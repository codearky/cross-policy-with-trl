# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch
from accelerate.logging import get_logger
from accelerate.utils import gather_object
from transformers import PreTrainedModel

from ..data_utils import apply_chat_template, is_conversational
from .cross_policy_grpo_config import CrossPolicyGRPOConfig
from .grpo_trainer import GRPOTrainer

logger = get_logger(__name__)


def _json_dumps_fallback(obj: Any) -> str:
    """
    JSON-dumps `obj` for dedup keys. Falls back to `repr` when not JSON-serializable.
    """
    try:
        return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=repr)
    except TypeError:
        return repr(obj)


@dataclass(frozen=True)
class SuccessBufferItem:
    prompt: Any
    completion: Any
    reward: float
    src: int

    def dedup_key(self) -> str:
        # Reward and src are intentionally excluded from the dedup key.
        return _json_dumps_fallback({"prompt": self.prompt, "completion": self.completion})


class SuccessBuffer:
    """
    Shared success buffer ℬ: stores verified-success samples (x, y, r, src) and can sample from other-policy items.
    """

    def __init__(self, max_size: int, dedup: bool = True, seed: int | None = None):
        if max_size < 0:
            raise ValueError(f"max_size must be >= 0, got {max_size}")
        self.max_size = max_size
        self.dedup = dedup
        self._rng = random.Random(seed)
        self._items: deque[SuccessBufferItem] = deque()
        self._dedup_keys: set[str] = set()

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: SuccessBufferItem) -> bool:
        """
        Add an item to the buffer. Returns True if inserted, False if dropped (e.g., dedup hit or max_size==0).
        """
        if self.max_size == 0:
            return False

        if self.dedup:
            key = item.dedup_key()
            if key in self._dedup_keys:
                return False

        self._items.append(item)
        if self.dedup:
            self._dedup_keys.add(item.dedup_key())

        # Evict oldest
        while len(self._items) > self.max_size:
            evicted = self._items.popleft()
            if self.dedup:
                self._dedup_keys.discard(evicted.dedup_key())
        return True

    def extend(self, items: list[SuccessBufferItem]) -> int:
        inserted = 0
        for it in items:
            inserted += int(self.add(it))
        return inserted

    def sample(self, k: int, exclude_src: int | None = None) -> list[SuccessBufferItem]:
        if k <= 0 or not self._items:
            return []

        if exclude_src is None:
            population = list(self._items)
        else:
            population = [it for it in self._items if it.src != exclude_src]
        if not population:
            return []

        if k >= len(population):
            # Deterministic-ish shuffle so callers don't always see the same prefix
            population = population[:]
            self._rng.shuffle(population)
            return population
        return self._rng.sample(population, k)


class JsonlSuccessBuffer(SuccessBuffer):
    """
    File-backed success buffer for sharing ℬ across multiple training runs.

    Stores items as JSON Lines (one dict per line). Keeps a local in-memory window up to `max_size`.
    """

    def __init__(self, path: str, max_size: int, dedup: bool = True, seed: int | None = None):
        super().__init__(max_size=max_size, dedup=dedup, seed=seed)
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # Touch file
        if not os.path.exists(path):
            with open(path, "a", encoding="utf-8"):
                pass
        self._file_offset = 0

    def _serialize(self, item: SuccessBufferItem) -> str:
        payload = {
            "prompt": item.prompt,
            "completion": item.completion,
            "reward": float(item.reward),
            "src": int(item.src),
        }
        return json.dumps(payload, ensure_ascii=False, default=repr)

    def append(self, items: list[SuccessBufferItem]) -> int:
        if not items or self.max_size == 0:
            return 0
        with open(self.path, "a", encoding="utf-8") as f:
            # Best-effort inter-process file lock (Linux/macOS). If unavailable, we still append.
            try:
                import fcntl  # noqa: WPS433

                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                fcntl = None  # type: ignore[assignment]
            try:
                for it in items:
                    f.write(self._serialize(it) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            finally:
                if fcntl is not None:  # type: ignore[truthy-bool]
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    except Exception:
                        pass
        return self.extend(items)

    def refresh(self) -> int:
        """
        Read newly appended lines into the local in-memory buffer window.
        """
        if self.max_size == 0:
            return 0
        inserted = 0
        # Handle truncation / rotation
        try:
            size = os.path.getsize(self.path)
        except OSError:
            size = 0
        if size < self._file_offset:
            self._file_offset = 0

        with open(self.path, "r", encoding="utf-8") as f:
            f.seek(self._file_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    it = SuccessBufferItem(
                        prompt=obj.get("prompt"),
                        completion=obj.get("completion"),
                        reward=float(obj.get("reward", 0.0)),
                        src=int(obj.get("src", -1)),
                    )
                except Exception:
                    continue
                inserted += int(self.add(it))
            self._file_offset = f.tell()
        return inserted


def pooled_advantages(rewards: torch.Tensor, mode: str = "zscore", eps: float = 1e-4) -> torch.Tensor:
    """
    Compute pooled advantages per prompt over the union 𝒴(x) across *all* policies.

    Args:
        rewards: Tensor of shape (N, B, K) where:
            - N policies
            - B prompts
            - K rollouts per policy per prompt
        mode: 'zscore' | 'rank' | 'none'
        eps: numerical stability
    Returns:
        Tensor of shape (N, B, K).
    """
    if rewards.dim() != 3:
        raise ValueError(f"Expected rewards with shape (N,B,K), got {tuple(rewards.shape)}")
    mode = mode.lower()

    if mode == "none":
        return rewards.clone()

    n, bsz, k = rewards.shape
    flat = rewards.reshape(n, bsz, k)
    advantages = torch.empty_like(flat)

    if mode == "zscore":
        for b in range(bsz):
            u = flat[:, b, :].reshape(-1)
            mean = u.mean()
            std = u.std(unbiased=False)
            if torch.isfinite(std) and std > 0:
                advantages[:, b, :] = (flat[:, b, :] - mean) / (std + eps)
            else:
                advantages[:, b, :].zero_()
        return advantages

    if mode == "rank":
        for b in range(bsz):
            u = flat[:, b, :].reshape(-1)
            m = u.numel()
            if m <= 1:
                advantages[:, b, :].zero_()
                continue
            order = torch.argsort(u)  # ascending
            ranks = torch.empty_like(u, dtype=torch.float32)
            ranks[order] = torch.arange(m, device=u.device, dtype=torch.float32)
            # Normalize to roughly [-0.5, 0.5]
            adv = ranks / float(m - 1) - 0.5
            advantages[:, b, :] = adv.view(n, k)
        return advantages

    raise ValueError(f"Unknown cross_policy_advantage_mode: {mode}. Expected 'zscore', 'rank', or 'none'.")

class CrossPolicyGRPOTrainer(GRPOTrainer):
    """
    Cross-policy GRPO trainer that *inherits GRPOTrainer features* and adds verified-success sharing (Algorithm-v1.md).

    Key idea:
      - keep the full GRPOTrainer generation/reward/advantage/loss pipeline
      - add a shared success buffer ℬ via the GRPOTrainer `_post_process_rewards` hook
      - add cross-policy SFT mixing (s==1) and periodic SFT-only updates (s!=1)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.args, CrossPolicyGRPOConfig):
            # Allow users to pass GRPOConfig; cross-policy features will be disabled.
            self.cp_args: CrossPolicyGRPOConfig | None = None
            self._cp_buffer = None
            self._cp_opt_steps = 0
            return

        self.cp_args: CrossPolicyGRPOConfig = self.args
        self._cp_opt_steps = 0

        # Shared success buffer ℬ
        if self.cp_args.cross_policy_success_buffer_path:
            self._cp_buffer: SuccessBuffer = JsonlSuccessBuffer(
                path=self.cp_args.cross_policy_success_buffer_path,
                max_size=self.cp_args.cross_policy_success_buffer_size,
                dedup=self.cp_args.cross_policy_success_buffer_dedup,
                seed=self.cp_args.seed,
            )
        else:
            self._cp_buffer = SuccessBuffer(
                max_size=self.cp_args.cross_policy_success_buffer_size,
                dedup=self.cp_args.cross_policy_success_buffer_dedup,
                seed=self.cp_args.seed,
            )

    # --- Success buffer update hook ---
    def _post_process_rewards(
        self,
        mode: str,
        inputs: list[dict[str, torch.Tensor | Any]],
        prompts: list[Any],
        completions: list[Any],
        completion_ids_list: list[list[int]],
        rewards_per_func: torch.Tensor,
        rewards: torch.Tensor,
    ) -> None:
        if self.cp_args is None:
            return
        if mode != "train":
            return
        if self._cp_buffer is None or self.cp_args.cross_policy_success_buffer_size <= 0:
            return

        tau = float(self.cp_args.cross_policy_success_threshold)
        if tau is None:
            return

        # Determine local slice into the gathered rewards tensor.
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        local_rewards = rewards[process_slice]

        local_success_items: list[SuccessBufferItem] = []
        for p, c, r in zip(prompts, completions, local_rewards.tolist(), strict=True):
            if r >= tau:
                local_success_items.append(
                    SuccessBufferItem(
                        prompt=p,
                        completion=c,
                        reward=float(r),
                        src=int(self.cp_args.cross_policy_policy_id),
                    )
                )

        # Gather only the successes across processes (avoids gathering all prompts/completions).
        gathered = gather_object(local_success_items)
        if self.accelerator.is_main_process:
            # `gather_object` returns one object per process
            all_items: list[SuccessBufferItem] = []
            for obj in gathered:
                if obj:
                    all_items.extend(obj)
            if isinstance(self._cp_buffer, JsonlSuccessBuffer):
                self._cp_buffer.append(all_items)
            else:
                self._cp_buffer.extend(all_items)

    # --- SFT loss on buffer samples ---
    def _compute_cross_policy_sft_loss(
        self,
        model: PreTrainedModel,
        batch: list[SuccessBufferItem],
    ) -> torch.Tensor:
        tok = self.processing_class  # tokenizer or processor
        if not hasattr(tok, "pad_token_id"):
            raise TypeError("Cross-policy SFT loss currently requires a tokenizer-like processing_class.")

        input_ids_list: list[torch.Tensor] = []
        labels_list: list[torch.Tensor] = []
        attention_mask_list: list[torch.Tensor] = []

        for item in batch:
            prompt = item.prompt
            completion = item.completion

            # Conversational: prompt and completion are lists of messages
            if is_conversational({"prompt": prompt}):
                rendered = apply_chat_template(
                    {"prompt": prompt, "completion": completion},
                    tok,
                    tools=self.tools,
                    **(self.chat_template_kwargs or {}),
                )
                prompt_text = rendered["prompt"]
                completion_text = rendered["completion"]
            else:
                prompt_text = str(prompt)
                completion_text = str(completion)

            full_text = prompt_text + completion_text

            full_ids = tok(full_text, add_special_tokens=False)["input_ids"]
            prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
            prompt_len = len(prompt_ids)

            labels = full_ids.copy()
            labels[:prompt_len] = [-100] * prompt_len

            input_ids_list.append(torch.tensor(full_ids, dtype=torch.long, device=self.accelerator.device))
            labels_list.append(torch.tensor(labels, dtype=torch.long, device=self.accelerator.device))
            attention_mask_list.append(torch.ones(len(full_ids), dtype=torch.long, device=self.accelerator.device))

        max_len = max(t.size(0) for t in input_ids_list)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=self.accelerator.device)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long, device=self.accelerator.device)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long, device=self.accelerator.device)
        for i in range(len(batch)):
            L = input_ids_list[i].size(0)
            input_ids[i, :L] = input_ids_list[i]
            labels[i, :L] = labels_list[i]
            attn[i, :L] = attention_mask_list[i]

        out = model(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
        return out.loss

    def _sample_cross_policy_sft_batch(self, k: int) -> list[SuccessBufferItem]:
        if self.cp_args is None or self._cp_buffer is None or k <= 0:
            return []
        if isinstance(self._cp_buffer, JsonlSuccessBuffer):
            # Best-effort refresh before sampling
            self._cp_buffer.refresh()
        return self._cp_buffer.sample(k, exclude_src=int(self.cp_args.cross_policy_policy_id))

    # --- Stage 6 (s==1) mixing ---
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        if return_outputs:
            return loss
        if self.cp_args is None or not model.training:
            return loss

        s = int(self.cp_args.cross_policy_interval)
        alpha = float(self.cp_args.cross_policy_mix_alpha)
        sft_bs = int(self.cp_args.cross_policy_sft_batch_size)

        if s == 1 and alpha > 0.0 and sft_bs > 0:
            batch = self._sample_cross_policy_sft_batch(sft_bs)
            if batch:
                with self.compute_loss_context_manager():
                    sft_loss = self._compute_cross_policy_sft_loss(model, batch)
                # Scale similarly to per-step losses under grad-accumulation
                sft_loss = sft_loss / max(int(self.current_gradient_accumulation_steps), 1)
                loss = (1.0 - alpha) * loss + alpha * sft_loss
        return loss

    # --- Stage 7 (s!=1) periodic SFT-only extra steps ---
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # First: normal GRPO step
        out = super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure=optimizer_closure)

        if self.cp_args is None:
            return out

        s = int(self.cp_args.cross_policy_interval)
        sft_steps = int(self.cp_args.cross_policy_sft_steps)
        sft_bs = int(self.cp_args.cross_policy_sft_batch_size)

        self._cp_opt_steps += 1

        if s == 1 or s <= 0 or sft_steps <= 0 or sft_bs <= 0:
            return out

        # Only run stage-7 SFT steps every s optimizer steps
        if self._cp_opt_steps % s != 0:
            return out

        # Deepspeed/FSDP stepping is more complex; keep it explicit.
        if getattr(self, "is_deepspeed_enabled", False) or getattr(self, "is_fsdp_enabled", False):
            logger.warning(
                "Cross-policy stage-7 extra SFT steps are currently skipped under DeepSpeed/FSDP. "
                "Use cross_policy_interval==1 to mix SFT into the main loss instead."
            )
            return out

        model = self.model
        model.train()
        for _ in range(sft_steps):
            batch = self._sample_cross_policy_sft_batch(sft_bs)
            if not batch:
                break
            optimizer.zero_grad(set_to_none=True)
            with self.compute_loss_context_manager():
                sft_loss = self._compute_cross_policy_sft_loss(model, batch)
            self.accelerator.backward(sft_loss)
            optimizer.step()

        return out


