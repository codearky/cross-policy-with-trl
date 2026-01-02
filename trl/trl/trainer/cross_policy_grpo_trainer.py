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
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

import torch
from accelerate.logging import get_logger
from accelerate.state import PartialState
from accelerate.utils import gather_object
from transformers import PreTrainedModel

from ..data_utils import apply_chat_template, is_conversational
from .cross_policy_grpo_config import CrossPolicyGRPOConfig
from .grpo_trainer import GRPOTrainer

logger = get_logger(__name__)


def _log_info(message: str) -> None:
    if PartialState._shared_state:
        logger.info(message)
    else:
        import logging

        logging.getLogger("trl.cross_policy").info(message)


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

    def available_count(self, exclude_src: int | None = None) -> int:
        if exclude_src is None:
            return len(self._items)
        return sum(1 for it in self._items if it.src != exclude_src)

    def _build_counter(self, items: list[SuccessBufferItem]) -> Counter:
        if self.dedup:
            return Counter(it.dedup_key() for it in items)
        return Counter(id(it) for it in items)

    def sample(self, k: int, exclude_src: int | None = None, consume: bool = False) -> list[SuccessBufferItem]:
        if k <= 0 or not self._items:
            return []

        if exclude_src is None:
            population = list(self._items)
        else:
            population = [it for it in self._items if it.src != exclude_src]
        if not population:
            return []

        # Determine chosen items (without replacement)
        if k >= len(population):
            chosen = population[:]
            self._rng.shuffle(chosen)
        else:
            chosen = self._rng.sample(population, k)

        if consume and chosen:
            self.consume_items(chosen)

        return chosen

    def consume_items(self, items: list[SuccessBufferItem]) -> None:
        if not items:
            return

        counters = self._build_counter(items)
        new_items: deque[SuccessBufferItem] = deque()
        new_dedup: set[str] = set()

        for it in self._items:
            key = it.dedup_key() if self.dedup else id(it)
            if counters.get(key, 0) > 0:
                counters[key] -= 1
                continue
            new_items.append(it)
            if self.dedup:
                new_dedup.add(it.dedup_key())

        self._items = new_items
        if self.dedup:
            self._dedup_keys = new_dedup


class JsonlSuccessBuffer(SuccessBuffer):
    """
    File-backed success buffer for sharing ℬ across multiple training runs.

    Stores items as JSON Lines (one dict per line). Keeps a local in-memory window up to `max_size`.
    """

    def __init__(self, path: str, max_size: int, dedup: bool = True, seed: int | None = None):
        super().__init__(max_size=max_size, dedup=dedup, seed=seed)
        self.path = path
        self.consumed_path = path + ".consumed_keys.jsonl"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        # Touch file
        if not os.path.exists(path):
            with open(path, "a", encoding="utf-8"):
                pass
        if not os.path.exists(self.consumed_path):
            with open(self.consumed_path, "a", encoding="utf-8"):
                pass
        self._file_offset = 0
        self._consumed_offset = 0
        self._consumed_keys: set[str] = set()

    def _serialize(self, item: SuccessBufferItem) -> str:
        payload = {
            "prompt": item.prompt,
            "completion": item.completion,
            "reward": float(item.reward),
            "src": int(item.src),
        }
        return json.dumps(payload, ensure_ascii=False, default=repr)

    def _lock_file(self, f):
        try:
            import fcntl  # noqa: WPS433

            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            return fcntl
        except Exception:
            return None

    def _unlock_file(self, f, fcntl_mod):
        if fcntl_mod is None:
            return
        try:
            fcntl_mod.flock(f.fileno(), fcntl_mod.LOCK_UN)
        except Exception:
            pass

    def append(self, items: list[SuccessBufferItem]) -> int:
        if not items or self.max_size == 0:
            return 0
        with open(self.path, "a", encoding="utf-8") as f:
            fcntl_mod = self._lock_file(f)
            try:
                for it in items:
                    f.write(self._serialize(it) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            finally:
                self._unlock_file(f, fcntl_mod)
        return self.extend(items)

    def _refresh_consumed(self) -> int:
        inserted = 0
        try:
            size = os.path.getsize(self.consumed_path)
        except OSError:
            size = 0
        if size < self._consumed_offset:
            self._consumed_offset = 0

        with open(self.consumed_path, "r", encoding="utf-8") as f:
            f.seek(self._consumed_offset)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = obj.get("key")
                if isinstance(key, str) and key not in self._consumed_keys:
                    self._consumed_keys.add(key)
                    inserted += 1
            self._consumed_offset = f.tell()
        return inserted

    def _mark_consumed(self, items: list[SuccessBufferItem]) -> None:
        if not items:
            return
        keys = [it.dedup_key() for it in items]
        with open(self.consumed_path, "a", encoding="utf-8") as f:
            fcntl_mod = self._lock_file(f)
            try:
                for k in keys:
                    f.write(json.dumps({"key": k}, ensure_ascii=False) + "\n")
                f.flush()
                try:
                    os.fsync(f.fileno())
                except Exception:
                    pass
            finally:
                self._unlock_file(f, fcntl_mod)
        self._consumed_keys.update(keys)

    def refresh(self) -> int:
        """
        Read newly appended lines into the local in-memory buffer window.
        """
        if self.max_size == 0:
            return 0
        # Also refresh consumed keys so we don't re-introduce consumed samples.
        self._refresh_consumed()
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
                # Skip consumed
                if it.dedup_key() in self._consumed_keys:
                    continue
                inserted += int(self.add(it))
            self._file_offset = f.tell()
        return inserted

    def consume_items(self, items: list[SuccessBufferItem]) -> None:
        if not items:
            return
        super().consume_items(items)
        self._mark_consumed(items)


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
            self._cp_log_prefix = ""
            return

        self.cp_args: CrossPolicyGRPOConfig = self.args
        self._cp_opt_steps = 0

        # Create log prefix with policy ID for differentiation
        policy_id = self.cp_args.cross_policy_policy_id
        self._cp_log_prefix = f"[Policy{policy_id}] "

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

    def _log_cp_metrics(self, metrics: dict[str, float], mode: str = "train") -> None:
        """
        Add cross-policy metrics to the parent's metrics dict.

        Metrics are accumulated and logged together with other training metrics
        when the parent's log() method is called, ensuring proper wandb step handling.
        """
        if not metrics:
            return
        for key, value in metrics.items():
            self._metrics[mode][key].append(value)

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
                if not obj:
                    continue
                if isinstance(obj, list):
                    all_items.extend(obj)
                else:
                    all_items.append(obj)
            if isinstance(self._cp_buffer, JsonlSuccessBuffer):
                added = self._cp_buffer.append(all_items)
            else:
                added = self._cp_buffer.extend(all_items)
            if added:
                logger.info(
                    f"{self._cp_log_prefix}[CrossPolicy] Added {added} successes "
                    f"(buffer size={len(self._cp_buffer)})"
                )
                self._log_cp_metrics(
                    {
                        f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/buffer/added": float(added),
                        f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/buffer/size": float(len(self._cp_buffer)),
                    }
                )

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
        consume = bool(self.cp_args.cross_policy_success_buffer_consumable)
        require_full = bool(getattr(self.cp_args, "cross_policy_sft_require_full_batch", False))
        exclude_src = int(self.cp_args.cross_policy_policy_id)

        if require_full:
            batch = self._cp_buffer.sample(k, exclude_src=exclude_src, consume=consume)
            if len(batch) < k:
                _log_info(
                    f"{self._cp_log_prefix}[CrossPolicy] Skip SFT (need {k}, available {len(batch)}, buffer {len(self._cp_buffer)})"
                )
                self._log_cp_metrics(
                    {
                        f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/sft/skip": 1.0,
                        f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/buffer/size": float(len(self._cp_buffer)),
                    }
                )
                return []
            if consume:
                self._cp_buffer.consume_items(batch)
        else:
            batch = self._cp_buffer.sample(k, exclude_src=exclude_src, consume=consume)
        if batch:
            _log_info(
                f"{self._cp_log_prefix}[CrossPolicy] Pulled {len(batch)} samples for SFT "
                f"(consume={consume}, buffer {len(self._cp_buffer)})"
            )
            self._log_cp_metrics(
                {
                    f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/sft/batch_size": float(len(batch)),
                    f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/buffer/size": float(len(self._cp_buffer)),
                }
            )
        else:
            _log_info(f"{self._cp_log_prefix}[CrossPolicy] Skip SFT (buffer empty)")
            self._log_cp_metrics(
                {
                    f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/sft/skip": 1.0,
                    f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/buffer/size": float(len(self._cp_buffer)),
                }
            )
        return batch

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
                grpo_loss = loss
                with self.compute_loss_context_manager():
                    sft_loss = self._compute_cross_policy_sft_loss(model, batch)
                # Scale similarly to per-step losses under grad-accumulation
                sft_loss = sft_loss / max(int(self.current_gradient_accumulation_steps), 1)
                loss =  loss + alpha * sft_loss
                _log_info(
                    f"{self._cp_log_prefix}[CrossPolicy] Mixed SFT into GRPO step "
                    f"(alpha={alpha}, batch={len(batch)})"
                )
                self._log_cp_metrics(
                    {
                        f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/sft/mixed_batch_size": float(len(batch)),
                        f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/sft/alpha": float(alpha),
                        f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/loss/grpo": float(grpo_loss.detach().item()),
                        f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/loss/sft": float(sft_loss.detach().item()),
                        f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/buffer/size": float(len(self._cp_buffer)),
                    }
                )
            else:
                _log_info(f"{self._cp_log_prefix}[CrossPolicy] Mixed SFT skipped (no buffer batch)")
                self._log_cp_metrics({f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/sft/mixed_skip": 1.0})
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
                _log_info(f"{self._cp_log_prefix}[CrossPolicy] Stage-7 SFT skipped (buffer empty)")
                self._log_cp_metrics({f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/sft/stage7_skip": 1.0})
                break
            optimizer.zero_grad(set_to_none=True)
            with self.compute_loss_context_manager():
                sft_loss = self._compute_cross_policy_sft_loss(model, batch)
            self.accelerator.backward(sft_loss)
            optimizer.step()
            _log_info(
                f"{self._cp_log_prefix}[CrossPolicy] Stage-7 SFT step finished "
                f"(batch={len(batch)})"
            )
            self._log_cp_metrics(
                {
                    f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/sft/stage7_batch_size": float(len(batch)),
                    f"policy{self.cp_args.cross_policy_policy_id}/cross_policy/buffer/size": float(len(self._cp_buffer)),
                }
            )

        return out


