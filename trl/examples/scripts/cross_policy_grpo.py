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

"""
Cross-policy GRPO (s=1) example with two policies:
  - Qwen/Qwen2.5-0.5B
  - Qwen/Qwen3-0.6B

This uses `CrossPolicyGRPOTrainer`, which inherits all the usual `GRPOTrainer` features and adds:
  - a shared verified-success buffer ℬ (JSONL file)
  - s=1 mixing: L = (1-α)*L_GRPO + α*L_SFT, where L_SFT is computed on successes from the *other* policy

Run (sequential, trains both policies one after the other):

    pixi run train-cross-policy

Run in two terminals (better cross-policy interaction; set CUDA devices as needed):

    CUDA_VISIBLE_DEVICES=0 pixi run train-policy0
    CUDA_VISIBLE_DEVICES=1 pixi run train-policy1
"""

import argparse
import os
from pathlib import Path
import re

import torch
from datasets import Dataset
from datasets import load_dataset

from trl import CrossPolicyGRPOConfig, CrossPolicyGRPOTrainer


def _extract_completion_text(completion):
    # Non-conversational: completion is a string.
    if isinstance(completion, str):
        return completion
    # Conversational: completion is typically `[{role, content}, ...]`
    if isinstance(completion, list) and completion and isinstance(completion[0], dict) and "content" in completion[0]:
        return completion[0]["content"]
    return str(completion)


_NUM_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")


def _extract_gsm8k_gold(answer: str) -> str | None:
    # GSM8K format: "... #### 42"
    if answer is None:
        return None
    answer = str(answer)
    if "####" in answer:
        ans = answer.split("####")[-1].strip()
        m = _NUM_RE.search(ans)
        if m:
            return m.group(0).replace(",", "")
        return None
    # Fallback: last number in the string
    matches = _NUM_RE.findall(answer)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def _extract_pred_number(completion) -> str | None:
    text = _extract_completion_text(completion)
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def gsm8k_exact_reward(prompts, completions, answer, **kwargs):
    """
    Verifiable reward for GSM8K:
      - Extract gold final numeric answer from `answer` (after ####)
      - Extract predicted final numeric answer as the last number in the completion
      - Reward 1.0 if they match exactly as strings after removing commas, else 0.0
    """
    rewards = []
    for completion, gold in zip(completions, answer, strict=True):
        gold_num = _extract_gsm8k_gold(gold)
        pred_num = _extract_pred_number(completion)
        rewards.append(1.0 if (gold_num is not None and pred_num is not None and pred_num == gold_num) else 0.0)
    return rewards


def build_gsm8k_dataset(split: str, max_samples: int | None) -> Dataset:
    ds = load_dataset("gsm8k", "main", split=split)

    def to_prompt(ex):
        q = ex["question"].strip()
        # Keep prompting simple for base models: ask for final numeric answer.
        prompt = (
            "Solve the following grade school math problem.\n"
            "Return ONLY the final numeric answer.\n\n"
            f"Question: {q}\n"
            "Answer: "
        )
        return {"prompt": prompt, "answer": ex["answer"]}

    ds = ds.map(to_prompt, remove_columns=[c for c in ds.column_names if c not in {"question", "answer"}])
    # Keep only what we need at training time (reward uses `answer`)
    ds = ds.remove_columns(["question"])
    if max_samples is not None and max_samples > 0:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def train_one_policy(
    model_name_or_path: str,
    policy_id: int,
    train_dataset: Dataset,
    output_dir: str,
    success_buffer_path: str,
    *,
    max_steps: int,
    per_device_train_batch_size: int,
    num_generations: int,
    max_completion_length: int,
    alpha: float,
    sft_batch_size: int,
    tau: float,
):
    # Model dtype: prefer bf16 when on GPU (works well on Ampere+), else float32.
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    args = CrossPolicyGRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        beta=0.0,  # keep example lightweight; set >0 for KL-to-reference regularization
        report_to=[],
        save_strategy="no",
        eval_strategy="no",
        logging_steps=10,
        remove_unused_columns=False,  # keep `solution` for the reward function
        cross_policy_policy_id=policy_id,
        cross_policy_interval=1,  # s=1
        cross_policy_mix_alpha=alpha,
        cross_policy_sft_batch_size=sft_batch_size,
        cross_policy_success_threshold=tau,
        cross_policy_success_buffer_path=success_buffer_path,
    )
    args.model_init_kwargs = {"dtype": model_dtype}

    trainer = CrossPolicyGRPOTrainer(
        model=model_name_or_path,
        reward_funcs=gsm8k_exact_reward,
        args=args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="runs/cross-policy-grpo-qwen", help="Base output directory.")
    parser.add_argument(
        "--success_buffer_path",
        type=str,
        default=None,
        help="Path to a shared JSONL success buffer. Defaults to <output_dir>/success_buffer.jsonl",
    )
    parser.add_argument(
        "--reset_buffer",
        action="store_true",
        help="If set, deletes the success buffer JSONL before training (useful for fresh runs).",
    )
    parser.add_argument(
        "--policy_id",
        type=int,
        default=None,
        choices=[0, 1],
        help="If set, trains only that policy (0 or 1). Otherwise trains both sequentially.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train[:1%]",
        help="GSM8K dataset split expression (HF datasets syntax), e.g. 'train[:1%]'.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=256,
        help="Max number of GSM8K training examples to use (after split).",
    )

    # Training knobs (kept small by default; increase for real training)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=64)

    # Cross-policy knobs (s=1 mixing)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--sft_batch_size", type=int, default=8)
    parser.add_argument("--tau", type=float, default=1.0)

    args = parser.parse_args()

    base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    success_buffer_path = args.success_buffer_path or str(base_out / "success_buffer.jsonl")
    if args.reset_buffer and os.path.exists(success_buffer_path):
        os.remove(success_buffer_path)

    train_dataset = build_gsm8k_dataset(args.train_split, args.max_train_samples)

    models = [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen3-0.6B",
    ]

    def run_policy(pid: int):
        train_one_policy(
            model_name_or_path=models[pid],
            policy_id=pid,
            train_dataset=train_dataset,
            output_dir=str(base_out / f"policy{pid}"),
            success_buffer_path=success_buffer_path,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            num_generations=args.num_generations,
            max_completion_length=args.max_completion_length,
            alpha=args.alpha,
            sft_batch_size=args.sft_batch_size,
            tau=args.tau,
        )

    if args.policy_id is not None:
        run_policy(args.policy_id)
    else:
        # Sequential run (simple / works on 1 GPU). For better cross-policy interaction, run policy0 and policy1 in
        # parallel in two terminals (see module docstring).
        run_policy(0)
        run_policy(1)


if __name__ == "__main__":
    main()


