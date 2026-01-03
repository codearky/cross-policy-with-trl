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

Run cross-policy (parallel, trains both policies simultaneously on separate GPUs):

    pixi run train-cp

Run baseline (regular GRPO without cross-policy, on different GPUs):

    pixi run train-baseline

Run in two terminals (alternative approach):

    CUDA_VISIBLE_DEVICES=0 pixi run train-policy0
    CUDA_VISIBLE_DEVICES=1 pixi run train-policy1
"""

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig

from trl import CrossPolicyGRPOConfig, CrossPolicyGRPOTrainer, GRPOConfig, GRPOTrainer

# Add examples directory to path for math_utils import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from math_utils.data import answer_correct, completion_to_text, get_gsm8k_questions


def gsm8k_structured_reward(prompts, completions, answer, **kwargs):
    """
    Reward for GSM8K using the shared math grader.
    Extracts raw text from completions (supports chat-style outputs) and
    checks correctness via the XML-aware `answer_correct`.
    """
    texts = [completion_to_text(completion) for completion in completions]
    return [1.0 if answer_correct(text, gold) else 0.0 for text, gold in zip(texts, answer)]


def build_gsm8k_dataset(split: str, style: str, max_samples: int | None, *, tokenizer_name: str | None = None) -> Dataset:
    ds = get_gsm8k_questions(split=split, style=style, tokenizer_override=tokenizer_name)
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
    gradient_accumulation_steps: int,
    num_generations: int,
    max_completion_length: int,
    alpha: float,
    sft_batch_size: int,
    tau: float,
    warmup_steps: int,
    buffer_warmup_steps: int,
    gpu_id: int | None = None,
    eval_dataset: Dataset | None = None,
    eval_steps: int | None = None,
):
    # Model dtype: prefer bf16 when on GPU (works well on Ampere+), else float32.
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Extract model short name for logging
    model_short_name = model_name_or_path.split("/")[-1] if "/" in model_name_or_path else model_name_or_path

    # GPU assignment: use explicit gpu_id if provided, else fall back to policy_id-based assignment
    if torch.cuda.is_available():
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            print(f"[CrossPolicy] Assigned {model_short_name} to cuda:{gpu_id}")
        else:
            device_count = torch.cuda.device_count()
            if device_count > 1:
                device_id = policy_id % device_count
                torch.cuda.set_device(device_id)
                print(f"[CrossPolicy] Assigned {model_short_name} to cuda:{device_id}")
            else:
                print(f"[CrossPolicy] {model_short_name} running on the only available GPU (cuda:0)")
    else:
        print(f"[CrossPolicy] {model_short_name} running on CPU")
    run_name = f"policy{policy_id}_{model_short_name}"

    # Determine eval strategy based on whether eval is requested
    eval_strategy = "steps" if eval_dataset is not None and eval_steps else "no"

    args = CrossPolicyGRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=num_generations,  # Must be divisible by num_generations
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        beta=0.0,  # keep example lightweight; set >0 for KL-to-reference regularization
        report_to=["wandb"],
        run_name=run_name,
        save_strategy="no",
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        logging_steps=20,
        remove_unused_columns=False,  # keep `solution` for the reward function
        cross_policy_policy_id=policy_id,
        cross_policy_interval=1,  # s=1
        cross_policy_mix_alpha=alpha,
        cross_policy_sft_batch_size=sft_batch_size,
        cross_policy_warmup_steps=warmup_steps,
        cross_policy_buffer_warmup_steps=buffer_warmup_steps,
        cross_policy_success_threshold=tau,
        cross_policy_success_buffer_path=success_buffer_path,
        scale_rewards="none",  # Avoid z-score normalization for larger GRPO gradients
    )
    # Explicitly set device_map to pin model to specific GPU (device_map="auto" would use GPU 0)
    device_map = f"cuda:{gpu_id}" if gpu_id is not None else "auto"
    args.model_init_kwargs = {"dtype": model_dtype, "device_map": device_map}

    # LoRA configuration with rank 64 and alpha 64
    peft_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = CrossPolicyGRPOTrainer(
        model=model_name_or_path,
        reward_funcs=gsm8k_structured_reward,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Log model metadata to W&B
    if hasattr(trainer, 'accelerator') and trainer.accelerator.is_main_process:
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update({
                    "model_name": model_name_or_path,
                    "policy_id": policy_id,
                    "model_short_name": model_short_name,
                })
                wandb.run.tags = wandb.run.tags + (f"policy{policy_id}", model_short_name)
        except ImportError:
            pass

    trainer.train()
    trainer.save_model(args.output_dir)


def _train_worker(model_name_or_path, policy_id, train_dataset, output_dir, success_buffer_path, gpu_id, train_kwargs, eval_dataset=None):
    """
    Worker function for multiprocessing. Must be at module level to be picklable.
    """
    # CRITICAL: Set CUDA_VISIBLE_DEVICES before any CUDA initialization
    # This ensures the process only sees its assigned GPU (which becomes cuda:0)
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    train_one_policy(
        model_name_or_path=model_name_or_path,
        policy_id=policy_id,
        train_dataset=train_dataset,
        output_dir=output_dir,
        success_buffer_path=success_buffer_path,
        gpu_id=0,  # After CUDA_VISIBLE_DEVICES, the target GPU becomes cuda:0
        eval_dataset=eval_dataset,
        **train_kwargs,
    )


def train_baseline_policy(
    model_name_or_path: str,
    policy_id: int,
    train_dataset: Dataset,
    output_dir: str,
    *,
    max_steps: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_generations: int,
    max_completion_length: int,
    gpu_id: int | None = None,
    eval_dataset: Dataset | None = None,
    eval_steps: int | None = None,
):
    """
    Train a single policy with regular GRPO (no cross-policy mixing).
    Used as baseline for comparison.
    """
    # Model dtype: prefer bf16 when on GPU (works well on Ampere+), else float32.
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Extract model short name for logging
    model_short_name = model_name_or_path.split("/")[-1] if "/" in model_name_or_path else model_name_or_path

    # GPU assignment
    if torch.cuda.is_available():
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            print(f"[Baseline] Assigned {model_short_name} to cuda:{gpu_id}")
        else:
            device_count = torch.cuda.device_count()
            if device_count > 1:
                device_id = policy_id % device_count
                torch.cuda.set_device(device_id)
                print(f"[Baseline] Assigned {model_short_name} to cuda:{device_id}")
            else:
                print(f"[Baseline] {model_short_name} running on the only available GPU (cuda:0)")
    else:
        print(f"[Baseline] {model_short_name} running on CPU")

    run_name = f"baseline_{model_short_name}"

    # Determine eval strategy based on whether eval is requested
    eval_strategy = "steps" if eval_dataset is not None and eval_steps else "no"

    args = GRPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        beta=0.0,  # keep example lightweight; set >0 for KL-to-reference regularization
        report_to=["wandb"],
        run_name=run_name,
        save_strategy="no",
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        logging_steps=20,
        remove_unused_columns=False,  # keep `answer` for the reward function
        log_completions=True,
        scale_rewards="none",  # Avoid z-score normalization for larger GRPO gradients
    )
    # Explicitly set device_map to pin model to specific GPU (device_map="auto" would use GPU 0)
    device_map = f"cuda:{gpu_id}" if gpu_id is not None else "auto"
    args.model_init_kwargs = {"dtype": model_dtype, "device_map": device_map}

    # LoRA configuration with rank 64 and alpha 64
    peft_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = GRPOTrainer(
        model=model_name_or_path,
        reward_funcs=gsm8k_structured_reward,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Log model metadata to W&B
    if hasattr(trainer, 'accelerator') and trainer.accelerator.is_main_process:
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update({
                    "model_name": model_name_or_path,
                    "policy_id": policy_id,
                    "model_short_name": model_short_name,
                    "mode": "baseline",
                })
                wandb.run.tags = wandb.run.tags + ("baseline", model_short_name)
        except ImportError:
            pass

    trainer.train()
    trainer.save_model(args.output_dir)


def _baseline_worker(model_name_or_path, policy_id, train_dataset, output_dir, gpu_id, train_kwargs, eval_dataset=None):
    """
    Worker function for baseline multiprocessing. Must be at module level to be picklable.
    """
    # CRITICAL: Set CUDA_VISIBLE_DEVICES before any CUDA initialization
    # This ensures the process only sees its assigned GPU (which becomes cuda:0)
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    train_baseline_policy(
        model_name_or_path=model_name_or_path,
        policy_id=policy_id,
        train_dataset=train_dataset,
        output_dir=output_dir,
        gpu_id=0,  # After CUDA_VISIBLE_DEVICES, the target GPU becomes cuda:0
        eval_dataset=eval_dataset,
        **train_kwargs,
    )


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
        help="If set, trains only that policy (0 or 1). Otherwise trains both in parallel on separate GPUs.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run baseline training with regular GRPO (no cross-policy mixing).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1' or '2,3'). Defaults to 0,1.",
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
        default=8500,
        help="Max number of GSM8K training examples to use (after split).",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default=None,
        help="GSM8K eval split expression (e.g. 'test'). If not set, no eval is performed.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=100,
        help="Max number of eval examples to use.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Run evaluation every N steps. If not set, no eval is performed.",
    )
    parser.add_argument(
        "--prompt_styles",
        type=str,
        default="base",
        help="Comma-separated prompt styles per policy (each entry: 'base' or 'instruct').",
    )
    parser.add_argument(
        "--tokenizer_names",
        type=str,
        default=None,
        help="Comma-separated tokenizer/model names per policy (overrides --tokenizer_name).",
    )

    # Training knobs (kept small by default; increase for real training)
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for no limit)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=16)
    parser.add_argument("--max_completion_length", type=int, default=512)

    # Cross-policy knobs (s=1 mixing)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sft_batch_size", type=int, default=8)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=400,
        help="Optimizer steps to run before enabling cross-policy mixed loss (s=1).",
    )
    parser.add_argument(
        "--buffer_warmup_steps",
        type=int,
        default=300,
        help="Optimizer steps to run before writing successes to the shared buffer.",
    )

    args = parser.parse_args()

    # Parse GPU IDs
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
        if len(gpu_ids) < 2:
            gpu_ids = [gpu_ids[0], gpu_ids[0]]  # Use same GPU for both if only one provided
    else:
        gpu_ids = [0, 1]  # Default

    # Set output directory based on mode
    if args.baseline:
        base_out = Path(args.output_dir.replace("cross-policy", "baseline") if "cross-policy" in args.output_dir else args.output_dir + "-baseline")
    else:
        base_out = Path(args.output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    success_buffer_path = args.success_buffer_path or str(base_out / "success_buffer.jsonl")
    if args.reset_buffer and os.path.exists(success_buffer_path):
        os.remove(success_buffer_path)

    models = [
        # "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
    ]
    num_policies = len(models)

    def _parse_tokenizer_names() -> list[str | None]:
        if args.tokenizer_names:
            tokens = [tok.strip() for tok in args.tokenizer_names.split(",") if tok.strip()]
            if len(tokens) == 1 and num_policies > 1:
                tokens = tokens * num_policies
            if len(tokens) != num_policies:
                raise ValueError(
                    f"--tokenizer_names must provide 1 or {num_policies} entries; received {len(tokens)}."
                )
            return tokens
        # Default to each policy using its own model name as tokenizer.
        return models[:num_policies]

    policy_tokenizers = _parse_tokenizer_names()

    def _parse_prompt_styles() -> list[str]:
        valid_styles = {"base", "instruct"}
        styles = [style.strip() for style in args.prompt_styles.split(",") if style.strip()]
        if not styles:
            raise ValueError("--prompt_styles must specify at least one style (base or instruct).")
        if len(styles) == 1 and num_policies > 1:
            styles = styles * num_policies
        if len(styles) != num_policies:
            raise ValueError(
                f"--prompt_styles must provide 1 or {num_policies} entries; received {len(styles)}."
            )
        unknown = [s for s in styles if s not in valid_styles]
        if unknown:
            raise ValueError(f"Invalid prompt styles: {unknown}. Supported: {sorted(valid_styles)}.")
        return styles

    policy_styles = _parse_prompt_styles()

    print("[Config] Policy settings:")
    train_datasets = []
    eval_datasets = []
    for idx, tokenizer_name in enumerate(policy_tokenizers):
        style = policy_styles[idx]
        print(
            f"  - policy{idx}: model={models[idx]}, tokenizer={tokenizer_name}, style={style}"
        )
        ds = build_gsm8k_dataset(
            args.train_split,
            style,
            args.max_train_samples,
            tokenizer_name=tokenizer_name,
        )
        train_datasets.append(ds)
        
        # Build eval dataset for this policy using its own tokenizer and style
        if args.eval_split and args.eval_steps:
            eval_ds = build_gsm8k_dataset(
                args.eval_split,
                style,
                args.max_eval_samples,
                tokenizer_name=tokenizer_name,
            )
            eval_datasets.append(eval_ds)
            print(f"  - policy{idx} eval: {len(eval_ds)} examples from '{args.eval_split}' (style={style}, tokenizer={tokenizer_name})")
        else:
            eval_datasets.append(None)
    
    if args.eval_split and args.eval_steps:
        print(f"[Config] Eval: every {args.eval_steps} steps (each policy uses its own style and tokenizer)")

    if args.baseline:
        # Baseline mode: regular GRPO without cross-policy mixing
        baseline_kwargs = {
            "max_steps": args.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size + 4,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "eval_steps": args.eval_steps,
        }
        print(f"baseline_kwargs: {baseline_kwargs}")
        def run_baseline(pid: int, gpu_id: int | None = None):
            train_baseline_policy(
                model_name_or_path=models[pid],
                policy_id=pid,
                train_dataset=train_datasets[pid],
                output_dir=str(base_out / f"baseline{pid}"),
                gpu_id=gpu_id,
                eval_dataset=eval_datasets[pid],
                **baseline_kwargs,
            )

        if args.policy_id is not None:
            run_baseline(args.policy_id, gpu_ids[args.policy_id])
        else:
            # Parallel run
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if num_gpus < 2:
                print(f"[Baseline] Warning: Only {num_gpus} GPU(s) available. Running {models[0]} and {models[1]} sequentially.")
                run_baseline(0, gpu_ids[0])
                run_baseline(1, gpu_ids[1] if len(gpu_ids) > 1 else gpu_ids[0])
            else:
                print(f"[Baseline] Starting parallel training: {models[0]} on GPU {gpu_ids[0]}, {models[1]} on GPU {gpu_ids[1]}")
                ctx = mp.get_context("spawn")
                worker_args_0 = (models[0], 0, train_datasets[0], str(base_out / "baseline0"), gpu_ids[0], baseline_kwargs, eval_datasets[0])
                worker_args_1 = (models[1], 1, train_datasets[1], str(base_out / "baseline1"), gpu_ids[1], baseline_kwargs, eval_datasets[1])
                p0 = ctx.Process(target=_baseline_worker, args=worker_args_0)
                p1 = ctx.Process(target=_baseline_worker, args=worker_args_1)
                p0.start()
                p1.start()
                p0.join()
                p1.join()
                if p0.exitcode != 0:
                    print(f"[Baseline] {models[0]} exited with code {p0.exitcode}")
                if p1.exitcode != 0:
                    print(f"[Baseline] {models[1]} exited with code {p1.exitcode}")
    else:
        # Cross-policy mode
        train_kwargs = {
            "max_steps": args.max_steps,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_generations": args.num_generations,
            "max_completion_length": args.max_completion_length,
            "alpha": args.alpha,
            "sft_batch_size": args.sft_batch_size,
            "tau": args.tau,
            "warmup_steps": args.warmup_steps,
            "buffer_warmup_steps": args.buffer_warmup_steps,
            "eval_steps": args.eval_steps,
        }

        def run_policy(pid: int, gpu_id: int | None = None):
            train_one_policy(
                model_name_or_path=models[pid],
                policy_id=pid,
                train_dataset=train_datasets[pid],
                output_dir=str(base_out / f"policy{pid}"),
                success_buffer_path=success_buffer_path,
                gpu_id=gpu_id,
                eval_dataset=eval_datasets[pid],
                **train_kwargs,
            )

        if args.policy_id is not None:
            run_policy(args.policy_id, gpu_ids[args.policy_id])
        else:
            # Parallel run: spawn two processes, each on a dedicated GPU
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if num_gpus < 2:
                print(f"[CrossPolicy] Warning: Only {num_gpus} GPU(s) available. Running {models[0]} and {models[1]} sequentially.")
                run_policy(0, gpu_ids[0])
                run_policy(1, gpu_ids[1] if len(gpu_ids) > 1 else gpu_ids[0])
            else:
                print(f"[CrossPolicy] Starting parallel training: {models[0]} on GPU {gpu_ids[0]}, {models[1]} on GPU {gpu_ids[1]}")
                # Use 'spawn' to avoid CUDA context issues with forking
                ctx = mp.get_context("spawn")
                # Prepare arguments for the worker function (must be picklable)
                worker_args_0 = (models[0], 0, train_datasets[0], str(base_out / "policy0"), success_buffer_path, gpu_ids[0], train_kwargs, eval_datasets[0])
                worker_args_1 = (models[1], 1, train_datasets[1], str(base_out / "policy1"), success_buffer_path, gpu_ids[1], train_kwargs, eval_datasets[1])
                p0 = ctx.Process(target=_train_worker, args=worker_args_0)
                p1 = ctx.Process(target=_train_worker, args=worker_args_1)
                p0.start()
                p1.start()
                p0.join()
                p1.join()
                if p0.exitcode != 0:
                    print(f"[CrossPolicy] {models[0]} exited with code {p0.exitcode}")
                if p1.exitcode != 0:
                    print(f"[CrossPolicy] {models[1]} exited with code {p1.exitcode}")


if __name__ == "__main__":
    main()


