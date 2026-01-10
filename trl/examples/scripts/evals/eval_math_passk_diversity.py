"""
Lightweight evaluation to compare checkpoints on math-style tasks:
- pass@1 / pass@k (k completions per prompt)
- optional reward mean/std
- diversity: unique rate, distinct-1/2, optional self-BLEU

Designed for GSM8K/Math-style prompts from `trl.examples.math_utils`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import os
import multiprocessing as mp
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import AutoPeftModelForCausalLM
except Exception:  # pragma: no cover
    AutoPeftModelForCausalLM = None

from trl.examples.math_utils.data import (
    answer_correct,
    completion_to_text,
    get_questions,
    set_tokenizer_name,
)


def distinct_n(texts: Sequence[str], n: int) -> float:
    total = 0
    uniq = set()
    for t in texts:
        tokens = t.split()
        ngrams = list(zip(*[tokens[i:] for i in range(n)]))
        total += len(ngrams)
        uniq.update(ngrams)
    return float(len(uniq) / total) if total else 0.0


def try_self_bleu(texts: Sequence[str], sample_size: int = 200, n_gram: int = 4) -> float | None:
    try:
        import sacrebleu
    except Exception:
        return None

    if len(texts) < 2:
        return None
    pool = random.sample(texts, min(sample_size, len(texts)))
    scores: list[float] = []
    for idx, hyp in enumerate(pool):
        refs = pool[:idx] + pool[idx + 1 :]
        if not refs:
            continue
        scores.append(sacrebleu.sentence_bleu(hyp, refs, smooth_method="exp", max_ngram_order=n_gram).score)
    return float(sum(scores) / len(scores)) if scores else None


def render_prompt(prompt, tokenizer: AutoTokenizer) -> str:
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict) and "role" in prompt[0]:
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    return prompt


def load_model(model_path: str, dtype: torch.dtype, device_map: str | None):
    if AutoPeftModelForCausalLM is not None:
        try:
            return AutoPeftModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device_map)
        except Exception:
            pass
    return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device_map)


def generate_batch(
    model,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> list[list[str]]:
    """
    Sample k completions for a batch of prompts in one forward pass.
    Returns list of lists shaped [batch, k].
    """
    expanded_prompts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for p in prompts:
        start = cursor
        expanded_prompts.extend([p] * k)
        cursor += k
        spans.append((start, cursor))

    model_inputs = tokenizer(
        expanded_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else 0,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_lens = (model_inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    decoded: list[str] = []
    for row_idx, output in enumerate(outputs):
        start = int(prompt_lens[row_idx].item())
        gen_tokens = output[start:]
        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        decoded.append(text)

    grouped: list[list[str]] = []
    for s, e in spans:
        grouped.append(decoded[s:e])
    return grouped


def build_dataset(name: str, split: str, style: str, max_prompts: int | None, tokenizer_for_filter: str) -> Dataset:
    set_tokenizer_name(tokenizer_for_filter)
    ds = get_questions(name=name, split=split, style=style)
    if max_prompts is not None and max_prompts > 0:
        ds = ds.select(range(min(max_prompts, len(ds))))
    return ds


def evaluate_model(
    model_path: str,
    tokenizer_name: str,
    dataset: list[dict],
    *,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
    prompt_batch_size: int,
    gpu_id: int | None = None,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = load_model(model_path, dtype=dtype, device_map="auto")
    model.eval()

    pass1 = []
    passk = []
    rewards_flat = []
    lengths = []
    all_completions: list[str] = []
    records = []

    total_batches = math.ceil(len(dataset) / prompt_batch_size)
    for idx in tqdm(range(0, len(dataset), prompt_batch_size), desc=f"Eval {Path(model_path).name}", total=total_batches):
        batch = dataset[idx : idx + prompt_batch_size]
        prompts = [render_prompt(ex["prompt"], tokenizer) for ex in batch]
        completions_per_prompt = generate_batch(
            model,
            tokenizer,
            prompts,
            k=k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        for ex, completions in zip(batch, completions_per_prompt):
            answer = ex["answer"]
            texts = [completion_to_text(c) for c in completions]
            rewards = [1.0 if answer_correct(t, answer) else 0.0 for t in texts]

            pass1.append(rewards[0])
            passk.append(1.0 if any(rewards) else 0.0)
            rewards_flat.extend(rewards)
            lengths.extend([len(t.split()) for t in texts])
            all_completions.extend(texts)

            records.append(
                {
                    "prompt": ex["prompt"],
                    "answer": answer,
                    "completions": texts,
                    "rewards": rewards,
                }
            )

    unique_rate = len(set(all_completions)) / len(all_completions) if all_completions else 0.0
    metrics = {
        "model": model_path,
        "pass@1": float(np.mean(pass1)) if pass1 else math.nan,
        "pass@k": float(np.mean(passk)) if passk else math.nan,
        "reward_mean": float(np.mean(rewards_flat)) if rewards_flat else math.nan,
        "reward_std": float(np.std(rewards_flat)) if rewards_flat else math.nan,
        "distinct1": distinct_n(all_completions, 1),
        "distinct2": distinct_n(all_completions, 2),
        "unique_completion_rate": unique_rate,
        "mean_len_words": float(np.mean(lengths)) if lengths else math.nan,
    }
    self_bleu = try_self_bleu(all_completions)
    if self_bleu is not None:
        metrics["self_bleu"] = self_bleu

    return {"metrics": metrics, "records": records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pass@k + diversity eval for math datasets.")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model or adapter paths.")
    parser.add_argument(
        "--tokenizers",
        type=str,
        default=None,
        help="Comma-separated tokenizer names (defaults to model names).",
    )
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset name for math_utils.get_questions.")
    parser.add_argument("--split", type=str, default="test[:100]", help="HF datasets split expression.")
    parser.add_argument("--prompt_style", type=str, default="instruct", choices=["base", "instruct"])
    parser.add_argument("--max_prompts", type=int, default=100, help="Number of prompts to evaluate.")
    parser.add_argument("--samples_per_prompt", type=int, default=8, help="k for pass@k.")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0, help="0 disables top-k filtering.")
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU IDs per model; if fewer provided than models, remaining use auto placement.",
    )
    parser.add_argument(
        "--prompt_batch_size",
        type=int,
        default=1,
        help="Number of prompts evaluated per forward pass (each still gets k samples).",
    )
    parser.add_argument("--save_jsonl", type=str, default=None, help="Optional path to dump per-prompt completions.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_paths = [m.strip() for m in args.models.split(",") if m.strip()]
    tokenizer_names = (
        [t.strip() for t in args.tokenizers.split(",") if t.strip()] if args.tokenizers else model_paths
    )
    if len(tokenizer_names) == 1 and len(model_paths) > 1:
        tokenizer_names = tokenizer_names * len(model_paths)
    if len(tokenizer_names) != len(model_paths):
        raise ValueError("tokenizers count must match models (or provide a single tokenizer to reuse).")

    # Build a single shared dataset to keep prompts identical across models.
    dataset = build_dataset(
        name=args.dataset,
        split=args.split,
        style=args.prompt_style,
        max_prompts=args.max_prompts,
        tokenizer_for_filter=tokenizer_names[0],
    )
    dataset_list = list(dataset)

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")] if args.gpus else []

    manager = mp.Manager()
    shared_results = []
    procs = []

    def _worker(model_path, tok_name, gpu_id, shared_dict):
        res = evaluate_model(
            model_path=model_path,
            tokenizer_name=tok_name,
            dataset=dataset_list,
            k=args.samples_per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            prompt_batch_size=args.prompt_batch_size,
            gpu_id=gpu_id,
        )
        shared_dict["metrics"] = res["metrics"]
        shared_dict["records"] = res["records"]

    for idx, (model_path, tok_name) in enumerate(zip(model_paths, tokenizer_names)):
        shared_dict = manager.dict()
        shared_results.append(shared_dict)
        gpu_id = gpu_ids[idx] if idx < len(gpu_ids) else None
        p = mp.Process(target=_worker, args=(model_path, tok_name, gpu_id, shared_dict))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    all_results = []
    for model_path, shared_dict in zip(model_paths, shared_results):
        metrics = dict(shared_dict.get("metrics", {}))
        metrics["model"] = model_path
        all_results.append(metrics)

        if args.save_jsonl and "records" in shared_dict:
            out_path = Path(args.save_jsonl)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("a", encoding="utf-8") as f:
                for rec in shared_dict["records"]:
                    f.write(json.dumps({"model": model_path, **rec}) + "\n")

    print(json.dumps({"results": all_results}, indent=2))


if __name__ == "__main__":
    main()

