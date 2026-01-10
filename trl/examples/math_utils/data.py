"""
Implementation of data loading, formatting, and reward helpers for math datasets.

Ported from the PODS utils package to mirror GSM8K formatting and grading.
"""

from __future__ import annotations

import re
from typing import Any

from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset
from transformers import AutoTokenizer

from .grader import grade_answer

# Default tokenizer to use for building prompts (only relevant for instruct style).
tokenizer_name = "Qwen/Qwen2.5-3B-Instruct"


def set_tokenizer_name(name: str) -> None:
    """
    Set the global tokenizer/model repo used by dataset builders.

    The dataset mapping functions rely on `AutoTokenizer.from_pretrained(tokenizer_name)`
    to obtain the appropriate chat template (e.g., Qwen vs LLaMA). This setter lets
    entrypoints choose the active model family dynamically.
    """
    global tokenizer_name
    if isinstance(name, str) and len(name) > 0:
        tokenizer_name = name


SYSTEM_PROMPT_BASE = """
An AI assistant is given a math problem and solves it step by step. The assistant first thinks about the reasoning process in the mind and then concludes the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., 
<think>
Reasoning
</think>
<answer>
Answer
</answer>
"""

SYSTEM_PROMPT_INSTRUCT = """
You will be given a math question.
Reason about the question and correct your own mistakes if you make any.

Respond in the following format:
<think>
Reason here. Decide to answer when you are confident as your response length is limited.
</think>
<answer>
Answer here.
</answer>
"""


def completion_to_text(completion: Any) -> str:
    """
    Normalize a completion into a plain string.

    The GRPO trainer may return a string or a chat-style list of dicts. We only
    care about the text payload when grading.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict) and "content" in completion[0]:
        return completion[0]["content"]
    return str(completion)


def extract_xml_answer(text: str) -> str:
    # Extracts the answer block from the XML format
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_last_integer(text: str) -> str:
    # Extracts the last integer from the text
    numbers = re.findall(r"[\d\.\,]+", text)
    if numbers:
        return numbers[-1]
    return "-1"


def answer_correct(text: str, answer: str) -> bool:
    """
    Uses the math grader to check if the answer is correct.

    We try multiple views of the model output (full text, extracted <answer> block,
    and last integer) to be more permissive.
    """
    candidates = [text, extract_xml_answer(text), extract_last_integer(text)]
    return any(grade_answer(ans, answer) for ans in candidates)


def format_score(text: str) -> float:
    pattern = r"<think>.*</think>[ \n]?<answer>.*</answer>"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match:
        return max(0.0, 1 - 0.01 * (len(text) - len(match.group(0))))
    return 0.0


def format_correct(text: str) -> bool:
    # Checks if the text is in the correct format
    pattern = r"^[ \n]?<think>.*</think>[ \n]?<answer>.*</answer>[ \n]?$"
    match = re.match(pattern, text, flags=re.DOTALL)
    return match is not None


def filter_function(example, tokenizer: AutoTokenizer) -> bool:
    """
    Filter out examples that are too long.

    Supports both:
      - string prompts
      - conversational prompts (list-of-messages dicts with "role"/"content")
    """
    prompt = example["prompt"]
    # Conversational prompts: tokenize via chat template so length matches what the model will see.
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict) and "role" in prompt[0] and "content" in prompt[0]:
        ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
        # `apply_chat_template` can return a list[int] or a BatchEncoding-like object depending on tokenizer/version.
        if isinstance(ids, dict) and "input_ids" in ids:
            input_ids = ids["input_ids"]
        elif hasattr(ids, "input_ids"):
            input_ids = ids.input_ids
        else:
            input_ids = ids
        length = len(input_ids) if isinstance(input_ids, list) else int(input_ids.shape[-1])
        return length <= 512

    # Plain string prompt
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    return tokenized_prompt["input_ids"].shape[1] <= 512


def get_math8k_questions(split: str = "train", style: str = "base") -> Dataset:
    # Loads the Math8K dataset, split is either "train" or "test"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset("parquet", data_files=f"datasets/math8k/{split}.parquet")["train"]
    if style == "base":
        data = data.map(
            lambda x: {
                "prompt": SYSTEM_PROMPT_BASE + "Problem: " + x["question"] + "\nSolution: ",
                "answer": x["gt_answer"],
            }
        )
    elif style == "instruct":
        data = data.map(
            lambda x: {
                # Store as messages (not a rendered string) so each policy can apply its own chat template.
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_INSTRUCT},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": x["gt_answer"],
            }
        )
    else:
        raise ValueError(f"Unknown style: {style}")
    data = data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer})
    return data


def extract_gsm8k_answer(text: str) -> str | None:
    # Extracts the answer from the GSM8K dataset
    if text is None or "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "")


def get_gsm8k_questions(
    split: str = "train", style: str = "base", tokenizer_override: str | None = None
) -> Dataset:
    """
    Loads the GSM8K dataset and formats prompts to match the PODS setup.

    `split` can be any HF datasets split expression, e.g. "train", "train[:1%]", "test".
    """
    tokenizer_id = tokenizer_override or tokenizer_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    data = load_dataset("openai/gsm8k", "main", split=split)
    if style == "base":
        data = data.map(
            lambda x: {
                "prompt": SYSTEM_PROMPT_BASE + "Problem: " + x["question"] + "\nSolution: ",
                "answer": extract_gsm8k_answer(x["answer"]),
            }
        )
    elif style == "instruct":
        data = data.map(
            lambda x: {
                # Store as messages (not a rendered string) so cross-policy SFT can re-render with the local tokenizer.
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_INSTRUCT},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": extract_gsm8k_answer(x["answer"]),
            }
        )
    else:
        raise ValueError(f"Unknown style: {style}")
    data = data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer})
    return data


def extract_math_answer(text: str) -> str:
    # Extracts the answer from the Math dataset
    if "\\boxed{" not in text:
        return None
    answer = text.split("\\boxed{")[-1]
    answer = answer.split("}")[0]
    return answer.strip()


def get_math_questions(split: str = "train", style: str = "base") -> Dataset:
    # Loads the Math dataset, split is either "train" or "test"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    subsets = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]
    datasets = [load_dataset("EleutherAI/hendrycks_math", s, split=split) for s in subsets]
    data = concatenate_datasets(datasets)
    if style == "base":
        data = data.map(
            lambda x: {
                "prompt": SYSTEM_PROMPT_BASE + "Problem: " + x["problem"] + "\nSolution: ",
                "answer": extract_math_answer(x["solution"]),
            }
        )
    elif style == "instruct":
        data = data.map(
            lambda x: {
                # Store as messages (not a rendered string) so each policy can apply its own chat template.
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_INSTRUCT},
                    {"role": "user", "content": x["problem"]},
                ],
                "answer": extract_math_answer(x["solution"]),
            }
        )
    else:
        raise ValueError(f"Unknown style: {style}")
    data = data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer}).shuffle(seed=42)
    return data


def get_math500_questions(split: str = "test", style: str = "base") -> Dataset:
    # Loads the Math500 dataset, split is either "train" or "test"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset("HuggingFaceH4/MATH-500", split=split)
    if style == "base":
        data = data.map(
            lambda x: {
                "prompt": SYSTEM_PROMPT_BASE + "Problem: " + x["problem"] + "\nSolution: ",
                "answer": x["answer"],
            }
        )
    elif style == "instruct":
        data = data.map(
            lambda x: {
                # Store as messages (not a rendered string) so each policy can apply its own chat template.
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_INSTRUCT},
                    {"role": "user", "content": x["problem"]},
                ],
                "answer": x["answer"],
            }
        )
    else:
        raise ValueError(f"Unknown style: {style}")
    data = data.filter(filter_function, fn_kwargs={"tokenizer": tokenizer})
    return data


def get_amc23_questions(split: str = "test", style: str = "base") -> Dataset:
    # Loads the AMC23 dataset, split is either "train" or "test"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    data = load_dataset("zwhe99/amc23", split=split)
    if style == "base":
        data = data.map(
            lambda x: {
                "prompt": SYSTEM_PROMPT_BASE + "Problem: " + x["question"] + "\nSolution: ",
                "answer": str(int(x["answer"])),
            }
        )
    elif style == "instruct":
        data = data.map(
            lambda x: {
                # Store as messages (not a rendered string) so each policy can apply its own chat template.
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT_INSTRUCT},
                    {"role": "user", "content": x["question"]},
                ],
                "answer": str(int(x["answer"])),
            }
        )
    else:
        raise ValueError(f"Unknown style: {style}")
    data = data.cast_column("answer", Value("string")).filter(filter_function, fn_kwargs={"tokenizer": tokenizer})
    return data


def get_questions(name: str, split: str = "train", style: str = "base") -> Dataset:
    # Loads the dataset based on the name provided
    if name == "math8k":
        return get_math8k_questions(split, style)
    if name == "gsm8k":
        return get_gsm8k_questions(split, style)
    if name == "math500":
        return get_math500_questions(split, style)
    if name == "amc23":
        return get_amc23_questions(split, style)
    if name == "math":
        return get_math_questions(split, style)
    raise ValueError(f"Unknown dataset name: {name}")


# Reward functions
def correctness_reward_func(completions, answer, **kwargs) -> list[float]:
    # Correctness reward, 1.0 for correct, 0.0 for incorrect
    responses = [completion_to_text(completion) for completion in completions]
    return [1.0 if answer_correct(r, a) else 0.0 for r, a in zip(responses, answer)]


def format_reward_func(completions, answer, **kwargs) -> list[float]:
    # Format reward, 0.1 for correct format
    responses = [completion_to_text(completion) for completion in completions]
    return [0.1 * format_score(r) for r, a in zip(responses, answer)]


def length_penalty_func(completion_mask, max_completion_length, **kwargs) -> list[float]:
    # Length penalty, 0.5 for completion length >= max_completion_length
    completion_lengths = completion_mask.sum(dim=1).tolist()
    return [-0.5 if l >= max_completion_length else 0.0 for l in completion_lengths]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>") == 1:
        count += 0.25
    if text.count("</think>") == 1:
        count += 0.25
    if text.count("<answer>") == 1:
        count += 0.25
    if text.count("</answer>") == 1:
        count += 0.25
    return count


def xmlcount_reward_func(completions, answer, **kwargs) -> list[float]:
    # XML count reward, 0.025 for each XML tag
    contents = [completion_to_text(completion) for completion in completions]
    return [0.1 * count_xml(r) for r, a in zip(contents, answer)]


