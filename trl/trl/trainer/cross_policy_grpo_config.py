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

from dataclasses import dataclass, field

from .grpo_config import GRPOConfig


@dataclass
class CrossPolicyGRPOConfig(GRPOConfig):
    """
    Configuration for `CrossPolicyGRPOTrainer` (GRPO-CP in Algorithm-v1.md).

    This extends `GRPOConfig` with cross-policy success sharing knobs:
      - shared success buffer ℬ (size + τ threshold + dedup)
      - pooled-advantage mode across the union 𝒴(x) (zscore/rank/none)
      - cross-policy SFT mixing (α) and scheduling (interval s, sft_steps, sft_batch_size)
    """

    # Cross-policy success buffer ℬ
    cross_policy_policy_id: int = field(
        default=0,
        metadata={
            "help": "Identifier for this policy (src tag) when writing items into a shared success buffer. "
            "Use different values for different policies when training them in separate runs."
        },
    )
    cross_policy_success_buffer_path: str | None = field(
        default=None,
        metadata={
            "help": "Optional path to a JSONL file used as a shared cross-policy success buffer ℬ. "
            "If provided, the trainer will append successes to this file and sample from it."
        },
    )
    cross_policy_success_threshold: float = field(
        default=1.0,
        metadata={
            "help": "Success threshold τ. Samples with reward >= τ are added to the shared success buffer."
        },
    )
    cross_policy_success_buffer_size: int = field(
        default=10_000,
        metadata={"help": "Maximum number of (prompt, completion, reward, src) items to keep in the success buffer."},
    )
    cross_policy_success_buffer_dedup: bool = field(
        default=True,
        metadata={"help": "Whether to deduplicate success buffer entries by (prompt, completion)."},
    )

    # Pooled advantages across union group 𝒴(x)
    cross_policy_advantage_mode: str = field(
        default="zscore",
        metadata={
            "help": "How to compute pooled advantages over the union of all rollouts across policies for a prompt. "
            "Supported: 'zscore' (default), 'rank', 'none'."
        },
    )

    # Cross-policy SFT scheduling / mixing
    cross_policy_interval: int = field(
        default=1,
        metadata={
            "help": "Cross-policy interval s. If s==1, mix cross-policy SFT into every GRPO step (stage 6 special-case). "
            "If s!=1, run cross-policy SFT-only updates every s GRPO iterations (stage 7)."
        },
    )
    cross_policy_sft_steps: int = field(
        default=0,
        metadata={"help": "Number of cross-policy SFT-only gradient steps to run at each interval when s!=1."},
    )
    cross_policy_sft_batch_size: int = field(
        default=0,
        metadata={
            "help": "How many success-buffer samples (from other policies) to use per SFT step. 0 disables SFT steps."
        },
    )
    cross_policy_mix_alpha: float = field(
        default=0.0,
        metadata={
            "help": "Cross-policy mix weight α for the s==1 case. The mixed objective is "
            "L = (1-α)*L_GRPO + α*L_SFT. Set to 0 to disable mixing."
        },
    )


