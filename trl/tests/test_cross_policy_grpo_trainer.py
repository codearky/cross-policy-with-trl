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

import torch


def test_cross_policy_exports_importable():
    # Ensure lazy exports are wired.
    from trl import CrossPolicyGRPOConfig, CrossPolicyGRPOTrainer  # noqa: F401


def test_success_buffer_dedup_and_exclude_src():
    from trl.trainer.cross_policy_grpo_trainer import SuccessBuffer, SuccessBufferItem

    buf = SuccessBuffer(max_size=10, dedup=True, seed=0)

    it0 = SuccessBufferItem(prompt="p", completion="c", reward=1.0, src=0)
    it1 = SuccessBufferItem(prompt="p", completion="c", reward=1.0, src=1)
    it2 = SuccessBufferItem(prompt="p2", completion="c2", reward=1.0, src=1)

    assert buf.add(it0) is True
    # Dedup: same (prompt, completion) should be dropped, regardless of src
    assert buf.add(it1) is False
    assert buf.add(it2) is True

    # Exclude src=1 → should return only src=0 samples
    sampled = buf.sample(10, exclude_src=1)
    assert sampled
    assert all(x.src != 1 for x in sampled)


def test_pooled_advantages_zscore_and_rank():
    from trl.trainer.cross_policy_grpo_trainer import pooled_advantages

    # Constant rewards → zero advantages for zscore
    rewards = torch.ones(2, 3, 4)
    adv = pooled_advantages(rewards, mode="zscore")
    assert adv.shape == rewards.shape
    assert torch.allclose(adv, torch.zeros_like(adv))

    # Rank mode: should be finite and in [-0.5, 0.5]
    rewards = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)
    adv_rank = pooled_advantages(rewards, mode="rank")
    assert adv_rank.shape == rewards.shape
    assert torch.isfinite(adv_rank).all()
    assert adv_rank.min().item() >= -0.5 - 1e-6
    assert adv_rank.max().item() <= 0.5 + 1e-6


