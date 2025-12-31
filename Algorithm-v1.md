# Cross-Policy GRPO with Verified-Success Sharing (CP-GRPO-VS)

This document specifies an algorithm to **jointly train multiple heterogeneous LLM policies** (e.g., Llama + Qwen) in an **RL with Verifiable Rewards (RLVR)** setting, such that each policy benefits from **verified successful trajectories discovered by the others**, while keeping training stable and preventing collapse.

---

## 1. Setting

We assume a dataset/distribution of prompts:
- \(x \sim \mathcal{D}\)

Each policy \(\pi_i\) (for \(i \in \{1,\dots,N\}\)) produces a text trajectory \(y\) conditioned on \(x\):
- \(y \sim \pi_i(\cdot \mid x)\)

A **verifier** (programmatic checker / unit tests / exact-match answer checker) returns a reward:
- \(r(x,y) \in \mathbb{R}\) (often \( \{0,1\} \) in RLVR)

Each policy has a fixed **reference** \(\pi^{\text{ref}}_i\) (initial checkpoint or periodically updated anchor) for KL regularization.

Goal: improve each \(\pi_i\) under verifier reward while enabling **cross-policy transfer** from verified successes.

---



## 2. Algorithm (pseudocode)

```text
Algorithm: GRPO-CP

Inputs:
  - N policies {π_i} with references {π_ref_i}
  - verifier reward function r(x,y)
  - prompt distribution 𝒟
  - rollouts per policy per prompt K
  - thresholds: success τ
  - weights: λ_KL, α (cross policy mix)
  - success buffer ℬ (initially empty)
  - cross policy interval s
  - cross policy sft steps

Repeat for iterations t = 1..T:
  1) Sample batch of prompts B = {x_b} ~ 𝒟

  2) Rollout:
     For each prompt x_b:
       For each policy i:
         Sample K trajectories y_{i,b,1..K} ~ π_i(.|x_b)

  3) Verify rewards:
     For each x_b and each y in union group 𝒴(x_b):
       compute reward r_b(y) = r(x_b, y)

  4) Update shared success buffer:
     For each successful sample (x_b, y) with r_b(y) >= τ:
       Add (x_b, y, r_b(y), src=i) to ℬ
     Apply hygiene: deduplicate, etc.

  5) Compute pooled advantages for each prompt x_b:
     Let 𝒴(x_b) be the union of all rollouts across i
     Compute A_b(y) using Z-score or rank advantage over 𝒴(x_b)
     Note: this is optional. We can strat without by default and experiment with that later.

  6) GRPO update (per policy, on its own samples):
     For each policy i:
       L_GRPO_i = - E[ A_b(y) * log π_i(y|x) ] + λ_KL * KL(π_i || π_ref_i) (regular grpo loss)
       if cross policy interval s == 1, at every GRPO step:
          L_GRPO_i = compute grpo loss
          Sample min(`sft_batch_size`, <maximum valid samples from other policy in the buffer>) (x, y, r, src) from shared success buffer with src\not{=}i
          L_SFT = <sft_loss_with_respect_to_the_batch>
          L_ALL = mix L_GRPO with L_SFT (with correct maximization/minimization of the mixed objectives, i.e. correct sign)
          skip stage 7


  7) Success-sharing preference update (per policy), once in a s steps, and only if cross policy interval s != 1:
     For each policy i:
       For 0,..., `sft_steps`
         batch = Sample min(`sft_batch_size`, <maximum valid samples from other policy in the buffer>) (x, y, r, src) from shared success buffer with src\not{=}i
         if batch is empty:
           finish stage 7
         L_SFT_i = <sft_loss_with_respect_to_batch>
         take gradient step with respect to L_SFT_i


       


  8) Optional diversity regularization:
     Apply small repulsive term or heuristics (temperature, balancing, etc.)

Return trained policies {π_i}
