---
sidebar_position: 20
sidebar_label: "Lộ trình Theory Deep Dive"
---

# Lộ trình Theory Deep Dive

Phần này cung cấp các bài phân tích toán học chuyên sâu về các thuật toán và kỹ thuật được sử dụng trong TRL.

---

## Danh sách bài viết

### [Phân tích toán học PPO](theory_1_ppo_math)
Chi tiết GAE, clipped surrogate, value function loss, và policy gradient theorem trong ngữ cảnh RLHF.

### [Phân tích toán học GRPO](theory_2_grpo_math)
Chứng minh group-relative advantage estimator, so sánh với PPO baseline, variance reduction analysis.

### [Phân tích toán học DPO](theory_3_dpo_math)
Chứng minh closed-form solution từ RL objective, Bradley-Terry model, connection đến RLHF.

### [KL Divergence Variants](theory_4_kl_divergence)
So sánh các KL approximation: Schulman, forward KL, reverse KL, và bias correction trong importance sampling.

### [Loss Variants Analysis](theory_5_loss_variants)
Phân tích chi tiết 8 loss types trong GRPOTrainer: GRPO, DAPO, CISPO, VESPO, SAPO, BNPO, DR-GRPO, LUSPO.
