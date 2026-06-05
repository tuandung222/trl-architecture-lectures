---
sidebar_position: 40
sidebar_label: "Lộ trình Experiments"
---

# Lộ trình Experiments Deep Dive

Các bài phân tích thực nghiệm so sánh hiệu năng giữa các cấu hình TRL.

---

## Danh sách Experiments

### [GRPO vs DPO Comparison](exp_1_grpo_vs_dpo)
So sánh GRPO và DPO trên math reasoning tasks: chất lượng, tốc độ hội tụ, và VRAM usage.

### [Loss Variants Benchmark](exp_2_loss_variants_benchmark)
Benchmark 8 loss types trong GRPOTrainer trên cùng dataset và model.

### [vLLM Speedup Analysis](exp_3_vllm_speedup)
Đo tốc độ generation với và không có vLLM, phân tích importance sampling correction overhead.
