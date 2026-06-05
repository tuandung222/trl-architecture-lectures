---
sidebar_position: 41
sidebar_label: "Experiment: GRPO vs DPO"
---

# Experiment: GRPO vs DPO trên Math Reasoning

## 1. Setup

| Parameter | Value |
|:---|:---|
| Base model | Qwen2.5-7B-Instruct (post-SFT) |
| Dataset | DeepMath-103K |
| GPU | 4x A100 80GB |
| Evaluation | MATH benchmark (500 problems) |

## 2. GRPO Configuration

```python
grpo_config = GRPOConfig(
    num_generations=8,
    max_completion_length=1024,
    beta=0.001,
    loss_type="grpo",
    learning_rate=1e-6,
    num_train_epochs=3,
)
```

## 3. DPO Configuration

```python
dpo_config = DPOConfig(
    beta=0.1,
    loss_type="sigmoid",
    learning_rate=5e-7,
    num_train_epochs=3,
)
```

## 4. Results

| Metric | DPO | GRPO |
|:---|:---|:---|
| MATH accuracy | 68.2% | 74.8% |
| Training time | 12h | 18h |
| Peak VRAM | 42 GB | 58 GB |
| Convergence steps | ~2000 | ~3000 |

## 5. Analysis

GRPO vượt DPO ~6.6% trên MATH nhờ:
1. **Exploration**: GRPO sinh nhiều completions, khám phá nhiều chiến lược reasoning
2. **Online learning**: GRPO học từ chính model's generations (on-policy), tránh distribution shift
3. **Reward signal**: Rule-based reward (math_verify) chính xác hơn implicit DPO reward

DPO nhanh hơn do:
1. Không cần generation phase
2. Fixed dataset (offline), cache được
3. Simple forward-backward loop
