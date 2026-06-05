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

---

## 6. Training Dynamics

**Reward progression**:
- GRPO: reward tăng đều từ 0.35 lên 0.72 trong 3000 steps, không có reward hacking
- DPO: implicit reward (log-ratio) tăng nhanh hơn nhưng saturate sớm hơn

**KL divergence**:
- GRPO: KL giữ ổn định quanh 0.04 (beta=0.001, nhẹ penalty)
- DPO: implicit KL (log-ratio) giảm dần, cho thấy policy drift khỏi reference

**Entropy**:
- GRPO: entropy giảm chậm (0.15/step), policy giữ diversity tốt
- DPO: entropy giảm nhanh hơn (0.25/step), policy specialize nhanh hơn nhưng dễ mode collapse

---

## 7. VRAM Breakdown

| Component | DPO | GRPO |
|:---|:---|:---|
| Policy model | 14GB (bf16) | 14GB (bf16) |
| Reference model | 14GB (frozen) | 14GB (frozen) |
| Optimizer states | 8GB (AdamW) | 8GB (AdamW) |
| Activations | 6GB | 6GB |
| Generation KV cache | - | 12GB (8 seq x 1024 tok) |
| Loss computation | - | 4GB (per-token logps) |
| **Total** | **42GB** | **58GB** |

VRAM khác biệt chính: GRPO cần thêm KV cache cho generation phase và per-token logps tensor cho loss computation.

---

## 8. Hyperparameter Sensitivity

| Parameter | DPO sensitive | GRPO sensitive |
|:---|:---|:---|
| `beta` (KL penalty) | Cao: 0.05-0.5 optimal | Thấp: 0.001-0.1 đều ok |
| `learning_rate` | Trung bình: 1e-7 đến 1e-6 | Trung bình: 5e-7 đến 5e-6 |
| `num_generations` | N/A | Cao: 4-16 optimal |
| `max_completion_length` | N/A | Trung bình: 512-1024 |

DPO nhạy cảm với `beta` hơn vì nó trực tiếp ảnh hưởng đến strength của preference signal.

---

## 9. Decision Guide

| Scenario | Recommended | Reason |
|:---|:---|:---|
| Compute budget tight | DPO | 1.5x faster, ít VRAM hơn |
| Maximum accuracy | GRPO | +6.6% accuracy, on-policy learning |
| Preference data có sẵn | DPO | Không cần reward function |
| Rule-based reward có sẵn | GRPO | Tận dụng exact reward signal |
| Production deployment | DPO | Đơn giản, predictable |
| Research exploration | GRPO | Nhiều knobs để tune |
| Single GPU | DPO | 42GB fit trong 1x A100 |
| Multi-GPU | GRPO | Tận dụng generation parallelism |
