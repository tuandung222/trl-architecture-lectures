---
sidebar_position: 6.5
sidebar_label: "Bài 4b: RLOO Trainer"
---

# Bài 4b: RLOO (REINFORCE Leave-One-Out) Trainer

Sau khi hiểu sâu GRPO ([Bài 4](./lesson_4_grpo_internals.md)), bài này giới thiệu một phương pháp estimate advantage khác trong TRL: **RLOO** (REINFORCE Leave-One-Out). `RLOOTrainer` (1548 dòng, file `rloo_trainer.py`) là "người anh em" của `GRPOTrainer`, cùng chia sẻ generation pipeline nhưng khác ở cách tính baseline.

---

## 1. Leave-One-Out Baseline: Tại sao cần?

### 1.1. Vấn đề của GRPO Group Mean

GRPO dùng group mean làm baseline:

$$b_i^{\text{GRPO}} = \frac{1}{G} \sum_{j=1}^{G} r_j$$

Baseline này **biased** vì reward của chính completion $i$ được bao gồm trong baseline. Điều này tạo ra **negative bias**: completion tốt bị đánh giá thấp hơn thực tế vì chính nó kéo baseline lên.

### 1.2. Leave-One-Out: Unbiased Baseline

RLOO loại bỏ reward của chính completion $i$ khỏi baseline:

$$b_i^{\text{RLOO}} = \frac{1}{G-1} \sum_{j \neq i} r_j$$

Advantage RLOO:

$$A_i^{\text{RLOO}} = r_i - \frac{1}{G-1} \sum_{j \neq i} r_j$$

**Tính chất quan trọng**:
- **Unbiased**: baseline không phụ thuộc vào reward của chính completion
- **Lower variance** (khi G nhỏ): leave-one-out estimator có variance thấp hơn single-sample baseline
- **Cùng expected value**: $\mathbb{E}[A_i^{\text{RLOO}}] = \mathbb{E}[A_i^{\text{GRPO}}]$

### 1.3. Ví dụ số

Với G=4, rewards = [1.0, 0.5, 0.8, 0.3]:

| Completion | Reward | GRPO Baseline | GRPO Adv | RLOO Baseline | RLOO Adv |
|:---|:---|:---|:---|:---|:---|
| 1 | 1.0 | 0.65 | +0.35 | 0.53 | +0.47 |
| 2 | 0.5 | 0.65 | -0.15 | 0.70 | -0.20 |
| 3 | 0.8 | 0.65 | +0.15 | 0.60 | +0.20 |
| 4 | 0.3 | 0.65 | -0.35 | 0.77 | -0.47 |

RLOO advantages có magnitude lớn hơn GRPO, tức signal mạnh hơn cho gradient update.

---

## 2. Advantage Computation trong TRL

### 2.1. Code Walkthrough (`rloo_trainer.py`, lines 1297-1314)

```python
# rloo_trainer.py: Leave-one-out advantage computation
# grouped_rewards: [num_prompts, num_generations]

# Đếm số completion có reward hợp lệ (không NaN)
scorable_counts = (~torch.isnan(grouped_rewards)).sum(dim=1, keepdim=True)

# Tổng rewards (bỏ qua NaN)
grouped_sum = torch.nansum(grouped_rewards, dim=1, keepdim=True)

# Leave-one-out baseline: (sum - self) / (count - 1)
baselines = (grouped_sum - grouped_rewards) / (scorable_counts - 1)
advantages = rewards - baselines.view(-1)

# Normalize advantages (optional)
if self.normalize_advantages:
    advantages = (advantages - nanmean(advantages)) / (nanstd(advantages) + 1e-4)

# Zero out unscorable completions (NaN rewards)
advantages = torch.nan_to_num(advantages, nan=0.0)
```

**Điểm đáng chú ý**:
- `nansum` và `scorable_counts` xử lý trường hợp một số completions có reward NaN (ví dụ: async reward timeout)
- Khi chỉ có 1 scorable completion trong group: `(count - 1) = 0` dẫn đến `0/0 = NaN`, sau đó `nan_to_num` zero out
- `normalize_advantages` normalize trên toàn bộ advantages (không per-group)

### 2.2. So sánh với GRPO

```python
# GRPO (grpo_trainer.py): Group mean baseline
grouped = rewards.reshape(B, num_generations)
mean = grouped.mean(dim=1, keepdim=True)
std = grouped.std(dim=1, keepdim=True)
advantages = (grouped - mean) / (std + 1e-8)

# RLOO (rloo_trainer.py): Leave-one-out baseline
baselines = (grouped_sum - grouped_rewards) / (scorable_counts - 1)
advantages = grouped_rewards - baselines
# Normalize separately (optional)
```

Khác biệt chính: GRPO normalize **trong công thức advantage** (chia std), RLOO normalize **tùy chọn** sau khi tính advantage.

---

## 3. Sequence-Level Loss (Khác biệt quan trọng)

### 3.1. GRPO: Token-Level Clipped Surrogate

GRPO tính probability ratio **tại mỗi token**, rồi clip và aggregate:

```python
# GRPO: per-token ratio, rồi masked mean
log_ratio = per_token_logps - old_per_token_logps  # [B, T]
ratio = torch.exp(log_ratio)
loss = -torch.min(ratio * advantages, clamp(ratio) * advantages)
loss = (loss * completion_mask).sum() / completion_mask.sum()
```

### 3.2. RLOO: Sequence-Level Clipped Surrogate

RLOO sum log probs **toàn bộ completion** trước, rồi tính ratio ở sequence level (`rloo_trainer.py`, lines 1427-1438):

```python
# RLOO: sequence-level ratio
logps = (per_token_logps * completion_mask).sum(1)  # [B] sum per sequence
old_logps = inputs["old_logps"]
log_ratio = logps - old_logps  # [B]

coef_1 = torch.exp(log_ratio)
coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
per_sequence_loss1 = coef_1 * advantages
per_sequence_loss2 = coef_2 * advantages
loss = -torch.min(per_sequence_loss1, per_sequence_loss2).mean()
```

### 3.3. Tại sao khác nhau?

| Aspect | GRPO (Token-level) | RLOO (Sequence-level) |
|:---|:---|:---|
| Ratio granularity | Per-token | Per-sequence |
| Clipping | Per-token | Per-sequence |
| completion_mask trong loss | Dùng để weighted mean | Dùng để sum logps |
| KL penalty | Thêm vào loss | Trừ vào reward |
| Loss variants | 8 types | 1 type (standard clip) |

Sequence-level ratio đơn giản hơn nhưng mất thông tin về token-level dynamics. Với completions dài, một vài tokens có ratio rất cao nhưng trung bình sequence-level ratio vẫn bình thường, token-level clipping sẽ bắt được điều này.

---

## 4. KL Penalty: Trừ vào Reward (Không Thêm vào Loss)

### 4.1. GRPO Approach

```python
# GRPO: KL penalty thêm vào loss function
per_token_loss = -min(surr1, surr2) + beta * kl_approx
```

### 4.2. RLOO Approach

```python
# RLOO (rloo_trainer.py, line 1282-1288): KL trừ trực tiếp vào reward
per_token_kl = old_per_token_logps - ref_per_token_logps
kl = (per_token_kl * completion_mask).sum(-1)  # Sum KL per sequence
rewards = rewards - beta * kl  # Subtract from reward
```

### 4.3. Mathematical Equivalence

Cả hai approaches đều tương đương về mặt toán học:

$$\mathbb{E}[\nabla_\theta (r - \beta \cdot \text{KL})] = \mathbb{E}[\nabla_\theta r] - \beta \cdot \mathbb{E}[\nabla_\theta \text{KL}]$$

Nhưng implementation khác nhau:
- **GRPO**: KL penalty tác động lên **gradient của loss**, mỗi token bị penalize proportional với KL
- **RLOO**: KL penalty tác động lên **reward signal**, sau đó advantage computation lan truyền signal này

---

## 5. Configuration và Usage

### 5.1. RLOOConfig Key Parameters

```python
from trl import RLOOConfig

config = RLOOConfig(
    # Generation
    num_generations=4,           # G completions per prompt
    max_completion_length=256,   # Max tokens per completion
    temperature=1.0,             # Sampling temperature
    
    # KL penalty
    beta=0.04,                   # KL coefficient (0 = no KL)
    
    # Clipping
    epsilon_low=0.2,             # Lower clip bound
    epsilon_high=0.2,            # Upper clip bound
    
    # Advantage
    normalize_advantages=True,   # Normalize advantages globally
    
    # Multi-step updates
    num_iterations=1,            # Number of updates per generation batch
    
    # vLLM
    use_vllm=False,              # Use vLLM for generation
)
```

### 5.2. Training Example

```python
from trl import RLOOTrainer, RLOOConfig
from datasets import load_dataset

trainer = RLOOTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    args=RLOOConfig(
        num_generations=4,
        max_completion_length=256,
        beta=0.04,
        learning_rate=1e-6,
        num_train_epochs=3,
        bf16=True,
    ),
    reward_funcs=[accuracy_reward, format_reward],
    train_dataset=load_dataset("openai/gsm8k", "main", split="train"),
    processing_class=tokenizer,
)
trainer.train()
```

API gần như identical với `GRPOTrainer`, chỉ khác ở `num_generations` default (2 thay vì 16) và không có `loss_type` parameter.

---

## 6. So sánh RLOO vs GRPO vs PPO

| Feature | RLOO | GRPO | PPO |
|:---|:---|:---|:---|
| **Baseline estimator** | Leave-one-out | Group mean | Critic network |
| **Models needed** | Policy + Reference | Policy + Reference | Policy + Ref + Critic + Reward |
| **Loss granularity** | Sequence-level | Token-level | Token-level |
| **KL handling** | Subtract from reward | Add to loss | Add to reward |
| **Loss variants** | 1 (standard clip) | 8 types | 1 (standard clip) |
| **Advantage normalization** | Optional (global) | Built-in (per-group) | GAE-based |
| **vLLM support** | Yes | Yes | No (experimental) |
| **Complexity** | Lower | Higher | Highest |
| **Variance** | Lower (LOO) | Moderate | Lower (critic) |
| **Memory** | 2 models | 2 models | 4 models |

---

## 7. Khi nào dùng RLOO?

**Ưu điểm**:
1. **Lower variance** hơn GRPO khi G nhỏ (2-4 generations)
2. **Đơn giản hơn GRPO**: ít loss variants, ít config parameters
3. **Memory efficient**: chỉ cần 2 models (policy + reference), không cần critic
4. **Research-friendly**: baseline rõ ràng, dễ modify và experiment

**Nhược điểm**:
1. **Sequence-level loss** mất token-level information
2. **Không có loss variants** để adapt cho different task types
3. **Higher variance** hơn PPO (PPO có critic network làm baseline)

**Recommended khi**:
- Bạn muốn một alternative đơn giản cho GRPO
- G nhỏ (2-4 generations per prompt)
- Research experiments cần so sánh baseline estimators
- Bạn đã hiểu GRPO và muốn explore different advantage estimation strategies

---

## Xem thêm

- [Bài 4: GRPO Internals](./lesson_4_grpo_internals.md): GRPO group-relative advantage
- [Lý thuyết 2: GRPO Math](./theory_deep_dive/theory_2_grpo_math.md): Derivation chi tiết
- [Lý thuyết 4: KL Divergence](./theory_deep_dive/theory_4_kl_divergence.md): KL approximation methods
