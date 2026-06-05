---
sidebar_position: 24
sidebar_label: "KL Divergence Variants"
---

# KL Divergence Variants trong TRL

## 1. Exact KL Divergence

$$D_{KL}(\pi_\theta \| \pi_{ref}) = \sum_a \pi_\theta(a|s) \log \frac{\pi_\theta(a|s)}{\pi_{ref}(a|s)}$$

Tính exact KL cho toàn bộ vocabulary là $O(V)$ per token, quá đắt với V = 128K.

---

## 2. Schulman KL Approximation (TRL default)

$$D_{KL}^{Schulman}(t) = \frac{\pi_{ref}(t)}{\pi_\theta(t)} - \log\frac{\pi_{ref}(t)}{\pi_\theta(t)} - 1$$

Đây là xấp xỉ bậc hai: $e^x - x - 1 \approx \frac{x^2}{2}$ khi $x \approx 0$.

**Tính chất**:
* Luôn $\geq 0$ (do $e^x \geq x + 1$)
* Bằng 0 khi $\pi_\theta = \pi_{ref}$
* Per-token computation: $O(1)$ thay vì $O(V)$

Trong TRL GRPOTrainer:
```python
per_token_kl = torch.exp(ref_logps - per_token_logps) - (ref_logps - per_token_logps) - 1
```

---

## 3. Forward KL vs Reverse KL

* **Forward KL**: $D_{KL}(\pi_\theta \| \pi_{ref})$ - phạt khi $\pi_\theta$ đặt mass nơi $\pi_{ref}$ không có
* **Reverse KL**: $D_{KL}(\pi_{ref} \| \pi_\theta)$ - phạt khi $\pi_{ref}$ đặt mass nơi $\pi_\theta$ không có

RLHF thường dùng forward KL (penalty khi policy drift khỏi reference).

---

## 4. Bias Correction với Importance Sampling

Khi dùng vLLM, logprobs từ generation ($\pi_{old}$) khác với training ($\pi_\theta$). KL divergence cần hiệu chỉnh:

$$D_{KL}^{corrected} = D_{KL}^{Schulman} \times \frac{\pi_{old}(t)}{\pi_\theta(t)}$$

Trong TRL:
```python
if self.args.use_bias_correction_kl:
    per_token_kl = per_token_kl * coef_1  # coef_1 = IS ratio
```

---

## 5. Token-level vs Sequence-level KL

* **Token-level**: KL penalty cộng vào mỗi token reward (PPO style)
* **Sequence-level**: KL penalty trung bình trên toàn sequence (GRPO style)

$$D_{KL}^{seq} = \frac{1}{T}\sum_{t=1}^{T} D_{KL}(t)$$

GRPO sử dụng sequence-level KL trong loss:
```python
loss = per_token_loss + self.beta * per_token_kl  # broadcast [B,1] to [B,T]
```

---

## 6. So sánh số 4 phương pháp

Bảng dưới đây minh họa giá trị KL divergence khi $r = \frac{\pi_{ref}(t)}{\pi_\theta(t)}$ thay đổi:

| Ratio $r$ | Exact KL | Schulman ($e^x - x - 1$) | Forward KL | Reverse KL |
|:---|:---|:---|:---|:---|
| 0.5 | 0.193 | 0.148 | 0.193 | 0.307 |
| 0.8 | 0.023 | 0.020 | 0.023 | 0.028 |
| 1.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1.2 | 0.018 | 0.017 | 0.018 | 0.022 |
| 2.0 | 0.307 | 0.307 | 0.307 | 0.193 |
| 5.0 | 3.390 | 144.5 | 3.390 | 0.214 |

Nhận xét:
- **Schulman** gần đúng khi $r \approx 1$ (policy gần reference) nhưng **overestimate** khi $r$ lớn (policy drift xa)
- **Forward vs Reverse KL**: symmetric tại $r=1$, asymmetric tại $r$ lớn
- **Schulman overestimate** tại $r=5$ vì $e^x$ tăng exponential: đây là tính chất, không phải lỗi, nó phạt mạnh khi policy drift xa

---

## 7. KL trong từng Trainer

| Trainer | KL Method | Cách sử dụng | Code ref |
|:---|:---|:---|:---|
| GRPOTrainer | Schulman approx | Cộng vào loss (per-token) | `grpo_trainer.py` |
| RLOOTrainer | Exact (log ratio) | Trừ vào reward (per-sequence) | `rloo_trainer.py` |
| PPOTrainer | Token-level KL | Trừ vào reward (per-token) | `ppo_trainer.py` |
| DPOTrainer | Implicit | Không tính KL trực tiếp | `dpo_trainer.py` |

**Tại sao DPO không cần KL**: DPO đã implicit regularize KL qua công thức loss. Tham số $\beta$ trong DPO chính là KL penalty coefficient, nhưng không cần tính KL divergence vì nó được embedded trong log-ratio.

---

## 8. Practical Implications cho Beta Tuning

### 8.1. Khi nào beta = 0 an toàn?

- Khi training ngắn (vài trăm steps) và policy không drift xa reference
- Khi dùng LoRA với rank nhỏ (limited capacity = limited drift)
- Khi dataset lớn và model nhỏ (model không thể overfit)

### 8.2. Khi nào bias correction quan trọng?

- Khi dùng vLLM (logprobs khác nhau giữa generation và training)
- Khi `num_iterations > 1` (nhiều update steps trên cùng generation batch)
- Khi learning rate cao (policy drift nhanh)

### 8.3. Schulman vs Exact KL: Trade-off

| Factor | Schulman | Exact |
|:---|:---|:---|
| Speed | O(1) per token | O(1) per token |
| Accuracy | Approximate | Exact (single token) |
| Overestimation | Có khi $r$ lớn | Không |
| Default trong TRL | Có | Không |
| Recommendation | Production training | Research experiments |
