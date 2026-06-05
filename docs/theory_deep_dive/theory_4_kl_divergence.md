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
