---
sidebar_position: 25
sidebar_label: "Loss Variants Analysis"
---

# Phân tích 8 Loss Types trong GRPOTrainer

## 1. Standard GRPO (`loss_type="grpo"`)

$$L = -\frac{1}{N}\sum_{i}\min(\rho_i A_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)A_i)$$

Two-sided clipping. Normalization: mean trên completion tokens.

---

## 2. DAPO (`loss_type="dapo"`)

Dynamic Advantage Per-token Optimization: advantage được broadcast per-token thay vì per-sequence.

$$L = -\frac{1}{\sum M_i}\sum_{i}\sum_t M_{i,t}\min(\rho_{i,t} A_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)A_i)$$

Normalization: chia cho tổng completion tokens trên toàn batch (thay vì mean per-sample).

---

## 3. CISPO (`loss_type="cispo"`)

Clipped Importance Sampling Policy Optimization. One-sided clipping:

$$L = -\text{clamp}(\rho, \max=1+\epsilon) \cdot A \cdot \log\pi_\theta$$

**Đặc điểm**: Không clip từ dưới, chỉ giới hạn IS ratio từ trên. Dùng gradient của log-prob trực tiếp thay vì ratio x advantage.

---

## 4. VESPO (`loss_type="vespo"`)

Variance-reduced Expected Sequence Policy Optimization với gamma weights:

$$L = -\phi(w) \cdot A \cdot \log\pi_\theta$$

$$\phi(w) = e^\lambda \cdot w^k \cdot e^{-\lambda w}$$

$\phi(1) = 1$, $\phi(w)$ giảm khi $w$ lệch xa 1, giảm variance từ off-policy correction.

---

## 5. SAPO (`loss_type="sapo"`)

Soft Annealed Policy Optimization. Sigmoid thay thế hard clamp:

$$L = -\frac{4}{\tau}\sigma(\tau(\rho - 1)) \cdot A$$

Temperature $\tau$ khác nhau cho positive/negative advantages:
* $\tau_{pos}$: sharp sigmoid cho positive advantage (khuyến khích)
* $\tau_{neg}$: gentle sigmoid cho negative advantage (ức chế nhẹ)

---

## 6. BNPO (`loss_type="bnpo"`)

Batch-Normalized PPO. Tương tự GRPO nhưng normalization khác:

$$\hat{A}_i = \frac{A_i - \text{mean}_{batch}(A)}{\text{std}_{batch}(A) + \epsilon}$$

Normalize advantage trên toàn batch thay vì per-group.

---

## 7. DR-GRPO (`loss_type="dr_grpo"`)

Doubly-Robust GRPO. Thêm delta clipping:

$$\rho_{clipped} = \text{clip}(\rho, \min=0, \max=\delta)$$

Sau đó áp dụng standard two-sided clipping trên $\rho_{clipped}$. Giảm ảnh hưởng của outlier ratios.

---

## 8. LUSPO (`loss_type="luspo"`)

Length-Unbiased Sequence-level PO. Sequence-level importance sampling:

$$w_{seq} = \exp\left(\frac{1}{T}\sum_t \log\frac{\pi_\theta(t)}{\pi_{old}(t)}\right)$$

Thay vì token-level IS ratio, dùng geometric mean của ratios trên toàn sequence. Tránh bias từ completion length.

---

## So sánh tổng quan

| Loss | Clipping | IS Level | Normalization | Đặc điểm nổi bật |
|:---|:---|:---|:---|:---|
| GRPO | Two-sided | Token | Per-sample mean | Standard |
| DAPO | Two-sided | Token | Batch total tokens | Dynamic normalization |
| CISPO | One-sided (upper) | Token | Per-sample | Gradient-based |
| VESPO | Gamma weights | Token/Seq | Per-sample | Variance reduction |
| SAPO | Soft (sigmoid) | Token | Per-sample | Smooth clipping |
| BNPO | Two-sided | Token | Batch mean | Batch normalization |
| DR-GRPO | Delta + two-sided | Token | Per-sample | Robust to outliers |
| LUSPO | Two-sided | Sequence | Per-sample | Length-unbiased |
