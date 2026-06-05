---
sidebar_position: 22
sidebar_label: "Toán học GRPO"
---

# Phân tích toán học GRPO (Group Relative Policy Optimization)

## 1. Motivation: Tại sao loại bỏ Critic?

Trong PPO, Critic $V_\phi$ có kích thước tương đương Actor. VRAM cho 4 models:

$$\text{VRAM}_{PPO} = 2 \times \text{Model}_{Actor} + \text{Model}_{Critic} + \text{Model}_{Reward} \approx 4M$$

GRPO thay Critic bằng group mean reward:

$$\text{VRAM}_{GRPO} = 2 \times \text{Model}_{Actor} + \text{Model}_{Reward} \approx 2.5M$$

Tiết kiệm ~30-40% VRAM, cho phép batch size lớn hơn hoặc model lớn hơn.

---

## 2. Group-Relative Advantage Estimator

### 2.1. Định nghĩa

Với prompt $q$, sinh $G$ responses $\{o_1, ..., o_G\}$ từ $\pi_\theta$. Đánh giá reward $\{r_1, ..., r_G\}$.

$$\hat{A}_i^{GRPO} = \frac{r_i - \frac{1}{G}\sum_{j=1}^{G} r_j}{\sqrt{\frac{1}{G-1}\sum_{j=1}^{G}(r_j - \bar{r})^2} + \epsilon}$$

### 2.2. Tính chất thống kê

* $\mathbb{E}[\hat{A}_i] \approx 0$: advantage trung bình bằng 0 (unbiased khi $G \to \infty$)
* $\text{Var}[\hat{A}_i] \approx 1$: normalized variance
* Khi $G$ nhỏ (4-8): ước lượng có noise, nhưng practice cho thấy đủ tốt

### 2.3. So sánh với PPO baseline

| | PPO | GRPO |
|:---|:---|:---|
| Baseline | $V_\phi(s_t)$ (learned) | $\bar{r}$ (sampled) |
| Bias | Function approximation bias | Monte Carlo bias (small for large G) |
| Variance | Low (learned value) | Higher (sample-based) |
| Compute cost | Extra model forward/backward | Chỉ reward computation |

---

## 3. GRPO Objective

$$J_{GRPO}(\theta) = \mathbb{E}_{q, \{o_i\}_{i=1}^G} \left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\left\{\min(\rho_{i,t} \hat{A}_i, \text{clip}(\rho_{i,t}, 1-\epsilon, 1+\epsilon)\hat{A}_i) - \beta D_{KL}\right\}\right]$$

### 3.1. Per-token vs per-sequence normalization

Công thức trên normalize per-token ($\frac{1}{|o_i|}$). DAPO thay đổi thành per-completion-token normalization trên toàn batch.

### 3.2. Multi-step updates ($\mu > 1$)

Khi $\mu > 1$, cùng một batch completions được dùng $\mu$ lần. Importance ratio:

$$\rho_{i,t} = \frac{\pi_\theta(o_{i,t} | q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i,<t})}$$

$\theta_{old}$ là policy tại thời điểm generation. Sau $\mu$ updates, $\theta$ lệch xa khỏi $\theta_{old}$, nên clipping $\epsilon$ giới hạn damage.

---

## 4. Variance Reduction

### 4.1. Group size effect

Variance của advantage estimator giảm theo $G$:

$$\text{Var}[\hat{A}_i] \propto \frac{1}{G}$$

Tuy nhiên, $G$ lớn hơn = nhiều generation hơn = chậm hơn. Tradeoff tối ưu thường $G \in [4, 16]$.

### 4.2. Clipping as variance control

Clipping $\epsilon$ giới hạn magnitude của mỗi gradient step:

$$|\nabla_\theta L| \leq (1+\epsilon) |\hat{A}_i| \cdot |\nabla_\theta \log \pi_\theta|$$

Ngăn catastrophic updates khi advantage estimator có high variance.
