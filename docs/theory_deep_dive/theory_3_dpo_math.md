---
sidebar_position: 23
sidebar_label: "Toán học DPO"
---

# Phân tích toán học DPO (Direct Preference Optimization)

## 1. Derivation từ RL Objective

Bắt đầu từ KL-constrained RL:

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}} \mathbb{E}_{y \sim \pi(y|x)} [r(x,y)] - \beta D_{KL}(\pi(y|x) \| \pi_{ref}(y|x))$$

### 1.1. Optimal policy (closed-form)

Sử dụng variational calculus, optimal policy có dạng:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

với $Z(x) = \sum_y \pi_{ref}(y|x) \exp(r(x,y)/\beta)$ là partition function.

### 1.2. Reward reparameterization

Nghịch đảo công thức trên:

$$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**Key insight**: Reward được biểu diễn qua log probability ratio.

---

## 2. Bradley-Terry Preference Model

Preference data: $(x, y_w, y_l)$ với $y_w$ preferred over $y_l$.

$$p(y_w \succ y_l | x) = \frac{\exp(r(x, y_w))}{\exp(r(x, y_w)) + \exp(r(x, y_l))} = \sigma(r(x, y_w) - r(x, y_l))$$

Thay reward bằng reparameterization:

$$r(x, y_w) - r(x, y_l) = \beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

**Partition function $Z(x)$ triệt tiêu hoàn toàn**, đây là bước đột phá của DPO.

---

## 3. DPO Loss

$$L_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

### 3.1. Gradient analysis

$$\nabla_\theta L_{DPO} = -\beta \cdot \sigma(-\hat{r}) \cdot \left(\nabla_\theta \log\pi_\theta(y_w|x) - \nabla_\theta \log\pi_\theta(y_l|x)\right)$$

với $\hat{r} = \beta(\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)})$.

* Khi $\hat{r}$ lớn (model đã biết phân biệt): gradient nhỏ (converged)
* Khi $\hat{r}$ nhỏ (model chưa phân biệt được): gradient lớn (cần học thêm)

### 3.2. Implicit reward monotonicity

DPO đảm bảo implicit reward $r_\theta(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ tăng cho chosen và giảm cho rejected.

---

## 4. Các biến thể Loss

| Variant | Loss function | Đặc điểm |
|:---|:---|:---|
| sigmoid (DPO) | $-\log\sigma(\beta(\Delta\log\pi))$ | Standard |
| hinge | $\max(0, 1 - \beta\Delta\log\pi)$ | Hard margin |
| ipo | $(\Delta\log\pi - \frac{1}{2\beta})^2$ | Regularized |
| kto_pair | $1 - \sigma(\beta(\log\pi_\theta/\pi_{ref}))$ | Unpaired variant |
