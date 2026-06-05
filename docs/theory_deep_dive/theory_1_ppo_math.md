---
sidebar_position: 21
sidebar_label: "Toán học PPO"
---

# Phân tích toán học PPO (Proximal Policy Optimization)

## 1. Policy Gradient Theorem

Mục tiêu RL: tối đa hóa kỳ vọng tổng reward:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} r_t\right]$$

Policy gradient theorem cho ta:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t\right]$$

$A_t$ là advantage function, đo lường "tốt hơn trung bình bao nhiêu".

---

## 2. GAE (Generalized Advantage Estimation)

### 2.1. Bias-Variance Tradeoff

* $A_t^{(0)} = r_t + V(s_{t+1}) - V(s_t)$: 1-step TD, low variance, high bias
* $A_t^{(\infty)} = \sum_{l=0}^{T-t} \gamma^l r_{t+l} - V(s_t)$: Monte Carlo, zero bias, high variance

### 2.2. GAE as exponential weighted average

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$$

Với $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$.

* $\lambda = 0$: tương đương 1-step TD (bias cao, variance thấp)
* $\lambda = 1$: tương đương Monte Carlo (bias thấp, variance cao)
* $\lambda \in (0,1)$: tradeoff tối ưu

---

## 3. Clipped Surrogate Objective

### 3.1. Motivation

REINFORCE gradient có thể quá lớn khi $\pi_\theta(a_t|s_t)$ nhỏ, gây unstable updates. PPO giới hạn step size bằng clipping.

### 3.2. Công thức

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(\rho_t(\theta)\hat{A}_t, \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

Với $\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.

### 3.3. Tại sao min()?

* Khi $A_t > 0$ (action tốt): clip giới hạn ratio từ trên (không tăng quá nhiều)
* Khi $A_t < 0$ (action xấu): clip giới hạn ratio từ dưới (không giảm quá nhiều)
* min() đảm bảo conservative update theo cả hai hướng

---

## 4. Value Function Loss

$$L^{VF}(\phi) = \mathbb{E}_t\left[\left(V_\phi(s_t) - \hat{R}_t\right)^2\right]$$

Với $\hat{R}_t = \hat{A}_t + V_{\phi_{old}}(s_t)$ là target returns.

---

## 5. Total PPO Loss

$$L^{PPO} = L^{CLIP}(\theta) - c_1 L^{VF}(\phi) + c_2 S[\pi_\theta](s_t)$$

$S$ là entropy bonus khuyến khích exploration, $c_1$ và $c_2$ là weight coefficients.

Trong TRL PPOTrainer, KL penalty được cộng trực tiếp vào reward:

$$r_t^{KL} = r_t - \beta \cdot \text{KL\_token}_t$$

với $\text{KL\_token}_t = \log \pi_\theta(a_t|s_t) - \log \pi_{ref}(a_t|s_t)$.
