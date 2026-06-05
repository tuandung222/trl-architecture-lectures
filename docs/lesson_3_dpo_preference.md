---
sidebar_position: 5
sidebar_label: "Bài 3: DPO & Preference Optimization"
---

# Bài 3: DPO & Preference Optimization - Từ Toán đến Code

DPO (Direct Preference Optimization) là thuật toán đưa TRL lên bản đồ alignment. Bài này phân tích chi tiết toán học, cách hiện thực trong mã nguồn, và các biến thể quan trọng.

---

## 1. Toán học DPO

### 1.1. Từ RL objective đến closed-form

Bắt đầu từ constrained RL objective:

$$\max_{\pi_\theta} \mathbb{E}_{x, y \sim \pi_\theta} [r(x,y)] - \beta D_{KL}(\pi_\theta \| \pi_{ref})$$

Rafailov et al. (2023) chứng minh rằng optimal policy có dạng closed-form:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

Với $Z(x)$ là partition function. Nghịch đảo công thức trên:

$$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

### 1.2. Bradley-Terry preference model

Sử dụng Bradley-Terry model cho preference data: probability that $y_w$ (chosen) được ưu tiên hơn $y_l$ (rejected):

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

Thay thế $r$ bằng công thức trên, partition function $Z(x)$ bị triệt tiêu:

$$L_{DPO}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) \right]$$

**Đột phá**: Không cần train reward model riêng. Reward được implicitly định nghĩa qua log probability ratio giữa policy và reference.

---

## 2. Hiện thực trong TRL - DPOTrainer

### 2.1. DataCollator cho preference data

```python
@dataclass
class DataCollatorForPreference:
    """Collator cho preference data.
    
    Input: mỗi sample chứa prompt_ids, chosen_ids, rejected_ids
    Output: batch với input_ids (chosen + rejected concatenated),
            attention_mask, completion_mask, labels
    """
```

Batch được tổ chức sao cho **nửa đầu** là chosen completions, **nửa sau** là rejected completions. Điều này cho phép forward pass duy nhất qua cả policy và reference models.

### 2.2. Loss computation trong DPOTrainer

```python
def compute_loss(self, model, inputs):
    # Forward pass: policy model
    policy_chosen_logps = self.forward_pass(model, inputs, "chosen")
    policy_rejected_logps = self.forward_pass(model, inputs, "rejected")
    
    # Forward pass: reference model (no_grad)
    with torch.no_grad():
        ref_chosen_logps = self.forward_pass(self.ref_model, inputs, "chosen")
        ref_rejected_logps = self.forward_pass(self.ref_model, inputs, "rejected")
    
    # DPO loss
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    logits = chosen_logratios - rejected_logratios
    
    loss = -F.logsigmoid(self.beta * logits).mean()
    return loss
```

### 2.3. selective_log_softmax

Thay vì tính full softmax trên toàn bộ vocabulary (V = 128K tokens cho Llama), TRL dùng `selective_log_softmax`:

```python
def selective_log_softmax(logits, index):
    """Chỉ tính log P tại vị trí token cần thiết."""
    # log P(token_i) = logits[i, token_i] - logsumexp(logits[i, :])
    log_z = torch.logsumexp(logits, dim=-1)
    selected = torch.gather(logits, -1, index.unsqueeze(-1)).squeeze(-1)
    return selected - log_z
```

**Hiệu quả**: Giảm VRAM đáng kể vì không cần materialize full softmax distribution.

---

## 3. Reference Model Management

### 3.1. Ba chiến lược quản lý reference model

**Chiến lược 1: Separate model (mặc định)**
```python
self.ref_model = create_reference_model(model)
# Tạo bản sao model, đóng băng trọng số
# Tốn gấp đôi VRAM nhưng đơn giản nhất
```

**Chiến lược 2: PEFT-based (khi dùng LoRA)**
```python
if is_peft_model(model):
    self.ref_model = None
    # Reference = base model khi tắt adapter
    # use_adapter() context manager để switch on/off
```

**Chiến lược 3: EMA sync (SyncRefModelCallback)**
```python
# Reference model được cập nhật liên tục qua EMA
ref_params = alpha * ref_params + (1 - alpha) * policy_params
```

### 3.2. flush_left - Tối ưu memory

```python
def flush_left(tensor):
    """Dịch chuyển non-padding tokens về bên trái.
    Giúp giải phóng VRAM ở cuối tensor."""
```

---

## 4. Các biến thể Preference Optimization

### 4.1. CPO (Contrastive Preference Optimization)

Không cần reference model. Loss đơn giản hóa:

$$L_{CPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \pi_\theta(y_w|x) - \beta \log \pi_\theta(y_l|x)\right)\right]$$

**Ưu điểm**: Tiết kiệm VRAM (không cần ref model). **Nhược điểm**: Có thể drift xa khỏi SFT model ban đầu.

### 4.2. IPO (Identity Preference Optimization)

Thay sigmoid loss bằng squared loss:

$$L_{IPO} = \mathbb{E}\left[\left(\log \frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)} - \frac{1}{2\beta}\right)^2\right]$$

IPO tránh vấn đề over-optimization khi reward quá lớn.

### 4.3. KTO (Kahneman-Tversky Optimization)

Không cần paired data (chosen vs rejected). Sử dụng **unpaired** feedback (chỉ biết response này tốt hay xấu):

$$L_{KTO} = -\mathbb{E}\left[\max\left(0, 1 - \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right) \cdot \mathbb{1}[y \text{ is good}]\right]$$

KTO phù hợp khi chỉ có binary feedback (thumbs up/down) mà không có cặp so sánh.

---

## 5. DPOConfig - Tham số quan trọng

| Tham số | Mặc định | Mô tả |
|:---|:---|:---|
| `beta` | 0.1 | KL penalty coefficient |
| `loss_type` | "sigmoid" | sigmoid, hinge, ipo, kto_pair... |
| `label_smoothing` | 0.0 | Label smoothing cho conservative DPO |
| `use_logits_to_keep` | False | Chỉ compute logits cuối (tiết kiệm VRAM) |
| `precompute_ref_log_probs` | False | Cache ref log probs (tiết kiệm compute) |

### 5.1. logits_to_keep optimization

Khi `use_logits_to_keep=True`, chỉ compute logits cho K tokens cuối thay vì toàn bộ sequence. Tối ưu này đặc biệt hiệu quả với large vocabulary:

```python
# Thay vì: logits shape [B, T, V] với V=128K
# Chỉ cần: logits shape [B, K, V] với K nhỏ
# Giảm VRAM: T/K lần cho logits tensor
```

Bài tiếp theo đi sâu vào GRPO, trainer phức tạp và mạnh nhất trong TRL.
