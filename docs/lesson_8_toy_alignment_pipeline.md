---
sidebar_position: 10
sidebar_label: "Bài 8: Thực hành Toy Pipeline"
---

# Bài 8: Thực hành - Tự viết Toy Alignment Pipeline

Bài thực hành này yêu cầu bạn tự lập trình từ đầu một pipeline DPO và GRPO tối giản bằng PyTorch thuần, không dùng TRL. Mục tiêu: hiểu sâu từng bước tính toán bằng cách tự hiện thực.

---

## 1. Toy DPO Pipeline

### 1.1. Chuẩn bị

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_name = "gpt2"  # Dùng model nhỏ để test
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tạo reference model (copy + freeze)
ref_model = AutoModelForCausalLM.from_pretrained(model_name)
for p in ref_model.parameters():
    p.requires_grad = False
```

### 1.2. DPO Loss từ công thức toán

```python
def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    DPO Loss = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    
    Args:
        policy_chosen_logps: log pi_theta(y_w | x), shape [B]
        policy_rejected_logps: log pi_theta(y_l | x), shape [B]
        ref_chosen_logps: log pi_ref(y_w | x), shape [B]
        ref_rejected_logps: log pi_ref(y_l | x), shape [B]
        beta: KL penalty coefficient
    """
    chosen_logratios = policy_chosen_logps - ref_chosen_logps
    rejected_logratios = policy_rejected_logps - ref_rejected_logps
    logits = chosen_logratios - rejected_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    return loss
```

### 1.3. Forward pass tính log probabilities

```python
def get_log_probs(model, input_ids, attention_mask, completion_mask):
    """Tính sum of log probabilities trên completion tokens.
    
    Args:
        model: policy hoặc reference model
        input_ids: [B, T] full sequence (prompt + completion)
        attention_mask: [B, T]
        completion_mask: [B, T] 1 tại completion tokens, 0 tại prompt
    
    Returns:
        log_probs: [B] sum of log probs trên completion
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]  # Shift right
    labels = input_ids[:, 1:]            # Shift left
    
    # selective_log_softmax: chỉ tính tại positions cần thiết
    log_z = torch.logsumexp(logits, dim=-1)  # [B, T-1]
    selected = torch.gather(logits, -1, labels.unsqueeze(-1)).squeeze(-1)
    per_token_logps = selected - log_z  # [B, T-1]
    
    # Chỉ lấy completion tokens
    completion_logps = (per_token_logps * completion_mask[:, 1:]).sum(dim=-1)
    return completion_logps
```

### 1.4. Training loop

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Dummy preference data
chosen_texts = ["Question: 2+2? Answer: 4"]
rejected_texts = ["Question: 2+2? Answer: 5"]

for epoch in range(3):
    # Tokenize
    chosen_enc = tokenizer(chosen_texts, padding=True, return_tensors="pt")
    rejected_enc = tokenizer(rejected_texts, padding=True, return_tensors="pt")
    
    # Policy forward
    policy_chosen = get_log_probs(model, chosen_enc["input_ids"], 
                                   chosen_enc["attention_mask"],
                                   chosen_enc["attention_mask"])
    policy_rejected = get_log_probs(model, rejected_enc["input_ids"],
                                     rejected_enc["attention_mask"],
                                     rejected_enc["attention_mask"])
    
    # Reference forward (no grad)
    with torch.no_grad():
        ref_chosen = get_log_probs(ref_model, chosen_enc["input_ids"],
                                    chosen_enc["attention_mask"],
                                    chosen_enc["attention_mask"])
        ref_rejected = get_log_probs(ref_model, rejected_enc["input_ids"],
                                      rejected_enc["attention_mask"],
                                      rejected_enc["attention_mask"])
    
    # Compute loss
    loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
    
    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"Epoch {epoch}: DPO loss = {loss.item():.4f}")
```

---

## 2. Toy GRPO Pipeline

### 2.1. GRPO Loss

```python
def grpo_loss(
    per_token_logps: torch.Tensor,     # [B, T]
    old_per_token_logps: torch.Tensor,  # [B, T]
    ref_per_token_logps: torch.Tensor,  # [B, T]
    advantages: torch.Tensor,           # [B]
    completion_mask: torch.Tensor,      # [B, T]
    beta: float = 0.04,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """
    GRPO Loss với clipped surrogate.
    """
    # Probability ratio per token
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)
    
    # Clipped surrogate
    advantages_unsq = advantages.unsqueeze(1)  # [B, 1] -> broadcast to [B, T]
    surr1 = ratio * advantages_unsq
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages_unsq
    per_token_loss = -torch.min(surr1, surr2)
    
    # KL penalty (Schulman approximation)
    kl = torch.exp(ref_per_token_logps - per_token_logps) \
         - (ref_per_token_logps - per_token_logps) - 1
    per_token_loss = per_token_loss + beta * kl
    
    # Masked mean
    loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
    return loss
```

### 2.2. Group-relative advantage

```python
def compute_grpo_advantages(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
    """
    Compute group-relative advantages.
    
    Args:
        rewards: [B * G] flat tensor of rewards
        num_generations: G (number of generations per prompt)
    
    Returns:
        advantages: [B * G] normalized advantages
    """
    B = rewards.shape[0] // num_generations
    grouped = rewards.reshape(B, num_generations)
    
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    
    advantages = (grouped - mean) / (std + 1e-8)
    return advantages.flatten()
```

### 2.3. Full GRPO training step

```python
def grpo_training_step(model, ref_model, tokenizer, prompts, num_generations=4):
    """Một step GRPO hoàn chỉnh."""
    
    # Step 1: Generate G completions per prompt
    all_completions = []
    for prompt in prompts:
        for _ in range(num_generations):
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=50, do_sample=True)
            completion = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:])
            all_completions.append(completion)
    
    # Step 2: Compute rewards (dummy: length-based)
    rewards = torch.tensor([min(len(c), 100) / 100.0 for c in all_completions])
    
    # Step 3: Compute advantages
    advantages = compute_grpo_advantages(rewards, num_generations)
    
    # Step 4: Tokenize and compute log probs
    # ... (tương tự DPO, nhưng trên completions thay vì pairs)
    
    # Step 5: Compute GRPO loss và backward
    # ... (sử dụng grpo_loss function ở trên)
```

---

## 3. So sánh Toy vs TRL

| Khía cạnh | Toy Implementation | TRL GRPOTrainer |
|:---|:---|:---|
| Generation | HF generate() đơn giản | vLLM + IS correction |
| Reward | Dummy function | math_verify + async + multi-obj |
| Reference | Separate model copy | PEFT-based hoặc separate |
| Loss | Standard GRPO | 8 loss variants |
| Distributed | Single GPU | FSDP/DeepSpeed/DDP |
| Padding | Right padding | Left padding + flush_left |
| Multi-turn | Không | Environment factory |
| KL penalty | Schulman approx | Schulman + bias correction |

---

## 4. Bài tập mở rộng

1. **Thêm vLLM**: Thay HF generate bằng vLLM inference
2. **Thêm IS correction**: Implement importance sampling correction
3. **Thêm OPSM**: Implement off-policy sequence masking
4. **Thêm DAPO**: Thay đổi loss thành dynamic advantage per-token
5. **Thêm async reward**: Implement async code execution reward
6. **Thêm distributed**: Wrap bằng Accelerate cho multi-GPU

Hoàn thành tất cả 6 bài tập trên, bạn sẽ có một phiên bản TRL GRPOTrainer thu nhỏ của riêng mình.
