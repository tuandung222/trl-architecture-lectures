---
sidebar_position: 3
sidebar_label: "Bài 1: Kiến trúc TRL & Triết lý Thiết kế"
---

# Bài 1: Kiến trúc TRL & Triết lý Thiết kế

Bài này phân tích cấu trúc module, cơ chế lazy import, hệ thống kế thừa Trainer, và triết lý thiết kế giúp TRL trở thành thư viện alignment phổ biến nhất trong hệ sinh thái Hugging Face.

---

## 1. Cấu trúc module của TRL

```
trl/
  __init__.py          # Lazy module với _LazyModule
  trainer/             # Core trainers (stable API)
    base_trainer.py    # _BaseTrainer kế thừa HF Trainer
    base_config.py     # _BaseConfig mở rộng TrainingArguments
    sft_trainer.py     # SFTTrainer
    dpo_trainer.py     # DPOTrainer
    grpo_trainer.py    # GRPOTrainer
    kto_trainer.py     # KTOTrainer
    reward_trainer.py  # RewardTrainer
    rloo_trainer.py    # RLOOTrainer
    callbacks.py       # SyncRefModelCallback, LogCompletions
    utils.py           # Tiện ích chung (pad, selective_log_softmax...)
  experimental/        # Experimental trainers
    ppo/               # PPOTrainer
    bco/               # BCOTrainer
    cpo/               # CPOTrainer
    online_dpo/        # OnlineDPOTrainer
    ...
  models/              # Model utilities
    utils.py           # prepare_deepspeed, prepare_fsdp, unwrap
    activation_offloading.py
  rewards/             # Reward functions
    accuracy_rewards.py
    format_rewards.py
    other_rewards.py
  generation/          # vLLM integration
    vllm_generation.py
    vllm_client.py
  data_utils.py        # apply_chat_template, pack_dataset
  chat_template_utils.py
```

### 1.1. Nguyên tắc phân tầng

TRL phân chia rõ ràng giữa **stable** và **experimental**:

* `trl/trainer/`: Các trainer đã ổn định, có API contract rõ ràng. Bất kỳ breaking change nào đều được cảnh báo trước.
* `trl/experimental/`: Các trainer đang thử nghiệm, có thể thay đổi hoặc bị loại bỏ bất cứ lúc nào. PPO nằm ở đây.

Sự phân tầng này giúp người dùng production tự tin dùng stable trainers, trong khi researchers có thể thử nghiệm các thuật toán mới.

---

## 2. Cơ chế Lazy Import

File `trl/__init__.py` sử dụng `_LazyModule` để trì hoãn việc import cho đến khi thực sự cần:

```python
class _LazyModule:
    """Trì hoãn import cho đến khi attribute được truy cập."""
    
    def __getattr__(self, name):
        # Chỉ import module chứa symbol khi lần đầu truy cập
        ...
```

**Lý do thiết kế**: TRL phụ thuộc vào nhiều thư viện nặng (torch, transformers, accelerate, vllm, peft, deepspeed). Lazy import giúp:
1. Giảm thời gian `import trl` ban đầu
2. Tránh lỗi import khi một dependency tùy chọn không có (ví dụ: vLLM không cài)
3. Cho phép `from trl import GRPOTrainer` chỉ tải đúng module cần thiết

Cấu trúc `_import_structure` định nghĩa mapping rõ ràng:

```python
_import_structure = {
    "trainer": ["DPOTrainer", "GRPOTrainer", "SFTTrainer", ...],
    "models": ["create_reference_model"],
    "data_utils": ["apply_chat_template", "pack_dataset", ...],
    ...
}
```

---

## 3. _BaseTrainer - Kế thừa từ HF Trainer

Lớp `_BaseTrainer` trong `trl/trainer/base_trainer.py` kế thừa trực tiếp từ `transformers.Trainer`:

```python
class _BaseTrainer(Trainer):
    _tag_names = []
    _name = "Base"
    _paper = {}
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._send_telemetry()
```

### 3.1. Lợi ích của việc kế thừa Trainer

Khi kế thừa `transformers.Trainer`, TRL được thừa hưởng miễn phí:
* **Training loop**: gradient accumulation, mixed precision, gradient checkpointing
* **Distributed training**: DDP, FSDP, DeepSpeed (ZeRO-1/2/3)
* **Logging**: WandB, TensorBoard, Comet ML, Trackio
* **Checkpointing**: save/load model, optimizer, scheduler state
* **Evaluation**: eval loop, prediction, metric computation
* **Callback system**: early stopping, custom callbacks

### 3.2. Telemetry

Mỗi trainer gửi telemetry (tôn trọng `HF_HUB_DISABLE_TELEMETRY=1`):

```python
send_telemetry(
    topic=f"trl/{trainer}",
    library_name="trl",
    library_version=__version__,
    user_agent={
        "model_arch": model_arch,
        "peft": str(is_peft_model(self.model)).lower(),
        "distributed": distributed,  # "deepspeed", "fsdp", "ddp", "none"
        "world_size": world_size,    # "1", "2-8", "9-64", "65+"
        "device": device,
        "gpu": gpu,
    },
)
```

---

## 4. Hệ thống Config Dataclass

Mỗi trainer có một Config riêng kế thừa từ `_BaseConfig` (mở rộng `TrainingArguments`):

```python
@dataclass
class GRPOConfig(_BaseConfig):
    # GRPO-specific parameters
    num_generations: int = 8
    max_completion_length: int = 256
    temperature: float = 1.0
    beta: float = 0.04
    loss_type: str = "grpo"
    ...
```

### 4.1. Ưu điểm thiết kế

* **Type safety**: Dataclass đảm bảo đúng kiểu dữ liệu
* **CLI integration**: Tương thích `HfArgumentParser` để parse từ command line
* **Documentation**: Mỗi parameter có docstring chi tiết
* **Default values**: Giá trị mặc định hợp lý cho từng thuật toán
* **Validation**: Có thể validate trong `__post_init__`

### 4.2. Ví dụ: GRPOConfig có hơn 80 tham số

GRPOConfig bao gồm các nhóm tham số:
* Model: `model_init_kwargs`, `disable_dropout`, `cast_lm_head_to_fp32`
* Generation: `num_generations`, `max_completion_length`, `temperature`, `top_p`, `top_k`
* Loss: `loss_type`, `epsilon`, `epsilon_high`, `beta`, `num_iterations`
* vLLM: `use_vllm`, `vllm_mode`, `vllm_gpu_memory_utilization`, `vllm_tensor_parallel_size`
* Advanced: `importance_sampling_level`, `off_policy_mask_threshold`, `use_liger_kernel`

---

## 5. Callback System

TRL mở rộng hệ thống callback của HF Trainer:

### 5.1. SyncRefModelCallback

Đồng bộ trọng số reference model với policy model (dùng cho EMA-style training):

```python
class SyncRefModelCallback(TrainerCallback):
    def __init__(self, ref_model, alpha=0.999):
        self.ref_model = ref_model
        self.alpha = alpha
    
    def on_step_end(self, args, state, control, model, **kwargs):
        # EMA update: ref = alpha * ref + (1 - alpha) * model
        for ref_p, model_p in zip(self.ref_model.parameters(), model.parameters()):
            ref_p.data.mul_(self.alpha).add_(model_p.data, alpha=1 - self.alpha)
```

### 5.2. LogCompletionsCallback

Log các completion sinh ra trong quá trình training lên WandB/Comet:

```python
class LogCompletionsCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, model, **kwargs):
        # Generate completions and log to wandb Table
        ...
```

---

## 6. Utility Functions quan trọng

### 6.1. selective_log_softmax

Tính log softmax chỉ tại các vị trí token cần thiết, tránh tính toàn bộ vocabulary:

```python
def selective_log_softmax(logits, index):
    """Tính log P(index | logits) mà không materialize full softmax."""
    # log_softmax(x_i) = x_i - log(sum(exp(x_j)))
    # = x_i - logsumexp(x)
    log_z = torch.logsumexp(logits, dim=-1)
    selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    return selected_logits - log_z
```

### 6.2. disable_dropout_in_model

Tắt dropout để đảm bảo consistency giữa các forward pass:

```python
def disable_dropout_in_model(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
```

### 6.3. pad

Padding tensor với support cho left/right padding và multiple-of:

```python
def pad(tensors, padding_value=0, padding_side="right"):
    """Pad list of tensors to same length."""
    max_len = max(t.size(0) for t in tensors)
    padded = []
    for t in tensors:
        pad_size = max_len - t.size(0)
        if padding_side == "right":
            padded.append(F.pad(t, (0,) * (2 * t.dim() - 2) + (0, pad_size), value=padding_value))
        else:
            padded.append(F.pad(t, (0,) * (2 * t.dim() - 2) + (pad_size, 0), value=padding_value))
    return torch.stack(padded)
```

Bài tiếp theo sẽ đi sâu vào SFTTrainer, trainer đầu tiên trong pipeline alignment.
