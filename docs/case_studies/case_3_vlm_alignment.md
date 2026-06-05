---
sidebar_position: 33
sidebar_label: "Case Study: VLM Alignment"
---

# Case Study: Vision-Language Model Alignment

TRL hỗ trợ alignment cho VLMs (SmolVLM, Qwen-VL, Gemma3) thông qua ProcessorMixin và multimodal data processing.

---

## 1. VLM Detection

GRPOTrainer tự động phát hiện VLM từ processing class:

```python
if isinstance(processing_class, ProcessorMixin):
    self._tokenizer = processing_class.tokenizer
    self._is_vlm = True
else:
    self._tokenizer = processing_class
    self._is_vlm = False
```

---

## 2. Multimodal Data Processing

### 2.1. Dataset format

```json
{
  "prompt": "Describe this image",
  "image": "path/to/image.jpg"
}
```

Hoặc conversational format:

```json
{
  "messages": [
    {"role": "user", "content": [
      {"type": "image", "image": "path/to/image.jpg"},
      {"type": "text", "text": "What is in this image?"}
    ]},
    {"role": "assistant", "content": "A cat sitting on a sofa."}
  ]
}
```

### 2.2. prepare_multimodal_messages

```python
def prepare_multimodal_messages(messages, processor):
    """Xử lý multimodal content:
    1. Load images từ paths/URLs
    2. Process bằng processor.image_processor
    3. Trả về pixel_values, image_grid_thw
    """
    images = extract_images(messages)
    processed = processor.image_processor(images)
    return {
        "pixel_values": processed.pixel_values,
        "image_grid_thw": processed.image_grid_thw,
    }
```

---

## 3. Vision Placeholder Tokens

VLMs sử dụng special tokens để đánh dấu vị trí image trong text:

```python
# Resolve vision placeholder token IDs
for candidate in ("<|image_pad|>", "<|image|>"):
    tid = self._tokenizer.convert_tokens_to_ids(candidate)
    if tid != self._tokenizer.unk_token_id:
        self._image_pad_token_id = tid
        break
```

Các token này được thay thế bằng image features trong forward pass.

---

## 4. VLM Forward Pass

```python
# Trong _compute_loss
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=inputs.get("pixel_values"),
    image_grid_thw=inputs.get("image_grid_thw"),
    num_images=inputs.get("num_images"),
    pixel_attention_mask=inputs.get("pixel_attention_mask"),
    image_sizes=inputs.get("image_sizes"),
    token_type_ids=inputs.get("token_type_ids"),
    mm_token_type_ids=inputs.get("mm_token_type_ids"),
)
```

---

## 5. Supported VLMs

| Model | Architecture | Image Token |
|:---|:---|:---|
| SmolVLM | Idefics3 | image placeholder token |
| Qwen2-VL | Qwen2VL | image placeholder token |
| Qwen2.5-VL | Qwen2VL | image placeholder token |
| Gemma3 | Gemma3 | image placeholder token |
| InternVL | InternVL | image placeholder token |
| LLaVA | Llava | image placeholder token |

---

## 6. Example: Image QA Alignment

```python
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import accuracy_reward

trainer = GRPOTrainer(
    model="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    reward_funcs=accuracy_reward,
    args=GRPOConfig(
        output_dir="smolvlm-grpo",
        num_generations=8,
        max_completion_length=512,
        beta=0.04,
    ),
    train_dataset=image_qa_dataset,
)
trainer.train()
```
