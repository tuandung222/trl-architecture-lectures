---
sidebar_position: 43
sidebar_label: "Experiment: vLLM Speedup"
---

# Experiment: vLLM Generation Speedup Analysis

## Setup

| Parameter | Value |
|:---|:---|
| Model | Qwen2.5-7B-Instruct |
| GPU | 2x A100 80GB |
| Generation | 8 samples x 512 tokens |
| Batch size | 4 prompts per step |

## Results

| Method | Gen Time/step | Total Step Time | Speedup |
|:---|:---|:---|:---|
| HF generate | 45s | 62s | 1.0x |
| vLLM colocate | 8s | 32s | 1.9x |
| vLLM server | 6s | 28s | 2.2x |

## Importance Sampling Overhead

| IS Correction | Extra Time | Accuracy Impact |
|:---|:---|:---|
| None | +0s | Baseline |
| Token truncate | +0.5s | +0.2% |
| Token mask | +0.8s | +0.5% |
| Sequence truncate | +0.3s | +0.1% |

## Analysis

* vLLM cho **2x speedup** trên tổng step time
* IS correction overhead nhỏ (~1-2s) nhưng quan trọng cho quality
* Colocation mode đơn giản hơn nhưng bị giới hạn bởi VRAM sharing
* Server mode nhanh nhất nhưng cần GPU riêng cho vLLM
