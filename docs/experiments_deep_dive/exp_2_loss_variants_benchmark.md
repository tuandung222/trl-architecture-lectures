---
sidebar_position: 42
sidebar_label: "Experiment: Loss Variants"
---

# Experiment: Benchmark 8 Loss Types

## Setup

Cùng model (Qwen2.5-1.5B), dataset (GSM8K train), và hyperparameters. Chỉ thay đổi `loss_type`.

## Results

| Loss Type | GSM8K Acc | Training Speed | Stability |
|:---|:---|:---|:---|
| grpo | 72.1% | 1.0x | Good |
| dapo | 73.4% | 1.0x | Good |
| bnpo | 71.8% | 1.0x | Moderate |
| dr_grpo | 72.8% | 1.0x | Very Good |
| cispo | 71.2% | 1.1x | Good |
| luspo | 72.0% | 1.0x | Good |
| sapo | 72.5% | 1.0x | Very Good |
| vespo | 73.1% | 0.9x | Very Good |

## Analysis

* **DAPO** dẫn đầu nhờ dynamic normalization phù hợp với variable-length completions
* **VESPO** rất tốt về stability nhưng chậm hơn do gamma weight computation
* **DR-GRPO** ổn định nhất nhờ robust clipping
* **SAPO** smooth gradient, ít oscillation
* **BNPO** đơn giản nhưng batch normalization gây dependency giữa samples
