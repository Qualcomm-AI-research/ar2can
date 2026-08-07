---
base_model: black-forest-labs/FLUX.1-Kontext-dev
library_name: peft
license: bsd-3-clause-clear
tags:
  - lora
  - flux
  - face-composition
  - text-to-image
---

# Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation — Artist LoRA

LoRA adapter for **FLUX.1-Kontext-dev** used in the Artist stage of the
[ar2can](https://github.com/Qualcomm-AI-research/ar2can) multi-reference face composition pipeline.

**Project page:** https://qualcomm-ai-research.github.io/ar2can/

The Artist receives a probe canvas with reference faces pasted onto detected face
regions (with optional segmentation-based cropping and face alignment) and renders
the final photorealistic composition.

> **Note:** This adapter requires [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev), which is subject to its own license. You must accept the FLUX.1-Kontext-dev license on Hugging Face before use.

## Adapter config

| Parameter | Value |
|---|---|
| `peft_type` | LORA |
| `r` (rank) | 64 |
| `lora_alpha` | 128 |
| `lora_dropout` | 0.0 |
| `init_lora_weights` | gaussian |
| `use_dora` | false |
| `use_rslora` | false |
| PEFT version | 0.17.0 |

**Target modules:** `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0`,
`attn.add_q_proj`, `attn.add_k_proj`, `attn.add_v_proj`, `attn.to_add_out`,
`ff.net.0.proj`, `ff.net.2`, `ff_context.net.0.proj`, `ff_context.net.2`

## Usage

See the top-level [model card](../README.md) or the
[ar2can repository](https://github.com/Qualcomm-AI-research/ar2can) for full usage instructions.

## License

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
SPDX-License-Identifier: BSD-3-Clause-Clear
