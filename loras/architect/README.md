---
base_model: black-forest-labs/FLUX.1-schnell
library_name: peft
license: bsd-3-clause-clear
tags:
  - lora
  - flux
  - text-to-image
---

# Ar2Can: An Architect and an Artist Leveraging a Canvas for Multi-Human Generation — Architect LoRA

LoRA adapter for **FLUX.1-schnell** used in the Architect stage of the
[ar2can](https://github.com/Qualcomm-AI-research/ar2can) multi-reference face composition pipeline.

**Project page:** https://qualcomm-ai-research.github.io/ar2can/

Given a text prompt, the Architect generates a 1024×1024 *probe* image that
establishes scene layout and approximate face positions. The probe is then
passed to the Artist stage for reference-face inpainting.

> **Note:** This adapter requires [FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell), which is subject to its own license. You must accept the FLUX.1-schnell license on Hugging Face before use.

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
`ff.net.0.proj`, `ff.net.2`, `ff_context.net.0.proj`, `ff_context.net.2`, `proj_mlp`

## Usage

See the top-level [model card](../README.md) or the
[ar2can repository](https://github.com/Qualcomm-AI-research/ar2can) for full usage instructions.

## License

Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.  
SPDX-License-Identifier: BSD-3-Clause-Clear
