#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Standalone inference script for the ar2can two-stage pipeline.

Default inputs : faces/person_1.webp, faces/person_2.webp
Default prompt : "Two men in a mythical forest, with blue flames around them"

All defaults are overridable via CLI flags (see --help).
"""

import argparse
import os
from typing import Any

import numpy as np
import torch
from diffusers import DiffusionPipeline
from PIL import Image

from pipeline.artist_pipeline_multicond import FluxArtistPipeline
from pipeline.canvas import (
    FacerDetector,
    FacerParser,
    build_canvases,
    detect_best_face_box,
    pick_topk_boxes,
)

# ── Model dirs ──────────────────────────────────────────────────────────
ARCHITECT_DIR_DEFAULT = os.getenv("ARCHITECT_DIR", "black-forest-labs/FLUX.1-schnell")
ARTIST_DIR_DEFAULT = os.getenv("ARTIST_DIR", "black-forest-labs/FLUX.1-Kontext-dev")
ARCHITECT_LORA_DEFAULT = os.getenv("ARCHITECT_LORA", "loras/architect")
ARTIST_LORA_DEFAULT = os.getenv("ARTIST_LORA", "loras/artist")
HF_REPO_ID = os.getenv("AR2CAN_LORA_HF", "Qualcomm-AI-Research/ar2can")

# ── Pipeline defaults (mirror app.py) ───────────────────────────────────
DEFAULT_PROMPT = (
    "Three people standing side-by-side in a winter landscape, "
    "8K, ultra-realistic, "
    "realistic light rendering, high dynamic range. "
    "They are each holding a paper-cup "
    "of hot steaming tea. From left to right, the people are wearing: "
    "a white winter "
    "coat, a yellow furry coat, a black turtleneck sweater. Preserve exact facial identity."
)
DEFAULT_STEPS_ARCH = 4
DEFAULT_GUIDANCE_ARCH = 0.0
DEFAULT_STEPS_ARTIST = 20
DEFAULT_GUIDANCE_ARTIST = 2.6
DEFAULT_FACE_SCALE = 1.5
DEFAULT_TARGET_PAD = 1.0
DEFAULT_MASK_MODE = "tight"  # relaxed segmentation
DEFAULT_SEED = 0
DEFAULT_OUTPUT = "output.png"
DEFAULT_FACES = (
    "faces/person_1.webp",
    "faces/person_2.webp",
    "faces/person_3.webp",
)

DET_BACKBONE = os.getenv("DET_BACKBONE", "retinaface/mobilenet")
DET_SCORE = float(os.getenv("DET_SCORE", 0.5))
SEG_BACKBONE = os.getenv("SEG_PARSE_BACKBONE", "farl/celebm/448")
PROBE_H = PROBE_W = 1024


# ── Argument parsing ────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments for the ar2can inference
    script.
    """
    parser = argparse.ArgumentParser(description="ar2can inference")

    parser.add_argument(
        "--faces",
        nargs="+",
        default=list(DEFAULT_FACES),
        metavar="IMG",
        help="Reference face images (1–5)",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path")

    parser.add_argument("--architect-dir", default=ARCHITECT_DIR_DEFAULT)
    parser.add_argument("--architect-lora", default=ARCHITECT_LORA_DEFAULT)
    parser.add_argument("--artist-dir", default=ARTIST_DIR_DEFAULT)
    parser.add_argument("--artist-lora", default=ARTIST_LORA_DEFAULT)

    parser.add_argument("--steps-architect", type=int, default=DEFAULT_STEPS_ARCH)
    parser.add_argument("--guidance-architect", type=float, default=DEFAULT_GUIDANCE_ARCH)
    parser.add_argument("--steps-artist", type=int, default=DEFAULT_STEPS_ARTIST)
    parser.add_argument("--guidance-artist", type=float, default=DEFAULT_GUIDANCE_ARTIST)
    parser.add_argument("--face-scale", type=float, default=DEFAULT_FACE_SCALE)
    parser.add_argument("--target-pad", type=float, default=DEFAULT_TARGET_PAD)
    parser.add_argument(
        "--mask-mode",
        default=DEFAULT_MASK_MODE,
        choices=["relaxed", "loose", "tight"],
        help="Segmentation mask mode",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-seg", action="store_true", help="Disable segmentation")
    parser.add_argument("--no-align", action="store_true", help="Disable face alignment")
    parser.add_argument(
        "--no-fg-mask",
        dest="use_fg_mask",
        action="store_false",
        help="Disable foreground cond_image_mask (enabled by default)",
    )
    parser.set_defaults(use_fg_mask=True)

    return parser.parse_args()


# ── Model loading ───────────────────────────────────────────────────────
def load_models(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any, Any, Any]:
    """Load and return (architect_pipe, artist_pipe, detector, seg_parser).

    LoRA adapters are applied when the specified paths exist.  seg_parser is
    None when ``--no-seg`` is set or when the Facer parser fails to initialise.
    """
    try:
        from peft import PeftModel

        peft_ok = True
    except Exception:
        PeftModel = None
        peft_ok = False

    device_str = device.type

    # Architect
    print(f"[Load] Architect: {args.architect_dir}")
    architect_pipe = DiffusionPipeline.from_pretrained(args.architect_dir, torch_dtype=dtype).to(
        device
    )
    if peft_ok:
        try:
            if args.architect_lora and os.path.isfile(
                os.path.join(args.architect_lora, "adapter_model.safetensors")
            ):
                arch_src, arch_kw = args.architect_lora, {}
            else:
                print(f"[Load] Architect LoRA not found locally — loading from HuggingFace: {HF_REPO_ID}")
                arch_src, arch_kw = HF_REPO_ID, {"subfolder": "architect"}
            architect_pipe.transformer = PeftModel.from_pretrained(
                architect_pipe.transformer, arch_src, **arch_kw
            )
            architect_pipe.transformer.set_adapter("default")
            print(f"[Load] Architect LoRA: {arch_src}")
        except Exception as e:
            print(f"[Warn] Architect LoRA skipped: {e}")

    # Face detector
    print(f"[Load] Face detector: {DET_BACKBONE}")
    detector = FacerDetector(
        device_str=device_str,
        backbone=DET_BACKBONE,
        score_thresh=DET_SCORE,
    )

    # Segmentation parser
    seg_parser = None
    if not args.no_seg:
        try:
            print(f"[Load] Seg parser: {SEG_BACKBONE}")
            seg_parser = FacerParser(
                device_str=device_str,
                backbone=SEG_BACKBONE,
                det_backbone=DET_BACKBONE,
            )
        except Exception as e:
            print(f"[Warn] Segmentation disabled: {e}")

    # Artist
    print(f"[Load] Artist: {args.artist_dir}")
    artist_pipe = FluxArtistPipeline.from_pretrained(args.artist_dir, torch_dtype=dtype).to(device)
    if peft_ok:
        try:
            if args.artist_lora and os.path.isfile(
                os.path.join(args.artist_lora, "adapter_model.safetensors")
            ):
                art_src, art_kw = args.artist_lora, {}
            else:
                print(f"[Load] Artist LoRA not found locally — loading from HuggingFace: {HF_REPO_ID}")
                art_src, art_kw = HF_REPO_ID, {"subfolder": "artist"}
            artist_pipe.transformer = PeftModel.from_pretrained(
                artist_pipe.transformer, art_src, **art_kw
            )
            artist_pipe.transformer.set_adapter("default")
            artist_pipe.transformer = artist_pipe.transformer.merge_and_unload()
            print(f"[Load] Artist LoRA merged: {art_src}")
        except Exception as e:
            print(f"[Warn] Artist LoRA skipped: {e}")

    return architect_pipe, artist_pipe, detector, seg_parser


# ── Inference ───────────────────────────────────────────────────────────
def run(args: argparse.Namespace) -> Image.Image:
    """Run the full two-stage ar2can pipeline and save outputs.

    Stage A: Architect (FLUX.1-schnell + LoRA) generates a probe image that
    establishes the scene layout and face positions.

    Stage B: reference faces are segmentation-cropped, placed on per-subject
    canvases, and passed to the Artist (FLUX.1-Kontext-dev + LoRA) which
    renders the final composition.

    Saves ``args.output``, ``{stem}_probe{ext}``, and
    ``{stem}_canvas{ext}`` alongside the output path and returns the
    final result image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str: str = device.type
    dtype = torch.bfloat16 if device_str == "cuda" else torch.float32

    architect_pipe, artist_pipe, detector, seg_parser = load_models(args, device, dtype)

    # Load reference images
    refs: list[Image.Image] = []
    for p in args.faces:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Face image not found: {p}")
        refs.append(Image.open(p).convert("RGB"))
    k = len(refs)
    print(f"[Info] {k} reference image(s) loaded")

    prompt = args.prompt

    # Stage A — generate probe
    print("[Stage A] Generating probe with Architect …")
    gen_a = torch.Generator(device="cpu").manual_seed(args.seed)
    with torch.inference_mode():
        probe: Image.Image = (
            architect_pipe(
                prompt,
                guidance_scale=args.guidance_architect,
                num_inference_steps=args.steps_architect,
                height=PROBE_H,
                width=PROBE_W,
                generator=gen_a,
            )
            .images[0]
            .convert("RGB")
        )

    # Face detection on probe
    boxes, scores = detector.detect(probe)
    if len(boxes) >= k:
        boxes = pick_topk_boxes(boxes, scores, k)
    else:
        if boxes:
            best = pick_topk_boxes(boxes, scores, 1)[0]
            while len(boxes) < k:
                boxes.append(best.copy())
        else:
            per_row = int(np.ceil(np.sqrt(k)))
            rows = int(np.ceil(k / per_row))
            gw, gh = PROBE_W // per_row, PROBE_H // rows
            boxes = []
            for idx in range(k):
                r, c = divmod(idx, per_row)
                boxes.append(
                    [
                        c * gw,
                        r * gh,
                        min(PROBE_W, c * gw + gw),
                        min(PROBE_H, r * gh + gh),
                    ]
                )

    # Sort boxes top-to-bottom (tallest person first)
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)

    # Optional: detect faces in ref images for alignment
    ref_face_boxes: list[tuple[int, int, int, int] | None] = [None] * k
    if not args.no_align:
        ref_detector = FacerDetector(
            device_str=device_str,
            backbone=DET_BACKBONE,
            score_thresh=DET_SCORE,
        )
        for i, ref in enumerate(refs):
            try:
                ref_face_boxes[i] = detect_best_face_box(ref, ref_detector)
            except Exception:
                pass

    # Stage B — build per-subject canvases
    canvases, fg_masks, combined_canvas = build_canvases(
        refs=refs,
        boxes=boxes,
        probe_wh=(PROBE_W, PROBE_H),
        ref_face_boxes=ref_face_boxes,
        seg_parser=seg_parser,
        mask_mode=args.mask_mode,
        face_scale=args.face_scale,
        target_pad=args.target_pad,
        align=not args.no_align,
    )

    # Stage B — Artist
    print("[Stage B] Running Artist …")
    cond_masks = [m.astype(np.float32) for m in fg_masks] if args.use_fg_mask else None
    gen_b = torch.Generator(device="cpu").manual_seed(args.seed)

    with torch.inference_mode():
        if args.use_fg_mask:
            out = artist_pipe(
                image=canvases,
                prompt=prompt,
                height=PROBE_H,
                width=PROBE_W,
                num_inference_steps=args.steps_artist,
                guidance_scale=args.guidance_artist,
                generator=gen_b,
                cond_image_mask=cond_masks,
            )
        else:
            out = artist_pipe(
                image=canvases,
                prompt=prompt,
                height=PROBE_H,
                width=PROBE_W,
                num_inference_steps=args.steps_artist,
                guidance_scale=args.guidance_artist,
                generator=gen_b,
            )

    result: Image.Image = out.images[0] if hasattr(out, "images") else out[0]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    stem, ext = os.path.splitext(args.output)
    ext = ext or ".png"

    result.save(args.output)
    print(f"[Done] Saved → {args.output}")

    probe_path = f"{stem}_probe{ext}"
    probe.save(probe_path)
    print(f"[Done] Saved → {probe_path}")

    canvas_path = f"{stem}_canvas{ext}"
    combined_canvas.save(canvas_path)
    print(f"[Done] Saved → {canvas_path}")

    return result


if __name__ == "__main__":
    run(parse_args())
