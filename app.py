#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Gradio demo: two-stage (Architect -> Canvas -> Artist) pipeline.

with the SAME defaults as in pipeline/canvas.py, plus controls to:
  - upload up to 5 reference images
  - provide a text prompt
  - adjust Artist timesteps and guidance scale
  - optionally use foreground masks as cond_image_mask
    (from padded target boxes)
  - Prompt Cache with precomputed Architect probes
  - On startup: load/normalize the Prompt Cache from disk and AUTO-generate
    missing probes with Architect, saving them under saved_probes/

Environment overrides (optional):
  ARCHITECT_DIR, ARCHITECT_LORA, ARTIST_DIR, ARTIST_LORA, ARTIST_LORA_2
  SAVED_PROBES_DIR, PROMPT_CACHE_JSON, AUTO_GENERATE_CACHE_PROBES (1/0)

Run:
  python -m pip install -r requirements.txt
  python app.py
"""

import asyncio
import json
import os
from typing import Any

# Disable Gradio analytics (avoids outbound httpx call that
# fails with no internet)
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Force uvicorn to use the standard asyncio loop instead of uvloop.
# Older uvloop builds lack new_event_loop, causing uvicorn's
# server thread to crash
# before it can handle requests, which then makes Gradio's
# startup health-check fail.
try:
    import uvicorn.loops.auto as _uv_auto

    _uv_auto.auto_loop_factory = lambda use_subprocess=False: asyncio.new_event_loop
except Exception as _e:
    print(f"[Warn] Could not patch uvicorn loop factory: {_e}")

# Belt-and-suspenders: also patch the uvloop module attribute directly
try:
    import uvloop as _uvloop

    if not hasattr(_uvloop, "new_event_loop"):
        _uvloop.new_event_loop = asyncio.new_event_loop
except ImportError:
    pass

# --- Canvas visualization helpers ---
from math import ceil

import gradio as gr
import numpy as np
import torch
from PIL import Image


def _make_montage(canvases: list[Image.Image], cols: int = 3) -> Image.Image:
    """Tile a list of equally-sized canvases into a simple montage image."""
    if not canvases:
        return Image.new("RGB", (1024, 1024), color=(240, 240, 240))
    w, h = canvases[0].size
    n = len(canvases)
    cols = max(1, min(cols, n))
    rows = ceil(n / cols)
    out = Image.new("RGB", (cols * w, rows * h), color=(255, 255, 255))
    for i, im in enumerate(canvases):
        r, c = divmod(i, cols)
        out.paste(im, (c * w, r * h))
    return out


def visualize_generated_canvas(
    canvases: list[Image.Image],
    combined: Image.Image | None,
) -> Image.Image:
    """Return the combined canvas if provided; otherwise a montage
    of per-face canvases.
    This is meant for showing a debug preview in the UI.
    """
    if combined is not None:
        return combined
    return _make_montage(canvases, cols=3)


# Reuse utilities/classes from pipeline/canvas.py to avoid code drift.
from diffusers import DiffusionPipeline  # noqa: E402

from pipeline.artist_pipeline_multicond import (  # noqa: E402
    FluxArtistPipeline,
)
from pipeline.canvas import (  # noqa: E402
    FacerDetector,
    FacerParser,
    build_canvases,
    detect_best_face_box,
    pick_topk_boxes,
)

try:
    from peft import PeftModel  # type: ignore

    PEFT_AVAILABLE = True
except Exception:  # pragma: no cover
    PeftModel = None
    PEFT_AVAILABLE = False

# -----------------------------------------------------------------------------
# Defaults: mirror pipeline/canvas.py
# -----------------------------------------------------------------------------
PROBE_H = int(os.getenv("PROBE_H", 1024))
PROBE_W = int(os.getenv("PROBE_W", 1024))

DET_BACKBONE = os.getenv("DET_BACKBONE", "retinaface/mobilenet")
DET_SCORE = float(os.getenv("DET_SCORE", 0.5))

USE_SEGMENTATION = os.getenv("USE_SEGMENTATION", "1") not in {
    "0",
    "false",
    "False",
}
SEG_BACKBONE = os.getenv("SEG_PARSE_BACKBONE", "farl/celebm/448")

ALIGN_REF_FACES = os.getenv("ALIGN_REF_FACES", "1") not in {
    "0",
    "false",
    "False",
}
FACE_SCALE_DEFAULT = float(os.getenv("FACE_SCALE", 1.10))
TARGET_PAD_DEFAULT = float(os.getenv("TARGET_PAD", 1.20))

# Model locations (override via env on your rig)
ARCHITECT_DIR = os.getenv("ARCHITECT_DIR", "black-forest-labs/FLUX.1-schnell")
ARCHITECT_LORA = os.getenv("ARCHITECT_LORA", "loras/architect")
ARTIST_DIR = os.getenv("ARTIST_DIR", "black-forest-labs/FLUX.1-Kontext-dev")

ARTIST_LORA = os.getenv("ARTIST_LORA", "loras/artist")
ARTIST_LORA_2 = os.getenv("ARTIST_LORA_2", None)
HF_REPO_ID = os.getenv("AR2CAN_LORA_HF", "Qualcomm-AI-Research/ar2can")

# Architect defaults
GUIDANCE_ARCHITECT = float(os.getenv("GUIDANCE_ARCHITECT", 0.0))
STEPS_ARCHITECT = int(os.getenv("STEPS_ARCHITECT", 4))

# Artist defaults (user controls via sliders)
GUIDANCE_ARTIST_DEFAULT = float(os.getenv("GUIDANCE_ARTIST", 2.4))
STEPS_ARTIST_DEFAULT = int(os.getenv("STEPS_ARTIST", 20))

# Saved probes directory & manifest
SAVED_PROBES_DIR = os.getenv("SAVED_PROBES_DIR", "saved_probes")
MANIFEST_PATH = os.path.join(SAVED_PROBES_DIR, "manifest.json")
PROMPTS_TXT = os.path.join(SAVED_PROBES_DIR, "prompts.txt")
PROMPT_CACHE_JSON = os.getenv(
    "PROMPT_CACHE_JSON", MANIFEST_PATH
)  # prefer external JSON if provided
AUTO_GENERATE_CACHE_PROBES = os.getenv("AUTO_GENERATE_CACHE_PROBES", "1") not in {
    "0",
    "false",
    "False",
}

os.makedirs(SAVED_PROBES_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Prompt Cache (initial in-code defaults). You can pre-fill here if you want.
#   We WILL merge this with on-disk JSON (if present) and normalize before UI.
#   Each entry must be a dict with keys: 'prompt' (str) and
#   'probe_path' (str|None)
# -----------------------------------------------------------------------------

# --- Prompt Cache (same set, probes ignored now) ---
PROMPT_CACHE_DEFAULT: dict[str, dict[str, str | None]] = {
    "Baking_2": {
        "prompt": (
            "Two friends baking in the kitchen, 8K, ultra-realistic,"
            " realistic light rendering, high dynamic range. The person"
            " on the left is rolling dough, the person on the right is"
            " operating a stand mixer which contains the label"
            " 'AI Research'. Realistic limb placement and person"
            " ordering. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Baking_3": {
        "prompt": (
            "Three friends baking in the kitchen, 8K, ultra-realistic,"
            " realistic light rendering, high dynamic range. The person"
            " on the left is rolling dough, the person in the middle is"
            " observing, the person on the right is operating a stand"
            " mixer which contains the label 'AI Research'. Realistic"
            " limb placement and person ordering. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Sunflower_2": {
        "prompt": (
            "A stunning portrait of two people kneeling in a sunflower"
            " field. Realistic image, realistic lighting and shadows,"
            " brillaint photography. It is really sunny outside and"
            " their hair is lit with the sunrays. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Sunflower_3": {
        "prompt": (
            "A stunning portrait of three people kneeling in a"
            " sunflower field. Realistic image, realistic lighting and"
            " shadows, brillaint photography. It is really sunny"
            " outside and their hair is lit with the sunrays. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Lights_2": {
        "prompt": (
            "Two people standing under green northern lights, 8K,"
            " ultra-realistic, realistic light rendering,"
            " high dynamic range. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Lights_3": {
        "prompt": (
            "Three people, standing in the dark night, under green"
            " northern lights, 8K, ultra-realistic, realistic shadows"
            " and darkness, high dynamic range, group harmony. From"
            " left to right, the people are wearing: A black leather"
            " jacket,  A large winter coat, a Grey muffler. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Lights_4": {
        "prompt": (
            "Four people, standing in the dark night, under green"
            " northern lights, 8K, ultra-realistic, realistic shadows"
            " and darkness, high dynamic range, group harmony. From"
            " left to right, the people are wearing: A black leather"
            " jacket,  A large winter coat, a Grey muffler,"
            " A yellow sweater. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Coffee_2": {
        "prompt": (
            "Two friends laughing sitting on a couch in a coffee shop."
            " 8K, ultra-realistic, realistic light rendering, high"
            " dynamic range. The coffee table has a flowerpot. The"
            " person on the left is wearing sunglasses and a striped"
            " tshirt. The person on the right has a yellow shirt and"
            " red trousers. Realistic limb placements. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Coffee_4": {
        "prompt": (
            "Four friends laughing  sitting on a couch in a coffee"
            " shop. 8K, ultra-realistic, realistic light rendering,"
            " high dynamic range. The coffee table has a flowerpot."
            " From left to right, the people are wearing: sunglasses,"
            " black suit+blue tie, scarf, blue jumpsuit. Realistic"
            " limb placements and people ordering. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Coffee_3": {
        "prompt": (
            "Three friends laughing  sitting on a couch in a coffee"
            " shop. 8K, ultra-realistic, realistic light rendering,"
            " high dynamic range. The coffee table has a flowerpot."
            " The person on the left is wearing sunglasses. The person"
            " in the middle is wearing a black suit and blue tie. The"
            " person on the right has a scarf. Realistic limb"
            " placements. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Soccer_4": {
        "prompt": (
            "Four people playing soccer,  blue fire in the background,"
            " realistic light and shadows rendering, group harmony,"
            " ultra-realistic. blue fire in front of the players,"
            " covering them. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Soccer_3": {
        "prompt": (
            "A real full-length portrait image of three soccer players"
            " in action at a stadium under strobe lights.  blue fire"
            " in the background. Realistic image, realistic lighting"
            " and shadows, brillaint photography. The jerseys read"
            " 'AIR FC'. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Beach_2": {
        "prompt": (
            "A portrait picture of Two people sitting on a log, near a"
            " palm tree on a beach.The person on the left is wearing"
            " sunglasses and a denim shirt. The person on the  right"
            " is wearing sunglasses and barefoot. Ultra-Realistic. 8K."
            " Realistic image, realistic lighting and shadows,"
            " brilliant photography. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Beach_3": {
        "prompt": (
            "A portrait picture of Three people sitting on a log, near"
            " a palm tree on a beach.The person on the left is wearing"
            " sunglasses and a denim shirt. The person in the middle"
            " is wearing black sunglasses. The person on the  right is"
            " wearing sunglasses and barefoot. Ultra-Realistic. 8K."
            " Realistic image, realistic lighting and shadows,"
            " brilliant photography. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    # New entries you requested
    "Kitchen_2": {
        "prompt": (
            "A traditional portrait of Two people  in a kitchen: the"
            " person on the left is chopping onions with a knife, and"
            " the person on a right is stirring a large blue wok with"
            " one hand using a wooden spatula. Ultra-Realistic. 8K."
            " Realistic image, realistic lighting and shadows,"
            " brilliant photography. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Kitchen_3": {
        "prompt": (
            "A traditional portrait of a total number of Three people"
            " in a kitchen: the person on the left is chopping"
            " vegetables with a knife, the person in the middle is"
            " standing, and the person on the right is stirring a"
            " large blue wok with one hand using a wooden spatula."
            " Ultra-Realistic. 8K. Realistic image, realistic lighting"
            " and shadows, brilliant photography. Realistic limb"
            " placement and person ordering. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Mansion_2": {
        "prompt": (
            "A full-length portrait of Two people in a dark mansion;"
            " faces clearly visible. Smoke all around. The person on"
            " the left is wearing a white blazer, and has crossed"
            " arms. The person on the right is wearing a blue sweater"
            " and waving. Ultra-Realistic. 8K. Realistic image,"
            " realistic lighting and shadows, brilliant photography. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Mansion_3": {
        "prompt": (
            "A full-length portrait of Three people in a dark mansion;"
            " faces clearly visible. Smoke all around. The person on"
            " the left is wearing a white blazer, and has crossed"
            " arms. The person on the right is wearing a blue sweater"
            " and waving. Ultra-Realistic. 8K. Realistic image,"
            " realistic lighting and shadows, brilliant photography."
            " The person in the middle is wearing sunglasses. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Mansion_4": {
        "prompt": (
            "A dark full-length portrait of Four people in a mansion."
            " Smoke all around. The person on the left is wearing a"
            " white blazer, and has crossed arms. The person on the"
            " right is wearing a blue sweater and waving."
            " Ultra-Realistic. 8K. Realistic image, realistic darkness"
            " and shadows, brilliant photography. The person in the"
            " middle is wearing sunglasses. Realistic anatomy. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Patio_5": {
        "prompt": (
            "A DSLR picture of Five people in a patio shed outside a"
            " house. The sun is visible. lots of sunrays all over the"
            " people. Portrait picture. Ultra-Realistic. 8K. Realistic"
            " image, realistic lighting and shadows, brilliant"
            " photography. Sunlight falling on the people. realistic"
            " person ordering and anatomy. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "CrimeScene_5": {
        "prompt": (
            "Five investigators inspecting a crime scene; the person"
            " on the left is taking notes. Faces clearly visible."
            " Portrait picture. Ultra-Realistic. 8K. Realistic image,"
            " realistic lighting and shadows, brilliant photography."
            " Realistic occlusions between people. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Forest_2": {
        "prompt": (
            "A DSLR picture of  Two people in a sunny forest. Sunrays"
            " through the trees. Ultra-Realistic. 8K.. Realistic"
            " image, realistic lighting and shadows, brillaint"
            " photography. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Forest_3": {
        "prompt": (
            "A DSLR picture of  Three people in a sunny forest."
            " Sunrays through the trees. Ultra-Realistic. 8K.."
            " Realistic image, realistic lighting and shadows,"
            " brillaint photography. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
    "Forest_4": {
        "prompt": (
            "A DSLR picture of  Four people in a sunny forest. Sunrays"
            " through the trees. Ultra-Realistic. 8K.. Realistic"
            " image, realistic lighting and shadows, brillaint"
            " photography. Preserve exact facial identity."
        ),
        "probe_path": None,
    },
}

#

# Runtime working cache after initialization
PROMPT_CACHE: dict[str, dict[str, str | None]] = {}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _slugify(s: str) -> str:
    """Convert *s* to a URL-safe slug (lowercase alphanumeric + hyphens)."""
    keep = [c.lower() if c.isalnum() else "-" for c in s]
    out = "".join(keep)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "probe"


def _validate_cache_entry(name: str) -> tuple[bool, str]:
    """Ensure cache entry is a dict with a non-empty 'prompt' string."""
    if name not in PROMPT_CACHE:
        return False, f"[Cache] '{name}' not found."
    entry = PROMPT_CACHE.get(name)
    if not isinstance(entry, dict):
        return False, f"[Cache] Entry for '{name}' is not a dict."
    prompt_text = entry.get("prompt")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        return False, (f"[Cache] Entry for '{name}' is missing a non-empty 'prompt'.")
    return True, ""


def _cache_choices() -> list[str]:
    """Return names of all prompt-cache entries that have a valid
    prompt string.
    """
    # Only expose entries that have a valid dict with a non-empty 'prompt'
    ok_names = []
    for nm, ent in PROMPT_CACHE.items():
        if isinstance(ent, dict) and isinstance(ent.get("prompt"), str) and ent["prompt"].strip():
            ok_names.append(nm)
    return ok_names


# -----------------------------------------------------------------------------
# Globals for models (loaded once)
# -----------------------------------------------------------------------------
architect_pipe: DiffusionPipeline | None = None
artist_pipe: FluxArtistPipeline | None = None
detector: FacerDetector | None = None
seg_parser: FacerParser | None = None

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
device_str = "cuda" if device.type == "cuda" else "cpu"
architect_dtype = torch.bfloat16 if device_str == "cuda" else torch.float32
artist_dtype = torch.bfloat16 if device_str == "cuda" else torch.float32


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------
def load_models() -> None:
    """Load all pipeline models into the module-level globals.

    Models are only loaded once; subsequent calls are no-ops for already-loaded
    components.  LoRA adapters are applied when the configured paths exist.
    """
    global architect_pipe, artist_pipe, detector, seg_parser

    # Architect
    if architect_pipe is None:
        print("[Load] FLUX-Architect from:", ARCHITECT_DIR)
        architect_pipe = DiffusionPipeline.from_pretrained(
            ARCHITECT_DIR, torch_dtype=architect_dtype
        )
        architect_pipe = architect_pipe.to(device)
        if (
            PEFT_AVAILABLE
            and hasattr(architect_pipe, "transformer")
            and architect_pipe.transformer is not None
        ):
            try:
                if ARCHITECT_LORA and os.path.isfile(
                    os.path.join(ARCHITECT_LORA, "adapter_model.safetensors")
                ):
                    arch_src, arch_kw = ARCHITECT_LORA, {}
                else:
                    print(f"[Load] Architect LoRA not found locally — loading from HuggingFace: {HF_REPO_ID}")
                    arch_src, arch_kw = HF_REPO_ID, {"subfolder": "architect"}
                architect_pipe.transformer = PeftModel.from_pretrained(
                    architect_pipe.transformer, arch_src, **arch_kw
                )
                architect_pipe.transformer.set_adapter("default")
                print(f"[Load] Architect LoRA loaded: {arch_src}")
            except Exception as e:
                print(f"[Warn] Architect LoRA load failed: {e}")

    # Detector
    if detector is None:
        print(f"[Load] Facer detector: {DET_BACKBONE}" f" (score>={DET_SCORE}) on {device_str}")
        detector = FacerDetector(
            device_str=device_str,
            backbone=DET_BACKBONE,
            score_thresh=DET_SCORE,
        )

    # Segmentation parser (optional)
    if seg_parser is None and USE_SEGMENTATION:
        try:
            print(f"[Load] Facer parser: {SEG_BACKBONE} on {device_str}")
            seg_parser = FacerParser(
                device_str=device_str,
                backbone=SEG_BACKBONE,
                det_backbone=DET_BACKBONE,
            )
        except Exception as e:
            print(f"[Warn] Segmentation disabled (init failed): {e}")

    # Artist
    if artist_pipe is None:
        print("[Load] Artist from:", ARTIST_DIR)
        artist_pipe = FluxArtistPipeline.from_pretrained(ARTIST_DIR, torch_dtype=artist_dtype)
        artist_pipe = artist_pipe.to(device)
        if (
            PEFT_AVAILABLE
            and hasattr(artist_pipe, "transformer")
            and artist_pipe.transformer is not None
        ):
            try:
                if ARTIST_LORA and os.path.isfile(
                    os.path.join(ARTIST_LORA, "adapter_model.safetensors")
                ):
                    art_src, art_kw = ARTIST_LORA, {}
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
                print(f"[Warn] Artist LoRA load failed: {e}")
        if ARTIST_LORA_2:
            try:
                artist_pipe.load_lora_weights(ARTIST_LORA_2)
                print(f"[Load] Artist secondary LoRA loaded: {ARTIST_LORA_2}")
            except Exception as e:
                print(f"[Warn] Artist secondary LoRA load failed: {e}")


# -----------------------------------------------------------------------------
# Probe generation & cache initialization
# -----------------------------------------------------------------------------
def _generate_probe_with_architect(prompt_text: str, seed: int = 0) -> Image.Image:
    """Generate a probe image with the Architect pipeline.

    Args:
        prompt_text: Scene description prompt.
        seed: RNG seed for reproducibility.

    Returns:
        A 1024×1024 RGB probe image.
    """
    if architect_pipe is None:
        raise RuntimeError("Architect model is not loaded.")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    with torch.inference_mode():
        img = architect_pipe(
            prompt_text,
            guidance_scale=GUIDANCE_ARCHITECT,
            num_inference_steps=STEPS_ARCHITECT,
            height=PROBE_H,
            width=PROBE_W,
            generator=gen,
        ).images[0]
    return img.convert("RGB")


def _ensure_saved_probe_for_cache_name(name: str, seed: int = 0) -> tuple[str | None, str]:
    """Generate and persist a probe image for the cache entry *name*
    if missing.

    Args:
        name: Key in ``PROMPT_CACHE``.
        seed: RNG seed passed to the Architect.

    Returns:
        A tuple ``(probe_path, message)`` where *probe_path* is the saved file
        path (or ``None`` on failure) and *message* describes the outcome.
    """
    ok, err = _validate_cache_entry(name)
    if not ok:
        return None, err
    entry = PROMPT_CACHE[name]
    prompt_text = entry["prompt"].strip()
    current_path = entry.get("probe_path")
    if current_path and os.path.isfile(current_path):
        return (
            current_path,
            f"[Cache] '{name}' already has probe: {current_path}",
        )

    os.makedirs(SAVED_PROBES_DIR, exist_ok=True)
    fname = f"{_slugify(name)}.jpg"
    out_path = os.path.join(SAVED_PROBES_DIR, fname)
    img = _generate_probe_with_architect(prompt_text, seed=seed)
    img.save(out_path, quality=92)

    entry["probe_path"] = out_path

    # Persist manifest & prompts list (full rewrite for simplicity)
    try:
        manifest = []
        for k, v in PROMPT_CACHE.items():
            manifest.append(
                {
                    "name": k,
                    "prompt": v.get("prompt", ""),
                    "probe_path": v.get("probe_path", None),
                }
            )
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2)
        with open(PROMPTS_TXT, "w") as f:
            for k, v in PROMPT_CACHE.items():
                p = v.get("prompt")
                if isinstance(p, str) and p.strip():
                    f.write(p.strip() + "\n")
    except Exception as e:
        return (
            out_path,
            f"[Cache] Saved probe but failed to update" f" manifest/prompts: {e}",
        )

    return (
        out_path,
        f"[Cache] Generated & saved probe for '{name}' -> {out_path}",
    )


def _load_cache_from_json(path: str) -> dict[str, dict[str, str | None]]:
    """Load cache entries from a JSON manifest:
    list of {name, prompt, probe_path}.
    """
    result: dict[str, dict[str, str | None]] = {}
    if not path or not os.path.isfile(path):
        return result
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                name = item.get("name") if isinstance(item, dict) else None
                prompt = item.get("prompt") if isinstance(item, dict) else None
                probe_path = item.get("probe_path") if isinstance(item, dict) else None
                if (
                    isinstance(name, str)
                    and name.strip()
                    and isinstance(prompt, str)
                    and prompt.strip()
                ):
                    result[name] = {
                        "prompt": prompt.strip(),
                        "probe_path": probe_path,
                    }
    except Exception as e:
        print(f"[Cache] Failed to load JSON cache from {path}: {e}")
    return result


def initialize_prompt_cache(auto_generate: bool = True) -> None:
    """Build PROMPT_CACHE before startup.

    - Start from PROMPT_CACHE_DEFAULT
    - Merge in on-disk JSON (PROMPT_CACHE_JSON if exists)
    - Normalize entries (dict with non-empty 'prompt')
    - If auto_generate: run Architect to fill missing probes,
      saving under saved_probes/
    """
    global PROMPT_CACHE

    # 1) Start with defaults
    merged: dict[str, dict[str, str | None]] = {}
    for name, ent in PROMPT_CACHE_DEFAULT.items():
        if isinstance(ent, dict) and isinstance(ent.get("prompt"), str) and ent["prompt"].strip():
            merged[name] = {
                "prompt": ent["prompt"].strip(),
                "probe_path": ent.get("probe_path"),
            }

    # 2) Merge in JSON if available
    disk_cache = _load_cache_from_json(PROMPT_CACHE_JSON)
    for name, ent in disk_cache.items():
        merged[name] = {
            "prompt": ent.get("prompt", "").strip(),
            "probe_path": ent.get("probe_path"),
        }

    # 3) Normalize & drop invalid
    PROMPT_CACHE = {}
    for name, ent in merged.items():
        if isinstance(ent, dict) and isinstance(ent.get("prompt"), str) and ent["prompt"].strip():
            PROMPT_CACHE[name] = {
                "prompt": ent["prompt"].strip(),
                "probe_path": ent.get("probe_path"),
            }

    # 4) Optionally auto-generate any missing probes now (before UI)
    if auto_generate and PROMPT_CACHE:
        print(f"[Cache] Auto-generating missing probes" f" (count={len(PROMPT_CACHE)}) ...")
        for name in list(PROMPT_CACHE.keys()):
            path, msg = _ensure_saved_probe_for_cache_name(name, seed=0)
            print(msg)
    else:
        print("[Cache] Auto-generate disabled or no cache entries present.")


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
def run_inference(  # noqa: D417
    prompt: str,
    img1: Image.Image | None,
    img2: Image.Image | None,
    img3: Image.Image | None,
    img4: Image.Image | None,
    img5: Image.Image | None,
    steps_artist: int,
    guidance_artist: float,
    use_fg_cond_mask: bool,
    cached_name: str,
    use_cached: bool,
    face_scale: float,
    target_pad: float,
    mask_mode: str = "loose",
    height1: float = 1,
    height2: float = 2,
    height3: float = 3,
    height4: float = 4,
    height5: float = 5,
) -> tuple[list[Image.Image], Image.Image]:
    """Run the two-stage ar2can pipeline and return generated images.

    Args:
        prompt: Scene description text.
        img1–img5: Reference face images (``None`` slots are ignored).
        steps_artist: Number of Artist diffusion steps.
        guidance_artist: Artist CFG guidance scale.
        use_fg_cond_mask: Whether to pass foreground masks as
            ``cond_image_mask``.
        cached_name: Name of a cached probe entry to use (when *use_cached*).
        use_cached: Use the pre-generated probe for *cached_name* instead of
            running the Architect.
        face_scale: Scale factor applied during face-aligned paste.
        target_pad: Box expansion ratio applied to the detected probe faces.
        mask_mode: Segmentation mask mode (``"loose"`` or ``"tight"``).
        height1–height5: Height-rank values for ordering reference images
            top-to-bottom in the scene (rank 1 = tallest).

    Returns:
        A tuple ``(final_images, combined_canvas)``.

    Raises:
        RuntimeError: If the models have not been loaded before calling.
        gr.Error: If no reference images are provided.
    """
    if architect_pipe is None or artist_pipe is None or detector is None:
        raise RuntimeError("Models are not loaded. Call load_models() before run_inference().")
    # Collect up to 5 images, paired with their height ranks
    all_imgs = [img1, img2, img3, img4, img5]
    all_ranks = [height1, height2, height3, height4, height5]
    ref_pairs = [
        (im, float(hr) if hr is not None else float(i + 1))
        for i, (im, hr) in enumerate(zip(all_imgs, all_ranks))
        if im is not None
    ]
    if len(ref_pairs) == 0:
        raise gr.Error("Please upload at least one reference image (up to 5).")

    # Sort refs by height rank ascending (rank 1 = tallest ->
    # highest face position)
    ref_pairs_sorted = sorted(ref_pairs, key=lambda x: x[1])
    refs: list[Image.Image] = [p[0] for p in ref_pairs_sorted]
    k = min(5, len(refs))

    # Stage A: Probe (cached if requested, else Architect)
    probe: Image.Image
    if use_cached and cached_name:
        ok, err = _validate_cache_entry(cached_name)
        if ok:
            entry = PROMPT_CACHE[cached_name]
            pth = entry.get("probe_path")
            if (not pth) or (pth and not os.path.isfile(pth)):
                pth, msg = _ensure_saved_probe_for_cache_name(cached_name, seed=0)
                print(msg)
            if pth and os.path.isfile(pth):
                print(f"[Cache] Using precomputed probe for" f" '{cached_name}' -> {pth}")
                probe = Image.open(pth).convert("RGB")
            else:
                print(
                    f"[Cache] Could not resolve probe for"
                    f" '{cached_name}'. Falling back to Architect."
                )
                probe = _generate_probe_with_architect(prompt_text=prompt, seed=0)
        else:
            print(err)
            probe = _generate_probe_with_architect(prompt_text=prompt, seed=0)
    else:
        probe = _generate_probe_with_architect(prompt_text=prompt, seed=0)

    # Detection on probe
    boxes, scores = detector.detect(probe)

    # Adjust number of boxes to number of refs
    if len(boxes) >= k:
        boxes = pick_topk_boxes(boxes, scores, k)
    else:
        if boxes:
            best = pick_topk_boxes(boxes, scores, 1)[0]
            while len(boxes) < k:
                boxes.append(best.copy())
        else:
            # Fallback: grid over probe size
            per_row = int(np.ceil(np.sqrt(k)))
            rows = int(np.ceil(k / per_row))
            boxes = []
            gw = PROBE_W // per_row
            gh = PROBE_H // rows
            for idx in range(k):
                r, c = divmod(idx, per_row)
                x1, y1 = c * gw, r * gh
                x2, y2 = min(PROBE_W, x1 + gw), min(PROBE_H, y1 + gh)
                boxes.append([x1, y1, x2, y2])

    # Sort boxes by vertical center ascending: lower y_center =
    # higher in image = taller position.
    # refs are already sorted by height rank (rank 1 = tallest),
    # so refs[0] maps to boxes[0] (highest).
    boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)

    # Optional reference face detection for alignment
    ref_face_boxes: list[tuple[int, int, int, int] | None] = [None] * k
    if ALIGN_REF_FACES:
        ref_detector = FacerDetector(
            device_str=device_str,
            backbone=DET_BACKBONE,
            score_thresh=DET_SCORE,
        )
        for i in range(k):
            try:
                ref_face_boxes[i] = detect_best_face_box(refs[i], ref_detector)
            except Exception:
                ref_face_boxes[i] = None

    # Build per-subject canvases & foreground masks
    canvases, fg_masks, combined_canvas = build_canvases(
        refs=refs,
        boxes=boxes,
        probe_wh=(PROBE_W, PROBE_H),
        ref_face_boxes=ref_face_boxes,
        seg_parser=seg_parser if USE_SEGMENTATION else None,
        mask_mode=mask_mode,
        face_scale=face_scale,
        target_pad=target_pad,
        align=ALIGN_REF_FACES,
    )

    # Prepare cond_image_mask based on toggle
    cond_masks: list[np.ndarray] | None = None
    if use_fg_cond_mask:
        cond_masks = [m.astype(np.float32) for m in fg_masks]

    # Stage B: Artist call
    gen2 = torch.Generator(device="cpu").manual_seed(0)
    # Disabled: "Distorted, distoreted limbs, distorted legs,
    # extra fingers, misplaced limbs, misplaced hands, bad anatomy,
    # unrealistic"
    negative_prompt = None

    final_images: list[Image.Image] = []
    try:
        if use_fg_cond_mask:
            out = artist_pipe(
                image=canvases,  # list of PIL Images
                prompt=prompt,
                height=1024,
                width=1024,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps_artist),
                guidance_scale=float(guidance_artist),
                generator=gen2,
                cond_image_mask=cond_masks,  # None or list of float32 masks
            )
            if hasattr(out, "images"):
                imgs = out.images
                if isinstance(imgs, list):
                    final_images.extend(imgs)
                else:
                    final_images.append(imgs)
        else:
            out = artist_pipe(
                image=combined_canvas,  # list of PIL Images
                prompt=prompt,
                height=1024,
                width=1024,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps_artist),
                guidance_scale=float(guidance_artist),
                generator=gen2,
                cond_image_mask=None,  # None or list of float32 masks
            )
            if hasattr(out, "images"):
                imgs = out.images
                if isinstance(imgs, list):
                    final_images.extend(imgs)
                else:
                    final_images.append(imgs)

    except Exception as e:
        # Fallback: one-by-one with corresponding mask (if provided)
        print(f"[Info] Artist list-call failed: {e}." " Falling back to per-canvas loop.")
        for i, can in enumerate(canvases):
            out = artist_pipe(
                image=can,
                prompt=prompt,
                height=1024,
                width=1024,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps_artist),
                guidance_scale=float(guidance_artist),
                generator=gen2,
                cond_image_mask=(cond_masks[i] if (cond_masks is not None) else None),
            )
            if hasattr(out, "images"):
                if isinstance(out.images, list):
                    final_images.extend(out.images)
                else:
                    final_images.append(out.images)

    if not final_images:
        final_images = canvases

    return final_images, combined_canvas


# -----------------------------------------------------------------------------
# Build UI AFTER initializing cache
# -----------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks UI for the ar2can demo."""
    css = """
    :root { --app-blue: #1e3a8a; --text-white: #ffffff; }

    /* EXCLUDE the top banner from the global background rule */
    .gradio-container,
    .gradio-container *:not(img):not(canvas):not(svg):not(#top-banner) {
      background-color: var(--app-blue) !important;
      color: var(--text-white) !important;
      border-color: rgba(255,255,255,0.35) !important;
    }

    /* Top red banner (now protected) */
    #top-banner {
      position: sticky; top: 0; z-index: 9999; width: 100%;
      background: #b30000 !important; /* add !important */
      color: #fff !important;
      text-align: center; padding: 14px 16px;
      font-weight: 700; font-size: 22px;
    }

    /* Inputs & buttons */
    input, textarea, select {
      background-color: var(--app-blue) !important;
      color: #ffffff !important;
    }
    input::placeholder, textarea::placeholder {
      color: rgba(255,255,255,0.85) !important;
    }
    #btn-generate, #btn-fill {
        background-color: #4B0082 !important; /* Dark purple */
        color: #ffffff !important;
        border: none !important;
    }
    * { box-shadow: none !important; }
    """

    demo = gr.Blocks(css=css, title="The Impossible Group Picture")
    with demo:
        gr.HTML('<div id="top-banner">The Impossible Group Picture</div>')
        gr.Markdown(
            """
            Upload **up to 5** reference images, enter a **prompt**,
            and adjust **Artist steps** and **guidance scale**.
            """
        )

        # Layout: left (prompt + controls + refs) | right (gallery)
        with gr.Row(equal_height=False):
            # LEFT COLUMN
            with gr.Column(scale=2):
                with gr.Group():
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe your scene...",
                        lines=2,
                    )
                    with gr.Row():
                        face_scale = gr.Slider(
                            minimum=0.5,
                            maximum=2.5,
                            value=FACE_SCALE_DEFAULT,
                            step=0.05,
                            label="Face scale (alignment)",
                        )
                        target_pad = gr.Slider(
                            minimum=0.5,
                            maximum=3.0,
                            value=TARGET_PAD_DEFAULT,
                            step=0.05,
                            label="Target pad (box expansion)",
                        )
                        steps = gr.Slider(
                            minimum=5,
                            maximum=60,
                            value=STEPS_ARTIST_DEFAULT,
                            step=1,
                            label="Time steps",
                        )
                        guidance = gr.Slider(
                            minimum=0.0,
                            maximum=10.0,
                            value=GUIDANCE_ARTIST_DEFAULT,
                            step=0.1,
                            label="Guidance scale",
                        )
                    use_fg = gr.Checkbox(
                        value=True,
                        label=("Use foreground cond_image_mask" " (from target boxes)"),
                    )
                    mask_mode = gr.Radio(
                        choices=["loose", "tight"],
                        value="loose",
                        label="Segmentation mask mode",
                        info=(
                            "Loose = face + neck + torso (with dilation)."
                            " Tight = face + hair only."
                        ),
                    )
                    run_btn = gr.Button(
                        "Generate",
                        variant="primary",
                        elem_id="btn-generate",
                    )

                gr.Markdown("### Reference Images (up to 5)")
                gr.Markdown(
                    "*Set **Height rank** for each person"
                    " (1 = tallest). Faces are assigned to detected"
                    " positions sorted top-to-bottom.*"
                )
                with gr.Row():
                    with gr.Column(min_width=160):
                        img1 = gr.Image(type="pil", label="Ref 1")
                        h1 = gr.Number(
                            value=1,
                            label="Height rank",
                            precision=0,
                            minimum=1,
                        )
                    with gr.Column(min_width=160):
                        img2 = gr.Image(type="pil", label="Ref 2")
                        h2 = gr.Number(
                            value=2,
                            label="Height rank",
                            precision=0,
                            minimum=1,
                        )
                    with gr.Column(min_width=160):
                        img3 = gr.Image(type="pil", label="Ref 3")
                        h3 = gr.Number(
                            value=3,
                            label="Height rank",
                            precision=0,
                            minimum=1,
                        )
                    with gr.Column(min_width=160):
                        img4 = gr.Image(type="pil", label="Ref 4")
                        h4 = gr.Number(
                            value=4,
                            label="Height rank",
                            precision=0,
                            minimum=1,
                        )
                    with gr.Column(min_width=160):
                        img5 = gr.Image(type="pil", label="Ref 5")
                        h5 = gr.Number(
                            value=5,
                            label="Height rank",
                            precision=0,
                            minimum=1,
                        )

            # RIGHT COLUMN (results beside the prompt)
            with gr.Column(scale=3):
                gallery = gr.Gallery(
                    label="Outputs",
                    columns=2,
                    height=700,
                    preview=True,
                )

        # --- Prompt Cache Section at the bottom ---
        gr.Markdown("### Prompt Cache (precomputed Architect probes)")
        with gr.Row():
            cached_selector = gr.Radio(
                choices=_cache_choices(),
                label="Choose a cached prompt",
                interactive=True,
            )
            cached_preview = gr.Image(label="Cached probe preview", visible=True)
            canvas_preview = gr.Image(label="Generated canvas preview", visible=True)
            cached_use = gr.Checkbox(value=True, label="Use selected cached probe")
            fill_btn = gr.Button("Fill prompt from selection", elem_id="btn-fill")

        # Cache interactions
        def _on_cache_change(
            name: str | None,
        ) -> tuple[Any, Image.Image | None]:
            """Update the prompt textbox and probe preview when
            the cache selection changes.
            """
            if not name:
                return gr.update(), gr.update(value=None)
            ok, err = _validate_cache_entry(name)
            if not ok:
                return gr.update(value=f"{err}"), gr.update(value=None)
            d = PROMPT_CACHE[name]
            txt = d["prompt"]
            pth = d.get("probe_path")
            img = Image.open(pth) if (pth and os.path.isfile(pth)) else None
            return (gr.update(value=txt), img)

        cached_selector.change(
            fn=_on_cache_change,
            inputs=[cached_selector],
            outputs=[prompt, cached_preview],
            show_progress=False,
        )

        def _fill_prompt(name: str | None, current: str) -> str:
            """Return the cached prompt text for *name*, or
            *current* if unavailable.
            """
            if not name:
                return current
            ok, err = _validate_cache_entry(name)
            if not ok:
                return current
            return PROMPT_CACHE[name]["prompt"]

        fill_btn.click(
            fn=_fill_prompt,
            inputs=[cached_selector, prompt],
            outputs=[prompt],
            show_progress=False,
        )

        # Inference wiring (includes mask toggle + cache selection)
        run_btn.click(
            fn=run_inference,
            inputs=[
                prompt,
                img1,
                img2,
                img3,
                img4,
                img5,
                steps,
                guidance,
                use_fg,
                cached_selector,
                cached_use,
                face_scale,
                target_pad,
                mask_mode,
                h1,
                h2,
                h3,
                h4,
                h5,
            ],
            outputs=[gallery, canvas_preview],
            show_progress=True,
            api_name="generate",
        )

        gr.Markdown(
            """
            **Notes**
            - Architect: steps = 4, guidance = 0.0 (fixed defaults).
            - Probe size: 1024x1024; detection via RetinaFace
              (score >= 0.5). If no detections, a grid fallback is used.
            - Alignment uses detected reference faces (if found).
              You can now tune `face_scale` and `target_pad` via the
              sliders.
            - Segmentation (FaRL) is enabled by default; if parser
              init fails it is silently disabled.
            - Toggle uses **foreground masks** (binary 0/1, float32)
              as `cond_image_mask` for Artist.
            - Prompt Cache is **pre-initialized before launch** from
              JSON/defaults; missing probes are auto-generated and
              saved under `saved_probes/` so you don't need to click
              anything.
            """
        )

    return demo


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if device_str != "cuda":
        print("[Warn] CUDA not available; running on CPU. This may be slow.")

    # 1) Load models first because we may need Architect to
    # generate cached probes
    load_models()
    if architect_pipe is None or artist_pipe is None or detector is None:
        raise RuntimeError("Model loading failed; cannot start the demo.")

    # 2) Initialize the Prompt Cache from defaults + JSON, and
    # auto-generate probes as needed
    initialize_prompt_cache(auto_generate=AUTO_GENERATE_CACHE_PROBES)

    # 3) Build UI AFTER cache is ready (radio choices reflect
    # the finalized cache)
    demo = build_ui()

    # 4) Launch
    try:
        demo.queue(max_size=12)
    except TypeError:
        demo.queue()

    # Use share=True for quick public link; set server_name/port
    # for SSH tunnel as needed
    demo.launch(
        server_name="0.0.0.0",
        server_port=7865,
        share=False,
        show_error=True,
    )
