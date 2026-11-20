#!/usr/bin/env python3
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Canvas and segmentation helpers for the ar2can demo.
Ported from layered_demo.py — excludes the CLI main() block and
diffusion pipeline imports.
"""

import logging
from typing import Optional

# Install dependencies from the repository root:
# python -m pip install -r requirements.txt (pins the pyfacer distribution).
import facer
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "FacerDetector",
    "FacerParser",
    "pick_topk_boxes",
    "detect_best_face_box",
    "build_single_canvas_1024",
    "build_canvases",
    "FGMASK_PAD",
    "MASK_THRESHOLD",
]

# =============================================================================
# Constants
# =============================================================================

# Padding ratio for the FOREGROUND MASK boxes (gives "room" around
# detected targets).
FGMASK_PAD: float = 2

# Threshold for binarising segmentation masks
# (pixel value > MASK_THRESHOLD → keep).
MASK_THRESHOLD: int = 127

# Proportion of image height used for downward neck-dilation in loose
# mask mode.
NECK_ROWS_RATIO: float = 0.18


# =============================================================================
# Box utilities
# =============================================================================


def pick_topk_boxes(boxes: list[list[float]], scores: list[float], k: int) -> list[list[float]]:
    """Return top-k boxes by score; if fewer than k, returns all."""
    if not boxes:
        return []
    score_arr = np.array([scores[i] if i < len(scores) else 0.0 for i in range(len(boxes))])
    order = np.argsort(score_arr)[::-1]
    return [boxes[int(i)] for i in order[:k]]


def box_center_size(
    box: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    """Return (center_x, center_y, width, height) for a (x1, y1, x2, y2)
    box.
    """
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = x1 + 0.5 * w, y1 + 0.5 * h
    return cx, cy, w, h


def expand_box_ratio(
    box: tuple[int, int, int, int],
    ratio: float,
    img_wh: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Expand *box* symmetrically about its centre by *ratio*
    (e.g. 1.2 = +20%).

    Optionally clamps the result to the image boundary given by *img_wh*
    (width, height).  Returns the box unchanged when *ratio* <= 1.0.
    """
    if ratio <= 1.0:
        return box
    x1, y1, x2, y2 = box
    cx, cy, w, h = box_center_size((x1, y1, x2, y2))
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    nx1, ny1 = int(round(cx - 0.5 * nw)), int(round(cy - 0.5 * nh))
    nx2, ny2 = nx1 + nw, ny1 + nh
    if img_wh is not None:
        W, H = img_wh
        nx1, ny1 = int(np.clip(nx1, 0, W)), int(np.clip(ny1, 0, H))
        nx2, ny2 = int(np.clip(nx2, 0, W)), int(np.clip(ny2, 0, H))
    return nx1, ny1, nx2, ny2


def clamp_box_to_canvas(
    box: tuple[int, int, int, int], wh: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Clamp a box to [0,W]x[0,H] and ensure it has positive area.

    If the clamped box collapses to zero area (degenerate input), returns the
    sentinel ``(0, 0, 1, 1)`` to guarantee a 1-pixel safe box for
    downstream use.
    """
    x1, y1, x2, y2 = box
    W, H = wh
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return (0, 0, 1, 1)
    return (x1, y1, x2, y2)


# =============================================================================
# Detection helpers
# =============================================================================


class FacerDetector:
    """Detect human/face boxes using RetinaFace via Facer."""

    def __init__(
        self,
        device_str: str = "cuda",
        backbone: str = "retinaface/mobilenet",
        score_thresh: float = 0.5,
    ) -> None:
        """Initialise the detector.

        Args:
            device_str: PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
                Falls back to ``"cpu"`` automatically when CUDA is unavailable.
            backbone: Facer detector backbone identifier.
            score_thresh: Minimum confidence score to keep a detection.
        """
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            device_str = "cpu"
        self.device_str = device_str
        self.detector = facer.face_detector(backbone, device=device_str)
        self.score_thresh = score_thresh

    @staticmethod
    def pil_to_bchw(img: Image.Image, device_str: str) -> torch.Tensor:
        """Convert a PIL image to a ``(1, 3, H, W)`` uint8 tensor on
        *device_str*.
        """
        arr = np.array(img.convert("RGB"), copy=False)
        t = torch.from_numpy(arr).to(torch.uint8).to(device_str)  # H,W,3
        t = t.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
        return t

    @torch.inference_mode()
    def detect(self, img: Image.Image) -> tuple[list[list[float]], list[float]]:
        """Run face detection on *img*.

        Returns:
            A tuple ``(boxes, scores)`` where each box is
            ``[x1, y1, x2, y2]`` in pixel coordinates.
        """
        bchw = self.pil_to_bchw(img, self.device_str)
        faces = self.detector(bchw)
        boxes, scores = [], []
        if faces["scores"].numel() == 0:
            return boxes, scores
        keep = faces["scores"] >= self.score_thresh
        rects = faces["rects"][keep].detach().cpu().numpy() if keep.any() else np.empty((0, 4))
        scs = faces["scores"][keep].float().cpu().numpy().tolist() if keep.any() else []
        for r in rects:
            boxes.append([float(r[0]), float(r[1]), float(r[2]), float(r[3])])
        scores = scs
        return boxes, scores


def detect_best_face_box(
    img: Image.Image,
    detector: "FacerDetector",
) -> tuple[int, int, int, int] | None:
    """Return the highest-scoring face box in *img*, or ``None`` if
    none found.
    """
    boxes, scores = detector.detect(img)
    if not boxes:
        return None
    best_i = 0
    if scores:
        best_i = max(
            range(len(boxes)),
            key=lambda i: scores[i] if i < len(scores) else 0.0,
        )
    x1, y1, x2, y2 = boxes[best_i]
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


# =============================================================================
# Segmentation (FaRL parser in Facer)
# =============================================================================
class FacerParser:
    """Face/upper-body parsing mask via Facer FaRL; returns an L-mode
    mask (255=keep).

    Backbone choice matters for neck coverage:
      farl/celebm/448  — CelebAMask-HQ (19 classes); class 17=neck,
                         18=cloth. Recommended.
      farl/lapa/448    — LaPa (11 classes); neck is background
                         (class 0) — neck NOT included.
    """

    def __init__(
        self,
        device_str: str = "cuda",
        backbone: str = "farl/celebm/448",
        det_backbone: str = "retinaface/mobilenet",
    ) -> None:
        """Initialise the parser (backbone and device — see class
        docstring).
        """
        if device_str == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            device_str = "cpu"
        self.device_str = device_str
        self.backbone = backbone
        self.parser = facer.face_parser(backbone, device=device_str)
        self.detector = facer.face_detector(det_backbone, device=device_str)

    @staticmethod
    def pil_to_bchw_uint8(img: Image.Image, device_str: str) -> torch.Tensor:
        """Convert a PIL image to a ``(1, 3, H, W)`` uint8 tensor on
        *device_str*.
        """
        arr = np.array(img.convert("RGB"), copy=False)
        t = torch.from_numpy(arr).to(torch.uint8).to(device_str)  # H,W,3
        t = t.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
        return t

    def make_mask(self, img: Image.Image, mode: str = "loose") -> Image.Image | None:
        """mode="tight" — face + hair only; excludes neck and cloth.
        mode="loose" — face + hair + neck + torso + downward dilation
        (default).
        """
        try:
            bchw = self.pil_to_bchw_uint8(img, self.device_str)
            with torch.inference_mode():
                det = self.detector(bchw)
                if det is None or det.get("scores", None) is None or det["scores"].numel() == 0:
                    return None
                scores = det["scores"]
                image_ids = torch.zeros_like(scores, dtype=torch.long)
                data = {
                    "rects": det.get("rects"),
                    "points": det.get("points"),
                    "scores": scores,
                    "image_ids": image_ids,
                }
                out = self.parser(bchw, data)
            seg = None
            if isinstance(out, dict):
                if "seg" in out and isinstance(out["seg"], dict) and "logits" in out["seg"]:
                    seg = out["seg"]["logits"]
                elif "seg" in out and isinstance(out["seg"], torch.Tensor):
                    seg = out["seg"]
                elif "seg_logits" in out:
                    seg = out["seg_logits"]
                elif "logits" in out:
                    seg = out["logits"]
            if seg is None:
                return None
            labels = seg.argmax(dim=1)  # (1,H,W)
            label_map = labels[0].detach().to("cpu").numpy().astype(np.int32)
            keep = label_map != 0
            is_celebm = "celebm" in self.backbone
            if mode == "tight" and is_celebm:
                # CelebAMask-HQ: exclude neck (17) and cloth (18)
                keep = keep & ~np.isin(label_map, [17, 18])
            elif mode == "loose":
                # Extend the mask downward to cover the neck region.
                # farl/celebm/448 labels neck natively (class 17);
                # farl/lapa/448 does not.
                # Either way, a small downward dilation ensures the
                # neck below the chin is covered.
                neck_rows = max(1, int(label_map.shape[0] * NECK_ROWS_RATIO))
                extended = keep.copy()
                for shift in range(1, neck_rows + 1):
                    extended[shift:] |= keep[:-shift]
                keep = extended
            mask = keep.astype(np.uint8) * 255
            return Image.fromarray(mask, mode="L")
        except Exception:
            logger.exception("make_mask failed; returning None")
            return None


# =============================================================================
# Canvas builders (1024x1024)
# =============================================================================


def _scale_box_to_canvas(
    box_probe: list[float],
    probe_wh: tuple[int, int],
    canvas_wh: tuple[int, int] = (1024, 1024),
) -> tuple[int, int, int, int]:
    """Scale a box from probe-image coordinates to canvas coordinates."""
    x1, y1, x2, y2 = box_probe
    Wp, Hp = probe_wh
    W_canvas, H_canvas = canvas_wh
    sx = (W_canvas / float(Wp)) if Wp > 0 else 1.0
    sy = (H_canvas / float(Hp)) if Hp > 0 else 1.0
    x1s = int(round(min(x1 * sx, x2 * sx)))
    x2s = int(round(max(x1 * sx, x2 * sx)))
    y1s = int(round(min(y1 * sy, y2 * sy)))
    y2s = int(round(max(y1 * sy, y2 * sy)))
    x1s = max(0, min(W_canvas, x1s))
    x2s = max(0, min(W_canvas, x2s))
    y1s = max(0, min(H_canvas, y1s))
    y2s = max(0, min(H_canvas, y2s))
    if x2s <= x1s or y2s <= y1s:
        return 0, 0, 1, 1
    return x1s, y1s, x2s, y2s


def _apply_seg_to_fg_mask(
    fg_mask: np.ndarray,
    px: int,
    py: int,
    nw: int,
    nh: int,
    mask_resized: Image.Image | None,
    W: int,
    H: int,
) -> bool:
    """Project a resized segmentation mask (L-mode PIL) onto the
    canvas-sized fg_mask numpy array.
    Returns True if any pixels were applied, False otherwise.
    """
    if mask_resized is None or nw <= 0 or nh <= 0:
        return False
    cx1, cy1 = max(0, px), max(0, py)
    cx2, cy2 = min(W, px + nw), min(H, py + nh)
    if cx2 <= cx1 or cy2 <= cy1:
        return False
    mask_arr = (np.array(mask_resized) > MASK_THRESHOLD).astype(np.uint8)
    # mx1/my1 are the offsets into mask_resized that correspond to the visible
    # canvas region [cx1:cx2, cy1:cy2] — i.e. how much of the pasted image
    # is cropped by the canvas boundary.
    mx1, my1 = cx1 - px, cy1 - py
    fg_mask[cy1:cy2, cx1:cx2] |= mask_arr[my1 : my1 + (cy2 - cy1), mx1 : mx1 + (cx2 - cx1)]
    return True


def _paste_keep_aspect_with_mask(
    canvas: Image.Image,
    ref_img: Image.Image,
    ref_mask: Image.Image | None,
    box: tuple[int, int, int, int],
) -> tuple[int, int, int, int, Image.Image | None]:
    """Paste *ref_img* into *box* on *canvas* preserving aspect ratio.

    Returns ``(px, py, nw, nh, mask_resized)`` for downstream fg_mask
    projection.
    *mask_resized* is ``None`` when *ref_mask* is ``None``; callers must handle
    that case (``_apply_seg_to_fg_mask`` already does so).
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return 0, 0, 0, 0, None
    img = ref_img.convert("RGB")
    iw, ih = img.size
    if iw == 0 or ih == 0:
        return 0, 0, 0, 0, None
    scale = min(bw / iw, bh / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    img_resized = img.resize((nw, nh), Image.LANCZOS)
    px = x1 + (bw - nw) // 2
    py = y1 + (bh - nh) // 2
    mask_resized = None
    if ref_mask is not None:
        if ref_mask.mode != "L":
            ref_mask = ref_mask.convert("L")
        mask_resized = ref_mask.resize((nw, nh), Image.NEAREST)
        canvas.paste(img_resized, (px, py), mask_resized)
    else:
        canvas.paste(img_resized, (px, py))
    return px, py, nw, nh, mask_resized


def paste_face_aligned_with_mask(
    canvas: Image.Image,
    ref_img: Image.Image,
    ref_mask: Image.Image | None,
    src_face_box: tuple[int, int, int, int],
    tgt_box: tuple[int, int, int, int],
    face_scale: float = 1.10,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int, int, int, Image.Image | None]:
    """Align the source face box center to the target box center, scale
    the entire reference image so that the face roughly fits inside
    the target box (times face_scale), and paste.
    Returns (px, py, nw, nh, mask_resized) where px,py is paste origin
    and nw,nh the resized reference image size; mask_resized is the
    resized segmentation mask if provided.
    """
    if ref_img.mode not in ("RGB", "RGBA"):
        ref_img = ref_img.convert("RGB")
    tx1, ty1, tx2, ty2 = tgt_box
    tcx, tcy, tw, th = box_center_size((tx1, ty1, tx2, ty2))
    sx1, sy1, sx2, sy2 = src_face_box
    scx, scy, sw, sh = box_center_size((sx1, sy1, sx2, sy2))
    s = max(1e-3, min(tw / float(sw), th / float(sh)) * float(face_scale))
    iw, ih = ref_img.size
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    img_resized = ref_img.resize((nw, nh), Image.LANCZOS)
    t_scx, t_scy = scx * s, scy * s
    px = int(round(tcx - t_scx))
    py = int(round(tcy - t_scy))
    # Optional mask handling
    mask_resized = None
    if ref_mask is not None:
        if ref_mask.mode != "L":
            ref_mask = ref_mask.convert("L")
        mask_resized = ref_mask.resize((nw, nh), Image.NEAREST)
    # Optional background fill (only where we draw)
    if img_resized.mode != "RGBA" and fill is not None:
        W, H = canvas.size
        bx1, by1 = max(0, px), max(0, py)
        bx2, by2 = min(W, px + nw), min(H, py + nh)
        if bx2 > bx1 and by2 > by1:
            bg = Image.new("RGB", (bx2 - bx1, by2 - by1), color=fill)
            canvas.paste(bg, (bx1, by1))
    if mask_resized is not None:
        canvas.paste(img_resized, (px, py), mask_resized)
    else:
        canvas.paste(img_resized, (px, py))
    return px, py, nw, nh, mask_resized


def build_single_canvas_1024(
    ref_path: str,
    box_ref_space: list[float],  # target box in probe coords
    bbox_ref_wh: tuple[int, int],  # (probe W, probe H)
    ref_face_box: tuple[int, int, int, int] | None,
    face_scale: float = 1.10,
    target_pad: float = 1.00,
    use_segmentation: bool = True,
    seg_parser: FacerParser | None = None,
    canvas_color: tuple[int, int, int] = (255, 255, 255),
) -> tuple[Image.Image, np.ndarray]:
    """Build a single 1024x1024 canvas for one subject and its foreground mask.
    - Pastes the reference (aligned if ref_face_box is present;
      otherwise aspect-preserving).
    - Foreground mask is derived from the **padded detected target
      box** (FGMASK_PAD).
    """
    W = H = 1024
    canvas = Image.new("RGB", (W, H), color=canvas_color)
    fg_mask = np.zeros((H, W), dtype=np.uint8)

    try:
        ref_img = Image.open(ref_path).convert("RGB")
    except Exception:
        logger.exception(
            "Failed to open reference image %r; returning blank canvas",
            ref_path,
        )
        return canvas, fg_mask

    ref_seg = None
    if use_segmentation and seg_parser is not None:
        ref_seg = seg_parser.make_mask(ref_img)

    # Scale detected probe-space box to 1024 canvas and pad for alignment paste
    tgt_scaled = _scale_box_to_canvas(box_ref_space, bbox_ref_wh, (W, H))
    tx1, ty1, tx2, ty2 = tgt_scaled
    tgt_padded = expand_box_ratio((tx1, ty1, tx2, ty2), target_pad, (W, H))

    if ref_face_box is not None:
        paste_coords = paste_face_aligned_with_mask(
            canvas=canvas,
            ref_img=ref_img,
            ref_mask=ref_seg,
            src_face_box=ref_face_box,
            tgt_box=tgt_padded,
            face_scale=face_scale,
            fill=canvas_color,
        )
    else:
        paste_coords = _paste_keep_aspect_with_mask(canvas, ref_img, ref_seg, tgt_padded)

    # Foreground mask: use projected segmentation (includes neck/hair)
    # when available, otherwise fall back to padded target box.
    px, py, nw, nh, mask_resized = paste_coords
    if not _apply_seg_to_fg_mask(fg_mask, px, py, nw, nh, mask_resized, W, H):
        fg_box = expand_box_ratio(tgt_padded, FGMASK_PAD, (W, H))
        fg_box = clamp_box_to_canvas(fg_box, (W, H))
        x1m, y1m, x2m, y2m = fg_box
        fg_mask[y1m:y2m, x1m:x2m] = 1

    return canvas, fg_mask


# =============================================================================
# Seg-crop helper + unified canvas builder
# =============================================================================


def _crop_to_seg_bounds(
    img: Image.Image,
    mask: Image.Image,
    pad_px: int = 4,
) -> tuple[Image.Image, Image.Image, int, int]:
    """Crop *img* and *mask* to the tight bounding box of non-zero mask pixels
    plus *pad_px* border pixels.

    Returns (cropped_img, cropped_mask, origin_x, origin_y).
    If the mask is entirely empty the originals are returned with
    origin (0, 0).
    """
    mask_arr = np.array(mask)
    rows = np.any(mask_arr > MASK_THRESHOLD, axis=1)
    cols = np.any(mask_arr > MASK_THRESHOLD, axis=0)
    if not rows.any() or not cols.any():
        return img, mask, 0, 0
    row_idx = np.where(rows)[0]
    col_idx = np.where(cols)[0]
    y1, y2 = int(row_idx[0]), int(row_idx[-1]) + 1
    x1, x2 = int(col_idx[0]), int(col_idx[-1]) + 1
    H_m, W_m = mask_arr.shape
    y1 = max(0, y1 - pad_px)
    y2 = min(H_m, y2 + pad_px)
    x1 = max(0, x1 - pad_px)
    x2 = min(W_m, x2 + pad_px)
    return img.crop((x1, y1, x2, y2)), mask.crop((x1, y1, x2, y2)), x1, y1


def build_canvases(
    refs: list[Image.Image],
    boxes: list[list[float]],
    probe_wh: tuple[int, int],
    ref_face_boxes: list[tuple[int, int, int, int] | None],
    seg_parser: Optional["FacerParser"] = None,
    mask_mode: str = "loose",
    face_scale: float = 1.10,
    target_pad: float = 1.0,
    align: bool = True,
    canvas_wh: tuple[int, int] = (1024, 1024),
    canvas_color: tuple[int, int, int] = (255, 255, 255),
) -> tuple[list[Image.Image], list[np.ndarray], Image.Image]:
    """Build per-subject canvases and foreground masks for the Artist pipeline.

    For each reference image:
      - Runs segmentation (if seg_parser is provided) and crops to the tight
        segmentation bounds to eliminate whitespace.
      - Scales the detected probe-space face box to the canvas and expands it
        by *target_pad*.
      - Pastes the (cropped) reference via face-aligned paste when a face box
        is available, otherwise via aspect-preserving paste.
      - Builds a per-subject fg_mask from the projected segmentation, falling
        back to the padded target box when segmentation is unavailable.

    Also accumulates all subjects onto a single *combined_canvas*.

    Returns (canvases, fg_masks, combined_canvas).
    """
    W, H = canvas_wh
    combined_canvas = Image.new("RGB", (W, H), color=canvas_color)
    canvases: list[Image.Image] = []
    fg_masks: list[np.ndarray] = []

    # Local copy so we don't mutate the caller's list when adjusting
    # crop origins.
    face_boxes = list(ref_face_boxes)

    for i, ref_img in enumerate(refs):
        canvas = Image.new("RGB", (W, H), color=canvas_color)
        fg_mask = np.zeros((H, W), dtype=np.uint8)

        ref_img = ref_img.convert("RGB")
        ref_seg = None
        if seg_parser is not None:
            try:
                ref_seg = seg_parser.make_mask(ref_img, mode=mask_mode)
            except Exception:
                logger.exception(
                    "Segmentation failed for image %d; skipping mask",
                    i,
                )

        # Crop to tight segmentation bounds to eliminate whitespace.
        if ref_seg is not None:
            ref_img, ref_seg, crop_ox, crop_oy = _crop_to_seg_bounds(ref_img, ref_seg)
            if face_boxes[i] is not None:
                fx1, fy1, fx2, fy2 = face_boxes[i]
                face_boxes[i] = (
                    fx1 - crop_ox,
                    fy1 - crop_oy,
                    fx2 - crop_ox,
                    fy2 - crop_oy,
                )

        tgt_scaled = _scale_box_to_canvas(boxes[i], probe_wh, canvas_wh)
        tx1, ty1, tx2, ty2 = tgt_scaled
        tgt_padded = expand_box_ratio((tx1, ty1, tx2, ty2), target_pad, (W, H))

        if align and face_boxes[i] is not None:
            try:
                paste_coords = paste_face_aligned_with_mask(
                    canvas=canvas,
                    ref_img=ref_img,
                    ref_mask=ref_seg,
                    src_face_box=face_boxes[i],
                    tgt_box=tgt_padded,
                    face_scale=face_scale,
                    fill=canvas_color,
                )
                paste_face_aligned_with_mask(
                    canvas=combined_canvas,
                    ref_img=ref_img,
                    ref_mask=ref_seg,
                    src_face_box=face_boxes[i],
                    tgt_box=tgt_padded,
                    face_scale=face_scale,
                    fill=canvas_color,
                )
            except Exception:
                logger.exception(
                    "Face-aligned paste failed for image %d;" " falling back",
                    i,
                )
                paste_coords = _paste_keep_aspect_with_mask(canvas, ref_img, ref_seg, tgt_padded)
                _paste_keep_aspect_with_mask(combined_canvas, ref_img, ref_seg, tgt_padded)
        else:
            paste_coords = _paste_keep_aspect_with_mask(canvas, ref_img, ref_seg, tgt_padded)
            _paste_keep_aspect_with_mask(combined_canvas, ref_img, ref_seg, tgt_padded)

        if paste_coords is not None:
            px, py, nw, nh, mask_resized = paste_coords
            if not _apply_seg_to_fg_mask(fg_mask, px, py, nw, nh, mask_resized, W, H):
                # Seg unavailable — expand tgt_padded by FGMASK_PAD
                # for the fg mask.
                fg_box = clamp_box_to_canvas(
                    expand_box_ratio(tgt_padded, FGMASK_PAD, (W, H)),
                    (W, H),
                )
                x1m, y1m, x2m, y2m = fg_box
                fg_mask[y1m:y2m, x1m:x2m] = 1
        else:
            fg_box = clamp_box_to_canvas(
                expand_box_ratio(tgt_padded, FGMASK_PAD, (W, H)),
                (W, H),
            )
            x1m, y1m, x2m, y2m = fg_box
            fg_mask[y1m:y2m, x1m:x2m] = 1

        canvases.append(canvas)
        fg_masks.append(fg_mask)

    return canvases, fg_masks, combined_canvas
