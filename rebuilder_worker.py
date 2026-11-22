#!/usr/bin/env python3
"""
rebuilder_worker.py - CPU/GPU inpainting & ASS vector mesh generator.

Paired with ass-inpaint-mesh.lua for Aegisub Automation 4.
Parses ASS \\clip(m ...) polygons (including lines and cubic Beziers),
inpaints masked regions on a padded bounding box using OpenCV (Telea / NS)
or optional CPU/GPU LaMa AI, and merges the resulting grid into optimized
\\p1 ASS vector rectangles grouped by color.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

WORKER_VERSION = "1.3.0"
MAX_RECTS_PER_EVENT = 1500
VALID_ENGINES = ("telea", "ns", "lama")

ENGINE_ALIASES: Dict[str, str] = {
    "telea": "telea",
    "opencv fast (telea)": "telea",
    "opencv_telea": "telea",
    "ns": "ns",
    "navier-stokes": "ns",
    "opencv smooth (navier-stokes)": "ns",
    "opencv_ns": "ns",
    "lama": "lama",
    "lama ai (cpu - high quality)": "lama",
    "lama ai (gpu - high quality)": "lama",
    "lama_gpu": "lama",
    "simple-lama": "lama",
}

DEVICE_ALIASES: Dict[str, str] = {
    "cpu": "cpu",
    "gpu": "gpu",
    "gpu (cuda / mps)": "gpu",
    "cuda": "cuda",
    "mps": "mps",
    "directml": "gpu",
    "dml": "gpu",
    "auto": "auto",
    "auto (prefer gpu)": "auto",
}

_LAMA_CACHE: Dict[str, Dict[str, Any]] = {}


@dataclass
class RectPatch:
    """Represents a single merged rectangle in script coordinate space."""

    x1: int
    y1: int
    x2: int
    y2: int
    bgr: Tuple[float, float, float]
    area: int
    rect_id: int = -1
    r0: int = 0
    c0: int = 0
    r1: int = 0
    c1: int = 0


@dataclass
class FrameSpec:
    """Represents a single frame target to process."""

    frame_index: int
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    image_path: Optional[str] = None


def normalize_engine(engine_str: str) -> str:
    """Map GUI dropdown labels or CLI aliases to canonical engine identifiers."""
    key = (engine_str or "telea").strip().lower()
    if key in ENGINE_ALIASES:
        return ENGINE_ALIASES[key]
    if "lama" in key:
        return "lama"
    if "navier" in key or key == "ns":
        return "ns"
    return "telea"


def normalize_device(device_str: Optional[str], engine_raw: Optional[str] = None) -> str:
    """Resolve canonical compute device preference ('cpu', 'gpu', 'cuda', 'mps', 'auto')."""
    key = (device_str or "").strip().lower()
    if key in DEVICE_ALIASES:
        pref = DEVICE_ALIASES[key]
    elif "gpu" in key or "cuda" in key:
        pref = "gpu"
    elif "mps" in key:
        pref = "mps"
    elif "auto" in key:
        pref = "auto"
    else:
        pref = "cpu"

    eng_key = (engine_raw or "").strip().lower()
    if "gpu" in eng_key and pref == "cpu":
        pref = "gpu"
    return pref


def resolve_compute_device(device_pref: str) -> Tuple[Optional[Any], str, Optional[str]]:
    """
    Resolve requested compute device ('cpu', 'gpu', 'cuda', 'mps', 'auto')
    to an active PyTorch device, device label, and optional fallback warning.
    """
    pref = normalize_device(device_pref)
    try:
        import torch
    except ImportError as exc:
        if pref in ("gpu", "cuda", "mps"):
            return None, "cpu", f"GPU requested ({pref}), but PyTorch is not installed ({exc}); fell back to CPU."
        return None, "cpu", None

    if pref == "cpu":
        return torch.device("cpu"), "cpu", None

    if pref in ("gpu", "cuda", "auto") and torch.cuda.is_available():
        return torch.device("cuda"), "cuda", None

    if pref in ("gpu", "mps", "auto") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps", None

    if pref in ("gpu", "auto"):
        try:
            import torch_directml  # type: ignore

            return torch_directml.device(), "directml", None
        except Exception:
            pass

    if pref in ("gpu", "cuda", "mps"):
        return (
            torch.device("cpu"),
            "cpu",
            "GPU requested, but PyTorch did not detect an active CUDA/MPS GPU; fell back to CPU.",
        )

    return torch.device("cpu"), "cpu", None


def format_ass_number(val: float) -> str:
    """Format float cleanly for ASS tags (e.g. 0.6 -> '0.6', 1.0 -> '1')."""
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return f"{val:.3f}".rstrip("0").rstrip(".")


def bgr_to_ass_color(bgr: Tuple[int, int, int]) -> str:
    """Convert integer (B, G, R) tuple to ASS &HBBGGRR& hex color string."""
    b = max(0, min(255, int(bgr[0])))
    g = max(0, min(255, int(bgr[1])))
    r = max(0, min(255, int(bgr[2])))
    return f"&H{b:02X}{g:02X}{r:02X}&"


def ms_to_ass_time(ms: int) -> str:
    """Format milliseconds as ASS timestamp H:MM:SS.cs."""
    ms = max(0, int(ms))
    cs_total = round(ms / 10.0)
    cs = cs_total % 100
    s_total = cs_total // 100
    s = s_total % 60
    m_total = s_total // 60
    m = m_total % 60
    h = m_total // 60
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def extract_raw_clip_inner(clip_str: str) -> str:
    """Extract the inner expression from \\clip(...) or \\iclip(...), or return trimmed string."""
    s = (clip_str or "").strip()
    m = re.match(r"^\\?i?clip\((.*)\)$", s, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return s


def _eval_cubic_bezier(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
) -> List[Tuple[float, float]]:
    """Sample a cubic Bezier curve adaptively based on its control-polygon length."""
    chord = (
        math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        + math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        + math.hypot(p3[0] - p2[0], p3[1] - p2[1])
    )
    steps = max(12, min(64, int(math.ceil(chord / 2.0))))
    pts: List[Tuple[float, float]] = []
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1.0 - t
        mt2 = mt * mt
        t2 = t * t
        w0 = mt2 * mt
        w1 = 3.0 * mt2 * t
        w2 = 3.0 * mt * t2
        w3 = t2 * t
        x = w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0]
        y = w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1]
        pts.append((x, y))
    return pts


def _eval_uniform_bspline(ctrl_pts: Sequence[Tuple[float, float]], closed: bool = False) -> List[Tuple[float, float]]:
    """Sample a sequence of uniform cubic B-spline control points (ASS 's'/'p'/'c' commands)."""
    if len(ctrl_pts) < 4:
        return list(ctrl_pts[1:])

    pts = list(ctrl_pts)
    if closed:
        pts = pts + pts[:3]
    else:
        # Repeat endpoints so the spline passes through the first and last points.
        pts = [pts[0], pts[0]] + pts + [pts[-1], pts[-1]]

    out: List[Tuple[float, float]] = []
    for i in range(len(pts) - 3):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        seg_len = (
            math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            + math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            + math.hypot(p3[0] - p2[0], p3[1] - p2[1])
        )
        steps = max(8, min(32, int(math.ceil(seg_len / 2.0))))
        for s in range(1, steps + 1):
            t = s / steps
            t2 = t * t
            t3 = t2 * t
            b0 = (-t3 + 3 * t2 - 3 * t + 1) / 6.0
            b1 = (3 * t3 - 6 * t2 + 4) / 6.0
            b2 = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6.0
            b3 = t3 / 6.0
            x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
            y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
            out.append((x, y))
    return out


def parse_ass_clip(clip_str: str) -> Tuple[List[List[Tuple[float, float]]], str]:
    """
    Parse ASS \\clip(...) vector drawing or rectangular coordinates into polygons.

    Supports:
    - Rectangular clip: (x1, y1, x2, y2)
    - Optional scale prefix: (scale, m x y ...)
    - Vector commands: m, n, l, b, s, p, c

    Returns:
        (contours, normalized_raw_clip_inner)
    """
    raw_inner = extract_raw_clip_inner(clip_str)
    if not raw_inner:
        raise ValueError("Empty \\clip string provided.")

    # Check for optional scale prefix: "<int>, m ..."
    scale_factor = 1.0
    drawing_str = raw_inner
    scale_match = re.match(r"^\s*(\d+)\s*,\s*(.+)$", raw_inner, flags=re.DOTALL)
    if scale_match:
        maybe_scale = int(scale_match.group(1))
        rest = scale_match.group(2).strip()
        if re.search(r"[mnlbspc]", rest, flags=re.IGNORECASE):
            if maybe_scale >= 1:
                scale_factor = float(1 << (maybe_scale - 1))
            drawing_str = rest

    # Check for rectangular clip: 4 numbers with no command letters
    if not re.search(r"[a-zA-Z]", drawing_str):
        num_tokens = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", drawing_str)
        if len(num_tokens) == 4:
            x1, y1, x2, y2 = (float(v) for v in num_tokens)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            contour = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            return [contour], raw_inner
        raise ValueError(f"Could not parse rectangular \\clip coordinates: {raw_inner!r}")

    # Tokenize ASS drawing commands and numbers
    token_pattern = re.compile(
        r"([mnlbspcMNLBSPC])|([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    )
    tokens: List[Tuple[str, str]] = []
    for match in token_pattern.finditer(drawing_str):
        if match.group(1):
            tokens.append(("CMD", match.group(1).lower()))
        else:
            tokens.append(("NUM", match.group(2)))

    contours: List[List[Tuple[float, float]]] = []
    current_contour: List[Tuple[float, float]] = []
    spline_points: List[Tuple[float, float]] = []
    in_spline = False
    cmd: Optional[str] = None
    idx = 0

    def flush_spline(closed: bool = False) -> None:
        nonlocal in_spline, spline_points
        if in_spline and len(spline_points) >= 2:
            current_contour.extend(_eval_uniform_bspline(spline_points, closed=closed))
        in_spline = False
        spline_points = []

    def flush_contour() -> None:
        nonlocal current_contour
        flush_spline(closed=False)
        if len(current_contour) >= 3:
            contours.append(current_contour)
        current_contour = []

    def read_nums(count: int) -> Optional[List[float]]:
        nonlocal idx
        vals: List[float] = []
        scan = idx
        while scan < len(tokens) and tokens[scan][0] == "NUM" and len(vals) < count:
            vals.append(float(tokens[scan][1]) / scale_factor)
            scan += 1
        if len(vals) < count:
            return None
        idx = scan
        return vals

    while idx < len(tokens):
        ttype, tval = tokens[idx]
        if ttype == "CMD":
            if tval == "c":
                flush_spline(closed=True)
                idx += 1
                continue
            if tval != "p" and in_spline:
                flush_spline(closed=False)
            cmd = tval
            idx += 1
            continue

        if cmd is None:
            # Implicit move if coordinates appear before any command
            cmd = "m"

        if cmd in ("m", "n"):
            coords = read_nums(2)
            if coords is None:
                break
            flush_contour()
            current_contour.append((coords[0], coords[1]))
            # Subsequent pairs without a new command act as implicit 'l' in libass
            cmd = "l"
        elif cmd == "l":
            coords = read_nums(2)
            if coords is None:
                break
            current_contour.append((coords[0], coords[1]))
        elif cmd == "b":
            coords = read_nums(6)
            if coords is None:
                break
            p0 = current_contour[-1] if current_contour else (0.0, 0.0)
            p1 = (coords[0], coords[1])
            p2 = (coords[2], coords[3])
            p3 = (coords[4], coords[5])
            current_contour.extend(_eval_cubic_bezier(p0, p1, p2, p3))
        elif cmd in ("s", "p"):
            coords = read_nums(2)
            if coords is None:
                break
            if not in_spline:
                in_spline = True
                start_pt = current_contour[-1] if current_contour else (coords[0], coords[1])
                spline_points = [start_pt]
            spline_points.append((coords[0], coords[1]))
        else:
            idx += 1

    flush_contour()

    if not contours:
        raise ValueError(f"No valid polygon contours found in \\clip: {raw_inner!r}")

    return contours, raw_inner


def rasterize_clip_mask(
    contours: Sequence[Sequence[Tuple[float, float]]],
    video_width: int,
    video_height: int,
    script_width: Optional[int] = None,
    script_height: Optional[int] = None,
) -> np.ndarray:
    """Rasterize ASS contours into a uint8 binary mask (0 or 255) at video resolution."""
    sx = float(video_width) / float(script_width) if script_width and script_width > 0 else 1.0
    sy = float(video_height) / float(script_height) if script_height and script_height > 0 else 1.0

    mask = np.zeros((video_height, video_width), dtype=np.uint8)
    poly_arrays: List[np.ndarray] = []

    for contour in contours:
        if len(contour) < 3:
            continue
        arr = np.asarray(contour, dtype=np.float64)
        arr[:, 0] *= sx
        arr[:, 1] *= sy
        pts_int = np.round(arr).astype(np.int32)
        poly_arrays.append(pts_int)

    if poly_arrays:
        cv2.fillPoly(mask, poly_arrays, 255)

    return mask


def _get_lama_model(device_pref: str = "cpu") -> Tuple[Optional[Any], str, Optional[str]]:
    """Lazy-load SimpleLama on the requested compute device (CPU or GPU), caching per device."""
    torch_dev, dev_label, dev_warn = resolve_compute_device(device_pref)
    cache_entry = _LAMA_CACHE.get(dev_label)
    if cache_entry is not None:
        return cache_entry["model"], dev_label, cache_entry["error"] or dev_warn

    entry: Dict[str, Any] = {"model": None, "error": None}
    _LAMA_CACHE[dev_label] = entry
    try:
        from simple_lama_inpainting import SimpleLama

        sig = inspect.signature(SimpleLama.__init__)
        if "device" in sig.parameters and torch_dev is not None:
            model = SimpleLama(device=torch_dev)
        else:
            model = SimpleLama()
        entry["model"] = model
        return model, dev_label, dev_warn
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        entry["error"] = err_msg
        combined = f"{dev_warn} {err_msg}".strip() if dev_warn else err_msg
        return None, dev_label, combined


def harmonize_inpainted_boundary(
    orig_crop_bgr: np.ndarray,
    inpainted_crop_bgr: np.ndarray,
    crop_mask: np.ndarray,
) -> np.ndarray:
    """
    Correct neural inpainting (e.g. SimpleLama) brightness/tint shifts on flat
    and white backgrounds by aligning both global outer-ring DC bias and local
    perimeter step jumps against untouched surrounding pixels, then compositing
    cleanly onto original unmasked pixels.
    """
    m_bin = (crop_mask > 0).astype(np.uint8)
    if not np.any(m_bin):
        return orig_crop_bgr.copy()

    orig_f = orig_crop_bgr.astype(np.float32)
    inp_f = inpainted_crop_bgr.astype(np.float32)

    k3 = np.ones((3, 3), dtype=np.uint8)
    k7 = np.ones((7, 7), dtype=np.uint8)
    k11 = np.ones((11, 11), dtype=np.uint8)

    outer_ring = (cv2.dilate(m_bin, k11) > 0) & (cv2.dilate(m_bin, k3) == 0)

    # 1. Global outer-ring correction (fixes full-crop decoder DC/tint bias)
    if np.count_nonzero(outer_ring) >= 16:
        outer_delta = np.median(orig_f[outer_ring] - inp_f[outer_ring], axis=0)
        outer_delta = np.clip(outer_delta, -20.0, 20.0)
        inp_f = np.clip(inp_f + outer_delta[None, None, :], 0.0, 255.0)

    # 2. Local perimeter step correction (fixes interior-only gray shift on white/solid regions)
    eroded = cv2.erode(m_bin, k7)
    inner_ring = (m_bin > 0) & (eroded == 0)
    if np.count_nonzero(inner_ring) < 12:
        inner_ring = m_bin > 0

    if np.any(outer_ring) and np.any(inner_ring):
        outer_w = outer_ring.astype(np.float32)
        inner_w = inner_ring.astype(np.float32)
        w_out = cv2.boxFilter(
            outer_w, ddepth=cv2.CV_32F, ksize=(15, 15), normalize=False, borderType=cv2.BORDER_CONSTANT
        )
        w_in = cv2.boxFilter(
            inner_w, ddepth=cv2.CV_32F, ksize=(15, 15), normalize=False, borderType=cv2.BORDER_CONSTANT
        )
        valid = inner_ring & (w_out >= 8.0) & (w_in >= 4.0)
        if np.count_nonzero(valid) >= 12:
            sum_orig = cv2.boxFilter(
                orig_f * outer_w[..., None],
                ddepth=cv2.CV_32F,
                ksize=(15, 15),
                normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )
            sum_inp = cv2.boxFilter(
                inp_f * inner_w[..., None],
                ddepth=cv2.CV_32F,
                ksize=(15, 15),
                normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )
            local_orig = sum_orig[valid] / w_out[valid, None]
            local_inp = sum_inp[valid] / w_in[valid, None]
            seam_delta = np.median(local_orig - local_inp, axis=0)
            seam_delta = np.clip(seam_delta, -20.0, 20.0)
            inp_f = np.clip(inp_f + seam_delta[None, None, :], 0.0, 255.0)

    # 3. Preserve exact untouched exterior pixels and blend 3px inner edge softly
    soft_alpha = cv2.GaussianBlur(m_bin.astype(np.float32), (5, 5), 0)[..., None]
    soft_alpha = np.where(m_bin[..., None] > 0, np.maximum(soft_alpha, 0.85), 0.0)
    blended = np.clip(inp_f * soft_alpha + orig_f * (1.0 - soft_alpha), 0.0, 255.0).astype(np.uint8)
    return blended


def inpaint_region(
    frame_bgr: np.ndarray,
    orig_mask: np.ndarray,
    engine: str = "telea",
    dilate_px: int = 3,
    grid_size: int = 4,
    device: str = "cpu",
    harmonize: Optional[bool] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], str, str, Optional[str]]:
    """
    Crop to a padded bounding box around orig_mask, dilate the mask, and run inpainting.

    Returns:
        (inpainted_crop_bgr, (crop_x0, crop_y0, crop_x1, crop_y1), used_engine, used_device, warning_msg)
    """
    h, w = frame_bgr.shape[:2]
    ys, xs = np.nonzero(orig_mask)
    if len(xs) == 0:
        raise ValueError("Rasterized \\clip mask is empty (outside video bounds or zero area).")

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    dilate_px = max(0, int(dilate_px))
    pad = max(32, int(grid_size) * 4, dilate_px * 4)

    crop_x0 = max(0, x0 - dilate_px - pad)
    crop_y0 = max(0, y0 - dilate_px - pad)
    crop_x1 = min(w, x1 + dilate_px + pad)
    crop_y1 = min(h, y1 + dilate_px + pad)

    crop_bgr = frame_bgr[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    crop_mask = orig_mask[crop_y0:crop_y1, crop_x0:crop_x1].copy()

    if dilate_px > 0:
        ksize = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        crop_mask = cv2.dilate(crop_mask, kernel, iterations=1)

    canonical_engine = normalize_engine(engine)
    device_pref = normalize_device(device, engine)
    should_harmonize = (canonical_engine == "lama") if harmonize is None else bool(harmonize)
    warning: Optional[str] = None
    used_engine = canonical_engine
    used_device = "cpu"
    inpaint_radius = max(3, dilate_px + 2)

    if canonical_engine == "lama":
        lama_model, dev_label, load_msg = _get_lama_model(device_pref)
        if lama_model is not None:
            try:
                crop_h, crop_w = crop_bgr.shape[:2]
                pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                pil_mask = Image.fromarray(crop_mask, mode="L")
                res_pil = lama_model(pil_img, pil_mask)
                if res_pil.size != (crop_w, crop_h):
                    res_pil = res_pil.crop((0, 0, crop_w, crop_h))
                inpainted_crop = cv2.cvtColor(np.asarray(res_pil, dtype=np.uint8), cv2.COLOR_RGB2BGR)
                if should_harmonize:
                    inpainted_crop = harmonize_inpainted_boundary(crop_bgr, inpainted_crop, crop_mask)
                return inpainted_crop, (crop_x0, crop_y0, crop_x1, crop_y1), "lama", dev_label, load_msg
            except Exception as exc:
                warning = f"LaMa inference failed ({type(exc).__name__}: {exc}); fell back to OpenCV Navier-Stokes."
        else:
            warning = f"LaMa unavailable ({load_msg}); fell back to OpenCV Navier-Stokes."
        used_engine = "ns"

    cv_flag = cv2.INPAINT_NS if used_engine == "ns" else cv2.INPAINT_TELEA
    inpainted_crop = cv2.inpaint(crop_bgr, crop_mask, inpaintRadius=inpaint_radius, flags=cv_flag)
    if harmonize is True:
        inpainted_crop = harmonize_inpainted_boundary(crop_bgr, inpainted_crop, crop_mask)
    return inpainted_crop, (crop_x0, crop_y0, crop_x1, crop_y1), used_engine, used_device, warning


def _quantize_rect_colors(rects: List[RectPatch], tolerance: int) -> Dict[Tuple[int, int, int], List[RectPatch]]:
    """
    Cluster near-identical rectangle colors within a small Euclidean tolerance so
    non-adjacent tiles share exact \\1c&HBBGGRR& tags, then group by final BGR tuple.
    """
    if not rects:
        return {}

    if tolerance <= 0:
        groups: Dict[Tuple[int, int, int], List[RectPatch]] = defaultdict(list)
        for rect in rects:
            key = (
                max(0, min(255, int(round(rect.bgr[0])))),
                max(0, min(255, int(round(rect.bgr[1])))),
                max(0, min(255, int(round(rect.bgr[2])))),
            )
            groups[key].append(rect)
        return groups

    snap_thresh = max(1.0, float(tolerance) * 0.35)
    # Process largest-area rectangles first so dominant colors anchor clusters
    indexed = sorted(enumerate(rects), key=lambda item: item[1].area, reverse=True)

    clusters: List[Dict[str, Any]] = []
    assignments: Dict[int, int] = {}

    for orig_idx, rect in indexed:
        rb, rg, rr = rect.bgr
        best_cluster_idx = -1
        best_dist = snap_thresh + 1.0

        for c_idx, cluster in enumerate(clusters):
            cb, cg, cr = cluster["mean_bgr"]
            dist = math.sqrt((rb - cb) ** 2 + (rg - cg) ** 2 + (rr - cr) ** 2)
            if dist <= snap_thresh and dist < best_dist:
                best_dist = dist
                best_cluster_idx = c_idx

        if best_cluster_idx < 0:
            clusters.append(
                {
                    "sum_b": rb * rect.area,
                    "sum_g": rg * rect.area,
                    "sum_r": rr * rect.area,
                    "weight": float(rect.area),
                    "mean_bgr": (rb, rg, rr),
                }
            )
            best_cluster_idx = len(clusters) - 1
        else:
            cl = clusters[best_cluster_idx]
            cl["sum_b"] += rb * rect.area
            cl["sum_g"] += rg * rect.area
            cl["sum_r"] += rr * rect.area
            cl["weight"] += float(rect.area)
            w_tot = cl["weight"]
            cl["mean_bgr"] = (cl["sum_b"] / w_tot, cl["sum_g"] / w_tot, cl["sum_r"] / w_tot)

        assignments[orig_idx] = best_cluster_idx

    groups = defaultdict(list)
    for orig_idx, rect in enumerate(rects):
        mean_b, mean_g, mean_r = clusters[assignments[orig_idx]]["mean_bgr"]
        key = (
            max(0, min(255, int(round(mean_b)))),
            max(0, min(255, int(round(mean_g)))),
            max(0, min(255, int(round(mean_r)))),
        )
        groups[key].append(rect)

    return groups


def build_vector_mesh_with_grid(
    inpainted_crop: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    orig_mask: np.ndarray,
    grid_size: int = 4,
    tolerance: int = 6,
    script_width: Optional[int] = None,
    script_height: Optional[int] = None,
) -> Tuple[Dict[Tuple[int, int, int], List[RectPatch]], Dict[str, int], Optional[np.ndarray]]:
    """
    Sample the inpainted region on a grid, apply greedy 2D rectangle merging,
    and return (color_groups_map, stats_dict, rect_id_grid).
    """
    vid_h, vid_w = orig_mask.shape[:2]
    crop_x0, crop_y0, _, _ = crop_box
    g = max(1, min(16, int(grid_size)))
    tol = max(0, min(30, int(tolerance)))

    # Expand orig_mask by 1px so boundary tiles covering anti-aliased clip edges are included
    kernel_1px = np.ones((3, 3), dtype=np.uint8)
    coverage_mask = cv2.dilate(orig_mask, kernel_1px, iterations=1)

    ys, xs = np.nonzero(coverage_mask)
    if len(xs) == 0:
        return {}, {"active_cells": 0, "merged_rects": 0, "color_groups": 0}, None

    bx0 = int(xs.min())
    bx1 = int(xs.max()) + 1
    by0 = int(ys.min())
    by1 = int(ys.max()) + 1

    rows = int(math.ceil((by1 - by0) / float(g)))
    cols = int(math.ceil((bx1 - bx0) / float(g)))

    active = np.zeros((rows, cols), dtype=bool)
    cell_colors = np.zeros((rows, cols, 3), dtype=np.float32)
    cell_weights = np.zeros((rows, cols), dtype=np.float32)
    rect_grid = np.full((rows, cols), -1, dtype=np.int32)

    active_cells = 0
    for r in range(rows):
        y_start = by0 + r * g
        y_end = min(by1, y_start + g)
        cy_start = y_start - crop_y0
        cy_end = y_end - crop_y0
        for c in range(cols):
            x_start = bx0 + c * g
            x_end = min(bx1, x_start + g)
            if not np.any(coverage_mask[y_start:y_end, x_start:x_end]):
                continue
            cx_start = x_start - crop_x0
            cx_end = x_end - crop_x0
            patch = inpainted_crop[cy_start:cy_end, cx_start:cx_end]
            if patch.size == 0:
                continue
            mean_bgr = patch.mean(axis=(0, 1))
            active[r, c] = True
            cell_colors[r, c] = mean_bgr
            cell_weights[r, c] = float((y_end - y_start) * (x_end - x_start))
            active_cells += 1

    visited = ~active
    tol_sq = float(tol) * float(tol)

    sx = float(script_width) / float(vid_w) if script_width and script_width > 0 else 1.0
    sy = float(script_height) / float(vid_h) if script_height and script_height > 0 else 1.0

    merged_rects: List[RectPatch] = []

    for r in range(rows):
        for c in range(cols):
            if visited[r, c]:
                continue

            seed_b = float(cell_colors[r, c, 0])
            seed_g = float(cell_colors[r, c, 1])
            seed_r = float(cell_colors[r, c, 2])

            def can_merge(rr: int, cc: int) -> bool:
                if visited[rr, cc]:
                    return False
                db = float(cell_colors[rr, cc, 0]) - seed_b
                dg = float(cell_colors[rr, cc, 1]) - seed_g
                dr = float(cell_colors[rr, cc, 2]) - seed_r
                return (db * db + dg * dg + dr * dr) <= tol_sq

            # Candidate 1: Expand horizontally first, then vertically
            w1 = 1
            while c + w1 < cols and can_merge(r, c + w1):
                w1 += 1
            h1 = 1
            while r + h1 < rows:
                row_ok = True
                for cc in range(c, c + w1):
                    if not can_merge(r + h1, cc):
                        row_ok = False
                        break
                if not row_ok:
                    break
                h1 += 1

            # Candidate 2: Expand vertically first, then horizontally
            h2 = 1
            while r + h2 < rows and can_merge(r + h2, c):
                h2 += 1
            w2 = 1
            while c + w2 < cols:
                col_ok = True
                for rr in range(r, r + h2):
                    if not can_merge(rr, c + w2):
                        col_ok = False
                        break
                if not col_ok:
                    break
                w2 += 1

            if (w1 * h1) >= (w2 * h2):
                best_w, best_h = w1, h1
            else:
                best_w, best_h = w2, h2

            visited[r : r + best_h, c : c + best_w] = True

            # Weighted average BGR over merged grid cells
            sub_weights = cell_weights[r : r + best_h, c : c + best_w]
            w_sum = float(sub_weights.sum())
            if w_sum > 0:
                sub_colors = cell_colors[r : r + best_h, c : c + best_w]
                avg_b = float((sub_colors[:, :, 0] * sub_weights).sum() / w_sum)
                avg_g = float((sub_colors[:, :, 1] * sub_weights).sum() / w_sum)
                avg_r = float((sub_colors[:, :, 2] * sub_weights).sum() / w_sum)
            else:
                avg_b, avg_g, avg_r = seed_b, seed_g, seed_r

            px1 = bx0 + c * g
            py1 = by0 + r * g
            px2 = min(vid_w, bx0 + (c + best_w) * g)
            py2 = min(vid_h, by0 + (r + best_h) * g)

            sx1 = int(round(px1 * sx))
            sy1 = int(round(py1 * sy))
            sx2 = int(round(px2 * sx))
            sy2 = int(round(py2 * sy))
            if sx2 <= sx1:
                sx2 = sx1 + 1
            if sy2 <= sy1:
                sy2 = sy1 + 1

            rid = len(merged_rects)
            rect_grid[r : r + best_h, c : c + best_w] = rid
            merged_rects.append(
                RectPatch(
                    x1=sx1,
                    y1=sy1,
                    x2=sx2,
                    y2=sy2,
                    bgr=(avg_b, avg_g, avg_r),
                    area=(sx2 - sx1) * (sy2 - sy1),
                    rect_id=rid,
                    r0=r,
                    c0=c,
                    r1=r + best_h,
                    c1=c + best_w,
                )
            )

    color_groups = _quantize_rect_colors(merged_rects, tol)
    stats = {
        "active_cells": active_cells,
        "merged_rects": len(merged_rects),
        "color_groups": len(color_groups),
    }
    return color_groups, stats, rect_grid


def build_vector_mesh(
    inpainted_crop: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    orig_mask: np.ndarray,
    grid_size: int = 4,
    tolerance: int = 6,
    script_width: Optional[int] = None,
    script_height: Optional[int] = None,
) -> Tuple[Dict[Tuple[int, int, int], List[RectPatch]], Dict[str, int]]:
    """
    Sample the inpainted region on a grid and apply greedy 2D rectangle merging.

    Returns:
        (color_groups_map, stats_dict)
    """
    color_groups, stats, _ = build_vector_mesh_with_grid(
        inpainted_crop=inpainted_crop,
        crop_box=crop_box,
        orig_mask=orig_mask,
        grid_size=grid_size,
        tolerance=tolerance,
        script_width=script_width,
        script_height=script_height,
    )
    return color_groups, stats


def apply_layer_aware_overlap(
    events_plan: Sequence[Tuple[int, str, List[RectPatch]]],
    rect_grid: Optional[np.ndarray],
    grid_size: int,
    overlap_px: int,
    script_width: Optional[int] = None,
    script_height: Optional[int] = None,
    base_layer_rect_ids: Optional[Tuple[int, Sequence[int]]] = None,
) -> None:
    """
    Expand lower-layer rectangles into adjacent strictly higher-layer (or outer-boundary)
    neighbors 1 px at a time with diagonal corner safety and zero same-layer overlap
    (safe for both Union/Non-Zero and Even-Odd ASS rasterizers).
    """
    ov = max(0, int(overlap_px))
    if ov <= 0 or rect_grid is None or rect_grid.size == 0 or not events_plan:
        return

    max_id = -1
    if base_layer_rect_ids is not None:
        for rid in base_layer_rect_ids[1]:
            if rid > max_id:
                max_id = rid
    grid_rects_exist = False
    min_x = 10**9
    min_y = 10**9
    max_x = -(10**9)
    max_y = -(10**9)
    for _, _, chunk in events_plan:
        for rect in chunk:
            if rect.rect_id > max_id:
                max_id = rect.rect_id
            if rect.r1 > rect.r0 and rect.c1 > rect.c0:
                grid_rects_exist = True
                if rect.x1 < min_x:
                    min_x = rect.x1
                if rect.y1 < min_y:
                    min_y = rect.y1
                if rect.x2 > max_x:
                    max_x = rect.x2
                if rect.y2 > max_y:
                    max_y = rect.y2
    if max_id < 0 or not grid_rects_exist:
        return

    id_to_layer = np.full(max_id + 1, -1, dtype=np.int32)
    if base_layer_rect_ids is not None:
        base_l = int(base_layer_rect_ids[0])
        for rid in base_layer_rect_ids[1]:
            if rid >= 0:
                id_to_layer[rid] = base_l
    for layer_idx, _, chunk in events_plan:
        for rect in chunk:
            if rect.rect_id >= 0:
                id_to_layer[rect.rect_id] = int(layer_idx)

    layer_grid = np.where(rect_grid >= 0, id_to_layer[np.maximum(rect_grid, 0)], -1)
    rows, cols = layer_grid.shape

    sw_limit = int(script_width) if (script_width and script_width > 0) else 0
    sh_limit = int(script_height) if (script_height and script_height > 0) else 0

    min_x = max(0, min_x - ov)
    min_y = max(0, min_y - ov)
    max_x = max_x + ov
    max_y = max_y + ov
    if sw_limit > 0:
        max_x = min(sw_limit, max_x)
    if sh_limit > 0:
        max_y = min(sh_limit, max_y)

    px_w = max_x - min_x
    px_h = max_y - min_y
    if px_w <= 0 or px_h <= 0:
        return

    use_px = (px_w * px_h) <= 16_000_000
    if use_px:
        claimed_px = np.full((px_h, px_w), -1, dtype=np.int32)
        claimed_grid_layer = None
        claimed_grid_owner = None
    else:
        claimed_px = None
        claimed_grid_layer = np.full((rows, cols), -1, dtype=np.int32)
        claimed_grid_owner = np.full((rows, cols), -1, dtype=np.int32)

    for layer_idx, _, chunk in events_plan:
        self_layer = int(layer_idx)
        for rect in chunk:
            if rect.r1 > rect.r0 and rect.c1 > rect.c0:
                if use_px:
                    lx0 = max(0, rect.x1 - min_x)
                    ly0 = max(0, rect.y1 - min_y)
                    lx1 = min(px_w, rect.x2 - min_x)
                    ly1 = min(px_h, rect.y2 - min_y)
                    if lx1 > lx0 and ly1 > ly0:
                        claimed_px[ly0:ly1, lx0:lx1] = self_layer
                else:
                    claimed_grid_layer[rect.r0 : rect.r1, rect.c0 : rect.c1] = self_layer
                    claimed_grid_owner[rect.r0 : rect.r1, rect.c0 : rect.c1] = rect.rect_id

        def _grid_ok(gr0: int, gr1: int, gc0: int, gc1: int) -> bool:
            sr0 = max(0, gr0)
            sr1 = min(rows, gr1)
            sc0 = max(0, gc0)
            sc1 = min(cols, gc1)
            if sr0 >= sr1 or sc0 >= sc1:
                return True
            lg = layer_grid[sr0:sr1, sc0:sc1]
            return bool(np.all((lg == -1) | (lg > self_layer)))

        for rect in chunk:
            if rect.r1 <= rect.r0 or rect.c1 <= rect.c0:
                continue
            r0, c0, r1, c1 = rect.r0, rect.c0, rect.r1, rect.c1
            cell_w = max(1.0, float(rect.x2 - rect.x1) / float(max(1, c1 - c0)))
            cell_h = max(1.0, float(rect.y2 - rect.y1) / float(max(1, r1 - r0)))
            nx1, ny1, nx2, ny2 = rect.x1, rect.y1, rect.x2, rect.y2
            rid = rect.rect_id

            def _strip_free(sx1: int, sy1: int, sx2: int, sy2: int, gr0: int, gr1: int, gc0: int, gc1: int) -> bool:
                if use_px:
                    lx0 = max(0, sx1 - min_x)
                    ly0 = max(0, sy1 - min_y)
                    lx1 = min(px_w, sx2 - min_x)
                    ly1 = min(px_h, sy2 - min_y)
                    if lx1 > lx0 and ly1 > ly0:
                        return not bool(np.any(claimed_px[ly0:ly1, lx0:lx1] == self_layer))
                    return True
                sr0 = max(0, gr0)
                sr1 = min(rows, gr1)
                sc0 = max(0, gc0)
                sc1 = min(cols, gc1)
                if sr0 >= sr1 or sc0 >= sc1:
                    return True
                return not bool(
                    np.any(
                        (claimed_grid_layer[sr0:sr1, sc0:sc1] == self_layer)
                        & (claimed_grid_owner[sr0:sr1, sc0:sc1] != rid)
                    )
                )

            def _claim_strip(sx1: int, sy1: int, sx2: int, sy2: int, gr0: int, gr1: int, gc0: int, gc1: int) -> None:
                if use_px:
                    lx0 = max(0, sx1 - min_x)
                    ly0 = max(0, sy1 - min_y)
                    lx1 = min(px_w, sx2 - min_x)
                    ly1 = min(px_h, sy2 - min_y)
                    if lx1 > lx0 and ly1 > ly0:
                        claimed_px[ly0:ly1, lx0:lx1] = self_layer
                else:
                    sr0 = max(0, gr0)
                    sr1 = min(rows, gr1)
                    sc0 = max(0, gc0)
                    sc1 = min(cols, gc1)
                    if sr0 < sr1 and sc0 < sc1:
                        claimed_grid_layer[sr0:sr1, sc0:sc1] = self_layer
                        claimed_grid_owner[sr0:sr1, sc0:sc1] = rid

            # Right
            for _ in range(ov):
                cand = nx2 + 1
                if sw_limit > 0 and cand > sw_limit:
                    break
                cc = c1 + int(math.floor((cand - rect.x2 - 1e-6) / cell_w))
                gr0 = r0 - int(math.ceil(max(0.0, (rect.y1 - ny1) / cell_h - 1e-9)))
                gr1 = r1 + int(math.ceil(max(0.0, (ny2 - rect.y2) / cell_h - 1e-9)))
                if not _grid_ok(gr0, gr1, cc, cc + 1):
                    break
                if not _strip_free(nx2, ny1, cand, ny2, gr0, gr1, cc, cc + 1):
                    break
                _claim_strip(nx2, ny1, cand, ny2, gr0, gr1, cc, cc + 1)
                nx2 = cand

            # Left
            for _ in range(ov):
                cand = nx1 - 1
                if cand < 0:
                    break
                cc = c0 - 1 - int(math.floor((rect.x1 - cand - 1e-6) / cell_w))
                gr0 = r0 - int(math.ceil(max(0.0, (rect.y1 - ny1) / cell_h - 1e-9)))
                gr1 = r1 + int(math.ceil(max(0.0, (ny2 - rect.y2) / cell_h - 1e-9)))
                if not _grid_ok(gr0, gr1, cc, cc + 1):
                    break
                if not _strip_free(cand, ny1, nx1, ny2, gr0, gr1, cc, cc + 1):
                    break
                _claim_strip(cand, ny1, nx1, ny2, gr0, gr1, cc, cc + 1)
                nx1 = cand

            # Bottom
            for _ in range(ov):
                cand = ny2 + 1
                if sh_limit > 0 and cand > sh_limit:
                    break
                rr = r1 + int(math.floor((cand - rect.y2 - 1e-6) / cell_h))
                gc0 = c0 - int(math.ceil(max(0.0, (rect.x1 - nx1) / cell_w - 1e-9)))
                gc1 = c1 + int(math.ceil(max(0.0, (nx2 - rect.x2) / cell_w - 1e-9)))
                if not _grid_ok(rr, rr + 1, gc0, gc1):
                    break
                if not _strip_free(nx1, ny2, nx2, cand, rr, rr + 1, gc0, gc1):
                    break
                _claim_strip(nx1, ny2, nx2, cand, rr, rr + 1, gc0, gc1)
                ny2 = cand

            # Top
            for _ in range(ov):
                cand = ny1 - 1
                if cand < 0:
                    break
                rr = r0 - 1 - int(math.floor((rect.y1 - cand - 1e-6) / cell_h))
                gc0 = c0 - int(math.ceil(max(0.0, (rect.x1 - nx1) / cell_w - 1e-9)))
                gc1 = c1 + int(math.ceil(max(0.0, (nx2 - rect.x2) / cell_w - 1e-9)))
                if not _grid_ok(rr, rr + 1, gc0, gc1):
                    break
                if not _strip_free(nx1, cand, nx2, ny1, rr, rr + 1, gc0, gc1):
                    break
                _claim_strip(nx1, cand, nx2, ny1, rr, rr + 1, gc0, gc1)
                ny1 = cand

            rect.x1 = nx1
            rect.y1 = ny1
            rect.x2 = max(nx1 + 1, nx2)
            rect.y2 = max(ny1 + 1, ny2)
            rect.area = (rect.x2 - rect.x1) * (rect.y2 - rect.y1)


def format_ass_events(
    color_groups: Dict[Tuple[int, int, int], List[RectPatch]],
    raw_clip_inner: str,
    blur: float = 0.6,
    frame_spec: Optional[FrameSpec] = None,
    base_layer: int = 0,
    style: str = "Default",
    actor: str = "",
    overlap_px: int = 0,
    rect_grid: Optional[np.ndarray] = None,
    grid_size: int = 4,
    script_width: Optional[int] = None,
    script_height: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Format grouped rectangles into ASS event records."""
    blur_val = max(0.0, float(blur))
    blur_str = format_ass_number(blur_val)
    clip_tag = f"\\clip({raw_clip_inner})"

    # Order color groups deterministically from largest total area to smallest
    sorted_groups = sorted(
        color_groups.items(),
        key=lambda item: sum(rect.area for rect in item[1]),
        reverse=True,
    )

    events_plan: List[Tuple[int, str, List[RectPatch]]] = []
    eff_overlap = max(0, int(overlap_px))

    if eff_overlap > 0 and sorted_groups:
        bx1 = min(r.x1 for _, rects in sorted_groups for r in rects)
        by1 = min(r.y1 for _, rects in sorted_groups for r in rects)
        bx2 = max(r.x2 for _, rects in sorted_groups for r in rects)
        by2 = max(r.y2 for _, rects in sorted_groups for r in rects)

        dom_bgr, dom_rects = sorted_groups[0]
        pad = max(2, eff_overlap)
        ux1 = max(0, bx1 - pad)
        uy1 = max(0, by1 - pad)
        ux2 = min(int(script_width), bx2 + pad) if (script_width and script_width > 0) else (bx2 + pad)
        uy2 = min(int(script_height), by2 + pad) if (script_height and script_height > 0) else (by2 + pad)
        ux2 = max(ux1 + 1, ux2)
        uy2 = max(uy1 + 1, uy2)

        backing_rect = RectPatch(
            x1=ux1,
            y1=uy1,
            x2=ux2,
            y2=uy2,
            bgr=(float(dom_bgr[0]), float(dom_bgr[1]), float(dom_bgr[2])),
            area=(ux2 - ux1) * (uy2 - uy1),
            rect_id=-1,
            r0=0,
            c0=0,
            r1=0,
            c1=0,
        )
        events_plan.append((int(base_layer), bgr_to_ass_color(dom_bgr), [backing_rect]))
        base_layer_rect_ids: Optional[Tuple[int, Sequence[int]]] = (
            int(base_layer),
            [r.rect_id for r in dom_rects if r.rect_id >= 0],
        )

        layer_offset = 1
        for bgr_key, rects in sorted_groups[1:]:
            ass_color = bgr_to_ass_color(bgr_key)
            rects_sorted = sorted(rects, key=lambda r: (r.y1, r.x1))
            for chunk_start in range(0, len(rects_sorted), MAX_RECTS_PER_EVENT):
                chunk = [
                    RectPatch(
                        x1=r.x1,
                        y1=r.y1,
                        x2=r.x2,
                        y2=r.y2,
                        bgr=r.bgr,
                        area=r.area,
                        rect_id=r.rect_id,
                        r0=r.r0,
                        c0=r.c0,
                        r1=r.r1,
                        c1=r.c1,
                    )
                    for r in rects_sorted[chunk_start : chunk_start + MAX_RECTS_PER_EVENT]
                ]
                layer = int(base_layer) + layer_offset
                layer_offset += 1
                events_plan.append((layer, ass_color, chunk))

        if rect_grid is not None:
            apply_layer_aware_overlap(
                events_plan=events_plan,
                rect_grid=rect_grid,
                grid_size=grid_size,
                overlap_px=eff_overlap,
                script_width=script_width,
                script_height=script_height,
                base_layer_rect_ids=base_layer_rect_ids,
            )
    else:
        layer_offset = 0
        for bgr_key, rects in sorted_groups:
            ass_color = bgr_to_ass_color(bgr_key)
            # Sort rectangles top-to-bottom, left-to-right for clean ASS strings
            rects_sorted = sorted(rects, key=lambda r: (r.y1, r.x1))

            for chunk_start in range(0, len(rects_sorted), MAX_RECTS_PER_EVENT):
                chunk = [
                    RectPatch(
                        x1=r.x1,
                        y1=r.y1,
                        x2=r.x2,
                        y2=r.y2,
                        bgr=r.bgr,
                        area=r.area,
                        rect_id=r.rect_id,
                        r0=r.r0,
                        c0=r.c0,
                        r1=r.r1,
                        c1=r.c1,
                    )
                    for r in rects_sorted[chunk_start : chunk_start + MAX_RECTS_PER_EVENT]
                ]
                layer = int(base_layer) + layer_offset
                layer_offset += 1
                events_plan.append((layer, ass_color, chunk))

    events: List[Dict[str, Any]] = []
    start_ms = frame_spec.start_ms if frame_spec else None
    end_ms = frame_spec.end_ms if frame_spec else None
    start_ass = ms_to_ass_time(start_ms) if start_ms is not None else "0:00:00.00"
    end_ass = ms_to_ass_time(end_ms) if end_ms is not None else "0:00:05.00"

    for layer, ass_color, chunk in events_plan:
        draw_cmds = " ".join(
            f"m {r.x1} {r.y1} l {r.x2} {r.y1} {r.x2} {r.y2} {r.x1} {r.y2}"
            for r in chunk
        )
        text = (
            f"{{\\an7\\pos(0,0)\\bord0\\shad0\\blur{blur_str}"
            f"\\1c{ass_color}{clip_tag}\\p1}}{draw_cmds}"
        )
        dialogue = f"Dialogue: {layer},{start_ass},{end_ass},{style},{actor},0,0,0,,{text}"

        events.append(
            {
                "layer": layer,
                "frame": frame_spec.frame_index if frame_spec else 0,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "color": ass_color,
                "rect_count": len(chunk),
                "text": text,
                "dialogue": dialogue,
            }
        )

    return events


def _extract_frame_ffmpeg(video_path: str, frame_index: int) -> Optional[np.ndarray]:
    """Fallback frame extraction via ffmpeg CLI when cv2.VideoCapture fails."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-i",
            video_path,
            "-vf",
            f"select=eq(n\\,{frame_index})",
            "-vsync",
            "0",
            "-vframes",
            "1",
            tmp_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            return None
        img = cv2.imread(tmp_path, cv2.IMREAD_COLOR)
        return img
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def extract_frames(
    frame_specs: Sequence[FrameSpec],
    video_path: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """Load target frames from per-frame image paths, cv2.VideoCapture, or ffmpeg fallback."""
    loaded: Dict[int, np.ndarray] = {}
    video_targets: List[FrameSpec] = []

    for spec in frame_specs:
        if spec.image_path:
            if not os.path.exists(spec.image_path):
                raise FileNotFoundError(f"Frame image not found: {spec.image_path}")
            img = cv2.imread(spec.image_path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to decode image file: {spec.image_path}")
            loaded[spec.frame_index] = img
        else:
            video_targets.append(spec)

    if not video_targets:
        return loaded

    if not video_path:
        raise ValueError("Neither image_path nor video_path was provided to read video frames.")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    unique_indices = sorted({s.frame_index for s in video_targets})
    cap = cv2.VideoCapture(video_path)
    cur_pos = -1

    try:
        if cap.isOpened():
            for target_idx in unique_indices:
                if target_idx < 0:
                    target_idx = 0
                if cur_pos != target_idx:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(target_idx))
                ok, frame = cap.read()
                if ok and frame is not None:
                    loaded[target_idx] = frame
                    cur_pos = target_idx + 1
                else:
                    cur_pos = -1
    finally:
        cap.release()

    # Fallback to ffmpeg for any frames cv2 failed to read
    for target_idx in unique_indices:
        if target_idx not in loaded:
            frame = _extract_frame_ffmpeg(video_path, max(0, target_idx))
            if frame is None:
                raise RuntimeError(
                    f"Failed to extract frame {target_idx} from video {video_path!r} "
                    "(both OpenCV VideoCapture and ffmpeg failed)."
                )
            loaded[target_idx] = frame

    return loaded


def _probe_video_color_info(video_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Query video color_range and color_space via ffprobe when available."""
    if not video_path or not os.path.exists(video_path):
        return None, None
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        return None, None
    try:
        proc = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=color_range,color_space",
                "-of",
                "json",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None, None
        info = json.loads(proc.stdout)
        streams = info.get("streams") or []
        if not streams:
            return None, None
        st = streams[0]
        cr = st.get("color_range")
        cs = st.get("color_space")
        return (str(cr).lower() if cr else None), (str(cs).lower() if cs else None)
    except Exception:
        return None, None


def process_job(params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full rasterization -> inpainting -> mesh generation pipeline."""
    clip_str = params.get("clip") or ""
    engine = str(params.get("engine") or "telea")
    device_raw = params.get("device")
    device_pref = normalize_device(str(device_raw) if device_raw is not None else None, engine)
    grid_size = max(1, min(16, int(params.get("grid_size", 4))))
    tolerance = max(0, min(30, int(params.get("tolerance", 6))))
    blur = max(0.0, float(params.get("blur", 0.6)))
    enable_overlap = params.get("enable_overlap")
    raw_overlap = int(params.get("overlap_px", 0))
    eff_overlap = 0 if enable_overlap is False else max(0, min(8, raw_overlap))
    dilate_px = max(0, int(params.get("dilate", 3)))
    base_layer = int(params.get("base_layer", 0))
    style = str(params.get("style") or "Default")
    actor = str(params.get("actor") or "")
    ycbcr_matrix = str(params.get("ycbcr_matrix") or "").strip()
    harmonize_opt = params.get("harmonize")

    video_path = params.get("video_path") or params.get("video")
    image_path = params.get("image_path") or params.get("image")

    raw_frames = params.get("frames")
    frame_specs: List[FrameSpec] = []
    if isinstance(raw_frames, list) and raw_frames:
        for item in raw_frames:
            if isinstance(item, dict):
                frame_specs.append(
                    FrameSpec(
                        frame_index=int(item.get("frame", 0)),
                        start_ms=int(item["start_ms"]) if item.get("start_ms") is not None else None,
                        end_ms=int(item["end_ms"]) if item.get("end_ms") is not None else None,
                        image_path=item.get("image_path") or image_path,
                    )
                )
            else:
                frame_specs.append(FrameSpec(frame_index=int(item), image_path=image_path))
    else:
        frame_specs.append(
            FrameSpec(
                frame_index=int(params.get("frame", 0)),
                start_ms=int(params["start_ms"]) if params.get("start_ms") is not None else None,
                end_ms=int(params["end_ms"]) if params.get("end_ms") is not None else None,
                image_path=image_path,
            )
        )

    contours, raw_clip_inner = parse_ass_clip(clip_str)
    frames_map = extract_frames(frame_specs, video_path=video_path)

    first_img = next(iter(frames_map.values()))
    img_h, img_w = first_img.shape[:2]
    vid_w = int(params.get("video_width") or params.get("width") or img_w)
    vid_h = int(params.get("video_height") or params.get("height") or img_h)
    script_w = int(params.get("script_width") or vid_w)
    script_h = int(params.get("script_height") or vid_h)

    orig_mask = rasterize_clip_mask(
        contours=contours,
        video_width=vid_w,
        video_height=vid_h,
        script_width=script_w,
        script_height=script_h,
    )

    all_events: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if ycbcr_matrix.upper().startswith("PC."):
        video_range, video_space = _probe_video_color_info(video_path)
        if video_range == "tv":
            warnings.append(
                f"ASS header sets 'YCbCr Matrix: {ycbcr_matrix}' (full range 0-255), while the video stream "
                f"uses limited-range 'tv' ({video_space or 'bt709'}). In Aegisub (File -> Properties), "
                "set YCbCr Matrix to 'TV.709' so white does not shift toward light gray."
            )
        elif video_range is None and vid_w >= 1280:
            warnings.append(
                f"ASS header sets 'YCbCr Matrix: {ycbcr_matrix}'. Most 1080p HD videos are limited-range "
                "TV.709 (16-235); unless your video is true full-range, switch File -> Properties -> "
                "YCbCr Matrix to 'TV.709' to avoid white-to-gray mismatch."
            )

    total_active_cells = 0
    total_merged_rects = 0
    total_color_groups = 0
    used_engine = normalize_engine(engine)
    used_device = "cpu"

    for spec in frame_specs:
        frame_bgr = frames_map[spec.frame_index]
        if frame_bgr.shape[1] != vid_w or frame_bgr.shape[0] != vid_h:
            frame_bgr = cv2.resize(frame_bgr, (vid_w, vid_h), interpolation=cv2.INTER_LINEAR)

        inpainted_crop, crop_box, actual_engine, actual_device, warn = inpaint_region(
            frame_bgr=frame_bgr,
            orig_mask=orig_mask,
            engine=engine,
            dilate_px=dilate_px,
            grid_size=grid_size,
            device=device_pref,
            harmonize=harmonize_opt,
        )
        used_engine = actual_engine
        used_device = actual_device
        if warn and warn not in warnings:
            warnings.append(warn)

        color_groups, stats, rect_grid = build_vector_mesh_with_grid(
            inpainted_crop=inpainted_crop,
            crop_box=crop_box,
            orig_mask=orig_mask,
            grid_size=grid_size,
            tolerance=tolerance,
            script_width=script_w,
            script_height=script_h,
        )
        total_active_cells += stats["active_cells"]
        total_merged_rects += stats["merged_rects"]
        total_color_groups += stats["color_groups"]

        frame_events = format_ass_events(
            color_groups=color_groups,
            raw_clip_inner=raw_clip_inner,
            blur=blur,
            frame_spec=spec,
            base_layer=base_layer,
            style=style,
            actor=actor,
            overlap_px=eff_overlap,
            rect_grid=rect_grid,
            grid_size=grid_size,
            script_width=script_w,
            script_height=script_h,
        )
        all_events.extend(frame_events)

    return {
        "status": "ok",
        "worker_version": WORKER_VERSION,
        "engine_requested": normalize_engine(engine),
        "engine_used": used_engine,
        "device_requested": device_pref,
        "device_used": used_device,
        "warnings": warnings,
        "events": all_events,
        "stats": {
            "frames_processed": len(frame_specs),
            "events_generated": len(all_events),
            "active_cells": total_active_cells,
            "merged_rects": total_merged_rects,
            "color_groups": total_color_groups,
            "overlap_px": eff_overlap,
            "backing_layer": bool(eff_overlap > 0 and total_color_groups > 0),
            "video_resolution": [vid_w, vid_h],
            "script_resolution": [script_w, script_h],
        },
    }


def _parse_frames_arg(
    frames_arg: Optional[str],
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> Optional[List[Any]]:
    """Parse --frames (JSON array or comma-separated ints) or --start-frame/--end-frame."""
    if frames_arg:
        stripped = frames_arg.strip()
        if stripped.startswith("["):
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return parsed
        return [int(tok.strip()) for tok in stripped.split(",") if tok.strip()]

    if start_frame is not None and end_frame is not None:
        if end_frame < start_frame:
            end_frame = start_frame
        return list(range(start_frame, end_frame + 1))

    if start_frame is not None:
        return [start_frame]

    return None


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Erase masked text with CPU/GPU inpainting and rebuild as ASS \\p1 vector meshes."
    )
    parser.add_argument("--config", help="Path to JSON configuration payload.")
    parser.add_argument("--video", dest="video_path", help="Path to source video file.")
    parser.add_argument("--image", dest="image_path", help="Path to pre-extracted frame PNG/image.")
    parser.add_argument("--clip", help="ASS \\clip(...) vector drawing string.")
    parser.add_argument("--width", type=int, dest="video_width", help="Video width in pixels.")
    parser.add_argument("--height", type=int, dest="video_height", help="Video height in pixels.")
    parser.add_argument("--script-width", type=int, help="ASS PlayResX script width.")
    parser.add_argument("--script-height", type=int, help="ASS PlayResY script height.")
    parser.add_argument(
        "--engine",
        help="Inpainting engine: telea | ns | lama (or Aegisub dropdown labels).",
    )
    parser.add_argument(
        "--device",
        help="Compute device: cpu | gpu | cuda | mps | auto (or Aegisub dropdown labels).",
    )
    parser.add_argument("--grid-size", type=int, help="Grid patch resolution in px (1..16).")
    parser.add_argument("--tolerance", type=int, help="Color quantization/merge tolerance (0..30).")
    parser.add_argument("--blur", type=float, help="Edge feather / \\blur value (default 0.6).")
    parser.add_argument("--overlap", type=int, dest="overlap_px", help="Layer-aware edge overlap in px (0..8).")
    parser.add_argument("--dilate", type=int, help="Mask dilation radius in px (default 3).")
    parser.add_argument("--ycbcr-matrix", dest="ycbcr_matrix", help="ASS [Script Info] YCbCr Matrix header value.")
    parser.add_argument("--no-harmonize", action="store_true", help="Disable boundary color harmonization for LaMa.")
    parser.add_argument("--frame", type=int, help="Single frame index to process.")
    parser.add_argument("--frames", help="JSON array or comma-separated list of frame indices.")
    parser.add_argument("--start-frame", type=int, help="First frame index (inclusive).")
    parser.add_argument("--end-frame", type=int, help="Last frame index (inclusive).")
    parser.add_argument("--base-layer", type=int, help="Starting ASS layer number.")
    parser.add_argument("--style", help="ASS style name for formatted Dialogue lines.")
    parser.add_argument("--actor", help="ASS actor field for formatted Dialogue lines.")
    parser.add_argument("--output", help="Write result to file path instead of stdout.")
    parser.add_argument(
        "--format",
        choices=("json", "ass"),
        default="json",
        help="Output format: 'json' (default) or raw 'ass' Dialogue lines.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    params: Dict[str, Any] = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8") as fh:
            loaded_cfg = json.load(fh)
        if not isinstance(loaded_cfg, dict):
            raise ValueError("JSON config root must be an object.")
        params.update(loaded_cfg)

    cli_overrides = {
        "video_path": args.video_path,
        "image_path": args.image_path,
        "clip": args.clip,
        "video_width": args.video_width,
        "video_height": args.video_height,
        "script_width": args.script_width,
        "script_height": args.script_height,
        "engine": args.engine,
        "device": args.device,
        "grid_size": args.grid_size,
        "tolerance": args.tolerance,
        "blur": args.blur,
        "overlap_px": args.overlap_px,
        "dilate": args.dilate,
        "ycbcr_matrix": args.ycbcr_matrix,
        "frame": args.frame,
        "base_layer": args.base_layer,
        "style": args.style,
        "actor": args.actor,
    }
    for k, v in cli_overrides.items():
        if v is not None:
            params[k] = v
    if args.no_harmonize:
        params["harmonize"] = False

    parsed_frames = _parse_frames_arg(args.frames, args.start_frame, args.end_frame)
    if parsed_frames is not None:
        params["frames"] = parsed_frames

    try:
        result = process_job(params)
    except Exception as exc:
        err_payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        if args.format == "json":
            err_text = json.dumps(err_payload, ensure_ascii=False)
            if args.output and args.output != "-":
                with open(args.output, "w", encoding="utf-8") as fh:
                    fh.write(err_text)
            else:
                print(err_text)
        else:
            print(err_payload["error"], file=sys.stderr)
        return 1

    if result.get("warnings"):
        for w in result["warnings"]:
            print(f"[rebuilder_worker] Warning: {w}", file=sys.stderr)

    if args.format == "ass":
        out_text = "\n".join(ev["dialogue"] for ev in result["events"]) + ("\n" if result["events"] else "")
    else:
        out_text = json.dumps(result, ensure_ascii=False)

    if args.output and args.output != "-":
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(out_text)
    else:
        sys.stdout.write(out_text)
        if not out_text.endswith("\n"):
            sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
