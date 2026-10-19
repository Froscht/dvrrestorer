"""OSD detection via edge-density + bbox-stability heuristic.

Analog FPV OSDs sit at the frame edges and persist across many seconds.
We sample N frames, run Canny on each, and threshold the mean edge map:
pixels where edges show up in a large fraction of samples are OSD. Then
we feather the binary mask into a soft alpha so copy-back doesn't leave
hard outlines.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def apply_soft_dilation(
    mask_bool: np.ndarray, dilate_px: int = 3, blur_sigma: float = 1.5
) -> np.ndarray:
    """Binary mask → float32 soft-alpha mask in [0, 1].

    Dilates by `dilate_px` (3x3 kernel iterated), then gaussian-blurs the
    result by `blur_sigma` to feather the edges.
    """
    m = mask_bool.astype(np.uint8) * 255
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        m = cv2.dilate(m, k, iterations=int(dilate_px))
    if blur_sigma > 0:
        m = cv2.GaussianBlur(m, (0, 0), float(blur_sigma))
    return (m.astype(np.float32) / 255.0).clip(0.0, 1.0)


def detect_osd_mask(
    path: Path, sample_every: int = 5, max_samples: int = 60
) -> np.ndarray:
    """Return a float32 HxW soft-alpha mask marking the OSD region."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Choose evenly-spaced sample indices
    if total <= 0:
        indices = list(range(max_samples))
    else:
        step = max(1, max(sample_every, total // max_samples))
        indices = list(range(0, total, step))[:max_samples]

    acc: np.ndarray | None = None
    count = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(i))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edges = cv2.dilate(edges, kernel, iterations=1)
        e = (edges > 0).astype(np.float32)
        acc = e if acc is None else acc + e
        count += 1
    cap.release()

    if acc is None or count == 0:
        return np.zeros((height, width), dtype=np.float32)

    mean_edges = acc / count
    threshold = max(0.6, float(mean_edges.max()) * 0.6)
    bm = mean_edges >= threshold
    return apply_soft_dilation(bm, dilate_px=3, blur_sigma=1.5)
