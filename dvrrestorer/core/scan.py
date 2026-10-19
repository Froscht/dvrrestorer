"""Per-frame metrics on an FPV DVR clip.

All measures are computed at a downscaled resolution (480x270) because
they are approximately scale-invariant and this keeps a 6-minute clip
under a few seconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameMetrics:
    frame: int
    t_sec: float
    mean_lum: float
    chroma_noise: float
    row_streak: float
    sharpness: float
    black_ratio: float
    white_ratio: float
    freeze_mad: float | None


def _chroma_noise(frame_bgr: np.ndarray) -> float:
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    u = yuv[:, :, 1].astype(np.float32)
    v = yuv[:, :, 2].astype(np.float32)
    u_lp = cv2.GaussianBlur(u, (0, 0), 3)
    v_lp = cv2.GaussianBlur(v, (0, 0), 3)
    return float(np.hypot(u - u_lp, v - v_lp).mean())


def _row_streak(gray: np.ndarray) -> float:
    top = gray[: gray.shape[0] // 2].astype(np.float32)
    row_means = top.mean(axis=1)
    row_diffs = np.abs(np.diff(row_means))
    return float(np.percentile(row_diffs, 95))


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _dropout_ratios(gray: np.ndarray) -> tuple[float, float]:
    return float((gray < 10).mean()), float((gray > 245).mean())


def scan(path: Path, sample_every: int = 2) -> list[FrameMetrics]:
    """Sample frames at `sample_every` interval and compute metrics."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    out: list[FrameMetrics] = []
    prev_gray: np.ndarray | None = None
    idx = -1

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if sample_every > 1 and idx % sample_every:
                continue

            small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            black, white = _dropout_ratios(gray)
            freeze = None
            if prev_gray is not None:
                freeze = float(np.abs(prev_gray.astype(np.int16) - gray.astype(np.int16)).mean())

            out.append(FrameMetrics(
                frame=idx,
                t_sec=idx / fps,
                mean_lum=float(gray.mean()),
                chroma_noise=_chroma_noise(small),
                row_streak=_row_streak(gray),
                sharpness=_sharpness(gray),
                black_ratio=black,
                white_ratio=white,
                freeze_mad=freeze,
            ))
            prev_gray = gray
    finally:
        cap.release()

    return out
