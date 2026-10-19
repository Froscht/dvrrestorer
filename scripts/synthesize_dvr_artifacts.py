"""Generate synthetic test clips with controlled DVR artifacts.

Used by the test suite as a stand-in for real analog FPV DVR files until
the user supplies originals. Produces an mp4 with (a) a slowly moving
gradient and a few static shapes as content, and the ability to overlay
chroma noise, RF streaks, and darkening per time range.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def _draw_frame(canvas: np.ndarray, t_sec: float) -> np.ndarray:
    """Draw a moving gradient + a few static shapes onto canvas."""
    h, w = canvas.shape[:2]
    # Horizontal gradient that shifts over time
    shift = int((t_sec * 60) % w)
    x = (np.arange(w) + shift) % w
    grad = (x * 255 // w).astype(np.uint8)
    canvas[:, :, 1] = np.broadcast_to(grad, (h, w))  # green channel
    canvas[:, :, 0] = 100  # blue baseline
    canvas[:, :, 2] = 60   # red baseline
    # Static shapes
    cv2.rectangle(canvas, (w // 8, h // 4), (w // 3, 3 * h // 4), (200, 200, 60), -1)
    cv2.circle(canvas, (2 * w // 3, h // 2), min(h, w) // 6, (40, 180, 220), -1)
    # Moving marker
    mx = int((t_sec * 120) % w)
    cv2.circle(canvas, (mx, h // 6), 8, (255, 255, 255), -1)
    return canvas


def make_clean_clip(
    path: Path,
    duration_s: float = 10.0,
    fps: float = 30.0,
    size: tuple[int, int] = (960, 540),
) -> None:
    """Write a short synthetic clean clip to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    total = int(round(duration_s * fps))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(total):
        t = i / fps
        frame = _draw_frame(canvas.copy(), t)
        writer.write(frame)
    writer.release()


def add_chroma_noise(frame: np.ndarray, strength: float) -> np.ndarray:
    """Add gaussian noise to U/V channels only. strength ∈ [0, 1]."""
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV).astype(np.int16)
    sigma = max(0.0, float(strength)) * 25.0
    noise = np.random.normal(0.0, sigma, size=yuv.shape[:2] + (2,)).astype(np.int16)
    yuv[:, :, 1] = np.clip(yuv[:, :, 1] + noise[:, :, 0], 0, 255)
    yuv[:, :, 2] = np.clip(yuv[:, :, 2] + noise[:, :, 1], 0, 255)
    return cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)


def add_rf_streaks(
    frame: np.ndarray,
    strength: float,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Overlay a rotated sinusoidal stripe pattern on the Y channel."""
    h, w = frame.shape[:2]
    amp = max(0.0, float(strength)) * 30.0
    # Generate horizontal sinusoidal pattern (frequency ≈ 1 cycle per 6 px)
    rows = np.arange(h, dtype=np.float32)[:, None]
    pattern = np.sin(rows * (2 * np.pi / 6.0)).astype(np.float32) * amp
    pattern = np.broadcast_to(pattern, (h, w)).copy()
    if angle_deg != 0.0:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle_deg), 1.0)
        pattern = cv2.warpAffine(
            pattern, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV).astype(np.float32)
    yuv[:, :, 0] = np.clip(yuv[:, :, 0] + pattern, 0, 255)
    return cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)


def darken(frame: np.ndarray, gamma: float) -> np.ndarray:
    """Gamma-darken via LUT. gamma = 1.0 + 2.0 * strength (per plan)."""
    g = max(0.01, float(gamma))
    inv = 1.0 / g
    lut = np.array(
        [((i / 255.0) ** inv) * 255.0 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(frame, lut)


def draw_fake_osd(
    frame: np.ndarray, text: str = "15.5V", pos: tuple[float, float] = (0.75, 0.3)
) -> np.ndarray:
    """Paint a static OSD-like text block onto `frame` at a relative position.

    `pos` is (x_frac, y_frac) in [0, 1]. Also draws a few battery icons so
    the edge-density heuristic has something to lock onto.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    px = int(pos[0] * w)
    py = int(pos[1] * h)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.4, h / 400.0)
    cv2.putText(out, text, (px, py), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, text, (px, py), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    # Battery icon beside the text
    bx, by = px - int(0.08 * w), py - int(0.03 * h)
    cv2.rectangle(out, (bx, by), (bx + 24, by + 12), (255, 255, 255), 1)
    cv2.rectangle(out, (bx + 24, by + 3), (bx + 27, by + 9), (255, 255, 255), -1)
    return out


def make_artifacted_clip(
    path: Path,
    duration_s: float = 10.0,
    fps: float = 30.0,
    size: tuple[int, int] = (960, 540),
    ranges: Sequence[tuple[float, float, float, float, float]] = (),
) -> None:
    """Write a clip with per-time-range noise/streak/dark control.

    `ranges` entries: (t_start, t_end, noise_strength, streak_strength, dark_strength).
    Multiple ranges may overlap; applied in order.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {path}")
    total = int(round(duration_s * fps))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(total):
        t = i / fps
        frame = _draw_frame(canvas.copy(), t)
        for t0, t1, noise_s, streak_s, dark_s in ranges:
            if not (t0 <= t <= t1):
                continue
            if dark_s > 0.0:
                frame = darken(frame, gamma=1.0 + 2.0 * dark_s)
            if streak_s > 0.0:
                frame = add_rf_streaks(frame, strength=streak_s)
            if noise_s > 0.0:
                frame = add_chroma_noise(frame, strength=noise_s)
        writer.write(frame)
    writer.release()
