"""RF-stripe / horizontal-beat removal via FFT-wedge notch.

The target pattern is a quasi-periodic stripe along one direction (0° =
horizontal). We rotate the Y plane so the target becomes horizontal,
transform to the frequency domain, zero (or attenuate) the spectral
column around DC that carries pure horizontal-stripe energy (high
vertical frequency, near-zero horizontal frequency), inverse-transform,
rotate back, and recombine with the original chroma.
"""

from __future__ import annotations

import cv2
import numpy as np


def _apply_mask(result: np.ndarray, original: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return result
    a = alpha.astype(np.float32).clip(0.0, 1.0)[..., None]
    out = result.astype(np.float32) * (1.0 - a) + original.astype(np.float32) * a
    return out.round().clip(0, 255).astype(np.uint8)


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if angle_deg == 0.0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def fix_lines(
    frame_bgr: np.ndarray,
    alpha_mask: np.ndarray | None,
    strength: float,
    direction_deg: float = 0.0,
) -> np.ndarray:
    """Remove periodic streaks oriented at `direction_deg` from `frame_bgr`.

    strength ∈ [0, 1]; 0 is identity.
    """
    s = max(0.0, min(1.0, float(strength)))
    if s == 0.0:
        return frame_bgr.copy()

    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    y = yuv[:, :, 0].astype(np.float32)
    y_rot = _rotate(y, -float(direction_deg))

    h, w = y_rot.shape
    F = np.fft.fft2(y_rot)
    # Horizontal stripes = vertical frequency content near zero horizontal
    # frequency. In the shifted spectrum this is a thin vertical band at
    # x = w/2. Build a notch mask that preserves DC + the lowest few
    # vertical frequencies and attenuates the rest of that column.
    Fs = np.fft.fftshift(F)
    cy, cx = h // 2, w // 2
    notch = np.ones_like(Fs, dtype=np.float32)
    # Half-width of the notch (in columns) — wider at higher strength.
    halfw = max(0, int(round(1 + s * 2)))
    # Vertical extent of the band — exclude a central DC region sized by
    # (1 - strength) so very low vertical frequencies (slow luminance
    # changes) survive.
    dc_half = max(1, int(round((1.0 - s) * 0.15 * h) + 2))
    attenuation = 1.0 - s  # strength=1 → full zero, strength=0.2 → 0.8 keep
    for dx in range(-halfw, halfw + 1):
        col = cx + dx
        if 0 <= col < w:
            notch[: cy - dc_half, col] = attenuation
            notch[cy + dc_half + 1 :, col] = attenuation

    Fs_filtered = Fs * notch
    F_filtered = np.fft.ifftshift(Fs_filtered)
    y_fixed_rot = np.real(np.fft.ifft2(F_filtered))
    y_fixed = _rotate(y_fixed_rot.astype(np.float32), float(direction_deg))
    y_fixed = np.clip(y_fixed, 0, 255).astype(np.uint8)

    yuv_out = yuv.copy()
    yuv_out[:, :, 0] = y_fixed
    result = cv2.cvtColor(yuv_out, cv2.COLOR_YUV2BGR)
    return _apply_mask(result, frame_bgr, alpha_mask)
