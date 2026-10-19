"""Low-light boost: gamma + CLAHE blend, OSD-mask-aware."""

from __future__ import annotations

import cv2
import numpy as np


def _apply_mask(result: np.ndarray, original: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return result
    a = alpha.astype(np.float32).clip(0.0, 1.0)[..., None]
    out = result.astype(np.float32) * (1.0 - a) + original.astype(np.float32) * a
    return out.round().clip(0, 255).astype(np.uint8)


def fix_dark(
    frame_bgr: np.ndarray,
    alpha_mask: np.ndarray | None,
    strength: float,
    shadow_boost: float = 0.5,
) -> np.ndarray:
    """Boost dark regions. strength ∈ [0, 1]. 0 is identity.

    Combines a strength-scaled gamma LUT with CLAHE on the L channel; the
    mix is biased toward pure gamma when `shadow_boost` → 0 and toward
    the CLAHE-amplified result when `shadow_boost` → 1.
    """
    s = max(0.0, min(1.0, float(strength)))
    if s == 0.0:
        return frame_bgr.copy()

    # Gamma pass: gamma > 1 lifts shadows (brightens dark values).
    gamma = 1.0 + s * 1.5  # strength=0 → 1.0 (no-op); strength=1 → 2.5
    inv = 1.0 / gamma
    lut = np.array(
        [((i / 255.0) ** inv) * 255.0 for i in range(256)], dtype=np.uint8
    )
    gamma_out = cv2.LUT(frame_bgr, lut)

    # CLAHE on LAB L channel
    lab = cv2.cvtColor(gamma_out, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=1.0 + s * 2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    clahe_out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    sb = max(0.0, min(1.0, float(shadow_boost)))
    mix = gamma_out.astype(np.float32) * (1.0 - sb) + clahe_out.astype(np.float32) * sb
    result = mix.round().clip(0, 255).astype(np.uint8)

    return _apply_mask(result, frame_bgr, alpha_mask)
