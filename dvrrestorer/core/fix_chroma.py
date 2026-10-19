"""Chroma denoise: spatial Non-Local Means + optional flow-median temporal.

For pure spatial denoise with no neighbors, we rely on OpenCV's fast NLM
on BGR. For temporal denoise, we warp each neighbor to the current
frame's coordinates via DIS optical flow, then take the per-pixel median
on U/V only (preserving luma detail).
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


def _warp_to(target_gray: np.ndarray, src_bgr: np.ndarray, src_gray: np.ndarray,
             flow_obj) -> np.ndarray:
    flow = flow_obj.calc(src_gray, target_gray, None)
    h, w = target_gray.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = xs + flow[..., 0]
    map_y = ys + flow[..., 1]
    return cv2.remap(src_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def fix_chroma(
    frame_bgr: np.ndarray,
    prev_frames: list[np.ndarray],
    next_frames: list[np.ndarray],
    alpha_mask: np.ndarray | None,
    strength: float,
    temporal_w: float = 0.5,
) -> np.ndarray:
    """Chroma denoise. strength ∈ [0, 1]; 0 is identity."""
    s = max(0.0, min(1.0, float(strength)))
    if s == 0.0:
        return frame_bgr.copy()
    tw = max(0.0, min(1.0, float(temporal_w)))

    # Spatial Non-Local Means on BGR
    h_param = s * 10.0
    spatial = cv2.fastNlMeansDenoisingColored(
        frame_bgr,
        None,
        h=h_param,
        hColor=h_param,
        templateWindowSize=7,
        searchWindowSize=21,
    )

    use_temporal = tw > 0.0 and (prev_frames or next_frames)
    if use_temporal:
        flow_obj = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        cur_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        warped_bgr = [frame_bgr]
        for neighbor in (prev_frames or []) + (next_frames or []):
            n_gray = cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY)
            warped_bgr.append(_warp_to(cur_gray, neighbor, n_gray, flow_obj))
        stack_yuv = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2YUV) for f in warped_bgr], axis=0)
        median_yuv = np.median(stack_yuv, axis=0).astype(np.uint8)
        # Replace U/V with median, keep original Y.
        current_yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
        temporal_yuv = current_yuv.copy()
        temporal_yuv[:, :, 1] = median_yuv[:, :, 1]
        temporal_yuv[:, :, 2] = median_yuv[:, :, 2]
        temporal = cv2.cvtColor(temporal_yuv, cv2.COLOR_YUV2BGR)
        mix = spatial.astype(np.float32) * (1.0 - tw) + temporal.astype(np.float32) * tw
        result = mix.round().clip(0, 255).astype(np.uint8)
    else:
        result = spatial

    return _apply_mask(result, frame_bgr, alpha_mask)
