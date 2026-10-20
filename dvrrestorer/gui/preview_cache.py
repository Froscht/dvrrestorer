"""Single-frame, baseline-only preview renderer with an LRU cache.

Opens one VideoCapture per instance, seeks by frame index, downscales, and
runs fix_dark -> fix_lines -> fix_chroma with EMPTY temporal context (the
preview never looks at neighbouring frames). Fixed results are cached in an
OrderedDict LRU keyed on frame index and 2-decimal-rounded parameters.

Not thread-safe: intended for one caller at a time (the Tk main thread).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from dvrrestorer.core import fix_chroma, fix_dark, fix_lines, video_io
from dvrrestorer.core.pipeline import Keyframe, interpolate


class PreviewRenderer:
    def __init__(
        self,
        input_path: Path,
        osd_mask: np.ndarray | None,
        cache_size: int = 32,
        downscale: float = 0.5,
    ) -> None:
        self._info = video_io.probe(Path(input_path))
        self._downscale = downscale
        self._cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._cap = cv2.VideoCapture(str(input_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        self._mask: np.ndarray | None = None
        if osd_mask is not None and osd_mask.size > 0:
            self._mask = cv2.resize(
                osd_mask, (0, 0), fx=downscale, fy=downscale,
                interpolation=cv2.INTER_AREA,
            )

    def __enter__(self) -> "PreviewRenderer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _decode(self, frame_idx: int) -> np.ndarray:
        last = max(0, self._info.frame_count - 1)
        idx = max(0, min(int(frame_idx), last))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, last)
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError(f"Cannot decode frame {frame_idx}")
        return cv2.resize(
            frame, (0, 0), fx=self._downscale, fy=self._downscale,
            interpolation=cv2.INTER_AREA,
        )

    def _lookup(self, key: tuple) -> np.ndarray | None:
        out = self._cache.get(key)
        if out is not None:
            self._cache.move_to_end(key)
        return out

    def _store(self, key: tuple, out: np.ndarray) -> None:
        self._cache[key] = out
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def render(
        self,
        frame_idx: int,
        keyframes: list[Keyframe],
        direction_deg: float = 0.0,
    ) -> np.ndarray:
        t_sec = frame_idx / self._info.fps if self._info.fps > 0 else 0.0
        dark, lines, chroma = interpolate(keyframes, t_sec)
        key = (frame_idx, round(dark, 2), round(lines, 2), round(chroma, 2),
               round(direction_deg, 2))
        hit = self._lookup(key)
        if hit is not None:
            return hit

        frame = self._decode(frame_idx)
        out = fix_dark.fix_dark(frame, self._mask, strength=dark)
        out = fix_lines.fix_lines(out, self._mask, strength=lines,
                                  direction_deg=direction_deg)
        out = fix_chroma.fix_chroma(out, [], [], self._mask, strength=chroma)
        self._store(key, out)
        return out

    def render_original(self, frame_idx: int) -> np.ndarray:
        key = ("orig", frame_idx)
        hit = self._lookup(key)
        if hit is not None:
            return hit
        out = self._decode(frame_idx)
        self._store(key, out)
        return out
