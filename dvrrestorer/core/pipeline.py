"""Chunked, OSD-aware restoration pipeline.

For every chunk of frames the pipeline:
1. Decodes the chunk plus `chroma_lookahead` extra on each end for
   temporal context in fix_chroma.
2. For each frame interpolates the keyframe values at t_sec.
3. Applies fix_dark → fix_lines → fix_chroma in order, all alpha-masked
   against the OSD mask so painted overlays stay intact.
4. Writes to a FrameWriter.

After every chunk is written, remux_audio pulls the original audio
stream back onto the cleaned video (optional-safe when the source has
no audio).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from dvrrestorer.core import fix_chroma, fix_dark, fix_lines, osd, video_io


@dataclass
class Keyframe:
    t_sec: float
    dark: float
    lines: float
    chroma: float


@dataclass
class PipelineConfig:
    keyframes: list[Keyframe]
    chunk_size: int = 32
    chroma_lookahead: int = 2
    direction_deg: float = 0.0
    temporal_w: float = 0.5


ProgressCB = Callable[[int, int], None]


def interpolate(keyframes: list[Keyframe], t_sec: float) -> tuple[float, float, float]:
    """Linear interpolation between keyframes with clamp outside range.

    Duplicate-t_sec entries resolve to the later one; after a stable
    sort by t_sec a duplicate t_sec just means the second value wins.
    """
    if not keyframes:
        return (0.0, 0.0, 0.0)
    kfs = sorted(enumerate(keyframes), key=lambda it: (it[1].t_sec, it[0]))
    sorted_kfs = [k for _, k in kfs]
    first, last = sorted_kfs[0], sorted_kfs[-1]
    if t_sec <= first.t_sec:
        # If multiple keyframes sit exactly at t_sec, prefer the last one
        # at that time.
        exact = [k for k in sorted_kfs if k.t_sec == t_sec]
        if exact:
            return (exact[-1].dark, exact[-1].lines, exact[-1].chroma)
        return (first.dark, first.lines, first.chroma)
    if t_sec >= last.t_sec:
        exact = [k for k in sorted_kfs if k.t_sec == t_sec]
        if exact:
            return (exact[-1].dark, exact[-1].lines, exact[-1].chroma)
        return (last.dark, last.lines, last.chroma)
    # Find bracketing pair
    for i in range(len(sorted_kfs) - 1):
        a, b = sorted_kfs[i], sorted_kfs[i + 1]
        if a.t_sec <= t_sec <= b.t_sec:
            if b.t_sec == a.t_sec:
                return (b.dark, b.lines, b.chroma)
            u = (t_sec - a.t_sec) / (b.t_sec - a.t_sec)
            return (
                a.dark + u * (b.dark - a.dark),
                a.lines + u * (b.lines - a.lines),
                a.chroma + u * (b.chroma - a.chroma),
            )
    return (last.dark, last.lines, last.chroma)


def run(
    input_path: Path,
    output_path: Path,
    config: PipelineConfig,
    osd_mask: np.ndarray | None = None,
    progress_cb: ProgressCB | None = None,
) -> None:
    """Run the restoration pipeline end-to-end."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    info = video_io.probe(input_path)

    if osd_mask is None:
        osd_mask = osd.detect_osd_mask(input_path)

    # Decode every frame into memory in chunks so we always have the
    # neighbors we need for temporal denoise.
    lookahead = max(0, int(config.chroma_lookahead))

    tmp_out = output_path.with_suffix(".video_only" + output_path.suffix)
    with video_io.FrameWriter(
        tmp_out, fps=info.fps, size=(info.width, info.height)
    ) as writer:
        all_frames: list[np.ndarray] = []
        for _, frame in video_io.decode_frames(input_path):
            all_frames.append(frame)

        total = len(all_frames)
        for i, frame in enumerate(all_frames):
            t_sec = i / info.fps
            dark_s, lines_s, chroma_s = interpolate(config.keyframes, t_sec)

            # Resize mask lazily if the clip resolution differs from the
            # mask (shouldn't happen but keeps us honest).
            mask = osd_mask
            if mask is not None and mask.shape != (info.height, info.width):
                mask = cv2.resize(mask, (info.width, info.height), interpolation=cv2.INTER_LINEAR)

            out = fix_dark.fix_dark(frame, mask, strength=dark_s)
            out = fix_lines.fix_lines(out, mask, strength=lines_s,
                                     direction_deg=config.direction_deg)

            prev = all_frames[max(0, i - lookahead): i]
            nxt = all_frames[i + 1: i + 1 + lookahead]
            out = fix_chroma.fix_chroma(
                out, prev, nxt, mask,
                strength=chroma_s, temporal_w=config.temporal_w,
            )

            writer.write(out)
            if progress_cb:
                progress_cb(i + 1, total)

    try:
        video_io.remux_audio(tmp_out, input_path, output_path)
    finally:
        try:
            tmp_out.unlink(missing_ok=True)
        except Exception:
            pass
