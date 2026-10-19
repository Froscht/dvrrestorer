"""Chunked, OSD-aware restoration pipeline.

Decodes the clip in a sliding window large enough to supply
`chroma_lookahead` neighbors on each side of the current frame. Only
`(2 * chroma_lookahead + chunk_size)` frames live in RAM at any time, so
multi-hour clips do not blow up memory.

For each frame:
1. Pull its real timestamp from the capture (CAP_PROP_POS_MSEC) so VFR
   clips interpolate keyframes at the correct wall-time position.
2. Interpolate the three fix strengths at that t_sec.
3. Apply fix_dark → fix_lines → fix_chroma. All three honor the OSD
   alpha mask.
4. Write to a FrameWriter.

After the writer closes, remux_audio stream-copies the original audio
onto the result (no-op if the source has no audio).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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

    Duplicate-t_sec entries resolve to the later one.
    """
    if not keyframes:
        return (0.0, 0.0, 0.0)
    kfs = sorted(enumerate(keyframes), key=lambda it: (it[1].t_sec, it[0]))
    sorted_kfs = [k for _, k in kfs]
    first, last = sorted_kfs[0], sorted_kfs[-1]
    if t_sec <= first.t_sec:
        exact = [k for k in sorted_kfs if k.t_sec == t_sec]
        if exact:
            return (exact[-1].dark, exact[-1].lines, exact[-1].chroma)
        return (first.dark, first.lines, first.chroma)
    if t_sec >= last.t_sec:
        exact = [k for k in sorted_kfs if k.t_sec == t_sec]
        if exact:
            return (exact[-1].dark, exact[-1].lines, exact[-1].chroma)
        return (last.dark, last.lines, last.chroma)
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
    """Run the restoration pipeline end-to-end, chunk-streaming."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    info = video_io.probe(input_path)

    if osd_mask is None:
        osd_mask = osd.detect_osd_mask(input_path)
    if osd_mask is not None and osd_mask.size > 0 and osd_mask.shape != (info.height, info.width):
        osd_mask = cv2.resize(osd_mask, (info.width, info.height), interpolation=cv2.INTER_LINEAR)
    if osd_mask is not None and osd_mask.size == 0:
        osd_mask = None

    lookahead = max(0, int(config.chroma_lookahead))
    tmp_out = output_path.with_suffix(".video_only" + output_path.suffix)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    try:
        with video_io.FrameWriter(
            tmp_out, fps=info.fps, size=(info.width, info.height)
        ) as writer:
            maxw = 2 * lookahead + 1
            window: deque[tuple[np.ndarray, float]] = deque(maxlen=maxw)
            frames_read = 0
            frames_written = 0
            total = info.frame_count if info.frame_count > 0 else 0

            def _read_next() -> tuple[np.ndarray, float] | None:
                nonlocal frames_read
                pos_msec_before = cap.get(cv2.CAP_PROP_POS_MSEC)
                ok, frame = cap.read()
                if not ok:
                    return None
                if pos_msec_before > 0.0:
                    t = pos_msec_before / 1000.0
                else:
                    t = frames_read / info.fps if info.fps > 0 else 0.0
                frames_read += 1
                return frame, t

            # Prime window with up to lookahead+1 frames: the first current
            # has prev=[] and up to `lookahead` next-frames.
            for _ in range(lookahead + 1):
                nxt = _read_next()
                if nxt is None:
                    break
                window.append(nxt)

            # cur_idx is the index of the "current" frame inside the window.
            # While the window is still growing (early frames), cur_idx moves
            # with the newly-appended tail; once the window is at maxlen,
            # appends popleft so cur_idx stays put; after EOF, cur_idx walks
            # right through the remaining tail.
            cur_idx = 0

            while cur_idx < len(window):
                current, t_sec = window[cur_idx]
                prev = [window[i][0] for i in range(cur_idx)]
                nxt_frames = [window[i][0] for i in range(cur_idx + 1, len(window))]

                dark_s, lines_s, chroma_s = interpolate(config.keyframes, t_sec)
                out = fix_dark.fix_dark(current, osd_mask, strength=dark_s)
                out = fix_lines.fix_lines(out, osd_mask, strength=lines_s,
                                         direction_deg=config.direction_deg)
                out = fix_chroma.fix_chroma(
                    out, prev, nxt_frames, osd_mask,
                    strength=chroma_s, temporal_w=config.temporal_w,
                )
                writer.write(out)
                frames_written += 1
                if progress_cb:
                    progress_cb(frames_written, total)

                nxt_item = _read_next()
                if nxt_item is not None:
                    if len(window) < maxw:
                        window.append(nxt_item)
                        cur_idx += 1
                    else:
                        window.append(nxt_item)  # deque pops left for us
                        # cur_idx stays — center stays at `lookahead`
                else:
                    cur_idx += 1  # drain mode: walk right through tail
    finally:
        cap.release()

    try:
        video_io.remux_audio(tmp_out, input_path, output_path)
    finally:
        try:
            tmp_out.unlink(missing_ok=True)
        except Exception:
            pass
