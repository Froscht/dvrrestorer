"""Video I/O: probe, decode, encode, audio remux, side-by-side compare.

All video output goes through the ffmpeg binary bundled with
imageio-ffmpeg so we never rely on the system installation. OpenCV is
used only for decoding (its VideoCapture handles the common containers
well enough) and for raw frame processing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import imageio_ffmpeg
import numpy as np


# On Windows, suppress the console flash that pops up for every subprocess.
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffprobe_exe() -> str | None:
    """Return the ffprobe binary alongside imageio-ffmpeg's ffmpeg, if any.

    imageio-ffmpeg ships only ffmpeg; ffprobe may or may not be a sibling.
    When it isn't, we fall back to parsing ffmpeg's stderr.
    """
    ff = Path(_ffmpeg_exe())
    for name in ("ffprobe.exe", "ffprobe"):
        cand = ff.with_name(name)
        if cand.exists():
            return str(cand)
    return None


def _run(argv: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv,
        check=False,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=_CREATE_NO_WINDOW,
    )


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int
    has_audio: bool


def probe(path: Path) -> VideoInfo:
    """Return metadata about `path`. Never crashes on VFR containers —
    returns whatever the container declares; the caller decides whether
    to trust the declared fps."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    has_audio = _detect_audio_stream(path)
    return VideoInfo(path=path, fps=fps, frame_count=frame_count,
                     width=width, height=height, has_audio=has_audio)


def _detect_audio_stream(path: Path) -> bool:
    ffprobe = _ffprobe_exe()
    if ffprobe:
        p = _run(
            [
                ffprobe, "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "json", str(path),
            ],
            capture=True,
        )
        try:
            data = json.loads(p.stdout or b"{}")
        except json.JSONDecodeError:
            return False
        return any(s.get("codec_type") == "audio" for s in data.get("streams", []))
    # Fallback: parse ffmpeg -i stderr for "Audio:" lines.
    p = _run([_ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture=False)
    return b"Audio:" in (p.stderr or b"")


def decode_frames(
    path: Path, start: int = 0, end: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(frame_index, bgr_frame)` for frames in `[start, end)`.

    `end=None` means decode to the end of the clip.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
    idx = start
    try:
        while True:
            if end is not None and idx >= end:
                break
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
    finally:
        cap.release()


class FrameWriter:
    """Pipe raw BGR frames to ffmpeg, out as libx264 mp4.

    Context-manager; closes stdin and waits for ffmpeg on `__exit__`.
    """

    def __init__(
        self,
        path: Path,
        fps: float,
        size: tuple[int, int],
        crf: int = 18,
        preset: str = "veryfast",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.size = (int(size[0]), int(size[1]))
        self.crf = int(crf)
        self.preset = preset
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> "FrameWriter":
        w, h = self.size
        argv = [
            _ffmpeg_exe(),
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", f"{self.fps}",
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(self.crf),
            "-preset", self.preset,
            str(self.path),
        ]
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=_CREATE_NO_WINDOW,
        )
        return self

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("FrameWriter is not open")
        if frame_bgr.dtype != np.uint8:
            frame_bgr = frame_bgr.astype(np.uint8)
        if frame_bgr.shape[1] != self.size[0] or frame_bgr.shape[0] != self.size[1]:
            raise ValueError(
                f"Frame shape {frame_bgr.shape[1]}x{frame_bgr.shape[0]} "
                f"does not match writer size {self.size[0]}x{self.size[1]}"
            )
        self._proc.stdin.write(frame_bgr.tobytes())

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            # communicate() drains stderr; wait() alone deadlocks if the
            # stderr PIPE fills.
            _, stderr = self._proc.communicate(timeout=300)
            if self._proc.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg writer exited {self._proc.returncode}: "
                    f"{(stderr or b'').decode(errors='replace')[-800:]}"
                )
        finally:
            self._proc = None

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def remux_audio(
    video_only_path: Path, audio_source: Path, out_path: Path
) -> None:
    """Copy the audio stream from `audio_source` onto `video_only_path`.

    When `audio_source` has no audio stream, the output is video-only but
    still valid — covered by Review-focus #3.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        _ffmpeg_exe(),
        "-y",
        "-i", str(video_only_path),
        "-i", str(audio_source),
        "-c:v", "copy",
        "-c:a", "copy",     # stream-copy audio: no generation loss
        "-map", "0:v:0",
        "-map", "1:a?",     # all audio streams; '?' = optional, no-op if absent
        str(out_path),
    ]
    p = _run(argv)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg remux failed: {p.stderr.decode(errors='replace')}")


def stack_compare(before_path: Path, after_path: Path, out_path: Path) -> None:
    """Produce a side-by-side (hstack) mp4 of before and after."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        _ffmpeg_exe(),
        "-y",
        "-i", str(before_path),
        "-i", str(after_path),
        "-filter_complex", "[0:v][1:v]hstack=inputs=2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        str(out_path),
    ]
    p = _run(argv)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg hstack failed: {p.stderr.decode(errors='replace')}")
