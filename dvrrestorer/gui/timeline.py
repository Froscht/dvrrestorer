"""Bottom timeline: zone bars, three keyframe tracks, scrub cursor."""
from __future__ import annotations

import tkinter as tk
from typing import Callable

import customtkinter as ctk

from dvrrestorer.core.pipeline import Keyframe
from dvrrestorer.core.zone_detector import Zone

TRACKS = ("dark", "lines", "chroma")
ZONE_COLORS = {
    "dark_scene": "#FFA500",
    "chroma_noise": "#20B2AA",
    "row_streak": "#8A2BE2",
    "mixed": "#808080",
}
_RADIUS = 5


def _pixel_to_time(x: int, width: int, duration_s: float) -> float:
    return x / width * duration_s


def _time_to_pixel(t: float, width: int, duration_s: float) -> int:
    return int(round(t / duration_s * width))


def _pick_keyframe_at(
    x: int,
    y: int,
    keyframes: list[Keyframe],
    width: int,
    duration_s: float,
    track_pixel_height: int,
    tolerance: int = 6,
) -> tuple[int, str] | None:
    if y < 0 or track_pixel_height <= 0:
        return None
    row = y // track_pixel_height
    if row >= len(TRACKS) or duration_s <= 0 or width <= 0:
        return None
    best: tuple[int, int] | None = None  # (distance, index)
    for i, k in enumerate(keyframes):
        d = abs(_time_to_pixel(k.t_sec, width, duration_s) - x)
        if d <= tolerance and (best is None or d < best[0]):
            best = (d, i)
    return None if best is None else (best[1], TRACKS[row])


def _diamond_color(strength: float) -> str:
    shade = max(0, min(255, int(80 + strength * 175)))
    return f"#{shade:02x}{shade:02x}{shade:02x}"


class TimelineWidget(ctk.CTkFrame):
    def __init__(
        self,
        parent,
        on_keyframe_select: Callable[[int, str], None],
        on_keyframe_move: Callable[[int, float], None],
        on_time_scrub: Callable[[float], None],
        on_keyframe_delete: Callable[[int], None],
    ):
        super().__init__(parent)
        self._on_select = on_keyframe_select
        self._on_move = on_keyframe_move
        self._on_scrub = on_time_scrub
        self._on_delete = on_keyframe_delete
        self._duration = 0.0
        self._zones: list[Zone] = []
        self._keyframes: list[Keyframe] = []
        self._cursor = 0.0
        self._selected: tuple[int, str] | None = None
        self._track_pixel_height = 30
        self._dragged = False

        self._canvas = tk.Canvas(
            self, height=self._track_pixel_height * len(TRACKS),
            bg="#2b2b2b", highlightthickness=0,
        )
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<B1-Motion>", self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Delete>", lambda _e: self.delete_selected())
        self._canvas.bind("<Configure>", lambda _e: self._redraw())

    # --- public API ---
    def set_duration(self, seconds: float) -> None:
        self._duration = float(seconds)
        self._redraw()

    def set_zones(self, zones: list[Zone]) -> None:
        self._zones = list(zones)
        self._redraw()

    def set_keyframes(self, keyframes: list[Keyframe]) -> None:
        self._keyframes = keyframes
        self._redraw()

    def set_cursor(self, t_sec: float) -> None:
        self._cursor = t_sec
        self._redraw()

    def selected_keyframe_index(self) -> int | None:
        return None if self._selected is None else self._selected[0]

    def delete_selected(self) -> None:
        if self._selected is None:
            return
        index = self._selected[0]
        if 0 <= index < len(self._keyframes):
            self._keyframes.pop(index)
        self._selected = None
        self._redraw()
        self._on_delete(index)

    def _select_internal(self, index: int, track: str) -> None:
        self._selected = (index, track)

    # --- geometry ---
    def _width(self) -> int:
        return max(1, self._canvas.winfo_width())

    # --- events ---
    def _on_click(self, event) -> None:
        self._canvas.focus_set()
        self._dragged = False
        if self._duration <= 0:
            return
        w = self._width()
        hit = _pick_keyframe_at(event.x, event.y, self._keyframes, w, self._duration,
                                self._track_pixel_height)
        if hit is not None:
            self._selected = hit
            self._on_select(*hit)
            self._redraw()
        else:
            self._on_scrub(_pixel_to_time(event.x, w, self._duration))

    def _on_drag(self, event) -> None:
        if self._selected is None or self._duration <= 0:
            return
        t = _pixel_to_time(event.x, self._width(), self._duration)
        self._keyframes[self._selected[0]].t_sec = min(max(t, 0.0), self._duration)
        self._dragged = True
        self._redraw()

    def _on_release(self, _event) -> None:
        if self._dragged and self._selected is not None:
            idx = self._selected[0]
            self._on_move(idx, self._keyframes[idx].t_sec)
        self._dragged = False

    # --- painting ---
    def _redraw(self) -> None:
        c = self._canvas
        c.delete("all")
        w, th = self._width(), self._track_pixel_height
        for row in range(len(TRACKS)):
            c.create_rectangle(0, row * th, w, (row + 1) * th, fill="#3a3a3a", outline="#222222")
        if self._duration <= 0:
            return
        for z in self._zones:
            x0 = _time_to_pixel(z.t_start, w, self._duration)
            x1 = _time_to_pixel(z.t_end, w, self._duration)
            color = ZONE_COLORS.get(z.dominant, ZONE_COLORS["mixed"])
            c.create_rectangle(x0, 0, x1, th * len(TRACKS), fill=color, outline="", stipple="gray50")
        for k in self._keyframes:
            x = _time_to_pixel(k.t_sec, w, self._duration)
            for row, name in enumerate(TRACKS):
                cy = row * th + th // 2
                c.create_polygon(x, cy - _RADIUS, x + _RADIUS, cy, x, cy + _RADIUS, x - _RADIUS, cy,
                                 fill=_diamond_color(getattr(k, name)), outline="#ffffff")
        cx = _time_to_pixel(self._cursor, w, self._duration)
        c.create_line(cx, 0, cx, th * len(TRACKS), fill="#ff4040", width=2)
