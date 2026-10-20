"""Right-hand panel: four sliders tuning fix strengths at the selected keyframe."""
from __future__ import annotations

from typing import Callable

import customtkinter as ctk

# (key, label, min, max)
_SLIDERS = (
    ("dark", "Dark lift", 0.0, 1.0),
    ("lines", "Line removal", 0.0, 1.0),
    ("chroma", "Chroma fix", 0.0, 1.0),
    ("direction_deg", "Direction (deg)", -45.0, 45.0),
)


class ParamsPane(ctk.CTkFrame):
    def __init__(self, parent, on_change: Callable[[dict], None]):
        super().__init__(parent)
        self._on_change = on_change
        self._sliders: dict[str, ctk.CTkSlider] = {}
        self._labels: dict[str, ctk.CTkLabel] = {}
        for key, text, lo, hi in _SLIDERS:
            ctk.CTkLabel(self, text=text, anchor="w").pack(fill="x", padx=8, pady=(8, 0))
            slider = ctk.CTkSlider(
                self, from_=lo, to=hi, command=lambda _v, k=key: self._on_slide(k)
            )
            slider.set(0.0)
            slider.pack(fill="x", padx=8)
            value = ctk.CTkLabel(self, text="0.00", anchor="e")
            value.pack(fill="x", padx=8)
            self._sliders[key] = slider
            self._labels[key] = value

    def _on_slide(self, key: str) -> None:
        self._labels[key].configure(text=f"{self._sliders[key].get():.2f}")
        self._emit()

    def set_values(self, dark: float, lines: float, chroma: float, direction_deg: float) -> None:
        for key, v in zip(self._sliders, (dark, lines, chroma, direction_deg)):
            self._sliders[key].set(v)
            self._labels[key].configure(text=f"{v:.2f}")

    def get_values(self) -> dict:
        return {k: s.get() for k, s in self._sliders.items()}

    def _emit(self) -> None:
        self._on_change(self.get_values())
