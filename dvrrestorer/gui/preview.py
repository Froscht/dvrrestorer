"""Center panel: shows the current preview frame with a Before/After toggle."""
from __future__ import annotations

from typing import Callable

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image


class PreviewPane(ctk.CTkFrame):
    def __init__(self, parent, on_toggle_before_after: Callable[[bool], None] | None = None):
        super().__init__(parent)
        self._on_toggle = on_toggle_before_after
        self._showing_before = False
        self._shape: tuple[int, int] | None = None
        self._current_image: ctk.CTkImage | None = None
        self._is_before = False

        self._label = ctk.CTkLabel(self, text="")
        self._label.pack(fill="both", expand=True)
        self._overlay = ctk.CTkLabel(self, text="")
        self._overlay.place(relx=0.02, rely=0.02, anchor="nw")
        self._button = ctk.CTkButton(self, text="Before/After", command=self._toggle)
        self._button.pack(pady=6)

    def set_frame(self, bgr: np.ndarray, is_before: bool = False) -> None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        self._current_image = ctk.CTkImage(light_image=Image.fromarray(rgb), size=(w, h))
        self._label.configure(image=self._current_image)
        self._shape = (h, w)
        self._is_before = is_before

    def set_overlay_text(self, text: str) -> None:
        self._overlay.configure(text=text)

    def current_shape(self) -> tuple[int, int] | None:
        return self._shape

    def is_showing_before(self) -> bool:
        return self._showing_before

    def _toggle(self) -> None:
        self._showing_before = not self._showing_before
        if self._on_toggle is not None:
            self._on_toggle(self._showing_before)
