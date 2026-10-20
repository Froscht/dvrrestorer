"""Main DVR Restorer window: wires preview, timeline and params together."""

from __future__ import annotations

import argparse
import copy
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np

from dvrrestorer.core import pipeline, video_io
from dvrrestorer.core.osd import detect_osd_mask
from dvrrestorer.core.pipeline import PipelineConfig
from dvrrestorer.core.scan import scan
from dvrrestorer.core.zone_detector import auto_keyframes, detect_zones
from dvrrestorer.gui.params import ParamsPane
from dvrrestorer.gui.preview import PreviewPane
from dvrrestorer.gui.preview_cache import PreviewRenderer
from dvrrestorer.gui.project import ProjectState, load_project, save_project
from dvrrestorer.gui.timeline import TimelineWidget


class App(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("DVR Restorer")
        self.geometry("1280x800")
        self._state = ProjectState.default()
        self._renderer: PreviewRenderer | None = None
        self._osd_mask: np.ndarray | None = None
        self._scan_done_event = threading.Event()
        self._cursor_t: float = 0.0
        self._scan_gen: int = 0
        self._export_in_progress = False

        menubar = tk.Menu(self)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open...", command=self._menu_open)
        filemenu.add_command(label="Save Profile...", command=self._menu_save_profile)
        filemenu.add_command(label="Load Profile...", command=self._menu_load_profile)
        filemenu.add_separator()
        filemenu.add_command(label="Export...", command=self._menu_export)
        filemenu.add_separator()
        filemenu.add_command(label="Quit", command=self.destroy)
        menubar.add_cascade(label="File", menu=filemenu)
        self.config(menu=menubar)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        self._preview = PreviewPane(self, on_toggle_before_after=self._on_toggle_before_after)
        self._preview.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._params = ParamsPane(self, on_change=self._on_params_change)
        self._params.grid(row=0, column=1, sticky="ns", padx=4, pady=4)
        self._timeline = TimelineWidget(
            self,
            on_keyframe_select=self._on_keyframe_select,
            on_keyframe_move=self._on_keyframe_move,
            on_time_scrub=self._on_time_scrub,
            on_keyframe_delete=self._on_keyframe_delete,
        )
        self._timeline.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        self._status = ctk.CTkLabel(self, text="")
        self._status.grid(row=2, column=0, columnspan=2, sticky="ew")

    # ---- menu handlers ----
    def _menu_open(self) -> None:
        p = filedialog.askopenfilename(
            title="Open clip",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.ts"), ("All", "*.*")],
        )
        if p:
            self.open_clip(Path(p))

    def _menu_save_profile(self) -> None:
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if p:
            save_project(self._state, Path(p))
            self._status.configure(text=f"Saved {p}")

    def _menu_load_profile(self) -> None:
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not p:
            return
        try:
            loaded = load_project(Path(p))
        except Exception as exc:  # noqa: BLE001
            self._status.configure(text=f"Load failed: {exc}")
            return
        self._state.keyframes = loaded.keyframes
        self._state.direction_deg = loaded.direction_deg
        self._timeline.set_keyframes(self._state.keyframes)
        self._status.configure(text=f"Loaded {p}")
        self._rerender()

    def _menu_export(self) -> None:
        if self._state.input_path is None:
            self.export(Path("unused.mp4"))
            return
        p = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4", "*.mp4")])
        if p:
            self.export(Path(p))

    # ---- public API ----
    def open_clip(self, path: Path) -> None:
        path = Path(path)
        try:
            info = video_io.probe(path)
            if info.fps <= 0:
                raise ValueError(f"Invalid fps: {info.fps}")
        except Exception as e:  # noqa: BLE001
            self._status.configure(text=f"Open failed: {e}")
            return
        self._scan_gen += 1
        gen = self._scan_gen
        self._state = ProjectState(
            input_path=path,
            duration_s=info.frame_count / info.fps,
            fps=info.fps,
            width=info.width,
            height=info.height,
            keyframes=[],
            direction_deg=0.0,
            osd_mask_path=None,
        )
        self._osd_mask = None
        self._cursor_t = 0.0
        self._scan_done_event.clear()
        self._init_renderer()
        self._timeline.set_duration(self._state.duration_s)
        self._timeline.set_keyframes([])
        self._timeline.set_zones([])
        self._status.configure(text="Scanning...")
        threading.Thread(target=self._scan_worker, args=(path, gen), daemon=True).start()

    def export(self, out_path: Path) -> int:
        if self._state.input_path is None:
            if self.winfo_viewable():
                try:
                    messagebox.showerror("No clip", "Open a clip first")
                except Exception:  # noqa: BLE001
                    pass
            self._status.configure(text="No clip loaded - open one first")
            return 1
        if not self._scan_done_event.is_set():
            self._status.configure(text="Scan still running - wait for it to finish")
            return 1
        if self._export_in_progress:
            self._status.configure(text="Export already running")
            return 1
        keyframes_snapshot = copy.deepcopy(self._state.keyframes)
        direction_snapshot = self._state.direction_deg
        osd_mask_snapshot = None if self._osd_mask is None else self._osd_mask.copy()
        input_snapshot = self._state.input_path
        self._export_in_progress = True
        threading.Thread(
            target=self._export_worker,
            args=(Path(out_path), input_snapshot, keyframes_snapshot,
                  direction_snapshot, osd_mask_snapshot),
            daemon=True,
        ).start()
        return 0

    def on_scrub(self, t_sec: float) -> None:
        self._on_time_scrub(t_sec)

    # ---- internals ----
    def _init_renderer(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._renderer = PreviewRenderer(self._state.input_path, self._osd_mask)

    def _scan_worker(self, path: Path, gen: int) -> None:
        try:
            rows = scan(path, sample_every=3)
            zones = detect_zones(rows)
            kfs = auto_keyframes(zones, clip_duration_s=self._state.duration_s)
            osd_mask = detect_osd_mask(path)
            self.after(0, self._apply_scan_result, zones, kfs, osd_mask, gen)
        except Exception as exc:  # noqa: BLE001
            self.after(0, lambda e=exc: self._scan_failed(e, gen))

    def _scan_failed(self, exc: Exception, gen: int) -> None:
        if gen != self._scan_gen:
            return
        self._status.configure(text=f"Scan failed: {exc}")
        self._scan_done_event.set()

    def _apply_scan_result(self, zones, kfs, osd_mask, gen: int) -> None:
        if gen != self._scan_gen:
            return  # stale scan from a previously opened clip
        self._osd_mask = osd_mask
        self._state.keyframes = kfs
        self._init_renderer()
        self._timeline.set_zones(zones)
        self._timeline.set_keyframes(kfs)
        self._status.configure(text=f"Scan done: {len(zones)} zones, {len(kfs)} keyframes")
        self._scan_done_event.set()
        self._rerender()

    def _rerender(self) -> None:
        if self._state.input_path is not None:
            self._on_time_scrub(self._cursor_t)

    def _on_time_scrub(self, t_sec: float) -> None:
        if self._state.input_path is None:
            return
        if self._renderer is None:
            self._init_renderer()
        self._cursor_t = t_sec
        frame_count = int(self._state.duration_s * self._state.fps)
        frame_idx = max(0, min(int(t_sec * self._state.fps), max(0, frame_count - 1)))
        before = self._preview.is_showing_before()
        if before:
            bgr = self._renderer.render_original(frame_idx)
        else:
            bgr = self._renderer.render(frame_idx, self._state.keyframes, self._state.direction_deg)
        self._preview.set_frame(bgr, is_before=before)
        self._timeline.set_cursor(t_sec)

    def _on_toggle_before_after(self, showing: bool) -> None:
        self._preview.set_overlay_text("BEFORE" if showing else "AFTER")
        self._rerender()

    def _on_params_change(self, values: dict) -> None:
        idx = self._timeline.selected_keyframe_index()
        if idx is None:
            self._status.configure(text="No keyframe selected")
            return
        if not (0 <= idx < len(self._state.keyframes)):
            return
        kf = self._state.keyframes[idx]
        kf.dark = values["dark"]
        kf.lines = values["lines"]
        kf.chroma = values["chroma"]
        self._state.direction_deg = values["direction_deg"]
        self._rerender()
        self._timeline.set_keyframes(self._state.keyframes)

    def _on_keyframe_select(self, idx: int, track: str) -> None:
        if not (0 <= idx < len(self._state.keyframes)):
            return
        kf = self._state.keyframes[idx]
        self._cursor_t = kf.t_sec
        self._params.set_values(kf.dark, kf.lines, kf.chroma, self._state.direction_deg)
        self._rerender()

    def _on_keyframe_move(self, idx: int, new_t: float) -> None:
        if 0 <= idx < len(self._state.keyframes):
            self._state.keyframes[idx].t_sec = new_t
        self._rerender()

    def _on_keyframe_delete(self, idx: int) -> None:
        # The timeline shares the list by reference and already popped it.
        self._rerender()

    def _export_worker(self, out_path: Path, input_path: Path, keyframes,
                       direction_deg: float, osd_mask) -> None:
        try:
            config = PipelineConfig(keyframes=keyframes, direction_deg=direction_deg)

            def progress_cb(cur: int, total: int) -> None:
                self.after(0, lambda c=cur, t=total: self._status.configure(text=f"Export: {c}/{t}"))

            pipeline.run(input_path, out_path, config, osd_mask, progress_cb)
            self.after(0, lambda: self._status.configure(text=f"Export done: {out_path}"))
        except Exception as exc:  # noqa: BLE001
            self.after(0, lambda e=exc: self._status.configure(text=f"Export failed: {e}"))
        finally:
            self.after(0, lambda: setattr(self, "_export_in_progress", False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dvrrestorer-gui")
    p.add_argument("--open", type=Path, help="Open this clip on startup")
    args = p.parse_args(argv)
    app = App()
    if args.open:
        app.after(100, lambda: app.open_clip(args.open))
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
