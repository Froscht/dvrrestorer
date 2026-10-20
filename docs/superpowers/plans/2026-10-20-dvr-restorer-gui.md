# DVR Restorer GUI (Phase 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put a CustomTkinter desktop GUI on top of the headless MVP so the user can open a clip, scrub the timeline, tune per-keyframe fix strengths with live preview, and export — without ever touching the CLI.

**Architecture:** `dvrrestorer.gui.app.App` composes three widgets (`TimelineWidget`, `PreviewPane`, `ParamsPane`) around a `ProjectState` dataclass. All heavy lifting (scan, OSD-detect, pipeline.run on export) runs in background threads; the preview pipeline is a baseline-only, quarter-res, single-frame renderer with an LRU cache so scrubbing stays under 100 ms. GUI code is a thin layer over `dvrrestorer.core.*` — no new DSP.

**Tech Stack:** customtkinter 5.x, tkinter, Pillow (ImageTk), pytest. Reuses MVP: core.pipeline, core.scan, core.zone_detector, core.osd, core.fix_*, profiles.

**Spec:** `docs/superpowers/specs/2026-10-19-dvr-restorer-design.md` (sections "Timeline + Zone-Detection" and "Preview-Strategie" cover the GUI side of the spec).

## Global Constraints

- Python 3.11+.
- GUI package: `dvrrestorer.gui`.
- CustomTkinter is the only GUI toolkit (spec).
- Preview MUST be baseline-only (no ML) and run at quarter-res (960×540 or less) — fix_chroma gets an empty temporal context for the preview path (spec).
- Every long-running operation (scan, OSD-detect, export) runs on a background `threading.Thread`; the Tk event loop is never blocked for more than one frame-render.
- A `ProjectState` roundtrip through `profiles.save_profile` / `load_profile` must preserve all keyframes and `direction_deg`.
- Alpha-mask contract from the MVP is unchanged — fix_* calls use the OSD mask the GUI holds (`None` if the user hasn't computed it yet).

## Review Focus

1. **Scrub while scan is still running** — user drags the timeline cursor before the background scan has produced zones; preview must still render using whatever keyframes exist (empty = defaults). Pin test in **Task 6** (`test_app_scrub_before_scan_done`).
2. **Export with no clip loaded** — hitting Export without a file must show an error and not crash. Pin in **Task 6** (`test_app_export_without_input_errors`).
3. **Save profile with zero keyframes** — `interpolate([], t)` already returns zeroes; save/load through `profiles` must not crash and the loaded profile stays empty. Pin in **Task 1** (`test_project_state_save_empty_keyframes`).
4. **Scrubbing past the clip end** — user drags the cursor beyond `t_end`; `interpolate` must clamp (already tested in MVP), but the preview cache must not request a non-existent frame index. Pin in **Task 2** (`test_preview_cache_clamps_oversized_index`).
5. **Delete the currently-selected keyframe** — selection pointer dangles; next scrub must not NPE. Pin in **Task 4** (`test_timeline_delete_selected_keyframe_clears_selection`).

---

### Task 1: Project State + GUI Package Skeleton

**Files:**
- Create: `dvrrestorer/gui/__init__.py` (empty)
- Create: `dvrrestorer/gui/project.py`
- Create: `tests/test_project_state.py`

**Interfaces:**
- Consumes: `dvrrestorer.core.pipeline.Keyframe`, `dvrrestorer.profiles` (save/load).
- Produces:
  - `@dataclass ProjectState { input_path: Path | None, duration_s: float, fps: float, width: int, height: int, keyframes: list[Keyframe], direction_deg: float, osd_mask_path: Path | None }`
  - `save_project(state: ProjectState, path: Path) -> None` (wraps `profiles.save_profile` + stores input_path alongside)
  - `load_project(path: Path) -> ProjectState`
  - `ProjectState.default() -> ProjectState`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_project_state.py
def test_project_state_default_is_empty():
    from dvrrestorer.gui.project import ProjectState
    s = ProjectState.default()
    assert s.input_path is None
    assert s.keyframes == []

def test_project_state_save_load_roundtrip(tmp_path):
    from dvrrestorer.gui.project import ProjectState, save_project, load_project
    from dvrrestorer.core.pipeline import Keyframe
    s = ProjectState(
        input_path=tmp_path / "in.mp4", duration_s=5.0, fps=30.0,
        width=960, height=540,
        keyframes=[Keyframe(0, 0.2, 0.2, 0.2), Keyframe(5, 0.5, 0.4, 0.3)],
        direction_deg=3.0, osd_mask_path=None,
    )
    p = tmp_path / "proj.json"
    save_project(s, p)
    out = load_project(p)
    assert out.keyframes == s.keyframes
    assert out.direction_deg == 3.0
    assert out.input_path == s.input_path

def test_project_state_save_empty_keyframes(tmp_path):
    """Review-focus #3: zero-keyframe projects must roundtrip."""
    from dvrrestorer.gui.project import ProjectState, save_project, load_project
    s = ProjectState.default()
    s.input_path = tmp_path / "in.mp4"
    p = tmp_path / "empty.json"
    save_project(s, p)
    out = load_project(p)
    assert out.keyframes == []
    assert out.input_path == s.input_path
```

- [ ] **Step 2: Run tests to verify they fail** — `pytest tests/test_project_state.py -v`

- [ ] **Step 3: Implement `ProjectState` dataclass, `save_project`, `load_project`, `default()`.**

Save-format: wrap `profiles.save_profile`'s dict with extra keys `input_path`, `duration_s`, `fps`, `width`, `height`, `osd_mask_path`, under schema key `"gui_project_version": 1`.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add dvrrestorer/gui/__init__.py dvrrestorer/gui/project.py tests/test_project_state.py
git commit -m "feat(gui): ProjectState with profile-backed persistence"
```

---

### Task 2: Preview Cache + Single-Frame Baseline Renderer

**Files:**
- Create: `dvrrestorer/gui/preview_cache.py`
- Create: `tests/test_preview_cache.py`

**Interfaces:**
- Consumes: `core.video_io.probe`, `core.fix_dark`, `core.fix_lines`, `core.fix_chroma`, `core.pipeline.interpolate`.
- Produces:
  - `class PreviewRenderer`:
    - `__init__(input_path: Path, osd_mask: np.ndarray | None, cache_size: int = 32, downscale: float = 0.5)`
    - `render(frame_idx: int, keyframes: list[Keyframe], direction_deg: float = 0.0) -> np.ndarray` (BGR uint8, downscaled)
    - `render_original(frame_idx: int) -> np.ndarray` (same shape, no fixes — for before/after toggle)
    - `close() -> None`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preview_cache.py
import numpy as np

def test_preview_renders_frame_at_downscale(synth_clip_noisy):
    from dvrrestorer.gui.preview_cache import PreviewRenderer
    from dvrrestorer.core.pipeline import Keyframe
    r = PreviewRenderer(synth_clip_noisy, osd_mask=None, downscale=0.5)
    out = r.render(100, [Keyframe(0, 0.3, 0.3, 0.3)])
    assert out.dtype == np.uint8
    assert out.shape[0] == 540 // 2  # noisy clip is 540h
    assert out.shape[1] == 960 // 2
    r.close()

def test_preview_cache_hit_is_fast(synth_clip_noisy):
    import time
    from dvrrestorer.gui.preview_cache import PreviewRenderer
    from dvrrestorer.core.pipeline import Keyframe
    r = PreviewRenderer(synth_clip_noisy, osd_mask=None)
    kfs = [Keyframe(0, 0.3, 0.3, 0.3)]
    r.render(100, kfs)
    t0 = time.perf_counter()
    r.render(100, kfs)   # same params → cache hit
    dt = time.perf_counter() - t0
    assert dt < 0.01
    r.close()

def test_preview_cache_clamps_oversized_index(synth_clip_noisy):
    """Review-focus #4."""
    from dvrrestorer.gui.preview_cache import PreviewRenderer
    from dvrrestorer.core.pipeline import Keyframe
    r = PreviewRenderer(synth_clip_noisy, osd_mask=None)
    out = r.render(999999, [Keyframe(0, 0.3, 0.3, 0.3)])
    assert out is not None and out.ndim == 3
    r.close()
```

- [ ] **Step 2: Run tests** → FAIL.

- [ ] **Step 3: Implement `PreviewRenderer`.**

Core behavior:
- Open `cv2.VideoCapture` once per instance (reuse across renders).
- `render(frame_idx, keyframes, direction_deg)` seeks via `cap.set(CAP_PROP_POS_FRAMES, min(frame_idx, frame_count-1))`, reads one frame, downscales (fx=fy=downscale, `INTER_AREA`), computes `t_sec = frame_idx / fps`, pulls (dark,lines,chroma) from `interpolate`, runs `fix_dark → fix_lines → fix_chroma(prev=[], next=[], ...)` with the downscaled OSD mask, returns BGR uint8.
- Cache key: `(frame_idx, dark_rounded, lines_rounded, chroma_rounded, direction_deg_rounded)` with 2-decimal rounding; `collections.OrderedDict` LRU, bounded by `cache_size`.
- `render_original` only decodes+downscales, no fixes.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add dvrrestorer/gui/preview_cache.py tests/test_preview_cache.py
git commit -m "feat(gui): quarter-res preview renderer with LRU cache"
```

---

### Task 3: ParamsPane Widget

**Files:**
- Create: `dvrrestorer/gui/params.py`
- Create: `tests/test_params_pane.py`

**Interfaces:**
- Consumes: nothing from previous tasks (pure UI).
- Produces:
  - `class ParamsPane(ctk.CTkFrame)`:
    - `__init__(parent, on_change: Callable[[dict], None])`
    - `set_values(dark: float, lines: float, chroma: float, direction_deg: float) -> None`
    - `get_values() -> dict` with keys `dark`, `lines`, `chroma`, `direction_deg`

Sliders: dark/lines/chroma in `[0, 1]`, direction_deg in `[-45, 45]`. `on_change` fires with the full dict after any slider release.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_params_pane.py
import pytest
pytestmark = pytest.mark.gui

def test_params_pane_roundtrip():
    import customtkinter as ctk
    from dvrrestorer.gui.params import ParamsPane
    root = ctk.CTk()
    seen = []
    pane = ParamsPane(root, on_change=seen.append)
    pane.set_values(dark=0.4, lines=0.5, chroma=0.6, direction_deg=12.0)
    v = pane.get_values()
    assert v == {"dark": 0.4, "lines": 0.5, "chroma": 0.6, "direction_deg": 12.0}
    root.destroy()

def test_params_pane_change_callback():
    import customtkinter as ctk
    from dvrrestorer.gui.params import ParamsPane
    root = ctk.CTk()
    seen = []
    pane = ParamsPane(root, on_change=seen.append)
    pane.set_values(dark=0.4, lines=0.5, chroma=0.6, direction_deg=0.0)
    pane._emit()      # internal test hook: fire callback with current values
    assert seen and seen[-1]["dark"] == 0.4
    root.destroy()
```

Register the `gui` marker in `pyproject.toml` under `[tool.pytest.ini_options] markers` so unmarked runs skip it on headless CI. Local runs include it by default.

- [ ] **Step 2: Run tests** → FAIL.

- [ ] **Step 3: Implement `ParamsPane` with four `CTkSlider`s plus labels. Expose `_emit()` as a test-visible hook.**

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add dvrrestorer/gui/params.py tests/test_params_pane.py pyproject.toml
git commit -m "feat(gui): ParamsPane with four sliders + on_change callback"
```

---

### Task 4: TimelineWidget

**Files:**
- Create: `dvrrestorer/gui/timeline.py`
- Create: `tests/test_timeline.py`

**Interfaces:**
- Consumes: `core.pipeline.Keyframe`, `core.zone_detector.Zone`.
- Produces:
  - `class TimelineWidget(ctk.CTkCanvas)`:
    - `__init__(parent, on_keyframe_select, on_keyframe_move, on_time_scrub, on_keyframe_delete)`
    - `set_duration(seconds: float) -> None`
    - `set_zones(zones: list[Zone]) -> None`
    - `set_keyframes(keyframes: list[Keyframe]) -> None`
    - `set_cursor(t_sec: float) -> None`
    - `selected_keyframe_index() -> int | None`
    - `delete_selected() -> None`
  - Module-level pure helpers (headless-testable):
    - `_pixel_to_time(x: int, width: int, duration_s: float) -> float`
    - `_time_to_pixel(t: float, width: int, duration_s: float) -> int`
    - `_pick_keyframe_at(x: int, y: int, keyframes: list[Keyframe], width: int, duration_s: float, track_pixel_height: int, tolerance: int = 6) -> tuple[int, str] | None` — returns `(index, track_name)` or None. `track_name` ∈ {"dark","lines","chroma"}.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_timeline.py
import pytest

def test_pixel_time_roundtrip():
    from dvrrestorer.gui.timeline import _pixel_to_time, _time_to_pixel
    assert _time_to_pixel(_pixel_to_time(200, 400, 10.0), 400, 10.0) == 200

def test_pick_keyframe_hits_closest_within_tolerance():
    from dvrrestorer.gui.timeline import _pick_keyframe_at
    from dvrrestorer.core.pipeline import Keyframe
    kfs = [Keyframe(0.0, 0.3, 0.3, 0.3), Keyframe(5.0, 0.5, 0.5, 0.5)]
    # 400 px wide, 10 s → 5s is at x=200.  Track pixel height = 30; dark on row 0.
    hit = _pick_keyframe_at(x=201, y=5, keyframes=kfs, width=400, duration_s=10.0,
                            track_pixel_height=30, tolerance=6)
    assert hit == (1, "dark")

def test_pick_keyframe_returns_none_when_far():
    from dvrrestorer.gui.timeline import _pick_keyframe_at
    from dvrrestorer.core.pipeline import Keyframe
    kfs = [Keyframe(5.0, 0.3, 0.3, 0.3)]
    hit = _pick_keyframe_at(x=10, y=5, keyframes=kfs, width=400, duration_s=10.0,
                            track_pixel_height=30, tolerance=6)
    assert hit is None

@pytest.mark.gui
def test_timeline_delete_selected_keyframe_clears_selection():
    """Review-focus #5."""
    import customtkinter as ctk
    from dvrrestorer.gui.timeline import TimelineWidget
    from dvrrestorer.core.pipeline import Keyframe
    root = ctk.CTk()
    tl = TimelineWidget(root, on_keyframe_select=lambda *a: None,
                        on_keyframe_move=lambda *a: None,
                        on_time_scrub=lambda *a: None,
                        on_keyframe_delete=lambda *a: None)
    tl.set_duration(10.0)
    tl.set_keyframes([Keyframe(1.0, 0.3, 0.3, 0.3), Keyframe(5.0, 0.5, 0.5, 0.5)])
    tl._select_internal(index=1, track="dark")   # internal test hook
    tl.delete_selected()
    assert tl.selected_keyframe_index() is None
    root.destroy()
```

- [ ] **Step 2: Run tests** → FAIL.

- [ ] **Step 3: Implement helpers (`_pixel_to_time`, `_time_to_pixel`, `_pick_keyframe_at`) first; they take no Tk. Then implement `TimelineWidget` using `tkinter.Canvas` (CustomTkinter does not ship a CTkCanvas; use `ctk.CTkFrame` + `tk.Canvas` child).**

Draw order each repaint:
- Background per track (grey)
- Zone bars (coloured by dominant metric: dark=orange, chroma_noise=teal, row_streak=purple)
- Keyframes as filled diamonds (one per track per Keyframe) at `(time_to_pixel(k.t_sec), track_y)`; filled brightness = strength
- Cursor as vertical line
- Click handler: `_pick_keyframe_at` → fires `on_keyframe_select(index, track)`; miss on a track → fires `on_time_scrub(t_sec)`
- Drag: moves the selected keyframe; releases fire `on_keyframe_move(index, new_t_sec)`
- `Delete` key: `on_keyframe_delete(index)` and clear selection

- [ ] **Step 4: Run tests** → PASS. Non-GUI tests always; `-m gui` for the Tk ones (local dev + CI with display).

- [ ] **Step 5: Commit**

```bash
git add dvrrestorer/gui/timeline.py tests/test_timeline.py
git commit -m "feat(gui): TimelineWidget with zone bars + keyframe diamonds"
```

---

### Task 5: PreviewPane

**Files:**
- Create: `dvrrestorer/gui/preview.py`
- Create: `tests/test_preview_pane.py`

**Interfaces:**
- Consumes: nothing new (just numpy arrays + Pillow).
- Produces:
  - `class PreviewPane(ctk.CTkFrame)`:
    - `__init__(parent, on_toggle_before_after: Callable[[bool], None] | None = None)`
    - `set_frame(bgr: np.ndarray, is_before: bool = False) -> None`
    - `set_overlay_text(text: str) -> None`
    - `current_shape() -> tuple[int, int] | None` (height, width)
    - `is_showing_before() -> bool`

The pane holds a `CTkLabel` whose `image` is a `CTkImage` built from the BGR array (convert to RGB, wrap in PIL.Image, pass to CTkImage). A `CTkButton` labelled "Before/After" toggles `_showing_before` and fires `on_toggle_before_after(new_state)` — the App listens and re-renders with `render_original` instead of `render` while active. The current mode is drawn as a corner badge via `set_overlay_text` ("BEFORE" / "AFTER").

- [ ] **Step 1: Write failing tests**

```python
# tests/test_preview_pane.py
import pytest
import numpy as np
pytestmark = pytest.mark.gui


def test_preview_pane_sets_shape():
    import customtkinter as ctk
    from dvrrestorer.gui.preview import PreviewPane
    root = ctk.CTk()
    pane = PreviewPane(root)
    f = np.zeros((270, 480, 3), dtype=np.uint8)
    pane.set_frame(f)
    assert pane.current_shape() == (270, 480)
    root.destroy()


def test_preview_pane_before_after_toggle_fires_callback():
    import customtkinter as ctk
    from dvrrestorer.gui.preview import PreviewPane
    root = ctk.CTk()
    seen = []
    pane = PreviewPane(root, on_toggle_before_after=seen.append)
    pane._toggle()  # internal test hook
    assert seen == [True]
    pane._toggle()
    assert seen == [True, False]
    root.destroy()
```

- [ ] **Step 2: Run tests** → FAIL.

- [ ] **Step 3: Implement `PreviewPane`.**

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add dvrrestorer/gui/preview.py tests/test_preview_pane.py
git commit -m "feat(gui): PreviewPane with CTkImage frame display"
```

---

### Task 6: App Main Window + Wiring

**Files:**
- Create: `dvrrestorer/gui/app.py`
- Create: `tests/test_app.py`
- Modify: `pyproject.toml` — add `dvrrestorer-gui = "dvrrestorer.gui.app:main"` under `[project.scripts]`; add `customtkinter>=5.2` + `Pillow>=10` to `dependencies`.

**Interfaces:**
- Consumes: all previous tasks + `core.scan`, `core.zone_detector`, `core.osd`, `core.pipeline`.
- Produces:
  - `class App(ctk.CTk)`:
    - `__init__()`
    - `open_clip(path: Path) -> None` — kicks off scan+osd thread, populates timeline when done
    - `export(out_path: Path) -> None` — runs `pipeline.run` in a thread, shows progress in a status bar
    - test hooks: `_state: ProjectState`, `_scan_done_event: threading.Event`
  - `main(argv: list[str] | None = None) -> int`

Wiring, in `App.__init__`:
- Menu bar: Open / Save Profile / Load Profile / Export / Quit
- Layout (CTkFrame grid):
  ```
  ┌───────────────────────────────┬───────────┐
  │                               │ ParamsPane│
  │        PreviewPane            │           │
  │                               │           │
  ├───────────────────────────────┴───────────┤
  │               TimelineWidget              │
  └───────────────────────────────────────────┘
  ```
- `TimelineWidget.on_time_scrub(t_sec)` → compute frame_idx = t_sec * fps, call `PreviewRenderer.render(...)` with current keyframes, `PreviewPane.set_frame(result)`.
- `PreviewPane.on_toggle_before_after(showing_before)` → re-render the current frame via `render_original` when True, `render(...)` when False.
- `ParamsPane.on_change(values)` → update the selected keyframe in `_state.keyframes`, re-render preview, redraw timeline.
- `TimelineWidget.on_keyframe_select(i, track)` → push its values into `ParamsPane.set_values(...)`.
- `TimelineWidget.on_keyframe_move(i, t)` → mutate `_state.keyframes[i].t_sec`, redraw.
- `TimelineWidget.on_keyframe_delete(i)` → pop from `_state.keyframes`, redraw.
- Open menu → `open_clip(path)`: spawn a worker thread that runs `scan` + `detect_zones` + `auto_keyframes` + `detect_osd_mask`, then marshals back to the Tk main loop via `self.after(0, ...)` to populate the widgets.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_app.py
import pytest
import threading
pytestmark = pytest.mark.gui


def test_app_smoke_create_and_destroy():
    from dvrrestorer.gui.app import App
    app = App()
    app.update_idletasks()
    app.destroy()


def test_app_scrub_before_scan_done(synth_clip_noisy):
    """Review-focus #1: scrubbing before scan completes must render."""
    from dvrrestorer.gui.app import App
    app = App()
    # Simulate "clip loaded, but scan not done": set state manually.
    from dvrrestorer.gui.project import ProjectState
    from dvrrestorer.core.video_io import probe
    info = probe(synth_clip_noisy)
    app._state = ProjectState(
        input_path=synth_clip_noisy, duration_s=info.frame_count / info.fps,
        fps=info.fps, width=info.width, height=info.height,
        keyframes=[], direction_deg=0.0, osd_mask_path=None,
    )
    app._init_renderer()   # test hook: builds PreviewRenderer from state
    app.on_scrub(2.0)      # test hook equivalent of TimelineWidget callback
    assert app._preview.current_shape() is not None
    app.destroy()


def test_app_export_without_input_errors(tmp_path):
    """Review-focus #2."""
    from dvrrestorer.gui.app import App
    app = App()
    rc = app.export(tmp_path / "out.mp4")   # returns status code
    assert rc != 0
    app.destroy()
```

- [ ] **Step 2: Run tests** → FAIL.

- [ ] **Step 3: Implement `App` and `main`.**

- `App.open_clip(path)`:
  1. `probe(path)` on the main thread (cheap), update `_state`.
  2. Launch `threading.Thread(target=self._scan_worker, daemon=True)`.
  3. `_scan_worker`: run scan → zone_detector → auto_keyframes → detect_osd_mask. When done, `self.after(0, lambda: self._apply_scan_result(...))` populates widgets.
- `App.on_scrub(t_sec)`:
  1. Guard: if `_state.input_path is None`, no-op.
  2. frame_idx = int(t_sec * _state.fps); clamp.
  3. `PreviewRenderer.render(frame_idx, _state.keyframes, _state.direction_deg)` → `_preview.set_frame`.
- `App.export(out_path)`:
  1. Guard: if `_state.input_path is None`, show messagebox, return non-zero.
  2. Launch export thread; progress via `after(0, status_bar.set)`.
  3. Return 0.
- `main(argv)`: parse an optional positional `--open PATH`; create `App`; optionally `app.after(100, app.open_clip, Path(argv.open))`; `app.mainloop()`.

- [ ] **Step 4: Add `dvrrestorer-gui` to `pyproject.toml` scripts; add `customtkinter`, `Pillow` to dependencies; `pip install -e .`**

- [ ] **Step 5: Run tests** → PASS.

- [ ] **Step 6: Manual smoke:** `dvrrestorer-gui` opens, File→Open a sample clip, scrub, adjust a slider, File→Export writes `.mp4`.

- [ ] **Step 7: Commit**

```bash
git add dvrrestorer/gui/app.py tests/test_app.py pyproject.toml
git commit -m "feat(gui): main App wiring preview↔timeline↔params + export"
```

---

## Final Verification

- [ ] Full suite: `pytest -v -m "not gui" && pytest -v -m gui` — all green.
- [ ] Confirm all five Review-Focus pin tests exist and pass.
- [ ] Manual smoke: `dvrrestorer-gui` on the real YouTube-sourced FPV clip — open, scrub, tune, export. Compare output to the CLI result.
- [ ] Tag `v0.2.0 — GUI`.

Next after this plan: Phase 3 (dropout detection + ML denoiser backend wiring), separate plan.
