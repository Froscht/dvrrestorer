"""Capture screenshots of the DVR Restorer GUI for the README / docs.

Reproducible fallback shots: generates a synthetic artifacted clip, boots
the GUI, waits for the automatic scan to finish so zone bars and
auto-keyframes are populated, then grabs three PNGs:

  docs/screenshots/gui_full.png     — the whole window
  docs/screenshots/gui_preview.png  — preview pane only
  docs/screenshots/gui_timeline.png — timeline strip only

Run from the repo root:  python -m scripts.make_screenshots

Replace the generated PNGs with real FPV captures before publishing — the
synthetic clip is deliberately ugly for test coverage.
"""

from __future__ import annotations

from pathlib import Path

from PIL import ImageGrab

from dvrrestorer.gui.app import App
from scripts.synthesize_dvr_artifacts import make_artifacted_clip


REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "docs" / "screenshots"
FIXTURE = OUT_DIR / "_fixture.mp4"
SCRUB_TIME_S = 3.0
SETTLE_MS = 400
POLL_MS = 100
SCAN_TIMEOUT_MS = 30_000


def _ensure_fixture() -> Path:
    if FIXTURE.exists():
        return FIXTURE
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    make_artifacted_clip(
        FIXTURE,
        duration_s=8.0,
        fps=30.0,
        size=(960, 540),
        ranges=[
            (0.0, 2.5, 0.0, 0.0, 0.8),
            (2.5, 5.0, 0.0, 0.7, 0.0),
            (5.0, 8.0, 0.8, 0.0, 0.0),
        ],
    )
    return FIXTURE


def _grab_widget(widget, path: Path) -> None:
    widget.update_idletasks()
    x = widget.winfo_rootx()
    y = widget.winfo_rooty()
    w = widget.winfo_width()
    h = widget.winfo_height()
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    print(f"wrote {path.relative_to(REPO_ROOT)}  ({w}x{h})")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixture = _ensure_fixture()

    app = App()
    deadline = {"left": SCAN_TIMEOUT_MS}

    def do_capture() -> None:
        _grab_widget(app, OUT_DIR / "gui_full.png")
        _grab_widget(app._preview, OUT_DIR / "gui_preview.png")
        _grab_widget(app._timeline, OUT_DIR / "gui_timeline.png")
        app.destroy()

    def after_scan() -> None:
        app.on_scrub(SCRUB_TIME_S)
        app.after(SETTLE_MS, do_capture)

    def wait_for_scan() -> None:
        if app._scan_done_event.is_set():
            after_scan()
            return
        deadline["left"] -= POLL_MS
        if deadline["left"] <= 0:
            print("timeout waiting for scan — capturing anyway")
            after_scan()
            return
        app.after(POLL_MS, wait_for_scan)

    def start() -> None:
        app.open_clip(fixture)
        app.after(POLL_MS, wait_for_scan)

    app.after(100, start)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
