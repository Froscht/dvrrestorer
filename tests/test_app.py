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
    from dvrrestorer.gui.project import ProjectState
    from dvrrestorer.core.video_io import probe
    app = App()
    info = probe(synth_clip_noisy)
    app._state = ProjectState(
        input_path=synth_clip_noisy, duration_s=info.frame_count / info.fps,
        fps=info.fps, width=info.width, height=info.height,
        keyframes=[], direction_deg=0.0, osd_mask_path=None,
    )
    app._init_renderer()
    app.on_scrub(2.0)
    assert app._preview.current_shape() is not None
    app.destroy()


def test_app_export_without_input_errors(tmp_path):
    """Review-focus #2."""
    from dvrrestorer.gui.app import App
    app = App()
    rc = app.export(tmp_path / "out.mp4")
    assert rc != 0
    app.destroy()
