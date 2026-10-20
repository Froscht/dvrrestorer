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
