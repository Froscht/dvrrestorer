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
