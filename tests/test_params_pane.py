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


def test_params_pane_fires_on_slider_release():
    import customtkinter as ctk
    from dvrrestorer.gui.params import ParamsPane
    root = ctk.CTk()
    seen = []
    pane = ParamsPane(root, on_change=seen.append)
    pane.set_values(dark=0.4, lines=0.5, chroma=0.6, direction_deg=0.0)
    pane.pack()
    root.update()
    assert not seen
    pane._sliders["dark"]._canvas.event_generate("<ButtonRelease-1>")
    root.update()
    assert seen and seen[-1]["dark"] == 0.4
    root.destroy()
