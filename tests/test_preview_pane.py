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
