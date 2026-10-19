import cv2
import numpy as np


def test_fix_dark_zero_strength_is_identity():
    from dvrrestorer.core.fix_dark import fix_dark
    f = (np.random.rand(50, 80, 3) * 100).astype(np.uint8)
    out = fix_dark(f, None, strength=0.0)
    assert (out == f).all()


def test_fix_dark_increases_mean_luminance_for_dark_input():
    from dvrrestorer.core.fix_dark import fix_dark
    f = np.full((50, 80, 3), 20, dtype=np.uint8)
    out = fix_dark(f, None, strength=0.8)
    assert cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean() > 20 + 20


def test_fix_dark_respects_osd_mask():
    from dvrrestorer.core.fix_dark import fix_dark
    f = np.full((50, 80, 3), 20, dtype=np.uint8)
    mask = np.zeros((50, 80), dtype=np.float32)
    mask[:10, :10] = 1.0  # top-left is OSD, must stay 20
    out = fix_dark(f, mask, strength=0.8)
    assert out[5, 5].mean() == 20
    assert out[40, 70].mean() > 25
