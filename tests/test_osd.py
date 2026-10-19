import numpy as np


def test_detect_osd_mask_finds_painted_osd(synth_clip_with_osd):
    from dvrrestorer.core.osd import detect_osd_mask
    mask = detect_osd_mask(synth_clip_with_osd)
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0
    h, w = mask.shape
    tr = mask[: h // 2, w // 2 :]
    rest = mask.copy()
    rest[: h // 2, w // 2 :] = 0
    assert tr.mean() > rest.mean() * 3


def test_apply_soft_dilation_grows_and_feathers():
    from dvrrestorer.core.osd import apply_soft_dilation
    bm = np.zeros((100, 100), dtype=bool)
    bm[40:60, 40:60] = True
    soft = apply_soft_dilation(bm, dilate_px=3, blur_sigma=1.5)
    assert soft.shape == (100, 100)
    assert soft[50, 50] > 0.95
    assert soft[0, 0] < 0.05
    # Edge pixel should fall between the two.
    edge = soft[62, 50]
    assert 0.0 < edge < 1.0
