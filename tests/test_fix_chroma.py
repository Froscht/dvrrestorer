import cv2
import numpy as np


def test_fix_chroma_zero_strength_identity():
    from dvrrestorer.core.fix_chroma import fix_chroma
    f = (np.random.rand(50, 80, 3) * 255).astype(np.uint8)
    out = fix_chroma(f, [], [], None, strength=0.0)
    assert (out == f).all()


def test_fix_chroma_reduces_chroma_std_on_noisy_input():
    from dvrrestorer.core.fix_chroma import fix_chroma
    from scripts.synthesize_dvr_artifacts import add_chroma_noise
    np.random.seed(0)
    base = np.full((80, 100, 3), 128, dtype=np.uint8)
    noisy = add_chroma_noise(base, strength=0.7)
    cleaned = fix_chroma(noisy, [], [], None, strength=0.7)

    def chroma_std(img):
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return yuv[:, :, 1].std() + yuv[:, :, 2].std()

    assert chroma_std(cleaned) < chroma_std(noisy) * 0.6


def test_warp_to_aligns_shifted_source_to_target():
    """Direct test of _warp_to: a horizontally-shifted source warps back
    toward the target. Reproduces the Critical review finding — flow
    direction swap would move the source further away, not toward.
    """
    import cv2
    import numpy as np
    from dvrrestorer.core.fix_chroma import _warp_to
    rng = np.random.default_rng(0)
    base = rng.integers(0, 256, size=(80, 120, 3), dtype=np.uint8)
    shifted = np.roll(base, shift=5, axis=1)
    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    shifted_gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
    flow_obj = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    warped = _warp_to(target_gray=base_gray, src_bgr=shifted,
                      src_gray=shifted_gray, flow_obj=flow_obj)
    center = slice(15, 65), slice(20, 100)
    diff_warped = np.abs(warped[center].astype(int) - base[center].astype(int)).mean()
    diff_raw = np.abs(shifted[center].astype(int) - base[center].astype(int)).mean()
    assert diff_warped < diff_raw * 0.5
