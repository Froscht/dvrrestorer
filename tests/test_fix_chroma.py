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
    base = np.full((80, 100, 3), 128, dtype=np.uint8)
    noisy = add_chroma_noise(base, strength=0.7)
    cleaned = fix_chroma(noisy, [], [], None, strength=0.7)

    def chroma_std(img):
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        return yuv[:, :, 1].std() + yuv[:, :, 2].std()

    assert chroma_std(cleaned) < chroma_std(noisy) * 0.5
