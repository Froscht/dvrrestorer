import numpy as np


def test_fix_lines_zero_strength_identity():
    from dvrrestorer.core.fix_lines import fix_lines
    f = (np.random.rand(60, 80, 3) * 255).astype(np.uint8)
    out = fix_lines(f, None, strength=0.0)
    assert (out == f).all()


def test_fix_lines_reduces_horizontal_stripe_energy():
    from dvrrestorer.core.fix_lines import fix_lines
    from scripts.synthesize_dvr_artifacts import add_rf_streaks
    base = np.full((120, 160, 3), 128, dtype=np.uint8)
    noisy = add_rf_streaks(base, strength=0.8, angle_deg=0.0)
    cleaned = fix_lines(noisy, None, strength=0.7, direction_deg=0.0)

    def streak_energy(img):
        g = img[:, :, 1].astype(np.float32)
        F = np.fft.fft2(g)
        return float(np.abs(F[1:5, :]).sum())

    assert streak_energy(cleaned) < streak_energy(noisy) * 0.5
