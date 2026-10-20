import numpy as np


def test_preview_renders_frame_at_downscale(synth_clip_noisy):
    from dvrrestorer.gui.preview_cache import PreviewRenderer
    from dvrrestorer.core.pipeline import Keyframe
    r = PreviewRenderer(synth_clip_noisy, osd_mask=None, downscale=0.5)
    out = r.render(100, [Keyframe(0, 0.3, 0.3, 0.3)])
    assert out.dtype == np.uint8
    assert out.shape[0] == 540 // 2  # noisy clip is 540h
    assert out.shape[1] == 960 // 2
    r.close()


def test_preview_cache_hit_is_fast(synth_clip_noisy):
    import time
    from dvrrestorer.gui.preview_cache import PreviewRenderer
    from dvrrestorer.core.pipeline import Keyframe
    r = PreviewRenderer(synth_clip_noisy, osd_mask=None)
    kfs = [Keyframe(0, 0.3, 0.3, 0.3)]
    r.render(100, kfs)
    t0 = time.perf_counter()
    r.render(100, kfs)   # same params -> cache hit
    dt = time.perf_counter() - t0
    assert dt < 0.01
    r.close()


def test_preview_cache_clamps_oversized_index(synth_clip_noisy):
    """Review-focus #4."""
    from dvrrestorer.gui.preview_cache import PreviewRenderer
    from dvrrestorer.core.pipeline import Keyframe
    r = PreviewRenderer(synth_clip_noisy, osd_mask=None)
    out = r.render(999999, [Keyframe(0, 0.3, 0.3, 0.3)])
    assert out is not None and out.ndim == 3
    r.close()
