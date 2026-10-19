def test_detect_zones_on_noisy_clip(synth_clip_noisy):
    from dvrrestorer.core.scan import scan
    from dvrrestorer.core.zone_detector import detect_zones
    rows = scan(synth_clip_noisy, sample_every=3)
    zones = detect_zones(rows)
    assert len(zones) >= 1
    noisy_zone = next((z for z in zones if 3.0 <= z.t_start <= 5.5), None)
    assert noisy_zone is not None
    assert noisy_zone.dominant in {"chroma_noise", "mixed"}


def test_auto_keyframes_count_bounded(synth_clip_noisy):
    from dvrrestorer.core.scan import scan
    from dvrrestorer.core.zone_detector import detect_zones, auto_keyframes
    rows = scan(synth_clip_noisy, sample_every=3)
    zones = detect_zones(rows)
    kfs = auto_keyframes(zones, clip_duration_s=10.0)
    assert len(kfs) <= 20
    assert any(3.0 <= k.t_sec <= 5.5 and k.chroma > 0.3 for k in kfs)


def test_zone_detector_short_clip_with_problem():
    """Review-focus #4: a short clip where ALL frames are problem frames
    must still produce usable bookend keyframes without crashing. (If a
    sub-min-zone-length problem exists, we fall back to the bookends;
    this still gives interpolate() a valid two-point curve.)
    """
    from dvrrestorer.core.scan import FrameMetrics
    from dvrrestorer.core.zone_detector import detect_zones, auto_keyframes
    rows = []
    for i in range(10):
        rows.append(FrameMetrics(
            frame=i, t_sec=i * 0.08, mean_lum=20.0, chroma_noise=8.0,
            row_streak=1.0, sharpness=100.0, black_ratio=0.0,
            white_ratio=0.0, freeze_mad=None,
        ))
    # 0.72s clip, every frame noisy → the one found zone is shorter than
    # the 1.0s min, so detect_zones drops it; auto_keyframes still emits
    # bookends so interpolation stays well-defined.
    zones = detect_zones(rows, min_zone_len_s=1.0)
    kfs = auto_keyframes(zones, clip_duration_s=0.72)
    assert len(kfs) >= 2
    ts = [k.t_sec for k in kfs]
    assert ts == sorted(ts)
    assert ts[0] == 0.0 and ts[-1] == 0.72


def test_zone_detector_short_clip_with_min_relaxed_detects():
    """When min_zone_len_s is set tight enough, a short noisy region is
    actually detected and a keyframe sits inside it."""
    from dvrrestorer.core.scan import FrameMetrics
    from dvrrestorer.core.zone_detector import detect_zones, auto_keyframes
    rows = []
    # Quiet start (14 frames), noisy end (6 frames). The 70th percentile
    # of chroma_noise is 0.1, but detect_zones floors the threshold at 0.5
    # so the 10.0-valued tail is cleanly flagged.
    for i in range(20):
        rows.append(FrameMetrics(
            frame=i, t_sec=i * 0.1, mean_lum=100.0,
            chroma_noise=10.0 if i >= 14 else 0.1,
            row_streak=0.5, sharpness=200.0, black_ratio=0.0,
            white_ratio=0.0, freeze_mad=None,
        ))
    zones = detect_zones(rows, min_zone_len_s=0.3)
    assert zones, "expected to detect the noise tail"
    kfs = auto_keyframes(zones, clip_duration_s=1.9)
    assert any(k.chroma > 0.4 for k in kfs)
