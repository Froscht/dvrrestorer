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


def test_zone_detector_short_clip():
    """Review-focus #4: clip shorter than min zone length."""
    from dvrrestorer.core.scan import FrameMetrics
    from dvrrestorer.core.zone_detector import detect_zones, auto_keyframes
    rows = [
        FrameMetrics(
            frame=i, t_sec=i * 0.1, mean_lum=50.0, chroma_noise=1.0,
            row_streak=1.0, sharpness=100.0, black_ratio=0.0,
            white_ratio=0.0, freeze_mad=None,
        )
        for i in range(10)
    ]  # 1-second clip
    zones = detect_zones(rows, min_zone_len_s=1.0)
    kfs = auto_keyframes(zones, clip_duration_s=1.0)
    assert len(kfs) >= 2
