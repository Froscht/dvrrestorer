def test_scan_clean_clip_has_low_noise_and_streak(synth_clip_clean):
    from dvrrestorer.core.scan import scan
    rows = scan(synth_clip_clean, sample_every=5)
    assert len(rows) > 0
    assert max(r.chroma_noise for r in rows) < 3.0
    assert max(r.row_streak for r in rows) < 5.0


def test_scan_noisy_clip_detects_peaks(synth_clip_noisy):
    from dvrrestorer.core.scan import scan
    rows = scan(synth_clip_noisy, sample_every=5)
    noisy = [r for r in rows if 3.0 <= r.t_sec <= 5.0]
    quiet = [r for r in rows if r.t_sec < 2.0]
    assert noisy and quiet
    assert max(r.chroma_noise for r in noisy) > 2 * max(r.chroma_noise for r in quiet)
