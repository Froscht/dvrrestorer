def test_interpolate_at_keyframe_exact():
    from dvrrestorer.core.pipeline import Keyframe, interpolate
    kfs = [Keyframe(0.0, 0.1, 0.2, 0.3), Keyframe(2.0, 0.5, 0.6, 0.7)]
    assert interpolate(kfs, 0.0) == (0.1, 0.2, 0.3)
    assert interpolate(kfs, 2.0) == (0.5, 0.6, 0.7)


def test_interpolate_linear_midpoint():
    from dvrrestorer.core.pipeline import Keyframe, interpolate
    kfs = [Keyframe(0.0, 0.0, 0.0, 0.0), Keyframe(2.0, 1.0, 1.0, 1.0)]
    assert interpolate(kfs, 1.0) == (0.5, 0.5, 0.5)


def test_pipeline_keyframe_edge_cases():
    """Review-focus #5: duplicate t_sec, outside-range."""
    from dvrrestorer.core.pipeline import Keyframe, interpolate
    kfs = [Keyframe(1.0, 0.3, 0.3, 0.3), Keyframe(2.0, 0.7, 0.7, 0.7)]
    assert interpolate(kfs, 0.0) == (0.3, 0.3, 0.3)
    assert interpolate(kfs, 5.0) == (0.7, 0.7, 0.7)
    kfs2 = [Keyframe(1.0, 0.1, 0.1, 0.1), Keyframe(1.0, 0.9, 0.9, 0.9)]
    assert interpolate(kfs2, 1.0) == (0.9, 0.9, 0.9)


def test_pipeline_runs_end_to_end(synth_clip_noisy, tmp_path):
    from dvrrestorer.core.pipeline import run, PipelineConfig, Keyframe
    out = tmp_path / "cleaned.mp4"
    cfg = PipelineConfig(keyframes=[Keyframe(0.0, 0.5, 0.5, 0.5)])
    run(synth_clip_noisy, out, cfg)
    assert out.exists() and out.stat().st_size > 1000
    from dvrrestorer.core.video_io import probe
    assert probe(out).frame_count == probe(synth_clip_noisy).frame_count


def test_pipeline_streams_without_ram_blowup(synth_clip_noisy, tmp_path):
    """Review-finding Critical #1: pipeline must process chunk-wise,
    not load the entire clip into memory. For a 300-frame 960x540
    noisy clip the peak traced RAM must stay well under the ~470 MB
    needed to hold every decoded frame in a list.
    """
    import tracemalloc
    from dvrrestorer.core.pipeline import run, PipelineConfig, Keyframe
    out = tmp_path / "cleaned_stream.mp4"
    cfg = PipelineConfig(
        keyframes=[Keyframe(0.0, 0.2, 0.2, 0.2)],
        chunk_size=32,
    )
    tracemalloc.start()
    run(synth_clip_noisy, out, cfg)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 120 * 1024 * 1024, f"peak traced={peak/1e6:.0f}MB"
