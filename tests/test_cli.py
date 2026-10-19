import pytest


def test_cli_auto_produces_output(synth_clip_noisy, tmp_path):
    from dvrrestorer.cli import main
    out = tmp_path / "cleaned.mp4"
    rc = main([str(synth_clip_noisy), "--out", str(out), "--auto"])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 1000


def test_cli_preset_mutually_exclusive_with_auto(synth_clip_noisy, tmp_path):
    from dvrrestorer.cli import main
    with pytest.raises(SystemExit):
        main([str(synth_clip_noisy), "--preset", "x.json", "--auto"])


def test_cli_rejects_output_equal_to_input(synth_clip_noisy):
    """--out == input would overwrite the source before remux completes."""
    from dvrrestorer.cli import main
    rc = main([str(synth_clip_noisy), "--out", str(synth_clip_noisy), "--auto"])
    assert rc != 0
