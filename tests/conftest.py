"""Shared pytest fixtures.

Synthetic test clips are generated lazily on first use and reused across
the session (large clips take a second or two to render, so we don't
rebuild per test).
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def synth_clip_clean(tmp_path_factory):
    """10s, 30fps, 960x540 clean synthetic clip."""
    from scripts.synthesize_dvr_artifacts import make_clean_clip
    d = tmp_path_factory.mktemp("synth")
    out = d / "clean.mp4"
    make_clean_clip(out, duration_s=10.0, fps=30.0, size=(960, 540))
    return out


@pytest.fixture(scope="session")
def synth_clip_noisy(tmp_path_factory):
    """10s synthetic clip with chroma noise 3-5s and streaks 6-8s."""
    from scripts.synthesize_dvr_artifacts import make_artifacted_clip
    d = tmp_path_factory.mktemp("synth")
    out = d / "noisy.mp4"
    make_artifacted_clip(
        out,
        duration_s=10.0,
        fps=30.0,
        size=(960, 540),
        ranges=[
            (3.0, 5.0, 0.6, 0.0, 0.0),  # chroma noise band
            (6.0, 8.0, 0.0, 0.6, 0.0),  # RF streak band
        ],
    )
    return out
