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


@pytest.fixture(scope="session")
def synth_clip_with_osd(tmp_path_factory):
    """3s clip with a persistent OSD on a smoothly moving gradient.

    Gradient has no stable edges; only the OSD provides consistently-placed
    edges, so edge-stability-over-time pinpoints the OSD.
    """
    import cv2
    import numpy as np
    from scripts.synthesize_dvr_artifacts import draw_fake_osd
    d = tmp_path_factory.mktemp("synth")
    out = d / "with_osd.mp4"
    w, h = 480, 270
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(90):  # 3s
        shift = int((i * 2) % w)
        x = (np.arange(w) + shift) % w
        grad = (x * 255 // w).astype(np.uint8)
        bg = np.broadcast_to(grad, (h, w)).copy()
        bg = np.stack([bg, bg, bg], axis=-1)
        frame = draw_fake_osd(bg, text="15.5V", pos=(0.75, 0.3))
        writer.write(frame)
    writer.release()
    return out
