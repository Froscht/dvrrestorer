"""Zone detection and automatic keyframe placement.

Takes the list of FrameMetrics from scan() and segments it into
contiguous zones where one of three problem metrics (chroma_noise,
row_streak, low luma) is dominant. Then auto_keyframes converts those
zones into a list of initial Keyframes the user can tune.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from dvrrestorer.core.pipeline import Keyframe
from dvrrestorer.core.scan import FrameMetrics


METRIC_NAMES = ("chroma_noise", "row_streak", "low_lum")


@dataclass
class Zone:
    t_start: float
    t_end: float
    dominant: str
    metrics_summary: dict[str, float] = field(default_factory=dict)


def _problem_flags(metrics: list[FrameMetrics]) -> dict[str, np.ndarray]:
    chroma = np.array([m.chroma_noise for m in metrics])
    streak = np.array([m.row_streak for m in metrics])
    lum = np.array([m.mean_lum for m in metrics])
    # Thresholds relative to the clip itself (robust to synthetic / real scale).
    chroma_t = np.percentile(chroma, 70) if chroma.size else 0.0
    streak_t = np.percentile(streak, 70) if streak.size else 0.0
    # "Low lum" uses the inverted percentile: frames below the 30th percentile
    # are dark-flagged.
    lum_t = np.percentile(lum, 30) if lum.size else 0.0
    return {
        "chroma_noise": chroma > max(chroma_t, 0.5),
        "row_streak": streak > max(streak_t, 0.5),
        "low_lum": lum < max(lum_t, 1.0),
    }


def detect_zones(
    metrics: list[FrameMetrics],
    min_zone_len_s: float = 1.0,
    merge_gap_s: float = 0.5,
) -> list[Zone]:
    """Return problem zones sorted by t_start."""
    if not metrics:
        return []
    flags = _problem_flags(metrics)
    any_flag = flags["chroma_noise"] | flags["row_streak"] | flags["low_lum"]

    zones: list[tuple[float, float, list[int]]] = []  # (t0, t1, indices)
    in_zone = False
    start_i = 0
    for i, is_prob in enumerate(any_flag):
        if is_prob and not in_zone:
            in_zone = True
            start_i = i
        elif not is_prob and in_zone:
            in_zone = False
            idxs = list(range(start_i, i))
            zones.append((metrics[start_i].t_sec, metrics[i - 1].t_sec, idxs))
    if in_zone:
        idxs = list(range(start_i, len(metrics)))
        zones.append((metrics[start_i].t_sec, metrics[-1].t_sec, idxs))

    # Merge gaps shorter than merge_gap_s
    merged: list[tuple[float, float, list[int]]] = []
    for z in zones:
        if merged and z[0] - merged[-1][1] <= merge_gap_s:
            merged[-1] = (merged[-1][0], z[1], merged[-1][2] + z[2])
        else:
            merged.append(z)

    # Drop zones shorter than min_zone_len_s
    kept = [z for z in merged if (z[1] - z[0]) >= min_zone_len_s]

    out: list[Zone] = []
    for t0, t1, idxs in kept:
        chroma = np.mean([metrics[i].chroma_noise for i in idxs])
        streak = np.mean([metrics[i].row_streak for i in idxs])
        lum = np.mean([metrics[i].mean_lum for i in idxs])
        # Dominance: whichever flag share is highest (relative strength).
        scores = {
            "chroma_noise": np.mean(flags["chroma_noise"][idxs]),
            "row_streak": np.mean(flags["row_streak"][idxs]),
            "low_lum": np.mean(flags["low_lum"][idxs]),
        }
        top = max(scores.values())
        tops = [k for k, v in scores.items() if v == top]
        dominant = tops[0] if len(tops) == 1 else "mixed"
        out.append(Zone(
            t_start=float(t0), t_end=float(t1),
            dominant=dominant,
            metrics_summary={
                "chroma_noise": float(chroma),
                "row_streak": float(streak),
                "mean_lum": float(lum),
            },
        ))
    return out


def auto_keyframes(zones: list[Zone], clip_duration_s: float, max_keyframes: int = 20) -> list[Keyframe]:
    """Generate initial keyframes from zones.

    Always emits two bookend keyframes at t=0 and t=clip_duration_s with
    low defaults (0.2 each) so interpolation outside any zone is well-
    defined. Per zone, one midpoint keyframe with strengths derived from
    the summary metrics.
    """
    kfs: list[Keyframe] = []
    default_dark = 0.2
    default_lines = 0.2
    default_chroma = 0.2

    kfs.append(Keyframe(0.0, default_dark, default_lines, default_chroma))

    zone_kfs: list[tuple[float, Keyframe]] = []
    for z in zones:
        t_mid = (z.t_start + z.t_end) / 2.0
        chroma_s = float(np.clip(z.metrics_summary.get("chroma_noise", 0.0) / 10.0, 0.2, 1.0))
        lines_s = float(np.clip(z.metrics_summary.get("row_streak", 0.0) / 10.0, 0.2, 1.0))
        lum = z.metrics_summary.get("mean_lum", 128.0)
        dark_s = float(np.clip((80.0 - lum) / 80.0, 0.2, 1.0))
        # Boost the dominant track
        if z.dominant == "chroma_noise":
            chroma_s = max(chroma_s, 0.5)
        elif z.dominant == "row_streak":
            lines_s = max(lines_s, 0.5)
        elif z.dominant == "low_lum":
            dark_s = max(dark_s, 0.5)
        strength_sum = chroma_s + lines_s + dark_s
        zone_kfs.append((strength_sum, Keyframe(t_mid, dark_s, lines_s, chroma_s)))

    # Keep the strongest zones until budget is reached
    zone_kfs.sort(key=lambda t: -t[0])
    budget = max_keyframes - 2  # bookends
    for _, k in zone_kfs[: max(0, budget)]:
        kfs.append(k)

    kfs.append(Keyframe(float(clip_duration_s), default_dark, default_lines, default_chroma))
    kfs.sort(key=lambda k: k.t_sec)
    return kfs
