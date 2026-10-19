"""Headless CLI entry point.

Usage:
    dvrrestorer-cli INPUT [--out PATH] [--preset PATH | --auto]
                           [--compare] [--chunk-size N] [--direction-deg DEG]

Exactly one of --preset or --auto must be given. --auto runs scan +
zone_detector + auto_keyframes to build a default timeline; --preset
loads a saved profile.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dvrrestorer.core import pipeline, scan as scan_mod, video_io, zone_detector
from dvrrestorer.core.pipeline import PipelineConfig
from dvrrestorer.profiles import load_profile


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dvrrestorer-cli",
                                description="Restore analog FPV DVR footage.")
    p.add_argument("input", help="input video path")
    p.add_argument("--out", help="output cleaned video path (default: <input_stem>_cleaned.mp4)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--auto", action="store_true",
                       help="auto-detect zones and place keyframes")
    group.add_argument("--preset", help="path to a saved profile JSON")
    p.add_argument("--compare", action="store_true",
                   help="also write a side-by-side <out>.compare.mp4")
    p.add_argument("--chunk-size", type=int, default=32)
    p.add_argument("--direction-deg", type=float, default=0.0,
                   help="angle in degrees of the dominant stripe direction")
    return p


def _progress(current: int, total: int) -> None:
    if total <= 0:
        return
    if current == 1 or current == total or current % 30 == 0:
        print(f"  frame {current}/{total}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"error: input does not exist: {in_path}", file=sys.stderr)
        return 2

    out_path = Path(args.out) if args.out else in_path.with_name(
        in_path.stem + "_cleaned.mp4"
    )
    if out_path.resolve() == in_path.resolve():
        print("error: --out must not point to the input file", file=sys.stderr)
        return 2

    direction_deg = float(args.direction_deg)

    if args.preset:
        _name, keyframes, saved_dir = load_profile(Path(args.preset))
        if saved_dir and direction_deg == 0.0:
            direction_deg = saved_dir
    else:
        print("scanning...", file=sys.stderr)
        info = video_io.probe(in_path)
        metrics = scan_mod.scan(in_path, sample_every=3)
        zones = zone_detector.detect_zones(metrics)
        clip_duration = info.frame_count / info.fps if info.fps > 0 else 0.0
        keyframes = zone_detector.auto_keyframes(zones, clip_duration_s=clip_duration)
        print(f"  {len(zones)} zone(s), {len(keyframes)} keyframe(s)", file=sys.stderr)

    cfg = PipelineConfig(
        keyframes=keyframes,
        chunk_size=int(args.chunk_size),
        direction_deg=direction_deg,
    )

    print(f"restoring → {out_path}", file=sys.stderr)
    pipeline.run(in_path, out_path, cfg, progress_cb=_progress)
    print("done.", file=sys.stderr)

    if args.compare:
        cmp_path = out_path.with_name(out_path.stem + ".compare.mp4")
        print(f"writing compare → {cmp_path}", file=sys.stderr)
        video_io.stack_compare(in_path, out_path, cmp_path)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
