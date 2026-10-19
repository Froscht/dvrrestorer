"""Profile JSON persistence (timeline snapshots)."""

from __future__ import annotations

import json
from pathlib import Path

from dvrrestorer.core.pipeline import Keyframe

SCHEMA_VERSION = 1


def save_profile(
    path: Path,
    name: str,
    keyframes: list[Keyframe],
    direction_deg: float = 0.0,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": SCHEMA_VERSION,
        "name": name,
        "direction_deg": float(direction_deg),
        "keyframes": [
            {
                "t_sec": float(k.t_sec),
                "dark": float(k.dark),
                "lines": float(k.lines),
                "chroma": float(k.chroma),
            }
            for k in keyframes
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_profile(path: Path) -> tuple[str, list[Keyframe], float]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    name = str(data.get("name", path.stem))
    direction_deg = float(data.get("direction_deg", 0.0))
    kfs = [
        Keyframe(
            t_sec=float(k["t_sec"]),
            dark=float(k["dark"]),
            lines=float(k["lines"]),
            chroma=float(k["chroma"]),
        )
        for k in data.get("keyframes", [])
    ]
    return name, kfs, direction_deg


def list_profiles(dir_path: Path) -> list[str]:
    """Return sorted profile names found in `dir_path`."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return []
    names: list[str] = []
    for p in sorted(dir_path.glob("*.json")):
        try:
            name, _, _ = load_profile(p)
            names.append(name)
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return names
