"""GUI project state and persistence (superset of the profile schema)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from dvrrestorer.core.pipeline import Keyframe
from dvrrestorer.profiles import SCHEMA_VERSION

GUI_PROJECT_VERSION = 1


@dataclass
class ProjectState:
    input_path: Path | None
    duration_s: float
    fps: float
    width: int
    height: int
    keyframes: list[Keyframe] = field(default_factory=list)
    direction_deg: float = 0.0
    osd_mask_path: Path | None = None

    @classmethod
    def default(cls) -> ProjectState:
        return cls(
            input_path=None, duration_s=0.0, fps=0.0, width=0, height=0,
            keyframes=[], direction_deg=0.0, osd_mask_path=None,
        )


def save_project(state: ProjectState, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "gui_project_version": GUI_PROJECT_VERSION,
        "version": SCHEMA_VERSION,
        "name": path.stem,
        "input_path": None if state.input_path is None else str(state.input_path),
        "duration_s": float(state.duration_s),
        "fps": float(state.fps),
        "width": int(state.width),
        "height": int(state.height),
        "direction_deg": float(state.direction_deg),
        "osd_mask_path": (
            None if state.osd_mask_path is None else str(state.osd_mask_path)
        ),
        "keyframes": [
            {
                "t_sec": float(k.t_sec),
                "dark": float(k.dark),
                "lines": float(k.lines),
                "chroma": float(k.chroma),
            }
            for k in state.keyframes
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_project(path: Path) -> ProjectState:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ip = data.get("input_path")
    mp = data.get("osd_mask_path")
    return ProjectState(
        input_path=None if ip is None else Path(ip),
        duration_s=float(data.get("duration_s", 0.0)),
        fps=float(data.get("fps", 0.0)),
        width=int(data.get("width", 0)),
        height=int(data.get("height", 0)),
        keyframes=[
            Keyframe(
                t_sec=float(k["t_sec"]),
                dark=float(k["dark"]),
                lines=float(k["lines"]),
                chroma=float(k["chroma"]),
            )
            for k in data.get("keyframes", [])
        ],
        direction_deg=float(data.get("direction_deg", 0.0)),
        osd_mask_path=None if mp is None else Path(mp),
    )
