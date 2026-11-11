# SPDX-License-Identifier: Apache-2.0
"""MuJoCo entry options from the scene's MJCF `<option>` (coordinator ruling at the end of [internal note]).

The MuJoCo entry's kwargs are built in three layers. First the capability defaults. Then the MJCF `<option>` values
named in HONOURED. Last the route's overrides, each with a reason (`alloy.scene.entries.mujoco_overrides`). Every
value taken from a file, and every override of one, is listed in `RouteReport.entry_options`. The files are the scene's
MJCF robots, then its MJCF object cards. For each key the first file that sets it wins, and a disagreement is reported.
noslip_iterations / noslip_tolerance go through alloy.scene.noslip_route. The integrator stays Alloy's choice (the
§3.5 integrator table); the report notes it when a file asks for another.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("alloy.scene")
HONOURED: tuple[str, ...] = ("impratio", "cone", "solver", "iterations", "tolerance")
REPORTED = ("integrator",)
_PARSE: dict[str, Callable[[str], object]] = {"impratio": float, "iterations": lambda v: int(float(v)),
                                              "tolerance": float, "cone": str.lower, "solver": str.lower,
                                              "integrator": str.lower}


@dataclass(frozen=True)
class MjcfOptions:
    values: dict[str, object] = field(default_factory=dict)      # HONOURED keys -> kwarg value
    source: dict[str, str] = field(default_factory=dict)         # key -> file name that set it
    asked: dict[str, tuple[object, str]] = field(default_factory=dict)   # REPORTED keys -> (value, file)
    conflicts: tuple[str, ...] = ()


def read_options(path: str) -> dict[str, object]:
    """HONOURED and REPORTED keys of a file's <option> elements (later elements override earlier, as MuJoCo)."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        log.warning("mjcf options: cannot read %s (%s); none taken from it", path, exc)
        return {}
    out: dict[str, object] = {}
    for opt in root.iter("option"):
        out.update({k: _PARSE[k](v) for k, v in opt.attrib.items() if k in _PARSE})
    return out


def scene_options(paths: Sequence[str]) -> MjcfOptions:
    """Merge the files' options in order: the first file that sets a key wins; disagreements are listed."""
    values: dict[str, object] = {}
    source: dict[str, str] = {}
    asked: dict[str, tuple[object, str]] = {}
    conflicts: list[str] = []
    for path in paths:
        name = Path(path).parent.name + "/" + Path(path).name
        for key, value in read_options(path).items():
            if key in REPORTED:
                asked.setdefault(key, (value, name))
                continue
            if key not in values:
                values[key], source[key] = value, name
            elif values[key] != value:
                conflicts.append(f"{key}: {source[key]} sets {values[key]}, {name} sets {value}; "
                                 f"using {values[key]} (robots first, then objects, in scene order)")
    return MjcfOptions(values, source, asked, tuple(conflicts))


def option_files(robots: Sequence, objects: Sequence, asset: Callable[[str], str]) -> list[str]:
    """The scene's MJCF robot files, then its MJCF object cards' files (absolute paths)."""
    from alloy.cards.types import MJCFSource

    paths = [asset(r.path) for r in robots if r.kind == "mjcf"]
    for t in objects:
        shape = getattr(getattr(t.obj, "card", None), "shape", None)
        if isinstance(shape, MJCFSource):
            paths.append(shape.path)
    return paths


def report_lines(opts: MjcfOptions, kw: Mapping[str, object], overrides: Mapping[str, tuple[object, str]]) -> list[str]:
    """Route-report lines: each MJCF value taken, each one the route overrides (with its reason), conflicts."""
    lines = []
    for key in HONOURED:
        if key not in opts.values:
            continue
        mine, src = opts.values[key], opts.source[key]
        if key in overrides and overrides[key][0] != mine:
            lines.append(f"mujoco {key}={kw[key]}: {src} sets {mine}; route override: {overrides[key][1]}")
        else:
            lines.append(f"mujoco {key}={mine} from {src}")
    for key, (value, src) in opts.asked.items():
        if key in kw and kw[key] != value:
            why = overrides[key][1] if key in overrides else "the capability default (§3.5 integrator table)"
            lines.append(f"mujoco {key}={kw[key]}: {src} asks {value}; Alloy's integrator route: {why}")
    for key, (value, why) in overrides.items():          # set by the route where no file sets the key
        if key not in opts.values and key not in opts.asked and kw.get(key) == value:
            lines.append(f"mujoco {key}={value}: no file sets it; route: {why}")
    lines.extend(f"mujoco option conflict, {c}" for c in opts.conflicts)
    return lines


__all__ = ["HONOURED", "MjcfOptions", "option_files", "read_options", "report_lines", "scene_options"]
