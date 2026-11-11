# SPDX-License-Identifier: Apache-2.0
"""The deposition plug-in's tools: markers that lay ink and wipers that lift it, from per-tool config tuples.

A config gives each tool parameter as a tuple with one value per tool, or one value for every tool. ``tool_specs``
expands them into ``ToolSpec`` rows (host) and ``tip_array`` into a device array of ``Tip`` structs. The contact of
every tool is the removal plug-in's compliant pad (``alloy.plugins.removal.kernels.Pad``: Winkler pressure,
Kelvin-Voigt damping, Coulomb friction), with an optional dome: a marker nib's face is a paraboloid of apex radius
``dome_radius`` (sag r²/2R, a sphere's small-patch limit, a ≪ R), so its contact patch grows with the force
(``a = sqrt(2·R·δ)``, ``F = k·π·R·δ²``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.removal.kernels import Pad

KINDS = {"marker": 0, "wiper": 1}
PER_TOOL = ("kinds", "colours", "tip_center", "tip_normal", "tip_radius", "dome_radius", "stiffness", "damping",
            "friction", "flow", "reservoir", "wipe_length")


@wp.struct
class Tip:
    pad: Pad                     # contact geometry and compliance (removal's pad)
    curvature: wp.float32        # 1 / dome radius [1/m]; 0: a flat face
    kind: wp.int32               # 0 marker, 1 wiper
    colour: wp.int32             # ink channel a marker lays
    flow: wp.float32             # marker: areal ink flux per contact pressure, j = flow·p [kg/(N·s)]
    wipe_length: wp.float32      # wiper: lifts 1 - exp(-cover·ds / wipe_length) of every channel [m]


@dataclass(frozen=True)
class ToolSpec:
    """One tool, expanded from the config tuples."""

    label: str
    kind: str
    colour: int
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    radius: float
    dome_radius: float
    stiffness: float
    damping: float
    friction: float
    flow: float
    reservoir: float
    wipe_length: float

    @property
    def contact_stiffness(self) -> float:
        """The pad pressed in over its whole face [N/m]: the worst case for the explicit step (a dome is softer)."""
        return self.stiffness * math.pi * self.radius**2


def _per_tool(c: Any, name: str, n: int) -> list[Any]:
    value = getattr(c, name)
    seq = list(value) if isinstance(value, (list, tuple)) and _is_sequence_field(name, value) else [value]
    if len(seq) == 1:
        return seq * n
    if len(seq) != n:
        raise ValueError(f"{name} has {len(seq)} values for {n} tools: give one per tool or one for all")
    return seq


def _is_sequence_field(name: str, value: Sequence[Any]) -> bool:
    """Vector fields (tip_center, tip_normal) are per-tool only when they hold vectors."""
    if name in ("tip_center", "tip_normal"):
        return len(value) > 0 and isinstance(value[0], (list, tuple))
    return True


def tool_specs(c: Any) -> list[ToolSpec]:
    """Expand and check the per-tool config tuples."""
    n = len(c.tools)
    if n < 1:
        raise ValueError("tools must name at least one tool body")
    cols = {name: _per_tool(c, name, n) for name in PER_TOOL}
    specs = []
    for t in range(n):
        row = {name: cols[name][t] for name in PER_TOOL}
        spec = ToolSpec(label=str(c.tools[t]), kind=str(row["kinds"]), colour=int(row["colours"]),
                        center=tuple(float(v) for v in row["tip_center"]),
                        normal=tuple(float(v) for v in row["tip_normal"]), radius=float(row["tip_radius"]),
                        dome_radius=float(row["dome_radius"]), stiffness=float(row["stiffness"]),
                        damping=float(row["damping"]), friction=float(row["friction"]), flow=float(row["flow"]),
                        reservoir=float(row["reservoir"]), wipe_length=float(row["wipe_length"]))
        _check(spec, len(c.channels))
        specs.append(spec)
    return specs


def _check(s: ToolSpec, channels: int) -> None:
    if s.kind not in KINDS:
        raise ValueError(f"tool {s.label!r}: kind must be one of {sorted(KINDS)}, got {s.kind!r}")
    if not 0 <= s.colour < channels:
        raise ValueError(f"tool {s.label!r}: colour {s.colour} is not a channel index (0..{channels - 1})")
    positive = {"tip_radius": s.radius, "stiffness": s.stiffness}
    nonneg = {"dome_radius": s.dome_radius, "damping": s.damping, "friction": s.friction, "flow": s.flow,
              "reservoir": s.reservoir, "wipe_length": s.wipe_length}
    bad = [k for k, v in positive.items() if not (math.isfinite(v) and v > 0.0)]
    bad += [k for k, v in nonneg.items() if not (math.isfinite(v) and v >= 0.0)]
    if bad:
        raise ValueError(f"tool {s.label!r}: {bad} must be finite and positive (tip_radius, stiffness) or >= 0")
    if len(s.center) != 3 or len(s.normal) != 3 or np.linalg.norm(s.normal) <= 0.0:
        raise ValueError(f"tool {s.label!r}: tip_center and tip_normal need 3 values, the normal non-zero")
    if s.kind == "wiper" and s.wipe_length <= 0.0:
        raise ValueError(f"tool {s.label!r}: a wiper needs wipe_length > 0")


def tip_array(specs: list[ToolSpec], slip_eps: float, device: Any) -> wp.array:
    """The tools as a device array of ``Tip`` structs (index = tool order)."""
    tips = []
    for s in specs:
        pad = Pad()
        pad.center = wp.vec3(*s.center)
        pad.normal = wp.vec3(*(np.asarray(s.normal, np.float64) / np.linalg.norm(s.normal)))
        pad.radius, pad.stiffness, pad.damping = s.radius, s.stiffness, s.damping
        pad.friction, pad.slip_eps = s.friction, float(slip_eps)
        tip = Tip()
        tip.pad = pad
        tip.curvature = 1.0 / s.dome_radius if s.dome_radius > 0.0 else 0.0
        tip.kind, tip.colour = KINDS[s.kind], s.colour
        tip.flow, tip.wipe_length = s.flow, s.wipe_length
        tips.append(tip)
    return wp.array(tips, dtype=Tip, device=device)


__all__ = ["KINDS", "Tip", "ToolSpec", "tip_array", "tool_specs"]
