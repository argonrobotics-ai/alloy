# SPDX-License-Identifier: Apache-2.0
"""Shared parts of the drilling task ports: Gym-Anything's six drilling tasks, scored by their v1 checks.

Each task (``alloy.tasks.drilling_<task>``) is a stack of static drilled parts (``Drill`` layers, top first), a
scripted spindle (``FeedHead``) that ``drilling_script.Driller`` moves from mark to mark, and, for the bolt and anchor
tasks, pins (``Drill(pins=...)``) that ``PushHead`` hands push into the holes (``drilling_script.Inserter``). The
scene has no gravity: the scripted heads carry every body (the bolts' weights, under 2 N, are left out), and no body
has solver collision (the drill owns the bit's contact and the pins'; the parts are static).

The checks are ports of the source's v1 ``Evaluator`` (success = every check passes), read from the drill's hole
records, its pin states and the bodies' poses. ``progress`` is the fraction of checks passed (the source's opt-in
contract with its progress ramps is not ported). A check the scene cannot measure, or that could not fail here, is
listed in ``unscored`` with the reason and left out of the success and the progress (``Verdict``): the static parts'
moves and turns (no vise or bracket compliance), a hole's wander (the bit does not walk), heat and chipping, an
anchor's spin (no nut is turned), and what a body does once its hand lets go (no gravity).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import body_xform
from alloy.plugins.holes.table import Hole
from alloy.plugins.npgeom import quat_rotate

STEP = 6.25e-4              # s: the drill's exchange step (control_dt / 32)
CONTROL_DT = 0.02           # s
GAP = 1.0e-3                # m: tip above the face, along the drill axis, when an operation starts
SETTLE = 0.1                # s: an operation's time past its ramp (the head closes its lag)
HEAD_HALF = 0.03            # m: the spindle housing's half height (its tip offsets start below it)
HEAD_SPAN = 0.15            # m: its half width: 3 kg with a hand drill's inertia across the axis (0.023 kg m^2),
HEAD_MASS = 3.0             # kg  so the bit's wall springs stay stable at the tip's lever arm
HOLDER_HALF = 0.2           # m: a pin's holder (the hand and forearm) half extent
HOLDER_MASS = 4.0           # kg: with its inertia (m s^2 / 6 = 0.107 kg m^2 about its centre, 0.267 about the pin's
                            # origin 0.2 m below) the pin springs stay stable at the shank's far end
MARK_RADIUS = 0.020         # m: a hole farther than this from a mark is not the mark's (the source's radius)
PARK = (0.6, 0.0, 0.4)      # m: where idle tools wait, away from the parts
RPM = 2.0 * math.pi / 60.0  # rad/s per rpm
STATIC = "the part is static (no vise, clamp or bracket compliance): it reads 0 in every world"
NO_GRAVITY = "the scene has no gravity: nothing pulls a body once its hand lets go"


@dataclass(frozen=True)
class Op:
    """One scripted drilling operation: the bit at ``mark`` (world x, y on the top face) plus ``offset``, its axis
    ``tilt`` degrees off the face normal (about world y), fed ``depth`` past the face along its axis at ``feed`` per
    revolution, spinning at ``spin``; ``hammer`` is the rotary hammer's (blow energy J, blow rate 1/s)."""

    mark: tuple[float, float]
    diameter: float
    depth: float
    feed: float                                   # m/rev
    spin: float                                   # rad/s
    tilt: float = 0.0                             # deg
    offset: tuple[float, float] = (0.0, 0.0)      # m: aim error
    hammer: tuple[float, float] = (0.0, 0.0)
    settle: float = SETTLE                        # s past the ramp
    stop_on_jam: bool = True                      # an operator stops at a bar; False: pushes on to the ramp's end

    @property
    def rate(self) -> float:
        """Feed rate [m/s]."""
        return self.feed * abs(self.spin) / (2.0 * math.pi)

    @property
    def direction(self) -> np.ndarray:
        t = math.radians(self.tilt)
        return np.array([math.sin(t), 0.0, -math.cos(t)])

    @property
    def seconds(self) -> float:
        return (GAP + self.depth) / self.rate + self.settle


@dataclass(frozen=True)
class Insert:
    """A pin pushed into the hole nearest ``mark`` (world x, y) with ``force`` [N], its nut torqued to ``torque``
    [N m] (anchors) once the push has run."""

    pin: str
    mark: tuple[float, float]
    force: float = 40.0
    torque: float = 0.0
    lift: float = 0.004                           # m: its tip above the top face at the start


@dataclass(frozen=True)
class Plan:
    """One world's script: the drilling operations in order, whether the spindle is retracted and stopped at the
    end (``park``), then the pins pushed in."""

    ops: tuple[Op, ...]
    park: bool = True
    inserts: tuple[Insert, ...] = ()


@dataclass(frozen=True)
class Check:
    """One v1 check: its name as the source words it, the verdict, the measured value and its bound."""

    name: str
    passed: bool
    value: Any = None
    bound: str = ""
    kind: str = "goal"                            # or "constraint" (the source's two kinds; both decide success)


@dataclass
class Verdict:
    """A world's checks; success = every check passes (the source's v1 ``Evaluator``)."""

    checks: list[Check]
    unscored: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def progress(self) -> float:
        return sum(c.passed for c in self.checks) / max(len(self.checks), 1)

    @property
    def failed(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]

    def get(self, name: str) -> Check:
        hits = [c for c in self.checks if c.name == name]
        if not hits:
            raise KeyError(f"no check {name!r}; checks: {[c.name for c in self.checks]}")
        return hits[0]


def angle_deg(a: Any, b: Any) -> float:
    """Angle between two lines [deg] (the source's ``_angle_deg``: sign-free)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))


def face_normal(face: np.ndarray) -> np.ndarray:
    return quat_rotate(np.asarray(face, np.float64)[3:], np.array([0.0, 0.0, 1.0]))


def in_plane(face: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """Distance from ``p`` to ``q`` across the face's normal (world points) [m]."""
    n = face_normal(face)
    d = np.asarray(p, np.float64) - np.asarray(q, np.float64)
    return float(np.linalg.norm(d - n * float(d @ n)))


def nearest(drill: Any, w: int, layer: int, body_q: np.ndarray, mark: Any) -> tuple[Hole, float] | None:
    """The hole of ``layer`` in world ``w`` whose entry lies nearest ``mark`` (world point) across the face, with that
    distance (the source's ``nearest``); None without holes."""
    face = drill.face(w, layer, body_q)
    best = None
    for h in drill.holes(w, layer):
        e = in_plane(face, h.in_world(face)[0], mark)
        if best is None or e < best[1]:
            best = (h, e)
    return best


def mark_point(mark: tuple[float, float], top: float) -> np.ndarray:
    return np.array([mark[0], mark[1], top])


def through_checks(label: str, hit: tuple[Hole, float] | None, d: float, tol: float,
                   max_d: float | None = None) -> tuple[list[Check], Hole | None]:
    """The source's ``_through_hole_checks``: through at ``d`` [m], entry within ``tol`` of the mark, the axis within
    1 deg of the face normal, and (``max_d``) the diameter between ``d`` and ``max_d`` (the hole's diameter plus twice
    its oversize: 0 here, the hole is cut along one axis)."""
    if hit is None or not hit[0].through_at(d):
        why = None if hit is None else ("not through" if not hit[0].through else "through at a smaller diameter only")
        return [Check(f"{label}: through hole at the mark", False, why)], None
    h, e = hit
    out = [Check(f"{label}: through hole at the mark", True, "through"),
           Check(f"{label}: entry within {tol * 1e3:.1f} mm of the mark", e <= tol, round(e * 1e3, 3),
                 f"<= {tol * 1e3:.1f} mm"),
           Check(f"{label}: axis within 1 deg of the face normal", h.tilt_deg <= 1.0, round(h.tilt_deg, 3),
                 "<= 1 deg")]
    if max_d is not None:
        dm = h.diameter * 1e3
        out.append(Check(f"{label}: diameter at most {max_d * 1e3:g} mm", dm <= max_d * 1e3 + 1e-6 and
                         dm >= d * 1e3 - 1e-6, round(dm, 3), f"{d * 1e3:g}-{max_d * 1e3:g} mm"))
    return out, h


class VariantTask:
    """Mixin for the task dataclasses: ``variants`` is ((name, Plan), ...), one per world (the last repeats)."""

    variants: tuple[tuple[str, Plan], ...]

    def plans(self, worlds: int) -> list[Plan]:
        return [self.variants[min(w, len(self.variants) - 1)][1] for w in range(worlds)]

    def variant(self, w: int) -> str:
        return self.variants[min(w, len(self.variants) - 1)][0]


@functools.lru_cache(maxsize=None)
def density(card: str) -> float:
    """A material card's density [kg/m^3]."""
    from alloy.cards import CardLibrary  # noqa: PLC0415

    return float(CardLibrary().material(card).density.value)


def report(verdicts: list[Verdict]) -> list[dict[str, Any]]:
    """Per world: success, progress, the failed checks and every check's value (JSON-ready)."""
    return [{"world": w, "success": v.success, "progress": round(v.progress, 4), "failed": v.failed,
             "unscored": list(v.unscored), "checks": {c.name: c.value for c in v.checks}, **v.detail}
            for w, v in enumerate(verdicts)]


# ------------------------------------------------------------------ scene parts

def add_part(b: newton.ModelBuilder, label: str, center: tuple[float, float, float],
             half: tuple[float, float, float]) -> None:
    """A static box part (a drill layer is its top face)."""
    b.add_shape_box(-1, xform=body_xform(*center), hx=half[0], hy=half[1], hz=half[2], label=label)


def _ghost(density: float) -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(density=density, has_shape_collision=False)


def add_spindle(b: newton.ModelBuilder, label: str = "spindle") -> None:
    """The scripted spindle's housing (no solver collision), parked."""
    body = b.add_body(xform=body_xform(*PARK), label=label)
    b.add_shape_box(body, hx=HEAD_SPAN, hy=HEAD_SPAN, hz=HEAD_HALF,
                    cfg=_ghost(HEAD_MASS / (8.0 * HEAD_SPAN * HEAD_SPAN * HEAD_HALF)))


def add_pin(b: newton.ModelBuilder, label: str, slot: int) -> None:
    """A pin's body (origin at its shank top) with the hand's holder above it, parked beside the spindle."""
    body = b.add_body(xform=body_xform(PARK[0], 0.5 * (slot + 1), PARK[2]), label=label)
    b.add_shape_box(body, xform=body_xform(0.0, 0.0, HOLDER_HALF), hx=HOLDER_HALF, hy=HOLDER_HALF, hz=HOLDER_HALF,
                    cfg=_ghost(HOLDER_MASS / (2.0 * HOLDER_HALF) ** 3))


def bit_tip(depth: float) -> tuple[float, float, float]:
    """The drill's tip in the spindle frame for a bit that reaches ``depth`` without the housing touching a part."""
    return (0.0, 0.0, -HEAD_HALF - depth - 0.02)


__all__ = ["CONTROL_DT", "GAP", "MARK_RADIUS", "NO_GRAVITY", "PARK", "RPM", "STATIC", "STEP", "Check", "Insert",
           "Op", "Plan", "VariantTask", "Verdict", "add_part", "add_pin", "add_spindle", "angle_deg", "bit_tip",
           "density", "face_normal", "in_plane", "mark_point", "nearest", "report", "through_checks"]
