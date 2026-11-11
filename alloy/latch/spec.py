# SPDX-License-Identifier: Apache-2.0
"""Generic sprung latch: the data spec shared by the MJCF and ModelBuilder back ends.

A latch assembly is a root part (static or dynamic) carrying sprung members. A member is a rigid part on a
``hinge`` or ``slide`` joint to its parent (the root or another member), pushed by a preloaded spring onto its rest
stop. Joint coordinate ``q``: 0 at the rest stop, positive = deflection off the stop (pressing a tab, pulling a
sleeve back); the joint range ``[0, travel]`` is the pair of stops. The spring force is ``-rate·(q - springref)`` with
``springref = -preload/rate`` beyond the stop, so at rest the member presses its stop with ``preload``.

Frames: the root frame is the assembly's frame. A member frame has its origin at the joint anchor and the axes of
its parent's frame at ``q = 0``; its shapes and centre of mass are given in it. Units SI, angles in rad.

Lessons applied ([internal reference], read-only): CON-45 (a light sprung member is regularised with body
inertia or mass, not armature, to omega·h <= 0.35 at the design step); CON-47 (members never collide with their own
assembly: explicit excludes even when the parent is welded to the world; the stop is the joint limit, made stiff);
CON-61 (ramps and catches below atan(1/mu_max), see ``alloy.latch.design``); CON-62 (the regularisation is sized
once at ``design_dt`` and held when the step is halved).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]            # x y z w

ROOT = "root"
ROLES = ("body", "guide", "tab", "catch", "ramp", "pad", "stop", "grip", "visual")
TOUCHED = frozenset({"guide", "tab", "catch", "ramp", "pad"})   # faces that carry the latch and location forces
                                                                # (stiff / precision contact on both sides)
OMEGA_H_MAX = 0.35                                    # CON-45 / CON-20: regularised mode omega·h at design_dt


class LatchSpecError(ValueError):
    """A latch spec that cannot be built (names the field)."""


def _unit(v: Vec3, what: str) -> Vec3:
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-12:
        raise LatchSpecError(f"{what}: zero vector")
    return (v[0] / n, v[1] / n, v[2] / n)


@dataclass(frozen=True)
class Box:
    """A box shape: half extents, pose in its part's frame, role (``TOUCHED`` roles carry the latch forces)."""
    name: str
    half: Vec3
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (0.0, 0.0, 0.0, 1.0)
    role: str = "body"
    collide: bool = True

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise LatchSpecError(f"box {self.name}: role {self.role!r} not in {ROLES}")
        if min(self.half) <= 0.0:
            raise LatchSpecError(f"box {self.name}: half extents must be > 0, got {self.half}")
        n = math.sqrt(sum(c * c for c in self.quat))
        if abs(n - 1.0) > 1e-6:
            raise LatchSpecError(f"box {self.name}: quat not unit ({n})")

    @property
    def volume(self) -> float:
        return 8.0 * self.half[0] * self.half[1] * self.half[2]


@dataclass(frozen=True)
class Spring:
    """Spring on a member's joint: ``rate`` N/m or N·m/rad, ``preload`` N or N·m on the rest stop, ``damping``
    N·s/m or N·m·s/rad (joint damping; implicit in MuJoCo)."""
    rate: float
    preload: float = 0.0
    damping: float = 0.0

    def __post_init__(self) -> None:
        if self.rate <= 0.0 or self.preload < 0.0 or self.damping < 0.0:
            raise LatchSpecError(f"spring: rate > 0, preload >= 0, damping >= 0 (got {self})")

    @property
    def springref(self) -> float:
        """Joint coordinate of zero spring force (beyond the rest stop)."""
        return -self.preload / self.rate

    def force(self, q: float) -> float:
        """Generalised spring force at ``q`` (negative = toward the rest stop)."""
        return -self.rate * (q - self.springref)


@dataclass(frozen=True)
class Member:
    """A sprung part. ``anchor``/``axis`` in the parent frame; ``travel`` = stop-to-stop range (rad or m);
    ``mass`` kg (physical, spread over the shapes by volume unless ``com`` is given).

    ``regularise`` puts the CON-45 floor (omega·h <= 0.35) in body mass/inertia (``"mass"``) or in the joint's
    armature (``"armature"``: the parent keeps the member's physical mass; DC3's light preloaded pads).
    ``stop_stiffness`` (N/m for a slide, N·m/rad for a hinge) is a floor on the rest stop's static stiffness, set
    through a flat ``solimplimit`` (CON-47); ``None`` keeps MuJoCo's default impedance on the direct solreflimit.
    ``snap_step`` (rad or m) bounds the travel in one ``design_dt`` step under the preload alone, the snap when a
    contact lets go: the regularised inertia is at least ``preload·dt²/snap_step`` (DC4: a preloaded DIMM clip
    jumped ~6 deg in one step). ``None`` keeps the omega·h floor only."""
    name: str
    kind: Literal["hinge", "slide"]
    anchor: Vec3
    axis: Vec3
    travel: float
    spring: Spring
    shapes: tuple[Box, ...]
    mass: float
    parent: str = ROOT
    com: Vec3 | None = None
    regularise: Literal["mass", "armature"] = "mass"
    stop_stiffness: float | None = None
    snap_step: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("hinge", "slide"):
            raise LatchSpecError(f"member {self.name}: kind {self.kind!r}")
        if self.travel <= 0.0 or self.mass <= 0.0 or not self.shapes:
            raise LatchSpecError(f"member {self.name}: travel > 0, mass > 0 and >= 1 shape required")
        if self.regularise not in ("mass", "armature"):
            raise LatchSpecError(f"member {self.name}: regularise {self.regularise!r} (mass | armature)")
        if self.stop_stiffness is not None and self.stop_stiffness <= 0.0:
            raise LatchSpecError(f"member {self.name}: stop_stiffness must be > 0")
        if self.snap_step is not None and self.snap_step <= 0.0:
            raise LatchSpecError(f"member {self.name}: snap_step must be > 0")
        object.__setattr__(self, "axis", _unit(self.axis, f"member {self.name} axis"))


@dataclass(frozen=True)
class Coupling:
    """``q_follower = c0 + c1·q_leader + c2·q_leader² + ...`` (MuJoCo joint equality polycoef, 5 terms)."""
    leader: str
    follower: str
    polycoef: tuple[float, ...] = (0.0, 1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Catch:
    """A catch face on ``part`` (member or ROOT): box ``box`` is the catch; it is engaged when it sits behind
    the partner's keeper box along the partner's insertion axis and overlaps it along ``lift`` (the direction the
    member deflects to release, in ``part``'s frame at rest) by at least ``min_overlap`` m (state.py)."""
    name: str
    part: str
    box: str
    lift: Vec3
    min_overlap: float = 1.0e-4


@dataclass(frozen=True)
class ContactPair:
    """A contact kept between two boxes on different parts of one assembly, e.g. a detent bump on a slider against
    the body near rest. CON-47 excludes every other intra-assembly contact; this one is an explicit MuJoCo pair, which
    the exclude does not filter. It carries its own friction ``mu`` and the precision contact: timeconst
    ``timeconst`` (None: 2 x design_dt, D22), ``dampratio`` and a flat ``solimp``."""
    box1: str
    box2: str
    mu: float
    timeconst: float | None = None
    dampratio: float = 1.0
    solimp: tuple[float, float, float, float, float] = (0.99, 0.99, 0.001, 0.5, 2.0)
    condim: int = 3

    def __post_init__(self) -> None:
        if self.mu < 0.0 or self.dampratio <= 0.0 or (self.timeconst is not None and self.timeconst <= 0.0):
            raise LatchSpecError(f"contact {self.box1}/{self.box2}: mu >= 0, dampratio > 0, timeconst > 0")
        if self.condim not in (1, 3, 4, 6):
            raise LatchSpecError(f"contact {self.box1}/{self.box2}: condim {self.condim} not in 1, 3, 4, 6")


@dataclass(frozen=True)
class LatchSpec:
    """An assembly: root shapes + mass, members, couplings, catches. ``static`` roots are welded to the world.
    ``design_dt`` sizes the CON-45 regularisation and the stop solref; keep it at the scene's coarsest step."""
    name: str
    root_shapes: tuple[Box, ...]
    root_mass: float
    members: tuple[Member, ...] = ()
    couplings: tuple[Coupling, ...] = ()
    catches: tuple[Catch, ...] = ()
    static: bool = False
    design_dt: float = 1.25e-3
    root_com: Vec3 | None = None
    stop_omega_h: float = 0.3                  # CON-47 direct stop solref (-(w/dt)^2, -2 zeta w/dt)
    notes: dict = field(default_factory=dict)  # free text per field (sources), carried into the MJCF comment
    contacts: tuple[ContactPair, ...] = ()     # intra-assembly contacts kept despite CON-47 (detents)

    def __post_init__(self) -> None:
        names = [ROOT] + [m.name for m in self.members]
        if len(set(names)) != len(names):
            raise LatchSpecError(f"{self.name}: duplicate member names {names}")
        for m in self.members:
            if m.parent not in names or m.parent == m.name:
                raise LatchSpecError(f"{self.name}: member {m.name} parent {m.parent!r} unknown")
        if names.index(ROOT) != 0 or any(names.index(m.parent) >= names.index(m.name) for m in self.members):
            raise LatchSpecError(f"{self.name}: members must follow their parents")
        boxes = [b.name for b in self.root_shapes] + [b.name for m in self.members for b in m.shapes]
        if len(set(boxes)) != len(boxes):
            raise LatchSpecError(f"{self.name}: duplicate shape names")
        for c in self.couplings:
            if c.leader not in names[1:] or c.follower not in names[1:]:
                raise LatchSpecError(f"{self.name}: coupling {c} names an unknown member")
        for c in self.catches:
            part = self.part_shapes(c.part)
            if c.box not in [b.name for b in part]:
                raise LatchSpecError(f"{self.name}: catch {c.name} box {c.box!r} not on {c.part}")
        if self.design_dt <= 0.0:
            raise LatchSpecError(f"{self.name}: design_dt must be > 0")
        for c in self.contacts:
            self._check_contact(c, boxes)

    def _check_contact(self, c: ContactPair, boxes: list[str]) -> None:
        for n in (c.box1, c.box2):
            if n not in boxes:
                raise LatchSpecError(f"{self.name}: contact box {n!r} unknown")
        (pa, a), (pb, b) = self.box(c.box1), self.box(c.box2)
        if pa == pb:
            raise LatchSpecError(f"{self.name}: contact {c.box1}/{c.box2} is within part {pa} (rigid, no contact)")
        if not all(x.collide and x.role != "visual" for x in (a, b)):
            raise LatchSpecError(f"{self.name}: contact {c.box1}/{c.box2} names a non-colliding box")

    def contact_solref(self, c: ContactPair) -> tuple[float, float]:
        """(timeconst, dampratio) of a kept contact: ``c.timeconst`` or 2 x design_dt (D22)."""
        return (c.timeconst if c.timeconst is not None else 2.0 * self.design_dt, c.dampratio)

    def member(self, name: str) -> Member:
        for m in self.members:
            if m.name == name:
                return m
        raise KeyError(name)

    def part_shapes(self, part: str) -> tuple[Box, ...]:
        return self.root_shapes if part == ROOT else self.member(part).shapes

    def box(self, name: str) -> tuple[str, Box]:
        """(part, box) for a shape name."""
        for part in (ROOT, *[m.name for m in self.members]):
            for b in self.part_shapes(part):
                if b.name == name:
                    return part, b
        raise KeyError(name)
