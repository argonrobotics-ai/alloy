# SPDX-License-Identifier: Apache-2.0
"""The neutral model every Alloy reader fills and every writer reads (MJCF, URDF, USD, cards).

One ``IOModel`` describes one world: rigid bodies with their collision shapes and mass properties, the joints that
connect them, surface materials and equality constraints. It is plain data (SI units, frames as 7-tuples
``(x, y, z, qx, qy, qz, qw)``, quaternions in Warp order) and has no Newton or MuJoCo objects in it, so writers can
be tested without a simulator.

Example::

    from alloy.io.model import IOModel, IOBody, IOShape, IOMaterial
    box = IOShape(name="box", kind="box", size=(0.03, 0.03, 0.03), material="wood")
    body = IOBody(name="box", pose=(0, 0, 0.05, 0, 0, 0, 1), mass=0.2, shapes=(box,))
    model = IOModel(name="demo", bodies=[body], materials={"wood": IOMaterial("wood", 720.0, 0.4, 0.3, 0.5)})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Pose = tuple[float, float, float, float, float, float, float]
IDENTITY: Pose = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
ShapeKind = Literal["box", "sphere", "capsule", "cylinder", "cone", "ellipsoid", "plane", "mesh"]
JointKind = Literal["free", "fixed", "revolute", "prismatic", "ball"]
Motion = Literal["dynamic", "kinematic", "static"]


@dataclass(frozen=True)
class IOMesh:
    """A triangle mesh in its shape frame (scale already applied), metres."""

    vertices: np.ndarray            # [V, 3] float64
    faces: np.ndarray               # [F, 3] int32

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", np.asarray(self.vertices, np.float64).reshape(-1, 3))
        object.__setattr__(self, "faces", np.asarray(self.faces, np.int32).reshape(-1, 3))


@dataclass(frozen=True)
class IOMaterial:
    """Surface and bulk values of one material. ``origin`` says where they came from ("card" or "file")."""

    name: str
    density: float                  # kg/m^3 (0 = unknown)
    mu_static: float
    mu_dynamic: float
    restitution: float
    torsional_lever: float = 0.0    # m (Alloy/Newton torsional friction is a lever arm)
    rolling_lever: float = 0.0      # m
    contact_stiffness: float | None = None  # N/m per material (Alloy D22); None = engine default
    origin: Literal["card", "file"] = "card"


@dataclass(frozen=True)
class IOShape:
    """One collision (and visual) shape. ``size`` follows Alloy's ``Primitive`` convention: box half-extents,
    sphere (radius,), capsule/cylinder/cone (radius, half_height) along the shape's +Z, ellipsoid semi-axes,
    plane (half_width, half_length) with 0 meaning infinite. Cones point their apex to +Z."""

    name: str
    kind: ShapeKind
    size: tuple[float, ...] = ()
    pose: Pose = IDENTITY           # in the body frame
    material: str = ""
    mesh: IOMesh | None = None
    convex: bool = True             # mesh collision uses the convex hull
    collide: bool = True
    visible: bool = True
    solref: tuple[float, float] | None = None   # MuJoCo contact (time constant s, damping ratio); None = unknown


@dataclass(frozen=True)
class IOBody:
    """A rigid body at its initial world pose; ``com`` and ``inertia`` (about the COM) are in the body frame."""

    name: str
    pose: Pose = IDENTITY
    mass: float = 0.0
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    inertia: tuple[tuple[float, float, float], ...] = ((0.0,) * 3,) * 3
    shapes: tuple[IOShape, ...] = ()
    motion: Motion = "dynamic"
    group: str = ""                 # owning object or robot name
    gravcomp: float = 0.0           # MuJoCo gravity compensation (1 = the body's weight is cancelled)


@dataclass(frozen=True)
class IOJoint:
    """A joint from ``parent`` (None = world) to ``child``. Frames: ``parent_pose`` in the parent body frame,
    ``child_pose`` in the child body frame; ``axis`` is in the joint frame. Limits are rad or m."""

    name: str
    kind: JointKind
    parent: str | None
    child: str
    parent_pose: Pose = IDENTITY
    child_pose: Pose = IDENTITY
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lower: float | None = None
    upper: float | None = None
    effort: float | None = None     # N or N·m
    velocity: float | None = None   # m/s or rad/s
    damping: float = 0.0
    friction: float = 0.0
    armature: float = 0.0
    stiffness: float = 0.0          # position-drive gain (N/m or N·m/rad), 0 = no drive
    drive_damping: float = 0.0      # velocity gain of the drive
    q0: float = 0.0                 # initial coordinate
    spring: float = 0.0             # passive spring stiffness about ``spring_ref`` (N/m or N·m/rad)
    spring_ref: float = 0.0         # coordinate where the passive spring is relaxed
    drive_dampratio: float = 0.0    # MuJoCo ``dampratio`` of a position servo (used when ``drive_damping`` is 0)
    motor_gear: float | None = None  # a force/torque motor on the joint (MuJoCo ``<motor gear=>``); None = none
    ctrl_range: tuple[float, float] | None = None   # control range of the joint's actuator
    actuated: bool = False          # an actuator drives this joint (a robot joint)


@dataclass(frozen=True)
class IOEquality:
    """A weld or connect constraint between two bodies (``body2`` None = world); ``anchor`` in body1's frame."""

    name: str
    kind: Literal["weld", "connect"]
    body1: str
    body2: str | None
    anchor: Pose = IDENTITY


@dataclass(frozen=True)
class IOPair:
    """One explicit contact pair between two shapes (MuJoCo ``<contact><pair>``): its friction, and its restitution
    when the pair's contact damping gives one."""

    shape1: str
    shape2: str
    mu: float
    restitution: float | None = None


@dataclass(frozen=True)
class Unmapped:
    """One value or feature a format cannot carry. ``item`` names it (object or field), ``reason`` says what was
    done instead and what to change to keep it."""

    format: str
    item: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"format": self.format, "item": self.item, "reason": self.reason}


@dataclass
class IOModel:
    """One world. ``groups`` maps each object/robot name to its kind ("object", "robot", "articulated")."""

    name: str
    bodies: list[IOBody] = field(default_factory=list)
    world_shapes: list[IOShape] = field(default_factory=list)   # static shapes (no body)
    joints: list[IOJoint] = field(default_factory=list)
    equalities: list[IOEquality] = field(default_factory=list)
    materials: dict[str, IOMaterial] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    shape_groups: dict[str, str] = field(default_factory=dict)  # world shape name -> group
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    timestep: float | None = None
    unmapped: list[Unmapped] = field(default_factory=list)
    pairs: list[IOPair] = field(default_factory=list)
    unwritable: dict[str, str] = field(default_factory=dict)    # child body -> joint the writers cannot express

    def body(self, name: str) -> IOBody:
        """The body called ``name``; raises KeyError listing the known names."""
        for b in self.bodies:
            if b.name == name:
                return b
        raise KeyError(f"no body {name!r}; bodies: {[b.name for b in self.bodies]}")

    def joint_of(self, child: str) -> IOJoint | None:
        """The joint whose child is ``child`` (None for a body without one)."""
        return next((j for j in self.joints if j.child == child), None)

    def note(self, fmt: str, item: str, reason: str) -> None:
        """Record an ``Unmapped`` row once."""
        row = Unmapped(fmt, item, reason)
        if row not in self.unmapped:
            self.unmapped.append(row)
