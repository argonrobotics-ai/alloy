# SPDX-License-Identifier: Apache-2.0
"""World-template construction for the assembler (SPEC §3.4 steps 3-4).

Each object is emitted in its own frame on a fresh sub-builder and merged at its placement pose with
`add_builder(sub, xform=pose)`; the entity counters before/after give its `IndexRanges`. The `alloy:*_object` and
`alloy:*_material` ids (int32, one frequency each, builder.py:1345-1377) are written over those ranges afterwards.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.cards.types import MaterialCard, Placement, ResolvedObject, RobotRef, Surface
from alloy.core.errors import AssemblyError
from alloy.core.quantity import Q
from alloy.core.registry import SolverCapability
from alloy.route.types import Route, RouteDecision
from alloy.scene.emitters.common import RIGID_GAP
from alloy.scene.emitters.dispatch import ROUTE_EMITTER
from alloy.scene.robot_visuals import add_mjcf as add_mjcf_with_visuals
from alloy.scene.types import IndexRanges

log = logging.getLogger("alloy.scene")

GRAVITY = (0.0, 0.0, -9.81)
ALLOY_ATTRS = (("body_object", "BODY"), ("shape_object", "SHAPE"), ("particle_object", "PARTICLE"),
               ("shape_material", "SHAPE"), ("particle_material", "PARTICLE"))
KINDS = ("body", "shape", "joint", "particle", "tri", "tet", "edge")


def register_alloy_attributes(b: newton.ModelBuilder) -> None:
    """The five `alloy:*` int32 attributes (default −1 = not an Alloy object / no material)."""
    CA, F = newton.ModelBuilder.CustomAttribute, newton.Model.AttributeFrequency
    for name, freq in ALLOY_ATTRS:
        b.add_custom_attribute(CA(name=name, dtype=wp.int32, frequency=getattr(F, freq), namespace="alloy",
                                  default=-1))


def make_builder(caps: Iterable[SolverCapability], *, topology: bool = False) -> newton.ModelBuilder:
    """A builder with gravity, the alloy attributes and every chosen capability's attributes registered; with
    `topology`, also WP10's `topology:*` bond and crumble attributes (before any object is emitted)."""
    b = newton.ModelBuilder(gravity=GRAVITY)
    register_alloy_attributes(b)
    for cap in caps:
        if cap.register_attributes is not None:
            cap.register_attributes(b)
    if topology:
        from alloy.topology import bonds

        bonds.register_custom_attributes(b)
    return b


def counts(b: newton.ModelBuilder) -> dict[str, int]:
    return {"body": b.body_count, "shape": b.shape_count, "joint": b.joint_count, "particle": b.particle_count,
            "tri": b.tri_count, "tet": b.tet_count, "edge": b.edge_count}


def ranges_between(before: Mapping[str, int], after: Mapping[str, int]) -> IndexRanges:
    return IndexRanges(**{k: range(before[k], after[k]) for k in KINDS})


def pose_transform(pose: Sequence[float]) -> wp.transform:
    if len(pose) != 7:
        raise AssemblyError(f"pose must be (x, y, z, qx, qy, qz, qw), got {tuple(pose)}")
    q = np.asarray(pose[3:], np.float64)
    q = q / np.linalg.norm(q)
    return wp.transform(wp.vec3(*pose[:3]), wp.quat(*q))


@dataclass
class TemplateObject:
    name: str
    obj: ResolvedObject | None            # None for robots
    decision: RouteDecision
    placement: Placement | RobotRef
    material_key: str
    ranges: IndexRanges = field(default_factory=IndexRanges.empty)


@dataclass
class Template:
    builder: newton.ModelBuilder
    objects: list[TemplateObject]
    robot_materials: dict[str, MaterialCard]


def robot_material(name: str, b: newton.ModelBuilder, shapes: range) -> MaterialCard:
    """Synthetic material for a robot's imported shapes: the median imported μ, e = 0, default stiffness."""
    mus = [float(b.shape_material_mu[s]) for s in shapes] or [0.5]
    note = "from the robot file (median imported shape mu)"
    mu = float(np.median(mus))
    surface = Surface(mu_static=Q(mu, "model", note=note), mu_dynamic=Q(mu, "model", note=note),
                      restitution=Q(0.0, "model", note="robot contacts carry no restitution in the robot file"))
    return MaterialCard(name=f"robot:{name}", density=Q(1000.0, "default"), surface=surface)


def emit_robot(b: newton.ModelBuilder, robot: RobotRef, make_sub: Callable[[], newton.ModelBuilder],
               asset: Callable[[str], str] = str) -> IndexRanges:
    """Import a robot file (path resolved against the card library root by `asset`) at its base pose. Its shapes get
    the robot's ``contact_gap`` (default ``RIGID_GAP``, as every scene object), not Newton's ``rigid_gap`` = 0.1 m:
    that made contacts of every shape pair within 0.2 m and overflowed MuJoCo Warp's contact buffer in packed scenes.
    A geom's own MJCF ``gap`` attribute is kept."""
    sub = make_sub()
    sub.rigid_gap = RIGID_GAP if robot.contact_gap is None else float(robot.contact_gap)
    importer = {"mjcf": lambda p: add_mjcf_with_visuals(sub, p),   # an MJCF plus its visual layer, when it has one
                "urdf": sub.add_urdf, "usd": sub.add_usd}.get(robot.kind)
    if importer is None:
        raise AssemblyError(f"robot {robot.name}: unknown kind {robot.kind!r}")
    from alloy.io.robot_file import robot_source  # noqa: PLC0415 - alloy.io imports alloy.scene

    # the file as the importer needs it (alloy.io.robot_file): an unexpanded xacro URDF is refused with the command,
    # a URDF floating joint on an empty root link is a free base, a USD stage in other length units is read in metres
    source, options = robot_source(robot.kind, asset(robot.path))
    importer(source, **options)
    before = counts(b)
    b.add_builder(sub, xform=pose_transform(robot.base_pose))
    return ranges_between(before, counts(b))


def require_moving(b: newton.ModelBuilder, objects: Sequence[TemplateObject], asset: Callable[[str], str], tree):
    """Refuse a scene whose robot files hold only static geometry while nothing in it has a joint, with the fix
    (the MuJoCo converter raised a bare ValueError). Fires only with robot files and no joint at all; a scene
    without robot files is never refused here. Returns ``tree`` (the emission result) unchanged."""
    from alloy.io.robot_file import require_joint  # noqa: PLC0415 - alloy.io imports alloy.scene

    require_joint(b.joint_count, [asset(t.placement.path) for t in objects if t.obj is None])
    return tree


def emit_object(b: newton.ModelBuilder, t: TemplateObject, cap: SolverCapability | None,
                make_sub: Callable[[], newton.ModelBuilder]) -> IndexRanges:
    """Emit one routed object at its placement pose; STATIC objects use the stock route emitter."""
    sub = make_sub()
    emitter = cap.emitter if cap is not None and cap.emitter is not None else ROUTE_EMITTER
    emitter.emit(sub, t.obj, t.decision)
    rotate_particles(sub, t.placement.pose)
    before = counts(b)
    b.add_builder(sub, xform=pose_transform(t.placement.pose))
    return ranges_between(before, counts(b))


def rotate_particles(sub: newton.ModelBuilder, pose: Sequence[float]) -> None:
    """Turn the sub-builder's particles (positions and velocities) about its origin by the placement's rotation.
    Newton's ``add_builder`` adds only the translation to particles (builder.py ``_merge_builder_copies``:
    ``particle_q += offsets``) while bodies and shapes take the whole transform, so without this a rotated towel, soft
    part or MPM block lands unrotated (SCR-SOFT-2-4). Rest shapes need nothing: triangle and tet rest poses, spring
    lengths and bend angles are relative to the particles. No rotation: the particles are untouched (bit for bit)."""
    q = np.asarray(pose[3:], np.float64)
    if not sub.particle_count or not q[:3].any():
        return
    rot = quat_matrix(q / np.linalg.norm(q))
    for name in ("particle_q", "particle_qd"):
        v = np.asarray(getattr(sub, name), np.float64).reshape(-1, 3) @ rot.T
        setattr(sub, name, [tuple(row) for row in v.tolist()])


def quat_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of a unit quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.asarray([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tag(b: newton.ModelBuilder, r: IndexRanges, object_id: int, material_id: int) -> None:
    """Write the alloy object/material ids over an object's template ranges."""
    for key, rng, val in (("alloy:body_object", r.body, object_id), ("alloy:shape_object", r.shape, object_id),
                          ("alloy:particle_object", r.particle, object_id),
                          ("alloy:shape_material", r.shape, material_id),
                          ("alloy:particle_material", r.particle, material_id)):
        values = b.custom_attributes[key].values
        for i in rng:
            values[i] = val


def add_spares(b: newton.ModelBuilder, obj: ResolvedObject, n: int) -> range:
    """Inactive spare particles (flags 0) parked at the object origin for topology events (D19, §3.8)."""
    start = b.particle_count
    for _ in range(n):
        b.add_particle((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.0, radius=1e-4, flags=0)
    return range(start, b.particle_count)


def free_body_count(objs: Sequence[ResolvedObject]) -> int:
    """Free bodies per world for the pile rule: dynamic objects (grains counted by the router separately)."""
    return sum(1 for o in objs if o.card.motion == "dynamic")


def stride_ranges(template_counts: Mapping[str, int]) -> IndexRanges:
    """Per-world offsets after replicate as ranges of the stride length: global id = template id + w·len(range)."""
    return IndexRanges(**{k: range(template_counts[k]) for k in KINDS})


def is_compliant(d: RouteDecision) -> bool:
    return d.route == Route.RIGID_COMPLIANT and "k_compliant" in d.numbers


def finite_or(x: float, default: float) -> float:
    return x if math.isfinite(x) else default
