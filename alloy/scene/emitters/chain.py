# SPDX-License-Identifier: Apache-2.0
"""CHAIN emitter (D7, SPEC §3.8 Rods): rigid capsule links joined by D6 joints with two bend axes (k = EI/l) and one
twist axis (k = GJ/l), half-length end segments (L = (N−1)·l), capsule self-collision with adjacent links excluded
by Newton's joint parent filter (builder.py:6158-6165; shapes are added before each joint).

MuJoCo receives the hinge springs as `mujoco:dof_passive_stiffness` (solvers/mujoco/solver_mujoco.py:1764); other
joint-space solvers get `target_ke` toward 0. `implicitfast` integrates springs explicitly, so that variant adds
armature a = max(0, 2·k·h²/4 − I_axis) per axis (ω·h ≤ √2 on the stiffest local mode) and damping ≥ k·h/2.

Every rotational dof of a CHAIN rope, the root joint's included, gets armature ≥ `ARMATURE_FLOOR` (1e-6 kg·m²),
whatever the integrator. MuJoCo Warp's Newton solver factorises its Hessian in float32, and contact-loaded light
links make it non-finite: A10's routed rope (0.56 g links) went non-finite in 16 of 16 worlds at 1e-9, 10 of 16 at
1e-8, none at 1e-7 or 1e-6, and armature on any one dof group (bend, twist or the root) did not suffice. The floor
is up to ~40x a thin link's own bend inertia, but it acts on relative joint rates: the hanging-chain swing ω₁ moves
≤ 0.28 % (A11's chains at 0.5 / 0.8 m, A10's rope; MuJoCo Warp CPU) and D7's period and twist relaxation on T09's
cable do not move at 1e-6 (docs/notes/wp2-results.md "CHAIN armature floor"). A thin rod's first torsional mode does:
on A11.torsion's 2 mm rod the floor is 1,923x a link's axial inertia and rings it 2.58x slow at a 25 µs step
(docs/known_gaps/rod-twist-dynamics.md).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.cards.types import CurveSource, ResolvedObject
from alloy.route.types import RodCard, RouteDecision
from alloy.scene.emitters.common import Z_TO_X, shape_config

CHAIN_DT_DEFAULT = 1.25e-3  # s, rigid grid dt (D4) the armature is sized for
ROD_CLAMP_NUMBER = "rod_clamp"  # decision number: weld link 0 to the world (add_rod(clamp="start", route="chain"))
ARMATURE_SAFETY = 2.0
ARMATURE_FLOOR = 1e-6       # kg·m², every rotational CHAIN dof incl. the root (module docstring)


@dataclass(frozen=True)
class ChainHandles:
    bodies: tuple[int, ...]
    joints: tuple[int, ...]
    shapes: tuple[int, ...]
    lengths: tuple[float, ...]


@dataclass(frozen=True)
class ChainOptions:
    capability: str = "mujoco"
    dt: float = CHAIN_DT_DEFAULT
    clamp: bool = False                       # weld link 0 to the world (cantilever self-test)
    pin: bool = False                         # ball joint from the world to link 0's outer end (laws A11/R9)
    collide: bool = True
    cfg: newton.ModelBuilder.ShapeConfig | None = None


def link_frames(points: Sequence[Sequence[float]] | None, length: float, n: int) -> tuple[np.ndarray, np.ndarray,
                                                                                         np.ndarray]:
    """Link start points [N+1, 3], orientations [N, 4] (float64 quaternions x y z w, x along each link) and arc
    lengths [N] (half-length-end layout).

    Straight rods lie on +x from the origin; a polyline is resampled by arc length (chords between link ends),
    with orientations parallel-transported so the rest twist is zero. The transport runs in float64 (`_transport`).
    """
    l = length / (n - 1)
    s = np.concatenate([[0.0], 0.5 * l + l * np.arange(n - 1), [length]])
    if points is None:
        nodes = np.stack([s, np.zeros_like(s), np.zeros_like(s)], axis=1)
    else:
        pts = np.asarray(points, np.float64)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        s = s * (arc[-1] / length)
        nodes = np.stack([np.interp(s, arc, pts[:, k]) for k in range(3)], axis=1)
    quats = np.zeros((n, 4))
    prev, prev_dir = np.array([0.0, 0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    for i in range(n):
        d = nodes[i + 1] - nodes[i]
        cur = d / np.linalg.norm(d)
        prev = _transport(prev, prev_dir, cur)
        prev_dir = cur
        quats[i] = prev
    arcs = np.diff(np.concatenate([[0.0], 0.5 * l + l * np.arange(n - 1), [length]]))
    return nodes, quats, arcs


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions stored x y z w (warp's layout), float64."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _transport(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotate frame q by the minimal rotation taking unit direction a onto b, in float64 with the angle from
    atan2(|a x b|, a . b). A float32 `cos > 1 - 1e-12` test dropped every bend under about 3e-4 rad per link (WP8:
    A12's 1e-4 L imperfection, 2e-7 to 9e-6 rad per link, came out as a straight column)."""
    axis = np.cross(a, b)
    sin = float(np.linalg.norm(axis))
    cos = float(np.dot(a, b))
    if sin == 0.0:
        if cos > 0.0:
            return q
        axis = np.cross(a, (0.0, 0.0, 1.0)) if abs(a[2]) < 0.9 else np.cross(a, (0.0, 1.0, 0.0))   # antiparallel: π
        sin, angle = float(np.linalg.norm(axis)), math.pi
    else:
        angle = math.atan2(sin, cos)
    half = 0.5 * angle
    r = np.array([*(axis / sin * math.sin(half)), math.cos(half)])
    return _qmul(r, q)


def _link_inertia(rho: float, r: float, li: float, arc: float) -> tuple[float, wp.mat33]:
    """Mass from the link's arc length (a curved card keeps ρ·A·L), inertia of the chord-length cylinder."""
    m = rho * math.pi * r * r * arc
    it, ia = m * (3.0 * r * r + li * li) / 12.0, 0.5 * m * r * r
    return m, wp.mat33(ia, 0.0, 0.0, 0.0, it, 0.0, 0.0, 0.0, it)


def _damping(rod: RodCard, k: float, w1: float, opts: ChainOptions) -> float:
    """Stiffness-proportional damping kd = k·τ with τ = 2ζ/ω1 (ζ = rod.damping_ratio at the first mode)."""
    tau = 2.0 * rod.damping_ratio / w1 if w1 > 0 else 0.0
    if rod.integrator == "implicitfast":
        tau = max(tau, 0.5 * opts.dt)
    return k * tau


def _dofs(rod: RodCard, l: float, inertia: wp.mat33, w1: float, opts: ChainOptions) -> tuple[list, list[float]]:
    Dof = newton.ModelBuilder.JointDofConfig
    kb, kt = rod.EI / l, rod.GJ / l
    arm_b = arm_t = 0.0
    if rod.integrator == "implicitfast":
        h2 = opts.dt * opts.dt
        arm_b = max(0.0, ARMATURE_SAFETY * kb * h2 / 4.0 - float(inertia[1, 1]))
        arm_t = max(0.0, ARMATURE_SAFETY * kt * h2 / 4.0 - float(inertia[0, 0]))
    arm_b, arm_t = max(arm_b, ARMATURE_FLOOR), max(arm_t, ARMATURE_FLOOR)
    axes = (((0.0, 1.0, 0.0), kb, arm_b), ((0.0, 0.0, 1.0), kb, arm_b), ((1.0, 0.0, 0.0), kt, arm_t))
    mujoco = opts.capability == "mujoco"
    dofs = [Dof(axis=ax, armature=arm, damping=_damping(rod, k, w1, opts),
                target_ke=0.0 if mujoco else k, target_pos=0.0) for ax, k, arm in axes]
    return dofs, [kb, kb, kt]


def first_mode(rod: RodCard) -> float:
    """Clamped Euler-Bernoulli first bending mode ω1 = β1²·√(EI/(ρA·L⁴)) [rad/s] (sets the damping time)."""
    lin = rod.density * math.pi * rod.radius**2
    return 1.8751040687119611**2 * math.sqrt(rod.EI / (lin * rod.length**4))


def build_chain(b: newton.ModelBuilder, rod: RodCard, points: Sequence | None = None,
                opts: ChainOptions = ChainOptions(), label: str = "chain") -> ChainHandles:
    """Emit the chain of `rod`; the root link gets a free joint, or a fixed weld when `opts.clamp`."""
    n = rod.links
    nodes, quats, arcs = link_frames(points, rod.length, n)
    w1 = first_mode(rod)
    cfg = opts.cfg or newton.ModelBuilder.ShapeConfig(density=0.0)
    cfg.density = 0.0
    if not opts.collide:
        cfg.has_shape_collision = False
        cfg.has_particle_collision = False
    l = rod.length / (n - 1)
    bodies, joints, shapes, lens = [], [], [], []
    for i in range(n):
        li = float(np.linalg.norm(nodes[i + 1] - nodes[i]))
        mass, inertia = _link_inertia(rod.density, rod.radius, li, float(arcs[i]))
        com = wp.vec3(*(0.5 * (nodes[i] + nodes[i + 1])))
        body = b.add_link(xform=wp.transform(com, wp.quat(*quats[i])), mass=mass, inertia=inertia,
                          label=f"{label}/{i}")
        shapes.append(b.add_shape_capsule(body, radius=rod.radius, half_height=max(0.5 * li - rod.radius, 1e-6),
                                          xform=wp.transform(wp.vec3(), Z_TO_X), cfg=cfg))
        joints.append(_joint(b, body, i, bodies, quats, lens, li, (rod, l, inertia, w1, opts), com))
        bodies.append(body)
        lens.append(li)
    b.add_articulation(joints, label=label)
    return ChainHandles(tuple(bodies), tuple(joints), tuple(shapes), tuple(lens))


def _floor_angular(b: newton.ModelBuilder, joint: int) -> int:
    """ARMATURE_FLOOR on the angular dofs of `joint` (the root's free or ball joint); returns the joint."""
    q0 = b.joint_qd_start[joint]
    lin, ang = b.joint_dof_dim[joint]
    for d in range(q0 + lin, q0 + lin + ang):
        b.joint_armature[d] = max(b.joint_armature[d], ARMATURE_FLOOR)
    return joint


def _joint(b: newton.ModelBuilder, body: int, i: int, bodies: list, quats: list, lens: list, li: float,
           dof_args: tuple, com: wp.vec3) -> int:
    rod, l, inertia, w1, opts = dof_args
    if i == 0:
        root = wp.transform(com, wp.quat(*quats[0]))
        if opts.clamp:
            return b.add_joint_fixed(-1, body, parent_xform=root, child_xform=wp.transform_identity())
        if opts.pin:
            end = wp.transform(wp.vec3(-0.5 * li, 0.0, 0.0), wp.quat_identity())
            return _floor_angular(b, b.add_joint_ball(-1, body, parent_xform=wp.transform_multiply(root, end),
                                                      child_xform=end))
        return _floor_angular(b, b.add_joint_free(body, parent_xform=root, child_xform=wp.transform_identity()))
    q_rel = wp.quat(*_qmul(_qconj(quats[i - 1]), quats[i]))      # float64: keeps sub-µrad bends (A12)
    pxf = wp.transform(wp.vec3(0.5 * lens[-1], 0.0, 0.0), q_rel)
    cxf = wp.transform(wp.vec3(-0.5 * li, 0.0, 0.0), wp.quat_identity())
    dofs, ks = _dofs(rod, l, inertia, w1, opts)
    extra = {"custom_attributes": {"mujoco:dof_passive_stiffness": ks}} if opts.capability == "mujoco" else {}
    return b.add_joint_d6(bodies[-1], body, parent_xform=pxf, child_xform=cxf, angular_axes=dofs, **extra)


def emit_chain(b: newton.ModelBuilder, obj: ResolvedObject, d: RouteDecision) -> None:
    """CHAIN route: the router's RodCard (decision numbers) along the card's curve or the rod axis (+x)."""
    from alloy.route.rodcheck import rod_card_from_decision

    rod = rod_card_from_decision(obj, d)
    opts = ChainOptions(capability=d.entry, dt=float(d.numbers.get("chain_dt", CHAIN_DT_DEFAULT)),
                        clamp=bool(d.numbers.get("tree_root", 0.0)        # a tree weld's root (SCR-WP2-9)
                                   or d.numbers.get(ROD_CLAMP_NUMBER, 0.0)),  # or an API clamp (S4)
                        cfg=shape_config(obj.material, density=0.0))
    build_chain(b, rod, rod_points(obj, rod.length), opts, label=obj.card.name)


def rod_points(obj: ResolvedObject, length: float) -> list[tuple[float, float, float]]:
    """The rod's centre line in the object frame: the card curve, else the primitive's axis (Newton Z for
    capsules/cylinders, the longest axis of a box), centred on the object origin."""
    shape = obj.card.shape
    if isinstance(shape, CurveSource):
        return [tuple(map(float, p)) for p in shape.points]
    axis = np.zeros(3)
    kind = getattr(shape, "kind", None)
    if kind == "box":
        axis[int(np.argmax(shape.size[:3]))] = 1.0
    else:
        axis[2] = 1.0
    h = 0.5 * length * axis
    return [tuple(-h), tuple(h)]
