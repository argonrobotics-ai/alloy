# SPDX-License-Identifier: Apache-2.0
"""ROD_VBD emitter (SCR-WP9-ROD): a rod card as Newton's discrete Cosserat rod for `alloy.solvers.rod.SolverVBDRod`.

`ModelBuilder.add_rod` makes capsule segments along their local z and one ROD joint between neighbours, with
stretch, shear, bend and twist. SolverVBDRod solves each chain with one Gauss-Newton direct step.

**Layout:** D7, as the CHAIN emitter (`chain.link_frames`). N segments, half-length end segments, joints
h = L / (N - 1) apart.

**Per joint:**
- stretch E A / h;
- shear kappa G A / h on both transverse axes, with Timoshenko's kappa = 6 (1 + nu) / (7 + 6 nu) for a solid round
  section and G A = E A / (2 (1 + nu)) from the card's EA. For an isotropic card that is the same G A as from GJ; a
  card with a measured, softer EI or GJ (a cable's composite) keeps its jacket's shear;
- bend E I / h on both axes;
- twist G J / h.
- So a clamped rod's tip twists T L / (G J) under an end torque T (N - 1 joints in series), and its bends match
  CHAIN's.
- E and G come from the card's EI and GJ (I = pi r^4 / 4, J = 2 I), and nu = EI / GJ - 1.

**Mass and inertia:** each segment is a solid cylinder of its own arc length, as CHAIN's `_link_inertia`; the
capsule's caps add none. So m = rho A l_i, I_axis = m r^2 / 2, I_t = m (3 r^2 + l_i^2) / 12.
- No armature. Twist and rolling keep their physical inertia: applied torques enter the solve as a potential (the
  solver's docstring, Applied torques). An armature would stall rolling, since axial spin is rolling.

**Damping:** stiffness-proportional on bend and twist, kd = k tau, with tau = 2 zeta / omega1 (zeta the card's damping
ratio at the first bending mode; CHAIN's `_damping` without its implicitfast floor).
- Stretch and shear are left undamped; backward Euler damps their stiff modes anyway.
- Damping them as well put kd / dt near 1e8 on those slots. float32 rounding of (u - u_prev) then became a force
  far above a segment's weight: A11.cantilever's gamma-0.5 rod drooped +0.87 % past the exact discrete chain. With
  bend and twist damping only it is -0.069 %.

**Clamp:** "start", "end" or "both" make that end segment kinematic, held at its built pose.

**Pin:** "start", "end" or "both" hold that end node in place and leave it free to turn (laws A11.hanging_chain and
A11.catenary). Each pin is a ball joint from the world to the end segment, a loop closure outside the rod's
articulation; add_rod's own articulation (a free root joint) stays, so Newton's forward kinematics matches the build.
SolverVBDRod evaluates the pins with Newton's own joint terms (its non-chain joints).

**Routed rods** (`emit_rod_vbd`, ROD_VBD): the router's RodCard from the decision numbers, along the card's curve or
the primitive's axis (`chain.rod_points`), with the card's surface values. A tree weld's root (`tree_root`) is
clamped at the start, as CHAIN welds it.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.cards.types import ResolvedObject
from alloy.route.types import RodCard, RouteDecision
from alloy.scene.emitters.chain import ChainHandles, first_mode, link_frames, rod_points
from alloy.scene.emitters.common import shape_config

Clamp = Literal["start", "end", "both"] | None
_CLAMPED = {"start": (0,), "end": (-1,), "both": (0, -1), None: ()}


def joint_stiffness(rod: RodCard, h: float) -> dict[str, float]:
    """The ROD joints' stiffnesses for joints h apart (module docstring)."""
    r = rod.radius
    area, inertia = math.pi * r * r, 0.25 * math.pi * r ** 4
    nu = min(max(rod.EI / rod.GJ - 1.0, 0.0), 0.5)
    kappa = 6.0 * (1.0 + nu) / (7.0 + 6.0 * nu)
    return {"stretch": rod.EA / h, "shear": kappa * rod.EA / (2.0 * (1.0 + nu)) / h, "bend": rod.EI / h,
            "twist": rod.GJ / h}


def damping_time(rod: RodCard) -> float:
    """tau = 2 zeta / omega1 [s], the stiffness-proportional damping time (0 for an undamped card)."""
    w1 = first_mode(rod)
    return 2.0 * rod.damping_ratio / w1 if w1 > 0.0 and rod.damping_ratio > 0.0 else 0.0


def build_rod_vbd(b: newton.ModelBuilder, rod: RodCard, points: Sequence[Sequence[float]] | None, *,
                  clamp: Clamp = None, pin: Clamp = None, cfg: newton.ModelBuilder.ShapeConfig | None = None,
                  label: str = "rod", plastic: tuple[float, float] | None = None) -> ChainHandles:
    """Add `rod` to `b` along `points` (a polyline, resampled by arc length; None: +x from the origin) with
    `rod.links` segments. `cfg` carries the segments' surface values (default Newton's, density from the card).
    `plastic`: (σ_y0, H) [Pa] makes the ROD joints J2 (alloy.solvers.rod.plastic). Returns the segment bodies, the ROD
    joints, the shapes and each segment's arc length."""
    n = int(rod.links)
    if n < 2:
        raise ValueError(f"ROD_VBD rod {label!r}: needs at least 2 segments, got {n}")
    for key, val in (("clamp", clamp), ("pin", pin)):
        if val not in _CLAMPED:
            raise ValueError(f"ROD_VBD rod {label!r}: {key} must be 'start', 'end', 'both' or None, got {val!r}")
    nodes, _, arcs = link_frames(points, rod.length, n)
    h = float(np.sum(arcs)) / (n - 1)
    k = joint_stiffness(rod, h)
    tau = damping_time(rod)
    shape = newton.ModelBuilder.ShapeConfig(density=rod.density) if cfg is None else cfg
    bodies, joints = b.add_rod(
        rod=newton.Rod(nodes.astype(np.float32), radius=rod.radius), cfg=shape,
        stretch_stiffness=k["stretch"], stretch_damping=0.0,
        shear_stiffness=k["shear"], shear_damping=0.0,
        bend_stiffness=k["bend"], bend_damping=k["bend"] * tau,
        twist_stiffness=k["twist"], twist_damping=k["twist"] * tau, label=label)
    _set_segment_inertia(b, bodies, rod, nodes, arcs)
    if plastic is not None:
        from alloy.solvers.rod.plastic import mark_joints  # noqa: PLC0415

        mark_joints(b, joints, plastic[0], plastic[1], rod.radius, h)
    for i in _CLAMPED[clamp]:
        b.body_flags[bodies[i]] = int(newton.BodyFlags.KINEMATIC)
    for i in _CLAMPED[pin]:
        _pin(b, bodies[i], nodes[0 if i == 0 else -1], label)
    shapes = tuple(int(s) for bi in bodies for s in b.body_shapes[bi])
    return ChainHandles(tuple(int(x) for x in bodies), tuple(int(j) for j in joints), shapes,
                        tuple(float(a) for a in arcs))


def _pin(b: newton.ModelBuilder, body: int, node: np.ndarray, label: str) -> int:
    """A ball joint from the world at `node` (at build) to `body` (module docstring, Pin). Newton takes a world-root
    joint outside the rod's articulation as a loop closure."""
    tf = b.body_q[body]
    p = wp.vec3(*(float(v) for v in node))
    local = wp.quat_rotate_inv(tf.q, p - tf.p)
    with warnings.catch_warnings():         # beside the root's FREE joint, which constrains nothing in VBD
        warnings.filterwarnings("ignore", message="Adding a BALL joint between parent -1")
        return b.add_joint_ball(-1, body, parent_xform=wp.transform(p, wp.quat_identity()),
                                child_xform=wp.transform(local, wp.quat_identity()), label=f"{label}_pin{body}")


def _set_segment_inertia(b: newton.ModelBuilder, bodies: Sequence[int], rod: RodCard, nodes: np.ndarray,
                         arcs: np.ndarray) -> None:
    """Each segment a solid cylinder of its arc length (mass) and chord length (inertia), axis along local z."""
    r, rho = rod.radius, rod.density
    for i, bi in enumerate(bodies):
        li = float(np.linalg.norm(nodes[i + 1] - nodes[i]))
        m = rho * math.pi * r * r * float(arcs[i])
        it, ia = m * (3.0 * r * r + li * li) / 12.0, 0.5 * m * r * r
        b.body_mass[bi], b.body_inv_mass[bi] = m, 1.0 / m
        b.body_inertia[bi] = wp.mat33(it, 0.0, 0.0, 0.0, it, 0.0, 0.0, 0.0, ia)
        b.body_inv_inertia[bi] = wp.mat33(1.0 / it, 0.0, 0.0, 0.0, 1.0 / it, 0.0, 0.0, 0.0, 1.0 / ia)


def emit_rod_vbd(b: newton.ModelBuilder, obj: ResolvedObject, d: RouteDecision) -> ChainHandles:
    """ROD_VBD route (module docstring, Routed rods)."""
    from alloy.route.rodcheck import rod_card_from_decision

    rod = rod_card_from_decision(obj, d)
    clamp: Clamp = "start" if d.numbers.get("tree_root", 0.0) else None
    return build_rod_vbd(b, rod, rod_points(obj, rod.length), clamp=clamp,
                         cfg=shape_config(obj.material, density=0.0), label=obj.card.name, plastic=j2_yield(obj))


def j2_yield(obj: ResolvedObject) -> tuple[float, float] | None:
    """(σ_y0, H) [Pa] of a `plastic.model: j2` material, else None."""
    p = obj.material.plastic
    if p is None or p.model != "j2" or p.yield_stress is None:
        return None
    return float(p.yield_stress.value), float(p.hardening_modulus.value) if p.hardening_modulus is not None else 0.0


__all__ = ["Clamp", "build_rod_vbd", "damping_time", "emit_rod_vbd", "j2_yield", "joint_stiffness"]
