# SPDX-License-Identifier: Apache-2.0
"""Pad drag law rigs (SCR-WP11-6's full-surface "against" drag; SCR-SC4-4's edge/face soft contacts): does a pad
pressed onto cloth on a frictionless table drag the cloth its own way, with per-vertex and full-surface contacts?

By Coulomb a pressed pad with friction drags cloth that lies on a frictionless table along with it: nothing holds
the cloth back but its inertia. ``run_plain_drag``: one SolverVBD, a kinematic mesh-box pad (C7's 16 x 6.4 x 20 mm
YAM pad tip) pressed onto C7's towel and slid along +y. ``run_flat_pinch``: WP11's pinch-from-flat rig ([internal note]
notes/wp11-grasp/flat_pinch.py) through the lagged MuJoCo -> VBD Scheduler: two pads on y sliders under a z-servo
carriage press about 2 N each and close; ``FlatPinch.lone`` opens the jaw to 100 mm so pad b stays on the bare table
and pad a drags the cloth alone.

With both pads on the cloth the friction is not converged at the towel's 10 VBD iterations: the two pads' friction
differs and the whole cloth drifts (WP11's 3.9 mm "against" the pad was that drift; 160 iterations bring it to
0.15 mm). That is WP11's SCR-WP11-6 step 2, not a sign error in the edge/face records: a lone pad drags the cloth
its own way in both contact modes. Per-vertex contacts gather that cloth into a ridge only by pushing cloth triangles
into the pads (``FlatPinchResult.tris_in_pads``), so they are no reference for it. Rows: docs/notes/sc4-pad-drag.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.coupling.pinch_rigs import (CELL, GRIP_KP, OPEN_GAP, PAD_HALF, PAD_MASS, RADIUS, SHAPE_KE, SOFT_KE,
                                       TOWEL, PinchPhase, _cloth_scheduler, _finalize, _run_script, mesh_box)
from alloy.coupling.proxy_contacts import ProxyContactPipeline

CARRIAGE_KG = 1.0
PRESS_KE = 2.0e3                # soft carriage z servo [N/m] (2 N per mm), so the press is set by force


@dataclass(frozen=True)
class PadDragResult:
    pad_travel: float           # pad (a) travel along +y [m]
    under_travel: float         # mean y travel of the two cloth vertices nearest the pad's centre at the start [m]
    cloth_travel: float         # mean y travel of the whole cloth [m]
    tris_in_pads: int           # cloth triangles with a sample point > 0.1 mm inside a pad at the end
    finite: bool


def _cloth_grid(b: newton.ModelBuilder, nx: int, ny: int, x0: float, y0: float, cell: float) -> None:
    b.add_cloth_grid(pos=wp.vec3(x0, y0, RADIUS), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=nx, dim_y=ny,
                     cell_x=cell, cell_y=cell, mass=TOWEL.mass * (cell / CELL) ** 2, tri_ke=1.0e3, tri_ka=1.0e3,
                     tri_kd=1.0e-2, edge_ke=1.0e-3, edge_kd=1.0e-5, particle_radius=RADIUS)


def _table(b: newton.ModelBuilder, mu: float) -> None:
    cfg = b.default_shape_cfg.copy()
    cfg.mu = mu
    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.01), wp.quat_identity()), hx=0.2, hy=0.2, hz=0.01, cfg=cfg)


def _nearest(q: np.ndarray, xy: np.ndarray, k: int = 2) -> np.ndarray:
    return np.argsort(np.linalg.norm(q[:, :2] - xy[:2], axis=1))[:k]


def tris_in_boxes(q: np.ndarray, tris: np.ndarray, centres: np.ndarray, half: tuple[float, float, float],
                  shrink: float = 1.0e-4) -> int:
    """Cloth triangles (28 barycentric sample points each) with a point inside an axis-aligned box ``half``
    (less ``shrink``) around any of ``centres``: cloth that went into a pad."""
    g = np.array([(i / 6, j / 6) for i in range(7) for j in range(7 - i)])
    w = np.stack([1.0 - g.sum(1), g[:, 0], g[:, 1]], 1)
    pts = np.einsum("sk,tkd->tsd", w, q[tris])
    hit = np.zeros(len(tris), bool)
    for c in np.atleast_2d(centres):
        hit |= ((np.abs(pts - c[:3]) < np.asarray(half) - shrink).all(-1)).any(1)
    return int(hit.sum())


@dataclass(frozen=True)
class PlainDrag:
    """``run_plain_drag``: the pad presses ``depth`` into the cloth's top, settles 0.1 s, slides at ``speed``."""
    mu_pad: float = 2.0
    mu_table: float = 0.0
    depth: float = 1.0e-5
    dt: float = 1.0e-3
    speed: float = 0.0375
    seconds: float = 0.2
    iterations: int = 10


def run_plain_drag(full_surface: bool, law: PlainDrag = PlainDrag()) -> PadDragResult:
    """One SolverVBD; the kinematic pad's pose is set every step (body_qd 0), per-vertex or full-surface contacts."""
    b = newton.ModelBuilder()
    b.default_shape_cfg.ke, b.default_shape_cfg.kd = SHAPE_KE, 1.0e3
    _table(b, law.mu_table)
    pcfg = b.default_shape_cfg.copy()
    pcfg.mu = law.mu_pad
    z = PAD_HALF[2] + 2.0 * RADIUS - law.depth
    pad = b.add_body(xform=wp.transform((0.0, -0.0195, z), wp.quat_identity()), mass=0.5, is_kinematic=True,
                     label="pad")
    shape = b.add_shape_mesh(pad, mesh=mesh_box(PAD_HALF), cfg=pcfg)
    _cloth_grid(b, 7, 11, -0.03, -0.045, CELL)
    b.color()
    m = b.finalize(device="cpu")
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = SOFT_KE, 1.0e-2, 0.45
    solver = newton.solvers.SolverVBD(m, iterations=law.iterations)
    pipe = (ProxyContactPipeline(full_surface=True, full_surface_shapes=[shape], soft_contact_gap=0.002)(m)
            if full_surface else newton.CollisionPipeline(m, soft_contact_gap=0.002))
    contacts = pipe.contacts()
    s0, s1, ctl = m.state(), m.state(), m.control()
    q0, bq0 = s0.particle_q.numpy().copy(), s0.body_q.numpy()[pad].copy()
    under = _nearest(q0, bq0)
    n_settle, n_move = int(round(0.1 / law.dt)), int(round(law.seconds / law.dt))
    qd = np.zeros((m.body_count, 6), np.float32)
    for k in range(n_settle + n_move):
        bq = s0.body_q.numpy()
        bq[pad, :3] = bq0[:3]
        bq[pad, 1] += law.speed * max(0, k - n_settle) * law.dt
        s0.body_q.assign(bq)
        s0.body_qd.assign(qd)
        s0.clear_forces()
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctl, contacts, law.dt)
        s0, s1 = s1, s0
    q, bq = s0.particle_q.numpy(), s0.body_q.numpy()
    return PadDragResult(pad_travel=float(bq[pad, 1] - bq0[1]), under_travel=float((q[under, 1] - q0[under, 1]).mean()),
                         cloth_travel=float((q[:, 1] - q0[:, 1]).mean()),
                         tris_in_pads=tris_in_boxes(q, m.tri_indices.numpy(), bq[pad:pad + 1], PAD_HALF),
                         finite=bool(np.isfinite(q).all()))


@dataclass(frozen=True)
class FlatPinch:
    """``run_flat_pinch`` (WP11's flat_pinch.py): pads 10 mm up are lowered over 0.25 s (0.4 s phase) to press about
    ``press`` on the cloth, then close for ``close`` s at ``grip`` N. The cloth's +y edge is ``inset`` beyond the jaw
    centre line; 0.04675 centres C7's 11-cell towel on the jaw (the two pads see mirror-image cloth)."""
    lone: bool = False
    inset: float = 0.04675
    width: float = 0.06
    length: float = 0.09
    cell: float = CELL
    mu_pad: float = 2.0
    mu_table: float = 0.0
    mu_cloth: float = 0.45
    press: float = 4.0
    grip: float = 15.0
    close: float = 0.15
    iterations: int = 10


def _flat_model(law: FlatPinch) -> tuple[newton.Model, float, float]:
    """Static table, carriage on a z servo, pads a (-y) and b (+y) on y sliders, cloth lying flat; (model, pad start
    centre height, carried weight)."""
    from alloy._vendor.newton.solvers import SolverMuJoCo

    gap = 0.100 if law.lone else OPEN_GAP
    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.ke, b.default_shape_cfg.kd, b.default_shape_cfg.mu = SHAPE_KE, 1.0e3, law.mu_pad
    _table(b, law.mu_table)
    z0 = PAD_HALF[2] + 0.010
    lift = b.add_link(xform=wp.transform((0.0, 0.0, z0 + 0.06), wp.quat_identity()), label="carriage")
    ccfg = b.default_shape_cfg.copy()
    ccfg.density = CARRIAGE_KG / 8.0e-6
    b.add_shape_box(lift, hx=0.01, hy=0.01, hz=0.01, cfg=ccfg)
    joints = [b.add_joint_prismatic(-1, lift, axis=(0.0, 0.0, 1.0), label="lift", target_ke=PRESS_KE,
                                    parent_xform=wp.transform((0.0, 0.0, z0 + 0.06), wp.quat_identity()),
                                    target_kd=2.0 * math.sqrt(PRESS_KE * 2.1),
                                    actuator_mode=newton.JointTargetMode.POSITION)]
    pcfg = b.default_shape_cfg.copy()
    pcfg.density = PAD_MASS / (8 * PAD_HALF[0] * PAD_HALF[1] * PAD_HALF[2])
    for sign, label in ((-1.0, "pad_a"), (1.0, "pad_b")):
        y = sign * (0.5 * gap + PAD_HALF[1])
        pad = b.add_link(xform=wp.transform((0.0, y, z0), wp.quat_identity()), label=label)
        b.add_shape_mesh(pad, mesh=mesh_box(PAD_HALF), cfg=pcfg)
        joints.append(b.add_joint_prismatic(lift, pad, axis=(0.0, 1.0, 0.0), label=label + "_slide",
                                            parent_xform=wp.transform((0.0, y, -0.06), wp.quat_identity()),
                                            target_ke=GRIP_KP, target_kd=2.0 * math.sqrt(GRIP_KP * PAD_MASS),
                                            actuator_mode=newton.JointTargetMode.POSITION))
    b.add_articulation(joints)
    inset, length = (0.020, 0.110) if law.lone else (law.inset, law.length)   # lone: pad a on the cloth, b off it
    nx, ny = int(round(law.width / law.cell)), int(round(length / law.cell))
    _cloth_grid(b, nx, ny, -0.5 * nx * law.cell, inset - ny * law.cell, law.cell)
    m = _finalize(b, SOFT_KE)
    m.soft_contact_mu = law.mu_cloth
    return m, z0, (CARRIAGE_KG + 2 * PAD_MASS) * 9.81


def run_flat_pinch(full_surface: bool, law: FlatPinch = FlatPinch()) -> PadDragResult:
    """WP11's pinch from flat through the lagged Scheduler; the result reads pad a and the cloth after the close."""
    model, z0, weight = _flat_model(law)
    sched = _cloth_scheduler(model, full_surface, law.iterations, "lagged", RADIUS)
    down = -(z0 - PAD_HALF[2]) + 2.0 * RADIUS + (weight - law.press) / PRESS_KE  # bottoms on the cloth, pressing
    lower = PinchPhase("lower", 0.40, down, False, ramp=0.25)
    script = (lower, PinchPhase("close", lower.end + law.close, down, True))
    seen: dict[str, np.ndarray] = {}

    def observe(ph: PinchPhase, k: int, s: newton.State) -> None:
        if ph.name == "lower":
            seen["q0"], seen["bq0"] = s.particle_q.numpy().copy(), s.body_q.numpy().copy()

    s = _run_script(sched, model, script, observe, law.grip, 40.0, half_gap=0.5 * (0.100 if law.lone else OPEN_GAP))
    q, bq, q0, bq0 = s.particle_q.numpy(), s.body_q.numpy(), seen["q0"], seen["bq0"]
    under = _nearest(q0, bq0[1])
    return PadDragResult(pad_travel=float(bq[1, 1] - bq0[1, 1]),
                         under_travel=float((q[under, 1] - q0[under, 1]).mean()),
                         cloth_travel=float((q[:, 1] - q0[:, 1]).mean()),
                         tris_in_pads=tris_in_boxes(q, model.tri_indices.numpy(), bq[1:3], PAD_HALF),
                         finite=bool(np.isfinite(q).all()))
