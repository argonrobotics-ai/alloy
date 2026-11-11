# SPDX-License-Identifier: Apache-2.0
"""SCR-SC4-4 CPU rigs for rigid features against coarse cloth (MuJoCo bodies, VBD cloth, lagged Scheduler).

C7's towel has 8.5 mm edges and 1 mm particles, so its vertices leave 6.5 mm gaps; the YAM pad tips are 6.4 mm thick
with 2.1 mm ridges. VBD's per-vertex body contact never sees a feature passing between vertices, so the pads thread
through the cloth and lift it when they open. The rigs here reproduce that at test size:

- ``run_blade``: a 2 x 4 mm mesh blade on a servo slider pushed 15 mm through a hanging cloth, aimed at a cell centre
  (every vertex stays > r from it). Per-vertex contact: the blade passes, the cloth does not move. Full-surface
  contact (``ProxyContactPipeline(full_surface=True)``): the cloth is pushed ahead of the blade.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.newton_compat import SolverCoupledProxy
from alloy.coupling.proxy_contacts import ProxyContactPipeline
from alloy.coupling.scheduler import Scheduler
from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy

CELL, RADIUS = 0.0085, 0.001          # C7's towel: edge [m], particle radius [m]
SOFT_KE, SHAPE_KE = 1.0e4, 2.5e5      # C7's soft / pad contact stiffness [N/m]


@dataclass(frozen=True)
class Cloth:
    """A rig cloth: edge and particle radius [m], vertex mass [kg], VBD ``soft_contact_ke`` [N/m] (its self-contact
    and the soft half of every body-contact record)."""

    cell: float
    radius: float
    mass: float
    soft_ke: float


TOWEL = Cloth(CELL, RADIUS, 2.0e-5, SOFT_KE)        # C7's terry towel
FILM = Cloth(0.010, 5.0e-4, 1.4e-6, 3.0e2)          # C16's HDPE film: 1 cm cells, 1.4 mg vertices, soft ke 300
_BOX_FACES = np.array([0, 1, 3, 0, 3, 2, 4, 6, 7, 4, 7, 5, 0, 4, 5, 0, 5, 1, 2, 3, 7, 2, 7, 6, 0, 2, 6, 0, 6, 4,
                       1, 5, 7, 1, 7, 3], np.int32)


def mesh_box(half: tuple[float, float, float]) -> newton.Mesh:
    """A closed box triangle mesh (outward winding) with half sizes ``half``."""
    v = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float32) * np.asarray(half)
    return newton.Mesh(v, _BOX_FACES)


def cup_mesh(r_out: float = 0.040, r_in: float = 0.0382, height: float = 0.11, floor: float = 0.002, n: int = 96):
    """A closed thin-walled cup (JOB2's beaker: 40 mm / 38.2 mm radii), outward winding: (vertices, indices)."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1)
    v = np.concatenate([ring * [r_out, r_out, 0], ring * [r_out, r_out, 0] + [0, 0, height],
                        ring * [r_in, r_in, 0] + [0, 0, height], ring * [r_in, r_in, 0] + [0, 0, floor],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, floor]]])
    ob, ot, it, ib, c0, c1 = 0, n, 2 * n, 3 * n, 4 * n, 4 * n + 1
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[ob + i, ob + j, ot + j], [ob + i, ot + j, ot + i]]            # outer wall, normal out
        f += [[ot + i, ot + j, it + j], [ot + i, it + j, it + i]]            # rim, normal up
        f += [[it + i, it + j, ib + j], [it + i, ib + j, ib + i]]            # inner wall, normal to the axis
        f += [[ib + i, ib + j, c1]]                                          # cavity floor, normal up
        f += [[ob + j, ob + i, c0]]                                          # outer bottom, normal down
    v, f = np.asarray(v, np.float32), np.asarray(f, np.int32)
    assert np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() > 0.0   # outward
    return v, f.ravel()


def segment_crossings(q: np.ndarray, tri: np.ndarray, a: np.ndarray, b: np.ndarray) -> int:
    """Cloth triangles (``tri`` into ``q``) crossed by the segment a -> b (Moller-Trumbore)."""
    p0, p1, p2 = q[tri[:, 0]], q[tri[:, 1]], q[tri[:, 2]]
    d = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    e1, e2 = p1 - p0, p2 - p0
    h = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = np.asarray(a, np.float64) - p0
    u = np.einsum("ij,ij->i", s, h) * inv
    qv = np.cross(s, e1)
    v = (qv @ d) * inv
    t = np.einsum("ij,ij->i", e2, qv) * inv
    return int((ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0) & (t <= 1)).sum())


def _cloth_scheduler(model: newton.Model, full_surface: bool, iterations: int = 10, mode: str = "lagged",
                     radius: float = RADIUS, friction_epsilon: float = 1.0e-3,
                     policy: SchedulerPolicy | None = None, solver_kwargs: dict | None = None,
                     route: bool = False, on_solver: Callable[[Any], None] | None = None) -> Scheduler:
    """MuJoCo -> VBD lagged (or staggered) Scheduler; with ``full_surface`` the VBD entry also gets make_stepper's
    proxy contact view (reduced hulls) for the moving shapes. ``policy``: the Scheduler's (default
    SchedulerPolicy()). ``on_solver(solver)`` runs on the VBD solver once it is built (e.g. attach-on-grasp)."""
    from alloy._vendor.newton.solvers import SolverMuJoCo

    from alloy.coupling.full_surface import contact_view

    moving = np.flatnonzero(model.shape_body.numpy() >= 0)                 # the plywood keeps per-vertex contact
    view = contact_view(moving)
    e_m = SolverCoupledProxy.Entry(name="mjc", solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False,
                                                                             njmax=50),
                                   bodies=list(range(model.body_count)), joints=list(range(model.joint_count)))
    e_v = SolverCoupledProxy.Entry(name="vbd", particles=list(range(model.particle_count)),
                                   solver=lambda v: _built((_route_vbd if route else _vbd)(
                                       v, radius, iterations, friction_epsilon, solver_kwargs), on_solver),
                                   configure_view=view if full_surface else None)
    pipe = ProxyContactPipeline(full_surface=full_surface, full_surface_shapes=moving, soft_contact_gap=0.002)
    relax = "aitken" if mode == "staggered" else "fixed"
    proxy = SolverCoupledProxy.Proxy(source="mjc", destination="vbd", bodies=list(range(model.body_count)),
                                     mode=mode, proxy_relaxation_mode=relax, collision_pipeline=pipe,
                                     collide_interval=1)
    return Scheduler(model, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False)],
                     [PairPolicy("mjc", "vbd", mode=mode, proxy=proxy)], policy or SchedulerPolicy())


def _built(solver: Any, on_solver: Callable[[Any], None] | None) -> Any:
    if on_solver is not None:
        on_solver(solver)
    return solver


def _vbd(view: Any, r: float, iterations: int = 10, friction_epsilon: float = 1.0e-3,
         extra: dict | None = None) -> Any:
    kw = dict(iterations=iterations, friction_epsilon=friction_epsilon, particle_enable_self_contact=True,
              particle_self_contact_radius=r, particle_self_contact_margin=2.0 * r,
              rigid_body_particle_contact_buffer_size=2048)
    return newton.solvers.SolverVBD(view, **{**kw, **(extra or {})})


ROUTE_VBD_KWARGS = {"alloy_initial_guess": "adaptive", "alloy_chebyshev_rho": 0.9,
                    "alloy_chebyshev_hold_self_contact": True,
                    "particle_enable_self_contact": True, "particle_self_contact_margin": 0.002,
                    "alloy_self_contact_speed": 3.0}       # the shipped cloth route's kwargs (cloth_fold_scene, 11-10)
ROUTE_ITERATIONS = 40                                       # ... and its 40 iterations (cloth_accuracy default)
STICK_KEY = "body_particle_friction_stick"                  # WP11's opt-in stick friction (branch wp11-friction-stick)


def _route_vbd(view: Any, r: float, iterations: int = ROUTE_ITERATIONS, friction_epsilon: float = 1.0e-2,
               extra: dict | None = None) -> Any:
    """The VBD solver as the cloth route builds it (``alloy.solvers.vbd.make_vbd`` with the route's kwargs; the rig's
    contact list size; the self-contact margin 2 mm = 2 r is the route's, ``r`` is not passed)."""
    from alloy.solvers.vbd import make_vbd

    kw = {**ROUTE_VBD_KWARGS, "iterations": iterations, "friction_epsilon": friction_epsilon,
          "rigid_body_particle_contact_buffer_size": 2048, **(extra or {})}
    return make_vbd(view, kw)


def _finalize(b: newton.ModelBuilder, soft_ke: float = SOFT_KE) -> newton.Model:
    from alloy._vendor.newton.solvers import SolverMuJoCo

    top = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(top)
    top.add_world(b)
    top.color()
    m = top.finalize(device="cpu")
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = soft_ke, 1.0e-2, 0.45
    return m


BLADE_HALF = (0.015, 0.001, 0.002)    # long along x (its travel), 2 mm (y) x 4 mm (z) section [m]
BLADE_Z = 0.10


def blade_model(cells: int = 6) -> newton.Model:
    """A blade on an x slider (servo kp 1e4) 5 mm before a hanging cloth in the plane x = 0 (top edge fixed),
    aimed at the centre of a cell."""
    from alloy._vendor.newton.solvers import SolverMuJoCo

    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.ke, b.default_shape_cfg.kd, b.default_shape_cfg.mu = SHAPE_KE, 1.0e3, 0.4
    cfg = b.default_shape_cfg.copy()
    mass, kp = 0.05, 1.0e4
    cfg.density = mass / (8 * BLADE_HALF[0] * BLADE_HALF[1] * BLADE_HALF[2])
    x0 = -0.005 - BLADE_HALF[0]
    body = b.add_link(xform=wp.transform((x0, 0.0, BLADE_Z), wp.quat_identity()), label="blade")
    b.add_shape_mesh(body, mesh=mesh_box(BLADE_HALF), cfg=cfg)
    j = b.add_joint_prismatic(-1, body, axis=(1.0, 0.0, 0.0), label="slide", target_ke=kp,
                              parent_xform=wp.transform((x0, 0.0, BLADE_Z), wp.quat_identity()),
                              target_kd=2.0 * math.sqrt(kp * mass), actuator_mode=newton.JointTargetMode.POSITION)
    b.add_articulation([j])
    rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)     # grid x -> -z, y -> y, normal -> x
    off = (cells / 2 - 0.5) * CELL                                            # a cell centre at (0, 0, BLADE_Z)
    b.add_cloth_grid(pos=wp.vec3(0.0, -off, BLADE_Z + off), rot=rot, vel=wp.vec3(0.0), dim_x=cells, dim_y=cells,
                     cell_x=CELL, cell_y=CELL, mass=2.0e-5, fix_left=True, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=1.0e-2,
                     edge_ke=1.0e-3, edge_kd=1.0e-5, particle_radius=RADIUS)
    return _finalize(b)


@dataclass(frozen=True)
class BladeResult:
    tip_x: float              # blade tip at the end [m] (cloth plane at x = 0)
    cloth_x_max: float        # furthest cloth particle along the push [m]
    threaded: int             # cloth triangles the blade's axis crosses at the end
    finite: bool


def run_blade(full_surface: bool, travel: float = 0.020, speed: float = 0.10, seconds: float = 0.3,
              dt_x: float = 1.0 / 480.0) -> BladeResult:
    model = blade_model()
    sched = _cloth_scheduler(model, full_surface)
    s0, s1, ctl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    glob = newton.CollisionPipeline(model)
    contacts = glob.contacts()
    tgt = ctl.joint_target_q if hasattr(ctl, "joint_target_q") else ctl.joint_target_pos
    t = 0.0
    for _ in range(int(round(seconds / dt_x))):
        tgt.assign(np.array([min(travel, speed * t)], np.float32))
        s0.clear_forces()
        glob.collide(s0, contacts)
        sched.step(s0, s1, ctl, contacts, dt_x)
        s0, s1 = s1, s0
        t += dt_x
    q = s0.particle_q.numpy().astype(np.float64)
    xb = float(s0.body_q.numpy()[0, 0])
    axis = (np.array([xb - BLADE_HALF[0], 0.0, BLADE_Z]), np.array([xb + BLADE_HALF[0], 0.0, BLADE_Z]))
    return BladeResult(tip_x=xb + BLADE_HALF[0], cloth_x_max=float(q[:, 0].max()),
                       threaded=segment_crossings(q, model.tri_indices.numpy(), *axis),
                       finite=bool(np.isfinite(q).all()))


PAD_HALF = (0.008, 0.0032, 0.010)     # pad half sizes: x (width), y (thickness, the jaw axis), z (height) [m]
GRIP_KP, PAD_MASS = 1.0e4, 0.56       # C7's grip servo gain [N/m] and finger proxy mass [kg]
OPEN_GAP = 0.030                      # inner-face gap with both pads at the open stop [m]
PAD_Z = 0.10                          # pad centres at the start [m]


LIFT_KE = 1.0e5                       # carriage z servo gain [N/m]: its error x LIFT_KE is the vertical load


def gripper_model(cells: int = 6, cloth: Cloth = TOWEL, fix_bottom: bool = False, *, closed: bool = False,
                  film_top: float | None = None, fix_top: bool = True, pad_mass: float = PAD_MASS) -> newton.Model:
    """A carriage on a z slider (servo) carrying two pads on y sliders (q > 0 closes pad a, q < 0 closes pad b) open
    around a hanging double cloth layer (y = -r, +r; top edges fixed 15 mm above the pads, ``fix_bottom``: the bottom
    ones 16 mm below). ``closed``: faces start on the layers (y = +-2r); ``film_top``: top edge [m]; ``fix_top``."""
    from alloy._vendor.newton.solvers import SolverMuJoCo

    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.ke, b.default_shape_cfg.kd, b.default_shape_cfg.mu = SHAPE_KE, 1.0e3, 0.4
    lift = b.add_link(xform=wp.transform((0.0, 0.0, PAD_Z + 0.06), wp.quat_identity()), label="carriage")
    cfg = b.default_shape_cfg.copy()
    cfg.density = 1.0 / 8.0e-6                                               # 1 kg in a 2 cm cube, far above
    b.add_shape_box(lift, hx=0.01, hy=0.01, hz=0.01, cfg=cfg)
    joints = [b.add_joint_prismatic(-1, lift, axis=(0.0, 0.0, 1.0), label="lift", target_ke=LIFT_KE,
                                    parent_xform=wp.transform((0.0, 0.0, PAD_Z + 0.06), wp.quat_identity()),
                                    target_kd=2.0 * math.sqrt(LIFT_KE), actuator_mode=newton.JointTargetMode.POSITION)]
    pcfg = b.default_shape_cfg.copy()
    pcfg.density = pad_mass / (8 * PAD_HALF[0] * PAD_HALF[1] * PAD_HALF[2])
    for sign, label in ((-1.0, "pad_a"), (1.0, "pad_b")):
        y = sign * ((2.0 * cloth.radius if closed else 0.5 * OPEN_GAP) + PAD_HALF[1])
        pad = b.add_link(xform=wp.transform((0.0, y, PAD_Z), wp.quat_identity()), label=label)
        b.add_shape_mesh(pad, mesh=mesh_box(PAD_HALF), cfg=pcfg)
        joints.append(b.add_joint_prismatic(lift, pad, axis=(0.0, 1.0, 0.0), label=label + "_slide",
                                            parent_xform=wp.transform((0.0, y, -0.06), wp.quat_identity()),
                                            target_ke=GRIP_KP, target_kd=2.0 * math.sqrt(GRIP_KP * pad_mass),
                                            actuator_mode=newton.JointTargetMode.POSITION))
    b.add_articulation(joints)
    rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)     # grid y -> z: a cloth in the xz plane
    top = PAD_Z + PAD_HALF[2] + 0.015 if film_top is None else film_top
    c = cloth
    for y in (-c.radius, c.radius):
        b.add_cloth_grid(pos=wp.vec3(-0.5 * cells * c.cell, y, top - cells * c.cell), rot=rot, vel=wp.vec3(0.0),
                         dim_x=cells, dim_y=cells, cell_x=c.cell, cell_y=c.cell, mass=c.mass, fix_top=fix_top,
                         fix_bottom=fix_bottom, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=1.0e-2, edge_ke=1.0e-3,
                         edge_kd=1.0e-5, particle_radius=c.radius)
    return _finalize(b, c.soft_ke)


@dataclass(frozen=True)
class PinchPhase:
    name: str
    end: float                # phase end [s]
    lift: float               # carriage target [m]
    close: bool               # pads driven to closed (grip_n cap) or to the open stop (open_n cap)
    ramp: float = 0.0         # [s] the carriage target moves from the previous phase's at constant speed over this


PINCH_SCRIPT = (PinchPhase("close", 0.35, 0.0, True), PinchPhase("hold", 0.50, 0.0, True),
                PinchPhase("open", 0.75, 0.0, False), PinchPhase("rise", 1.05, 0.030, False))


def _pad_targets(q: np.ndarray, close: bool, grip_n: float, open_n: float,
                 half_gap: float = 0.5 * OPEN_GAP) -> np.ndarray:
    """Kit grip servo: goal closed (q = +-``half_gap``, the start's face half-gap) or open (0), target within F/kp."""
    goal = np.array([half_gap, -half_gap]) if close else np.zeros(2)
    lead = (grip_n if close else open_n) / GRIP_KP
    return np.clip(goal, q - lead, q + lead)


@dataclass(frozen=True)
class PinchResult:
    gap_hold: float           # inner-face gap at the end of the hold [m]
    open_left: float          # largest |slide| from the open stop at the end of the open [m]
    cloth_rise: float         # largest particle rise over the 30 mm rise of the open pads [m]
    pass_through: int         # worlds flagged by the pass-through detector
    work: float               # interface work at the end [J]
    finite: bool


def _run_script(sched: Scheduler, model: newton.Model, script: Sequence[PinchPhase], observe: Callable[..., None],
                grip_n: float = 15.0, open_n: float = 40.0, period: float = 1.0 / 60.0,
                on_exchange: Callable[[PinchPhase, float], None] | None = None, *, half_gap: float = 0.5 * OPEN_GAP,
                monitor: Any = None) -> newton.State:
    """Drive the gripper rig through ``script``: servo targets once per control ``period`` (closed goal: the start's
    face half-gap ``half_gap``), the Scheduler's own clock per period; ``observe(phase, k, state)`` after every period,
    ``on_exchange(phase, t)`` and ``monitor`` (a HealthMonitor) after every exchange. Returns the last state."""
    s0, s1, ctl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    glob = newton.CollisionPipeline(model)
    contacts = glob.contacts()
    k, last, prev = 0, None, 0.0
    for ph in script:
        start = k * period
        while k < round(ph.end / period):
            q = s0.joint_q.numpy()
            frac = min(1.0, ((k + 1) * period - start) / ph.ramp) if ph.ramp > 0.0 else 1.0
            lift = prev + (ph.lift - prev) * frac
            ctl.joint_target_q.assign(np.array([lift, *_pad_targets(q[1:3].astype(np.float64), ph.close,
                                                                    grip_n, open_n, half_gap)], np.float32))
            dt, _ = sched.choose_clock(period, last)
            for j in range(int(round(period / dt))):
                s0.clear_forces()
                glob.collide(s0, contacts)
                sched.step(s0, s1, ctl, contacts, dt)
                s0, s1 = s1, s0
                if monitor is not None:
                    monitor.update(int(round((k * period + (j + 1) * dt) / dt)), s1, s0, contacts, dt=dt)
                if on_exchange is not None:
                    on_exchange(ph, k * period + (j + 1) * dt)
            last, k = sched.report(), k + 1
            observe(ph, k, s0)
        prev = ph.lift
    return s0


def _pass_through_count(sched: Scheduler) -> int:
    from alloy.coupling.pass_through import PASS_THROUGH_BITS

    flags = [x[0] for x in sched.capacity_flags() if isinstance(x, tuple) and x[1] == PASS_THROUGH_BITS]
    return int(sum(int(f.numpy().sum()) for f in flags))


def run_pinch(full_surface: bool = True, mode: str = "lagged", grip_n: float = 15.0, open_n: float = 40.0,
              period: float = 1.0 / 60.0, *, cloth: Cloth = TOWEL) -> PinchResult:
    """Close on the double layer of ``cloth`` at ``grip_n``, hold, open at ``open_n``, rise 30 mm (``PINCH_SCRIPT``).
    Servo targets once per control ``period``; the Scheduler's own clock picks the exchange step (as Simulation runs
    it: the watchdog's escalation halves it)."""
    model = gripper_model(cloth=cloth)
    sched = _cloth_scheduler(model, full_surface, mode=mode, radius=cloth.radius)
    seen: dict[str, np.ndarray] = {}

    def observe(ph: PinchPhase, k: int, s: newton.State) -> None:
        seen[ph.name] = np.concatenate([s.body_q.numpy()[1:3, 1], s.joint_q.numpy()[1:3]])
        seen[ph.name + "_z"] = s.particle_q.numpy()[:, 2].copy()

    s0 = _run_script(sched, model, PINCH_SCRIPT, observe, grip_n, open_n, period)
    return PinchResult(gap_hold=float(seen["hold"][1] - seen["hold"][0]) - 2.0 * PAD_HALF[1],
                       open_left=float(np.abs(seen["open"][2:]).max()),
                       cloth_rise=float((seen["rise_z"] - seen["open_z"]).max()),
                       pass_through=_pass_through_count(sched),
                       work=float(np.sum(sched.report().interface_work)),
                       finite=bool(np.isfinite(s0.particle_q.numpy()).all()))


@dataclass(frozen=True)
class DragResult:
    gap_spread: float         # max - min pad gap over the late hold and the move [m] (a bouncing clamp: mm)
    follow: float             # how far the clamped cloth moved with the pads (they moved ``travel``) [m]
    peak: float               # largest |z force| of the cloth on the pads during the ramp, per exchange [N]
    sliding: float            # mean |z force| over the ramp's last third, per exchange: mu N while it slides [N]
    pass_through: int
    finite: bool


def run_drag(travel: float, ramp: float = 0.15, full_surface: bool = True, grip_n: float = 15.0, *,
             cloth: Cloth = TOWEL, iterations: int = 10, friction_epsilon: float = 1.0e-3,
             policy: SchedulerPolicy | None = None) -> DragResult:
    """SCR-SC4-6: close on the double layer (top and bottom edges fixed) at ``grip_n``, hold, then move the carriage
    by ``travel`` at constant speed over ``ramp`` with the pads closed: down for ``travel`` > 0, a lift for
    ``travel`` < 0. The clamped cloth (the particles between the pads at the hold) follows while the cloth's pull is
    below mu N (two faces x mu x ``grip_n``) and slides at it. VBD's friction is smoothed Coulomb: below
    friction_epsilon of slip speed it is mu f.n f1(v / eps), so a held cloth creeps; a ramp much faster than eps
    measures mu N itself. The force is the cloth's on the pads (coupling endpoints' f_used), free of the carriage's
    inertia and of the rig's weight."""
    period, t0 = 1.0 / 60.0, 0.50
    script = (PinchPhase("close", 0.35, 0.0, True), PinchPhase("hold", t0, 0.0, True),
              PinchPhase("move", t0 + ramp + 0.15, -travel, True, ramp=ramp))
    model = gripper_model(cloth=cloth, fix_bottom=True)
    sched = _cloth_scheduler(model, full_surface, iterations, "lagged", cloth.radius, friction_epsilon, policy)
    acc: dict[str, Any] = {"sel": None, "z0": 0.0, "z": 0.0, "gaps": [], "fz": [], "late": []}

    def observe(ph: PinchPhase, k: int, s: newton.State) -> None:
        bq, pq = s.body_q.numpy(), s.particle_q.numpy()
        t = k * period
        if ph.name == "hold" and t >= t0 - 0.05:
            acc["gaps"].append(float(bq[2, 1] - bq[1, 1]) - 2.0 * PAD_HALF[1])
            if acc["sel"] is None:
                acc["sel"] = ((np.abs(pq[:, 0]) < PAD_HALF[0]) & (np.abs(pq[:, 2] - bq[1, 2]) < PAD_HALF[2])
                              & (np.abs(pq[:, 1]) < 0.004))
            acc["z0"] = float(pq[acc["sel"], 2].mean())
        if ph.name == "move" and t <= t0 + ramp + 1e-9:
            acc["gaps"].append(float(bq[2, 1] - bq[1, 1]) - 2.0 * PAD_HALF[1])
            acc["z"] = float(pq[acc["sel"], 2].mean())

    def on_exchange(ph: PinchPhase, t: float) -> None:
        if ph.name != "move" or t > t0 + ramp + 1e-9:
            return
        fz = abs(_pad_fz(sched))
        acc["fz"].append(fz)
        if t > t0 + ramp * 2.0 / 3.0:
            acc["late"].append(fz)

    s0 = _run_script(sched, model, script, observe, grip_n, period=period, on_exchange=on_exchange)
    return DragResult(gap_spread=float(np.ptp(acc["gaps"])), follow=(acc["z0"] - acc["z"]) * math.copysign(1.0, travel),
                      peak=float(max(acc["fz"])), sliding=float(np.mean(acc["late"])),
                      pass_through=_pass_through_count(sched), finite=bool(np.isfinite(s0.particle_q.numpy()).all()))


def _pad_fz(sched: Scheduler) -> float:
    """z force the cloth puts on the two pads (bodies 1, 2), from the coupling endpoints' f_used [N]."""
    tot = 0.0
    eps = sched.endpoint_sets
    for es in (eps() if callable(eps) else eps):
        if es.kind != "body" or es.count == 0:
            continue
        g, f = es.src_global.numpy(), es.f_used.numpy()
        tot += float(sum(f[i, 2] for i in range(len(g)) if int(g[i]) in (1, 2)))
    return tot
