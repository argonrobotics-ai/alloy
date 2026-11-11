# SPDX-License-Identifier: Apache-2.0
"""Sewing ply pinch rig (XB4-7 sewing proof gap, coordinator 2026-11-06 18:40Z): ``run_ply_pinch``.

In XB4-7's sewing rebuild ([internal note] xb4/gym-anything/proofs/sewing), FR3 finger pads land on a stack of two muslin
plies lying on the table, then close to gather it; the world latched INVALID (PEN_FAIL: 0.61 mm against the 0.305 mm
limit, half the ply thickness) at 10 N and at 3 N. This rig reproduces that at test size, on the lagged MuJoCo -> VBD
Scheduler of WP4's pinch rigs (``alloy.coupling.pinch_rigs``):
- a light carriage on a z servo carries two box pads on y sliders (WP4's 16 x 6.4 x 20 mm pads), open above the stack;
- ``PlyStack`` muslin plies (their params.py: 0.61 mm free thickness, 125 g/m2, ply-ply mu 0.65) lie flat on a static
  table, one particle radius = half the thickness;
- phases: lower onto the stack and press at ``press_n`` (the servo's lead below the contact height), close at
  ``grip_n``, lift ``lift`` and hold.
A HealthMonitor (level full) checks every exchange; per period the rig also measures, under each pad, how far the
plies' particles reach into the pad's bottom face and into the table, and the smallest ply-to-ply spacing.

The pads must press the plies (GAP-10A's finding, 2026-11-10). Cloth contact is per vertex, and a 6.4 mm pad on the
10 mm grid has at most one vertex row under it. At the first rig's 20 mm opening that row sat at the pads' inner
edges: during a 100 N press the pads slid 0.3 mm outward, left it, and sank between the vertices to the table (pad
bottoms 4 um under the table top, MuJoCo Warp out of constraint rows 3,171 times at njmax 50); the lift then gathered
cloth that lay above the faces' bottoms, which read as a 96 % carry in a world with no health flag. So:
- the default opening (``SEATED_GAP``) centres each pad on the vertex row at y = +-10 mm;
- a run also reports ``plies_under`` (the fewest plies with a vertex under a pad during press and close),
  ``pad_bottom`` (the lowest pad bottom over the table top) and ``row_overflow`` (exchange steps where MuJoCo Warp
  ran out of rows or contacts; ``NJMAX`` is sized so a pad on the table does not);
- ``PinchResult.pressed`` is true when every ply had a vertex under each pad throughout, no pad reached the table,
  and nothing overflowed. A run that is not ``pressed`` did not pinch the stack, whatever its carry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.coupling.pinch_rigs import GRIP_KP, LIFT_KE, PAD_HALF, SHAPE_KE, _pad_targets, _pass_through_count, mesh_box

CLEARANCE = 0.002                      # pad bottoms above the stack top at the start [m]
SEATED_GAP = 2.0 * (0.010 - PAD_HALF[1])   # open pads' inner-face gap that centres them on the y = +-10 mm rows [m]
NJMAX = 400                            # MuJoCo Warp's rows per world (50 overflowed with a pad on the table)
PRESSING = ("press", "close")          # the phases in which the pads bear on the stack
TABLE_HALF = (0.15, 0.15, 0.01)        # static table box half sizes; its top is z = 0 [m]


@dataclass(frozen=True)
class PlyStack:
    """Muslin plies lying flat, centred under the pads (XB4-7's card, from their params.py:213-235)."""

    count: int = 2
    thickness: float = 0.61e-3         # free thickness [m]; particle radius = half of it
    areal_density: float = 0.1252      # [kg/m2]
    cell: float = 0.010                # grid spacing [m] (the proof's 10 mm)
    size: tuple[float, float] = (0.08, 0.08)
    soft_ke: float = 1.0e4             # VBD soft contact stiffness [N/m] (the proof's card value)
    mu: float = 0.65                   # VBD's one particle mu (their ply-ply value)
    tri_ke: float = 400.0              # membrane mu / lambda [N/m] (proof's KES mapping, 09:25Z)
    tri_ka: float = 3000.0
    edge_ke: float = 1.0e-5            # bending
    self_radius: float | None = None   # VBD self-contact radius [m]; None: the particle radius (the proof ran C7's 1 mm)

    @property
    def radius(self) -> float:
        return 0.5 * self.thickness

    @property
    def top(self) -> float:
        """Stack top at rest [m] (table top z = 0)."""
        return self.count * self.thickness


@dataclass(frozen=True)
class PinchSpec:
    """The pinch: forces [N], pad layout, contact path and phase times [s]."""

    press_n: float = 10.0              # vertical load on the stack while pressing
    grip_n: float = 10.0               # closing force per pad
    open_gap: float = SEATED_GAP       # inner-face gap of the open pads [m]; 0.020 (the first rig) lands on edges
    pad_mass: float = 0.2              # each pad [kg]: 20 g FR3 fingertips put the Scheduler's clock at ~1e-4 s (CPU)
    carriage_mass: float = 0.2
    shape_ke: float = SHAPE_KE         # pad / table shape contact stiffness [N/m]
    pad_mu: float = 0.8                # their pad friction
    table_mu: float = 0.3
    full_surface: bool = False         # WP4's full-surface proxy contacts for the pads
    log_barrier: bool | None = None    # VBD body-particle normal contact on the self-contact log barrier; None: make_vbd's
    iterations: int = 20               # VBD iterations (the proof's VbdSettings)
    lower: float = 0.3
    press: float = 0.3
    close: float = 0.4
    lift: float = 0.010                # [m]
    lift_time: float = 0.3
    hold: float = 0.3


@dataclass
class PinchResult:
    pen_fail: bool                     # PEN_FAIL / PAIR_PEN_FAIL from the HealthMonitor
    max_pen: float                     # deepest penetration health measured [m]
    flags: int
    phase: dict[str, dict[str, float]] = field(default_factory=dict)   # per phase: worst pad / table depth, ply gap
    carry: float = float("nan")        # mean rise of the particles pinched between the faces, close end -> hold end [m]
    pad_rise: float = float("nan")     # the pads' rise over the same span [m]
    pass_through: int = 0
    finite: bool = True
    failure: str = ""                  # the HealthMonitor's first-failure text (world 0)
    plies: int = 0                     # the stack's ply count
    plies_under: int = 0               # fewest plies with a vertex under a pad over press and close (``plies``: seated)
    pad_bottom: float = float("nan")   # lowest pad bottom above the table top over press and close [m]
    row_overflow: int = 0              # exchange steps where MuJoCo Warp ran out of constraint rows or contacts

    @property
    def pressed(self) -> bool:
        """The pads bore on every ply through press and close, stayed off the table, and no row was dropped."""
        return self.plies_under == self.plies > 0 and self.pad_bottom > 0.0 and self.row_overflow == 0


def _cloth(b: newton.ModelBuilder, plies: PlyStack) -> None:
    nx, ny = (int(round(s / plies.cell)) for s in plies.size)
    mass = plies.areal_density * plies.cell ** 2
    for k in range(plies.count):
        z = plies.radius + k * plies.thickness + 1.0e-6
        b.add_cloth_grid(pos=wp.vec3(-0.5 * plies.size[0], -0.5 * plies.size[1], z), rot=wp.quat_identity(),
                         vel=wp.vec3(0.0), dim_x=nx, dim_y=ny, cell_x=plies.cell, cell_y=plies.cell, mass=mass,
                         tri_ke=plies.tri_ke, tri_ka=plies.tri_ka, tri_kd=1.0e-2, edge_ke=plies.edge_ke,
                         edge_kd=1.0e-6, particle_radius=plies.radius)


def ply_pinch_model(plies: PlyStack = PlyStack(), spec: PinchSpec = PinchSpec()) -> newton.Model:
    """Table, carriage (body 0, z servo: joint 0), pads (bodies 1, 2; y sliders: joints 1, 2, q > 0 closes), plies."""
    from alloy._vendor.newton.solvers import SolverMuJoCo

    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    b.default_shape_cfg.ke, b.default_shape_cfg.kd = spec.shape_ke, 1.0e3
    table = b.default_shape_cfg.copy()
    table.mu = spec.table_mu
    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -TABLE_HALF[2]), wp.quat_identity()), hx=TABLE_HALF[0],
                    hy=TABLE_HALF[1], hz=TABLE_HALF[2], cfg=table)
    zc = plies.top + CLEARANCE + PAD_HALF[2]                                  # pad centres at the start
    above = 0.03
    lift = b.add_link(xform=wp.transform((0.0, 0.0, zc + above), wp.quat_identity()), label="carriage")
    cfg = b.default_shape_cfg.copy()
    cfg.density = spec.carriage_mass / 1.25e-7
    cfg.has_shape_collision = cfg.has_particle_collision = False
    b.add_shape_box(lift, hx=0.0025, hy=0.0025, hz=0.0025, cfg=cfg)
    m_all = spec.carriage_mass + 2.0 * spec.pad_mass
    joints = [b.add_joint_prismatic(-1, lift, axis=(0.0, 0.0, 1.0), label="lift", target_ke=LIFT_KE,
                                    parent_xform=wp.transform((0.0, 0.0, zc + above), wp.quat_identity()),
                                    target_kd=2.0 * math.sqrt(LIFT_KE * m_all),
                                    actuator_mode=newton.JointTargetMode.POSITION)]
    pcfg = b.default_shape_cfg.copy()
    pcfg.mu = spec.pad_mu
    pcfg.density = spec.pad_mass / (8 * PAD_HALF[0] * PAD_HALF[1] * PAD_HALF[2])
    for sign, label in ((-1.0, "pad_a"), (1.0, "pad_b")):
        y = sign * (0.5 * spec.open_gap + PAD_HALF[1])
        pad = b.add_link(xform=wp.transform((0.0, y, zc), wp.quat_identity()), label=label)
        b.add_shape_mesh(pad, mesh=mesh_box(PAD_HALF), cfg=pcfg)
        joints.append(b.add_joint_prismatic(lift, pad, axis=(0.0, 1.0, 0.0), label=label + "_slide",
                                            parent_xform=wp.transform((0.0, y, -above), wp.quat_identity()),
                                            target_ke=GRIP_KP, target_kd=2.0 * math.sqrt(GRIP_KP * spec.pad_mass),
                                            actuator_mode=newton.JointTargetMode.POSITION))
    b.add_articulation(joints)
    _cloth(b, plies)
    top = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(top)
    top.add_world(b)
    top.color()
    m = top.finalize(device="cpu")
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = plies.soft_ke, 1.0e-2, plies.mu
    return m


PHASES = ("lower", "press", "close", "lift", "hold")
CLOSED = ("close", "lift", "hold")


def _lift_target(name: str, frac: float, q: float, lead: float, z_hold: float, spec: PinchSpec) -> float:
    """The carriage target: lower (a ramp to 2 mm below the stack's top, never more than ``lead`` below q), press /
    close in force mode (``lead`` below q, re-centred each period: the load is ``press_n``), lift / hold above the
    height the close ended at."""
    if name == "lower":
        return max(-(CLEARANCE + 0.002) * frac, q - lead)
    if name in ("press", "close"):
        return q - lead
    return z_hold + spec.lift * (frac if name == "lift" else 1.0)


def _drive(sched: Any, m: newton.Model, spec: PinchSpec, observe: Any, monitor: Any,
           period: float = 1.0 / 60.0) -> newton.State:
    """Servo targets once per control ``period`` (the Scheduler's own clock inside it), ``observe(phase, k, state)``
    after every period, ``monitor`` after every exchange. Returns the last state."""
    s0, s1, ctl = m.state(), m.state(), m.control()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
    glob = newton.CollisionPipeline(m)
    contacts = glob.contacts()
    ends = np.cumsum([spec.lower, spec.press, spec.close, spec.lift_time, spec.hold])
    lead = (spec.press_n - 9.81 * (spec.carriage_mass + 2.0 * spec.pad_mass)) / LIFT_KE
    k, last, z_hold, start = 0, None, 0.0, 0.0
    for name, end in zip(PHASES, ends):
        while k < round(end / period):
            q = s0.joint_q.numpy().astype(np.float64)
            frac = min(1.0, ((k + 1) * period - start) / (end - start))
            if name == "close":
                z_hold = q[0]
            z = _lift_target(name, frac, q[0], lead, z_hold, spec)
            pads = _pad_targets(q[1:3], name in CLOSED, spec.grip_n, 40.0, 0.5 * spec.open_gap)
            ctl.joint_target_q.assign(np.array([z, *pads], np.float32))
            dt, _ = sched.choose_clock(period, last)
            for j in range(int(round(period / dt))):
                s0.clear_forces()
                glob.collide(s0, contacts)
                sched.step(s0, s1, ctl, contacts, dt)
                s0, s1 = s1, s0
                monitor.update(int(round((k * period + (j + 1) * dt) / dt)), s1, s0, contacts, dt=dt)
            last, k = sched.report(), k + 1
            observe(name, k, s0)
        start = end
    return s0


def _measure(s: newton.State, plies: PlyStack, per_ply: int) -> dict[str, float]:
    """Under each pad's bottom face: the deepest particle reach into the pad (d > r: its centre is inside) and into
    the table, and the smallest ply-to-ply spacing of matching vertices (2 r at rest, < 0: crossed). Also ``under``,
    the fewer of the two pads' counts of plies with a vertex under the pad, and ``bottom``, the lower pad bottom's
    height over the table top."""
    bq, pq = s.body_q.numpy(), s.particle_q.numpy()
    r = plies.radius
    out = {"pad": -np.inf, "table": float(np.max(r - pq[:, 2])), "ply_gap": np.inf, "under": plies.count,
           "bottom": float(np.min(bq[1:3, 2]) - PAD_HALF[2])}
    for body in (1, 2):
        px, py, pz = bq[body, :3]
        under = (np.abs(pq[:, 0] - px) < PAD_HALF[0]) & (np.abs(pq[:, 1] - py) < PAD_HALF[1])
        out["under"] = min(out["under"], int(under.reshape(plies.count, per_ply).any(axis=1).sum()))
        if under.any():
            out["pad"] = max(out["pad"], float(np.max(pq[under, 2] + r - (pz - PAD_HALF[2]))))
    if plies.count > 1:
        z = pq[:, 2].reshape(plies.count, per_ply)
        out["ply_gap"] = float(np.min(np.diff(z, axis=0)))
    return out


def vbd_kwargs(iterations: int, self_radius: float, log_barrier: bool | None = None) -> dict[str, Any]:
    """The cloth entry's SolverVBD kwargs (WP4's pinch rigs' settings), built by the route's ``make_vbd``."""
    kw = {"iterations": iterations, "friction_epsilon": 1.0e-3, "particle_enable_self_contact": True,
          "particle_self_contact_radius": self_radius, "particle_self_contact_margin": 2.0 * self_radius,
          "rigid_body_particle_contact_buffer_size": 2048}
    if log_barrier is not None:
        kw["rigid_soft_contact_use_log_barrier"] = bool(log_barrier)
    return kw


def cloth_scheduler(m: newton.Model, iterations: int, self_radius: float, *, full_surface: bool = False,
                    log_barrier: bool | None = None, policy: Any = None) -> Any:
    """WP4's lagged MuJoCo -> VBD Scheduler (``pinch_rigs._cloth_scheduler``) with the cloth entry from ``make_vbd``;
    every body on the MuJoCo entry, every particle on the VBD entry. ``policy``: a SchedulerPolicy (default: its
    defaults)."""
    from alloy._vendor.newton.solvers import SolverMuJoCo

    from alloy.core.newton_compat import SolverCoupledProxy
    from alloy.coupling.full_surface import contact_view
    from alloy.coupling.proxy_contacts import ProxyContactPipeline
    from alloy.coupling.scheduler import Scheduler
    from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy
    from alloy.solvers.vbd import make_vbd

    moving = np.flatnonzero(m.shape_body.numpy() >= 0)
    kw = vbd_kwargs(iterations, self_radius, log_barrier)
    e_m = SolverCoupledProxy.Entry(name="mjc", solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False,
                                                                           njmax=NJMAX),
                                   bodies=list(range(m.body_count)), joints=list(range(m.joint_count)))
    e_v = SolverCoupledProxy.Entry(name="vbd", particles=list(range(m.particle_count)), solver=lambda v: make_vbd(v, kw),
                                   configure_view=contact_view(moving) if full_surface else None)
    pipe = ProxyContactPipeline(full_surface=full_surface, full_surface_shapes=moving, soft_contact_gap=0.002)
    proxy = SolverCoupledProxy.Proxy(source="mjc", destination="vbd", bodies=list(range(m.body_count)), mode="lagged",
                                     proxy_relaxation_mode="fixed", collision_pipeline=pipe, collide_interval=1)
    return Scheduler(m, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False)],
                     [PairPolicy("mjc", "vbd", mode="lagged", proxy=proxy)], policy or SchedulerPolicy())


def run_ply_pinch(plies: PlyStack = PlyStack(), spec: PinchSpec = PinchSpec(), verbose: bool = False) -> PinchResult:
    """Lower, press, close, lift and hold (module docstring) through the lagged MuJoCo -> VBD Scheduler."""
    from alloy.precision.health import HealthMonitor
    from alloy.precision.types import CheckPolicy, HealthFlags

    m = ply_pinch_model(plies, spec)
    sched = cloth_scheduler(m, spec.iterations, plies.self_radius or plies.radius, full_surface=spec.full_surface,
                            log_barrier=spec.log_barrier)
    from alloy.integration.stepper import entry_solvers

    # solvers=: health reads MuJoCo Warp's own overflow word only from the solvers it is given (WP1, 2026-11-10)
    monitor = _Watch(HealthMonitor(m, CheckPolicy(level="full", every=1, full_every=1),
                                   capacity_flags=sched.capacity_flags(),
                                   solvers=list(entry_solvers(sched).values())), sched)
    per_ply = m.particle_count // plies.count
    res = PinchResult(pen_fail=False, max_pen=0.0, flags=0, plies=plies.count, plies_under=plies.count,
                      pad_bottom=float("inf"))
    marks: dict[str, Any] = {}

    def observe(name: str, k: int, s: newton.State) -> None:
        row = res.phase.setdefault(name, {"pad": -np.inf, "table": -np.inf, "ply_gap": np.inf})
        got = _measure(s, plies, per_ply)
        if verbose and k % 6 == 0:
            rep = sched.report()
            print(f"t {k / 60:.3f} {name} dt_x {rep.dt_x:.2e} pad {got['pad'] * 1e3:.3f} table {got['table'] * 1e3:.3f}"
                  f" ply_gap {got['ply_gap'] * 1e3:.3f} mm", flush=True)
        for key in ("pad", "table"):
            row[key] = max(row[key], got[key])
        row["ply_gap"] = min(row["ply_gap"], got["ply_gap"])
        if name in PRESSING:
            res.plies_under, res.pad_bottom = min(res.plies_under, got["under"]), min(res.pad_bottom, got["bottom"])
        bq, pq = s.body_q.numpy(), s.particle_q.numpy()
        if name == "close":                                  # the pinched particles: between the faces at the close
            half = 0.5 * abs(bq[2, 1] - bq[1, 1]) - PAD_HALF[1] + plies.radius
            ids = np.flatnonzero((np.abs(pq[:, 1] - 0.5 * (bq[1, 1] + bq[2, 1])) < half)
                                 & (np.abs(pq[:, 0] - bq[1, 0]) < PAD_HALF[0]))
            marks["close"] = (ids, float(pq[ids, 2].mean()) if ids.size else np.nan, float(bq[1, 2]))
        elif name == "hold" and "close" in marks:
            ids, z0, p0 = marks["close"]
            marks["hold"] = (float(pq[ids, 2].mean()) - z0 if ids.size else np.nan, float(bq[1, 2]) - p0)

    s = _drive(sched, m, spec, observe, monitor)
    rep = monitor.poll()
    res.flags = int(rep.flags[0])
    res.pen_fail = bool(res.flags & int(HealthFlags.PEN_FAIL | HealthFlags.PAIR_PEN_FAIL))
    res.max_pen = float(np.nan_to_num(rep.max_pen_m[0]))
    res.failure = rep.failure_text(0) if res.flags else ""
    res.carry, res.pad_rise = marks.get("hold", (np.nan, np.nan))
    res.pass_through = _pass_through_count(sched)
    res.finite = bool(np.isfinite(s.particle_q.numpy()).all())
    res.row_overflow = monitor.overflow
    return res


class _Watch:
    """The run's HealthMonitor, plus a count of the exchange steps in which MuJoCo Warp had more constraint rows or
    contacts than its buffers hold (it drops the rest and only prints a line: no health flag)."""

    def __init__(self, monitor: Any, sched: Any) -> None:
        self.monitor, self.overflow = monitor, 0
        self.data = getattr(sched._entries["mjc"].solver, "mjw_data", None)

    def update(self, *args: Any, **kwargs: Any) -> None:
        self.monitor.update(*args, **kwargs)
        d = self.data
        if d is not None:
            rows, contacts = int(d.nefc.numpy().max()), int(d.nacon.numpy()[0])
            self.overflow += int(rows > int(d.njmax) or contacts > int(d.naconmax))

    def poll(self) -> Any:
        return self.monitor.poll()


def _main(argv: list[str] | None = None) -> None:
    import argparse
    import dataclasses
    import json

    ap = argparse.ArgumentParser(description="sewing ply pinch rig (one variant per call)")
    ap.add_argument("--press", type=float, default=10.0)
    ap.add_argument("--grip", type=float, default=10.0)
    ap.add_argument("--cell", type=float, default=0.010)
    ap.add_argument("--plies", type=int, default=2)
    ap.add_argument("--soft-ke", type=float, default=1.0e4)
    ap.add_argument("--shape-ke", type=float, default=SHAPE_KE)
    ap.add_argument("--full-surface", action="store_true")
    ap.add_argument("--iterations", type=int, default=20)
    ap.add_argument("--pad-mass", type=float, default=0.2)
    ap.add_argument("--self-radius", type=float, default=None)
    ap.add_argument("--open-gap", type=float, default=SEATED_GAP)
    ap.add_argument("--log-barrier", choices=("default", "on", "off"), default="default")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    plies = PlyStack(count=a.plies, cell=a.cell, soft_ke=a.soft_ke, self_radius=a.self_radius)
    spec = PinchSpec(press_n=a.press, grip_n=a.grip, shape_ke=a.shape_ke, full_surface=a.full_surface,
                     iterations=a.iterations, pad_mass=a.pad_mass, carriage_mass=a.pad_mass,
                     open_gap=a.open_gap, log_barrier={"default": None, "on": True, "off": False}[a.log_barrier])
    r = run_ply_pinch(plies, spec, verbose=not a.quiet)
    print("PLY_PINCH", json.dumps({"plies": dataclasses.asdict(plies), "spec": dataclasses.asdict(spec),
                                   "result": {**dataclasses.asdict(r), "pressed": r.pressed}}, default=float),
          flush=True)


if __name__ == "__main__":
    _main()
