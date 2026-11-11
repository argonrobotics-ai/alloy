# SPDX-License-Identifier: Apache-2.0
"""WP4 test helpers: the coupling-time box-on-cloth law and a Simulation-like period runner (SPEC §2.3, D18).

``coupling_box_on_cloth`` rebuilds the scene of coupling-time §3.2 (a 10 cm box owned by SolverMuJoCo dropped on a
0.5 m cloth strip owned by SolverVBD, pinned left and right, 1 kg, VBD-side contact k = 1e5 N/m), replicated over
``n`` worlds, optionally with a per-world drop-height offset. ``PeriodRunner`` runs control periods the way
Simulation does (choose_clock once per period, exchanges of dt_x, report, rollback through ``WorldSnapshot`` when the
stepper asks for it) and records per-exchange interface data for the CPL metrics.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverMuJoCo, SolverVBD

from alloy.core.newton_compat import SolverCoupledProxy
from alloy.core.snapshot import WorldSnapshot
from alloy.coupling.onset import escalated_beyond_onset

BOX_HALF = 0.05          # m
CLOTH_Z = 1.0            # m
DROP_GAP = 0.012         # m, box bottom above the cloth at t = 0


@dataclass(frozen=True)
class ClothSceneSpec:
    """Scene parameters (defaults = coupling-time §3.2)."""

    box_mass: float = 0.2            # kg
    cloth_mass: float = 1.0          # kg, total
    cells: int = 12
    size: float = 0.5                # m
    contact_ke: float = 1.0e5        # N/m, VBD body-particle contact
    contact_mu: float = 0.5
    shape_ke: float = 2.0e4          # N/m
    vbd_iterations: int = 10
    njmax: int = 50
    damped: bool = True              # False: no cloth (tri/edge) or contact damping (CPL-4 conservative scene)


def _world_builder(spec: ClothSceneSpec, dz: float = 0.0) -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    b.default_shape_cfg.ke = spec.shape_ke
    SolverMuJoCo.register_custom_attributes(b)
    z = CLOTH_Z + BOX_HALF + DROP_GAP + dz
    box = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, z), q=wp.quat_identity()), label="box")
    cfg = b.default_shape_cfg.copy()
    cfg.density = spec.box_mass / (2.0 * BOX_HALF) ** 3
    if not spec.damped:
        cfg.kd = 0.0
    b.add_shape_box(box, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg)
    n = spec.cells
    damp = 1.0 if spec.damped else 0.0
    b.add_cloth_grid(pos=wp.vec3(-0.5 * spec.size, -0.5 * spec.size, CLOTH_Z), rot=wp.quat_identity(),
                     vel=wp.vec3(0.0), fix_left=True, fix_right=True, dim_x=n, dim_y=n, cell_x=spec.size / n,
                     cell_y=spec.size / n, mass=spec.cloth_mass / float((n + 1) ** 2), tri_ke=1.0e4, tri_ka=1.0e4,
                     tri_kd=1.0e2 * damp, edge_ke=0.01, edge_kd=1.0e-4 * damp, particle_radius=0.01)
    return b


def coupling_box_on_cloth(n: int, spec: ClothSceneSpec = ClothSceneSpec(),
                          drop_offsets: Sequence[float] | None = None, device: str | None = None) -> newton.Model:
    """``n`` worlds of the box-on-cloth law; ``drop_offsets[w]`` raises world w's box [m]."""
    top = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(top)
    offsets = list(drop_offsets) if drop_offsets is not None else [0.0] * n
    if len(offsets) != n:
        raise ValueError(f"drop_offsets has {len(offsets)} values for {n} worlds")
    if all(o == offsets[0] for o in offsets):
        top.replicate(_world_builder(spec, offsets[0]), copy_count=n)
    else:
        for dz in offsets:
            top.add_world(_world_builder(spec, dz))
    top.color()
    model = top.finalize(device=device)
    model.soft_contact_ke = spec.contact_ke
    model.soft_contact_mu = spec.contact_mu
    if not spec.damped:
        model.soft_contact_kd = 0.0
    return model


def cloth_entries(model: newton.Model, spec: ClothSceneSpec = ClothSceneSpec(),
                  vbd_substeps: int = 1, mjc_substeps: int = 1) -> tuple[Any, Any]:
    """(MuJoCo entry owning every body/joint, VBD entry owning every particle)."""
    e_m = SolverCoupledProxy.Entry(
        name="mjc", solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=spec.njmax),
        bodies=list(range(model.body_count)), joints=list(range(model.joint_count)), substeps=mjc_substeps)
    e_v = SolverCoupledProxy.Entry(
        name="vbd", solver=lambda v: SolverVBD(model=v, iterations=spec.vbd_iterations),
        particles=list(range(model.particle_count)), substeps=vbd_substeps)
    return e_m, e_v


def cloth_proxy(model: newton.Model, mode: str = "lagged", relaxation_mode: str = "fixed") -> Any:
    return SolverCoupledProxy.Proxy(source="mjc", destination="vbd", bodies=list(range(model.body_count)),
                                    mode=mode, proxy_relaxation_mode=relaxation_mode,
                                    collision_pipeline=lambda m: newton.CollisionPipeline(m), collide_interval=1)


def stock_proxy_solver(model: newton.Model, spec: ClothSceneSpec = ClothSceneSpec(), iterations: int = 2,
                       mode: str = "lagged") -> SolverCoupledProxy:
    """Newton's stock SolverCoupledProxy on the same scene (the CPU equality reference)."""
    e_m, e_v = cloth_entries(model, spec)
    cfg = SolverCoupledProxy.Config(proxies=[cloth_proxy(model, mode)], iterations=iterations)
    return SolverCoupledProxy(model=model, entries=[e_m, e_v], coupling=cfg)


@dataclass
class ExchangeLog:
    """Per-exchange interface records (final pass), one row per exchange: arrays over endpoints or worlds."""

    f_used: list[np.ndarray] = field(default_factory=list)      # (P, 6)
    f_raw: list[np.ndarray] = field(default_factory=list)       # (P, 6)
    meff: np.ndarray | None = None                              # (P,)
    res_vel: list[np.ndarray] = field(default_factory=list)     # (W,)
    res_pos: list[np.ndarray] = field(default_factory=list)
    iters: list[np.ndarray] = field(default_factory=list)
    body_q: list[np.ndarray] = field(default_factory=list)      # (B, 7)
    body_qd: list[np.ndarray] = field(default_factory=list)
    dt: list[float] = field(default_factory=list)               # exchange length [s]
    ledger: list[dict[str, np.ndarray]] = field(default_factory=list)   # Scheduler.energy_ledger() per exchange

    def _rows(self) -> list[list]:
        return [getattr(self, f.name) for f in dataclasses.fields(self) if isinstance(getattr(self, f.name), list)]

    def move_since(self, start: int, into: "ExchangeLog") -> None:
        """Move the rows recorded after the first ``start`` exchanges (a rolled-back attempt) to ``into``."""
        into.meff = self.meff
        for mine, theirs in zip(self._rows(), into._rows()):
            theirs.extend(mine[start:])
            del mine[start:]


class PeriodRunner:
    """Eager §2.3 period loop for one stepper on one model (no contact layer; Newton collision pipeline)."""

    def __init__(self, model: newton.Model, stepper: Any, control_dt: float, *, log_exchanges: bool = False,
                 log_speed_only: bool = False) -> None:
        self.model, self.stepper, self.control_dt = model, stepper, control_dt
        self.state, self.next = model.state(), model.state()
        self.control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state)
        self.pipeline = newton.CollisionPipeline(model)
        self.contacts = self.pipeline.contacts()
        extra = dict(stepper.state_arrays()) if hasattr(stepper, "state_arrays") else {}
        maps = dict(stepper.state_world_maps()) if hasattr(stepper, "state_world_maps") else {}
        self.snapshot = WorldSnapshot(model, self.state, extra=extra, extra_world=maps)
        self.last = None
        self.periods = 0
        self.rollbacks = 0                      # re-runs for an escalation (the watchdog ladder)
        self.onset_reruns = 0                   # finer re-runs of a force jump (SCR-WP11-4): not instability
        self.seen_escalated = np.zeros(int(model.world_count), dtype=bool)   # ESCALATED (not a force jump) or FAILED
        self.log = ExchangeLog() if log_exchanges else None           # the accepted exchanges
        self.discarded = ExchangeLog() if log_exchanges else None     # the exchanges of rolled-back attempts
        self.speed_only = log_speed_only

    def exchange(self, dt: float) -> None:
        self.state.clear_forces()
        self.pipeline.collide(self.state, self.contacts)
        self.stepper.step(self.state, self.next, self.control, self.contacts, dt)
        self.state, self.next = self.next, self.state
        if self.log is not None:
            self.log.dt.append(dt)
            self._record()

    def _record(self) -> None:
        lg, st = self.log, self.stepper
        if self.speed_only:
            lg.body_qd.append(np.array(self.state.body_qd.numpy(), copy=True))
            return
        sets = getattr(st, "endpoint_sets", ())
        if sets:
            es = sets[0]
            lg.f_used.append(np.array(es.f_used.numpy(), copy=True))
            lg.f_raw.append(np.array(es.f_raw.numpy(), copy=True))
            lg.meff = np.array(es.meff.numpy(), copy=True)
            loop = st.iteration_loop
            lg.res_vel.append(np.array(loop.res_vel.numpy(), copy=True))
            lg.res_pos.append(np.array(loop.res_pos.numpy(), copy=True))
            lg.iters.append(np.array(loop.iters_used.numpy(), copy=True))
        if hasattr(st, "energy_ledger"):
            lg.ledger.append(st.energy_ledger())
        lg.body_q.append(np.array(self.state.body_q.numpy(), copy=True))
        lg.body_qd.append(np.array(self.state.body_qd.numpy(), copy=True))

    def run_period(self) -> Any:
        """One control period with up to the stepper's rollback budget of re-runs; returns the final report. The
        exchanges of a rolled-back attempt move from ``log`` to ``discarded``, so ``log`` is the accepted run."""
        start = len(self.log.dt) if self.log is not None else 0
        while True:
            dt_x, _ = self.stepper.choose_clock(self.control_dt, self.last)
            self.snapshot.save(self.state)
            for _ in range(int(round(self.control_dt / dt_x))):
                self.exchange(dt_x)
            report = self.stepper.report()
            self.last = report
            unstable = escalated_beyond_onset(report)
            self.seen_escalated |= unstable | (np.asarray(report.watch_state) == 4)   # 3 marks every rolled-back world
            if hasattr(self.stepper, "rollback_requested") and self.stepper.rollback_requested(report):
                self.snapshot.restore(self.state)
                self.stepper.on_rollback()
                if unstable.any():
                    self.rollbacks += 1
                else:
                    self.onset_reruns += 1
                if self.log is not None:
                    self.log.move_since(start, self.discarded)
                continue
            self.periods += 1
            return report

    def run(self, periods: int) -> list[Any]:
        return [self.run_period() for _ in range(periods)]


def cloth_scheduler(model: newton.Model, spec: ClothSceneSpec = ClothSceneSpec(), *, policy: Any = None,
                    mode: str = "lagged", vbd_adaptive: bool = True, vbd_initial: int = 1,
                    max_rung: int = 64) -> Any:
    """``alloy.coupling.scheduler.Scheduler`` on the cloth scene: MuJoCo fixed at 1 substep, VBD built at
    ``max_rung`` (D21) and started at ``vbd_initial``. Staggered pairs use Aitken."""
    from alloy.coupling.scheduler import Scheduler
    from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy

    e_m, e_v = cloth_entries(model, spec, vbd_substeps=max_rung if vbd_adaptive else vbd_initial)
    relax = "aitken" if mode == "staggered" else "fixed"
    pair = PairPolicy("mjc", "vbd", mode=mode, proxy=cloth_proxy(model, mode, relax))
    return Scheduler(model, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, initial_rung=vbd_initial,
                                                                         adaptive=vbd_adaptive)],
                     [pair], policy if policy is not None else SchedulerPolicy())


def fixed_clock_policy(dt_x: float, **kwargs: Any) -> Any:
    """SchedulerPolicy whose exchange clock stays at the grid value ``dt_x`` until an escalation."""
    from alloy.coupling.types import ClockPolicy, SchedulerPolicy

    return SchedulerPolicy(clock=ClockPolicy(dt_x_max=dt_x, omega_dt_max=float("inf")), **kwargs)


def record_metric(metric: str, **values: Any) -> None:
    """Append one metric row (JSON line) to ``$ALLOY_METRICS_OUT`` or ``bench/out/wp4/metrics.jsonl``."""
    import json
    import os
    import time
    from pathlib import Path

    path = Path(os.environ.get("ALLOY_METRICS_OUT", "bench/out/wp4/metrics.jsonl"))
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"metric": metric, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **values}
    with path.open("a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")


def cpl_force_mismatch(log: ExchangeLog, gravity: float = 9.81, active_frac: float = 1e-3) -> np.ndarray:
    """CPL-1 samples: |F_used − F_raw| / max(|F_raw|, m_eff·g) per (exchange, endpoint) with contact
    (|F_raw| > active_frac·m_eff·g); linear force parts."""
    used = np.array(log.f_used)[..., :3]
    raw = np.array(log.f_raw)[..., :3]
    mg = log.meff[None, :] * gravity
    raw_n = np.linalg.norm(raw, axis=-1)
    ratio = np.linalg.norm(used - raw, axis=-1) / np.maximum(raw_n, mg)
    return ratio[raw_n > active_frac * mg]


def fake_drop_model(heights: Sequence[float], device: str | None = None) -> newton.Model:
    """Deterministic (NATIVE) coupling scene for the GPU bitwise tests: per world a 0.2 kg "box" body at
    ``heights[w]`` (source entry, no floor) and a resting "pad" body (destination entry, floor at z = 0). The box's
    proxy falls onto the destination's floor; the momentum harvest returns the floor impulse."""
    top = newton.ModelBuilder()
    for h in heights:
        b = newton.ModelBuilder()
        cfg = b.default_shape_cfg.copy()
        cfg.density = 0.2 / (2.0 * BOX_HALF) ** 3
        box = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, float(h)), q=wp.quat_identity()), label="box")
        b.add_shape_box(box, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg)
        pad = b.add_body(xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.1), q=wp.quat_identity()), label="pad")
        b.add_shape_box(pad, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, cfg=cfg)
        top.add_world(b)
    return top.finalize(device=device)


def fake_scheduler(model: newton.Model, *, mode: str = "staggered", policy: Any = None) -> Any:
    """Scheduler over two FakeSolver entries (box -> pad entry, proxy of every box), FakeSolver floor radius 0.1 m."""
    from alloy.coupling.scheduler import Scheduler
    from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy
    from alloy.testing.core import FakeSolver

    labels = list(model.body_label)
    boxes = [i for i, name in enumerate(labels) if name.endswith("box")]
    pads = [i for i, name in enumerate(labels) if name.endswith("pad")]
    joint_body = model.joint_child.numpy()
    src = SolverCoupledProxy.Entry(
        name="src", solver=lambda v: FakeSolver(v, FakeSolver.Config(floor_z=None)), bodies=boxes,
        joints=[j for j, b in enumerate(joint_body) if b in set(boxes)], substeps=4)
    dst = SolverCoupledProxy.Entry(
        name="dst", solver=lambda v: FakeSolver(v, FakeSolver.Config(floor_z=0.0)), bodies=pads,
        joints=[j for j, b in enumerate(joint_body) if b in set(pads)], substeps=4)
    relax = "aitken" if mode == "staggered" else "fixed"
    proxy = SolverCoupledProxy.Proxy(source="src", destination="dst", bodies=boxes, mode=mode,
                                     proxy_relaxation_mode=relax)
    return Scheduler(model, [AlloyEntry(src, adaptive=False), AlloyEntry(dst, adaptive=False)],
                     [PairPolicy("src", "dst", mode=mode, proxy=proxy)],
                     policy if policy is not None else SchedulerPolicy())


def cloth_seed_model(masses: Sequence[float], speeds: Sequence[float], contact_ke: float,
                     spec: ClothSceneSpec = ClothSceneSpec(), device: str | None = None) -> newton.Model:
    """CPL-5 seeds: world w has a box of ``masses[w]`` kg moving down at ``speeds[w]`` m/s at t = 0 above the cloth
    (cloth mass ``spec.cloth_mass``); every world shares the model-wide contact stiffness ``contact_ke``."""
    if len(masses) != len(speeds):
        raise ValueError("masses and speeds need one value per world")
    top = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(top)
    for m in masses:
        top.add_world(_world_builder(dataclasses.replace(spec, box_mass=float(m))))
    top.color()
    model = top.finalize(device=device)
    model.soft_contact_ke = float(contact_ke)
    model.soft_contact_mu = spec.contact_mu
    qd = model.joint_qd.numpy()
    starts = model.joint_qd_start.numpy()
    for w, v in enumerate(speeds):                     # one free joint per world, linear dofs first
        qd[starts[w] + 2] = -float(v)
    model.joint_qd.assign(qd)
    return model


def resting_boxes_model(worlds: int, per_world: int, device: str | None = None,
                        never_sleep: Sequence[int] = ()) -> newton.Model:
    """CPL-8 scene (T12-like layout): ``per_world`` boxes resting on a ground plane, two thirds on the floor in a grid
    and one third stacked on top of them (box-box resting contacts), MuJoCo contacts. Bodies listed in
    ``never_sleep`` (per-world index) get ``mujoco:sleep_policy`` NEVER (an object the robot keeps awake)."""
    from alloy.core.defaults import RIGID_GAP_M
    from alloy.coupling.sleep import POLICY_NEVER, apply_body_sleep_policies

    b = newton.ModelBuilder()
    b.rigid_gap = RIGID_GAP_M            # never Newton's 0.1 m default (SPEC §4.2): it links resting boxes 5 cm apart
    SolverMuJoCo.register_custom_attributes(b)
    base = per_world - per_world // 3
    side = int(np.ceil(np.sqrt(base)))
    for k in range(per_world):
        slot = k if k < base else k - base
        x, y = 0.15 * (slot % side), 0.15 * (slot // side)
        z = 0.0505 if k < base else 0.1515
        body = b.add_body(xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()))
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    if never_sleep:
        apply_body_sleep_policies(b, {int(k): POLICY_NEVER for k in never_sleep})
    top = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(top)
    top.rigid_gap = RIGID_GAP_M
    top.replicate(b, copy_count=worlds)
    top.add_ground_plane()
    return top.finalize(device=device)


def _admm_world(dz: float = 0.0, dim: int = 6) -> newton.ModelBuilder:
    """Newton's MuJoCo-VBD ADMM example, smaller: a ball attached to a pinned cloth centre (rigid-particle rows) and a
    pendulum link carrying a VBD payload through a ball joint (rigid-rigid rows)."""
    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    cz = 2.0
    p0 = b.particle_count
    b.add_cloth_grid(pos=wp.vec3(-0.3, -0.3, cz), rot=wp.quat_identity(), vel=wp.vec3(0.0), fix_left=True,
                     fix_right=True, dim_x=dim, dim_y=dim, cell_x=0.1, cell_y=0.1, mass=0.05, tri_ke=1.0e3,
                     tri_ka=1.0e3, tri_kd=10.0, edge_ke=0.01, edge_kd=1e-4, particle_radius=0.01)
    centre = p0 + (dim // 2) * (dim + 1) + dim // 2
    ball = b.add_body(xform=wp.transform(p=wp.vec3(0.0, 0.0, cz - 0.08 - dz), q=wp.quat_identity()), mass=0.5,
                      inertia=wp.mat33(np.eye(3) * 5.0e-3), label="ball")
    b.add_shape_sphere(ball, radius=0.08)
    b.add_attachment_particle_body(centre, ball, body_point=wp.vec3(0.0, 0.0, 0.08), stiffness=1.0e3)
    anchor = wp.vec3(1.4, 0.0, 2.2)
    link = b.add_link(xform=wp.transform(p=anchor + wp.vec3(0.28, 0.0, 0.0), q=wp.quat_identity()), mass=0.6,
                      inertia=wp.mat33(np.eye(3) * 1.0e-2), label="link")
    b.add_shape_box(link, hx=0.28, hy=0.045, hz=0.045)
    j = b.add_joint_revolute(parent=-1, child=link, axis=wp.vec3(0.0, 1.0, 0.0), target_kd=0.5,
                             parent_xform=wp.transform(p=anchor, q=wp.quat_identity()),
                             child_xform=wp.transform(p=wp.vec3(-0.28, 0.0, 0.0), q=wp.quat_identity()))
    b.add_articulation([j], label="pendulum")
    payload = b.add_body(xform=wp.transform(p=anchor + wp.vec3(0.68 + dz, 0.0, 0.0), q=wp.quat_identity()),
                         mass=0.35, inertia=wp.mat33(np.eye(3) * 6.0e-3), label="payload")
    b.add_shape_box(payload, hx=0.12, hy=0.09, hz=0.09)
    b.add_joint_ball(parent=link, child=payload, friction=1.0,
                     parent_xform=wp.transform(p=wp.vec3(0.28, 0.0, 0.0), q=wp.quat_identity()),
                     child_xform=wp.transform(p=wp.vec3(-0.12, 0.0, 0.0), q=wp.quat_identity()),
                     collision_filter_parent=True)
    return b


def admm_model(offsets: Sequence[float], device: str | None = None) -> newton.Model:
    top = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(top)
    for dz in offsets:
        top.add_world(_admm_world(dz))
    top.color()
    model = top.finalize(device=device)
    model.soft_contact_ke = 0.0          # the attachment binds ball and cloth; soft contact would fight it
    model.soft_contact_kd = 0.0
    return model


def admm_scheduler(model: newton.Model, policy: Any = None, rho: float = 50.0, gamma: float = 0.1) -> Any:
    """SchedulerADMM: MuJoCo owns ball + link, VBD owns payload + cloth (example settings ρ 50, γ 0.1)."""
    from alloy.coupling.scheduler import SchedulerADMM
    from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy

    labels = list(model.body_label)
    child = model.joint_child.numpy()
    rigid = [i for i, n in enumerate(labels) if n.endswith(("ball", "link"))]
    vbd_b = [i for i, n in enumerate(labels) if n.endswith("payload")]
    e_m = SolverCoupledProxy.Entry(name="mjc", solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False,
                                                                             njmax=32),
                                   bodies=rigid, joints=[j for j, c in enumerate(child) if c in set(rigid)])
    jtype = model.joint_type.numpy()
    free = int(newton.JointType.FREE)
    e_v = SolverCoupledProxy.Entry(name="vbd", solver=lambda v: SolverVBD(model=v, iterations=8), bodies=vbd_b,
                                   joints=[j for j, c in enumerate(child) if c in set(vbd_b) and jtype[j] == free],
                                   particles=list(range(model.particle_count)))
    from alloy.core.newton_compat import SolverCoupledADMM
    cfg = SolverCoupledADMM.Config(rho=rho, gamma=gamma, baumgarte=0.01, joint_proximal_destination_entries=("mjc",))
    return SchedulerADMM(model, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False)],
                         [PairPolicy("mjc", "vbd", mode="admm")], policy if policy is not None else SchedulerPolicy(),
                         admm_config=cfg)


def cloth_sph_model(spec: ClothSceneSpec = ClothSceneSpec(), sph_cfg: Any = None, *, pool_x: float = 1.0,
                    pool_cells: tuple[int, int, int] = (6, 6, 4),
                    device: str | None = None) -> tuple[newton.Model, list[int]]:
    """One box-on-cloth world plus a resting SPH pool in its own tank ``pool_x`` m away, out of reach of the box and
    the cloth: the DC5 composition MuJoCo <-> VBD coupled + MuJoCo -> SPH one-way (SCR-DC5-6). Returns the model and
    the SPH particle ids."""
    from alloy.solvers.sph.build import add_box_tank, add_fluid_block
    from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes

    cfg = sph_cfg if sph_cfg is not None else DFSPHConfig(particle_radius=1.0e-3, coupling="one_way")
    b = _world_builder(spec)
    register_custom_attributes(b)
    s, (nx, ny, nz) = cfg.spacing, pool_cells
    add_box_tank(b, (pool_x, 0.0, 0.0), (pool_x + nx * s, ny * s, (nz + 4) * s), wall=cfg.h)
    first = b.particle_count
    add_fluid_block(b, cfg, (pool_x, 0.0, 0.0), pool_cells)
    sph_ids = list(range(first, b.particle_count))
    top = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(top)
    register_custom_attributes(top)
    top.add_world(b)
    top.color()
    model = top.finalize(device=device)
    model.soft_contact_ke, model.soft_contact_mu = spec.contact_ke, spec.contact_mu
    return model, sph_ids


def cloth_sph_stepper(model: newton.Model, sph_ids: Sequence[int], spec: ClothSceneSpec = ClothSceneSpec(), *,
                      sph_cfg: Any = None, sph_rung: int = 4, with_sph: bool = True, policy: Any = None) -> Any:
    """Scheduler over MuJoCo (box) <-> VBD (cloth only) lagged; with ``with_sph`` wrapped in a ``OneWayFollower``
    whose SolverDFSPH leaf (full model, one-way) runs at ``sph_rung``. Without it the pool is never stepped: the
    reference for "the follower changes nothing"."""
    from alloy.coupling.one_way import OneWayFollower, OneWayLeaf
    from alloy.coupling.scheduler import Scheduler
    from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy
    from alloy.solvers.sph.config import DFSPHConfig
    from alloy.solvers.sph.solver import SolverDFSPH

    pool = set(sph_ids)
    cloth = [i for i in range(model.particle_count) if i not in pool]
    e_m, _ = cloth_entries(model, spec)
    e_v = SolverCoupledProxy.Entry(name="vbd", solver=lambda v: SolverVBD(model=v, iterations=spec.vbd_iterations),
                                   particles=cloth)
    pol = policy if policy is not None else SchedulerPolicy()
    base = Scheduler(model, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False)],
                     [PairPolicy("mjc", "vbd", mode="lagged", proxy=cloth_proxy(model))], pol)
    if not with_sph:
        return base
    cfg = sph_cfg if sph_cfg is not None else DFSPHConfig(particle_radius=1.0e-3, coupling="one_way")
    leaf = OneWayLeaf("sph", SolverDFSPH(model, cfg), list(sph_ids), rung=sph_rung, adaptive=False)
    return OneWayFollower(base, model, [leaf], pol.clock)
