# SPDX-License-Identifier: Apache-2.0
"""CPL-6 (SPEC §5.2, SCR-DC5-6): wall time per sim-second of the multirate clock (MuJoCo at dt_x = 0.625 ms, a
one-way SPH follower at dt_x/ratio) against one global fine clock (both at dt_x/ratio), at ratio 4 (DC5's 0.16 ms)
and 8, with the subcycled fraction f (SPH's share of the fine-clock step time) and the trajectory and law errors of
both runs. Target: >= 5x where f <= 5 % at ratio 8 (ideal 1 / (f + (1 - f)/ratio)).

Scene per world: ``ALLOY_CPL6_BOXES`` free boxes resting on the floor in a grid (the rigid workload, no stacks so no
creep), a static tray with a ``ALLOY_CPL6_POOL`` SPH pool (r = 1 mm) and a hinged paddle (pendulum, L = 30 mm,
released at 0.2 rad) whose blade swings through the liquid: smooth body motion the liquid follows one-way (a dropped
box splashed chaotically, so its liquid trajectory differed between any two clocks). MuJoCo contacts, CUDA graphs.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverMuJoCo

from alloy.coupling.one_way import OneWayFollower, OneWayLeaf
from alloy.coupling.types import CouplingReport
from alloy.solvers.sph.build import add_box_tank, add_fluid_points, lattice_points
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.solver import SolverDFSPH
from alloy.testing.coupling import record_metric

DEV = os.environ.get("ALLOY_DEVICE", "cuda:0")
WORLDS = int(os.environ.get("ALLOY_CPL6_WORLDS", "1024"))
BOXES = int(os.environ.get("ALLOY_CPL6_BOXES", "16"))
POOL = tuple(int(x) for x in os.environ.get("ALLOY_CPL6_POOL", "10,4,2").split(","))
SIM_S = float(os.environ.get("ALLOY_CPL6_SECONDS", "0.3"))
DT_X = 0.625e-3
RATIOS = (4, 8)
HALF = 0.02                       # resting box half size [m]
TRAY_Z = 0.02                     # tray floor height: a tank floor on the ground plane launched the pool (1.4 m/s)
ARM, THETA0 = 0.03, 0.2           # paddle pendulum length [m] and release angle [rad] (blade clears the walls)
REPEATS, WINDOW = 5, 20           # timing: 5 x 20 graph launches (each one dt_x of sim time)
NCONMAX = max(128, 6 * BOXES + 32)   # per world: ~4 floor contacts per box + paddle (overflow is asserted)
NJMAX = 4 * NCONMAX


def _world(cfg: DFSPHConfig, pool: tuple[int, int, int] = POOL) -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(b)
    register_custom_attributes(b)
    side = int(np.ceil(np.sqrt(BOXES)))
    for k in range(BOXES):
        x, y = 0.1 + 0.06 * (k % side), 0.06 * (k // side)
        body = b.add_body(xform=wp.transform(p=wp.vec3(x, y, HALF + 5.0e-4), q=wp.quat_identity()))
        b.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF)
    s, (nx, ny, nz) = cfg.spacing, pool
    lo, hi = (-nx * s - 0.02, 0.0, TRAY_Z), (-0.02, ny * s, TRAY_Z + (nz + 8) * s)
    add_box_tank(b, lo, hi, wall=cfg.h)
    half, pivot = _blade(lo, s, nx, ny)
    pts = lattice_points(lo, pool, s)
    add_fluid_points(b, cfg, pts[~_inside_blade(pts, half, pivot, cfg.particle_radius)])
    _paddle(b, half, pivot)
    return b


def _blade(lo: tuple[float, float, float], s: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Blade half sizes (x 1.5 s, y 0.4·ny·s, z 1.5 s) and hinge position: the blade hangs ARM below the hinge
    (axis y), its bottom 0.75 s above the tray floor at rest, so it stays in the liquid over the whole swing."""
    half = np.array([1.5 * s, 0.4 * ny * s, 1.5 * s])
    pivot = np.array([lo[0] + 0.5 * nx * s, 0.5 * ny * s, lo[2] + 0.75 * s + half[2] + ARM])
    return half, pivot


def _inside_blade(pts: np.ndarray, half: np.ndarray, pivot: np.ndarray, r: float) -> np.ndarray:
    """Lattice points inside the blade at the release angle (with a particle radius margin): left out, as the
    floating-box scene leaves out the cells inside its box."""
    c, si = np.cos(THETA0), np.sin(THETA0)
    x_axis, z_axis = np.array([c, 0.0, -si]), np.array([si, 0.0, c])
    d = pts - (pivot - ARM * z_axis)
    local = np.stack([d @ x_axis, d[:, 1], d @ z_axis], axis=1)
    return np.all(np.abs(local) <= half + r, axis=1)


def _paddle(b: newton.ModelBuilder, half: np.ndarray, pivot_np: np.ndarray) -> None:
    hx, hy, hz = (float(v) for v in half)
    pivot = wp.vec3(*(float(v) for v in pivot_np))
    link = b.add_link(xform=wp.transform(p=wp.vec3(pivot[0], pivot[1], pivot[2] - ARM), q=wp.quat_identity()),
                      label="paddle")
    b.add_shape_box(link, hx=hx, hy=hy, hz=hz)
    joint = b.add_joint_revolute(parent=-1, child=link, parent_xform=wp.transform(p=pivot, q=wp.quat_identity()),
                                 child_xform=wp.transform(p=wp.vec3(0.0, 0.0, ARM), q=wp.quat_identity()),
                                 axis=(0.0, 1.0, 0.0))
    b.add_articulation([joint], label="paddle")      # FK and MuJoCo's state update only cover articulated joints
    b.joint_q[-1] = THETA0


def _model(cfg: DFSPHConfig, pool: tuple[int, int, int] = POOL) -> newton.Model:
    top = newton.ModelBuilder()
    SolverMuJoCo.register_custom_attributes(top)
    register_custom_attributes(top)
    top.rigid_gap = 5.0e-3
    top.replicate(_world(cfg, pool), copy_count=WORLDS)
    top.add_ground_plane()
    return top.finalize(device=DEV)


class _MuJoCoStepper:
    """Single-entry base on the full model (MuJoCo contacts), dt fixed by the caller."""

    def __init__(self, model: newton.Model):
        self.solver = SolverMuJoCo(model, use_mujoco_contacts=True, njmax=NJMAX, nconmax=NCONMAX)
        self.world_count = int(model.world_count)

    def step(self, state_in, state_out, control, contacts, dt):
        self.solver.step(state_in, state_out, control, None, dt)

    def choose_clock(self, control_dt, last):
        return DT_X, {"mjc": 1}

    def report(self):
        return CouplingReport.quiet(DT_X, {"mjc": 1}, self.world_count)

    def state_arrays(self):
        return {}

    def state_world_maps(self):
        return {}


class _Run:
    """One configuration: ``outer`` base steps of ``dt`` per dt_x, the SPH leaf at ``rung`` per base step."""

    def __init__(self, model, cfg, outer: int, rung: int, parts: str = "both"):
        self.model, self.outer, self.dt = model, outer, DT_X / outer
        self.base = _MuJoCoStepper(model)
        self.sph = SolverDFSPH(model, cfg)
        ids = np.flatnonzero(model.sph.phase.numpy() >= 0)
        self.stepper = OneWayFollower(self.base, model, [OneWayLeaf("sph", self.sph, ids, rung=rung, adaptive=False)])
        self.parts = parts
        self.s0, self.s1 = model.state(), model.state()
        self.control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.s0)
        self.rung = rung
        self.sph.prime(self.s0, self.dt / rung)               # hydrostatic warm start of the resting pool

    def _one(self):
        for _ in range(self.outer):
            if self.parts == "both":
                self.stepper.step(self.s0, self.s1, self.control, None, self.dt)
            elif self.parts == "mjc":
                self.base.step(self.s0, self.s1, self.control, None, self.dt)
            else:
                for _ in range(self.rung):
                    self.sph.step(self.s0, self.s1, None, None, self.dt / self.rung)
                    _copy(self.s1, self.s0)
                continue
            _copy(self.s1, self.s0)

    def advance(self, macro_steps: int) -> None:
        for _ in range(macro_steps):
            self._one()

    def time_per_sim_s(self) -> float:
        self._one()                                         # warm up (modules, allocations)
        with wp.ScopedCapture(device=DEV) as cap:
            self._one()
        times = []
        for _ in range(REPEATS):
            wp.synchronize_device(DEV)
            t0 = time.perf_counter()
            for _ in range(WINDOW):
                wp.capture_launch(cap.graph)
            wp.synchronize_device(DEV)
            times.append((time.perf_counter() - t0) / (WINDOW * DT_X))
        self.check_overflow()
        return float(np.median(times))

    def check_overflow(self) -> None:
        overflow = int(self.base.solver.mjw_data.overflow.numpy().max())
        assert overflow == 0, f"MuJoCo contact/row overflow ({overflow}): timing would measure dropped contacts"


def _copy(src, dst):
    for name in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd"):
        a = getattr(src, name)
        if a is not None and a.size:
            wp.copy(getattr(dst, name), a)


def _per_world_max(dq: np.ndarray, counts: int) -> np.ndarray:
    return np.linalg.norm(dq, axis=1).reshape(WORLDS, counts).max(axis=1)


def _errors(fine: _Run, multi: _Run, model) -> dict:
    """Trajectory difference (bodies, liquid centroid) and law errors (SPH density error, escapes, mass)."""
    bq_f, bq_m = fine.s0.body_q.numpy()[:, :3], multi.s0.body_q.numpy()[:, :3]
    ids = np.flatnonzero(model.sph.phase.numpy() >= 0)
    pq_f, pq_m = fine.s0.particle_q.numpy()[ids], multi.s0.particle_q.numpy()[ids]
    per_world = len(ids) // WORLDS
    cen = np.linalg.norm(pq_f.reshape(WORLDS, per_world, 3).mean(1) - pq_m.reshape(WORLDS, per_world, 3).mean(1),
                         axis=1)
    body = _per_world_max(bq_f - bq_m, model.body_count // WORLDS)
    rf, rm = fine.sph.report(), multi.sph.report()
    return {"body_diff_p50_m": float(np.median(body)), "body_diff_max_m": float(body.max()),
            "liquid_centroid_diff_p50_m": float(np.median(cen)), "liquid_centroid_diff_max_m": float(cen.max()),
            "density_err_fine": float(np.max(rf.density_err)), "density_err_multi": float(np.max(rm.density_err)),
            "escaped_fine": int(rf.escaped.sum()), "escaped_multi": int(rm.escaped.sum()),
            "nonfinite": int(not (np.all(np.isfinite(bq_m)) and np.all(np.isfinite(pq_m))))}


@pytest.mark.gpu
@pytest.mark.timeout(3600)
def test_cpl6_multirate_vs_global_fine_clock():
    cfg = DFSPHConfig(particle_radius=1.0e-3, coupling="one_way")
    model = _model(cfg)
    rows = []
    for ratio in RATIOS:
        t_multi = _Run(model, cfg, outer=1, rung=ratio).time_per_sim_s()
        t_fine = _Run(model, cfg, outer=ratio, rung=1).time_per_sim_s()
        t_mjc = _Run(model, cfg, outer=ratio, rung=1, parts="mjc").time_per_sim_s()
        t_sph = _Run(model, cfg, outer=1, rung=ratio, parts="sph").time_per_sim_s()
        f = t_sph / (t_sph + t_mjc)
        speedup = t_fine / t_multi
        ideal = 1.0 / (f + (1.0 - f) / ratio)
        multi, fine = _Run(model, cfg, outer=1, rung=ratio), _Run(model, cfg, outer=ratio, rung=1)
        macro = int(round(SIM_S / DT_X))
        multi.advance(macro)
        fine.advance(macro)
        multi.check_overflow()
        fine.check_overflow()
        err = _errors(fine, multi, model)
        row = {"ratio": ratio, "speedup": speedup, "f": f, "ideal": ideal, "wall_per_sim_s_multi": t_multi,
               "wall_per_sim_s_fine": t_fine, "wall_mjc_fine": t_mjc, "wall_sph_fine": t_sph, **err}
        rows.append(row)
        print(f"CPL-6 ratio {ratio}: speedup {speedup:.2f}x (ideal {ideal:.2f}x at f {f:.3f}); wall/sim-s multi "
              f"{t_multi:.3f} fine {t_fine:.3f} (mjc {t_mjc:.3f}, sph {t_sph:.3f}); errors {err}", flush=True)
    record_metric("CPL-6", device=DEV, worlds=WORLDS, boxes_per_world=BOXES, pool=list(POOL),
                  particles_per_world=int(np.prod(POOL)), dt_x=DT_X, sim_s=SIM_S, rows=rows,
                  timing=f"CUDA graph, {REPEATS}x{WINDOW} launches of dt_x")
    for row in rows:
        assert row["nonfinite"] == 0 and row["escaped_multi"] == 0


DC58_POOL = tuple(int(x) for x in os.environ.get("ALLOY_DC58_POOL", "16,8,3").split(","))
DC58_RUNG = 32                    # DC5's rung at r = 0.2 mm: 32 SPH steps per 0.625 ms exchange


def _one_way_scheduler(model, cfg):
    """Two-entry Scheduler: MuJoCo (bodies, joints) -> SolverDFSPH (one-way, Newton view) at DC58_RUNG."""
    from alloy.core.newton_compat import SolverCoupledProxy
    from alloy.coupling.scheduler import Scheduler
    from alloy.coupling.types import AlloyEntry, ClockPolicy, PairPolicy, SchedulerPolicy

    sph_ids = list(np.flatnonzero(model.sph.phase.numpy() >= 0))
    e_m = SolverCoupledProxy.Entry(name="mjc", bodies=list(range(model.body_count)),
                                   joints=list(range(model.joint_count)),
                                   solver=lambda v: SolverMuJoCo(v, use_mujoco_contacts=True, njmax=NJMAX,
                                                                 nconmax=NCONMAX))
    e_s = SolverCoupledProxy.Entry(name="sph", particles=sph_ids, substeps=64, solver=lambda v: SolverDFSPH(v, cfg))
    proxy = SolverCoupledProxy.Proxy(source="mjc", destination="sph", bodies=list(range(model.body_count)),
                                     mode="lagged", collision_pipeline=lambda _m: None)
    pol = SchedulerPolicy(clock=ClockPolicy(dt_x_max=DT_X, omega_dt_max=float("inf")))
    return Scheduler(model, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_s, initial_rung=DC58_RUNG, adaptive=False)],
                     [PairPolicy("mjc", "sph", proxy=proxy, one_way=True)], pol)


def _time_exchange(model, sched, passes: int) -> tuple[float, newton.State]:
    """Median wall time of one dt_x exchange (CUDA graph) with ``passes`` passes; the state after 40 exchanges."""
    sched._one_way_passes = passes
    sched.choose_clock(0.02, None)
    s0, s1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)

    def one():
        sched.step(s0, s1, None, None, DT_X)
        _copy(s1, s0)
    for _ in range(40):
        one()
    state = model.state()
    _copy(s0, state)
    with wp.ScopedCapture(device=DEV) as cap:
        one()
    times = []
    for _ in range(REPEATS):
        wp.synchronize_device(DEV)
        t0 = time.perf_counter()
        for _ in range(5):
            wp.capture_launch(cap.graph)
        wp.synchronize_device(DEV)
        times.append((time.perf_counter() - t0) / 5)
    return float(np.median(times)), state


@pytest.mark.gpu
@pytest.mark.timeout(3600)
def test_dc5_8_one_pass_time_saved_at_r_02mm():
    """SCR-DC5-8 on GPU: wall time per exchange of the one-way MuJoCo -> SPH pair with one pass vs the old two, at
    r = 0.2 mm and rung 32 (32 SPH steps per pass). The bitwise proof is the CPU test in test_one_way.py; here the
    GPU difference after 40 exchanges is reported (MuJoCo Warp float atomics may differ between re-steps)."""
    cfg = DFSPHConfig(particle_radius=2.0e-4, coupling="one_way")
    model = _model(cfg, DC58_POOL)
    t1, s_one = _time_exchange(model, _one_way_scheduler(model, cfg), 1)
    t2, s_two = _time_exchange(model, _one_way_scheduler(model, cfg), 2)
    ids = np.flatnonzero(model.sph.phase.numpy() >= 0)
    dq = float(np.max(np.abs(s_one.particle_q.numpy()[ids] - s_two.particle_q.numpy()[ids])))
    db = float(np.max(np.abs(s_one.body_q.numpy() - s_two.body_q.numpy())))
    record_metric("DC5-8", device=DEV, worlds=WORLDS, boxes_per_world=BOXES, pool=list(DC58_POOL), r=2.0e-4,
                  rung=DC58_RUNG, ms_per_exchange_one_pass=t1 * 1e3, ms_per_exchange_two_pass=t2 * 1e3,
                  saved_fraction=1.0 - t1 / t2, sph_steps_per_exchange=(DC58_RUNG, 2 * DC58_RUNG),
                  max_dq_sph_m=dq, max_dq_body_m=db)
    print(f"DC5-8: {t1 * 1e3:.2f} ms/exchange one pass vs {t2 * 1e3:.2f} two passes (saved {1 - t1 / t2:.1%}); "
          f"SPH steps {DC58_RUNG} vs {2 * DC58_RUNG}; after 40 exchanges max |dq| SPH {dq:.3e} m, bodies {db:.3e} m")
    assert t1 < t2
