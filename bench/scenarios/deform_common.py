# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the `deform` family: device-side schedules and recorders, world assembly, solver factories.

Every scripted input (gravity ramps, applied forces, kinematic poses) is a Warp kernel indexed by a device substep
counter, so the scenarios replay inside CUDA graphs (bench/SCENARIOS.md "No host sync inside step()"). Traces of
chosen points are written on device every `stride` substeps and read once in finish().

Analytic laws live in deform_laws.py, mesh and measurement geometry in deform_geom.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from alloy._vendor import newton
import numpy as np
import warp as wp

from bench.harness import Built, Config, Scenario

G = 9.81
REC_WORLDS = 64  # traces are kept for the first REC_WORLDS worlds only (laws need ncond worlds, not 4096)


# ---- kernels ----------------------------------------------------------------------------------------------------
@wp.func
def ease(u: float) -> float:
    """Smoothstep 0 -> 1 on u in [0, 1] (zero slope at both ends)."""
    x = wp.clamp(u, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@wp.kernel
def k_tick(counter: wp.array(dtype=wp.int32)):
    counter[0] = counter[0] + 1


@wp.kernel
def k_gravity_ramp(counter: wp.array(dtype=wp.int32), h: float, t0: float, t1: float, g0: wp.vec3,
                   gravity: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    t = float(counter[0]) * h
    gravity[i] = g0 * ease((t - t0) / wp.max(t1 - t0, 1.0e-9))


@wp.kernel
def k_particle_force(counter: wp.array(dtype=wp.int32), h: float, t0: float, t1: float,
                     ids: wp.array(dtype=wp.int32), force: wp.array(dtype=wp.vec3),
                     particle_f: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    t = float(counter[0]) * h
    p = ids[i]
    particle_f[p] = particle_f[p] + force[i] * ease((t - t0) / wp.max(t1 - t0, 1.0e-9))


@wp.kernel
def k_body_force(counter: wp.array(dtype=wp.int32), h: float, t0: float, t1: float,
                 ids: wp.array(dtype=wp.int32), force: wp.array(dtype=wp.vec3),
                 body_f: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    t = float(counter[0]) * h
    b = ids[i]
    f = force[i] * ease((t - t0) / wp.max(t1 - t0, 1.0e-9))
    body_f[b] = body_f[b] + wp.spatial_vector(f, wp.vec3(0.0))


@wp.kernel
def k_body_pose_table(counter: wp.array(dtype=wp.int32), h: float, row_dt: float,
                      ids: wp.array(dtype=wp.int32), table: wp.array2d(dtype=wp.transform),
                      body_q0: wp.array(dtype=wp.transform), body_q1: wp.array(dtype=wp.transform),
                      body_qd0: wp.array(dtype=wp.spatial_vector), body_qd1: wp.array(dtype=wp.spatial_vector)):
    """Kinematic body poses from a time table (rows at row_dt), linear in p and slerp in q; velocity by difference."""
    i = wp.tid()
    t = float(counter[0]) * h
    rows = table.shape[0]
    u = t / row_dt
    r0 = wp.min(int(u), rows - 1)
    r1 = wp.min(r0 + 1, rows - 1)
    a = wp.clamp(u - float(r0), 0.0, 1.0)
    T0 = table[r0, i]
    T1 = table[r1, i]
    p = wp.lerp(wp.transform_get_translation(T0), wp.transform_get_translation(T1), a)
    q = wp.quat_slerp(wp.transform_get_rotation(T0), wp.transform_get_rotation(T1), a)
    T = wp.transform(p, q)
    b = ids[i]
    v = wp.vec3(0.0)
    if r1 > r0:
        v = (wp.transform_get_translation(T1) - wp.transform_get_translation(T0)) / row_dt
    body_q0[b] = T
    body_q1[b] = T
    body_qd0[b] = wp.spatial_vector(v, wp.vec3(0.0))
    body_qd1[b] = wp.spatial_vector(v, wp.vec3(0.0))


@wp.kernel
def k_record_particles(counter: wp.array(dtype=wp.int32), stride: int, ids: wp.array(dtype=wp.int32),
                       particle_q: wp.array(dtype=wp.vec3), out: wp.array2d(dtype=wp.vec3)):
    i = wp.tid()
    n = counter[0] + 1
    if n % stride == 0:
        row = n / stride - 1
        if row < out.shape[0]:
            out[row, i] = particle_q[ids[i]]


@wp.kernel
def k_record_bodies(counter: wp.array(dtype=wp.int32), stride: int, ids: wp.array(dtype=wp.int32),
                    local: wp.array(dtype=wp.vec3), body_q: wp.array(dtype=wp.transform),
                    out: wp.array2d(dtype=wp.vec3)):
    i = wp.tid()
    n = counter[0] + 1
    if n % stride == 0:
        row = n / stride - 1
        if row < out.shape[0]:
            out[row, i] = wp.transform_point(body_q[ids[i]], local[i])


# ---- driver -------------------------------------------------------------------------------------------------------
@dataclass
class _Trace:
    kind: str  # "particle" or "body"
    ids: wp.array
    local: wp.array | None
    stride: int
    out: wp.array


@dataclass
class Driver:
    """Device clock for one Built: pre-solve inputs, post-solve recorders, one substep counter."""

    device: str
    h: float  # solver substep [s]
    counter: wp.array = None
    pre: list[Callable] = field(default_factory=list)
    traces: dict[str, _Trace] = field(default_factory=dict)

    def __post_init__(self):
        self.counter = wp.zeros(1, dtype=wp.int32, device=self.device)

    # inputs ---------------------------------------------------------------------------------------------------
    def gravity_ramp(self, model, t0: float, t1: float, g0=(0.0, 0.0, -G)) -> None:
        g, n = wp.vec3(*g0), model.gravity.shape[0]
        self.pre.append(lambda s0, s1: wp.launch(k_gravity_ramp, dim=n, device=self.device,
                                                 inputs=[self.counter, self.h, t0, t1, g, model.gravity]))

    def particle_force(self, ids: Sequence[int], forces: np.ndarray, t0: float, t1: float) -> None:
        if len(ids) == 0:
            return
        ids_w = wp.array(np.asarray(ids, np.int32), dtype=wp.int32, device=self.device)
        f_w = wp.array(np.asarray(forces, np.float32).reshape(-1, 3), dtype=wp.vec3, device=self.device)
        self.pre.append(lambda s0, s1: wp.launch(k_particle_force, dim=len(ids), device=self.device,
                                                 inputs=[self.counter, self.h, t0, t1, ids_w, f_w, s0.particle_f]))

    def body_force(self, ids: Sequence[int], forces: np.ndarray, t0: float, t1: float) -> None:
        if len(ids) == 0:
            return
        ids_w = wp.array(np.asarray(ids, np.int32), dtype=wp.int32, device=self.device)
        f_w = wp.array(np.asarray(forces, np.float32).reshape(-1, 3), dtype=wp.vec3, device=self.device)
        self.pre.append(lambda s0, s1: wp.launch(k_body_force, dim=len(ids), device=self.device,
                                                 inputs=[self.counter, self.h, t0, t1, ids_w, f_w, s0.body_f]))

    def body_pose_table(self, ids: Sequence[int], table: np.ndarray, row_dt: float) -> None:
        """table: (rows, len(ids), 7) transforms (px, py, pz, qx, qy, qz, qw)."""
        ids_w = wp.array(np.asarray(ids, np.int32), dtype=wp.int32, device=self.device)
        tab = wp.array(np.asarray(table, np.float32), dtype=wp.transform, device=self.device)
        self.pre.append(lambda s0, s1: wp.launch(
            k_body_pose_table, dim=len(ids), device=self.device,
            inputs=[self.counter, self.h, row_dt, ids_w, tab, s0.body_q, s1.body_q, s0.body_qd, s1.body_qd]))

    # recorders ------------------------------------------------------------------------------------------------
    def record_particles(self, name: str, ids: Sequence[int], stride: int, rows: int) -> None:
        ids_w = wp.array(np.asarray(ids, np.int32), dtype=wp.int32, device=self.device)
        out = wp.zeros((rows, len(ids)), dtype=wp.vec3, device=self.device)
        self.traces[name] = _Trace("particle", ids_w, None, stride, out)

    def record_bodies(self, name: str, ids: Sequence[int], local: np.ndarray, stride: int, rows: int) -> None:
        ids_w = wp.array(np.asarray(ids, np.int32), dtype=wp.int32, device=self.device)
        loc = wp.array(np.asarray(local, np.float32).reshape(-1, 3), dtype=wp.vec3, device=self.device)
        out = wp.zeros((rows, len(ids)), dtype=wp.vec3, device=self.device)
        self.traces[name] = _Trace("body", ids_w, loc, stride, out)

    def trace(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(t [rows], xyz [rows, n, 3]) of the rows recorded so far."""
        tr = self.traces[name]
        n = int(self.counter.numpy()[0])
        rows = min(n // tr.stride, tr.out.shape[0])
        t = (np.arange(rows) + 1) * tr.stride * self.h
        return t, tr.out.numpy()[:rows].astype(np.float64)

    def time(self) -> float:
        return float(self.counter.numpy()[0]) * self.h

    # hooks ----------------------------------------------------------------------------------------------------
    def run_pre(self, s0, s1) -> None:
        for fn in self.pre:
            fn(s0, s1)

    def run_post(self, s0) -> None:
        for tr in self.traces.values():
            if tr.kind == "particle":
                wp.launch(k_record_particles, dim=tr.ids.shape[0], device=self.device,
                          inputs=[self.counter, tr.stride, tr.ids, s0.particle_q, tr.out])
            else:
                wp.launch(k_record_bodies, dim=tr.ids.shape[0], device=self.device,
                          inputs=[self.counter, tr.stride, tr.ids, tr.local, s0.body_q, tr.out])
        wp.launch(k_tick, dim=1, device=self.device, inputs=[self.counter])


def rec_rows(horizon_s: float, h: float, stride: int, margin: int = 8) -> int:
    return int(np.ceil(horizon_s / (h * stride))) + margin


# ---- scenario base --------------------------------------------------------------------------------------------
class DeformScenario(Scenario):
    """Scenario whose substep runs Driver inputs before the solve and recorders after it (graph safe)."""

    family = "deform"

    def substep(self, b: Built, dt: float) -> None:
        drv: Driver = b.extra["driver"]
        b.state_0.clear_forces()
        drv.run_pre(b.state_0, b.state_1)
        if b.pipeline is not None:
            b.pipeline.collide(b.state_0, b.contacts)
        b.solver.step(b.state_0, b.state_1, b.control, b.contacts, dt)
        b.state_0, b.state_1 = b.state_1, b.state_0
        drv.run_post(b.state_0)

    @staticmethod
    def horizon(rec: dict) -> float:
        return float(rec["horizon_s"])


# ---- world assembly ---------------------------------------------------------------------------------------------
@dataclass
class Assembly:
    """A finalized-to-be builder with nworld worlds and each world's entity offsets."""

    builder: newton.ModelBuilder
    cond: np.ndarray  # condition index per world
    particle_start: np.ndarray
    body_start: np.ndarray
    joint_start: np.ndarray
    dof_start: np.ndarray


def assemble(templates: Sequence[newton.ModelBuilder], nworld: int, gravity: float = -G,
             register: Callable[[newton.ModelBuilder], None] | None = None) -> Assembly:
    """World w holds templates[w % len(templates)] (bench convention: conditions tile over worlds)."""
    main = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    if register is not None:
        register(main)
    ncond = len(templates)
    cond = np.arange(nworld) % ncond
    if ncond == 1:
        main.replicate(templates[0], copy_count=nworld)
    else:
        for w in range(nworld):
            main.add_world(templates[cond[w]])

    def starts(count: Callable[[newton.ModelBuilder], int]) -> np.ndarray:
        sizes = np.array([count(templates[c]) for c in cond], dtype=np.int64)
        return np.concatenate([[0], np.cumsum(sizes)[:-1]])

    return Assembly(main, cond, starts(lambda t: t.particle_count), starts(lambda t: t.body_count),
                    starts(lambda t: t.joint_count), starts(lambda t: len(t.joint_qd)))


def pin_particles(builder: newton.ModelBuilder, ids: Sequence[int]) -> None:
    """Make particles kinematic the way add_cloth_grid(fix_*) does: zero mass, ACTIVE flag cleared."""
    for i in ids:
        builder.particle_mass[i] = 0.0
        builder.particle_flags[i] = int(builder.particle_flags[i]) & ~int(newton.ParticleFlags.ACTIVE)


def make_kinematic_body(builder: newton.ModelBuilder, body: int) -> None:
    builder.body_mass[body] = 0.0
    builder.body_inv_mass[body] = 0.0
    builder.body_inertia[body] = wp.mat33(0.0)
    builder.body_inv_inertia[body] = wp.mat33(0.0)


def world_ids(start: np.ndarray, local: Sequence[int], worlds: Sequence[int]) -> np.ndarray:
    """Global indices of local entity ids for the given worlds, shape (len(worlds), len(local))."""
    return start[np.asarray(worlds)][:, None] + np.asarray(local, dtype=np.int64)[None, :]


def recorded_worlds(nworld: int, ncond: int = 1) -> list[int]:
    return list(range(min(nworld, max(REC_WORLDS, ncond))))


# ---- solver factories -------------------------------------------------------------------------------------------
def make_particle_solver(cfg: Config, model: newton.Model):
    """Cloth / soft-body solver from a Config (kwargs live in cfg.params so they appear in the record)."""
    S, p = newton.solvers, cfg.params
    if cfg.solver == "vbd":
        kw = dict(iterations=p.get("iterations", 10))
        if p.get("self_contact", False):
            kw.update(particle_enable_self_contact=True,
                      particle_self_contact_radius=p.get("self_contact_radius", 0.002),
                      particle_self_contact_margin=p.get("self_contact_margin", 0.003))
        if "tile_solve" in p:
            kw["particle_enable_tile_solve"] = p["tile_solve"]
        return S.SolverVBD(model, **kw)
    if cfg.solver == "style3d":
        return S.SolverStyle3D(model=model, iterations=p.get("iterations", 10))
    if cfg.solver == "xpbd":
        return S.SolverXPBD(model, iterations=p.get("iterations", 10))
    if cfg.solver == "semi_implicit":
        return S.SolverSemiImplicit(model)
    raise ValueError(f"unknown particle solver {cfg.solver!r}")


def law_record(rec: dict, err: float, unit: str | None = None, **metrics) -> None:
    """Fill the headline accuracy fields and merge scenario metrics (NaN becomes None)."""
    rec["accuracy_error"] = None if err is None or not np.isfinite(err) else float(err)
    if unit is not None:
        rec["accuracy_unit"] = unit
    rec["metrics"].update({k: _clean(v) for k, v in metrics.items()})


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_clean(x) for x in v.tolist()]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def period_from_crossings(t: np.ndarray, y: np.ndarray, skip: float = 0.0) -> float:
    """Mean period from upward mean-crossings of y(t) after time `skip` (NaN with < 3 crossings)."""
    sel = t >= skip
    t, y = t[sel], y[sel]
    if len(t) < 4:
        return float("nan")
    yc = y - y.mean()
    idx = np.where((yc[:-1] < 0) & (yc[1:] >= 0))[0]
    if len(idx) < 3:
        return float("nan")
    tc = t[idx] - yc[idx] * (t[idx + 1] - t[idx]) / (yc[idx + 1] - yc[idx])
    return float(np.mean(np.diff(tc)))


def flag_blowup(state, rec: dict, vmax: float = 50.0, xmax: float = 10.0) -> bool:
    """Mark the record unstable when particles run away (the harness only catches NaN/Inf; XPBD clamps speeds).

    True (and rec["stable"] = False, metrics.blowup = True) when any particle is faster than vmax [m/s] or farther
    than xmax [m] from the origin at the end of the run.
    """
    if state.particle_q is None or state.particle_q.shape[0] == 0:
        return False
    x = state.particle_q.numpy()
    v = state.particle_qd.numpy()
    bad = bool(np.max(np.linalg.norm(v, axis=1)) > vmax or np.max(np.linalg.norm(x, axis=1)) > xmax)
    if bad:
        rec["stable"] = False
        rec["metrics"]["blowup"] = True
    return bad


def pct(a: float, ref: float) -> float:
    return 100.0 * (a / ref - 1.0) if ref else float("nan")
