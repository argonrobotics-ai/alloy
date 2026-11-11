# SPDX-License-Identifier: Apache-2.0
"""Granular and fluids family (implicit MPM): G01 column collapse + repose, G02 dam break + settle,
G03 viscous gravity current + funnel, G04 blade digging. Spec: bench/SCENARIOS.md §1.4. Notes: bench/notes/granular.md.

    python -m bench.run --family granular --device cpu --ladder 1 --horizon 0.1 --repeat-max-nworld 0   # Mac smoke
    python -m bench.run --family granular --device cuda:0 --ladder 1,4,16,256                            # GPU VM

Scenes derive from Newton ac7f7830 newton/examples/mpm/: example_mpm_granular.py (sand material, solver defaults),
example_mpm_water_dam_break.py (water material, P0/Q1/pic, project_outside + tank clamp), example_mpm_viscous.py
(viscous material, funnel mesh), example_mpm_pinch_carry.py (kinematic body colliders driven by a device clock,
collect_collider_impulses), and alloy bench/proto/solvers/mpm_dambreak.py (Martin & Moyce slab geometry).

Layout: conditions tile over worlds (world w runs condition w % ncond). Worlds sit side by side in x/y with a gap of
several voxels, so they never share a grid cell; with params separate_worlds="auto" the solver also isolates them
(Config.separate_worlds) on CUDA. Newton's per-world NanoVDB grids are CUDA-only ("NanoVDB environment volumes can
only be built on CUDA devices", warp nanogrid.py:1216), so on CPU the spatial gap alone keeps worlds apart. The
effective value is recorded in metrics.separate_worlds. Each world confines its particles to a box (frictionless
walls) by static colliders plus a post-step clamp, as the dam-break example does. Stepping is in place
(solver.step(s, s)), as in the pinch-carry example; scripted colliders read a device clock, so CUDA graphs stay valid.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp
from alloy._vendor.newton.solvers import SolverImplicitMPM

from bench.harness import Built, Config, Scenario
from bench.scenarios.granular_laws import (  # noqa: F401 - laws re-exported for callers of this module
    MM_T, MM_Z, G, deposit_front, fit_power, fit_power_offset, heap_angle_deg, heap_profile, huppert_radius,
    lube_runout, nrmse, occupancy_volume, rankine_force, ritter_front, surface_volume)

LADDER = (1, 4, 16, 256, 1024)
MEASURE_MAX_WORLDS = 64  # host measurement covers the first worlds only (all conditions at every rung)
T_EASE = 0.5  # [s] kinematic drives ramp their velocity over this time (SCENARIOS.md: loads eased over >= 0.5 s)
OPEN = 1.0e3  # [m] "no wall" bound for the confinement clamp
SAND_MU = math.tan(math.radians(30.0))  # Drucker-Prager friction coefficient; configured angle = atan(mu) = 30 deg
SAND_RHO = 1500.0
WATER_RHO = 1000.0


# ---- warp kernels ------------------------------------------------------------------------------------------
@wp.kernel
def _advance_time(time: wp.array(dtype=float), dt: float):
    time[0] = time[0] + dt


@wp.func
def _clamp_axis(p: float, v: float, lo: float, hi: float):
    if p < lo:
        return lo, wp.max(v, 0.0)
    if p > hi:
        return hi, wp.min(v, 0.0)
    return p, v


@wp.kernel
def _clamp_box(q: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3), lo: wp.array(dtype=wp.vec3),
               hi: wp.array(dtype=wp.vec3)):
    """Frictionless confinement: project particles back into their world's box and drop outward velocity."""
    i = wp.tid()
    p, v, a, b = q[i], qd[i], lo[i], hi[i]
    x, vx = _clamp_axis(p[0], v[0], a[0], b[0])
    y, vy = _clamp_axis(p[1], v[1], a[1], b[1])
    z, vz = _clamp_axis(p[2], v[2], a[2], b[2])
    q[i] = wp.vec3(x, y, z)
    qd[i] = wp.vec3(vx, vy, vz)


@wp.kernel
def _drive_bodies(time: wp.array(dtype=float), body_ids: wp.array(dtype=int), p0: wp.array(dtype=wp.vec3),
                  vel: wp.array(dtype=wp.vec3), t_start: float, t_ease: float, body_q: wp.array(dtype=wp.transform),
                  body_qd: wp.array(dtype=wp.spatial_vector)):
    """Kinematic translation p0 + s(t) vel with the speed ramped linearly over t_ease after t_start."""
    k = wp.tid()
    t = time[0] - t_start
    s = float(0.0)
    f = float(0.0)
    if t > 0.0:
        if t < t_ease:
            s = 0.5 * t * t / t_ease
            f = t / t_ease
        else:
            s = t - 0.5 * t_ease
            f = 1.0
    b = body_ids[k]
    body_q[b] = wp.transform(p0[k] + s * vel[k], wp.quat_identity())
    body_qd[b] = wp.spatial_vector(f * vel[k], wp.vec3(0.0))


def sphere_penetration_mm(solver) -> np.ndarray:
    """Depth [mm] of every active MuJoCo contact (-dist, clipped at 0): MjData on the CPU reference, else mjw_data."""
    if getattr(solver, "use_mujoco_cpu", False):
        dc = solver.mj_data
        return np.clip(-np.asarray(dc.contact.dist[: dc.ncon], dtype=np.float64), 0.0, None) * 1e3
    d = getattr(solver, "mjw_data", None)
    if d is None:
        return np.zeros(0)
    n = min(int(d.nacon.numpy()[0]), int(d.naconmax))
    return np.clip(-d.contact.dist.numpy()[:n].astype(np.float64), 0.0, None) * 1e3 if n else np.zeros(0)


def drive_offset(t: float, t_start: float, t_ease: float = T_EASE) -> float:
    """Host mirror of _drive_bodies: displacement per unit speed at time t."""
    u = t - t_start
    return 0.0 if u <= 0 else (0.5 * u * u / t_ease if u < t_ease else u - 0.5 * t_ease)


# ---- scene description -------------------------------------------------------------------------------------
@dataclass
class KinematicBody:
    p0: tuple[float, float, float]
    vel: tuple[float, float, float]
    boxes: list[tuple[tuple[float, float, float], tuple[float, float, float], float]]  # (centre, half, yaw)
    mu: float = 0.0


@dataclass
class MpmWorld:
    """One world's scene in local coordinates (the builder adds the world origin)."""

    points: np.ndarray
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    material: dict[str, float]
    density: float
    bbox: tuple[float, float, float, float]  # local xy extent incl. walls: (xmin, ymin, xmax, ymax)
    statics: list[tuple[tuple[float, float, float], tuple[float, float, float], float]] = field(default_factory=list)
    meshes: list[tuple[np.ndarray, np.ndarray, float]] = field(default_factory=list)
    bodies: list[KinematicBody] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def jittered_grid(lo, hi, spacing: float, seed: int = 0) -> np.ndarray:
    """Cell-centred particle grid on [lo, hi] with +-1/4 spacing jitter (deterministic)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    axes = [np.arange(lo[k] + 0.5 * spacing, hi[k] - 0.25 * spacing, spacing) for k in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    rng = np.random.default_rng(seed)
    return pts + (rng.random(pts.shape) - 0.5) * 0.5 * spacing


def walls_for_box(lo, hi, thick: float, height: float, sides: str = "xXyY", mu: float = 0.0):
    """Static wall boxes just outside the faces of the confinement box ('x' = -x face, 'X' = +x face, ...)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    cx, cy = 0.5 * (lo[:2] + hi[:2])
    hx, hy = 0.5 * (hi[:2] - lo[:2]) + 2.0 * thick
    hz, cz = 0.5 * height, 0.5 * height
    out = []
    if "x" in sides:
        out.append(((lo[0] - thick, cy, cz), (thick, hy, hz), mu))
    if "X" in sides:
        out.append(((hi[0] + thick, cy, cz), (thick, hy, hz), mu))
    if "y" in sides:
        out.append(((cx, lo[1] - thick, cz), (hx, thick, hz), mu))
    if "Y" in sides:
        out.append(((cx, hi[1] + thick, cz), (hx, thick, hz), mu))
    return out


MPM_DEFAULTS: dict[str, Any] = dict(
    voxel=0.01, ppc=2, max_iterations=250, tolerance=1e-4, strain_basis="P0", collider_basis="S2",
    velocity_basis="Q1", warmstart_mode="auto", transfer_scheme="apic", integration_scheme="pic", solver="auto",
    critical_fraction=0.0, air_drag=1.0, grid_type="sparse", separate_worlds="auto", projection_frac=None,
    stack_worlds=False,  # diagnostic: every world at the same origin (needs separate_worlds; see _assemble)
    align_worlds=False,  # diagnostic: world origins rounded to the voxel lattice (same grid phase in every world)
)
_CONFIG_KEYS = ("max_iterations", "tolerance", "strain_basis", "collider_basis", "velocity_basis", "warmstart_mode",
                "transfer_scheme", "integration_scheme", "solver", "critical_fraction", "air_drag", "grid_type")


def mpm_config(name: str, voxel: float, dt: float, max_nworld: int, **params) -> Config:
    """Config for an implicit-MPM run; params override MPM_DEFAULTS and the scenario's own defaults."""
    return Config(name, "implicit_mpm", dt=dt, substeps=1, params=dict(voxel=voxel, **params), max_nworld=max_nworld)


# ---- MPM scenario base -------------------------------------------------------------------------------------
class MpmScenario(Scenario):
    """Builds tiled MPM worlds, steps in place with clamps and drives, and offers per-world particle access."""

    ladder = LADDER
    mpm_defaults: dict[str, Any] = {}
    floor_mu: float = 0.0
    drive_start: float = 0.0  # [s] when kinematic bodies start moving

    def conditions(self, config: Config) -> list[Any]:
        raise NotImplementedError

    def make_world(self, cond: Any, config: Config, p: dict[str, Any]) -> MpmWorld:
        raise NotImplementedError

    def params(self, config: Config) -> dict[str, Any]:
        return {**MPM_DEFAULTS, **self.mpm_defaults, **config.params}

    # -- build --------------------------------------------------------------------------------------------
    def build(self, nworld: int, device: str, config: Config) -> Built:
        if config.solver != "implicit_mpm":
            raise ValueError(f"{self.id}: unknown solver {config.solver!r}")
        p = self.params(config)
        conds = self.conditions(config)
        worlds = [self.make_world(conds[w % len(conds)], config, p) for w in range(nworld)]
        model, info = _assemble(worlds, p, self.floor_mu, device)
        solver, sep, capacity = _make_solver(model, p, device, nworld)
        state = model.state()
        state.mpm.particle_Jp.fill_(1.0)
        ex = dict(info, conds=conds, ncond=len(conds), p=p, h=config.dt / config.substeps, separate_worlds=sep, capacity=capacity,
                  time=wp.zeros(1, dtype=float, device=device), samples=[])
        if info["drive_ids"] is not None:
            self._drive(ex, state)
        solver.reset(state)
        solver.project_outside(state, state, ex["h"])
        ex["q0"] = state.particle_q.numpy().copy()
        return Built(model, solver, state, state, None, nworld=nworld, extra=ex)

    def _drive(self, ex: dict, state) -> None:
        wp.launch(_drive_bodies, dim=len(ex["drive_ids"]),
                  inputs=[ex["time"], ex["drive_ids"], ex["drive_p0"], ex["drive_vel"], self.drive_start, T_EASE],
                  outputs=[state.body_q, state.body_qd])

    # -- stepping ------------------------------------------------------------------------------------------
    def substep(self, b: Built, dt: float) -> None:
        ex, s = b.extra, b.state_0
        if ex["drive_ids"] is not None:
            self._drive(ex, s)
        b.solver.step(s, s, None, None, dt)
        b.solver.project_outside(s, s, dt)
        wp.launch(_clamp_box, dim=b.model.particle_count, inputs=[s.particle_q, s.particle_qd, ex["lo"], ex["hi"]])
        wp.launch(_advance_time, dim=1, inputs=[ex["time"], dt])

    # -- measurement helpers -------------------------------------------------------------------------------
    def measured_worlds(self, b: Built) -> range:
        return range(min(b.nworld, MEASURE_MAX_WORLDS))

    def local(self, b: Built, q: np.ndarray, w: int) -> np.ndarray:
        s, e = b.extra["ranges"][w]
        return q[s:e] - b.extra["origins"][w]

    def occupancy(self, b: Built, x: np.ndarray) -> float:
        return occupancy_volume(x, b.extra["p"]["voxel"], float(b.extra["p"]["ppc"]) ** 3)

    def body_force(self, b: Built) -> np.ndarray:
        """Force [N] on each body from the last step's collider impulses (impulse on the collider / h)."""
        imp, _, cid = b.solver.collect_collider_impulses(b.state_0)
        imp, cid = imp.numpy(), cid.numpy()
        cbi = b.solver.collider_body_index.numpy()
        body = np.where(cid >= 0, cbi[np.maximum(cid, 0)], -2)
        out = np.zeros((b.model.body_count, 3))
        ok = body >= 0
        for k in range(3):
            out[:, k] = np.bincount(body[ok], weights=imp[ok, k], minlength=b.model.body_count)
        return out / b.extra["h"]

    def floor_load(self, b: Built, w: int) -> float:
        """Vertical load [N] on the static colliders (floor, walls) inside world w's footprint."""
        imp, pos, cid = b.solver.collect_collider_impulses(b.state_0)
        imp, pos, cid = imp.numpy(), pos.numpy(), cid.numpy()
        cbi = b.solver.collider_body_index.numpy()
        static = (cid >= 0) & (cbi[np.maximum(cid, 0)] < 0)
        bb = b.extra["worlds"][w].bbox
        loc = pos - b.extra["origins"][w]
        inside = static & (loc[:, 0] > bb[0]) & (loc[:, 0] < bb[2]) & (loc[:, 1] > bb[1]) & (loc[:, 1] < bb[3])
        return float(-imp[inside, 2].sum() / b.extra["h"])

    def cond_of(self, b: Built, w: int) -> Any:
        return b.extra["conds"][w % b.extra["ncond"]]

    def common_finish(self, b: Built, rec: dict) -> None:
        ex = b.extra
        overflow = False
        try:
            b.solver.check_sparse_grid_rebuild_status()
        except RuntimeError as exc:  # recorded, never ignored: the law then fails
            overflow = True
            rec["notes"] += f"sparse grid overflow: {exc}; "
        q = b.state_0.particle_q.numpy()
        below = int(np.sum(q[:, 2] < -ex["p"]["voxel"]))
        rec["metrics"].update(
            particles=int(b.model.particle_count), particles_per_world=int(b.model.particle_count / b.nworld),
            separate_worlds=ex["separate_worlds"], grid_capacity=ex["capacity"], grid_overflow=overflow,
            conditions=[str(c) for c in ex["conds"]], conditions_covered=b.nworld >= ex["ncond"],
            measured_worlds=len(self.measured_worlds(b)), escaped_below_floor=below,
            mass_loss_pct=100.0 * below / max(1, b.model.particle_count))
        ex["overflow"] = overflow or below > 0


def _assemble(worlds: list[MpmWorld], p: dict, floor_mu: float, device: str):
    """One model with every world placed on an xy grid; returns (model, info dict for Built.extra)."""
    voxel = p["voxel"]
    spacing = voxel / p["ppc"]
    gap = max(8.0 * voxel, 0.05)
    pitch_x = max(w.bbox[2] - w.bbox[0] for w in worlds) + gap
    pitch_y = max(w.bbox[3] - w.bbox[1] for w in worlds) + gap
    ncol = int(math.ceil(math.sqrt(len(worlds))))
    main = newton.ModelBuilder()
    SolverImplicitMPM.register_custom_attributes(main)
    origins, ranges, lo, hi, mats = [], [], [], [], []
    drive_ids, drive_p0, drive_vel, world_bodies = [], [], [], []
    for w, sc in enumerate(worlds):
        origin = np.array([(w % ncol) * pitch_x - sc.bbox[0], (w // ncol) * pitch_y - sc.bbox[1], 0.0])
        if p["stack_worlds"]:  # identical coordinates in every world; only the per-world grids keep them apart
            origin = np.zeros(3)
        elif p["align_worlds"]:  # tiled, but every world sees the same grid phase (origin on the voxel lattice)
            origin = np.round(origin / voxel) * voxel
        wb = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(wb)
        pts = sc.points + origin
        n = len(pts)
        wb.add_particles(pos=pts.tolist(), vel=np.zeros_like(pts).tolist(), mass=[sc.density * spacing**3] * n,
                         radius=[0.5 * spacing] * n)
        for c, hh, mu in sc.statics:
            wb.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*(np.asarray(c) + origin)), wp.quat_identity()),
                             hx=hh[0], hy=hh[1], hz=hh[2], cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu))
        for verts, tris, mu in sc.meshes:
            mesh = newton.Mesh((verts + origin).astype(np.float32), tris, compute_inertia=False, is_solid=False)
            wb.add_shape_mesh(body=-1, mesh=mesh, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu))
        local_bodies = []
        for kb in sc.bodies:
            p0 = np.asarray(kb.p0) + origin
            body = wb.add_body(is_kinematic=True, xform=wp.transform(wp.vec3(*p0), wp.quat_identity()))
            for c, hh, yaw in kb.boxes:
                rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw))
                wb.add_shape_box(body, xform=wp.transform(wp.vec3(*c), rot), hx=hh[0], hy=hh[1], hz=hh[2],
                                 cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=kb.mu))
            local_bodies.append((body, p0, kb.vel))
        b0, s0 = main.body_count, main.particle_count
        main.add_world(wb)
        ranges.append((s0, main.particle_count))
        origins.append(origin)
        ids = []
        for body, p0, vel in local_bodies:
            drive_ids.append(b0 + body)
            drive_p0.append(p0)
            drive_vel.append(vel)
            ids.append(b0 + body)
        world_bodies.append(ids)
        lo.append(np.broadcast_to(np.asarray(sc.lo) + origin, (n, 3)))
        hi.append(np.broadcast_to(np.asarray(sc.hi) + origin, (n, 3)))
        mats.append((n, sc.material))
    main.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=floor_mu))
    model = main.finalize(device=device)
    _set_materials(model, mats)
    info = dict(origins=np.array(origins), ranges=ranges, world_bodies=world_bodies, spacing=spacing,
                worlds=worlds, lo=wp.array(np.concatenate(lo), dtype=wp.vec3, device=device),
                hi=wp.array(np.concatenate(hi), dtype=wp.vec3, device=device), drive_ids=None)
    if drive_ids:
        info.update(drive_ids=wp.array(drive_ids, dtype=int, device=device),
                    drive_p0=wp.array(np.array(drive_p0), dtype=wp.vec3, device=device),
                    drive_vel=wp.array(np.array(drive_vel), dtype=wp.vec3, device=device))
    return model, info


def _set_materials(model, mats: list[tuple[int, dict[str, float]]]) -> None:
    """Per-particle MPM material arrays from each world's material dict (unset keys keep Newton defaults)."""
    keys = sorted({k for _, m in mats for k in m})
    for key in keys:
        arr = getattr(model.mpm, key).numpy()
        pos = 0
        for n, m in mats:
            if key in m:
                arr[pos:pos + n] = m[key]
            pos += n
        getattr(model.mpm, key).assign(arr)


def _make_solver(model, p: dict, device: str, nworld: int):
    cfg = SolverImplicitMPM.Config()
    for key in _CONFIG_KEYS:
        setattr(cfg, key, p[key])
    cfg.voxel_size = p["voxel"]
    sep = p["separate_worlds"]
    sep = wp.get_device(device).is_cuda if sep == "auto" else bool(sep)
    cfg.separate_worlds = sep and nworld > 1
    if p["stack_worlds"] and nworld > 1 and not cfg.separate_worlds:
        raise ValueError("stack_worlds needs separate_worlds (CUDA only): stacked worlds would share one grid")
    # Rebuildable sparse grid with explicit capacity: an active cell holds >= 1 particle, so the particle count
    # bounds the active cells (graph capture needs a positive capacity; overflow is read in finish()).
    cfg.max_active_cell_count = int(model.particle_count) + 256
    solver = SolverImplicitMPM(model, config=cfg)
    if p["projection_frac"] is not None:  # the dam-break example projects at 0.25 voxel (default 0.01 voxel)
        ncol = int(solver.collider_body_index.shape[0])
        solver.setup_collider(collider_projection_threshold=[p["projection_frac"] * p["voxel"]] * ncol)
    return solver, bool(cfg.separate_worlds), cfg.max_active_cell_count


SAND = dict(friction=SAND_MU, tensile_yield_ratio=0.0, yield_pressure=1.0e12, yield_stress=0.0, viscosity=0.0)
WATER = dict(friction=0.0, tensile_yield_ratio=1.0, viscosity=0.0)


def _wall_thick(voxel: float) -> float:
    return max(1.5 * voxel, 0.01)


def _bbox(lo, hi, thick: float) -> tuple[float, float, float, float]:
    return (lo[0] - 4 * thick, lo[1] - 4 * thick, hi[0] + 4 * thick, hi[1] + 4 * thick)


def _per_condition(b: Built, values: dict[int, dict]) -> dict[str, dict]:
    """Group per-world dicts by condition label: {label: {key: [values over worlds]}}."""
    out: dict[str, dict] = {}
    for w, d in values.items():
        label = str(b.extra["conds"][w % b.extra["ncond"]])
        slot = out.setdefault(label, {})
        for k, v in d.items():
            slot.setdefault(k, []).append(v)
    return out


def _worst(values, key=abs) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return max(vals, key=key) if vals else None


# ---- G02: dam break + hydrostatic settle -------------------------------------------------------------------
class DamBreak(MpmScenario):
    """Martin & Moyce slab (column a wide, 2a high, frictionless channel 0.6 m) and a closed-box settle.

    Front Z = p99.5(x)/a sampled every 4 steps for t <= 0.25 s. Settle: surface height ratio p99.5(z)/p99.5(z0)
    (the prior's h_end/h0) and bottom-layer pressure trace(sigma)/3 (compression positive) vs rho g (h - z).
    """

    id = "G02_dam_break"
    family = "granular"
    title = "Quasi-2D Martin & Moyce dam break and closed-box hydrostatic settle (implicit MPM water)"
    horizon_s = 5.0
    sample_every = 4
    accuracy_unit = "nrmse"
    accuracy_ref = ("front Z(T) NRMSE (range-normalised) vs Martin & Moyce 1952 n^2=2 (table UNVERIFIED); "
                    "metrics: Ritter x_f <= a + 2t sqrt(gH), bottom p = rho g h, settle volume loss")
    A = 0.05715
    CHANNEL = 0.6
    T_FRONT = 0.25  # [s] front sampled up to here (T = 4.6)
    mpm_defaults = dict(max_iterations=50, tolerance=1e-3, warmstart_mode="particles", collider_basis="pic27",
                        projection_frac=0.25)  # example_mpm_water_dam_break.py defaults
    configs = tuple(
        mpm_config(f"mpm_v{v * 1e3:g}mm_cf{cf:g}", v, dt, cap, ppc=2, critical_fraction=cf)
        for v, dt, cap in ((0.01, 2e-3, 1024), (0.005, 1e-3, 256), (0.0025, 5e-4, 16)) for cf in (0.0, 0.5))

    def conditions(self, config: Config) -> list[str]:
        return ["dam_break", "settle"]

    def make_world(self, cond: str, config: Config, p: dict) -> MpmWorld:
        voxel, a = p["voxel"], self.A
        hw, thick = 2.0 * voxel, _wall_thick(voxel)
        x_max = self.CHANNEL if cond == "dam_break" else a
        lo, hi = (0.0, -hw, 0.0), (x_max, hw, OPEN)
        pts = np.clip(jittered_grid((0.0, -hw, 0.0), (a, hw, 2.0 * a), voxel / p["ppc"], seed=2), lo, hi)
        return MpmWorld(pts, lo, hi, dict(WATER), WATER_RHO, _bbox(lo, (x_max, hw), thick),
                        statics=walls_for_box(lo, (x_max, hw, 0.0), thick, 0.3), meta=dict(cond=cond))

    def on_start(self, b: Built, rec: dict) -> None:
        q0 = b.extra["q0"]
        b.extra["z0_p995"] = {w: float(np.percentile(self.local(b, q0, w)[:, 2], 99.5)) for w in self.measured_worlds(b)}
        b.extra["v0"] = {w: self.occupancy(b, self.local(b, q0, w)) for w in self.measured_worlds(b)}
        b.extra["x0_p995"] = {w: float(np.percentile(self.local(b, q0, w)[:, 0], 99.5)) for w in self.measured_worlds(b)}
        b.extra["front"] = {w: [] for w in self.measured_worlds(b)}
        b.extra["level"] = {w: [] for w in self.measured_worlds(b)}

    def measure(self, b: Built, t: float, rec: dict) -> None:
        early = t <= self.T_FRONT + 1e-9
        late = t - b.extra.get("t_level", -1.0) >= 0.1 - 1e-9
        if not (early or late):
            return
        if late:
            b.extra["t_level"] = t
        q = b.state_0.particle_q.numpy()
        for w in self.measured_worlds(b):
            x = self.local(b, q, w)
            if self.cond_of(b, w) == "dam_break" and early:
                shift = float(np.percentile(x[:, 0], 99.5)) - b.extra["x0_p995"][w]
                b.extra["front"][w].append((t, 1.0 + shift / self.A))
            elif self.cond_of(b, w) == "settle" and late:
                b.extra["level"][w].append((t, float(np.percentile(x[:, 2], 99.5)) / b.extra["z0_p995"][w]))

    def finish(self, b: Built, rec: dict) -> None:
        self.common_finish(b, rec)
        q = b.state_0.particle_q.numpy()
        stress = b.state_0.mpm.particle_stress.numpy()
        per: dict[int, dict] = {}
        for w in self.measured_worlds(b):
            per[w] = self._dam(b, w, q) if self.cond_of(b, w) == "dam_break" else self._settle(b, w, q, stress)
        pc = _per_condition(b, per)
        dam, settle = pc.get("dam_break", {}), pc.get("settle", {})
        nr = _worst(dam.get("front_nrmse", []))
        rec["accuracy_error"] = nr
        vol = _worst(settle.get("volume_loss_pct", []))
        prs = _worst(settle.get("pressure_err_pct", []))
        viol = max(dam.get("ritter_violations", [0]))
        checks = dict(front=nr is not None and nr <= 0.1, ritter=viol == 0,
                      pressure=prs is not None and abs(prs) <= 3.0, volume=vol is not None and vol <= 1.0)
        rec["metrics"].update(front_nrmse=nr, ritter_violations=viol, pressure_err_pct=prs, volume_loss_pct=vol,
                              law_checks=checks,
                              law_pass=all(checks.values()) and not b.extra["overflow"]
                              and rec["metrics"]["conditions_covered"], per_condition=pc)

    def _dam(self, b: Built, w: int, q: np.ndarray) -> dict:
        tz = np.array(b.extra["front"][w]) if b.extra["front"][w] else np.zeros((0, 2))
        a = self.A
        T = tz[:, 0] * math.sqrt(2 * G / a)
        use = MM_T <= (T.max() if T.size else 0.0)
        nr = nrmse(np.interp(MM_T[use], T, tz[:, 1]), MM_Z[use]) if use.sum() >= 3 else None
        z_ritter = ritter_front(tz[:, 0], a, 2 * a) / a
        return dict(front_nrmse=nr, front_points=int(use.sum()), ritter_violations=int(np.sum(tz[:, 1] > z_ritter)),
                    front_Z_at_T=[(round(float(t), 3), round(float(z), 3)) for t, z in zip(T[::5], tz[::5, 1])])

    def _settle(self, b: Built, w: int, q: np.ndarray, stress: np.ndarray) -> dict:
        """Volume from the surface height h = p99.5(z) (closed box, so h / h0 is the volume ratio). Bottom pressure:
        mean particle pressure of the lowest voxel layer vs rho g (h - z_layer) with the CURRENT surface height, so
        this check is independent of the volume check; the value vs the initial depth h0 is kept as
        pressure_err_vs_h0_pct. The profile slope above that layer and the floor load (collider impulses in this
        world's footprint vs its weight) separate a bottom boundary layer from global errors."""
        s, e = b.extra["ranges"][w]
        x, voxel, h0 = self.local(b, q, w), b.extra["p"]["voxel"], 2.0 * self.A
        h_now = float(np.percentile(x[:, 2], 99.5))
        ratio = h_now / b.extra["z0_p995"][w]
        p = np.trace(stress[s:e], axis1=1, axis2=2) / 3.0  # Newton's particle_stress is compression positive
        bottom = x[:, 2] < voxel
        p_bottom = float(np.mean(p[bottom]))
        surface = h_now + (h0 - b.extra["z0_p995"][w])  # p99.5 sits below the free surface; offset taken at t = 0
        p_law = WATER_RHO * G * (surface - float(np.mean(x[bottom, 2])))
        p_law_h0 = WATER_RHO * G * h0
        mid = (x[:, 2] > voxel) & (x[:, 2] < 0.8 * h_now)
        slope = float(np.polyfit(x[mid, 2], p[mid], 1)[0]) if mid.sum() > 10 else math.nan
        weight = float(b.model.particle_mass.numpy()[s:e].sum()) * G
        return dict(h_ratio=ratio, volume_loss_pct=100.0 * (1.0 - ratio), p_bottom_pa=p_bottom, p_law_pa=p_law,
                    pressure_err_pct=100.0 * (p_bottom - p_law) / p_law,
                    pressure_err_vs_h0_pct=100.0 * (p_bottom - p_law_h0) / p_law_h0,
                    bottom_layer_particles=int(bottom.sum()),
                    hydro_slope_err_pct=100.0 * (-slope / (WATER_RHO * G) - 1.0),
                    floor_load_err_pct=100.0 * (self.floor_load(b, w) / weight - 1.0),
                    occupancy_volume_loss_pct=100.0 * (1.0 - self.occupancy(b, x) / b.extra["v0"][w]),
                    level_trace=[(round(t, 2), round(r, 4)) for t, r in b.extra["level"][w][::5]])



# ---- G01: column collapse + lifting-cylinder heap ---------------------------------------------------------
COLLAPSE_L0 = {0.5: 0.08, 2.0: 0.05, 6.0: 0.04}  # [m] column length per aspect a = H / L0
HEAP_R, HEAP_H, LIFT_V = 0.04, 0.08, 0.02  # quarter heap: tube radius, fill height [m], lift speed [m/s]
# Drucker-Prager vs Mohr-Coulomb: Newton's cone is sqrt(J2) <= mu p (rheology_solver_kernels.py YieldParamVec with
# warp fem dof_mapper value_to_dof_3d: shear dofs are sigma_ij, normal dof tr(sigma)/sqrt(6)), which matches a
# Mohr-Coulomb angle phi = atan(mu) only in simple shear. The same mu matches other phi in other stress states:
DP_MC_DEG = {"simple_shear": math.degrees(math.atan(SAND_MU)),
             "triaxial_compression": math.degrees(math.asin(3 * math.sqrt(3) * SAND_MU / (6 + math.sqrt(3) * SAND_MU))),
             "triaxial_extension": math.degrees(math.asin(3 * math.sqrt(3) * SAND_MU / (6 - math.sqrt(3) * SAND_MU))),
             "plane_strain": math.degrees(math.atan(3 * SAND_MU / math.sqrt(9 - 12 * SAND_MU**2)))}
SPHERE_COLUMNS = {0.5: (8, 4), 2.0: (5, 10), 6.0: (4, 24)}  # rigid spheres: (columns along x, layers) per aspect


def tube_boxes(radius: float, thick: float, height: float, a0: float, a1: float, nseg: int):
    """Box segments approximating a tube wall over the arc [a0, a1] (body frame, base at z = 0)."""
    da = (a1 - a0) / nseg
    rc = radius + 0.5 * thick
    half_t = 1.15 * rc * math.tan(0.5 * da)
    out = []
    for k in range(nseg):
        th = a0 + (k + 0.5) * da
        out.append(((rc * math.cos(th), rc * math.sin(th), 0.5 * height), (0.5 * thick, half_t, 0.5 * height), th))
    return out


class SandCollapse(MpmScenario):
    """Planar collapse of sand columns (Lube et al. 2005) and a quarter lifting-cylinder heap (repose angle).

    Collapse: a frictionless quasi-2D slab (half-width 2 voxels) with a rough back wall and floor (mu of the sand).
    Runout = shift of the front p99.5(x) over L0. Heap: a smooth quarter tube (symmetry planes x = 0, y = 0) filled
    to HEAP_H and lifted at LIFT_V = 20 mm/s after 0.1 s (eased); the angle is a line fit of the surface on the
    flank. The angle depends on the lift speed (5 mm voxels, L4: 100 / 50 / 20 mm/s gave 16.6 / 19.3 / 21.8 deg;
    bench/notes/granular.md), so the speed is part of the scenario. The configured angle is atan(mu) (simple-shear
    match of the Drucker-Prager cone); metrics.dp_mc_equivalent_deg lists what the same mu means in other states.
    Rigid spheres (MuJoCo Warp, condim 3, no rolling resistance) run the collapse conditions only.
    """

    id = "G01_sand_collapse_repose"
    family = "granular"
    title = "Sand column collapse (a = 0.5, 2, 6) and lifting-cylinder heap (implicit MPM; rigid spheres coarse)"
    horizon_s = 3.0
    floor_mu = SAND_MU
    drive_start = 0.1
    accuracy_unit = "pct"
    accuracy_ref = ("max |runout - law| / law over a in {0.5, 2, 6}; law (L_inf - L0)/L0 = 1.2a (a < 2.3), "
                    "1.9a^(2/3) (Lube et al. PRE 72 041301, 2005); metrics: repose vs atan(mu) = 30 deg")
    mpm_defaults = dict(max_iterations=250, tolerance=1e-4)  # example_mpm_granular.py CLI defaults (S2 collider)
    configs = tuple(
        mpm_config(f"mpm_v{v * 1e3:g}mm_ppc{ppc}", v, dt, cap, ppc=ppc)
        for v, dt, caps in ((0.01, 2e-3, (1024, 256)), (0.005, 1e-3, (256, 16)), (0.0025, 5e-4, (16, 4)))
        for ppc, cap in zip((2, 4), caps)
    ) + (
        # diagnostic: the 10 mm ppc2 config with every world at the same origin (separate grids, CUDA only), to
        # tell float32 position offsets of the xy tiling from scheduling effects when identical worlds disagree
        mpm_config("mpm_v10mm_ppc2_stacked", 0.01, 2e-3, 256, ppc=2, stack_worlds=True),
        mpm_config("mpm_v10mm_ppc2_aligned", 0.01, 2e-3, 256, ppc=2, align_worlds=True),
    ) + tuple(
        Config(name, "mujoco", dt=1e-3, substeps=2, max_nworld=256,
               params=dict(diameter=0.01, density=2500.0, mu=SAND_MU, condim=condim, mu_rolling=roll, solver="cg",
                           cone="elliptic", integrator="implicitfast", iterations=100, ls_iterations=50,
                           contacts_per_sphere=16, rows_per_sphere=8 * condim, gap=0.001))
        # condim 3 = the study's pure spheres; condim 6 with a 0.5 mm rolling lever = its rolling-resistance variant
        for name, condim, roll in (("spheres_d10_mujoco", 3, 1e-4), ("spheres_d10_roll0.5mm_mujoco", 6, 5e-4)))

    def conditions(self, config: Config) -> list[tuple[str, float]]:
        collapse = [("collapse", a) for a in COLLAPSE_L0]
        return collapse if config.solver == "mujoco" else collapse + [("heap", HEAP_H / HEAP_R)]

    def make_world(self, cond, config: Config, p: dict) -> MpmWorld:
        kind, a = cond
        voxel, spacing = p["voxel"], p["voxel"] / p["ppc"]
        thick = _wall_thick(voxel)
        if kind == "collapse":
            l0 = COLLAPSE_L0[a]
            hw, x_max = 2.0 * voxel, l0 * (1.0 + 1.6 * lube_runout(a)) + 0.02
            lo, hi = (0.0, -hw, 0.0), (x_max, hw, OPEN)
            pts = np.clip(jittered_grid((0.0, -hw, 0.0), (l0, hw, a * l0), spacing, seed=1), lo, hi)
            statics = walls_for_box(lo, hi, thick, a * l0 + 0.05, "x", SAND_MU) + walls_for_box(lo, hi, thick,
                                                                                               a * l0 + 0.05, "XyY")
            return MpmWorld(pts, lo, hi, dict(SAND), SAND_RHO, _bbox(lo, hi, thick), statics, meta=dict(l0=l0))
        side = 0.18
        lo, hi = (0.0, 0.0, 0.0), (side, side, OPEN)
        pts = jittered_grid((0.0, 0.0, 0.0), (HEAP_R, HEAP_R, HEAP_H), spacing, seed=1)
        pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < HEAP_R - 0.25 * spacing]
        pts = np.clip(pts, lo, hi)
        tube = KinematicBody(p0=(0.0, 0.0, 0.001), vel=(0.0, 0.0, LIFT_V), mu=0.0,
                             boxes=tube_boxes(HEAP_R, thick, HEAP_H + 0.04, -0.15, 0.5 * math.pi + 0.15, 10))
        return MpmWorld(pts, lo, hi, dict(SAND), SAND_RHO, _bbox(lo, hi, thick),
                        walls_for_box(lo, hi, thick, 0.15), bodies=[tube])

    # -- rigid spheres (MuJoCo Warp) -----------------------------------------------------------------------
    def build(self, nworld: int, device: str, config: Config) -> Built:
        return self._build_spheres(nworld, device, config) if config.solver == "mujoco" else \
            super().build(nworld, device, config)

    def substep(self, b: Built, dt: float) -> None:
        if b.extra.get("kind") == "spheres":
            Scenario.substep(self, b, dt)
        else:
            super().substep(b, dt)

    def _build_spheres(self, nworld: int, device: str, config: Config) -> Built:
        """Columns of rigid spheres in a 3-sphere-wide frictionless slab. MuJoCo Warp needs identical worlds,
        so every world holds the same sphere count; spheres a condition does not use are parked on the floor
        beyond x = 1 m (never reached) and are excluded from the measurement."""
        p = config.params
        d, lanes = p["diameter"], 3
        pitch = 1.02 * d
        nmax = max(nx * nz for nx, nz in SPHERE_COLUMNS.values()) * lanes
        conds = self.conditions(config)
        gap = p["gap"]  # explicit: the builder default rigid_gap is 0.1 m, which floods the broadphase
        cfg = newton.ModelBuilder.ShapeConfig(mu=p["mu"], density=p["density"], gap=gap, mu_rolling=p["mu_rolling"])
        grain = {"mujoco:condim": p["condim"]}
        main = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(main)
        main.rigid_gap = gap
        rng = np.random.default_rng(3)
        for w in range(nworld):
            nx, nz = SPHERE_COLUMNS[conds[w % len(conds)][1]]
            col = [((i + 0.5) * pitch, (j - 1) * pitch, (k + 0.5) * pitch + 1e-4)
                   for i in range(nx) for j in range(lanes) for k in range(nz)]
            park = [(1.0 + (n // lanes) * 1.3 * d, ((n % lanes) - 1) * pitch, 0.5 * pitch)
                    for n in range(nmax - len(col))]
            wb = newton.ModelBuilder()
            newton.solvers.SolverMuJoCo.register_custom_attributes(wb)
            wb.rigid_gap = gap
            for pos in col + park:
                jit = rng.uniform(-0.02, 0.02, 3) * d
                body = wb.add_body(xform=wp.transform(wp.vec3(*(np.asarray(pos) + jit)), wp.quat_identity()))
                wb.add_shape_sphere(body, radius=0.49 * d, cfg=cfg, custom_attributes=grain)
            main.add_world(wb)
        hw, t = 1.5 * pitch + 5e-4, 0.02
        smooth = newton.ModelBuilder.ShapeConfig(mu=0.0, density=0.0, gap=gap)
        rough = newton.ModelBuilder.ShapeConfig(mu=p["mu"], density=0.0, gap=gap, mu_rolling=p["mu_rolling"])
        # MuJoCo mixes pair friction with max(), so mu = 0 on a wall alone still gives the spheres' mu. The higher
        # geom priority makes the wall's condim win: condim 1 = frictionless contact (also avoids MJ_MINMU NaN).
        frictionless = {"mujoco:condim": 1, "mujoco:geom_priority": 1}
        main.add_ground_plane(cfg=rough)
        for sgn in (-1.0, 1.0):
            main.add_shape_box(-1, xform=wp.transform(wp.vec3(1.0, sgn * (hw + t), 0.3), wp.quat_identity()),
                               hx=2.0, hy=t, hz=0.3, cfg=smooth, custom_attributes=frictionless)
        main.add_shape_box(-1, xform=wp.transform(wp.vec3(-t, 0.0, 0.3), wp.quat_identity()), hx=t, hy=hw + 2 * t,
                           hz=0.3, cfg=rough)
        model = main.finalize(device=device)
        ncon = p["contacts_per_sphere"] * nmax
        cpu_ref = bool(p.get("use_mujoco_cpu", False))  # offline reference (bench/scenarios/_granular_ref.py)
        solver = newton.solvers.SolverMuJoCo(
            model, solver=p["solver"], cone=p["cone"], integrator=p["integrator"], iterations=p["iterations"],
            ls_iterations=p["ls_iterations"], nconmax=ncon, njmax=p["rows_per_sphere"] * nmax,
            **(dict(use_mujoco_cpu=True, update_data_interval=0) if cpu_ref else {}))
        s0, s1, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        if cpu_ref:  # float64 MjData is never overwritten (update_data_interval 0): push the initial state once
            solver._update_mjc_data(solver.mj_data, model, s0)
        ex = dict(kind="spheres", conds=conds, ncond=len(conds), nmax=nmax, d=d, pitch=pitch, nconmax=ncon,
                  condim=p["condim"], mu_rolling=p["mu_rolling"],
                  q0=s0.body_q.numpy().copy())
        return Built(model, solver, s0, s1, control, nworld=nworld, extra=ex)

    # -- measurement ---------------------------------------------------------------------------------------
    def on_start(self, b: Built, rec: dict) -> None:
        b.extra["fronts"] = {w: [] for w in self.measured_worlds(b)}
        b.extra["x0"] = {w: self._front(b, b.extra["q0"], w) for w in self.measured_worlds(b)}

    def _front(self, b: Built, q: np.ndarray, w: int) -> float:
        """MPM: p99.5 of x. Spheres: front of the connected deposit (deposit_front), because a few perfect
        spheres roll away for ever and a percentile of ~100 grains is the farthest one."""
        if b.extra.get("kind") == "spheres":
            nx, nz = SPHERE_COLUMNS[self.cond_of(b, w)[1]]
            x = q[w * b.extra["nmax"]: w * b.extra["nmax"] + nx * nz * 3, 0]
            front, detached = deposit_front(x, 2.0 * b.extra["d"])
            b.extra.setdefault("detached", {})[w] = detached
            return front
        x = self.local(b, q, w)[:, 0]
        return float(np.percentile(x, 99.5))

    def measure(self, b: Built, t: float, rec: dict) -> None:
        b.extra["t_end"] = t
        if b.extra.get("kind") == "spheres":  # harness turns "_pen_mm" into penetration_p99_mm
            pen = sphere_penetration_mm(b.solver)
            if pen.size:
                b.extra.setdefault("_pen_mm", []).append(pen)
        q = (b.state_0.body_q if b.extra.get("kind") == "spheres" else b.state_0.particle_q).numpy()
        for w in self.measured_worlds(b):
            if self.cond_of(b, w)[0] == "collapse":
                b.extra["fronts"][w].append((t, self._front(b, q, w)))

    def finish(self, b: Built, rec: dict) -> None:
        spheres = b.extra.get("kind") == "spheres"
        if spheres:
            self._sphere_finish(b, rec)
        else:
            self.common_finish(b, rec)
        q = None if spheres else b.state_0.particle_q.numpy()
        per = {}
        for w in self.measured_worlds(b):
            kind, a = self.cond_of(b, w)
            if kind == "collapse":
                l0 = SPHERE_COLUMNS[a][0] * b.extra["pitch"] if spheres else COLLAPSE_L0[a]
                tr = b.extra["fronts"][w]
                runout = (tr[-1][1] - b.extra["x0"][w]) / l0 if tr else math.nan
                drift = 1e3 * (tr[-1][1] - tr[-2][1]) if len(tr) > 1 else math.nan
                per[w] = dict(runout=runout, law=lube_runout(a), runout_err_pct=100.0 * (runout / lube_runout(a) - 1),
                              front_drift_mm_last_sample=drift)
                if spheres:
                    per[w]["detached_grains"] = b.extra.get("detached", {}).get(w)
                else:  # diagnostic: front where the deposit surface first drops below one voxel (resolution-fair)
                    x, voxel = self.local(b, q, w), b.extra["p"]["voxel"]
                    xc, zs = heap_profile(x[:, 0], x[:, 2], voxel)
                    thin = np.nonzero(zs < voxel)[0]
                    front = float(xc[thin[0]] if thin.size else xc[-1]) if xc.size else math.nan
                    per[w].update(runout_1voxel=(front - l0) / l0,
                                  runout_1voxel_err_pct=100.0 * ((front - l0) / l0 / lube_runout(a) - 1.0),
                                  deposit_profile_mm=[(round(1e3 * u, 1), round(1e3 * v, 1)) for u, v in zip(xc, zs)][::2])
            else:
                x = self.local(b, q, w)
                r, bin_w = np.hypot(x[:, 0], x[:, 1]), max(b.extra["p"]["voxel"], 0.005)
                ang, nbin = heap_angle_deg(r, x[:, 2], bin_w)
                rc, zs = heap_profile(r, x[:, 2], bin_w)
                per[w] = dict(repose_deg=ang, repose_err_deg=ang - math.degrees(math.atan(SAND_MU)), flank_bins=nbin,
                              heap_height_mm=1e3 * float(np.percentile(x[:, 2], 99.5)),
                              tube_lift_mm=1e3 * LIFT_V * drive_offset(b.extra["t_end"], self.drive_start),
                              surface_profile_mm=[(round(1e3 * a_, 1), round(1e3 * b_, 1)) for a_, b_ in zip(rc, zs)])
        pc = _per_condition(b, per)
        errs = [d["runout_err_pct"] for w, d in per.items() if "runout_err_pct" in d]
        rep = [d["repose_err_deg"] for d in per.values() if "repose_err_deg" in d]
        rec["accuracy_error"] = _worst(errs)
        worst_rep = _worst(rep)
        checks = dict(runout=bool(errs) and all(math.isfinite(e) and abs(e) <= 10.0 for e in errs))
        if not spheres:
            checks["repose"] = worst_rep is not None and abs(worst_rep) <= 2.0
        rec["metrics"].update(runout_err_pct=_worst(errs), repose_err_deg=worst_rep,
                              configured_friction_deg=math.degrees(math.atan(SAND_MU)),
                              dp_mc_equivalent_deg=None if spheres else DP_MC_DEG, law_checks=checks,
                              law_pass=all(checks.values()) and not b.extra.get("overflow", False)
                              and rec["metrics"]["conditions_covered"], per_condition=pc)

    def _sphere_finish(self, b: Built, rec: dict) -> None:
        data = getattr(b.solver, "mjw_data", None)
        nacon = getattr(data, "nacon", None)
        nacon = int(nacon.numpy()[0]) if nacon is not None else None
        cap = b.extra["nconmax"] * b.nworld
        rec["metrics"].update(spheres_per_world=b.extra["nmax"], nconmax_per_world=b.extra["nconmax"],
                              contacts=nacon, contact_overflow=int(nacon is not None and nacon >= cap),
                              conditions=[str(c) for c in b.extra["conds"]], conditions_covered=b.nworld >= 3,
                              measured_worlds=len(self.measured_worlds(b)),
                              condim=b.extra["condim"], mu_rolling_m=b.extra["mu_rolling"],
                              note="rigid spheres: collapse only; front = connected deposit (deposit_front, gap 2d); "
                                   "condim 3 has no rolling resistance (study: +40-50 % runout)")
        b.extra["overflow"] = bool(rec["metrics"]["contact_overflow"])



# ---- G03: viscous gravity current + funnel ----------------------------------------------------------------
NU_SWEEP = (0.05, 0.1, 0.2)  # kinematic viscosity [m^2/s] (dynamic = rho nu: 50-200 Pa s)
SPREAD_V, SPREAD_R0 = 1.0e-3, 0.06  # full current volume [m^3] and initial column radius [m] (quarter simulated)
HUPPERT_P995 = 0.9906  # p99.5 of particle radius / r_N for Huppert's profile h ~ (1 - (r/r_N)^2)^(1/3)
FUNNEL = dict(aperture_radius=0.01, top_radius=0.05, height=0.2, z_offset=0.2)  # example_mpm_viscous defaults


class ViscousSpread(MpmScenario):
    """Quarter axisymmetric column of viscous fluid spreading on a no-slip plane (Huppert 1982), one nu per world,
    plus the example's funnel flow (nu = 0.05 m^2/s) as a robustness world.

    Front r_N = p99.5(radius) / 0.9906 (the percentile of Huppert's similarity profile). The exponent comes from
    r = c t^n over the late window t >= FIT_T0 (t0 = 0); the prefactor is mean(r / t^(1/8)) over the same window.
    A virtual-origin fit r = c (t - t0)^n from 0.5 s is kept as a diagnostic (spread_exponent_t0fit, t0_s).
    Volume: surface_volume (volume under the free surface, quarter mirrored), relative to t = 0; the voxel occupancy
    (compaction only) is reported beside it.
    """

    id = "G03_viscous_spread"
    family = "granular"
    title = "Axisymmetric viscous gravity current (Huppert 1982), nu sweep per world, plus funnel flow"
    horizon_s = 5.0
    sample_every = 20
    floor_mu = 10.0  # Coulomb mu >> 1 on the floor collider approximates the no-slip base of the theory
    accuracy_unit = "pct"
    accuracy_ref = ("max |prefactor / 0.894 - 1| over nu worlds; r = 0.894 (g V^3 t / (3 nu))^(1/8) (Huppert, JFM 121, "
                    "1982); metrics: exponent 1/8 +- 0.01, volume loss <= 1 %, funnel robustness")
    FIT_T0 = 2.0  # [s] start of the late-time fit window (r ~ 2 R0, current ~ 4 voxels thick at 5 mm)
    T0_FIT_FROM = 0.5  # [s] start of the virtual-origin diagnostic fit
    mpm_defaults = dict(max_iterations=250, tolerance=1e-6)  # example_mpm_viscous.py defaults (S2 collider)
    configs = (mpm_config("mpm_v5mm_ppc2", 0.005, 2.5e-3, 256, ppc=2),
               mpm_config("mpm_v2.5mm_ppc2", 0.0025, 1.25e-3, 16, ppc=2))

    def conditions(self, config: Config) -> list[tuple[str, float]]:
        return [("spread", nu) for nu in NU_SWEEP] + [("funnel", 0.05)]

    def make_world(self, cond, config: Config, p: dict) -> MpmWorld:
        kind, nu = cond
        voxel, spacing = p["voxel"], p["voxel"] / p["ppc"]
        thick = _wall_thick(voxel)
        mat = dict(friction=0.0, tensile_yield_ratio=1.0, viscosity=WATER_RHO * nu)
        if kind == "spread":
            side, h0 = 0.2, SPREAD_V / (math.pi * SPREAD_R0**2)
            lo, hi = (0.0, 0.0, 0.0), (side, side, OPEN)
            pts = jittered_grid((0.0, 0.0, 0.0), (SPREAD_R0, SPREAD_R0, h0), spacing, seed=4)
            pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < SPREAD_R0]
            return MpmWorld(np.clip(pts, lo, hi), lo, hi, mat, WATER_RHO, _bbox(lo, hi, thick),
                            walls_for_box(lo, hi, thick, 0.15), meta=dict(nu=nu, h0=h0))
        from alloy._vendor.newton.examples.mpm.example_mpm_viscous import Example as ViscousExample

        f = FUNNEL
        verts, tris = ViscousExample.create_funnel_mesh(f["aperture_radius"], f["top_radius"], f["height"],
                                                        f["z_offset"])
        r_top, z0, hgt = f["top_radius"], f["z_offset"], f["height"]
        pts = jittered_grid((-r_top, -r_top, z0), (r_top, r_top, z0 + hgt), spacing, seed=4)
        frac = np.clip((pts[:, 2] - z0) / hgt, 0.0, 1.0)  # cone filter with an inward margin, as the example
        r_max = f["aperture_radius"] + frac * (r_top - f["aperture_radius"]) - voxel
        keep = (np.hypot(pts[:, 0], pts[:, 1]) < r_max) & (pts[:, 2] > z0 + voxel) & (pts[:, 2] < z0 + hgt - voxel)
        side = 0.2
        lo, hi = (-side, -side, 0.0), (side, side, OPEN)
        return MpmWorld(pts[keep], lo, hi, mat, WATER_RHO, _bbox(lo, hi, thick), walls_for_box(lo, hi, thick, 0.1),
                        meshes=[(verts, tris, 0.0)], meta=dict(nu=nu))

    def surface_vol(self, b: Built, x: np.ndarray, bias: float = 0.0) -> tuple[float, float]:
        return surface_volume(x, b.extra["p"]["voxel"], b.extra["spacing"], mirror_xy=True, bias=bias)

    def on_start(self, b: Built, rec: dict) -> None:
        q0 = b.extra["q0"]
        spread = [w for w in self.measured_worlds(b) if self.cond_of(b, w)[0] == "spread"]
        b.extra["v0"] = {w: self.occupancy(b, self.local(b, q0, w)) for w in spread}
        # t = 0 surface excess of the jittered lattice calibrates the max-height bias (surface_volume)
        b.extra["vs_bias"] = {w: self.surface_vol(b, self.local(b, q0, w))[1] for w in spread}
        b.extra["vs0"] = {w: self.surface_vol(b, self.local(b, q0, w), b.extra["vs_bias"][w])[0] for w in spread}
        b.extra["radius"] = {w: [] for w in self.measured_worlds(b)}

    def measure(self, b: Built, t: float, rec: dict) -> None:
        b.extra["t_end"] = t
        q = b.state_0.particle_q.numpy()
        for w in self.measured_worlds(b):
            if self.cond_of(b, w)[0] == "spread":
                x = self.local(b, q, w)
                b.extra["radius"][w].append((t, float(np.percentile(np.hypot(x[:, 0], x[:, 1]), 99.5)) / HUPPERT_P995))

    def finish(self, b: Built, rec: dict) -> None:
        self.common_finish(b, rec)
        q = b.state_0.particle_q.numpy()
        per = {w: (self._spread(b, w, q) if self.cond_of(b, w)[0] == "spread" else self._funnel(b, w, q))
               for w in self.measured_worlds(b)}
        pc = _per_condition(b, per)
        spreads = [d for d in per.values() if "prefactor_err_pct" in d]
        pref = _worst([d["prefactor_err_pct"] for d in spreads])
        expo = _worst([d["spread_exponent"] for d in spreads], key=lambda n: abs(n - 0.125))
        vol = _worst([d["volume_loss_pct"] for d in spreads])
        funnel = [d for d in per.values() if "funnel_ok" in d]
        rec["accuracy_error"] = None if pref is None else abs(pref)
        checks = dict(exponent=expo is not None and abs(expo - 0.125) <= 0.01,
                      prefactor=pref is not None and abs(pref) <= 10.0, volume=vol is not None and vol <= 1.0)
        if funnel:
            checks["funnel"] = all(d["funnel_ok"] for d in funnel)
        rec["metrics"].update(spread_exponent=expo, prefactor_err_pct=pref, volume_loss_pct=vol, law_checks=checks,
                              law_pass=all(checks.values()) and not b.extra["overflow"] and len(checks) == 4
                              and rec["metrics"]["conditions_covered"],
                              fit_window_s=[self.FIT_T0, rec["horizon_s"]], per_condition=pc)

    def _spread(self, b: Built, w: int, q: np.ndarray) -> dict:
        nu = self.cond_of(b, w)[1]
        tr = np.array(b.extra["radius"][w]) if b.extra["radius"][w] else np.zeros((0, 2))
        x = self.local(b, q, w)
        v_end, excess_end = self.surface_vol(b, x, b.extra["vs_bias"][w])
        out = dict(nu=nu, r_end_mm=1e3 * tr[-1, 1] if len(tr) else math.nan,
                   volume_loss_pct=100.0 * (1.0 - v_end / b.extra["vs0"][w]),
                   volume_loss_uncalibrated_pct=100.0 * (1.0 - self.surface_vol(b, x)[0] / self.surface_vol(
                       b, self.local(b, b.extra["q0"], w))[0]),
                   surface_bias_mm=1e3 * b.extra["vs_bias"][w],
                   occupancy_volume_loss_pct=100.0 * (1.0 - self.occupancy(b, x) / b.extra["v0"][w]),
                   surface_volume0_ml=1e6 * b.extra["vs0"][w], nominal_quarter_volume_ml=1e6 * SPREAD_V / 4.0)
        if len(tr):
            out["r_end_err_pct"] = 100.0 * (tr[-1, 1] / float(huppert_radius(tr[-1, 0], SPREAD_V, nu)) - 1.0)
        out["surface_excess_end_mm"] = 1e3 * (excess_end - b.extra["vs_bias"][w])  # < 0: compaction under the surface
        out["radius_trace_mm"] = [(round(float(t), 3), round(1e3 * float(r), 2)) for t, r in tr[::10]]
        win = tr[tr[:, 0] >= self.FIT_T0] if len(tr) else tr
        if len(win) < 5:
            out.update(spread_exponent=None, prefactor_err_pct=None, fit_points=len(win))
            return out
        # Law fit: r = c t^n on the late window (t0 = 0; the similarity solution is the late-time attractor).
        n, _ = fit_power(win[:, 0], win[:, 1])
        pref = float(np.mean(win[:, 1] / win[:, 0] ** 0.125)) / (G * SPREAD_V**3 / (3.0 * nu)) ** 0.125
        out.update(spread_exponent=n, prefactor=pref, prefactor_err_pct=100.0 * (pref / 0.894 - 1.0),
                   fit_points=len(win))
        # Diagnostic: virtual-origin fit r = c (t - t0)^n from t >= T0_FIT_FROM (absorbs the initial slump).
        early = tr[tr[:, 0] >= self.T0_FIT_FROM]
        if len(early) >= 5:
            n0, _, t0 = fit_power_offset(early[:, 0], early[:, 1])
            out.update(spread_exponent_t0fit=n0, t0_s=t0)
        return out

    def _funnel(self, b: Built, w: int, q: np.ndarray) -> dict:
        x, f, voxel = self.local(b, q, w), FUNNEL, b.extra["p"]["voxel"]
        r = np.hypot(x[:, 0], x[:, 1])
        band = (x[:, 2] > f["z_offset"] + voxel) & (x[:, 2] < f["z_offset"] + f["height"] - voxel)
        r_wall = f["aperture_radius"] + (x[:, 2] - f["z_offset"]) / f["height"] * (f["top_radius"] - f["aperture_radius"])
        leaked = int(np.sum(band & (r > r_wall + 0.005 + voxel)))  # outside the 5 mm wall, still at funnel height
        below = int(np.sum(x[:, 2] < -voxel))
        flowed = float(np.mean(x[:, 2] < f["z_offset"]))
        return dict(funnel_flowed_frac=flowed, funnel_leaked=leaked, funnel_below_floor=below,
                    funnel_ok=bool(np.isfinite(x).all() and leaked == 0 and below == 0
                                   and (flowed > 0.0 or b.extra.get("t_end", 0.0) < 1.0)))



# ---- G04: blade digging -----------------------------------------------------------------------------------
DIG_DEPTHS = (0.01, 0.02, 0.04)  # [m]
BLADE_W, BLADE_V = 0.05, 0.01  # blade width [m] (Rankine reference) and drag speed [m/s]
BED_L, BED_H = 0.24, 0.06  # sand bed length and depth [m]


class BladeDigging(MpmScenario):
    """A smooth vertical blade spanning a frictionless plane-strain slab, dragged through a sand bed at 10 mm/s.

    The slab is narrower than the 50 mm blade (half-width max(2 voxels, 10 mm)); in plane strain the force per
    width is the same, so forces are reported per 50 mm (scaled by BLADE_W / slab width). The bed settles for
    0.3 s; the blade speed then ramps over 0.5 s. Force = -(sum of collider impulses on the blade along x) / dt,
    sampled every 20 steps. In plane strain the bulldozed sand cannot spill sideways, so the pile in front of the
    blade grows with travel and the force with it. The law window is therefore scaled with the depth (travel in
    [0.5 d, 1.5 d], DIG_WINDOW): the scene is then self-similar in d, so F ~ d^2 holds even with the pile, and the
    pile adds a depth-independent factor to F / F_Rankine. The second half of the whole travel (a fixed window,
    where the pile is the same size at every depth) is kept as the *_late diagnostics.
    """

    id = "G04_blade_digging"
    family = "granular"
    title = "50 mm kinematic blade dragged through sand at 10 mm/s, depth 10/20/40 mm (implicit MPM)"
    horizon_s = 8.0
    sample_every = 20
    floor_mu = SAND_MU
    drive_start = 0.3
    accuracy_unit = "decades"
    accuracy_ref = ("max |log10(F / F_Rankine)| over depths, F = mean force at travel in [0.5 d, 1.5 d], "
                    "F_R = 1/2 rho g d^2 w tan^2(45 + phi/2), phi = 30 deg (order of magnitude: pass <= 0.5); "
                    "metrics: exponent n in F ~ d^n = 2 +- 0.2, force CV in the window <= 15 %")
    DIG_WINDOW = (0.5, 1.5)  # law window: travel / depth
    mpm_defaults = dict(max_iterations=250, tolerance=1e-4)  # example_mpm_granular.py CLI defaults
    configs = (mpm_config("mpm_v5mm_ppc2", 0.005, 2.5e-3, 256, ppc=2),
               mpm_config("mpm_v2.5mm_ppc2", 0.0025, 1.25e-3, 16, ppc=2))

    def conditions(self, config: Config) -> list[float]:
        return list(DIG_DEPTHS)

    def make_world(self, depth: float, config: Config, p: dict) -> MpmWorld:
        voxel, spacing = p["voxel"], p["voxel"] / p["ppc"]
        thick, hw = _wall_thick(voxel), max(2.0 * voxel, 0.01)
        tb = max(2.0 * voxel, 0.006)
        x_face, z_tip, top = 0.04, BED_H - depth, BED_H + 0.05
        lo, hi = (0.0, -hw, 0.0), (BED_L, hw, OPEN)
        pts = jittered_grid((0.0, -hw, 0.0), (BED_L, hw, BED_H), spacing, seed=5)
        inside = (pts[:, 0] > x_face - tb - 0.5 * spacing) & (pts[:, 0] < x_face + 0.5 * spacing) & \
                 (pts[:, 2] > z_tip - 0.5 * spacing)
        blade = KinematicBody(p0=(x_face - 0.5 * tb, 0.0, 0.5 * (z_tip + top)), vel=(BLADE_V, 0.0, 0.0), mu=0.0,
                              boxes=[((0.0, 0.0, 0.0), (0.5 * tb, hw + thick, 0.5 * (top - z_tip)), 0.0)])
        return MpmWorld(np.clip(pts[~inside], lo, hi), lo, hi, dict(SAND), SAND_RHO, _bbox(lo, hi, thick),
                        walls_for_box(lo, hi, thick, top + 0.02), bodies=[blade], meta=dict(depth=depth, hw=hw))

    def on_start(self, b: Built, rec: dict) -> None:
        b.extra["force"] = {w: [] for w in self.measured_worlds(b)}

    def measure(self, b: Built, t: float, rec: dict) -> None:
        f = self.body_force(b)
        for w in self.measured_worlds(b):
            body = b.extra["world_bodies"][w][0]
            travel = BLADE_V * drive_offset(t, self.drive_start)
            b.extra["force"][w].append((t, travel, float(-f[body, 0]), float(f[body, 2])))

    def finish(self, b: Built, rec: dict) -> None:
        self.common_finish(b, rec)
        per = {w: self._steady(b, w) for w in self.measured_worlds(b)}
        pc = _per_condition(b, per)
        logs = [d["log10_rankine_ratio"] for d in per.values()]
        rec["accuracy_error"] = None if _worst(logs) is None else abs(_worst(logs))
        expo, offset_mm, k_ratio, ndep = _depth_fit(per, "force_n")
        expo_late = _depth_fit(per, "force_late_n")[0]
        cv = _worst([d["force_cv"] for d in per.values()])
        checks = dict(rankine=rec["accuracy_error"] is not None and rec["accuracy_error"] <= 0.5,
                      exponent=expo is not None and ndep == 3 and abs(expo - 2.0) <= 0.2,
                      cv=cv is not None and cv <= 0.15)
        rec["metrics"].update(exponent=expo, force_cv=cv, exponent_late=expo_late,
                              force_cv_late=_worst([d.get("force_late_cv") for d in per.values()]),
                              law_window_travel_over_depth=list(self.DIG_WINDOW),
                              effective_depth_offset_mm=offset_mm, k_over_rankine=k_ratio, law_checks=checks,
                              law_pass=all(checks.values()) and not b.extra["overflow"]
                              and rec["metrics"]["conditions_covered"], per_condition=pc)

    def _steady(self, b: Built, w: int) -> dict:
        depth = self.cond_of(b, w)
        hw = b.extra["worlds"][w].meta["hw"]
        tr = np.array(b.extra["force"][w]) if b.extra["force"][w] else np.zeros((0, 4))
        f_ref = rankine_force(SAND_RHO, depth, BLADE_W, math.degrees(math.atan(SAND_MU)))
        out = dict(depth=depth, rankine_n=f_ref, travel_mm=1e3 * float(tr[-1, 1]) if len(tr) else 0.0)
        tr = tr * np.array([1.0, 1.0, BLADE_W / (2.0 * hw), BLADE_W / (2.0 * hw)])  # per 50 mm of blade
        lo, hi = self.DIG_WINDOW
        law = _force_window(tr[(tr[:, 1] >= lo * depth) & (tr[:, 1] <= hi * depth)] if len(tr) else tr)
        late = _force_window(tr[(tr[:, 1] > 0.0) & (tr[:, 1] >= 0.5 * tr[-1, 1])] if len(tr) else tr)
        mean = law["mean"]
        out.update(force_n=mean, force_cv=law["cv"], force_cv_detrended=law["cv_detrended"],
                   force_slope_n_per_mm=law["slope"], vertical_force_up_n=law["fz"], steady_samples=law["n"],
                   rankine_ratio=mean / f_ref if math.isfinite(mean) else None,
                   log10_rankine_ratio=math.log10(mean / f_ref) if math.isfinite(mean) and mean > 0 else None,
                   force_late_n=late["mean"], force_late_cv=late["cv"], force_late_cv_detrended=late["cv_detrended"],
                   force_late_slope_n_per_mm=late["slope"],
                   rankine_ratio_late=late["mean"] / f_ref if math.isfinite(late["mean"]) else None,
                   force_trace=[(round(1e3 * float(u), 1), round(float(f), 4)) for u, f in tr[::10, 1:3]])
        return out


def _force_window(win: np.ndarray) -> dict:
    """Mean, CV, detrended CV, slope [N/mm] and mean vertical force of force samples (t, travel, fx, fz)."""
    if len(win) < 3:
        return dict(mean=math.nan, cv=None, cv_detrended=None, slope=None, fz=None, n=len(win))
    fx = win[:, 2]
    mean = float(np.mean(fx))
    slope, icpt = np.polyfit(1e3 * win[:, 1], fx, 1)
    resid = fx - (slope * 1e3 * win[:, 1] + icpt)
    return dict(mean=mean, cv=float(np.std(fx) / abs(mean)) if mean else None,
                cv_detrended=float(np.std(resid) / abs(mean)) if mean else None, slope=float(slope),
                fz=float(np.mean(win[:, 3])), n=len(win))


def _depth_fit(per: dict[int, dict], key: str) -> tuple[float | None, float | None, float | None, int]:
    """Exponent n of F ~ d^n over depths (mean over worlds per depth), and the diagnostic fit F = k (d + delta)^2
    (sqrt(F) linear in d): returns (n, delta [mm], k / k_Rankine, number of depths)."""
    by_depth: dict[float, list[float]] = {}
    for d in per.values():
        f = d.get(key)
        if f is not None and math.isfinite(f) and f > 0:
            by_depth.setdefault(d["depth"], []).append(f)
    if len(by_depth) < 2:
        return None, None, None, len(by_depth)
    dep = np.array(sorted(by_depth))
    fbar = np.array([np.mean(by_depth[k]) for k in dep])
    slope, icpt = np.polyfit(dep, np.sqrt(fbar), 1)
    k_rankine = rankine_force(SAND_RHO, 1.0, BLADE_W, math.degrees(math.atan(SAND_MU)))
    return fit_power(dep, fbar)[0], 1e3 * float(icpt / slope), float(slope**2 / k_rankine), len(by_depth)


SCENARIOS = [SandCollapse(), DamBreak(), ViscousSpread(), BladeDigging()]
