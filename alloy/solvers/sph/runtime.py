# SPDX-License-Identifier: Apache-2.0
"""Per-world statistics, the mass ledger, emitter management and estimator kernels of SolverDFSPH."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from alloy._vendor.newton import ParticleFlags

from alloy.solvers.sph.types import SPHReport

from . import emitters as em
from .bodies import FORCE_SCALE, WORK_SCALE
from .config import DFSPHConfig, sph_phase, stable_dt
from .kernels import FIX_SCALE_HOST
from .layout import ParticleLayout, spare_ids
from .patch import PatchEmitters
from .pressure import MODE_DIVERGENCE, Particles, is_fluid, max_err_kernel, scale_kernel

if TYPE_CHECKING:
    from .solver import SolverDFSPH

log = logging.getLogger("alloy.sph")
_ACTIVE = int(ParticleFlags.ACTIVE)


@wp.kernel(enable_backward=False)
def flux_export_kernel(flux_e: wp.array[wp.int64], flux_g: wp.array[wp.int64], z_ref: wp.array[wp.float64],
                       out: wp.array[wp.float64]):
    """W_ext per world [J] = flux_e + flux_g z_ref (emitters.flux_add), then zero the ledger."""
    w = wp.tid()
    out[w] = wp.float64(flux_e[w]) / wp.float64(WORK_SCALE) + wp.float64(flux_g[w]) / wp.float64(FORCE_SCALE) * z_ref[w]
    flux_e[w] = wp.int64(0)
    flux_g[w] = wp.int64(0)


@wp.kernel(enable_backward=False)
def _reset_kernel(world: wp.array[wp.int32], mask: wp.array[wp.bool], kv: wp.array[wp.float32],
                  kd: wp.array[wp.float32], flags0: wp.array[wp.int32], phase0: wp.array[wp.int8],
                  flags: wp.array[wp.int32], phase: wp.array[wp.int8]):
    """Masked worlds: warm starts cleared, particle flags and SPH phase back to their build-time values."""
    i = wp.tid()
    w = world[i]
    if w >= 0 and mask[w]:
        kv[i] = 0.0
        kd[i] = 0.0
        flags[i] = flags0[i]
        phase[i] = phase0[i]


@wp.kernel(enable_backward=False)
def _stable_dt_kernel(vmax: wp.array[wp.float32], cfl: float, spacing: float, dt_min: float, dt_max: float,
                      out: wp.array[wp.float32]):
    w = wp.tid()
    v = vmax[w]
    dt = dt_max
    if v > 0.0:
        dt = wp.clamp(cfl * spacing / v, dt_min, dt_max)
    out[w] = dt


@wp.kernel(enable_backward=False)
def _energy_kernel(p: Particles, gravity: wp.array[wp.vec3], z_ref: wp.array[wp.float64],
                   e: wp.array[wp.float64]):
    """KE + gravity PE about (0, 0, z_ref[world]) per particle [J] (D20)."""
    i = wp.tid()
    if not is_fluid(p, i):
        e[i] = wp.float64(0.0)
        return
    w = p.world[i]
    m = wp.float64(p.mass[i])
    v = p.qd[i]
    x = p.q[i]
    g = gravity[w]
    dz = wp.vec3(x[0], x[1], float(wp.float64(x[2]) - z_ref[w]))
    e[i] = m * (wp.float64(0.5) * wp.float64(wp.dot(v, v)) - wp.float64(wp.dot(g, dz)))


@wp.kernel(enable_backward=False)
def _speed_only_kernel(p: Particles, out: wp.array[wp.float32]):
    i = wp.tid()
    if is_fluid(p, i):
        s = wp.length(p.qd[i])
        if wp.isfinite(s):
            wp.atomic_max(out, p.world[i], s)
        else:
            wp.atomic_max(out, p.world[i], 1.0e30)


class WorldStats:
    """Device counters per world plus the host report."""

    def __init__(self, world_count: int, device: wp.context.Device):
        W = max(world_count, 1)
        self.world_count = world_count
        self.device = device
        z = lambda dt: wp.zeros(W, dtype=dt, device=device)  # noqa: E731
        self.overflow, self.escaped, self.projected = z(wp.int32), z(wp.int32), z(wp.int32)
        self.nonfinite, self.emit_overflow, self.swept = z(wp.int32), z(wp.int32), z(wp.int32)
        self.cavitating = z(wp.int32)  # last density solve, ambient pressure only (ambient.py)
        self.max_speed, self.den_err, self.div_err, self.den_err_max = (z(wp.float32) for _ in range(4))
        self.iters_den, self.iters_div = z(wp.int32), z(wp.int32)
        self.den_tol = z(wp.float32)  # effective density tolerance of the last step (solve_driver.py)
        self.mass_in, self.mass_out, self.mass0, self.mass_act = (z(wp.int64) for _ in range(4))
        self.flux_e, self.flux_g = z(wp.int64), z(wp.int64)  # energy in/out by emission/retirement (emitters.flux_add)
        self.flux_w = wp.zeros(W, dtype=wp.float64, device=device)  # export scratch

    def record_solve(self, divergence: bool, iters: wp.array, err_mean: wp.array) -> None:
        wp.copy(self.iters_div if divergence else self.iters_den, iters)
        wp.copy(self.div_err if divergence else self.den_err, err_mean)

    def record_density_tolerance(self, scale: wp.array, tol: float) -> None:
        wp.launch(scale_kernel, dim=self.world_count, inputs=[scale, float(tol), self.den_tol], device=self.device)

    def record_max_density_error(self, p: Particles, err: wp.array) -> None:
        self.den_err_max.zero_()
        wp.launch(max_err_kernel, dim=err.shape[0], inputs=[p, err, self.den_err_max], device=self.device)

    def speed(self, p: Particles, kernel) -> None:
        self.max_speed.zero_()
        wp.launch(kernel, dim=p.q.shape[0], inputs=[p, self.max_speed, self.nonfinite], device=self.device)

    def _active_mass(self, p: Particles | None) -> np.ndarray:
        self.mass_act.zero_()
        if p is not None:
            wp.launch(em.active_mass_kernel, dim=p.mass.shape[0], inputs=[p, self.mass_act], device=self.device)
        return self.mass_act.numpy()[: self.world_count]

    def init_mass(self, p: Particles | None) -> None:
        self._active_mass(p)
        wp.copy(self.mass0, self.mass_act)

    def ledger(self, p: Particles | None) -> dict[str, np.ndarray]:
        W = self.world_count
        return {"initial": self.mass0.numpy()[:W].copy(), "in": self.mass_in.numpy()[:W].copy(),
                "out": self.mass_out.numpy()[:W].copy(), "active": self._active_mass(p).copy(),
                "scale": np.float64(FIX_SCALE_HOST)}

    def report(self, p: Particles | None, max_iterations: int) -> SPHReport:
        W = self.world_count
        kg = lambda a: a.numpy()[:W].astype(np.float64) / FIX_SCALE_HOST  # noqa: E731
        f64 = lambda a: a.numpy()[:W].astype(np.float64)  # noqa: E731
        it_den, it_div = self.iters_den.numpy()[:W].copy(), self.iters_div.numpy()[:W].copy()
        return SPHReport(
            density_err=f64(self.den_err), divergence_err=f64(self.div_err), iters_density=it_den,
            iters_divergence=it_div, mass_active=self._active_mass(p).astype(np.float64) / FIX_SCALE_HOST,
            mass_in=kg(self.mass_in), mass_out=kg(self.mass_out), escaped=self.escaped.numpy()[:W].copy(),
            neighbor_overflow=self.overflow.numpy()[:W].copy(), max_speed=f64(self.max_speed),
            unconverged=(it_den >= max_iterations) | (it_div >= max_iterations), density_tol=f64(self.den_tol),
        )

    def diagnostics(self) -> dict[str, np.ndarray]:
        """Extra per-world counters beyond SPHReport (max density error, projections, non-finite, emitter overflow,
        fast-particle sweeps that stopped a particle at a wall; with an ambient pressure, particles of the last
        density solve at the cavitation floor, gauge -p_a: ambient.py)."""
        W = self.world_count
        return {"density_err_max": self.den_err_max.numpy()[:W].astype(np.float64),
                "projected": self.projected.numpy()[:W].copy(), "nonfinite": self.nonfinite.numpy()[:W].copy(),
                "emitter_overflow": self.emit_overflow.numpy()[:W].copy(),
                "swept": self.swept.numpy()[:W].copy(), "cavitating": self.cavitating.numpy()[:W].copy()}

    def cfl_dt(self, cfg: DFSPHConfig) -> float:
        v = float(np.max(self.max_speed.numpy()[: self.world_count])) if self.world_count else 0.0
        return stable_dt(cfg, v)

    def capacity(self) -> dict[str, int]:
        return {"neighbor_overflow": int(self.overflow.numpy().sum()),
                "emitter_overflow": int(self.emit_overflow.numpy().sum())}

    def state_arrays(self) -> dict[str, wp.array]:
        return {"sph.mass_in": self.mass_in, "sph.mass_out": self.mass_out, "sph.escaped": self.escaped}

    def reset(self, model, kv: wp.array, kd: wp.array, world_mask: wp.array | None,
              initial: tuple[wp.array, wp.array, wp.array] | None = None) -> None:
        """``initial`` = (flags0, phase0, phase): restore the build-time flags/phase of the masked worlds."""
        W = self.world_count
        if world_mask is None:
            mask = np.ones(W, dtype=bool)
        else:
            mask = world_mask.numpy()[:W].astype(bool)
        if model.particle_count:
            wm = wp.array(mask if W else np.zeros(1, bool), dtype=wp.bool, device=self.device)
            flags0, phase0, phase = initial
            wp.launch(_reset_kernel, dim=model.particle_count,
                      inputs=[model.particle_world, wm, kv, kd, flags0, phase0, model.particle_flags, phase],
                      device=self.device)
        for arr in (self.escaped, self.projected, self.overflow, self.nonfinite, self.emit_overflow, self.swept,
                    self.mass_in, self.mass_out, self.flux_e, self.flux_g):  # the ledger restarts with the restored particles (mass0 unchanged)
            a = arr.numpy()
            a[:W][mask] = 0
            arr.assign(a)


class EmitterManager:
    """Host bookkeeping of emitters and sinks; device work in ``step``."""

    def __init__(self, solver: SolverDFSPH, layout: ParticleLayout, stats: WorldStats):
        self.solver = solver
        self.layout = layout
        self.stats = stats
        self.table = em.EmitterTable()
        self.dev: em.EmitterDevice | None = None
        self._saved: dict[str, wp.array] = {}
        b = solver.config.world_bounds
        self.bounds = None if b is None else (wp.vec3(*b[0]), wp.vec3(*b[1]))
        self.patches = PatchEmitters(solver, self._unowned_spares)

    @property
    def active(self) -> bool:
        return self.dev is not None or self.bounds is not None or bool(self.patches.groups)

    def _spares(self, w: int) -> np.ndarray:
        """Inactive spares of world w not reserved by a patch emitter (ascending ids)."""
        m = self.solver.model
        ids = spare_ids(self.layout.phase.numpy(), m.particle_world.numpy(), m.particle_flags.numpy(), w, _ACTIVE)
        return np.setdiff1d(ids, self.patches.reserved(w)).astype(np.int32)

    def _unowned_spares(self, w: int) -> np.ndarray:
        """``_spares`` minus the pools emitters own (a pour's reservoir), for patch emitters to reserve: a patch added
        after a pour took 10 of its ids, which the pour would then re-place (AUDIT-STACK, 2026-11-11)."""
        t = self.table
        owned = [t.pool[e] for e, w2 in enumerate(t.world) if w2 == w and t.own[e]]
        return np.setdiff1d(self._spares(w), np.concatenate(owned)).astype(np.int32) if owned else self._spares(w)

    def add_patch_emitter(self, world_mask, body, local_points, velocity) -> int:
        """Register a patch group (patch.py); box emitters of its worlds re-split the remaining spares, so they
        must not have emitted yet."""
        t = self.table
        worlds = set(PatchEmitters.worlds_of(world_mask, local_points)) if len(local_points) == len(world_mask) else set()
        box = [e for e, w in enumerate(t.world) if w in worlds]
        if box and self.dev is not None and np.any(self.dev.cursor.numpy()[box] > 0):
            raise ValueError("add_patch_emitter: a box emitter of these worlds has emitted; add patch emitters first")
        g = self.patches.add(world_mask, body, local_points, velocity)
        for w in sorted({t.world[e] for e in box}):
            self._split_pools(w)
        if box:
            self._upload()
        return g

    def add_emitter(self, world: int, region: wp.transform, half_extents: wp.vec3, rate: float,
                    velocity: wp.vec3, pool: np.ndarray | None = None) -> int:
        if not (0 <= world < self.layout.world_count):
            raise ValueError(f"add_emitter: world {world} out of range")
        if rate <= 0.0:
            raise ValueError("add_emitter: rate must be > 0 [m^3/s]")
        cfg = self.solver.config
        X = wp.transform(*region) if not isinstance(region, wp.transform) else region
        v_local = np.array(wp.quat_rotate_inv(wp.transform_get_rotation(X), wp.vec3(*velocity)))
        offs = em.layer_offsets(np.asarray(half_extents, dtype=np.float64), v_local, cfg.spacing)
        rho0 = float(self.layout.rest_density.numpy()[self._first_sph(world)])
        mass = float(self.solver.model.particle_mass.numpy()[self._first_sph(world)])
        interval = len(offs) * (mass / rho0) / rate
        self._check_rate(offs, v_local, rate, cfg.spacing)
        self._check_inside_bounds(np.array(wp.transform_get_translation(X), dtype=np.float64))
        t = self.table
        t.world.append(world)
        t.xform.append(tuple(float(c) for c in X))
        t.velocity.append(tuple(float(c) for c in velocity))
        t.interval.append(interval)
        t.offsets.append(offs)
        t.own.append(pool is not None)            # a pour's own reservoir (its spares of this world, ascending)
        t.pool.append(np.zeros(0, np.int32) if pool is None else np.intersect1d(pool, self._spares(world)))
        self._split_pools(world)
        self._upload()
        log.info("SPH emitter %d: world %d, %d particles per layer every %.4g s", len(t.world) - 1, world,
                 len(offs), interval)
        return len(t.world) - 1

    def _first_sph(self, world: int) -> int:
        ws = self.layout.world_start
        ph = self.layout.phase.numpy()[ws[world]:ws[world + 1]]
        idx = np.nonzero(sph_phase(ph))[0]
        if len(idx) == 0:
            raise ValueError(f"world {world} has no SPH particles or spares")
        return int(ws[world] + idx[0])

    @staticmethod
    def _check_rate(offs: np.ndarray, v_local: np.ndarray, rate: float, spacing: float) -> None:
        area = len(offs) * spacing * spacing
        speed = float(np.linalg.norm(v_local))
        if speed > 0.0 and not 0.8 <= rate / (area * speed) <= 1.25:
            log.warning("SPH emitter rate %.3g m^3/s vs area*|v| %.3g m^3/s: layers will over/under-pack", rate,
                        area * speed)

    def _check_inside_bounds(self, centre: np.ndarray) -> None:
        """An emitter outside the escape box retires its liquid as it is emitted (counted in ``escaped``): warn."""
        if self.bounds is None:
            return
        lo, hi = np.array(self.bounds[0], dtype=np.float64), np.array(self.bounds[1], dtype=np.float64)
        if np.any(centre < lo) or np.any(centre > hi):
            log.warning("SPH add_emitter: region centre %s is outside world_bounds %s; its liquid retires as it is "
                        "emitted (report().escaped)", centre.tolist(), (lo.tolist(), hi.tolist()))

    def _split_pools(self, world: int) -> None:
        """Share the world's unused spares among its emitters that were given no pool (``em.split_pools``)."""
        cursors = self.dev.cursor.numpy() if self.dev is not None else np.zeros(len(self.table.world), np.int32)
        em.split_pools(self.table, world, self._spares(world), cursors)

    def add_sink(self, world: int, region: wp.transform, half_extents: wp.vec3) -> int:
        if not (0 <= world < self.layout.world_count):
            raise ValueError(f"add_sink: world {world} out of range")
        X = wp.transform(*region) if not isinstance(region, wp.transform) else region
        self.table.sink_world.append(world)
        self.table.sink_xform.append(tuple(float(c) for c in X))
        self.table.sink_half.append(tuple(float(c) for c in half_extents))
        self._upload()
        return len(self.table.sink_world) - 1

    def _upload(self) -> None:
        new = em.EmitterDevice(self.table, self.solver.device)
        new.carry_from(self.dev)
        self.dev = new
        self._saved.clear()

    def step(self, p: Particles, dt: float, bodies: tuple[wp.array, wp.array, wp.array] | None = None) -> None:
        """Sinks and escape box, box emitters, then patch emitters (``bodies`` = body_q, body_qd, body_com)."""
        d = self.dev
        if d is not None and (d.n_sinks or self.bounds is not None) or (d is None and self.bounds is not None):
            self._retire(p)
        if d is not None and d.n_emitters:
            self._emit(p, dt)
        if self.patches.groups and bodies is not None:
            self.patches.step(p, *bodies, self.stats.mass_in, self._flux())

    def _emit(self, p: Particles, dt: float) -> None:
        d, s = self.dev, self.stats
        dev = self.solver.device
        wp.launch(em.emit_clock_kernel, dim=d.n_emitters, inputs=[d.interval, d.t_acc, dt, d.enabled, d.emit_now],
                  device=dev)
        if d.n_offsets:
            wp.launch(em.emit_kernel, dim=d.n_offsets, device=dev,
                      inputs=[d.off_emitter, d.off_rank, d.off_local, d.emit_now, d.cursor, d.pool_begin,
                              d.pool_size, d.pool_ids, d.xform, d.velocity, d.world, p, s.mass_in, s.emit_overflow,
                              *self._flux()])
        wp.launch(em.advance_cursor_kernel, dim=d.n_emitters, inputs=[d.emit_now, d.n_layer, d.pool_size, d.cursor],
                  device=dev)

    def _retire(self, p: Particles) -> None:
        d, s = self.dev, self.stats
        if d is None:
            d = em.EmitterDevice(self.table, self.solver.device)
            self.dev = d
        lo, hi = self.bounds if self.bounds is not None else (wp.vec3(0.0), wp.vec3(0.0))
        wp.launch(em.retire_kernel, dim=p.q.shape[0], device=self.solver.device,
                  inputs=[p, d.n_sinks, d.s_world, d.s_xform, d.s_half, int(self.bounds is not None), lo, hi,
                          s.mass_out, s.escaped, *self._flux()])

    def _flux(self) -> tuple[wp.array, wp.array, wp.array]:
        return self.solver.model.gravity, self.stats.flux_e, self.stats.flux_g

    def export_flux_energy(self, health) -> None:
        """Adds the energy emitted minus the energy retired since the last call to ``health``'s W_ext
        (``HealthMonitor.add_external_work``, at its z_ref) and zeroes the ledger. Device only (capture-safe); call
        once per health check, after the step(s). The monitor's worlds must be this solver's worlds."""
        s = self.stats
        z_ref = health.state_arrays()["health.energy_z_ref"]
        wp.launch(flux_export_kernel, dim=self.layout.world_count, device=self.solver.device,
                  inputs=[s.flux_e, s.flux_g, z_ref, s.flux_w])
        health.add_external_work(s.flux_w)

    def _snap_arrays(self) -> dict[str, wp.array]:
        m = self.solver.model
        arrays = {"flags": m.particle_flags, "phase": self.layout.phase, "mass_in": self.stats.mass_in,
                  "mass_out": self.stats.mass_out, "escaped": self.stats.escaped}
        if self.dev is not None and self.dev.n_emitters:
            arrays.update({"t_acc": self.dev.t_acc, "cursor": self.dev.cursor})
        arrays.update(self.patches.arrays())
        return arrays

    def save(self) -> None:
        """Copy everything an emitter, sink or escape event mutates (coupling restart support)."""
        if not self.active:
            return
        for k, a in self._snap_arrays().items():
            if k not in self._saved:
                self._saved[k] = wp.empty_like(a)
            wp.copy(self._saved[k], a)

    def restore(self) -> None:
        if not self.active or not self._saved:
            return
        for k, a in self._snap_arrays().items():
            if k in self._saved:
                wp.copy(a, self._saved[k])

    def state_arrays(self) -> dict[str, wp.array]:
        out = self.patches.arrays()
        if self.dev is not None and self.dev.n_emitters:
            out.update({"sph.emit_t": self.dev.t_acc, "sph.emit_cursor": self.dev.cursor})
        return out

    def state_worlds(self) -> dict[str, wp.array]:
        out = self.patches.worlds()
        if self.dev is not None and self.dev.n_emitters:
            out.update({"sph.emit_t": self.dev.world, "sph.emit_cursor": self.dev.world})
        return out

    def reset(self, world_mask: wp.array | None) -> None:
        """Rewind the box emitters (``emitters.rewind``) and re-arm the patch emitters of the masked worlds (all when
        None)."""
        em.rewind(self.dev, world_mask, self.layout.world_count)
        if not self.patches.groups:
            return
        W = self.layout.world_count
        mask = np.ones(W, bool) if world_mask is None else world_mask.numpy()[:W].astype(bool)
        self.patches.reset(mask)


def stable_dt_per_world(solver: SolverDFSPH, p: Particles, out: wp.array) -> None:
    """out[w] = clamp(speed_margin·cfl·2r/|v|max, dt_min, dt_max): margin while fast, dt_max at rest (D6; COUPLE-9)."""
    cfg = solver.config
    W = solver.layout.world_count
    tmp = solver._tmp_w
    tmp.zero_()
    if solver.layout.n:
        wp.launch(_speed_only_kernel, dim=solver.layout.n, inputs=[p, tmp], device=solver.device)
    wp.launch(_stable_dt_kernel, dim=W, inputs=[tmp, cfg.speed_margin * cfg.cfl, cfg.spacing, cfg.dt_min,
                                               cfg.dt_max, out],
              device=solver.device)


def energy_per_world(solver: SolverDFSPH, p: Particles, z_ref: wp.array, out: wp.array) -> None:
    """KE + gravity PE of the liquid per world [J] (float64, deterministic order)."""
    n = solver.layout.n
    e = solver._tmp_e
    if n:
        wp.launch(_energy_kernel, dim=n, inputs=[p, solver.model.gravity, z_ref, e], device=solver.device)
    solver._reducer.sum(e, out)


def hydrostatic_guess(solver: SolverDFSPH, state) -> np.ndarray:
    """Host: k_i = |g| (z_top - z_i + r) per (world, horizontal 2r bin) column of active liquid, measured along
    gravity; the initial guess that ``SolverDFSPH.prime`` refines (k = p / rho0 [m^2/s^2])."""
    n = solver.layout.n
    out = np.zeros(max(n, 1), dtype=np.float32)
    if n == 0:
        return out
    m = solver.model
    q = state.particle_q.numpy().astype(np.float64)
    act = solver.layout.phase.numpy() >= 1
    act &= (m.particle_flags.numpy() & _ACTIVE) != 0
    world = m.particle_world.numpy()
    grav = m.gravity.numpy().astype(np.float64)
    sp = solver.config.spacing
    for w in np.unique(world[act]):
        g = grav[w]
        gn = float(np.linalg.norm(g))
        if gn == 0.0:
            continue
        up = -g / gn
        sel = np.nonzero(act & (world == w))[0]
        z = q[sel] @ up
        e1 = np.cross(up, [1.0, 0.0, 0.0] if abs(up[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        key = np.stack([np.floor(q[sel] @ e1 / sp), np.floor(q[sel] @ e2 / sp)], 1).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        top = np.full(inv.max() + 1, -np.inf)
        np.maximum.at(top, inv, z)
        out[sel] = (gn * np.maximum(top[inv] - z + 0.5 * sp, 0.0)).astype(np.float32)
    return out


def prime_hydrostatic(solver: SolverDFSPH, state, dt: float, tol: float = 1.0e-6,
                  max_iterations: int = 2000) -> int:
    """Host helper: give a fluid at rest its hydrostatic pressure before the first step.

    Solves the divergence constraint (no compression rate anywhere) for the velocity one gravity step would add,
    which stops the whole column (a one-step density solve would only stop the layers near the floor), and stores
    that pressure as the density warm start; velocities are left unchanged. With an ambient pressure and seal walls
    (ambient.py) the solve is two-sided on sealed liquid off its free surface, down to -p_a, so a column hanging
    under its piston starts at its hydrostatic tension. Returns the iterations used."""
    if solver.layout.n == 0:
        return 0
    cfg = solver.config
    saved = (cfg.max_divergence_error, cfg.max_iterations, cfg.warm_start_divergence, cfg.min_neighbors_divergence,
             cfg.max_error_factor, cfg.divergence_two_sided, cfg.warm_start_density)
    qd = wp.clone(state.particle_qd)
    p = solver._particles(state)
    solver._prepare_density(state, p)
    solver._kv.assign(hydrostatic_guess(solver, state) * float(dt))
    amb = solver._ambient
    try:
        cfg.max_divergence_error, cfg.max_iterations = tol, max_iterations
        cfg.warm_start_divergence, cfg.min_neighbors_divergence, cfg.max_error_factor = 1.0, 0.0, 1.0e9
        if amb is not None and amb.any_seal:  # sealed liquid may start in tension (ambient.py)
            amb.prime_bounds(p, solver._slot_struct(), solver._h, solver._kind, dt, solver._kd, solver.layout.n)
            cfg.divergence_two_sided, cfg.warm_start_density = True, 1.0
        solver._nonpressure(p, state, dt, *solver._body_state(state), predict=False)  # no pressure guess yet
        solver._solve(p, solver._slot_struct(), MODE_DIVERGENCE, dt)
    finally:
        (cfg.max_divergence_error, cfg.max_iterations, cfg.warm_start_divergence, cfg.min_neighbors_divergence,
         cfg.max_error_factor, cfg.divergence_two_sided, cfg.warm_start_density) = saved
        wp.copy(state.particle_qd, qd)
        solver._slot_impulse.zero_()
    wp.launch(scale_kernel, dim=solver.layout.n, inputs=[solver._kv, float(dt), solver._kd], device=solver.device)
    solver._kv.zero_()
    return int(solver.stats.iters_div.numpy()[: solver.layout.world_count].max())
