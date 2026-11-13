# SPDX-License-Identifier: Apache-2.0
"""SolverDFSPH: divergence-free SPH liquid as a Newton solver plug-in (SPEC §3.8, WP9).

One ``step(dt)`` is one DFSPH step (SPlisHSPlasH@dba6b26:SPlisHSPlasH/DFSPH/TimeStepDFSPH.cpp:117-250 order):
neighbours -> boundary slots -> density and factor -> divergence solve -> gravity, external forces, XSPH, surface
tension (csf.py: CSF with a contact angle; tension.py: Akinci, qualitative), viscosity -> density solve -> body
wrenches and embedded body integration -> advection -> projection -> sinks, escape box, box and patch emitters
(patch.py). Every reduction is per world and float-atomic free (NATIVE determinism, D9); iteration loops run under
``wp.capture_while`` (graph-capturable on CUDA, a host loop on CPU). The Scheduler supplies substeps through
``alloy_stable_dt``; ``choose_dt`` snaps the CFL step onto the D4 grid once per period.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from typing import ClassVar

import numpy as np
import warp as wp
from alloy._vendor.newton import Contacts, Control, Model, ModelFlags, ParticleFlags, State
from alloy._vendor.newton.solvers import SolverBase

from alloy.core.newton_compat import CouplingInterface
from alloy.core.protocols import DetCapability
from alloy.solvers.sph.types import SPHReport

from . import bodies as bd
from . import emitters as em
from . import pressure as pr
from .ambient import make_ambient
from .boundary import NB, BoundarySlots, compute_slots
from .config import (DFSPHConfig, full_neighbor_count, register_custom_attributes, resolve_viscosity, snap_down,
                     sph_phase)
from .integrate import SWEEP, advect_kernel, nonpressure_kernel, project_kernel, speed_kernel
from .krylov import ProjectedCG, SolveControl
from .layout import body_arrays, particle_layout, shape_table
from .neighbors import NeighborLists
from .reduce import WorldReducer
from .kernels import laplacian_lattice_factor, lattice_diagonal
from .viscosity import LAPLACE_D, ImplicitViscosity, guess_shift_kernel, pressure_guess_kernel, wall_shift_kernel
from .wall_viscosity import wall_coefficient_scale, wall_slots
from .csf import XSPH_MIN, make_tension
from .runtime import EmitterManager, WorldStats, energy_per_world, prime_hydrostatic, stable_dt_per_world
from .solve_driver import PressureSolves
from .coupling_hooks import SPHCouplingMixin
from .particle_walls import ParticleWalls
from .soft_walls import SoftWalls
from .ports import SPHPort
from .view_flags import ViewFlagMirror

log = logging.getLogger("alloy.sph")


class SolverDFSPH(PressureSolves, SPHCouplingMixin, SolverBase, CouplingInterface):
    """Divergence-free SPH (Bender & Koschier 2017) with SDF tube-map boundaries and embedded proxy bodies."""

    plugin_name: ClassVar[str] = "sph"
    routes: ClassVar[frozenset[str]] = frozenset({"liquid_sph"})
    determinism: ClassVar[DetCapability] = DetCapability.NATIVE
    # The coupled clock's safety on alloy_stable_dt: 1 (with ClockPolicy.own_cfl, clocks.entry_safety; S4), since that
    # estimate carries its own margin: config.speed_margin x the CFL step while the liquid moves fast (COUPLE-9)
    alloy_clock_safety: ClassVar[float] = 1.0
    # ... but not beside MPM (sand, paste): a slab floating on SPH water sat 2 points deeper at the CFL step itself
    # (tests/integration/test_sph_mpm_laws.py L1: draft 1.108 vs 1.088 of rho_s/rho_f H, bound 10 %; WP12 11-09)
    alloy_clock_safety_off_beside: ClassVar[tuple[str, ...]] = ("mpm",)
    Config = DFSPHConfig
    register_custom_attributes = staticmethod(register_custom_attributes)

    def __init__(self, model: Model, config: DFSPHConfig | None = None, *, capacity_per_world: int | None = None,
                 soft_particles=None, soft_half_thickness: float = 0.0, soft_inertia: bool = True):
        super().__init__(model)
        self.config = cfg = config or DFSPHConfig()
        cfg.validate()

        self.layout = particle_layout(model, cfg)
        self.leaks = None  # LeakMonitor when enabled (LIQ-8)
        self._flags0 = wp.clone(model.particle_flags) if self.layout.n else wp.zeros(1, dtype=wp.int32, device=model.device)
        self._phase0 = wp.clone(self.layout.phase) if self.layout.n else wp.zeros(1, dtype=wp.int8, device=model.device)
        self._check_capacity(capacity_per_world)
        self._tension = make_tension(model, self.layout, cfg, self.device)
        if self._tension is not None and self._tension.dt_limit < cfg.dt_max:  # capillary step (tension.py)
            self.config = cfg = dataclasses.replace(cfg, dt_max=max(self._tension.dt_limit, cfg.dt_min))
        self.viscosity = resolve_viscosity(cfg, self.layout.viscosity_max)
        self._xsph_c = max(cfg.xsph_c, XSPH_MIN) if cfg.surface_tension == "csf" else cfg.xsph_c  # csf.py
        self.pressure_solver = cfg.pressure_solver
        if self.pressure_solver == "cg" and not self.layout.uniform:
            log.warning("SolverDFSPH: pressure_solver 'cg' needs one particle mass and rest density; using 'jacobi'")
            self.pressure_solver = "jacobi"
        self._h, self._kind = cfg.h, cfg.kernel_id
        self._n_full = full_neighbor_count(cfg)
        self._tot_min = cfg.diagonal_floor * lattice_diagonal(cfg.spacing, cfg.h, cfg.kernel)  # pressure.py
        self._alloc()
        self.shapes = shape_table(model, cfg, self.device)
        self.soft = SoftWalls(model, self.shapes.arrays.reach, self.device, soft_particles, soft_half_thickness)
        self.shapes.arrays.soft, self._soft_back = self.soft.arrays, 0.0  # soft_walls.py, coupling_hooks.py
        self.inertia = self._make_inertia(soft_inertia)  # soft_inertia.py (SPH-VBD P2)
        self.pw = self._make_particle_walls()  # another entry's particles (particle_walls.py, SCR-WP12-MIX)
        self._ambient = make_ambient(model, cfg, self.layout.n, self.device, self.layout.world_start)  # SCR-SC10-1
        self.bodies = body_arrays(model, self.device)
        self._alloc_bodies()
        self.stats = WorldStats(self.layout.world_count, self.device)
        self.emitters = EmitterManager(self, self.layout, self.stats)
        self._mirror_flags = ViewFlagMirror(model, self.layout.phase, self.layout.n, self.device)  # SCR-OSS22-6
        self.stats.init_mass(self._particles_model())
        self.port = SPHPort(self)  # absorb / emit / retire between steps (ports.py)
        log.info("SolverDFSPH: %d SPH particles in %d worlds, r=%g m, h=%g m, kernel=%s, viscosity=%s, "
                 "%d boundary shapes", self.layout.sph_count, self.layout.world_count, cfg.particle_radius,
                 self._h, cfg.kernel, self.viscosity, self.shapes.count)

    # ------------------------------------------------------------------------------------------------- setup
    def _check_capacity(self, capacity_per_world: int | None) -> None:
        if capacity_per_world is None:
            return
        from alloy.core.errors import CapacityError

        ws = self.layout.world_start
        phase = self.layout.phase.numpy() if self.layout.n else np.zeros(0)
        for w in range(self.layout.world_count):
            have = int(np.count_nonzero(sph_phase(phase[ws[w]:ws[w + 1]])))
            if have < capacity_per_world:
                raise CapacityError(f"SolverDFSPH: world {w} has {have} SPH particle slots, "
                                    f"capacity_per_world={capacity_per_world}")

    def _alloc(self) -> None:
        n, d, W = self.layout.n, self.device, self.layout.world_count
        cfg = self.config
        self._nbrs = NeighborLists(n, cfg.neighbor_capacity, self._h, d, cfg.grid_dim,
                                   world_start=self.layout.world_start,  # world-local rows (L4b)
                                   skin=(cfg.reneighbor_skin or 0.0) * self._h,  # Verlet-skin reuse (L4a)
                                   spacing=cfg.spacing)
        self._slots = BoundarySlots.allocate(n, d)
        self._slot_impulse = wp.zeros((max(n, 1), NB), dtype=wp.vec3, device=d)
        self._w = pr.SolveArrays.allocate(n, d)
        f32 = lambda m: wp.zeros(max(m, 1), dtype=wp.float32, device=d)  # noqa: E731
        self._kv, self._kd = f32(n), f32(n)  # warm starts, stored dt-free (k dt, k dt^2)
        self._kv_saved, self._kd_saved = f32(n), f32(n)
        self._vtmp = wp.zeros(max(n, 1), dtype=wp.vec3, device=d)
        self._lower = f32(n)  # per-particle lower bound of k in the current solve
        self._reducer = WorldReducer(self.layout.world_start, d)
        self._n_active = wp.zeros(max(W, 1), dtype=wp.float64, device=d)
        self._err_sum = wp.zeros(max(W, 1), dtype=wp.float64, device=d)
        self._converged = wp.zeros(max(W, 1), dtype=wp.int32, device=d)
        self._iters = wp.zeros(max(W, 1), dtype=wp.int32, device=d)
        self._err_mean = wp.zeros(max(W, 1), dtype=wp.float32, device=d)
        self._err_max = wp.zeros(max(W, 1), dtype=wp.float32, device=d)
        self._n_unconv = wp.zeros(1, dtype=wp.int32, device=d)
        self._tmp_w = wp.zeros(max(W, 1), dtype=wp.float32, device=d)  # estimator scratch (capture-safe)
        self._tmp_e = wp.zeros(max(n, 1), dtype=wp.float64, device=d)
        self._ctl = SolveControl(self._converged, self._iters, self._err_mean, self._err_max, self._n_unconv,
                                 self._n_active)
        self._alloc_tolerance()
        self._cg = (ProjectedCG(n, W, d, self._reducer, self._w, self._lower, self._ctl)
                    if self.pressure_solver == "cg" else None)
        self._visc = (ImplicitViscosity(n, W, d, self._reducer, self.model.sph.viscosity, self._n_active,
                                        cfg.viscosity_tolerance, cfg.viscosity_max_iterations,
                                        LAPLACE_D / laplacian_lattice_factor(cfg.spacing, cfg.h, cfg.kernel),
                                        wall_coeff=float(cfg.boundary_viscosity or 0.0)
                                        * (wall_coefficient_scale(cfg.spacing, cfg.h, cfg.kernel)
                                           if cfg.boundary_viscosity else 0.0))
                      if self.viscosity == "weiler2018" else None)
        if cfg.boundary_viscosity and self._visc is None:
            raise ValueError(f"SolverDFSPH: boundary_viscosity needs the weiler2018 viscosity (resolved {self.viscosity!r})")
        self._empty_vec3 = wp.zeros(1, dtype=wp.vec3, device=d)
        self._empty_sv = wp.zeros(1, dtype=wp.spatial_vector, device=d)
        self._empty_tf = wp.zeros(1, dtype=wp.transform, device=d)

    def _alloc_bodies(self) -> None:
        B, d = self.bodies.count, self.device
        S = max(int(self.model.shape_count), 1)
        self._body_mode = wp.zeros(max(B, 1), dtype=wp.int32, device=d)
        self._step_f = wp.zeros((max(B, 1), 6), dtype=wp.int64, device=d)
        self._exch_j = wp.zeros((max(B, 1), 6), dtype=wp.int64, device=d)
        self._meter_j = wp.zeros((S, 3), dtype=wp.int64, device=d)
        self._meter_saved = wp.zeros((S, 3), dtype=wp.int64, device=d)   # the exchange's start (passes)
        self._shape_work = wp.zeros(S, dtype=wp.int64, device=d)  # per step (bodies.py, SCR-WP9-6)
        self._refresh_body_modes()

    def _refresh_body_modes(self) -> None:
        if self.bodies.count == 0:
            return
        modes = bd.body_modes(self.bodies.flags.numpy(), self.bodies.inv_mass.numpy(), self.config.integrate_bodies)
        self._body_mode.assign(modes)

    # ------------------------------------------------------------------------------------------------- structs
    def _particles(self, state: State) -> pr.Particles:
        p = pr.Particles()
        m = self.model
        p.q, p.qd = state.particle_q, state.particle_qd
        p.mass, p.rho0, p.flags = m.particle_mass, self.layout.rest_density, m.particle_flags
        p.phase, p.world = self.layout.phase, m.particle_world
        nb = self._nbrs
        p.nbr, p.nbr_hi, p.nbr_wide, p.wstart, p.count = nb.nbr, nb.hi, nb.wide, nb.base, nb.count
        return p

    def _particles_model(self) -> pr.Particles:
        """Particles struct on the model's own arrays (ledger and capacity checks; no state needed)."""
        return self._particles(self.model) if self.layout.n else None

    def _slot_struct(self) -> pr.Slots:
        s = pr.Slots()
        s.shape, s.psi, s.gpsi, s.vb = self._slots.shape, self._slots.psi, self._slots.gpsi, self._slots.vb
        s.reg = self._w.reg
        return s

    def _body_state(self, state: State) -> tuple[wp.array, wp.array]:
        if self.bodies.count == 0 or state.body_q is None:
            return self._empty_tf, self._empty_sv
        if state.body_q.shape[0] != self.bodies.count:  # e.g. a coupled parent's state given to prime/density_ratio
            raise ValueError(f"SolverDFSPH: state has {state.body_q.shape[0]} bodies, the SPH model "
                             f"{self.bodies.count}; inside SolverCoupledProxy pass the SPH entry's state "
                             "(entry_state('sph')), whose bodies are the view's own")
        return state.body_q, state.body_qd

    # ------------------------------------------------------------------------------------------------- step
    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None,
             dt: float, *, observables: SolverBase.Observables | None = None) -> None:
        del control, contacts, observables
        if not dt > 0.0:
            raise ValueError(f"SolverDFSPH.step: dt must be > 0, got {dt}")
        if self.layout.n == 0:
            return
        if self._tension is not None:
            self._tension.check_dt(dt)
        self._copy_state(state_in, state_out)
        p = self._particles(state_out)
        bq_in = self._body_state(state_in)[0]  # start-of-step poses (sweep, leak monitor)
        bq, bqd = self._body_state(state_out)
        self._begin_step()
        self._soft_step_start(state_out, dt)  # before the grid: particle walls are in it (particle_walls.py)
        self._nbrs.build(p.q, p.flags, p.phase, p.world, self.stats.overflow)
        self.pw.begin(state_out, self._nbrs.grid.id)
        compute_slots(p.q, p.flags, p.phase, p.world, self.shapes.arrays, bq, bqd, self.bodies.com, self._h,
                      self._kind, self._slots, self.device)
        sl = self._slot_struct()
        w = self._w
        self._launch(pr.density_alpha_kernel, [p, sl, self._h, self._kind, self._n_full, self._tot_min, w.ratio,
                                               w.alpha, w.neff, w.active])
        self._reducer.sum(w.active, self._n_active)
        self._flow_speed(p)
        if self.config.divergence_solve:
            self._solve(p, sl, pr.MODE_DIVERGENCE, dt)
            self._soft_refresh(state_out)
        self._nonpressure(p, state_in, dt, bq, bqd)
        self._soft_refresh(state_out)
        self._solve(p, sl, pr.MODE_DENSITY, dt)
        self._soft_refresh(state_out)
        self._move_bodies(state_in, bq, bqd, dt)
        self.pw.accumulate(p.q, p.world, self._slots, self._slot_impulse, self.layout.n,
                           self.config.coupling == "one_way")
        self._soft_end_step(state_out)
        self._launch(advect_kernel, [p, dt])
        self.soft.advance(state_out, dt)
        self.pw.advance(state_out, dt)
        sl_ = self._slots
        r = self.config.particle_radius
        self._launch(project_kernel, [p, sl_.shape, sl_.phi, sl_.xbl, sl_.nbl, sl_.face, sl_.vb, self.shapes.arrays,
                                      bq, bqd, self.bodies.com, PROJECT_EPS * r, self.stats.projected, bq_in, dt,
                                      SWEEP * self._h, r, self.stats.swept, self._shape_work])
        if self.leaks is not None:  # before the escape box retires particles, so their exits are classified
            self.leaks.observe(state_in, state_out, bq_in, bq)
        self.emitters.step(p, dt, (bq, bqd, self.bodies.com))
        self.stats.speed(p, speed_kernel)
        self._mirror_flags()  # poured / retired flags to the full model when SPH runs on a view

    def enable_leak_monitor(self, depth_tol: float | None = None):
        """Count particles whose step path crosses wall solid (LIQ-8, ``leaks.LeakMonitor``); returns the monitor."""
        from .leaks import DEPTH_TOL, LeakMonitor

        self.leaks = LeakMonitor(self, DEPTH_TOL if depth_tol is None else depth_tol)
        return self.leaks

    def _launch(self, kernel, inputs: list, dim: int | None = None) -> None:
        wp.launch(kernel, dim=self.layout.n if dim is None else dim, inputs=inputs, device=self.device)

    def _copy_state(self, state_in: State, state_out: State) -> None:
        if state_out is state_in:
            return
        wp.copy(state_out.particle_q, state_in.particle_q)
        wp.copy(state_out.particle_qd, state_in.particle_qd)
        if self.bodies.count and state_in.body_q is not None:
            wp.copy(state_out.body_q, state_in.body_q)
            wp.copy(state_out.body_qd, state_in.body_qd)

    def _begin_step(self) -> None:
        self._slot_impulse.zero_()
        self._shape_work.zero_()
        if self.bodies.count:
            self._step_f.zero_()

    def _nonpressure(self, p: pr.Particles, state_in: State, dt: float, bq: wp.array, bqd: wp.array,
                     predict: bool = True) -> None:
        pf = state_in.particle_f
        has_f = int(pf is not None)
        xsph = self._xsph_c if self.viscosity == "xsph" else 0.0
        self._launch(nonpressure_kernel, [p, self._h, self._kind, dt, self.model.gravity,
                                          pf if has_f else self._empty_vec3, has_f, self._w.ratio, xsph,
                                          self._vtmp])
        wp.copy(p.qd, self._vtmp, count=self.layout.n)
        if self._tension is not None:
            self._tension.apply(p, self._slots, self._w.ratio, self._h, self._kind, dt, self._slot_impulse)
        if self._visc is not None:
            self._visc.solve(p, self._w.ratio, self._h, self._kind, dt, self._wall_terms(p, dt, bq, bqd, predict),
                             self.config.particle_radius, self._slot_impulse)

    def _wall_terms(self, p: pr.Particles, dt: float, bq: wp.array, bqd: wp.array, predict: bool):
        """WallSlots of the wall viscosity (None when off); with ``predict`` the walls' velocities carry the step's
        pressure guess (viscosity.py, SCR-WP9-11; the prime has no guess yet). Work arrays: kappa, acc, and _vtmp
        (free between the non-pressure update and the density solve)."""
        if not self._visc.has_wall:
            return None
        w, shift = self._w, (self._empty_vec3, 0)
        if predict:
            sl = self._slot_struct()
            self._launch(pressure_guess_kernel, [p, self._kd, dt, 1.0, w.kappa])  # last step's pressure, undamped
            res = (self._ambient.resid, 1) if self._ambient is not None else (self._empty_vec3, 0)
            self._launch(guess_shift_kernel, [p, sl, self._h, self._kind, w.kappa, dt, *res, w.acc])
            self._launch(wall_shift_kernel, [p, sl, self._h, self._kind, w.acc, self._vtmp])
            shift = (self._vtmp, 1)
        return wall_slots(self._slots, self.shapes.arrays, bq, bqd, self.bodies.com, self.config, *shift)

    def _move_bodies(self, state_in: State, bq: wp.array, bqd: wp.array, dt: float) -> None:
        """Sum boundary reactions per shape (force meters) and per body, then integrate the bodies."""
        m, b, B = self.model, self.bodies, self.bodies.count
        if int(m.shape_count) > 0:
            wp.launch(bd.accumulate_kernel, dim=(self.layout.n, NB), device=self.device,
                      inputs=[self._slots.shape, self._slots.xb, self._slots.vb, self._slot_impulse, m.shape_body, bq,
                              b.com, dt, int(self.config.coupling == "one_way"), self._step_f, self._exch_j,
                              self._meter_j, self._shape_work])
        self.soft.accumulate(self._slots, self._slot_impulse, self.layout.n, self.config.coupling == "one_way")
        if B == 0:
            return
        bf = state_in.body_f
        has_f = int(bf is not None)
        wp.launch(bd.integrate_bodies_kernel, dim=B, device=self.device,
                  inputs=[self._body_mode, bq, bqd, bf if has_f else self._empty_sv, has_f, b.mass, b.inv_mass,
                          b.inertia, b.inv_inertia, b.com, b.world, m.gravity, self._step_f, dt, bq, bqd])

    def notify_model_changed(self, flags: int) -> None:
        if flags & (int(ModelFlags.BODY_INERTIAL_PROPERTIES) | int(ModelFlags.BODY_PROPERTIES)):
            self._refresh_body_modes()
        if flags & int(ModelFlags.SHAPE_PROPERTIES):
            self.shapes = shape_table(self.model, self.config, self.device)
            self.shapes.arrays.soft = self.soft.arrays
            self.pw.bind(self.shapes.arrays)
            if self._ambient is not None:
                self._ambient.refresh(self.model)

    def reset(self, state: State, world_mask: wp.array | None = None, flags: int | None = None) -> None:
        """Masked worlds (all when None) back to their build-time SPH configuration: warm starts and per-world
        counters cleared, particle flags and phase restored (particles retired by sinks or the escape box return;
        positions and velocities are the caller's, e.g. DirectSim.reset). WorldSnapshot.restore of
        ``alloy_state_arrays`` afterwards still wins (latch / rollback)."""
        del state, flags
        self.stats.reset(self.model, self._kv, self._kd, world_mask, (self._flags0, self._phase0, self.layout.phase))
        self.emitters.reset(world_mask)  # re-arms patch emitters
        self.port.reset(world_mask)
        self._mirror_flags()

    # ------------------------------------------------------------------------------------------------- reports
    def report(self) -> SPHReport:
        rep = self.stats.report(self._particles_model(), self.config.max_iterations)
        return self._tension.annotate(rep) if self._tension is not None else rep

    def prime(self, state: State, dt: float, tol: float = 1.0e-6, max_iterations: int = 2000) -> int:
        """Host helper: give a fluid at rest its hydrostatic pressure before the first step (``runtime.prime``);
        returns the iterations used. ``state`` is this solver's own (in a coupled sim the SPH entry's state)."""
        return prime_hydrostatic(self, state, dt, tol, max_iterations)

    def _prepare_density(self, state: State, p: pr.Particles) -> None:
        bq, bqd = self._body_state(state)
        self._nbrs.build(p.q, p.flags, p.phase, p.world, self.stats.overflow)
        self.soft.begin(state)
        self.pw.begin(state, self._nbrs.grid.id)
        compute_slots(p.q, p.flags, p.phase, p.world, self.shapes.arrays, bq, bqd, self.bodies.com, self._h,
                      self._kind, self._slots, self.device)
        w = self._w
        self._launch(pr.density_alpha_kernel, [p, self._slot_struct(), self._h, self._kind, self._n_full,
                                               self._tot_min, w.ratio, w.alpha, w.neff, w.active])
        self._reducer.sum(w.active, self._n_active)

    def density_ratio(self, state: State) -> tuple[np.ndarray, np.ndarray]:
        """Host helper (not capture-safe): density ratio rho/rho0 of every particle in ``state`` and the active-liquid
        mask, from fresh neighbour lists and boundary slots. Used by laws and tests (LIQ-2/3)."""
        if self.layout.n == 0:
            return np.zeros(0), np.zeros(0, dtype=bool)
        self._prepare_density(state, self._particles(state))
        w = self._w
        return w.ratio.numpy()[: self.layout.n].copy(), w.active.numpy()[: self.layout.n] > 0.0

    def mass_ledger(self) -> dict[str, np.ndarray]:
        """Exact int64 fixed-point ledger per world: initial, in, out, active (LIQ-1: active + out == initial + in)."""
        return self.stats.ledger(self._particles_model())

    def shape_impulse(self) -> np.ndarray:
        """Cumulative fluid impulse on each shape since the last ``reset_shape_meters`` [N·s], (S, 3) float64."""
        return bd.fixed_rows_to_float(self._meter_j.numpy(), bd.IMP_SCALE_HOST)

    def reset_shape_meters(self) -> None:
        self._meter_j.zero_()
        self._meter_saved.zero_()

    def alloy_shape_work(self, out: wp.array) -> None:
        """Work [J] each shape's boundary did on the liquid during the last step, at the wall's point velocity
        (pressure reactions and wall projections; static shapes 0), into ``out`` ([shape_count] float64, model shape
        indices). A driven wall's work is energy no wrench carries; ledgers count it for robot shapes (SCR-WP9-6)."""
        wp.launch(bd.work_to_float_kernel, dim=int(self.model.shape_count), inputs=[self._shape_work, out],
                  device=self.device)

    def export_flux_energy(self, health) -> None:
        """Energy emitted minus energy retired since the last call, into ``health``'s W_ext (a HealthMonitor:
        ``add_external_work`` at its z_ref), so emitters and sinks do not read as energy jumps. Capture-safe; call
        once per health check, after the step(s)."""
        self.emitters.export_flux_energy(health)

    def body_force(self) -> np.ndarray:
        """Fluid wrench on each body during the last step [N, N·m], (B, 6) float64 (world frame, about the COM)."""
        return bd.fixed_rows_to_float(self._step_f.numpy(), bd.FORCE_SCALE_HOST)

    def choose_dt(self, control_dt: float) -> float:
        """Host, once per period: CFL step from the last step's max speed, snapped down to the D4 grid."""
        return snap_down(self.stats.cfl_dt(self.config), control_dt)

    def retire(self, state: State, mask: wp.array) -> None:
        """Retire the masked liquid particles for good (``mask [n]`` int32; graph-safe, deterministic): mass and energy
        leave the ledgers as at a sink; ``port.taken[w]`` counts them (ports.py)."""
        self.port.retire(state, mask)

    def add_emitter(self, world: int, region: wp.transform, half_extents: wp.vec3, rate: float,
                    velocity: wp.vec3, pool: np.ndarray | None = None) -> int:
        """Emit liquid from a box region at ``rate`` [m^3/s] with ``velocity`` [m/s] by activating spares (``pool``)."""
        return self.emitters.add_emitter(world, region, half_extents, rate, velocity, pool=pool)

    def set_emitter_enabled(self, e: int, on: bool) -> None:
        """Open or close emitter ``e`` (``add_emitter``'s index) between steps: a closed emitter emits nothing and its
        clock pauses until it is opened again (a burette's stopcock). Host write, not inside a captured graph."""
        mgr = self.emitters
        n = len(mgr.table.world)
        if not 0 <= e < n or mgr.dev is None:
            raise ValueError(f"set_emitter_enabled: emitter {e} does not exist ({n} emitters)")
        flags = mgr.dev.enabled.numpy().copy()
        flags[e] = 1 if on else 0
        mgr.dev.enabled.assign(flags)

    def add_patch_emitter(self, world_mask, body, local_points, velocity=(0.0, 0.0, 0.0)) -> int:
        """Triggered one-shot emitter (SCR-DC5-5, ``patch.py``): for each masked world w, ``local_points[w]``
        (N_w, 3) in the frame of ``body`` (one index or one per world; this solver's model indices, -1 = world frame),
        moving with it plus ``velocity`` (body frame). Spares are reserved now; ``trigger`` arms worlds. Returns pid."""
        return self.emitters.add_patch_emitter(world_mask, body, local_points, velocity)

    def trigger(self, pid: int, world_flags: wp.array) -> None:
        """Arm patch emitter ``pid`` in the worlds where ``world_flags`` (int32 or bool, one per world) is set; the
        next step emits there once (capture-safe: one launch)."""
        self.emitters.patches.trigger(pid, world_flags)

    def add_sink(self, world: int, region: wp.transform, half_extents: wp.vec3) -> int:
        """Retire particles that enter a box region; their mass goes to ``mass_out``."""
        return self.emitters.add_sink(world, region, half_extents)

    # ------------------------------------------------------------------------------------------------- protocols
    def alloy_stable_dt(self, state: State, contacts: Contacts | None, out: wp.array) -> None:
        del contacts
        stable_dt_per_world(self, self._particles(state), out)

    def alloy_energy(self, state: State, z_ref: wp.array, out: wp.array) -> None:
        energy_per_world(self, self._particles(state), z_ref, out)

    def alloy_state_arrays(self, state: State) -> Mapping[str, wp.array]:
        del state
        arrays = {"sph.kappa_v": self._kv, "sph.kappa_d": self._kd, "sph.phase": self.layout.phase,
                  "particle_flags": self.model.particle_flags}
        if self.model.shape_count:                       # a re-run period must not count its walls' impulse twice
            arrays["sph.meter_j"] = self._meter_j        # (with no shape the meter is a (1, 3) placeholder)
        arrays.update(self.stats.state_arrays())
        arrays.update(self.emitters.state_arrays())
        arrays.update(self.port.state_arrays())
        return arrays

    def alloy_state_worlds(self) -> Mapping[str, wp.array]:
        """Item -> world maps of the per-item arrays in ``alloy_state_arrays`` (WorldSnapshot extra_world,
        SCR-WP9-4); per-world counters are world-major and need no map."""
        pw = self.model.particle_world
        maps = {"sph.kappa_v": pw, "sph.kappa_d": pw, "sph.phase": pw, "particle_flags": pw}
        if self.model.shape_count:
            maps["sph.meter_j"] = self.model.shape_world
        maps.update(self.emitters.state_worlds())
        return maps

    def alloy_capacity_report(self) -> Mapping[str, int]:
        return self.stats.capacity()

    @staticmethod
    def emit(builder, obj, decision) -> None:
        from .build import emit_object

        emit_object(builder, obj, decision)


_ACTIVE = int(ParticleFlags.ACTIVE)
PROJECT_EPS = 1.0e-3  # wall projection skin, in particle radii
