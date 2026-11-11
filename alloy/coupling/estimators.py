# SPDX-License-Identifier: Apache-2.0
"""Stability estimators for Newton solvers (SPEC §3.6 "Estimators WP4 registers", coupling-time §4.4.3).

``alloy_stable_dt`` writes the largest stable internal step per world [s] (float32, inf = no limit):
- SolverMuJoCo / SolverKamino: ``min(0.4·τ_eff, 0.5·margin/|v|max)``, ``τ_eff = max(τ_req, 2·dt)`` with ``dt`` the
  solver's last step (``mjw_model.opt.timestep``) and τ_req the smallest geom/pair solref time constant (implicit soft
  constraints: the clamp, mujoco_warp@a9daa05f:mujoco_warp/_src/constraint.py:104-105, keeps them stable at any dt,
  so √(K/m) would over-ask);
- SolverSemiImplicit / SolverFeatherstone (explicit penalty contact): ``0.4/ω``, ``ω = sqrt(K/m_min)``;
- SolverVBD / SolverXPBD: ``0.4·min(r/|v|)`` over active particles; SolverImplicitMPM: ``0.5·voxel/|v|max``.
Reductions are ``atomic_max``/``atomic_min`` on float32 (order-independent). ``alloy_energy`` uses the D20 energy
kernel on the solver's (entry) model.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import warp as wp

from alloy.core.defaults import RIGID_GAP_M
from alloy.core.registry import register_estimator
from alloy.coupling import energy as en

log = logging.getLogger("alloy.coupling")

DEFAULT_TIMECONST = 0.02     # s, MuJoCo default solref time constant (newton kernels.py:190-204 fallback)
V_FLOOR = wp.constant(1.0e-6)
INF = wp.constant(1.0e30)


@wp.kernel(enable_backward=False)
def body_speed_kernel(body_qd: wp.array[wp.spatial_vector], body_world: wp.array[wp.int32],
                      vmax: wp.array[float]):
    b = wp.tid()
    w = body_world[b]
    if w >= 0:
        wp.atomic_max(vmax, w, wp.length(wp.spatial_top(body_qd[b])))


@wp.kernel(enable_backward=False)
def particle_speed_kernel(particle_qd: wp.array[wp.vec3], particle_world: wp.array[wp.int32],
                          particle_mass: wp.array[float], vmax: wp.array[float]):
    i = wp.tid()
    w = particle_world[i]
    if w >= 0 and particle_mass[i] > 0.0:
        wp.atomic_max(vmax, w, wp.length(particle_qd[i]))


@wp.kernel(enable_backward=False)
def particle_ratio_kernel(particle_qd: wp.array[wp.vec3], particle_world: wp.array[wp.int32],
                          particle_mass: wp.array[float], particle_radius: wp.array[float],
                          ratio: wp.array[float]):
    i = wp.tid()
    w = particle_world[i]
    if w >= 0 and particle_mass[i] > 0.0:
        v = wp.max(wp.length(particle_qd[i]), V_FLOOR)
        wp.atomic_min(ratio, w, particle_radius[i] / v)


@wp.kernel(enable_backward=False)
def mujoco_dt_kernel(tau_req: wp.array[float], timestep: wp.array[float], margin: wp.array[float],
                     vmax: wp.array[float], out: wp.array[float]):
    w = wp.tid()
    ts = timestep[wp.min(w, timestep.shape[0] - 1)]
    tau_eff = wp.max(tau_req[w], 2.0 * ts)
    travel = 0.5 * margin[w] / wp.max(vmax[w], V_FLOOR)
    out[w] = wp.min(0.4 * tau_eff, travel)


@wp.kernel(enable_backward=False)
def scaled_kernel(scale: float, values: wp.array[float], out: wp.array[float]):
    """out = scale·values (inf stays inf)."""
    w = wp.tid()
    out[w] = scale * values[w]


@wp.kernel(enable_backward=False)
def cfl_kernel(length: float, factor: float, vmax: wp.array[float], out: wp.array[float]):
    w = wp.tid()
    out[w] = factor * length / wp.max(vmax[w], V_FLOOR)


class _EstimatorBase:
    """Shared energy path and scratch for one solver bound at construction."""

    def __init__(self, solver: Any) -> None:
        self.solver = solver
        self.model = solver.model
        self.device = self.model.device
        self.world_count = int(self.model.world_count)
        self._scratch = wp.zeros(self.world_count, dtype=float, device=self.device)
        self._energy_inputs: en.EnergyInputs | None = None

    def alloy_energy(self, state: Any, z_ref: wp.array, out: wp.array) -> None:
        if self._energy_inputs is None:
            self._energy_inputs = _energy_inputs(self.model)
            self._vmax = wp.zeros(self.world_count, dtype=float, device=self.device)
            self._bad = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        en.launch_energy(self._energy_inputs, state, z_ref, out, self._vmax, self._bad, self.device)

    def _body_vmax(self, state: Any) -> wp.array:
        self._scratch.zero_()
        if self.model.body_count:
            wp.launch(body_speed_kernel, dim=self.model.body_count,
                      inputs=[state.body_qd, self.model.body_world, self._scratch], device=self.device)
        return self._scratch


def _energy_inputs(model: Any) -> en.EnergyInputs:
    return en.energy_inputs(model)


def _per_world_min_positive(values: np.ndarray, worlds: int, default: float) -> np.ndarray:
    """values: (1 or W, n) -> (W,) min over positive entries, ``default`` where none."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim == 1:
        v = v[None, :]
    v = np.where(v > 0.0, v, np.inf)
    m = v.min(axis=1) if v.shape[1] else np.full(v.shape[0], np.inf)
    m = np.where(np.isfinite(m), m, default)
    return np.broadcast_to(m, (worlds,)).astype(np.float32) if m.size == 1 else m[:worlds].astype(np.float32)


class MuJoCoEstimator(_EstimatorBase):
    """``min(0.4·max(τ_req, 2·dt), 0.5·margin/|v|max)`` per world."""

    def __init__(self, solver: Any) -> None:
        super().__init__(solver)
        tau = self._tau_req()
        self.tau_req = wp.array(tau, dtype=float, device=self.device)
        self.margin = wp.array(np.full(self.world_count, self._margin(), np.float32), dtype=float, device=self.device)

    def _tau_req(self) -> np.ndarray:
        mjm = getattr(self.solver, "mjw_model", None)
        if mjm is None:
            return np.full(self.world_count, DEFAULT_TIMECONST, np.float32)
        tcs = [mjm.geom_solref.numpy()[..., 0]]
        if getattr(mjm, "pair_solref", None) is not None and mjm.pair_solref.size:
            tcs.append(mjm.pair_solref.numpy()[..., 0])
        per = [_per_world_min_positive(t.reshape(t.shape[0], -1), self.world_count, DEFAULT_TIMECONST) for t in tcs]
        return np.minimum.reduce(per)

    def _margin(self) -> float:
        gap = getattr(self.model, "shape_gap", None)
        if gap is None or self.model.shape_count == 0:
            return RIGID_GAP_M
        g = gap.numpy()
        g = g[g > 0.0]
        return float(g.min()) if g.size else RIGID_GAP_M

    def _timestep(self) -> wp.array:
        mjm = getattr(self.solver, "mjw_model", None)
        if mjm is not None:
            return mjm.opt.timestep
        return wp.full(1, DEFAULT_TIMECONST, dtype=float, device=self.device)

    def alloy_stable_dt(self, state: Any, contacts: Any, out: wp.array) -> None:
        vmax = self._body_vmax(state)
        wp.launch(mujoco_dt_kernel, dim=self.world_count,
                  inputs=[self.tau_req, self._timestep(), self.margin, vmax, out], device=self.device)


class KaminoEstimator(MuJoCoEstimator):
    """Kamino as MuJoCo (SPEC §3.6) with the default time constant (no solref arrays are read)."""

    def _tau_req(self) -> np.ndarray:
        return np.full(self.world_count, DEFAULT_TIMECONST, np.float32)

    def _timestep(self) -> wp.array:
        if not hasattr(self, "_ts"):
            self._ts = wp.full(1, 0.0, dtype=float, device=self.device)
        return self._ts


class PenaltyEstimator(_EstimatorBase):
    """Explicit penalty contact: 0.4/ω, ω = sqrt(K_max/m_min) (host constants, per world identical)."""

    def __init__(self, solver: Any) -> None:
        super().__init__(solver)
        m = self.model
        ke = float(getattr(m, "soft_contact_ke", 0.0) or 0.0)
        if m.shape_count:
            ke = max(ke, float(np.max(m.shape_material_ke.numpy())))
        masses = m.body_mass.numpy() if m.body_count else np.zeros(0)
        masses = masses[masses > 0.0]
        m_min = float(masses.min()) if masses.size else math.inf
        dt = 0.4 / math.sqrt(ke / m_min) if ke > 0.0 and math.isfinite(m_min) else math.inf
        self.dt = wp.full(self.world_count, min(dt, 1e30), dtype=float, device=self.device)

    def alloy_stable_dt(self, state: Any, contacts: Any, out: wp.array) -> None:
        wp.copy(out, self.dt)


class ParticleEstimator(_EstimatorBase):
    """VBD/XPBD displacement bound 0.4·min(r/|v|) per world."""

    def alloy_stable_dt(self, state: Any, contacts: Any, out: wp.array) -> None:
        self._scratch.fill_(INF)
        m = self.model
        if m.particle_count:
            wp.launch(particle_ratio_kernel, dim=m.particle_count,
                      inputs=[state.particle_qd, m.particle_world, m.particle_mass, m.particle_radius,
                              self._scratch], device=self.device)
        wp.launch(scaled_kernel, dim=self.world_count, inputs=[0.4, self._scratch, out], device=self.device)


class MPMEstimator(_EstimatorBase):
    """Implicit MPM particle-transfer CFL 0.5·voxel/|v|max."""

    def alloy_stable_dt(self, state: Any, contacts: Any, out: wp.array) -> None:
        self._scratch.zero_()
        m = self.model
        if m.particle_count:
            wp.launch(particle_speed_kernel, dim=m.particle_count,
                      inputs=[state.particle_qd, m.particle_world, m.particle_mass, self._scratch],
                      device=self.device)
        voxel = float(getattr(self.solver, "voxel_size", 0.0) or 0.0)
        wp.launch(cfl_kernel, dim=self.world_count, inputs=[voxel, 0.5, self._scratch, out], device=self.device)


@wp.kernel(enable_backward=False)
def clamp_kernel(lo: float, hi: float, out: wp.array[float]):
    w = wp.tid()
    out[w] = wp.min(wp.max(out[w], lo), hi)


class SPHEstimator(_EstimatorBase):
    """DFSPH CFL step per world: clamp(cfl·2r/|v|max, dt_min, dt_max) from the solver's own config (D6,
    ``alloy.solvers.sph.config.stable_dt``); the Scheduler's ``ClockPolicy.safety`` applies on top (as for MPM)."""

    def alloy_stable_dt(self, state: Any, contacts: Any, out: wp.array) -> None:
        self._scratch.zero_()
        m, cfg = self.model, self.solver.config
        if m.particle_count:
            wp.launch(particle_speed_kernel, dim=m.particle_count,
                      inputs=[state.particle_qd, m.particle_world, m.particle_mass, self._scratch],
                      device=self.device)
        wp.launch(cfl_kernel, dim=self.world_count, inputs=[float(cfg.spacing), float(cfg.cfl), self._scratch, out],
                  device=self.device)
        wp.launch(clamp_kernel, dim=self.world_count, inputs=[float(cfg.dt_min), float(cfg.dt_max), out],
                  device=self.device)


def _register_sph_estimator() -> None:
    """SolverDFSPH (WP9) when its package is importable; coupling does not require it."""
    try:
        from alloy.solvers.sph.solver import SolverDFSPH
    except ImportError:                                    # pragma: no cover - SPH package absent
        log.debug("SolverDFSPH not importable: no SPH stability estimator registered")
        return
    register_estimator(SolverDFSPH, SPHEstimator)


def register_builtin_estimators() -> None:
    """Register the estimators above for Newton's solver classes (idempotent)."""
    import alloy._vendor.newton.solvers as ns

    register_estimator(ns.SolverMuJoCo, MuJoCoEstimator)
    register_estimator(ns.SolverKamino, KaminoEstimator)
    register_estimator(ns.SolverSemiImplicit, PenaltyEstimator)
    register_estimator(ns.SolverFeatherstone, PenaltyEstimator)
    register_estimator(ns.SolverVBD, ParticleEstimator)
    register_estimator(ns.SolverXPBD, ParticleEstimator)
    register_estimator(ns.SolverImplicitMPM, MPMEstimator)
    _register_sph_estimator()
