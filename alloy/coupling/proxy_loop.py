# SPDX-License-Identifier: Apache-2.0
"""Residual-gated proxy iterations for one exchange (SPEC §3.6, coupling-time §4.4.2).

Pass 0 always runs with ``iteration_restart=False``; later passes run with ``iteration_restart=True`` either
unconditionally (k_min == k_max, e.g. lagged: exactly 2) or inside ``wp.capture_while`` on the device condition
``any world unconverged and k < k_max`` (a CUDA conditional node under graph capture, a host loop otherwise). Each
pass calls Newton's ``_step_proxy`` unchanged; the Scheduler's ``_blend_proxy_feedback`` override calls
``on_harvest`` to record residuals and blend per world.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import warp as wp

from alloy.coupling import aitken as ak
from alloy.coupling import capture_passes as cpm
from alloy.coupling import residuals as rs
from alloy.coupling.capture_passes import CapturedPasses
from alloy.coupling.endpoints import EndpointSet, capture_begin_positions
from alloy.coupling.freeze import SolverStateFreeze
from alloy.coupling.slew import FeedbackSlewLimit

log = logging.getLogger("alloy.coupling")


@dataclass(frozen=True)
class IterationConfig:
    """Resolved iteration settings (from ``SchedulerPolicy``/``InterfaceTolerance``/``AitkenPolicy``)."""

    k_min: int
    k_max: int
    tol_vel: float                 # [m/s]
    tol_pos: float                 # [m]
    tol_force: float               # [-] relative
    force_floor: float             # [N]
    gravity: float                 # [m/s^2] magnitude, for the m_eff·g force scale
    omega0: float
    omega0_strong: float
    omega_min: float
    omega_max: float
    omega_carry_min: float = 0.1
    carry_omega: bool = True
    slew_c: float | None = None    # SCR-SC4-4d: lagged body feedback slew limit (alloy.coupling.slew); None: off
    force_norm: str = "max"        # "max" | "l2" | "l2_free" (alloy.coupling.residuals.FORCE_NORMS)

    def __post_init__(self) -> None:
        if not 1 <= self.k_min <= self.k_max:
            raise ValueError(f"need 1 <= k_min <= k_max, got k_min={self.k_min}, k_max={self.k_max}")
        if self.force_norm not in rs.FORCE_NORMS:
            raise ValueError(f"force_norm must be one of {sorted(rs.FORCE_NORMS)}, got {self.force_norm!r}")


class ProxyIterationLoop:
    """Per-world convergence state and the pass loop for a ``SolverCoupledProxy`` subclass."""

    def __init__(self, solver: Any, sets: list[EndpointSet], cfg: IterationConfig,
                 capture_while: bool = True) -> None:
        self.solver = solver
        self.sets = sets
        self.cfg = cfg
        # CUDA conditional bodies may not allocate; Newton's solvers do inside a step, so for them the residual
        # loop is unrolled when capturing: to ``captured_passes`` passes, its short worlds re-run with the full loop
        # (capture_passes.py; frozen worlds keep their converged result).
        self.capture_while = capture_while
        self._by_mapping = {id(es.mapping): es for es in sets}
        model = solver.model
        self.device = model.device
        w = int(model.world_count)
        self.world_count = w
        self.res_vel = wp.zeros(w, dtype=float, device=self.device)
        self.res_pos = wp.zeros(w, dtype=float, device=self.device)
        self.res_force = wp.zeros(w, dtype=float, device=self.device)
        self.l2_sums = wp.zeros((w, 3), dtype=wp.float64, device=self.device)   # L2 force norm sums, per pass
        self.world_converged = wp.zeros(w, dtype=wp.int32, device=self.device)
        self.iters_used = wp.zeros(w, dtype=wp.int32, device=self.device)
        self.start_frozen = wp.zeros(w, dtype=wp.int32, device=self.device)
        self.n_unconverged = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.iteration = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.condition = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.freeze = SolverStateFreeze(model, {n: e.solver for n, e in solver._entries.items()},
                                        enabled=cfg.k_min < cfg.k_max)
        self.captured_passes = CapturedPasses(w, self.device, enabled=cfg.k_min < cfg.k_max and not capture_while,
                                              passes=cfg.k_min)       # lever #1: capture the passes that run
        self.slew = FeedbackSlewLimit(sets, cfg.slew_c) if cfg.slew_c is not None else None
        self._dt = 0.0

    # ------------------------------------------------------------------ exchange
    def run(self, state_in: Any, control: Any, contacts: Any, dt: float) -> None:
        """Run k_min..k_max passes of one exchange interval ``dt``."""
        self._dt = float(dt)
        self._begin(state_in)
        self._pass(state_in=state_in, control=control, contacts=contacts, dt=dt, restart=False)
        extra = self.cfg.k_max - 1
        unroll = self.cfg.k_min >= self.cfg.k_max or (not self.capture_while and cpm.capturing(self.device))
        if extra > 0 and unroll:
            cp, k_min, k_max = self.captured_passes, self.cfg.k_min, self.cfg.k_max
            for _ in range(cp.captured(k_min, k_max) - 1):
                self._pass(state_in=state_in, control=control, contacts=contacts, dt=dt, restart=True)
            if cp.capped(k_min, k_max):
                cp.mark(self.world_converged)
        elif extra > 0:
            wp.capture_while(self.condition, self._pass, state_in=state_in, control=control,
                             contacts=contacts, dt=dt, restart=True)
        self._end()

    def _begin(self, state_in: Any) -> None:
        dev = self.device
        wp.launch(rs.begin_exchange_kernel, dim=self.world_count,
                  inputs=[self.world_converged, self.iters_used, self.start_frozen], device=dev)
        self.iteration.zero_()
        capture_begin_positions(self.sets, state_in)
        if self.slew is not None:
            self.slew.begin()
        for es in self.sets:
            if es.staggered:
                omega0 = self.cfg.omega0_strong if es.strong else self.cfg.omega0
                wp.launch(ak.aitken_begin_kernel, dim=self.world_count,
                          inputs=[int(self.cfg.carry_omega), omega0, self.cfg.omega_carry_min, self.cfg.omega_max,
                                  es.aitken_omega, es.aitken_has_prev], device=dev)

    def _pass(self, state_in: Any, control: Any, contacts: Any, dt: float, restart: bool) -> None:
        dev = self.device
        self.freeze.begin_pass(self.world_converged)
        for es in self.sets:
            wp.launch(rs.gather_feedback_kernel, dim=es.count,
                      inputs=[es.proxy_global, es.mapping.coupling_forces, es.f_used], device=dev)
        wp.launch(rs.reset_active_residuals_kernel, dim=self.world_count,
                  inputs=[self.world_converged, self.res_vel, self.res_pos, self.res_force], device=dev)
        if self.cfg.force_norm != "max":
            wp.launch(rs.reset_active_l2_sums_kernel, dim=self.world_count, inputs=[self.world_converged, self.l2_sums],
                      device=dev)
        if restart:
            self.solver._distribute_state(state_in, dt=dt, iteration_restart=True)
        self.solver._step_proxy(state_in, control, contacts, dt, iteration_restart=restart)
        self._after_pass()

    def _after_pass(self) -> None:
        dev = self.device
        self.n_unconverged.zero_()
        wp.launch(rs.convergence_kernel, dim=self.world_count,
                  inputs=[self._params(), self.iteration, self.res_vel, self.res_pos, self.res_force,
                          self.l2_sums, self.world_converged, self.iters_used, self.n_unconverged], device=dev)
        self.freeze.capture_converged(self.world_converged)
        for es in self.sets:
            wp.launch(rs.freeze_new_worlds_kernel, dim=es.count,
                      inputs=[es.proxy_global, es.world, self.world_converged, es.f_used,
                              es.mapping.coupling_forces, es.frozen_out], device=dev)
        wp.launch(rs.settle_new_worlds_kernel, dim=self.world_count, inputs=[self.world_converged], device=dev)
        wp.launch(rs.advance_loop_kernel, dim=1,
                  inputs=[self.cfg.k_max, self.n_unconverged, self.iteration, self.condition], device=dev)

    def _end(self) -> None:
        """Frozen worlds carry the blended value (and solver state) from their converging pass into the next one."""
        self.freeze.finish(self.world_converged)
        for es in self.sets:
            wp.launch(rs.scatter_frozen_kernel, dim=es.count,
                      inputs=[es.proxy_global, es.world, self.world_converged, es.frozen_out,
                              es.mapping.coupling_forces], device=self.device)

    def _params(self) -> rs.ConvergenceParams:
        p = rs.ConvergenceParams()
        p.k_min = self.cfg.k_min
        p.k_max = self.cfg.k_max
        p.tol_vel = self.cfg.tol_vel
        p.tol_pos = self.cfg.tol_pos
        p.tol_force = self.cfg.tol_force
        p.check_force = int(any(es.staggered for es in self.sets))
        p.force_norm = rs.FORCE_NORMS[self.cfg.force_norm]
        p.force_floor = self.cfg.force_floor
        return p

    # ------------------------------------------------------------------ per-mapping hook
    def on_harvest(self, mapping: Any) -> bool:
        """Called after the destination harvest of ``mapping``; returns False if the mapping is not tracked."""
        es = self._by_mapping.get(id(mapping))
        if es is None:
            return False
        self._record_residuals(es)
        if es.staggered:
            self._blend_aitken(es)
        else:
            wp.launch(rs.scatter_frozen_kernel, dim=es.count,
                      inputs=[es.proxy_global, es.world, self.world_converged, es.f_used,
                              mapping.coupling_forces], device=self.device)
        if self.slew is not None:
            self.slew.apply(es, self.world_converged, self._dt)
        return True

    def _record_residuals(self, es: EndpointSet) -> None:
        src = self.solver._entries[es.src_name]
        dst = self.solver._entries[es.dst_name]
        if es.kind == "body":
            kernel, sq, sqd, dq, dqd = (rs.body_residual_kernel, src.state_1.body_q, src.state_1.body_qd,
                                        dst.state_1.body_q, dst.state_1.body_qd)
        else:
            kernel, sq, sqd, dq, dqd = (rs.particle_residual_kernel, src.state_1.particle_q,
                                        src.state_1.particle_qd, dst.state_1.particle_q, dst.state_1.particle_qd)
        inp = rs.ResidualInputs()
        inp.gravity, inp.force_floor, inp.staggered = self.cfg.gravity, self.cfg.force_floor, int(es.staggered)
        gate = getattr(self.solver, "_gate", None)          # lever #2: released endpoints add no residual
        idx = gate.set_index.get(id(es)) if gate is not None else None
        inp.gated = int(idx is not None)
        if idx is not None:
            inp.armed, inp.gate_index = gate.armed, idx
        wp.launch(kernel, dim=es.count,
                  inputs=[inp, self.iteration, es.src_local, es.proxy_local, es.proxy_global, es.world, es.meff,
                          self.world_converged, sq, sqd, dq, dqd, es.mapping.coupling_forces, es.f_used, es.f_raw,
                          es.x_prev, es.v_prev, self.res_vel, self.res_pos, self.res_force], device=self.device)
        if self.cfg.force_norm != "max":
            wp.launch(rs.force_l2_sums_kernel, dim=self.world_count,
                      inputs=[int(self.cfg.force_norm == "l2_free"), es.csr.start, es.csr.index, es.meff,
                              self.world_converged, es.f_raw, es.f_used, self.l2_sums], device=self.device)

    def _blend_aitken(self, es: EndpointSet) -> None:
        wp.launch(ak.aitken_update_kernel, dim=self.world_count,
                  inputs=[es.csr.start, es.csr.index, self.world_converged, es.f_raw, es.f_used, es.r_prev,
                          self.cfg.omega_min, self.cfg.omega_max, es.aitken_stats, es.aitken_omega,
                          es.aitken_has_prev], device=self.device)
        wp.launch(ak.aitken_blend_kernel, dim=es.count,
                  inputs=[es.proxy_global, es.world, self.world_converged, es.f_raw, es.f_used, es.aitken_omega,
                          es.r_prev, es.mapping.coupling_forces], device=self.device)

    def state_arrays(self) -> dict[str, wp.array]:
        out: dict[str, wp.array] = {}
        for n, es in enumerate(self.sets):
            out.update(es.state_arrays(f"proxy{n}"))
        return out

    def world_maps(self) -> dict[str, wp.array]:
        """coupling_forces is indexed by global body/particle id, r_prev by endpoint."""
        model = self.solver.model
        out: dict[str, wp.array] = {}
        for n, es in enumerate(self.sets):
            out[f"proxy{n}.coupling_forces"] = model.body_world if es.kind == "body" else model.particle_world
            out[f"proxy{n}.r_prev"] = es.world
        return out
