# SPDX-License-Identifier: Apache-2.0
"""Residual-gated ADMM iterations for one exchange (SPEC §3.6 SchedulerADMM, coupling-time §4.4, Boyd et al. 2011).

Mirrors ``SolverCoupledADMM._step_coupled`` (newton@ac7f7830:newton/_src/solvers/coupled/solver_coupled_admm.py:
2965-3026) and adds, after every dual update, per-world residuals over the ADMM rows (max-norms, order-independent):
primal ``W·|Jv − u|`` and dual ``ρ·W·|u − u_prev|``, converged when ``k ≥ k_min`` and both are below
``ε_abs + ε_rel·scale`` (scale = max row ``W·max(|Jv|, |u|)`` resp. ``W·|λ|``). A converged world is frozen: its rows
get back the (λ, u, Jv) its forces were built from, so later iterations reproduce it bitwise and the world's result
does not depend on the batch; the post-update values are carried into the next exchange. The interface work
``dt·W·λ·Jv`` is summed per world in int64 fixed point (deterministic integer atomics, D9). Under a CUDA capture the
loop is unrolled to ``captured_passes`` passes, and worlds left unconverged are re-run with the full loop
(``capture_passes``, lever #1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy.coupling import capture_passes as cpm
from alloy.coupling import residuals as rs
from alloy.coupling.capture_passes import CapturedPasses
from alloy.coupling.freeze import SolverStateFreeze

log = logging.getLogger("alloy.coupling")

WORK_QUANTUM = 2.0**-40          # J per int64 unit (≈ 9.1e-13 J; range ±8.4e6 J)


@dataclass(frozen=True)
class AdmmIterationConfig:
    k_min: int
    k_max: int
    eps_abs: float
    eps_rel: float


@dataclass
class RowGroup:
    """One Newton ADMM group plus its endpoint->world maps and freeze buffers (capacity-sized)."""

    group: Any
    ids_a: wp.array
    world_a: wp.array
    ids_b: wp.array
    world_b: wp.array
    prev: dict[str, wp.array] = field(default_factory=dict)
    frozen: dict[str, wp.array] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return int(self.group.count)

    @property
    def active_count(self) -> wp.array | None:
        return getattr(self.group, "active_count", None)


@wp.func
def _row_world(i: int, ids_a: wp.array[int], world_a: wp.array[wp.int32], ids_b: wp.array[int],
               world_b: wp.array[wp.int32]) -> int:
    a = ids_a[i]
    if a >= 0:
        return world_a[a]
    b = ids_b[i]
    if b >= 0:
        return world_b[b]
    return -1


@wp.struct
class RowArrays:
    ids_a: wp.array[int]
    world_a: wp.array[wp.int32]
    ids_b: wp.array[int]
    world_b: wp.array[wp.int32]
    active_count: wp.array[int]
    has_active: int
    W: wp.array[float]
    u: wp.array[wp.vec3]
    lam: wp.array[wp.vec3]
    Jv: wp.array[wp.vec3]
    u_prev: wp.array[wp.vec3]
    lam_prev: wp.array[wp.vec3]
    Jv_prev: wp.array[wp.vec3]
    u_frz: wp.array[wp.vec3]
    lam_frz: wp.array[wp.vec3]
    Jv_frz: wp.array[wp.vec3]


@wp.func
def _row_live(r: RowArrays, i: int) -> bool:
    if r.has_active != 0:
        return i < r.active_count[0]
    return True


@wp.kernel(enable_backward=False)
def stash_rows_kernel(r: RowArrays):
    i = wp.tid()
    r.u_prev[i] = r.u[i]
    r.lam_prev[i] = r.lam[i]
    r.Jv_prev[i] = r.Jv[i]


@wp.kernel(enable_backward=False)
def row_residual_kernel(r: RowArrays, rho: float, world_converged: wp.array[wp.int32], primal: wp.array[float],
                        dual: wp.array[float], scale_p: wp.array[float], scale_d: wp.array[float]):
    i = wp.tid()
    if not _row_live(r, i):
        return
    w = _row_world(i, r.ids_a, r.world_a, r.ids_b, r.world_b)
    if w < 0 or world_converged[w] != rs.ACTIVE:
        return
    wt = r.W[i]
    wp.atomic_max(primal, w, wt * wp.length(r.Jv[i] - r.u[i]))
    wp.atomic_max(dual, w, rho * wt * wp.length(r.u[i] - r.u_prev[i]))
    wp.atomic_max(scale_p, w, wt * wp.max(wp.length(r.Jv[i]), wp.length(r.u[i])))
    wp.atomic_max(scale_d, w, wt * wp.length(r.lam[i]))


@wp.kernel(enable_backward=False)
def freeze_rows_kernel(r: RowArrays, world_converged: wp.array[wp.int32]):
    """Newly converged worlds: keep the post-update rows aside; frozen and newly converged: restore pre-update."""
    i = wp.tid()
    w = _row_world(i, r.ids_a, r.world_a, r.ids_b, r.world_b)
    if w < 0:
        return
    state = world_converged[w]
    if state == rs.JUST_CONVERGED:
        r.u_frz[i] = r.u[i]
        r.lam_frz[i] = r.lam[i]
        r.Jv_frz[i] = r.Jv[i]
    if state != rs.ACTIVE:
        r.u[i] = r.u_prev[i]
        r.lam[i] = r.lam_prev[i]
        r.Jv[i] = r.Jv_prev[i]


@wp.kernel(enable_backward=False)
def unfreeze_rows_kernel(r: RowArrays, world_converged: wp.array[wp.int32]):
    i = wp.tid()
    w = _row_world(i, r.ids_a, r.world_a, r.ids_b, r.world_b)
    if w >= 0 and world_converged[w] == rs.FROZEN:
        r.u[i] = r.u_frz[i]
        r.lam[i] = r.lam_frz[i]
        r.Jv[i] = r.Jv_frz[i]


@wp.kernel(enable_backward=False)
def row_work_kernel(r: RowArrays, dt: float, quantum: float, work_q: wp.array[wp.int64]):
    i = wp.tid()
    if not _row_live(r, i):
        return
    w = _row_world(i, r.ids_a, r.world_a, r.ids_b, r.world_b)
    if w >= 0:
        e = dt * r.W[i] * wp.dot(r.lam[i], r.Jv[i])
        wp.atomic_add(work_q, w, wp.int64(wp.round(e / quantum)))


@wp.kernel(enable_backward=False)
def admm_convergence_kernel(k_min: int, eps_abs: float, eps_rel: float, iteration: wp.array[wp.int32],
                            primal: wp.array[float], dual: wp.array[float], scale_p: wp.array[float],
                            scale_d: wp.array[float], world_converged: wp.array[wp.int32],
                            iters_used: wp.array[wp.int32], n_unconverged: wp.array[wp.int32]):
    w = wp.tid()
    if world_converged[w] != rs.ACTIVE:
        return
    iters_used[w] = iters_used[w] + 1
    ok = (iteration[0] + 1 >= k_min and primal[w] <= eps_abs + eps_rel * scale_p[w]
          and dual[w] <= eps_abs + eps_rel * scale_d[w])
    if ok:
        world_converged[w] = rs.JUST_CONVERGED
    else:
        wp.atomic_add(n_unconverged, 0, 1)


@wp.kernel(enable_backward=False)
def add_work_kernel(quantum: float, work_q: wp.array[wp.int64], interface_work: wp.array[wp.float64]):
    w = wp.tid()
    interface_work[w] = interface_work[w] + wp.float64(work_q[w]) * wp.float64(quantum)


def row_groups(solver: Any) -> list[RowGroup]:
    """Every ADMM row group of a SolverCoupledADMM with endpoint world maps (as in its _reset_admm_history)."""
    out: list[RowGroup] = []
    rr = (*solver._admm_rr_groups, *solver._admm_rr_angular_groups, *solver._admm_rr_revolute_angular_groups,
          *solver._admm_rr_angular_friction_groups, *solver._admm_dynamic_rr_contact_groups)
    for g in rr:
        ea, eb = solver._entries[g.body_entry_name_a], solver._entries[g.body_entry_name_b]
        out.append(RowGroup(g, g.body_ids_a, ea.view.body_world, g.body_ids_b, eb.view.body_world))
    for g in (*solver._admm_rp_groups, *solver._admm_dynamic_rp_contact_groups):
        eb, ep = solver._entries[g.body_entry_name], solver._entries[g.particle_entry_name]
        out.append(RowGroup(g, g.body_ids, eb.view.body_world, g.particle_ids, ep.view.particle_world))
    for g in solver._admm_dynamic_pp_contact_groups:
        ea, eb = solver._entries[g.particle_entry_name_a], solver._entries[g.particle_entry_name_b]
        out.append(RowGroup(g, g.particle_ids_a, ea.view.particle_world, g.particle_ids_b, eb.view.particle_world))
    device = solver.model.device
    for rg in out:
        for key in ("u", "lambda_", "Jv"):
            rg.prev[key] = wp.zeros(rg.count, dtype=wp.vec3, device=device)
            rg.frozen[key] = wp.zeros(rg.count, dtype=wp.vec3, device=device)
    return out


def row_arrays(rg: RowGroup, dummy_count: wp.array) -> RowArrays:
    r = RowArrays()
    g = rg.group
    r.ids_a, r.world_a, r.ids_b, r.world_b = rg.ids_a, rg.world_a, rg.ids_b, rg.world_b
    r.active_count = rg.active_count if rg.active_count is not None else dummy_count
    r.has_active = int(rg.active_count is not None)
    r.W, r.u, r.lam, r.Jv = g.W, g.u, g.lambda_, g.Jv
    r.u_prev, r.lam_prev, r.Jv_prev = rg.prev["u"], rg.prev["lambda_"], rg.prev["Jv"]
    r.u_frz, r.lam_frz, r.Jv_frz = rg.frozen["u"], rg.frozen["lambda_"], rg.frozen["Jv"]
    return r


def _velocity_buffers(solver: Any) -> tuple[dict[str, wp.array], dict[str, wp.array]]:
    """ADMM's per-entry ``*_qd_k`` buffers (read by the proximal shift and Jv refresh of the next iteration) with
    their entry-view item->world maps."""
    from alloy._vendor import newton

    from alloy.core.worldops import item_world_for

    freq = newton.Model.AttributeFrequency
    arrays: dict[str, wp.array] = {}
    maps: dict[str, wp.array] = {}
    for name, entry in solver._entries.items():
        buf = solver._admm_buffers[name]
        for kind, frequency in (("body", freq.BODY), ("particle", freq.PARTICLE), ("joint", freq.JOINT_DOF)):
            arr = getattr(buf, f"{kind}_qd_k", None)
            world = item_world_for(entry.view, frequency) if arr is not None else None
            if arr is not None and arr.size and world is not None and world.shape[0] == arr.shape[0]:
                arrays[f"admm.{name}.{kind}_qd_k"], maps[f"admm.{name}.{kind}_qd_k"] = arr, world
    return arrays, maps


class AdmmIterationLoop:
    """Per-world convergence state and the iteration loop for a ``SolverCoupledADMM`` subclass."""

    def __init__(self, solver: Any, cfg: AdmmIterationConfig, capture_while: bool = False) -> None:
        self.solver, self.cfg, self.capture_while = solver, cfg, capture_while
        self.device = solver.model.device
        w = int(solver.model.world_count)
        self.world_count = w
        self.groups = row_groups(solver)
        self._dummy = wp.zeros(1, dtype=int, device=self.device)
        self.rows = [row_arrays(rg, self._dummy) for rg in self.groups]
        z = {"dtype": float, "device": self.device}
        self.primal, self.dual = wp.zeros(w, **z), wp.zeros(w, **z)
        self.scale_p, self.scale_d = wp.zeros(w, **z), wp.zeros(w, **z)
        self.res_pos = wp.zeros(w, **z)                 # ADMM has no source-proxy gap: stays 0
        self.world_converged = wp.zeros(w, dtype=wp.int32, device=self.device)
        self.iters_used = wp.zeros(w, dtype=wp.int32, device=self.device)
        self.start_frozen = wp.zeros(w, dtype=wp.int32, device=self.device)
        self.n_unconverged = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.iteration = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.condition = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.work_q = wp.zeros(w, dtype=wp.int64, device=self.device)
        extra, maps = _velocity_buffers(solver)
        self.freeze = SolverStateFreeze(solver.model, {n: e.solver for n, e in solver._entries.items()},
                                        enabled=cfg.k_min < cfg.k_max, extra=extra, extra_world=maps)
        self.captured_passes = CapturedPasses(w, self.device, enabled=cfg.k_min < cfg.k_max and not capture_while,
                                              passes=cfg.k_min)       # lever #1: capture the passes that run
        log.info("ADMM loop: %d row groups, k %d..%d", len(self.groups), cfg.k_min, cfg.k_max)

    def run(self, state_in: Any, control: Any, contacts: Any, dt: float, interface_work: wp.array) -> None:
        """One exchange: Newton's ADMM prologue, iteration 0, then residual-gated iterations 1..k_max-1."""
        s = self.solver
        s._refresh_collision_contact_groups(state_in)
        if float(s._coupling.gamma) > 0.0:
            s._refresh_admm_proximal_masks()
            s._refresh_admm_proximal_view_overrides(refresh_supported_solvers=True)
        self._snapshot_inputs()
        s._admm_begin_step(dt)
        wp.launch(rs.begin_exchange_kernel, dim=self.world_count,
                  inputs=[self.world_converged, self.iters_used, self.start_frozen], device=self.device)
        self.iteration.zero_()
        self._iterate(state_in=state_in, control=control, contacts=contacts, dt=dt, restart=False)
        extra = self.cfg.k_max - 1
        capturing = cpm.capturing(self.device)
        if extra > 0 and (self.cfg.k_min >= self.cfg.k_max or (capturing and not self.capture_while)):
            cp, k_min, k_max = self.captured_passes, self.cfg.k_min, self.cfg.k_max
            for _ in range(cp.captured(k_min, k_max) - 1):
                self._iterate(state_in=state_in, control=control, contacts=contacts, dt=dt, restart=True)
            if cp.capped(k_min, k_max):
                cp.mark(self.world_converged)
        elif extra > 0:
            wp.capture_while(self.condition, self._iterate, state_in=state_in, control=control, contacts=contacts,
                             dt=dt, restart=True)
        for r, rg in zip(self.rows, self.groups):
            wp.launch(unfreeze_rows_kernel, dim=rg.count, inputs=[r, self.world_converged], device=self.device)
        self.freeze.finish(self.world_converged)
        self._accumulate_work(dt, interface_work)

    def _snapshot_inputs(self) -> None:
        for name, entry in self.solver._entries.items():
            buf = self.solver._admm_buffers[name]
            for kind in ("body", "particle", "joint"):
                q_n = getattr(buf, f"{kind}_q_n")
                if q_n is None:
                    continue
                state = entry.state_0
                wp.copy(q_n, getattr(state, f"{kind}_q"))
                wp.copy(getattr(buf, f"{kind}_qd_n"), getattr(state, f"{kind}_qd"))
                wp.copy(getattr(buf, f"{kind}_qd_k"), getattr(state, f"{kind}_qd"))

    def _iterate(self, state_in: Any, control: Any, contacts: Any, dt: float, restart: bool) -> None:
        s = self.solver
        k = 1 if restart else 0                          # only k == 0 vs k > 0 matters to Newton's helpers
        self.freeze.begin_pass(self.world_converged)
        for name, entry in s._entries.items():
            s._prepare_admm_iteration_state(entry, s._admm_buffers[name], state_in, dt, iteration_restart=restart)
        s._accumulate_admm_forces(k, dt, refresh_jv=not restart, initialize_contact_u=not restart)
        for name, entry in s._entries.items():
            s._apply_admm_force_inputs(entry, s._admm_buffers[name], dt)
        for entry in s._entries.values():
            s._step_entry(entry, control, contacts, dt)
        for name, entry in s._entries.items():
            buf = s._admm_buffers[name]
            for kind in ("body", "particle", "joint"):
                qd_k = getattr(buf, f"{kind}_qd_k")
                if qd_k is not None:
                    wp.copy(qd_k, getattr(entry.state_1, f"{kind}_qd"))
        for r, rg in zip(self.rows, self.groups):
            wp.launch(stash_rows_kernel, dim=rg.count, inputs=[r], device=self.device)
        s._update_admm_dual(k, dt)
        self._after_iteration()

    def _after_iteration(self) -> None:
        dev, rho = self.device, float(self.solver._coupling.rho)
        for a, b, c in ((self.primal, self.dual, self.scale_p), (self.scale_d, self.scale_d, self.scale_d)):
            wp.launch(rs.reset_active_residuals_kernel, dim=self.world_count, inputs=[self.world_converged, a, b, c],
                      device=dev)
        for r, rg in zip(self.rows, self.groups):
            wp.launch(row_residual_kernel, dim=rg.count,
                      inputs=[r, rho, self.world_converged, self.primal, self.dual, self.scale_p, self.scale_d],
                      device=dev)
        self.n_unconverged.zero_()
        wp.launch(admm_convergence_kernel, dim=self.world_count,
                  inputs=[self.cfg.k_min, self.cfg.eps_abs, self.cfg.eps_rel, self.iteration, self.primal, self.dual,
                          self.scale_p, self.scale_d, self.world_converged, self.iters_used, self.n_unconverged],
                  device=dev)
        self.freeze.capture_converged(self.world_converged)
        for r, rg in zip(self.rows, self.groups):
            wp.launch(freeze_rows_kernel, dim=rg.count, inputs=[r, self.world_converged], device=dev)
        wp.launch(rs.settle_new_worlds_kernel, dim=self.world_count, inputs=[self.world_converged], device=dev)
        wp.launch(rs.advance_loop_kernel, dim=1,
                  inputs=[self.cfg.k_max, self.n_unconverged, self.iteration, self.condition], device=dev)

    def _accumulate_work(self, dt: float, interface_work: wp.array) -> None:
        self.work_q.zero_()
        for r, rg in zip(self.rows, self.groups):
            wp.launch(row_work_kernel, dim=rg.count, inputs=[r, float(dt), WORK_QUANTUM, self.work_q],
                      device=self.device)
        wp.launch(add_work_kernel, dim=self.world_count, inputs=[WORK_QUANTUM, self.work_q, interface_work],
                  device=self.device)

    def state_arrays(self) -> dict[str, wp.array]:
        out: dict[str, wp.array] = {}
        for n, rg in enumerate(self.groups):
            out[f"admm{n}.u"], out[f"admm{n}.lambda"] = rg.group.u, rg.group.lambda_
        return out
