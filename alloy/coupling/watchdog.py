# SPDX-License-Identifier: Apache-2.0
"""Energy watchdog: power bond, passive gain, watch states (SPEC §3.6, D3, D12, D20; coupling-time §4.4.5).

Per exchange and world:
- power bond ``ΔE_int += F_used·Δx_src − R·Δx_proxy + dt·(τ_used·ω_src − τ_R·ω_proxy)`` over proxy endpoints,
  where F_used is the feedback the source integrated and R the destination's harvest in the final pass. Δx are true
  displacements over the exchange, so the bond is exact for position updates ``x += h·v_end`` at any substep count;
  it is ≈ 0 for a consistent exchange and positive when the coupler injects energy (Sadjina et al. 2017);
- passive gain ``G = E − E_ref − W_ext`` (reported);
- states: 1 warn ``ΔE_int/E_scale > warn_gain_per_min·max(t, window)/60``; 2 escalated ``(ΔE_int − ΔE_int(t −
  window))/E_scale > escalate_gain_per_window`` or a final-pass source-proxy gap ``res_pos > res_pos_escalate``
  (10x the interface position tolerance: more passes do not close it, a finer clock does, coupling-time §3.2,
  §4.4.2); 4 failed: non-finite state or speed > ``abort_speed`` (D12).
  State 3 (rolled back) is written by the host. States only rise within a period.
- force jump (SCR-WP11-4): one exchange whose bond injects more than ``onset_limit`` x E_scale in a world the host
  armed (``onset_armed``: outside its cooldown) also escalates and sets ``onset_hit``;
  an escalation for the reasons above sets the sticky ``ladder_hit``, so ``onset_only`` never hides it. At first
  contact the lagged proxy books ``R^2 dt_x^2 / m`` (the destination's proxy runs a
  force-free step while the source decelerates under R); the Scheduler re-runs that period at a finer dt_x instead
  of climbing the escalation ladder. ``exchange_peak`` is the largest single-exchange injection of the attempt.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.coupling import energy as en
from alloy.coupling.endpoints import EndpointSet
from alloy.coupling.types import WatchState

log = logging.getLogger("alloy.coupling")


@wp.kernel(enable_backward=False)
def power_bond_body_kernel(
    csr_start: wp.array[wp.int32],
    csr_index: wp.array[wp.int32],
    src_local: wp.array[int],
    proxy_local: wp.array[int],
    x_begin: wp.array[wp.vec3],
    src_q: wp.array[wp.transform],
    src_qd: wp.array[wp.spatial_vector],
    dst_q: wp.array[wp.transform],
    dst_qd: wp.array[wp.spatial_vector],
    f_used: wp.array[wp.spatial_vector],
    f_raw: wp.array[wp.spatial_vector],
    staggered: int,
    dt: float,
    interface_work: wp.array[wp.float64],
):
    w = wp.tid()
    acc = wp.float64(0.0)
    for j in range(csr_start[w], csr_start[w + 1]):
        i = csr_index[j]
        s = src_local[i]
        p = proxy_local[i]
        x_src = wp.transform_get_translation(src_q[s])
        x_start = x_begin[i]
        if staggered != 0:
            x_start = x_src
        dx_src = x_src - x_begin[i]
        dx_proxy = wp.transform_get_translation(dst_q[p]) - x_start
        lin = wp.dot(wp.spatial_top(f_used[i]), dx_src) - wp.dot(wp.spatial_top(f_raw[i]), dx_proxy)
        ang = dt * (wp.dot(wp.spatial_bottom(f_used[i]), wp.spatial_bottom(src_qd[s]))
                    - wp.dot(wp.spatial_bottom(f_raw[i]), wp.spatial_bottom(dst_qd[p])))
        acc = acc + wp.float64(lin) + wp.float64(ang)
    interface_work[w] = interface_work[w] + acc


@wp.kernel(enable_backward=False)
def power_bond_particle_kernel(
    csr_start: wp.array[wp.int32],
    csr_index: wp.array[wp.int32],
    src_local: wp.array[int],
    proxy_local: wp.array[int],
    x_begin: wp.array[wp.vec3],
    src_q: wp.array[wp.vec3],
    dst_q: wp.array[wp.vec3],
    f_used: wp.array[wp.vec3],
    f_raw: wp.array[wp.vec3],
    staggered: int,
    interface_work: wp.array[wp.float64],
):
    w = wp.tid()
    acc = wp.float64(0.0)
    for j in range(csr_start[w], csr_start[w + 1]):
        i = csr_index[j]
        x_src = src_q[src_local[i]]
        x_start = x_begin[i]
        if staggered != 0:
            x_start = x_src
        lin = wp.dot(f_used[i], x_src - x_begin[i]) - wp.dot(f_raw[i], dst_q[proxy_local[i]] - x_start)
        acc = acc + wp.float64(lin)
    interface_work[w] = interface_work[w] + acc


@wp.struct
class WatchParams:
    warn_per_s: wp.float64          # fraction of E_scale per second of sim time
    warn_floor_s: wp.float64        # the warn allowance never drops below this many seconds
    escalate: wp.float64            # fraction of E_scale within one window
    abort_speed: float              # m/s
    res_pos_escalate: float         # m, final-pass |x_src - x_proxy| that escalates (unresolved interface)


@wp.kernel(enable_backward=False)
def watch_kernel(
    params: WatchParams,
    elapsed_ns: wp.array[wp.int64],
    energy: wp.array[wp.float64],
    energy_ref: wp.array[wp.float64],
    external_work: wp.array[wp.float64],
    interface_work: wp.array[wp.float64],
    window_base: wp.array[wp.float64],
    e_scale: wp.array[wp.float64],
    max_speed: wp.array[float],
    nonfinite: wp.array[wp.int32],
    res_pos: wp.array[float],
    work_before: wp.array[wp.float64],
    onset_limit: wp.array[wp.float64],
    exchange_peak: wp.array[wp.float64],
    onset_hit: wp.array[wp.int32],
    ladder_hit: wp.array[wp.int32],
    onset_armed: wp.array[wp.int32],
    gain: wp.array[wp.float64],
    watch_state: wp.array[wp.int32],
):
    w = wp.tid()
    gain[w] = energy[w] - energy_ref[w] - external_work[w]
    state = watch_state[w]
    if nonfinite[w] != 0 or not wp.isfinite(energy[w]) or max_speed[w] > params.abort_speed:
        watch_state[w] = 4
        return
    scale = e_scale[w]
    t = wp.float64(elapsed_ns[0]) * wp.float64(1.0e-9)
    injected = interface_work[w] / scale
    windowed = (interface_work[w] - window_base[w]) / scale
    bond = (interface_work[w] - work_before[w]) / scale        # this exchange's injection (SCR-WP11-4)
    exchange_peak[w] = wp.max(exchange_peak[w], bond)
    jump = bond > onset_limit[0] and onset_armed[w] != 0
    level = int(0)
    if windowed > params.escalate or res_pos[w] > params.res_pos_escalate:
        level = 2
        ladder_hit[w] = 1
    elif jump:
        level = 2
        onset_hit[w] = 1
    elif injected > params.warn_per_s * wp.max(t, params.warn_floor_s):
        level = 1
    if state == 3 and level == 2:
        state = 2                      # a rolled-back world that trips again in its re-run must ask for the next rung
    watch_state[w] = wp.max(state, level)


@wp.kernel(enable_backward=False)
def push_window_kernel(
    head: int,
    interface_work: wp.array[wp.float64],
    ring: wp.array2d[wp.float64],
    window_base: wp.array[wp.float64],
):
    """Period boundary: window_base = sample from ``len(ring)`` periods ago; store the current sample."""
    w = wp.tid()
    window_base[w] = ring[head, w]
    ring[head, w] = interface_work[w]


@wp.kernel(enable_backward=False)
def fill_ring_kernel(interface_work: wp.array[wp.float64], ring: wp.array2d[wp.float64]):
    r, w = wp.tid()
    ring[r, w] = interface_work[w]


@wp.kernel(enable_backward=False)
def clear_watch_kernel(watch_state: wp.array[wp.int32]):
    """New period: drop transient levels, keep FAILED (sticky until reset)."""
    w = wp.tid()
    if watch_state[w] != 4:
        watch_state[w] = 0


@wp.kernel(enable_backward=False)
def add_elapsed_kernel(ns: wp.int64, elapsed_ns: wp.array[wp.int64]):
    elapsed_ns[0] = elapsed_ns[0] + ns


@dataclass(frozen=True)
class WatchConfig:
    window: float                 # [s]
    warn_gain_per_min: float
    escalate_gain_per_window: float
    abort_speed: float            # [m/s]
    res_pos_escalate: float = math.inf   # [m] kinematic escalation (coupling-time §3.2: failure = source-proxy gap)


class EnergyWatchdog:
    """Owns the per-world energy, work and watch arrays of one Scheduler (W = model.world_count)."""

    def __init__(self, model: Any, cfg: WatchConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.device = model.device
        w = int(model.world_count)
        self.world_count = w
        f64 = {"dtype": wp.float64, "device": self.device}
        self.energy, self.energy_ref, self.e_scale = wp.zeros(w, **f64), wp.zeros(w, **f64), wp.zeros(w, **f64)
        self.z_ref, self.m_tot, self.l_char = wp.zeros(w, **f64), wp.zeros(w, **f64), wp.zeros(w, **f64)
        self.interface_work, self.external_work = wp.zeros(w, **f64), wp.zeros(w, **f64)
        self.window_base, self.gain = wp.zeros(w, **f64), wp.zeros(w, **f64)
        self.max_speed = wp.zeros(w, dtype=float, device=self.device)
        self.nonfinite = wp.zeros(w, dtype=wp.int32, device=self.device)
        self.watch_state = wp.zeros(w, dtype=wp.int32, device=self.device)
        self.elapsed_ns = wp.zeros(1, dtype=wp.int64, device=self.device)
        self.work_before, self.exchange_peak = wp.zeros(w, **f64), wp.zeros(w, **f64)      # SCR-WP11-4
        self.onset_hit = wp.zeros(w, dtype=wp.int32, device=self.device)
        self.ladder_hit = wp.zeros(w, dtype=wp.int32, device=self.device)
        self.onset_armed = wp.zeros(w, dtype=wp.int32, device=self.device)
        self.onset_limit = wp.full(1, math.inf, dtype=wp.float64, device=self.device)
        self.ring: wp.array | None = None
        self.head = 0
        self.inputs = self._energy_inputs(model)
        self.work_inputs = self._work_inputs(model)
        self.has_reference = False
        self._empty_sv = wp.empty(0, dtype=wp.spatial_vector, device=self.device)
        self._empty_v = wp.empty(0, dtype=wp.vec3, device=self.device)
        self._empty_f = wp.empty(0, dtype=float, device=self.device)

    def _energy_inputs(self, model: Any) -> en.EnergyInputs:
        return en.energy_inputs(model)

    def _work_inputs(self, model: Any) -> en.WorkInputs:
        return en.work_inputs(model)

    def ring_for(self, control_dt: float) -> None:
        """(Re)size the window ring to ceil(window / control_dt) periods."""
        n = max(1, int(math.ceil(self.cfg.window / control_dt - 1e-9)))
        if self.ring is None or self.ring.shape[0] != n:
            self.ring = wp.zeros((n, self.world_count), dtype=wp.float64, device=self.device)
            wp.copy(self.window_base, self.interface_work)
            self.head = 0

    def reset_reference(self, state: Any) -> None:
        """Datum, mass, extent, E₀ and E_scale from ``state``; clears work counters and watch states (D20)."""
        dev, w = self.device, self.world_count
        wp.launch(en.reference_kernel, dim=w, inputs=[self.inputs, state.body_q, state.particle_q,
                                                      self.z_ref, self.m_tot, self.l_char], device=dev)
        self._energy(state)
        wp.launch(en.scale_kernel, dim=w, inputs=[self.inputs.gravity, self.energy, self.m_tot, self.l_char,
                                                  self.energy_ref, self.e_scale], device=dev)
        for arr in (self.interface_work, self.external_work, self.window_base, self.gain, self.watch_state,
                    self.elapsed_ns):
            arr.zero_()
        if self.ring is not None:
            self.ring.zero_()
        self.head = 0
        self.has_reference = True

    def _energy(self, state: Any) -> None:
        en.launch_energy(self.inputs, state, self.z_ref, self.energy, self.max_speed, self.nonfinite, self.device)

    def accumulate_bond(self, es: EndpointSet, src: Any, dst: Any, dt: float) -> None:
        """Add one endpoint set's power bond for the final pass (source/destination entry end states)."""
        if es.count == 0:
            return
        if es.kind == "body":
            wp.launch(power_bond_body_kernel, dim=self.world_count,
                      inputs=[es.csr.start, es.csr.index, es.src_local, es.proxy_local, es.x_begin,
                              src.state_1.body_q, src.state_1.body_qd, dst.state_1.body_q, dst.state_1.body_qd,
                              es.f_used, es.f_raw, int(es.staggered), float(dt), self.interface_work],
                      device=self.device)
        else:
            wp.launch(power_bond_particle_kernel, dim=self.world_count,
                      inputs=[es.csr.start, es.csr.index, es.src_local, es.proxy_local, es.x_begin,
                              src.state_1.particle_q, dst.state_1.particle_q, es.f_used, es.f_raw,
                              int(es.staggered), self.interface_work], device=self.device)

    def after_exchange(self, state_in: Any, state_out: Any, control: Any, dt: float, res_pos: wp.array) -> None:
        """Energy, external work and watch states after the parent state is reconciled; ``res_pos`` [W] is the
        final-pass kinematic residual."""
        dev = self.device
        wp.launch(add_elapsed_kernel, dim=1, inputs=[wp.int64(round(dt * 1.0e9)), self.elapsed_ns], device=dev)
        self._external_work(state_in, state_out, control, dt)
        self._energy(state_out)
        cfg = self.cfg
        p = WatchParams()
        p.warn_per_s = wp.float64(cfg.warn_gain_per_min / 60.0)
        p.warn_floor_s = wp.float64(cfg.window)
        p.escalate = wp.float64(cfg.escalate_gain_per_window)
        p.abort_speed = float(cfg.abort_speed)
        p.res_pos_escalate = float(min(cfg.res_pos_escalate, 1.0e30))
        wp.launch(watch_kernel, dim=self.world_count,
                  inputs=[p, self.elapsed_ns, self.energy, self.energy_ref, self.external_work, self.interface_work,
                          self.window_base, self.e_scale, self.max_speed, self.nonfinite, res_pos, self.work_before,
                          self.onset_limit, self.exchange_peak, self.onset_hit, self.ladder_hit,
                          self.onset_armed, self.gain, self.watch_state],
                  device=dev)

    def _external_work(self, state_in: Any, state_out: Any, control: Any, dt: float) -> None:
        empty_sv, empty_v, empty_f = self._empty_sv, self._empty_v, self._empty_f
        joint_f = control.joint_f if control is not None and control.joint_f is not None else empty_f
        has_joint = joint_f.shape[0] > 0 and state_in.joint_qd is not None
        body = (state_in.body_f if state_in.body_f is not None else empty_sv, state_in.body_q, state_out.body_q,
                state_out.body_qd)
        particle = (state_in.particle_f if state_in.particle_f is not None else empty_v, state_in.particle_q,
                    state_out.particle_q)
        joint = tuple(a if has_joint else empty_f for a in (joint_f, state_in.joint_qd, state_out.joint_qd))
        en.launch_external_work(self.work_inputs, dt, body, particle, joint, self.external_work, self.device)

    def begin_exchange(self) -> None:
        """Device, before each exchange: the bond this exchange adds is measured from here (SCR-WP11-4)."""
        wp.copy(self.work_before, self.interface_work)

    def begin_attempt(self, onset_limit: float, armed: np.ndarray | None = None) -> None:
        """Host, before every run of a period (first or re-run): clear the force-jump peak and flags, set the
        single-exchange limit (fraction of E_scale; inf turns the trigger off for this attempt) and the worlds the
        trigger may flag (int32 [W]; None: all)."""
        self.exchange_peak.zero_()
        self.onset_hit.zero_()
        self.ladder_hit.zero_()
        self.onset_limit.fill_(float(onset_limit))
        if armed is None:
            self.onset_armed.fill_(1)
        else:
            self.onset_armed.assign(np.asarray(armed, dtype=np.int32))

    def onset_only(self) -> np.ndarray:
        """Host: worlds whose escalations this attempt were all force jumps (bool [W])."""
        return (self.onset_hit.numpy() != 0) & (self.ladder_hit.numpy() == 0)

    def begin_period(self, control_dt: float) -> None:
        """Host, once per period: advance the 1 s window and clear transient watch levels."""
        self.ring_for(control_dt)
        wp.launch(push_window_kernel, dim=self.world_count,
                  inputs=[self.head, self.interface_work, self.ring, self.window_base], device=self.device)
        self.head = (self.head + 1) % self.ring.shape[0]
        wp.launch(clear_watch_kernel, dim=self.world_count, inputs=[self.watch_state], device=self.device)

    def mark_rolled_back(self, world_mask: np.ndarray | None) -> None:
        """Host: record a rollback (state 3) for the masked worlds (all if None), keeping FAILED worlds."""
        ws = self.watch_state.numpy().copy()
        mask = np.ones_like(ws, dtype=bool) if world_mask is None else np.asarray(world_mask, dtype=bool)
        ws[mask & (ws != int(WatchState.FAILED))] = int(WatchState.ROLLED_BACK)
        self.watch_state.assign(ws)

    def mark_failed(self, world_mask: np.ndarray) -> None:
        """Host: set FAILED (sticky until reset) for the masked worlds."""
        ws = self.watch_state.numpy().copy()
        ws[np.asarray(world_mask, dtype=bool)] = int(WatchState.FAILED)
        self.watch_state.assign(ws)

    def rebase_window(self) -> None:
        """Host, after acting on an escalation: the 1 s window restarts from the current ΔE_int, so injection that
        was already handled (rolled back or answered with a finer clock) does not trip the next periods again."""
        wp.copy(self.window_base, self.interface_work)
        if self.ring is not None:
            wp.launch(fill_ring_kernel, dim=self.ring.shape, inputs=[self.interface_work, self.ring],
                      device=self.device)

    def state_arrays(self) -> dict[str, wp.array]:
        """Per-world (W,) arrays that a rollback must restore (the global elapsed clock is excluded)."""
        return {
            "watch.energy_ref": self.energy_ref, "watch.interface_work": self.interface_work,
            "watch.external_work": self.external_work, "watch.window_base": self.window_base,
            "watch.watch_state": self.watch_state,
        }
