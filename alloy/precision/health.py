# SPDX-License-Identifier: Apache-2.0
"""HealthMonitor: one ``StepHealth`` record per world, every check step, failing closed (SPEC §3.7, §4.3, D11).

``update`` launches kernels only (bar START_PEN's first judge); ``latch`` restores INVALID worlds from the last-good
``WorldSnapshot`` every step, so a latched world stays frozen and every other world is bitwise untouched (the caller
owns saving that snapshot: FakeSimulation saves every step, WP2's Simulation every period and also rolls back from
it); ``poll`` is the one host sync and raises ``PrecisionError`` under ``on_invalid="raise"``.

Two records: ``health`` is the per-step inbox the GapMonitor writes into before ``update`` (its per-pair
PAIR_PEN_WARN/FAIL, tunnel and overflow bits, ``max_pair_pen_m``, ``min_pair_gap_m``; D11); ``update`` folds it into
``record`` (the complete per-world StepHealth, what ``poll`` reports) and resets the inbox. The value-based pair
check uses the loosest pair limits, so with several pairs it can never latch a world the GapMonitor's exact per-pair
classification would not.

The record's latest values describe the latest check. A latched world is held at its last good pose, so they read
clean after its failure. The record therefore also keeps a first-failure snapshot (``fail_*``, ``health_failure``):
the values of the check that first made the world INVALID, with the deepest contact's shapes / particle, the pair id
and, when the caller passes ``dt`` to ``update``, the sim time. ``poll`` reports it as ``HealthReport.first_failure``;
summaries use ``HealthReport.failure_text``.

Check levels: ``cheap`` (every ``every`` steps) = finiteness, overflow sources, speed, |p|, rigid and soft contact
penetration (from a Newton ``Contacts``, and from every SolverMuJoCo's own MuJoCo Warp contacts, which on the native
contact route are the only ones, under ``health_pen``'s rule; a world no contact source covered reports ``max_pen_m``
NaN and ``pen_measured`` 0: SCR-SC0-1), precision-pair penetration (GapMonitor, D11), solver iterations, and energy
when enabled; ``full`` (every ``full_every`` steps) adds joint and weld violation. Energy (D20) is checked
for passive scenes (no joint drives, MuJoCo actuators or kinematic bodies) unless ``energy="on"``, in which case the
caller reports external work with ``add_external_work`` each step; ``energy="off"`` disables it. Work arrays registered
on the model (``health_work.register_work``: force plug-ins) are booked every step in either case.

Graph capture (PeriodGraph): scratch is preallocated; ``update``/``latch`` launch kernels only (run the first period
eagerly: START_PEN reads the host once there). Call ``reset_reference(state)`` before capturing; keep ``every = 1``
with ``full_every`` dividing the substeps per period: the check schedule is decided on the host from ``step``.
``first_bad_step`` comes from a device counter that every ``update`` call bumps (starting at ``start_step``), so it
stays right across graph replays; callers call ``update`` once per step.

Thin-part rule (coordinator ruling 2026-11-05 02:00Z and refinement): a rigid contact deeper than half the thinner
solid is PEN_FAIL. A solid is every colliding shape of one rigid body (a static shape: its decomposition parent),
and its thickness is its width along the contact normal (``health_source_shapes``). Soft contacts keep the shape's
own thickness.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.contact.types import PrecisionPairSpec
from alloy.core.errors import PrecisionError
from alloy.core.registry import HealthSample, health_adapter_for
from alloy.core.snapshot import WorldSnapshot
from alloy.precision import health_capacity as hc
from alloy.precision import health_joint
from alloy.precision.health_energy_gain import EnergyGain, energy_ref
from alloy.precision import health_failure as hf
from alloy.precision import health_kernels as hk
from alloy.precision import health_native
from alloy.precision.health_work import book_work
from alloy.precision import health_physics as hp
from alloy.precision import health_reset
from alloy.precision.config import far_is_invalid
from alloy.precision.determinism import effective_determinism, enable_solver_determinism, notice_floor, unwrap_solvers
from alloy.precision.health_adapters import (
    CapacitySource, contact_sources, mjw_contact_sources, register_builtin_adapters, solver_sources, user_sources,
)
from alloy.precision.health_cut_front import cut_front_growth
from alloy.precision.health_soft_pen import SoftPen
from alloy.precision.cloth_stretch import RangeChecks
from alloy.precision.self_contact import report_fields as sc_fields, state_arrays as sc_state_arrays
from alloy.precision.health_start import StartChecks
from alloy.precision.health_plugin import PluginChecks, ShareCheckMixin
from alloy.precision.health_setup import (
    WeldLimits, coord_worlds, energy_model, energy_scale_floor, shape_thickness, unjudged_world, weld_table,
)
from alloy.precision.health_source_shapes import build_solids
from alloy.precision.ladder import Thresholds, thresholds
from alloy.precision.types import (
    CheckPolicy, DepthCheck, Determinism, FramePolicy, HealthFlags, HealthReport,
    PrecisionConfig, PrecisionObservables, Severity, StepHealth,
)

log = logging.getLogger("alloy.precision")

EnergyMode = Literal["auto", "on", "off"]


def is_passive(model: newton.Model) -> bool:
    """No joint drives, MuJoCo actuators or kinematic bodies: total energy can only fall (D20)."""
    if model.joint_dof_count and model.joint_target_mode is not None and model.joint_target_mode.numpy().any():
        return False
    mj = getattr(model, "mujoco", None)
    trn = getattr(mj, "actuator_trntype", None)
    if trn is not None and trn.shape[0] > 0:
        return False
    if model.body_count and model.body_flags is not None:
        return not bool((model.body_flags.numpy() & int(newton.BodyFlags.KINEMATIC)).any())
    return True


def _initial_health(world_count: int, device) -> wp.array:
    init = wp.zeros(world_count, dtype=StepHealth, device="cpu").numpy()
    init["first_bad_step"] = -1
    init["residual"] = np.nan
    init["min_pair_gap_m"] = np.inf
    init["max_pen_m"] = np.nan  # not measured until a check reads a contact source
    init["max_pair_id"] = -1
    init["plugin_check"] = -1
    for name in ("fail_time_s", "fail_pen_m", "fail_pair_pen_m", "fail_speed_mps", "fail_joint_err_m",
                 "fail_joint_err_rad", "fail_energy_j", "fail_plugin_depth_m"):
        init[name] = np.nan  # first-failure snapshot: none yet
    for name in ("fail_shape_a", "fail_shape_b", "fail_particle", "fail_pair_id", "fail_body", "fail_plugin_check"):
        init[name] = -1
    return wp.array(init, dtype=StepHealth, device=device)


def _accum(world_count: int, device) -> hk.Accum:
    acc = hk.Accum()
    for name, dtype in (("bad_q", wp.int32), ("bad_qd", wp.int32), ("bad_particle", wp.int32),
                        ("overflow", wp.uint32), ("speed", wp.float32), ("abs_pos", wp.float32), ("arg", wp.uint64),
                        ("pen", wp.float32), ("pen_fail", wp.int32), ("pen_src", wp.int32),
                        ("pen_native", wp.float32), ("pen_native_fail", wp.int32), ("pen_arg", wp.uint64),
                        ("pen_shape_a", wp.int32), ("pen_shape_b", wp.int32), ("pen_particle", wp.int32),
                        ("cut_front", wp.float32), ("cut_front_hit", wp.int32),
                        ("plug_check", wp.int32), ("plug_depth", wp.float32),
                        ("joint_m", wp.float32), ("stretch_fail", wp.int32), ("pen_native_flag", wp.float32),
                        ("joint_rad", wp.float32), ("joint_fail", wp.int32), ("iters", wp.int32),
                        ("residual", wp.float32), ("energy", wp.float64)):
        setattr(acc, name, wp.zeros(world_count, dtype=dtype, device=device))
    return acc


def _limits(t: Thresholds, check_energy: bool, static_flags: int, native_pen_flags: bool = False) -> hk.Limits:
    lim = hk.Limits()
    lim.pen_warn, lim.pen_fail, lim.speed_fail = t.pen_warn_m, t.pen_fail_m, t.speed_fail_mps
    lim.pair_warn, lim.pair_fail = min(t.pair_warn_m, 3.0e38), min(t.pair_fail_m, 3.0e38)
    lim.energy_jump, lim.energy_s1 = t.energy_jump_rel, t.energy_s1_rel
    lim.r_warn, lim.r_fail = t.r_warn_m, t.r_fail_m
    lim.far_invalid = int(t.far_invalid)
    lim.check_energy = int(check_energy)
    lim.static_flags = wp.uint32(static_flags)
    lim.native_pen_flags = int(native_pen_flags)
    return lim


class HealthMonitor(ShareCheckMixin):
    """``HealthLike``: per-world health records for one Model (all worlds of one pack)."""

    def __init__(self, model: newton.Model, cfg: CheckPolicy, *, precision_pairs: Sequence[PrecisionPairSpec] = (),
                 capacity_flags: Sequence[wp.array | tuple[wp.array, int]] = (),
                 solvers: Sequence[newton.solvers.SolverBase] = (), config: PrecisionConfig | None = None,
                 energy: EnergyMode = "auto", design_force_n: float = 0.0, start_step: int = 0,
                 body_active: wp.array | None = None, rigid_contacts_unused: bool = False) -> None:
        register_builtin_adapters()
        self.model, self.cfg, self.config = model, cfg, config
        self.world_count, self.device = int(model.world_count), model.device
        frames = config.frames if config is not None else FramePolicy()
        self.thresholds = thresholds(cfg, frames, precision_pairs)
        self.health = _initial_health(self.world_count, self.device)   # per-step inbox (GapMonitor writes, D11)
        self.record = _initial_health(self.world_count, self.device)   # complete records (poll, latch)
        self._acc = _accum(self.world_count, self.device)
        self._time = wp.full(self.world_count, np.nan, dtype=wp.float64, device=self.device)  # update(dt=...)
        self._latched, self._bad = (wp.zeros(self.world_count, dtype=wp.int32, device=self.device) for _ in range(2))
        self._thick = wp.array(shape_thickness(model), dtype=wp.float32, device=self.device)
        self._pair_mark = wp.zeros(max(model.shape_count, 1), dtype=wp.int32, device=self.device)   # health_pen
        self._shape_allow = wp.zeros(model.shape_count, dtype=wp.float32, device=self.device)  # soft on purpose
        self._solids, _ = build_solids(model, self.device)  # rigid thin-part rule: body solids along the normal
        self._cut_blade = wp.zeros(0, dtype=wp.int32, device=self.device)      # SCR-SC5-3, set_cut_front
        self._soft_world = model.particle_world                               # skip_soft_contacts
        self.ranges = RangeChecks(cfg, self.world_count, self.device)        # STRAIN_RANGE, CLOTH_STRETCH
        self._soft_pen = SoftPen.build(model)    # SCR-SOFT-3-1: soft bodies inside each other (None: < 2 per world)
        self._blade_half = wp.zeros(0, dtype=wp.float32, device=self.device)
        self._cut_now = wp.zeros(0, dtype=wp.float32, device=self.device)      # per particle: deepest this check
        self._cut_floor = wp.zeros(0, dtype=wp.float32, device=self.device)    # per particle: normal-contact limit
        self._cut_first = wp.zeros(0, dtype=wp.float32, device=self.device)    # per particle: growth baseline
        self._q_world, self._qd_world = coord_worlds(model)
        self._welds = weld_table(model, WeldLimits(cfg.joint_limit_m, cfg.joint_limit_rad, design_force_n))
        self._solvers = list(solvers)
        if not self._solvers and (model.body_count or model.particle_count):
            log.warning("HealthMonitor: no solvers given, so no solver's own overflow counters are read (MuJoCo's "
                        "dropped constraint rows and contacts, VBD's self-contact rows): pass solvers=[...]")
        self._sources = user_sources(capacity_flags, self.world_count)
        self._sources += [s for solver in self._solvers for s in solver_sources(solver, cfg=cfg)]
        self._mjw_contacts = mjw_contact_sources(self._solvers, model)
        self._note = hk.overflow_note(self._mjw_contacts, self.device)       # CONTACT_OVERFLOW's one line
        self._contact_sources: dict[tuple[int, int], list[CapacitySource]] = {}
        self._rigid_unused = bool(rigid_contacts_unused)  # SCR-JOB1-3: the caller discards the rigid contact stream
        if self._rigid_unused:
            log.info("HealthMonitor: rigid contact stream declared unused: rigid-count and reducer overflow ignored")
        self._adapters = [(s, fn) for s in self._solvers if (fn := health_adapter_for(s)) is not None]
        self._plugins = PluginChecks(self.world_count, self.device)          # SCR-SC11-1 plug-in depth checks
        self._plugins.discover(list(unwrap_solvers(self._solvers)))
        self._sample = HealthSample(wp.zeros(self.world_count, dtype=wp.int32, device=self.device),
                                    wp.full(self.world_count, np.nan, dtype=wp.float32, device=self.device))
        self.det_mode, self.det_reasons = self._determinism()
        downgraded = config is not None and self.det_mode < notice_floor(config, self.device)
        self._energy_on = "energy" not in cfg.skip and (energy == "on" or (energy == "auto" and is_passive(model)))
        static = int(HealthFlags.DET_DOWNGRADED) if downgraded else 0
        self._lim = _limits(self.thresholds, self._energy_on, static, bool(cfg.native_pen_flags))
        self._lim.pen_dwell_s = cfg.native_pen_dwell_s          # GAP-E1; native_pen_hard goes to health_native
        self._energy = energy_model(model)
        if body_active is not None:
            self.set_body_active(body_active)
        self._step = wp.array([start_step - 1], dtype=wp.int64, device=self.device)  # bumped by every update
        self._ref = energy_ref(self.world_count, self.device)
        self._scale_floor = wp.array(energy_scale_floor(model), dtype=wp.float64, device=self.device)
        self._all = wp.ones(self.world_count, dtype=wp.int32, device=self.device)
        self._z_scratch = wp.zeros(self.world_count, dtype=wp.float64, device=self.device)
        self._e_scratch = wp.zeros(self.world_count, dtype=wp.float64, device=self.device)
        self._empty = {t: wp.zeros(0, dtype=t, device=self.device) for t in (wp.transform, wp.spatial_vector, wp.vec3)}
        self._empty_f32 = wp.zeros(0, dtype=wp.float32, device=self.device)
        self._ref_ready = False
        self.gain = EnergyGain(model, self.world_count)     # PHYSA-1: a steady energy gain is ENERGY_JUMP too
        self._sources.append(self.gain.source())
        self.start = StartChecks(model, cfg, self.thresholds, self.world_count, self.device, self._thick)  # WP5-14
        self._world_reset = health_reset.WorldReset(self)         # build-time copies for reset_worlds (SCR-WP5-19)
        log.info("HealthMonitor: %d worlds, level %s, energy %s, %d capacity sources, %d adapters, det %s%s",
                 self.world_count, cfg.level, "on" if self._energy_on else "off", len(self._sources),
                 len(self._adapters), self.det_mode.name, " (downgraded)" if downgraded else "")

    def _determinism(self) -> tuple[Determinism, tuple[str, ...]]:
        """Enable what the config asks for on the live solvers (SolverMuJoCo's ordered route), then report."""
        if self.config is None:
            return Determinism.NONE, ("no PrecisionConfig given: determinism not evaluated",)
        enable_solver_determinism(self.config, self._solvers)
        mode, reasons = effective_determinism(self.config, self._solvers, self.world_count)
        return mode, tuple(reasons)

    # ----- per step -----
    def is_check_step(self, step: int) -> bool:
        """True if ``update(step, ...)`` checks anything. With ``every > 1`` a caller saves its last-good snapshot
        only after such a step, so a latch never restores an unchecked state."""
        return self.cfg.level != "off" and step % max(self.cfg.every, 1) == 0

    def update(self, step: int, state_in: newton.State, state_out: newton.State, contacts: newton.Contacts | None,
               obs: PrecisionObservables | None = None, dt: float | None = None) -> None:
        """Run the checks due at ``step`` on ``state_out``: kernels only, except START_PEN's one-time judge (host reads;
        skipped under graph capture, health_start). ``dt``: the step's length (NaN snapshot times when None)."""
        cfg = self.cfg
        wp.launch(hp.bump_step, dim=1, inputs=[self._step], device=self.device)
        if self._energy_on:                   # F4: applied forces' work (plug-ins) is external work, every step
            if not self._ref_ready:           # the reference first: this step's work counts from its state_in
                self.reset_reference(state_in)
            book_work(self.model, self._ref)
        if dt is not None:
            wp.launch(hf.advance_time, dim=self.world_count, inputs=[wp.float64(dt), self._time], device=self.device)
        if self.start.pending and not self.is_check_step(step):     # START_PEN: the start state, once (eager)
            self.start.first(state_in, contacts, self._mjw_contacts, self._thick, None)
        if not self.is_check_step(step):
            return
        if self._energy_on and not self._ref_ready:
            self.reset_reference(state_in)
        dev, W = self.device, self.world_count
        wp.launch(hk.reset_accum, dim=W, inputs=[self._acc], device=dev)
        self._check_state(state_out)
        if contacts is not None:
            self._check_contacts(contacts, state_out)
        health_native.check_native(self, state_out)         # MuJoCo's own contacts (GAP-E1)
        self._plugins.launch(self._acc)
        self._check_sources()
        if self._soft_pen is not None:        # PEN_FAIL: a soft body's particle inside another soft body
            self._soft_pen.launch(state_out.particle_q, self._acc, int(HealthFlags.PEN_FAIL))
        self.ranges.launch(state_out, self._acc)      # STRAIN_RANGE (soft tets), CLOTH_STRETCH (cloth triangles)
        self._check_solvers(obs)
        hold_joints = health_joint.measure(self, step, state_out)    # 1: not measured now, keep the last verdict
        if self._energy_on:
            self._world_energy(state_out, self._acc.energy)
            self.gain.launch(self._acc.energy, self._ref, dt)
        if self.start.pending:
            self.start.first(state_in, contacts, self._mjw_contacts, self._thick, self._acc)
        wp.launch(hk.classify, dim=W, inputs=[self._step, self._acc, self._lim, self._ref, self._latched, self.health,
                  self.record, self._note, hold_joints, np.inf if dt is None else float(dt) * max(cfg.every, 1)],
                  device=dev)
        wp.launch(hf.first_failure, dim=W, inputs=[self._step, self._time, self._acc, self.record], device=dev)
        self.start.apply(state_out, self.model.body_world, dt, self.record)     # SPIN; sticky START_PEN

    def _check_state(self, state: newton.State) -> None:
        m, dev = self.model, self.device
        if m.body_count:
            wp.launch(hk.check_bodies, dim=m.body_count, inputs=[state.body_q, state.body_qd, m.body_world,
                                                                  self._acc], device=dev)
        if m.joint_coord_count and state.joint_q is not None:
            wp.launch(hk.check_coords, dim=m.joint_coord_count, inputs=[state.joint_q, self._q_world,
                                                                         self._acc.bad_q], device=dev)
        if m.joint_dof_count and state.joint_qd is not None:
            wp.launch(hk.check_coords, dim=m.joint_dof_count, inputs=[state.joint_qd, self._qd_world,
                                                                       self._acc.bad_qd], device=dev)
        if m.particle_count and state.particle_q is not None:
            wp.launch(hk.check_particles, dim=m.particle_count,
                      inputs=[state.particle_q, state.particle_qd, m.particle_flags, m.particle_world, self._acc],
                      device=dev)

    def _check_contacts(self, contacts: newton.Contacts, state: newton.State) -> None:
        m, dev = self.model, self.device
        body_q = state.body_q if state.body_q is not None else self._empty[wp.transform]
        pen_fail = float(self.thresholds.pen_fail_m)
        if contacts.rigid_contact_max and m.shape_count:
            wp.launch(hk.check_rigid_contacts, dim=contacts.rigid_contact_max,
                      inputs=[contacts.rigid_contact_count, contacts.rigid_contact_shape0, contacts.rigid_contact_shape1,
                              contacts.rigid_contact_point0, contacts.rigid_contact_point1, contacts.rigid_contact_normal,
                              contacts.rigid_contact_margin0, contacts.rigid_contact_margin1, m.shape_body, m.shape_world,
                              self._thick, body_q, pen_fail, self._solids, m.shape_type, m.shape_scale,
                              m.shape_transform, self._acc], device=dev)
            wp.launch(hf.resolve_rigid, dim=self.world_count, device=dev,
                      inputs=[contacts.rigid_contact_shape0, contacts.rigid_contact_shape1, self._acc])
            self._mark_pen_measured()
        if getattr(contacts, "soft_contact_max", 0) and m.particle_count and m.shape_count:
            wp.launch(hk.check_soft_contacts, dim=contacts.soft_contact_max,
                      inputs=[contacts.soft_contact_count, contacts.soft_contact_particle, contacts.soft_contact_shape,
                              contacts.soft_contact_body_pos, contacts.soft_contact_normal, state.particle_q,
                              m.particle_radius, self._soft_world, m.shape_body, self._thick, self._cut_blade,
                              self._blade_half, self._cut_now, self._cut_floor, body_q, pen_fail, self._acc],
                      device=dev)
            wp.launch(hf.resolve_soft, dim=self.world_count, device=dev,
                      inputs=[contacts.soft_contact_particle, contacts.soft_contact_shape, self._acc])
            if self._cut_first.shape[0]:
                wp.launch(cut_front_growth, dim=m.particle_count,
                          inputs=[self._cut_blade, m.particle_world, float(self.cfg.cut_front_growth_m),
                                  self._cut_now, self._cut_floor, self._cut_first, self._acc], device=dev)
                wp.launch(hf.resolve_particle, dim=self.world_count, inputs=[self._acc], device=dev)
            self._mark_pen_measured()
        key = (id(contacts), contacts.rigid_contact_count.ptr)
        if key not in self._contact_sources:
            self._contact_sources[key] = contact_sources(contacts, rigid=not self._rigid_unused)
            self._contact_sources[key] += [s for solver in self._solvers for s in solver_sources(solver, contacts)
                                           if s.kind == "over" and not (self._rigid_unused and s.rigid)]
        for src in self._contact_sources[key]:
            self._launch_source(src)

    def _mark_pen_measured(self) -> None:
        wp.launch(hk.mark_pen_measured, dim=self.world_count, inputs=[self.world_count, self._acc], device=self.device)

    def _check_sources(self) -> None:
        for src in self._sources:
            self._launch_source(src)

    def _launch_source(self, src: CapacitySource) -> None:
        W, dev, acc = self.world_count, self.device, self._acc
        if src.kind == "over":
            wp.launch(hc.flag_if_over, dim=W, inputs=[src.array, src.index, src.cap, wp.uint32(src.flag), acc],
                      device=dev)
        elif src.kind == "bool":
            wp.launch(hc.flag_if_true, dim=W, inputs=[src.array, W, wp.uint32(src.flag), acc], device=dev)
        elif src.kind == "bitmask":
            wp.launch(hc.flag_bitmask, dim=W, inputs=[src.array, *src.bits, acc], device=dev)
        else:
            wp.launch(hc.flag_if_nonzero, dim=W, device=dev,
                      inputs=[src.array, int(src.per_world), wp.uint32(src.flag), int(src.invalid), acc])

    def _check_solvers(self, obs: PrecisionObservables | None) -> None:
        W, dev = self.world_count, self.device
        for solver, adapter in self._adapters:
            self._sample.iters.zero_()
            self._sample.residual.fill_(np.nan)
            adapter(solver, self._sample)
            wp.launch(hk.merge_sample, dim=W, inputs=[self._sample.iters, self._sample.residual, self._acc], device=dev)
        if obs is not None and getattr(obs, "residual", None) is not None and getattr(obs, "iterations", None) is not None:
            wp.launch(hk.merge_sample, dim=W, inputs=[obs.iterations, obs.residual, self._acc], device=dev)

    def _check_joints(self, state: newton.State) -> None:
        health_joint.launch(self, state)

    # ----- energy bookkeeping (D20) -----
    def _world_energy(self, state: newton.State, out: wp.array) -> None:
        e = self._empty
        body_q = state.body_q if state.body_q is not None else e[wp.transform]
        body_qd = state.body_qd if state.body_qd is not None else e[wp.spatial_vector]
        pq = state.particle_q if state.particle_q is not None else e[wp.vec3]
        pqd = state.particle_qd if state.particle_qd is not None else e[wp.vec3]
        jq = state.joint_q if state.joint_q is not None and state.joint_q.size else self._empty_f32
        hp.launch_world_energy(self._energy, body_q, body_qd, pq, pqd, jq, self._ref.z_ref, out, self.device)

    def forget_seen(self) -> None:
        """``seen_flags`` back to the flags on now, ``peak_pen_m`` to the depth now: a settled state is a new start."""
        wp.launch(hp.forget_seen, dim=self.world_count, inputs=[self.record], device=self.device)

    def reset_reference(self, state: newton.State, world_mask: wp.array | None = None) -> None:
        """Energy reference at a reset (z_ref, E0, E_scale) for the masked worlds (all if None)."""
        dev, W, mask = self.device, self.world_count, world_mask if world_mask is not None else self._all
        body_q = state.body_q if state.body_q is not None else self._empty[wp.transform]
        pq = state.particle_q if state.particle_q is not None else self._empty[wp.vec3]
        wp.launch(hp.lowest_height, dim=W, inputs=[self._energy, body_q, pq, self._z_scratch], device=dev)
        wp.launch(hf.masked_copy, dim=W, inputs=[self._z_scratch, mask, self._ref.z_ref], device=dev)
        self._world_energy(state, self._e_scratch)
        wp.launch(hp.init_energy_ref, dim=W, inputs=[self._e_scratch, self._scale_floor, mask, self._ref], device=dev)
        self.gain.rebase(mask, self._ref)
        self._ref_ready = True

    def relax_thin_part_rule(self, shape_ids: Sequence[int]) -> None:
        """Shapes that cut or press into others by design (a blade edge): held to ``pen_s1_m`` only (SCR-SC0-1)."""
        thick = self._thick.numpy()
        thick[np.asarray(list(shape_ids), dtype=np.int64)] = np.inf
        self._thick.assign(thick)

    def relax_spin(self, bodies: Sequence[int]) -> None:      # spinning by design (a wheel): never SPIN
        self.start.relax_spin(bodies)

    reset_worlds = health_reset.reset_worlds          # a world reset clears its health (SCR-WP5-19)

    def set_soft_strain(self, measure: Any) -> None:
        """Soft tet strain each check step (WP10's ``SoftStrain``: float32 [W]); a world past soft_strain_range gets
        STRAIN_RANGE (DEGRADED: outside the soft FEM model's range, not a blow-up). Set before any graph capture."""
        self.ranges.soft = measure

    def set_cut_front(self, cut_front_blade: wp.array, blade_half_thickness: wp.array) -> None:
        """SCR-SC5-3, from the topology controller (both live on device; set before any graph capture):
        ``cut_front_blade`` int32 [particle_count], the body of the Blade cutting a phase-1 bond this particle twins
        (else -1); ``blade_half_thickness`` float32 [body_count], that blade face's half-thickness. Soft contacts of
        such a particle with that body leave the PEN rule for ``cut_front_pen_m`` / ``cut_front_steps``. As re-ruled
        (19:05Z) they are held to 2 x half-thickness + particle radius, and may not get deeper than the larger of a
        normal contact's limit (11:30Z) and ``CheckPolicy.cut_front_growth_m`` past the twin's depth at its first
        cut-front check (``health_cut_front``); past either is PEN_FAIL, and every other contact stays under the PEN rule. The per-particle baselines are not
        rolled back with a world; they reset when the twin leaves the cut front."""
        m = self.model
        if cut_front_blade.dtype != wp.int32 or cut_front_blade.shape[0] != m.particle_count:
            raise ValueError(f"cut_front_blade must be int32 of length {m.particle_count}")
        if blade_half_thickness.dtype != wp.float32 or blade_half_thickness.shape[0] != m.body_count:
            raise ValueError(f"blade_half_thickness must be float32 of length {m.body_count}")
        self._cut_blade, self._blade_half = cut_front_blade, blade_half_thickness
        self._cut_now = wp.full(m.particle_count, -np.inf, dtype=wp.float32, device=self.device)
        self._cut_floor = wp.full(m.particle_count, -np.inf, dtype=wp.float32, device=self.device)
        self._cut_first = wp.full(m.particle_count, np.nan, dtype=wp.float32, device=self.device)

    def skip_soft_contacts(self, particles: Sequence[int]) -> None:
        """Do not judge these particles' soft contacts: their solver (SPH, MPM) never reads them (OSS-17's cup, 11-07)."""
        self._soft_world = unjudged_world(self.model.particle_world, np.asarray(particles, np.int64))

    def set_body_active(self, body_active: wp.array) -> None:
        """Bodies with ``body_active == 0`` (int8 [body_count], retired by a crumble) leave the energy sums
        (SCR-WP10-2); pair it with ``reset_reference(state, topology.step_events)`` after ``topology.apply``."""
        if body_active.dtype != wp.int8 or body_active.shape[0] < self.model.body_count:
            raise ValueError(f"body_active must be int8 of length >= {self.model.body_count}")
        self._energy.body_active = body_active

    def state_arrays(self) -> dict[str, wp.array]:
        """Per-world arrays a world's rollback must restore (all world-major, [W]): records, latch state and the
        energy reference. The per-step accumulators are rebuilt every check and are not included."""
        out = {"health.record": self.record, "health.inbox": self.health, "health.latched": self._latched,
               "health.time": self._time}
        for name in ("z_ref", "e0", "e_max", "w_cum", "e_scale"):
            out[f"health.energy_{name}"] = getattr(self._ref, name)
        return {**out, **self.gain.state_arrays(), **sc_state_arrays(self._solvers)}

    def add_external_work(self, w_ext: wp.array) -> None:
        """Work done on each world by actuators and applied wrenches over the last step [J] (float64 [W])."""
        wp.launch(hp.add_work, dim=self.world_count, inputs=[w_ext, self._ref], device=self.device)

    # ----- latch and poll -----
    def latch(self, state_out: newton.State, snapshot: WorldSnapshot) -> None:
        """INVALID worlds: restore from ``snapshot`` (the caller's last good state) and freeze, sticky LATCHED."""
        wp.launch(hk.latch_masks, dim=self.world_count, inputs=[self.record, self._latched, self._bad],
                  device=self.device)
        snapshot.restore_selected(state_out, self._bad)

    def register_depth_check(self, name: str, depth: wp.array, limit_m: float) -> int:
        """SCR-SC11-1 (ruling 2026-11-04 18:25Z): a plug-in's per-world depth (float32 [world_count] in metres on
        this device; 0 or NaN: nothing crossed). Deeper than ``limit_m`` is PEN_FAIL (INVALID), named ``name`` in the
        record and the first-failure snapshot (``health_plugin``). Register before any graph capture. Returns the
        check's index."""
        return self._plugins.register(DepthCheck(name, depth, float(limit_m)))

    def poll(self) -> HealthReport:
        """Host copy of the records; raises ``PrecisionError`` on any INVALID world under ``on_invalid="raise"``."""
        h = self.record.numpy()
        sev = h["severity"].astype(np.int32)
        invalid = tuple(int(w) for w in np.flatnonzero(sev >= int(Severity.INVALID)))
        report = HealthReport(sev, h["flags"].astype(np.uint32), h["first_bad_step"].astype(np.int64),
                              h["max_pen_m"].astype(np.float64), h["max_pair_pen_m"].astype(np.float64),
                              h["max_speed_mps"].astype(np.float64), invalid, self.det_mode, self.det_reasons,
                              h["pen_measured"].astype(bool), h["cut_front_pen_m"].astype(np.float64),
                              h["cut_front_steps"].astype(np.int64), hf.first_failure_report(h, self._plugins.names),
                              **self.start.report_fields(), **self.ranges.report_fields(), **sc_fields(self._solvers),
                              seen_flags=h["seen_flags"].astype(np.uint32),
                              peak_pen_m=h["peak_pen_m"].astype(np.float64), **self.gain.report_fields(self._energy_on))
        if invalid and self.cfg.on_invalid == "raise":
            w = invalid[0]
            names = "|".join(f.name for f in HealthFlags if int(h["flags"][w]) & int(f))
            err = PrecisionError(f"{len(invalid)} INVALID world(s); world {w} at step {int(h['first_bad_step'][w])}: "
                                 f"{names}; first failure: {report.failure_text(w)}")
            err.report = report  # type: ignore[attr-defined]
            raise err
        return report


__all__ = ["HealthMonitor", "is_passive", "far_is_invalid"]
