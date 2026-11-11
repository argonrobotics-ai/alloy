# SPDX-License-Identifier: Apache-2.0
"""``BenchMonitors``: bench-only checks layered over ``StepHealth`` (D11, SPEC §3.11, bench §4.6).

Energy ledger (D20) over the **passive subsystem** (every dynamic body that is not robot, plus active particles):
E = KE + m·g·(z - z_min,0) per world in float64; unexplained gain per step = dE - W(applied wrenches, at the
midpoint velocity, plus the work robot shapes' boundaries did on a solver's particles, ``alloy_shape_work``,
SCR-WP9-6: a driven SPH cup); a step in which the robot touches a passive body is excused (a rigid contact, or
particles when no shape-work source books that work, e.g. a blade in a VBD solid). Reported per step and cumulative
over ``E_scale = max(E0, m_tot·g·L_char, 1e-3 J)`` (all dynamic bodies). In scenes without robot bodies the
solver's own ``alloy_energy`` (KE + PE + elastic, D20) replaces KE + PE when the runner passes it, so energy parked
in penalty contact springs is not read as a loss and then a gain. The robot's own drive work is not
estimated: Newton's XPBD drive moves a body ~4x faster in one step than a ``target_ke`` spring would, so no PD-force
quadrature gives its work (``[internal note]``). Plus non-target displacement
(VAL-116), teleport (CON-12), impossible motion (free body > v_max without robot contact), joint anchor separation,
a contact-buffer penetration fallback, non-finite state and max speed (so the bully classifies before WP5's
HealthMonitor is injected). Call ``update`` once per solver step (``observer``); once per control period also works
but sees robot contact only at period ends.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverBase

from alloy.bully import kernels as K
from alloy.bully.roles import KIND_FREE, KIND_NON_TARGET, KIND_ROBOT, BodyRoles, body_roles
from alloy.bully.types import BullyLimits
from alloy.core.defaults import E_SCALE_FLOOR_J
from alloy.core.protocols import ShapeWorkSource, StabilityEstimator

log = logging.getLogger("alloy.bully")

_F32_FIELDS = ("max_speed", "teleport", "nontarget_disp", "impossible_speed", "impossible_speed_d12",
               "contact_free_accel", "joint_sep", "max_pen", "pen_s1_ratio")


class BenchMonitors:
    """Per-world worst values since the last ``reset``; device kernels only in ``update``."""

    def __init__(self, model: newton.Model, limits: BullyLimits, non_target_bodies: Sequence[int] = (), *,
                 robot_bodies: Sequence[int] | None = None, thin_parts: Mapping[int, float] | None = None,
                 energy_source: StabilityEstimator | None = None, overflow: wp.array | None = None,
                 work_sources: Sequence[ShapeWorkSource] = ()) -> None:
        self.model, self.limits = model, limits
        self.overflow = overflow  # [W] int, the solver's per-step capacity bits (MJWarp Data.overflow), OR-ed per step
        dev, w = model.device, model.world_count
        roles = body_roles(model, non_target_bodies, robot_bodies, thin_parts)
        self._roles = roles
        self._kind = wp.array(roles.kind, dtype=wp.int32, device=dev)
        self._robot = wp.array((roles.kind & KIND_ROBOT != 0).astype(np.int32), dtype=wp.int32, device=dev)
        self._radius = wp.array(roles.radius, dtype=float, device=dev)
        self._thin = wp.array(roles.thin, dtype=float, device=dev)
        self._b_start = wp.array(roles.world_start, dtype=wp.int32, device=dev)
        self._b_index = wp.array(roles.world_index, dtype=wp.int32, device=dev)
        p_start, p_index = K.world_csr(_np(model.particle_world, model.particle_count), w)
        self._p_start = wp.array(p_start, dtype=wp.int32, device=dev)
        self._p_index = wp.array(p_index, dtype=wp.int32, device=dev)
        self._alloc(w, dev)
        self.work_sources = tuple(work_sources)
        self._shape_work = wp.zeros(max(int(model.shape_count), 1), dtype=wp.float64, device=dev)
        s_start, s_index = K.world_csr(_robot_shape_world(model, roles), w)
        self._rs_start = wp.array(s_start, dtype=wp.int32, device=dev)
        self._rs_index = wp.array(s_index, dtype=wp.int32, device=dev)
        self.steps = 0
        # a solver's own energy (D20: KE + PE + elastic, e.g. penalty contact springs) replaces KE + PE, but only in
        # scenes without robot bodies, whose drive work the ledger cannot measure
        self.energy_source = energy_source if not np.any(roles.kind & KIND_ROBOT) else None
        log.info("BenchMonitors: %d worlds, %d free / %d robot / %d non-target bodies, %d ledger bodies", w,
                 int(np.count_nonzero(roles.kind & KIND_FREE)), int(np.count_nonzero(roles.kind & KIND_ROBOT)),
                 int(np.count_nonzero(roles.kind & KIND_NON_TARGET)), roles.world_index.size)

    def _alloc(self, w: int, dev: str) -> None:
        m = self.model
        self.checks = K.BodyChecks()
        self.checks.nonfinite = wp.zeros(w, dtype=wp.int32, device=dev)
        for name in _F32_FIELDS:
            setattr(self.checks, name, wp.zeros(w, dtype=wp.float32, device=dev))
        f64 = ("_z_ref", "_e_scale", "_e_prev", "_e_now", "_w_force", "_cum")
        for name in f64:
            setattr(self, name, wp.zeros(w, dtype=wp.float64, device=dev))
        self._gain_step = wp.zeros(w, dtype=wp.float32, device=dev)
        self._gain_cum = wp.zeros(w, dtype=wp.float32, device=dev)
        self._touch_any = wp.zeros(m.body_count, dtype=wp.int32, device=dev)
        self._flight = K.FlightState()
        for name, dtype in (("exit_speed", wp.float32), ("free_t", wp.float32), ("touch_prev", wp.int32),
                            ("robot_prev", wp.int32)):
            setattr(self._flight, name, wp.zeros(m.body_count, dtype=dtype, device=dev))
        self._contacts_known = 0
        self._touch_robot = wp.zeros(m.body_count, dtype=wp.int32, device=dev)
        self._world_robot_touch = wp.zeros(w, dtype=wp.int32, device=dev)
        self._q_reset = wp.zeros(m.body_count, dtype=wp.transform, device=dev)
        self._overflow = wp.zeros(w, dtype=wp.int32, device=dev)

    # ------------------------------------------------------------------ reset (host sync allowed here)
    def reset(self, state: newton.State, control: newton.Control | None = None) -> None:
        """Baseline at the episode start: datum, E_scale, E0, reset poses; clears all maxima."""
        self._set_datum_and_scale(state)
        self._energy(state, self._e_prev)
        self._cum.zero_()
        self._gain_step.zero_()
        self._gain_cum.zero_()
        self.checks.nonfinite.zero_()
        for name in _F32_FIELDS:
            getattr(self.checks, name).zero_()
        if self.model.body_count:
            wp.copy(self._q_reset, state.body_q)
            speed = np.linalg.norm(state.body_qd.numpy()[:, :3], axis=1).astype(np.float32)
            self._flight.exit_speed.assign(speed)  # a body never in contact flies from its reset speed
        self._flight.free_t.zero_()
        self._touch_any.zero_()
        self._touch_robot.zero_()
        self._overflow.zero_()
        self.steps = 0

    def _set_datum_and_scale(self, state: newton.State) -> None:
        """D20 datum (lowest dynamic COM per world) and E_scale = max(E0, m_free·g·L_char, 1e-3 J): the mass that
        can carry energy in the ledger is the free bodies' and particles' (robot links and bodies welded to the world
        excluded); L_char is the extent of everything dynamic, robot reach included."""
        m, w = self.model, self.model.world_count
        p = _points(m, state, self._roles)
        z_ref, m_tot, l_char = np.zeros(w), np.zeros(w), np.zeros(w)
        for k in range(w):
            sel = p.world == k
            if not sel.any():
                continue
            z_ref[k] = p.x[sel, 2].min()
            m_tot[k] = p.mass[sel & p.free].sum()
            lo, hi = (p.x[sel] - p.radius[sel, None]).min(axis=0), (p.x[sel] + p.radius[sel, None]).max(axis=0)
            l_char[k] = float((hi - lo).max())
        self._z_ref.assign(z_ref)
        self._energy(state, self._e_now)
        g = np.linalg.norm(m.gravity.numpy()[:w], axis=1)
        scale = np.maximum(np.maximum(self._e_now.numpy(), m_tot * g * l_char), E_SCALE_FLOOR_J)
        self._e_scale.assign(scale)
        self.e_scale = scale

    # ------------------------------------------------------------------ per step (kernels only)
    def update(self, prev: newton.State, cur: newton.State, control: newton.Control | None,
               obs: SolverBase.Observables | None, dt: float, contacts: newton.Contacts | None = None) -> None:
        """Fold one step (or one period) ``prev -> cur`` of length ``dt`` into the per-world maxima."""
        del obs, control  # CONTACT_F friction loss (bench §4.6) is not used yet; the ledger is dE - W(wrenches)
        self._contacts(cur, contacts)
        self._body_checks(prev, cur, dt)
        self._energy(cur, self._e_now)
        self._force_work(prev, cur, dt)
        wp.launch(K.ledger_kernel, dim=self.model.world_count, device=self.model.device,
                  inputs=[self._e_now, self._e_prev, self._w_force, self._world_robot_touch, self._e_scale,
                          self._cum, self._gain_step, self._gain_cum])
        if self.overflow is not None:  # Newton clears Data.overflow before every step: fold it in after each one
            wp.launch(K.overflow_or_kernel, dim=self.model.world_count, device=self.model.device,
                      inputs=[self.overflow, K.CAPACITY_OVERFLOW_MASK, self._overflow])
        self.steps += 1

    def observer(self, state_in: newton.State, state_out: newton.State, control: newton.Control,
                 contacts: newton.Contacts | None, dt: float) -> None:
        """Per-substep hook signature used by ``DirectSim`` and the Simulation step-observer SCR."""
        self.update(state_in, state_out, control, None, dt, contacts)

    def _contacts(self, cur: newton.State, contacts: newton.Contacts | None) -> None:
        m = self.model
        if m.body_count:
            wp.copy(self._flight.touch_prev, self._touch_any)
            wp.copy(self._flight.robot_prev, self._touch_robot)
        self._contacts_known = int(contacts is not None)
        self._touch_any.zero_()
        self._touch_robot.zero_()
        self._world_robot_touch.zero_()
        if contacts is None or m.body_count == 0:
            return
        c = contacts
        if c.rigid_contact_max:
            wp.launch(K.contact_touch_kernel, dim=c.rigid_contact_max, device=m.device, inputs=[
                c.rigid_contact_count, c.rigid_contact_shape0, c.rigid_contact_shape1, m.shape_body, self._robot,
                m.body_world, self._touch_any, self._touch_robot, self._world_robot_touch])
            wp.launch(K.contact_pen_kernel, dim=c.rigid_contact_max, device=m.device, inputs=[
                c.rigid_contact_count, c.rigid_contact_shape0, c.rigid_contact_shape1, c.rigid_contact_point0,
                c.rigid_contact_point1, c.rigid_contact_normal, c.rigid_contact_margin0, c.rigid_contact_margin1,
                m.shape_body, m.shape_world, cur.body_q, self._thin, self.limits.pen_s1, self.checks])
        if getattr(c, "soft_contact_max", 0):
            wp.launch(K.soft_touch_kernel, dim=c.soft_contact_max, device=m.device, inputs=[
                c.soft_contact_count, c.soft_contact_shape, m.shape_body, self._robot, m.body_world,
                int(not self.work_sources), self._touch_any, self._world_robot_touch])

    def _body_checks(self, prev: newton.State, cur: newton.State, dt: float) -> None:
        m = self.model
        if m.body_count:
            wp.launch(K.body_checks_kernel, dim=m.body_count, device=m.device, inputs=[
                prev.body_q, prev.body_qd, cur.body_q, cur.body_qd, self._q_reset, m.body_com, m.body_world,
                self._kind, self._radius, self._touch_any, self._touch_robot, m.gravity, self.limits.v_max, dt,
                self.checks])
            wp.launch(K.impossible_speed_kernel, dim=m.body_count, device=m.device, inputs=[
                cur.body_qd, m.body_world, self._kind, self._touch_any, self._touch_robot, m.gravity,
                self._contacts_known, self.limits.v_max, self.limits.flight_margin, dt, self._flight, self.checks])
        if m.joint_count:
            wp.launch(K.joint_sep_kernel, dim=m.joint_count, device=m.device, inputs=[
                m.joint_type, m.joint_enabled, m.joint_parent, m.joint_child, m.joint_X_p, m.joint_X_c,
                m.joint_world, cur.body_q, self.checks])
        if m.particle_count:
            wp.launch(K.particle_checks_kernel, dim=m.particle_count, device=m.device, inputs=[
                cur.particle_q, cur.particle_qd, m.particle_flags, m.particle_world, self.checks])

    def _energy(self, state: newton.State, out: wp.array) -> None:
        m, w = self.model, self.model.world_count
        if self.energy_source is not None:
            self.energy_source.alloy_energy(state, self._z_ref, out)
            return
        out.zero_()
        if m.body_count:
            wp.launch(K.body_energy_kernel, dim=w, device=m.device, inputs=[
                state.body_q, state.body_qd, m.body_mass, m.body_inertia, m.body_com, m.gravity, self._z_ref,
                self._b_start, self._b_index], outputs=[out])
        if m.particle_count:
            wp.launch(K.particle_energy_kernel, dim=w, device=m.device, inputs=[
                state.particle_q, state.particle_qd, m.particle_mass, m.particle_flags, m.gravity, self._z_ref,
                self._p_start, self._p_index], outputs=[out])

    def _force_work(self, prev: newton.State, cur: newton.State, dt: float) -> None:
        """Work of the wrenches and particle forces applied over the step (in ``prev``, the step's input)."""
        m = self.model
        self._w_force.zero_()
        if m.body_count and prev.body_f is not None:
            wp.launch(K.body_force_work_kernel, dim=m.world_count, device=m.device, inputs=[
                prev.body_f, prev.body_qd, cur.body_qd, dt, self._b_start, self._b_index], outputs=[self._w_force])
        if m.particle_count and prev.particle_f is not None:
            wp.launch(K.particle_force_work_kernel, dim=m.world_count, device=m.device, inputs=[
                prev.particle_f, prev.particle_qd, cur.particle_qd, m.particle_flags, dt, self._p_start,
                self._p_index], outputs=[self._w_force])
        for src in self.work_sources:  # robot walls' work on a solver's particles (SCR-WP9-6)
            src.alloy_shape_work(self._shape_work)
            wp.launch(K.shape_work_kernel, dim=m.world_count, device=m.device,
                      inputs=[self._shape_work, self._rs_start, self._rs_index], outputs=[self._w_force])

    # ------------------------------------------------------------------ host report
    def report(self) -> dict[str, np.ndarray]:
        """Per-world worst values since reset: e_gain, e_gain_cum, nontarget_disp, contact_free_accel, ... ."""
        out = {name: getattr(self.checks, name).numpy().astype(np.float64) for name in _F32_FIELDS}
        out["nonfinite"] = self.checks.nonfinite.numpy().astype(np.float64)
        out["e_gain"] = self._gain_step.numpy().astype(np.float64)
        out["e_gain_cum"] = self._gain_cum.numpy().astype(np.float64)
        if self.overflow is not None:
            out["capacity_overflow"] = (self._overflow.numpy() != 0).astype(np.float64)
        return out


def shape_work_sources(solver: object) -> tuple[ShapeWorkSource, ...]:
    """The solver itself if it is a ``ShapeWorkSource``, else the ones among a stock coupled solver's entry solvers
    (``SolverCoupled._entries``, listed in ``PRIVATE_HOOKS``)."""
    if solver is None:
        return ()
    if isinstance(solver, ShapeWorkSource):
        return (solver,)
    entries = getattr(solver, "_entries", None) or {}
    subs = [getattr(e, "solver", None) for e in entries.values()]
    return tuple(s for s in subs if isinstance(s, ShapeWorkSource))


def _robot_shape_world(model: newton.Model, roles: BodyRoles) -> np.ndarray:
    """World of each shape on a robot body, -1 for every other shape (dropped by ``world_csr``)."""
    if not model.shape_count:
        return np.zeros(0, dtype=np.int32)
    body = model.shape_body.numpy()
    robot = (roles.kind & KIND_ROBOT) != 0
    on_robot = (body >= 0) & robot[np.maximum(body, 0)] if model.body_count else np.zeros(body.shape, dtype=bool)
    world = model.body_world.numpy()[np.maximum(body, 0)] if model.body_count else np.zeros_like(body)
    return np.where(on_robot, world, -1).astype(np.int32)


def _np(arr: wp.array | None, count: int) -> np.ndarray:
    return arr.numpy() if arr is not None and count else np.zeros(0, dtype=np.int32)


@dataclass(frozen=True)
class _Points:
    """Dynamic bodies and active particles at reset (host): COM, mass, world, radius, carries-E_scale-mass."""

    x: np.ndarray
    mass: np.ndarray
    world: np.ndarray
    radius: np.ndarray
    free: np.ndarray


def _points(m: newton.Model, state: newton.State, roles: BodyRoles) -> _Points:
    parts: list[tuple[np.ndarray, ...]] = []
    if m.body_count:
        q, com, mass = state.body_q.numpy(), m.body_com.numpy(), m.body_mass.numpy()
        dyn = mass > 0.0
        x = (q[:, :3] + _rotate(q[:, 3:7], com))[dyn]
        parts.append((x, mass[dyn], m.body_world.numpy()[dyn], roles.radius[dyn], (roles.kind & KIND_FREE != 0)[dyn]))
    if m.particle_count:
        act = (m.particle_flags.numpy() & 1).astype(bool) & (m.particle_mass.numpy() > 0.0)
        parts.append((state.particle_q.numpy()[act], m.particle_mass.numpy()[act], m.particle_world.numpy()[act],
                      m.particle_radius.numpy()[act], np.ones(int(act.sum()), dtype=bool)))
    if not parts:
        return _Points(np.zeros((0, 3)), np.zeros(0), np.zeros(0, np.int64), np.zeros(0), np.zeros(0, bool))
    return _Points(*(np.concatenate(cols) for cols in zip(*parts, strict=True)))


def _rotate(quat_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors by unit quaternions (x, y, z, w), row-wise."""
    u, s = quat_xyzw[:, :3], quat_xyzw[:, 3:4]
    return 2.0 * np.sum(u * v, axis=1, keepdims=True) * u + (s * s - np.sum(u * u, axis=1, keepdims=True)) * v \
        + 2.0 * s * np.cross(u, v)
