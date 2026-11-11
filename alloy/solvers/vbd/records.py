# SPDX-License-Identifier: Apache-2.0
"""Newton's view of a VBD step's body-particle records, before and after the step, for the post-step passes that read
them: the stick friction (``stick``) and the exit-speed cap (``exit_speed``).

The particle iteration reads a record's body point at the body's pose over the step: when a rigid solver outside VBD
integrates the bodies, the step's end (``state_out.body_q``) against its start (``state_in.body_q``); when VBD
integrates them itself, ``body_q`` against Newton's ``body_q_prev``. Its rigid finalization advances ``body_q_prev``
to the step's end, so after the step the start pose is the copy ``keep_start`` took before it (AUDIT-STACK 3b, the
stick's driven plate: reading the advanced history saw a moving plate as still).
"""

from __future__ import annotations

from typing import Any

import warp as wp

from alloy.solvers.film.grip import ContactView


def _empty(dtype: Any, device: Any) -> wp.array:
    return wp.zeros(1, dtype=dtype, device=device)


class RecordViews:
    """The body poses and record arrays of one VBD solver's steps, as its particle iteration reads them."""

    def __init__(self, solver: Any) -> None:
        m, dev = solver.model, solver.model.device
        self.solver = solver
        self.no_q, self.no_qd, self.no_vec = _empty(wp.transform, dev), _empty(wp.spatial_vector, dev), \
            _empty(wp.vec3, dev)
        own_bodies = m.body_count > 0 and not solver.integrate_with_external_rigid_solver
        self.q_start = wp.clone(solver.body_q_prev) if own_bodies else None    # VBD-integrated bodies' start

    def keep_start(self) -> None:
        """Before the step: keep the start pose of the bodies VBD integrates (module docstring)."""
        if self.q_start is not None:
            wp.copy(self.q_start, self.solver.body_q_prev)

    def bodies(self, state_in: Any, state_out: Any, after: bool) -> tuple[wp.array, wp.array | None, wp.array]:
        """(body_q, body_q_prev or None, body_qd) as Newton's particle iteration reads them (as film/grip.py), before
        the step or, with ``after``, after it (module docstring)."""
        s, m = self.solver, self.solver.model
        if m.body_count == 0 or state_in.body_q is None:
            return self.no_q, None, self.no_qd
        if s.integrate_with_external_rigid_solver:
            return state_out.body_q, state_in.body_q, state_out.body_qd
        if after:
            return state_out.body_q, self.q_start, state_out.body_qd
        return state_in.body_q, s.body_q_prev, state_in.body_qd

    def view(self, contacts: Any, state_in: Any, state_out: Any, dt: float, after: bool = False) -> ContactView:
        s, m = self.solver, self.solver.model
        bq, bqp, bqd = self.bodies(state_in, state_out, after)
        cv = ContactView()
        cv.count, cv.max_count, cv.corners = contacts.soft_contact_count, contacts.soft_contact_max, \
            contacts.soft_contact_indices
        cv.shape, cv.body_pos, cv.body_vel = contacts.soft_contact_shape, contacts.soft_contact_body_pos, \
            contacts.soft_contact_body_vel
        cv.normal, cv.shape_body, cv.shape_margin, cv.radius = contacts.soft_contact_normal, m.shape_body, \
            m.shape_margin, m.particle_radius
        cv.ke, cv.mu = s.body_particle_contact_penalty_k, s.body_particle_contact_material_mu
        cv.log_barrier = int(bool(s.rigid_soft_contact_use_log_barrier))
        cv.body_q, cv.body_q_prev, cv.use_prev = bq, bqp if bqp is not None else bq, int(bqp is not None)
        cv.body_qd, cv.body_com = bqd, m.body_com if m.body_count else self.no_vec
        cv.anchor, cv.dt, cv.eps_u = s.particle_q_prev, float(dt), float(s.friction_epsilon * dt)
        cv.v0 = self.no_vec
        return cv

    def bary(self, contacts: Any) -> wp.array:
        """The records' barycentric weights (read only for edge / face records, which always carry them)."""
        bary = getattr(contacts, "soft_contact_barycentric", None)
        return bary if bary is not None and bary.shape[0] > 0 else self.no_vec


__all__ = ["RecordViews"]
