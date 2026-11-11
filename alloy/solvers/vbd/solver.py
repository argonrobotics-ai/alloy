# SPDX-License-Identifier: Apache-2.0
"""SolverVBDMomentum: Newton's SolverVBD plus one exact coarse translation step per particle iteration (SCR-WP11-2).

Why. VBD's per-vertex Gauss-Seidel leaves a residual R_i = m_i (y_i − x_i)/dt² + f_i(x) after each iteration. The
internal forces sum to zero (translation-invariant energies), so Σ m_i (x_i − y_i)/dt² = −Σ R_i: the net residual
is the momentum error of the step, and it is the slowest mode of the sweep (a component's translation has stiffness
m/dt² only, while every vertex block also carries the membrane). On Newton ac7f7830 a spherical cap under a constant
load gains 3.3x the external impulse (A17.vbd_momentum; docs/notes/wp11-vbd-momentum.md).

What. Once per particle iteration, VBD's own incremental potential
    G(x) = Σ_i m_i/(2 dt²) |x_i − y_i|² + E_int(x)
is minimised exactly along each free component's translation x_i + δ. E_int (membrane, bending, springs, tets,
self-contact within the component) is invariant under it, so δ = −Σ m_i (x_i − y_i) / Σ m_i in closed form. G can
only decrease, a converged solution is unchanged (δ = 0 there), and afterwards Σ m_i (x_i − y_i) = 0: the momentum
changes by exactly the external impulse (gravity and `particle_f` are in y). One more exact descent step on the
same objective (a two-level / coarse-space correction), not a post-hoc momentum patch.

Where. Newton's colour sweep writes each colour's update into `particle_displacements` (relative to the positions
of the last collision detection) and only its penetration-free truncation writes `particle_q` from them. δ is added
to those displacements just before the truncation of the last colour of each iteration, with x read as base +
displacement: the truncation (self-contact planes, isotropic bound, rigid-soft DAT) keeps the last word on every
position at no extra launch. Where it clips a particle the step is no longer exact for that component, but safe.

Scope. A component is blocked (δ = 0, the stock path) in an iteration where it exchanges momentum with anything
else: a body/shape contact record touching any of its particles (particle, edge or face records), an enabled
body attachment, a self-contact pair with another component, or a pinned (m = 0) or inactive particle. Those forces
are not in y, so the closed form would remove them (the first prototype made a resting sheet jitter at 1 m/s).
Newton's per-vertex damping (tri_kd, edge_kd) drags a rigid translation; the step does not include it, so a free
damped component no longer feels that drag. A per-colour callback registered as external (`add_colour_callback`,
base.py) adds forces that are not in y, so the step is skipped while one is registered.

Cost and safety: three to seven launches per iteration with sizes fixed at construction (capture-safe), float64 sums
in a fixed order (deterministic in every Warp determinism mode).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import warp as wp

from alloy._vendor import newton

from . import coarse_contact as CC
from . import kernels as K
from .base import SolverVBDAlloy
from .components import ComponentLayout, build_layout

log = logging.getLogger("alloy.vbd")


@dataclass(frozen=True)
class CoarseArrays:
    """Device copies of the layout plus the per-iteration scratch (all sized at construction)."""

    comp_of: wp.array
    members: wp.array
    chunk_start: wp.array
    comp_chunk_start: wp.array
    g_chunk: wp.array        # [K] vec3d
    m_chunk: wp.array        # [K] float64
    pinned_chunk: wp.array   # [K] int32
    blocked: wp.array        # [C] int32
    delta: wp.array          # [C] vec3: the last translation applied (diagnostics, tests)

    @classmethod
    def upload(cls, lay: ComponentLayout, device) -> CoarseArrays:
        i32 = lambda a: wp.array(a, dtype=wp.int32, device=device)       # noqa: E731
        k, c = lay.chunk_count, lay.count
        return cls(comp_of=i32(lay.comp_of), members=i32(lay.members), chunk_start=i32(lay.chunk_start),
                   comp_chunk_start=i32(lay.comp_chunk_start), g_chunk=wp.zeros(k, dtype=wp.vec3d, device=device),
                   m_chunk=wp.zeros(k, dtype=wp.float64, device=device),
                   pinned_chunk=wp.zeros(k, dtype=wp.int32, device=device),
                   blocked=wp.zeros(c, dtype=wp.int32, device=device), delta=wp.zeros(c, dtype=wp.vec3, device=device))


class SolverVBDMomentum(SolverVBDAlloy):
    """SolverVBD with the exact per-component translation step in every particle iteration (module docstring),
    on SolverVBDAlloy (static-shape DAT under self-contact, SCR-SC8-3). Takes SolverVBD's arguments, plus
    ``coarse_contact`` (opt-in): components touching a body take Newton steps on their translation too
    (coarse_contact.py, SCR-WP11-6)."""

    def __init__(self, model: newton.Model, *args, coarse_contact: bool = False, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.coarse_layout = build_layout(model)
        self.coarse = CoarseArrays.upload(self.coarse_layout, self.device) if self.coarse_layout.count else None
        self._colors_left = 0
        self.coarse_contact = bool(coarse_contact) and self.coarse is not None
        self._cc: dict = {}                      # contact-step arrays, sized by the contact buffer at first use
        self._cc_sorted = False
        self._iter_states = None
        log.debug("SolverVBDMomentum: %d components, %d chunks", self.coarse_layout.count,
                  self.coarse_layout.chunk_count)

    def _initialize_particles(self, state_in, state_out, contacts, dt):
        self._cc_sorted = False                  # new contact records this step
        self._cc_iteration = 0
        super()._initialize_particles(state_in, state_out, contacts, dt)

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        groups = self.model.particle_color_groups
        self._colors_left = len(groups) if self.coarse is not None and self.model.particle_count else 0
        self._iter_states = (state_in, state_out, dt)
        try:
            super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        finally:
            self._colors_left = 0

    def _penetration_free_truncation(self, state, contacts):
        """Newton calls this after every colour; before the last one of an iteration, apply the coarse step."""
        if self._colors_left > 0:
            self._colors_left -= 1
            if self._colors_left == 0 and not self._colour_external:   # external callback forces: not in y
                self._coarse_step(state, contacts)
        super()._penetration_free_truncation(state, contacts)

    def _coarse_step(self, state, contacts) -> None:
        a, m, dev = self.coarse, self.model, self.device
        a.blocked.zero_()
        self._mark_blocked(contacts)
        wp.launch(K.chunk_sums, dim=self.coarse_layout.chunk_count,
                  inputs=[a.chunk_start, a.members, m.particle_mass, m.particle_flags, self.pos_prev_collision_detection,
                          self.particle_displacements, self.particle_q_prev, self.inertia_disp],
                  outputs=[a.g_chunk, a.m_chunk, a.pinned_chunk], device=dev)
        wp.launch(K.component_delta, dim=self.coarse_layout.count,
                  inputs=[a.comp_chunk_start, a.g_chunk, a.m_chunk, a.pinned_chunk, a.blocked], outputs=[a.delta],
                  device=dev)
        if self.coarse_contact and contacts is not None and contacts.soft_contact_max > 0:
            self._cc_iteration = getattr(self, "_cc_iteration", 0) + 1
            if self._cc_iteration > CC.SETTLE_ITERATIONS:             # contact normals settle first (CC docstring)
                self._contact_step(contacts)
        wp.launch(K.apply_delta, dim=m.particle_count, inputs=[a.comp_of, a.delta],
                  outputs=[self.particle_displacements], device=dev)

    def _mark_blocked(self, contacts) -> None:
        """Components that exchange momentum with something else this iteration (module docstring, Scope). With
        ``coarse_contact``, body-contact records also go to ``touching`` and everything else to ``blocked_other``."""
        a, m, dev = self.coarse, self.model, self.device
        cc = self._contact_arrays(contacts) if self.coarse_contact else None
        if contacts is not None and contacts.soft_contact_max > 0:
            for out in (a.blocked,) if cc is None else (a.blocked, cc["touching"]):
                wp.launch(K.mark_soft_contacts, dim=contacts.soft_contact_max,
                          inputs=[contacts.soft_contact_count, contacts.soft_contact_indices, a.comp_of],
                          outputs=[out], device=dev)
        for out in (a.blocked,) if cc is None else (a.blocked, cc["blocked_other"]):
            self._mark_other(out)

    def _mark_other(self, out) -> None:
        a, m, dev = self.coarse, self.model, self.device
        if m.attachment_particle_body_count > 0:
            wp.launch(K.mark_attachments, dim=m.attachment_particle_body_count,
                      inputs=[m.attachment_particle_body_particle, m.attachment_particle_body_enabled, a.comp_of],
                      outputs=[out], device=dev)
        if not self.particle_enable_self_contact:
            return
        info = self.trimesh_collision_info
        wp.launch(K.mark_vertex_triangle_pairs, dim=info.vertex_colliding_triangles_count.shape[0],
                  inputs=[info, m.tri_indices, a.comp_of], outputs=[out], device=dev)
        if m.edge_count > 0:
            wp.launch(K.mark_edge_edge_pairs, dim=info.edge_colliding_edges_count.shape[0],
                      inputs=[info, m.edge_indices, a.comp_of], outputs=[out], device=dev)

    # -- contact-aware coarse step (coarse_contact.py) ------------------------------------------------------------
    def _contact_arrays(self, contacts) -> dict:
        """The contact step's arrays, (re)allocated when the contact buffer grows; the per-iteration flags zeroed."""
        cc, c, dev = self._cc, self.coarse_layout.count, self.device
        r = int(contacts.soft_contact_max) if contacts is not None else 0
        if cc.get("records", -1) < r:
            n = self.model.particle_count
            cc.update(records=r, keys=wp.zeros(2 * max(r, 1), dtype=wp.int32, device=dev),
                      order=wp.zeros(2 * max(r, 1), dtype=wp.int32, device=dev),
                      starts=wp.zeros(c + 1, dtype=wp.int32, device=dev),
                      f=wp.zeros(max(r, 1), dtype=wp.vec3, device=dev),
                      h=wp.zeros(max(r, 1), dtype=wp.mat33, device=dev),
                      n=wp.zeros(max(r, 1), dtype=wp.vec3, device=dev),
                      x=wp.zeros(n, dtype=wp.vec3, device=dev), delta=wp.zeros(c, dtype=wp.vec3, device=dev),
                      active=wp.zeros(c, dtype=wp.int32, device=dev))
            self._cc_sorted = False
        if "touching" not in cc:
            cc.update(touching=wp.zeros(c, dtype=wp.int32, device=dev),
                      blocked_other=wp.zeros(c, dtype=wp.int32, device=dev),
                      empty_q=wp.empty(0, dtype=wp.transform, device=dev),
                      empty_qd=wp.empty(0, dtype=wp.spatial_vector, device=dev),
                      empty_v=wp.empty(0, dtype=wp.vec3, device=dev))
        cc["touching"].zero_()
        cc["blocked_other"].zero_()
        return cc

    def _sort_records(self, contacts, cc) -> None:
        """Records grouped by component, stable (record order within a component), once per step."""
        r, c, dev = cc["records"], self.coarse_layout.count, self.device
        wp.launch(CC.record_keys, dim=r, inputs=[contacts.soft_contact_count, contacts.soft_contact_indices,
                                                 self.coarse.comp_of, c], outputs=[cc["keys"], cc["order"]], device=dev)
        wp.utils.radix_sort_pairs(cc["keys"], cc["order"], r)
        wp.launch(CC.segment_starts, dim=c + 1, inputs=[cc["keys"], r], outputs=[cc["starts"]], device=dev)
        self._cc_sorted = True

    def _views(self, contacts) -> tuple:
        """Newton's choice of body arrays for particle contacts (solver_vbd: external rigid solver or not)."""
        m, cc = self.model, self._cc
        state_in, state_out, _ = self._iter_states
        if self.integrate_with_external_rigid_solver:
            q, q_prev, qd = state_out.body_q, state_in.body_q, state_out.body_qd
        else:
            q, q_prev, qd = state_in.body_q, (self.body_q_prev if m.body_count else None), state_in.body_qd
        b = CC.BodyView()
        b.shape_body, b.shape_margin = m.shape_body, m.shape_margin
        b.body_com = m.body_com if m.body_count else cc["empty_v"]
        b.body_q = q if q is not None else cc["empty_q"]
        b.body_q_prev = q_prev if q_prev is not None else cc["empty_q"]
        b.body_qd = qd if qd is not None else cc["empty_qd"]
        rec = CC.RecordView()
        rec.count, rec.indices = contacts.soft_contact_count, contacts.soft_contact_indices
        rec.bary, rec.shape = contacts.soft_contact_barycentric, contacts.soft_contact_shape
        rec.body_pos, rec.body_vel = contacts.soft_contact_body_pos, contacts.soft_contact_body_vel
        rec.normal, rec.ke = contacts.soft_contact_normal, self.body_particle_contact_penalty_k
        rec.kd, rec.mu = self.body_particle_contact_material_kd, self.body_particle_contact_material_mu
        return b, rec

    def _contact_step(self, contacts) -> None:
        """Newton steps on each touching component's translation; the result goes into the coarse delta."""
        a, cc, dev, m = self.coarse, self._cc, self.device, self.model
        if not self._cc_sorted:
            self._sort_records(contacts, cc)
        c, r, dt = self.coarse_layout.count, cc["records"], float(self._iter_states[2])
        wp.launch(CC.mark_active, dim=c, inputs=[cc["touching"], cc["blocked_other"], a.comp_chunk_start,
                                                 a.pinned_chunk], outputs=[cc["active"], cc["delta"]], device=dev)
        bodies, rec = self._views(contacts)
        pts = CC.ParticleView()
        pts.radius, pts.mass, pts.world, pts.gravity = m.particle_radius, m.particle_mass, m.particle_world, m.gravity
        for _ in range(CC.NEWTON_STEPS):
            wp.launch(CC.shifted_positions, dim=m.particle_count,
                      inputs=[a.comp_of, self.pos_prev_collision_detection, self.particle_displacements, cc["delta"],
                              cc["active"]], outputs=[cc["x"]], device=dev)
            barrier = bool(self.rigid_soft_contact_use_log_barrier)
            wp.launch(CC.record_terms, dim=r, inputs=[rec, bodies, pts, cc["x"], self.particle_q_prev,
                                                      self.friction_epsilon, dt, barrier],
                      outputs=[cc["f"], cc["h"], cc["n"]], device=dev)
            wp.launch(CC.component_newton, dim=c, inputs=[cc["active"], a.comp_chunk_start, a.g_chunk, a.m_chunk,
                                                          cc["starts"], cc["order"], cc["f"], cc["h"], cc["n"], dt],
                      outputs=[cc["delta"]], device=dev)
        wp.launch(CC.merge_delta, dim=c, inputs=[cc["active"], cc["delta"]], outputs=[a.delta], device=dev)


__all__ = ["CoarseArrays", "SolverVBDMomentum"]
