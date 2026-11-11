# SPDX-License-Identifier: Apache-2.0
"""SolverDFSPH's coupling hooks (Newton's coupled-solver interface; SPEC §3.6, SCR-WP9-2; soft walls: engine item 3).

Bodies: proxy bodies are integrated or kinematic per ``bodies.py``; the harvest returns the momentum change of
integrated bodies and the exchange-averaged fluid wrench of the others.

Particles: VBD surface vertices mapped as proxy particles are soft walls (``soft_walls.py``). The harvest returns
each vertex's exchange-averaged fluid force. Soft walls need a staggered proxy pair (end pose, end velocity; Aitken):
after the sync the first step moves the soft vertices back by v dt_x, so the wall runs from end - v dt_x to the end
pose (WP4 review, 2026-11-06). A lagged pair calls the rewind hook, which refuses: with lagged proxies a floating
soft box diverged (vmax 7.5 m/s) or settled at 0.08 of its draft under 0.10 W of fluid force (P1's L1 rows).
With shell inertia (P2, soft_inertia.py; on by default with the CG solve; open components only) the soft
vertices of a cloth or shell then move inside the
exchange as particles of VBD's mass under the liquid's impulse and VBD's implied non-fluid acceleration; F_used is
read at the pass's first step from the array the harvest writes (the coupler's proxy force array, the same every
pass), the begin and end velocities at the coupler's notifies.
"""

from __future__ import annotations

import logging

import warp as wp
from alloy._vendor.newton import Contacts, State, StateFlags

from . import bodies as bd
from .particle_walls import ParticleWalls
from .soft_inertia import SoftInertia

log = logging.getLogger("alloy.sph")

PARTICLE_Q = int(StateFlags.PARTICLE_Q)
PARTICLE_QD = int(StateFlags.PARTICLE_QD)


class SPHCouplingMixin:
    """Coupling hooks of SolverDFSPH; expects the solver's layout, bodies, warm starts, emitters and ``soft``."""

    def coupling_notify_input_state_update(self, state: State, flags: int, *, iteration_restart: bool = False,
                                           dt: float = 0.0) -> None:
        """Restart passes reuse the exchange's warm starts, emitter state and shape meters, so every pass of one
        exchange starts from the same solver state and ``shape_impulse`` counts the pass that sticks (a staggered
        pair's passes added up to 8.7x the walls' impulse before, WP9 (a1)); any notification starts a new exchange
        impulse window."""
        if self.layout.n == 0:
            return
        if int(flags) & PARTICLE_Q and (self.soft.count or self.pw.count):
            self._soft_back = float(dt)  # cleared by the rewind (lagged); kept after a staggered sync
        if int(flags) & PARTICLE_QD and self.inertia is not None:
            self.inertia.on_notify(state)  # begin state (copy-in), then end state (sync): soft_inertia.py
        if iteration_restart:
            wp.copy(self._kv, self._kv_saved)
            wp.copy(self._kd, self._kd_saved)
            wp.copy(self._meter_j, self._meter_saved)    # the shape meters count the pass that sticks only
            self.emitters.restore()
        else:
            wp.copy(self._kv_saved, self._kv)
            wp.copy(self._kd_saved, self._kd)
            wp.copy(self._meter_saved, self._meter_j)
            self.emitters.save()
        if self.bodies.count:
            self._exch_j.zero_()
        self.soft.new_window()
        self.pw.new_window()

    def coupling_harvest_proxy_wrenches(self, body_local_to_proxy_global: wp.array, out_body_f: wp.array, *,
                                        body_qd_before: wp.array, state: State, state_out: State,
                                        contacts: Contacts | None, dt: float) -> None:
        del state, contacts
        n = body_local_to_proxy_global.shape[0]
        if n == 0:
            return
        if not dt > 0.0:
            raise ValueError("SolverDFSPH harvest requires dt > 0")
        wp.launch(bd.harvest_kernel, dim=n, device=self.device,
                  inputs=[body_local_to_proxy_global, self._body_mode, body_qd_before, state_out.body_qd,
                          state_out.body_q, self.model.body_mass, self.model.body_inertia, self._exch_j, float(dt),
                          out_body_f])

    def coupling_rewind_proxy_particle(self, particle_local_to_proxy_global: wp.array, state: State,
                                       coupling_forces: wp.array, particle_gravity_acceleration: wp.array,
                                       dt: float) -> None:
        """Called for lagged particle proxies only: refused for soft walls (module doc)."""
        del state, coupling_forces, particle_gravity_acceleration, dt
        if particle_local_to_proxy_global.shape[0] and (self.soft.count or self.pw.count):
            raise ValueError("SolverDFSPH soft and particle walls need a staggered proxy pair (PairPolicy / Proxy mode "
                             "'staggered' with Aitken relaxation); lagged proxies diverge for a liquid-soft pair")

    def coupling_harvest_proxy_particle_forces(self, particle_local_to_proxy_global: wp.array, out_particle_f, *,
                                               dt: float, **kwargs) -> None:
        del kwargs
        if particle_local_to_proxy_global.shape[0] == 0:
            return
        if not dt > 0.0:
            raise ValueError("SolverDFSPH harvest requires dt > 0")
        if self.soft.count == 0 and self.pw.count == 0:
            raise ValueError("SolverDFSPH: proxy particles but neither soft faces nor particle walls")
        if self.inertia is not None:  # F_used of the next pass is read from this array (soft_inertia.py)
            self.inertia.capture_feedback(particle_local_to_proxy_global, out_particle_f)
        self.soft.harvest(particle_local_to_proxy_global, out_particle_f, dt)
        self.pw.harvest(particle_local_to_proxy_global, out_particle_f, dt)

    def coupling_prepare_proxy_contacts(self, state: State, contacts: Contacts | None, *,
                                        contacts_freshly_detected: bool = False) -> Contacts | None:
        del state, contacts_freshly_detected
        return contacts  # SolverDFSPH reads no Contacts rows: boundaries come from shape SDFs

    def coupling_supports_full_surface_soft_contacts(self) -> bool:
        return True  # contacts are ignored entirely, so any soft-contact layout is acceptable

    def coupling_supports_inertial_property_refresh(self) -> bool:
        return False  # body modes are recomputed on the host in notify_model_changed

    def _make_inertia(self, enabled: bool) -> SoftInertia | None:
        """Shell inertia (soft_inertia.py) for soft faces in a two-way pair with the CG solve; None otherwise."""
        if not enabled or self.soft.count == 0 or self.config.coupling == "one_way":
            return None
        if not int(self.soft.arrays.two_sided.numpy().sum()):  # closed bodies only: P1's prescribed walls exactly
            return None
        if self._cg is None:
            log.warning("SolverDFSPH: soft inertia needs pressure_solver 'cg'; soft walls stay prescribed (P1)")
            return None
        inertia = SoftInertia(self.soft, self._slots, self.device)
        self._cg.wall_inertia = inertia
        return inertia

    def _make_particle_walls(self) -> ParticleWalls:
        """Proxies without soft faces are particle walls (particle_walls.py); soft faces and particle walls in one view
        are refused (SCR-WP12-MIX R2: VBD with MPM stays refused)."""
        soft_vid = self.soft.vid.numpy() if self.soft.count else None
        pw = ParticleWalls(self.model, self.soft.base + self.soft.components, self._h, self._kind, self.device,
                           soft_vid)
        if pw.count and self.soft.count:
            raise ValueError(f"SolverDFSPH: {self.soft.count} soft faces and {pw.count} particle walls in one view "
                             "are not supported (a VBD body and an MPM material beside one liquid)")
        pw.bind(self.shapes.arrays)
        if pw.mask is not None:
            self._nbrs.set_walls(pw.mask)
        return pw

    def _soft_step_start(self, state: State, dt: float) -> None:
        """Step start: after a staggered sync move the soft vertices back by v dt_x once (with inertia: the pass's
        F_used and end velocities), set the shell's velocities for this step, then gather and refit."""
        if self.soft.count == 0 and self.pw.count == 0:
            return
        if self._soft_back > 0.0:
            self.soft.advance(state, -self._soft_back)
            self.pw.advance(state, -self._soft_back)
            if self.inertia is not None:
                self.inertia.pass_start(self.model, self._soft_back)
            self._soft_back = 0.0
        if self.inertia is not None:
            self.inertia.begin_step(state, dt)
        self.soft.begin(state)

    def _soft_refresh(self, state: State) -> None:
        """After a solve or the non-pressure stage: the shell's velocities from the step's impulses so far."""
        if self.inertia is not None:
            self.inertia.refresh(state, self._slot_impulse)

    def _soft_end_step(self, state: State) -> None:
        if self.inertia is not None:
            self.inertia.end_step(state)

