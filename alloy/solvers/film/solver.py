# SPDX-License-Identifier: Apache-2.0
"""SolverFilm: Alloy's VBD plus three thin-film features (SC8; design: docs/notes/sc8-film-design.md).

Base: ``SolverVBDMomentum`` (A17, WP11): for a film model, each free film component gets the exact per-component
translation step in every particle iteration, so a falling film keeps its momentum (FILM-4b: the 0.1 mm paper filter
fell at 0.65 m/s on stock VBD and at 1.05 m/s with the step, target 1.06-1.12; item 8 G2, WP11 11-06). A component
that is pinned or in contact keeps the stock iteration (A17's scope). A model without film items, or one with
``FilmConfig(momentum=False)``, gets no step. A free damped film no longer feels the rigid-translation drag of
per-vertex damping (tri_kd / edge_kd): that is A17's documented scope and physically right, but it is a change.

The base also brings SolverVBDAlloy's static-shape DAT (SCR-SC8-3: under particle self-contact, particles in contact
with a static shape are truncated against its division plane). Film models keep it; a model without film items turns
it off and steps as stock SolverVBD exactly.

``step`` = [air force into ``state_in.particle_f``] → ``SolverVBD.step`` → [caller forces restored] → [strain-limit
projection of ``state_out``] → [plastic hinge update]. Every stage is device kernels with sizes fixed at
construction (capture-safe, no host sync). A model with no film attributes set builds no film arrays, takes no momentum
step and steps through ``SolverVBD.step`` alone: bitwise the stock solver (default off).
"""

from __future__ import annotations

import logging

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.solvers.vbd.solver import SolverVBDMomentum

from . import aero, hinge, ledger, longrange, seam_vbd, seams, sheet_j2, skin, strain, stress
from .config import FilmConfig

try:                                     # grip.py uses Newton internals; a rename upstream must not break film imports
    from .grip import FilmGrip
except ImportError as _grip_error:       # pragma: no cover (canary: tests/solvers/film/test_film_grip.py)
    FilmGrip = None
    logging.getLogger("alloy.film").warning("SolverFilm: grip terms unavailable (%s); film contacts step without "
                                            "them (docs/notes/film-grip-c16.md)", _grip_error)
from .topology import DeviceTopology, build_topology

log = logging.getLogger("alloy.film")

PINNED_PIECE_START = "particle_pinned_piece_start"     # Newton SolverVBD keyword (WP11's pin bump, item 8 G1)


def _vbd_accepts(name: str) -> bool:
    """True when the pinned Newton's SolverVBD takes keyword ``name`` (older pins: the film keeps stock VBD)."""
    import inspect

    return name in inspect.signature(newton.solvers.SolverVBD.__init__).parameters


@wp.kernel(enable_backward=False)
def _reset_edges(edge_world: wp.array[wp.int32], mask: wp.array[wp.bool], rest0: wp.array[float],
                 rest: wp.array[float], plastic_work: wp.array[float]):
    e = wp.tid()
    w = edge_world[e]
    if w < 0:
        w = mask.shape[0] - 1          # Newton's reset mask: the last entry selects the global world (-1)
    if mask[w]:
        rest[e] = rest0[e]
        plastic_work[e] = 0.0


@wp.kernel(enable_backward=False)
def _reset_worlds(mask: wp.array[wp.bool], a: wp.array[wp.float64]):
    w = wp.tid()
    if mask[w]:
        a[w] = wp.float64(0.0)


class SolverFilm(SolverVBDMomentum):
    """SolverVBD with strain limiting, an elastoplastic hinge and air drag on the triangles the ``film:*``
    attributes switch on (config.py). Extra keyword ``film`` (FilmConfig); everything else goes to SolverVBD."""

    def __init__(self, model: newton.Model, *, film: FilmConfig | None = None, **vbd_kwargs) -> None:
        film = film if film is not None else FilmConfig()
        top = build_topology(model)
        if top.any and film.pinned_piece_start and _vbd_accepts(PINNED_PIECE_START):
            vbd_kwargs.setdefault(PINNED_PIECE_START, True)     # film entries only: default-off stays bitwise
        if top.seams is not None:                   # twins in different VBD colours (seam_vbd.recolor_twins)
            moved = seam_vbd.recolor_twins(model, top.seams)
            log.debug("SolverFilm: %d twin ends recoloured", moved)
        super().__init__(model, **vbd_kwargs)
        self.film = film
        self.alloy_exchange_caps = film.exchange_caps     # read by alloy.coupling.stiffness_cap
        self.film_topology = top
        self._film_on = top.any
        if not (self._film_on and film.momentum):
            self.coarse = None                  # no A17 translation step: SolverVBDMomentum iterates as stock
        if not self._film_on:
            self._static_dat_t = None           # nor SolverVBDAlloy's static-shape DAT: stock SolverVBD exactly
            log.debug("SolverFilm: no film attributes set; stepping as SolverVBD")
            return
        dev = model.device
        self._dtop = DeviceTopology.upload(top, dev)
        self._ns = model.film
        T, P = int(model.tri_count), int(model.particle_count)
        self._delta = wp.zeros((T, 3), dtype=wp.vec3, device=dev)
        self._active = wp.zeros(T, dtype=wp.int32, device=dev)
        C = len(top.pairs)
        self._pair_delta = wp.zeros((max(C, 1), 2), dtype=wp.vec3, device=dev)
        self._pair_active = wp.zeros(max(C, 1), dtype=wp.int32, device=dev)
        self._init_seams(top, dev)
        self._piece_base = None                          # VBD's triangle pieces; seams join them (seam_vbd.py)
        if self._seams is not None and getattr(self, "particle_cloth_piece", None) is not None:
            self._piece_base = wp.clone(self.particle_cloth_piece)
            self._piece_label = wp.zeros(self.piece_pinned.shape[0], dtype=wp.int32, device=dev)
        self._dx = wp.zeros(P, dtype=wp.vec3, device=dev)
        self.projection_reaction = wp.zeros(P, dtype=wp.vec3, device=dev)   # N, per pinned particle, last step
        self._count = wp.zeros(P, dtype=wp.int32, device=dev)
        self._f_tri = wp.zeros(T, dtype=wp.vec3, device=dev)
        self._f_air = wp.zeros(P, dtype=wp.vec3, device=dev)
        self._f_user = wp.zeros(P, dtype=wp.vec3, device=dev)
        self._f_scratch = wp.zeros(P, dtype=wp.vec3, device=dev)
        self._x_in = wp.zeros(P, dtype=wp.vec3, device=dev)
        self.plastic_work = wp.zeros(int(model.edge_count), dtype=float, device=dev)
        self._rest_angle0 = wp.clone(model.edge_rest_angle) if (top.plastic or top.j2) else None
        self._j2 = sheet_j2.SheetJ2(model, dev) if top.j2 else None          # J2 metal sheet hinges (sheet_j2.py)
        self.ledger = ledger.FilmLedger(max(int(model.world_count), 1), dev)
        self._ratio = wp.zeros(T, dtype=float, device=dev)
        self.last_particle_q = model.particle_q
        grip = (film.projection_friction or film.static_friction_hessian) and FilmGrip is not None
        self._grip = FilmGrip(self, film.projection_friction, film.static_friction_hessian) if grip else None
        entry = film.contact_skin_entry                  # VBD's skin overshoot at static shapes (skin.py)
        self._skin = skin.SkinLimit(model, top.particles, entry) if entry is not None else None
        # N per particle, last step: the static shapes' share the skin guard carried (None with the limit off)
        self.contact_guard_reaction = self._skin.reaction if self._skin is not None else None
        self._stress = self._stress_limiter(top) if film.limiter == "stress" else None
        # N per particle, last step: the stretch limit's pull (stress.py); with the projection, its reaction on pins
        self.limiter_force = self._stress.force if self._stress is not None else self.projection_reaction
        log.info("SolverFilm: %d limited / %d airborne triangles, %d plastic hinges, %d J2 hinges, %d film particles, "
                 "%d long-range limits, limiter %s", top.limited, top.airborne, top.plastic, top.j2,
                 len(top.particles), C, film.limiter)

    def _stress_limiter(self, top):
        """The stress limit (``FilmConfig.limiter="stress"``): k = E h A / (1 - nu^2) per limited triangle."""
        m, ns = self.model, self.model.film
        if top.seams is not None:
            raise ValueError("FilmConfig.limiter='stress' does not handle tear seams yet (their forces come from the "
                             "projection's sweeps); use limiter='projection' for a film with tear lines")
        ke, cap = ns.limit_ke.numpy(), ns.strain_cap.numpy()
        if np.any((cap > 0.0) & (ke <= 0.0)):
            raise ValueError("FilmConfig.limiter='stress' needs model.film.limit_ke on every limited triangle "
                             "(the film emitter writes it: emit.add_film_mesh)")
        if np.any((cap > 0.0) & (ns.compress_cap.numpy() < 1.0)):
            raise ValueError("FilmConfig.limiter='stress' has no compression cap; use limiter='projection'")
        k = np.where(cap > 0.0, ke * m.tri_areas.numpy(), 0.0)
        return stress.StressLimiter(self, top, self._dtop, k, self.film.stress_entry)

    def alloy_route_notes(self) -> dict[str, str]:
        """``sim.routes()`` row notes for the objects on this entry (alloy/api/sim.py)."""
        notes = {}
        if getattr(self, "_skin", None) is not None:
            notes["contact_skin"] = (f"a film vertex reaching a static shape enters its contact skin by at most "
                                     f"{100 * self._skin.entry:g} % of its contact radius per iteration, so a thin "
                                     "sheet over a sharp box edge stays out of the box; a film on a moving body is not "
                                     "covered (docs/known_gaps/film-thin-sheet-box-edge.md)")
        if getattr(self, "_stress", None) is not None:
            notes["film_limiter"] = ("past its stretch cap the film is an elastic membrane of its real stiffness "
                                     "inside the solve, so its tension is a force there (opt-in) "
                                     "(docs/known_gaps/film-projection-force.md)")
        cap = getattr(self, "alloy_exchange_cap_installed", None)          # set by the coupler when it caps
        if cap is not None:
            notes["exchange_caps"] = (f"the rigid bodies coupled to this film take its lagged contact force with "
                                      f"capped stiffness (c {cap[0]}) and damping (c_d {cap[1]}) per exchange, so a "
                                      "light body is not thrown off it (docs/known_gaps/paper-open-carton-peel.md)")
        return notes

    # --- step ---------------------------------------------------------------------------------------------------

    def _solve_particle_iteration(self, state_in, state_out, contacts, dt):
        self._skin_colour = 0                       # Newton truncates once per colour, in order (skin.py)
        try:
            super()._solve_particle_iteration(state_in, state_out, contacts, dt)
        finally:
            self._skin_colour = -1

    def _penetration_free_truncation(self, state, contacts):
        """Newton calls this after the initial guess and after every colour: first the film's skin entry limit
        (skin.py) on that colour's particles (all film particles after the guess), then the rest."""
        skin_on, stress_on = getattr(self, "_skin", None) is not None, getattr(self, "_stress", None) is not None
        if skin_on or stress_on:
            k = getattr(self, "_skin_colour", -1)
            groups = self.model.particle_color_groups
            group = groups[k] if 0 <= k < len(groups) else None
            if stress_on:
                self._stress.cut_entry(self, state, group)
            if skin_on:
                self._skin.apply(self, state, group)
            if k >= 0:
                self._skin_colour = k + 1
        super()._penetration_free_truncation(state, contacts)

    def step(self, state_in, state_out, control, contacts, dt: float, *, observables=None) -> None:
        if not self._film_on:
            super().step(state_in, state_out, control, contacts, dt, observables=observables)
            return
        top = self.film_topology
        if self.film.ledger:
            self._film_energy(state_in, self.ledger.e_pre)
        swapped = None
        if top.airborne:
            swapped = self._apply_air(state_in, dt)
        if self._piece_base is not None:
            self._join_pieces()
        if self._grip is not None:
            self._grip.begin_step(state_in, state_out, dt)
        if self._skin is not None:
            self._skin.begin_step(self.model, contacts, state_in.particle_q, state_in.body_q)
        super().step(state_in, state_out, control, contacts, dt, observables=observables)
        if top.airborne:
            self._restore_forces(state_in, swapped, state_out)
        if self._stress is not None:
            self._stress.after_step(state_out.particle_q)       # the limit acted inside the solve: readouts only
        elif top.limited and self.film.strain_iterations > 0:
            self._project(state_out, dt, contacts)
        if top.plastic:
            wp.launch(hinge.plastic_hinge, dim=self.model.edge_count,
                      inputs=[state_out.particle_q, self.model.edge_indices, self._ns.yield_angle,
                              self.model.edge_rest_angle, self.plastic_work], device=self.model.device)
        if self._j2 is not None:
            self._j2.step(self.model, state_out.particle_q, self._rest_angle0)
        self.last_particle_q = state_out.particle_q
        if self.film.ledger:
            L = self.ledger
            self._film_energy(state_out, L.e_post)
            wp.launch(ledger.step_gain_kernel, dim=L.e_pre.shape[0], inputs=[L.e_pre, L.e_post, L.air_step,
                                                                             L.step_gain_max],
                      device=self.model.device)

    def _film_energy(self, state, out: wp.array) -> None:
        d, m = self._dtop, self.model
        wp.launch(ledger.film_energy_kernel, dim=out.shape[0],
                  inputs=[d.world_start, d.particles, m.particle_mass, m.gravity, state.particle_q,
                          state.particle_qd, out], device=m.device)

    def _air_params(self, dt: float) -> aero.AirParams:
        c = self.film
        p = aero.AirParams()
        p.rho, p.cd, p.ct = float(c.air_density), float(c.drag_normal), float(c.drag_tangent)
        p.wind, p.dt, p.cap = wp.vec3(*c.wind), float(dt), int(c.drag_cap)
        return p

    def _apply_air(self, state_in, dt: float):
        """Saves the caller's particle forces, adds the air force; returns the array swapped in when the state had
        none (restored to None after the step)."""
        dev, m, d = self.model.device, self.model, self._dtop
        swapped = None
        if state_in.particle_f is None:
            self._f_scratch.zero_()
            state_in.particle_f, swapped = self._f_scratch, True
        else:
            wp.copy(self._f_user, state_in.particle_f)
        wp.copy(self._x_in, state_in.particle_q)
        p = self._air_params(dt)
        wp.launch(aero.tri_air_force, dim=m.tri_count,
                  inputs=[p, state_in.particle_q, state_in.particle_qd, m.tri_indices, self._ns.air, self._f_tri],
                  device=dev)
        wp.launch(aero.vertex_air_force, dim=len(self.film_topology.particles),
                  inputs=[p, d.vtx_start, d.vtx_tri, d.particles, self._f_tri, m.particle_inv_mass,
                          state_in.particle_qd, state_in.particle_f, self._f_air], device=dev)
        return swapped

    def _restore_forces(self, state_in, swapped, state_out) -> None:
        if swapped:
            state_in.particle_f = None
        else:
            wp.copy(state_in.particle_f, self._f_user)
        if self.film.ledger:
            d = self._dtop
            wp.launch(ledger.air_work_kernel, dim=self.ledger.air_work.shape[0],
                      inputs=[d.world_start, d.particles, self._f_air, self._x_in, state_out.particle_q,
                              self.ledger.air_work, self.ledger.air_step, self.ledger.air_since],
                      device=self.model.device)

    # --- seams (tear lines, seams.py) ------------------------------------------------------------------------------

    def _init_seams(self, top, dev) -> None:
        s = top.seams
        self._seams = seams.upload(s, dev) if s is not None else None
        self._n_seams = len(s.a) if s is not None else 0
        n_bonds = s.n_bonds if s is not None else 1
        self._bond_intact = wp.array(np.ones(n_bonds, np.int8), dtype=wp.int8, device=dev)
        self.bond_force = wp.full(n_bonds, -1.0, dtype=float, device=dev)        # -1: not supplied
        self.bond_threshold = wp.full(n_bonds, -1.0, dtype=float, device=dev)

    def topology_link(self, intact: wp.array):
        """Called by the TopologyController with its per-bond ``intact`` flags: broken seams then leave the
        projection. Returns (force, threshold) per bond, N, filled for this film's seams each step (-1 elsewhere),
        or None when the film has no seams."""
        if getattr(self, "_seams", None) is None:
            return None
        self._bond_intact = intact
        return self.bond_force, self.bond_threshold

    def _join_pieces(self) -> None:
        dev = self.model.device
        n_pieces, n_groups = self._piece_label.shape[0], len(self.film_topology.seams.world_start) - 1
        wp.launch(seam_vbd.init_labels, dim=n_pieces, inputs=[self._piece_label], device=dev)
        wp.launch(seam_vbd.join_pieces, dim=n_groups, inputs=[self._seams, self._bond_intact, self._piece_base,
                                                              self._piece_label], device=dev)
        wp.launch(seam_vbd.flatten_labels, dim=n_pieces, inputs=[self._piece_label], device=dev)
        wp.launch(seam_vbd.write_pieces, dim=self._piece_base.shape[0],
                  inputs=[self._piece_base, self._piece_label, self.particle_cloth_piece], device=dev)

    def _seam_force(self, x, dt: float) -> None:
        dev, m = self.model.device, self.model
        wp.launch(seams.seam_force, dim=self._n_seams,
                  inputs=[self._seams, self._bond_intact, x, m.tri_indices, 1.0 / (dt * dt), self._seam_alpha(dt),
                          self.bond_force, self.bond_threshold], device=dev)

    def _seam_alpha(self, dt: float) -> float:
        tau = float(self.film.seam_force_tau)
        return 1.0 if tau <= 0.0 else min(1.0, float(dt) / tau)

    def _project(self, state_out, dt: float, contacts=None) -> None:
        dev, m, d, n = self.model.device, self.model, self._dtop, len(self.film_topology.particles)
        x = state_out.particle_q
        C = len(self.film_topology.pairs)
        grip = self._grip is not None and self._grip.capacity(state_out, contacts)     # friction on the shift
        self.projection_reaction.zero_()
        if self._seams is not None:                      # the twin springs' share, from VBD's result
            wp.launch(seams.seam_spring, dim=self._n_seams,
                      inputs=[self._seams, self._bond_intact, x, m.spring_stiffness, float(dt)], device=dev)
        for _ in range(self.film.strain_iterations):
            wp.launch(strain.tri_corrections, dim=m.tri_count,
                      inputs=[x, m.particle_inv_mass, m.tri_indices, m.tri_poses, self._ns.strain_cap,
                              self._ns.compress_cap, float(self.film.mass_ratio_cap), self._delta, self._active],
                      device=dev)
            if C:
                wp.launch(longrange.pair_corrections, dim=C,
                          inputs=[x, m.particle_inv_mass, d.pairs, d.pair_rest, d.pair_cap,
                                  float(self.film.mass_ratio_cap), d.gate_start, d.gate, self._bond_intact],
                          outputs=[self._pair_delta, self._pair_active], device=dev)
            wp.launch(strain.vertex_count, dim=n,
                      inputs=[d.vtx_start, d.vtx_tri, d.particles, self._active, d.pair_start, d.pair_id,
                              self._pair_active, self._count], device=dev)
            if self._seams is not None:
                wp.launch(seams.seam_impulse, dim=self._n_seams,
                          inputs=[self._seams, self._bond_intact, m.particle_mass, m.particle_inv_mass, d.pairs,
                                  self._pair_delta, self._pair_active, self._count,
                                  float(self.film.strain_relaxation)], device=dev)
            wp.launch(strain.vertex_apply, dim=n,
                      inputs=[d.vtx_start, d.vtx_tri, d.vtx_slot, d.particles, m.tri_indices, self._delta,
                              self._active, d.pair_start, d.pair_id, d.pair_end, d.pairs, self._pair_delta,
                              self._pair_active, self._count, float(self.film.strain_relaxation),
                              int(self.film.jacobi == "vertex"), x, self._dx], device=dev)
            wp.launch(strain.vertex_reaction, dim=n,
                      inputs=[d.vtx_start, d.vtx_tri, d.particles, m.tri_indices, self._delta, self._active,
                              d.pair_start, d.pair_id, d.pair_end, d.pairs, self._pair_delta, self._pair_active,
                              self._count, m.particle_inv_mass, float(self.film.strain_relaxation),
                              int(self.film.jacobi == "vertex"), 1.0 / (dt * dt), self.projection_reaction],
                      device=dev)
        if self._seams is not None:
            self._seam_force(x, dt)
        if grip:
            self._grip.clamp(d.particles, x, self._dx, dt)
        if self._skin is not None:                   # the projection keeps static contacts out of the skin (skin.py)
            self._skin.guard(m, x, self._dx, dt)
        wp.launch(strain.velocity_from_shift, dim=n, inputs=[d.particles, self._dx, 1.0 / dt, state_out.particle_qd],
                  device=dev)
        if self.film.ledger:
            wp.launch(ledger.projection_energy_kernel, dim=self.ledger.projection_dE.shape[0],
                      inputs=[d.world_start, d.particles, m.particle_mass, m.gravity, self._dx, state_out.particle_qd,
                              1.0 / dt, self.ledger.projection_dE, self.ledger.projection_gain_max], device=dev)
        self._dx.zero_()

    def export_air_work(self, health) -> None:
        """Adds the air work done since the last call to ``health``'s W_ext (``HealthMonitor.add_external_work``,
        float64 [W]) and zeroes it. Device only (capture-safe); call once per health check, after the step(s).
        The entry's worlds must be the monitor's worlds (film entries span all worlds)."""
        if not self._film_on or not self.film_topology.airborne:
            return
        health.add_external_work(self.ledger.air_since)
        self.ledger.air_since.zero_()

    # --- state ----------------------------------------------------------------------------------------------------

    def reset(self, state, world_mask=None, flags=None):
        """SolverVBD.reset, then the film state of the selected worlds (all if ``world_mask`` is None): authored
        rest angles, plastic work, ledger."""
        out = super().reset(state, world_mask, flags)
        if not self._film_on:
            return out
        dev = self.model.device
        if world_mask is None:
            if self._rest_angle0 is not None:
                wp.copy(self.model.edge_rest_angle, self._rest_angle0)
            self.plastic_work.zero_()
            if self._j2 is not None:
                self._j2.reset(self.model)
            self.ledger.zero()
            if self._seams is not None:
                self._seams.f_avg.zero_()
            return out
        if self._rest_angle0 is not None:
            wp.launch(_reset_edges, dim=self.model.edge_count,
                      inputs=[self.edge_world(), world_mask, self._rest_angle0, self.model.edge_rest_angle,
                              self.plastic_work], device=dev)
        if self._j2 is not None:
            self._j2.reset(self.model, self.edge_world(), world_mask)
        for a in (*self.ledger.arrays().values(), self.ledger.air_since):
            wp.launch(_reset_worlds, dim=a.shape[0], inputs=[world_mask, a], device=dev)
        if self._seams is not None:
            wp.launch(seams.zero_average, dim=self._n_seams, inputs=[self._seams, world_mask], device=dev)
        return out

    def film_state_arrays(self) -> dict[str, wp.array]:
        """Arrays the next step reads beyond SolverVBD's (snapshot, digest): plastic rest angles, the J2 hinges' layer
        state, the ledger and the seam force averages."""
        if not self._film_on:
            return {}
        out = {"film_plastic_work": self.plastic_work, **self.ledger.arrays()}
        if self.film_topology.plastic or self.film_topology.j2:
            out["film_edge_rest_angle"] = self.model.edge_rest_angle
        if self._j2 is not None:
            out.update(self._j2.state_arrays())
        if self._seams is not None:
            out["film_seam_force_avg"] = self._seams.f_avg
        return out

    @property
    def has_plastic_strain(self) -> bool:
        """True when this film carries J2 sheet hinges (``plastic_strain`` then reads them)."""
        return self._j2 is not None

    def plastic_strain(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Per particle of this solver's model: (largest J2 layer ε̄p along it, has a J2 hinge), or None without J2
        sheets (host read; ``"<sheet>.plastic_strain"`` in the API state)."""
        return None if self._j2 is None else self._j2.vertex_strain(self.model)

    def health_view(self):
        """(scratch [T], limited triangle world ranges [W+1], limited triangle ids) for the health adapter, or None
        when no triangle is limited."""
        if not self._film_on or not self.film_topology.limited:
            return None
        return self._ratio, self._dtop.tri_world_start, self._dtop.limited_tris

    def edge_world(self) -> wp.array | None:
        """Item→world int32 map of the per-edge arrays (world of the edge's first vertex), built once."""
        if not self._film_on:
            return None
        if getattr(self, "_edge_world", None) is None:
            m = self.model
            pw = m.particle_world.numpy() if m.particle_world is not None else np.zeros(m.particle_count, np.int32)
            ew = pw[m.edge_indices.numpy()[:, 2]] if m.edge_count else np.zeros(0, np.int32)
            self._edge_world = wp.array(ew.astype(np.int32), dtype=wp.int32, device=m.device)
        return self._edge_world

    def strain_ratio(self, state) -> np.ndarray:
        """Host read: (s1 − 1)/strain_cap per limited triangle (laws and reports; never inside a step)."""
        if not self.film_topology.limited:
            return np.zeros(0)
        out = wp.zeros(self.model.tri_count, dtype=float, device=self.model.device)
        wp.launch(strain.tri_max_strain, dim=self.model.tri_count,
                  inputs=[state.particle_q, self.model.tri_indices, self.model.tri_poses, self._ns.strain_cap, out],
                  device=self.model.device)
        return out.numpy()[self._ns.strain_cap.numpy() > 0.0]
