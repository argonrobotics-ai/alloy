# SPDX-License-Identifier: Apache-2.0
"""Opt-in solve accelerators of SolverVBDAlloy (XB4 L5.drape, WP2 2026-11-08): the adaptive initial guess and
Chebyshev-accelerated particle iterations. Both change only where a step's iterations start and how successive iterates
combine; the step's objective (the inertia target y = x + h v + h^2 g and the energies) is untouched, so a converged
step is the same with or without them. Off (the defaults), nothing here runs.

Adaptive initial guess (Chen et al. 2024, "Vertex Block Descent", sec. 3.6). Newton starts every step's iterate at the
inertia target, x + h v + h^2 a_ext (a_ext = g + f/m). With few iterations most of h^2 a_ext is never undone, so a sheet
that should hang or rest settles far past its equilibrium (XB4's drape at 10 x 10: 13.4 % membrane strain, converged
0.36 %; with this start 1.1 % at the same cost, L4, 8 worlds). The adaptive start keeps only the share of a_ext the
particle realised over its last step: start = (v + s a_ext h) h, s = clamp(a_last . a_ext / |a_ext|^2, 0, 1), a_last =
(v - v_last) / h_last the acceleration realised over the last step (h_last that step's dt: a coupled clock may change
dt). In free fall s = 1 (within 0.1 % the start is Newton's, bit for bit); at rest s = 0, so a static equilibrium is a
fixed point at any iteration count. A pinned piece under Newton's ``particle_pinned_piece_start`` keeps Newton's x + h
v. Until a particle has two steps of history it keeps Newton's start.

History. Per particle: the last step's input velocity (``_accel_v_cur``), that step's dt (``_accel_h_cur``), the
acceleration realised over the step before (``_accel_a_last``) and how many are valid (``_accel_hist``, 0-2). A step
shifts them only when its input velocity differs from ``_accel_v_cur``, so a re-step from the same input (coupled
entries step twice per exchange; a rollback to the step's start) reads the same history; every step writes its dt, so
the shift divides by the dt of the step that produced the new velocity. ``reset(state, world_mask)`` clears the masked
worlds' history (they restart from Newton's start). The four arrays are VBD per-particle state for snapshots
(``alloy.coupling.adapters`` VBD_PARTICLE), so a restored world starts from its own history bit for bit. (The dt was a
host value before 2026-11-08: after another world's refined re-run, a kept world's next start divided by the refined dt
and left its reference, tests/coupling/test_rollback_bitwise.py, WP11.)

Chebyshev semi-iterative acceleration (VBD sec. 3.8; Wang 2015). After each particle iteration (every colour swept,
callbacks and the momentum step included), x(n) = x(n-2) + w_n (x~(n) - x(n-2)), w_1 = 1, w_2 = 2 / (2 - rho^2), w_n = 4
/ (4 - rho^2 w_n-1). The mix runs on displacements from the last collision-detection positions x_cd, never on
positions: d(n) = d(n-2) + w_n (d~(n) - d(n-2)) with d = x - x_cd, the solver's own particle_displacements. A step's
per-iteration motion can sit far below float32's resolution of the positions (CARD-8's plate at z 0.5 m: 0.007 um
against a 0.06 um ulp), so a mix of positions rounded every update to zero and froze slow sheets in place (WP11,
2026-11-08: CARD-8 w = 0 exactly). The kept iterates d(n-1), d(n-2) are rebased whenever x_cd moves (a new collision
detection): d += x_cd_old - x_cd, a difference of nearby positions, exact. The extrapolated displacement goes through
the solver's own penetration-free truncation (static-shape DAT, self-contact planes, the isotropic bound), so the
conservative bound still holds. Particles in a body/shape contact record or an enabled body attachment
are not extrapolated (``AccelState.mark_held``): with them, WP4's pinch drag let the clamped cloth slip at a tenth of mu
N. Particles held only by cloth-on-cloth contact (self-contact: fold plies, a crease) are extrapolated unless
``hold_self_contact`` (``alloy_chebyshev_hold_self_contact=True``) holds every particle of a self-contact pair too, marked
at every collision detection of the step. Extrapolated plies keep a folded towel moving (known gap
cloth-fold-unsettled-default, WP11's thirds-fold and tee spreads, 2026-11-09). Per-step scratch only (d(n-1), d(n-2), x_cd at the last mix, the held flags). rho is an estimate of the sweep's spectral radius, and too
high a value over-relaxes: on the A17 film (15 um, the stiffest membrane case in tests/solvers/vbd) rho 0.97 does not
converge (2.5e-3 m off at 30 iterations), while 0.9-0.95 converge there and faster than stock (2.4e-6 vs 1.8e-5 m at 100
iterations). With the contact mask it adds little where the strain sits in contact (XB4's drape: 1.5 % with both flags
vs 1.1 % with the adaptive start alone). Use rho <= 0.95; 0.9 is the tested value.

All kernels are per particle, without atomics (deterministic in every Warp mode), with launch sizes fixed at
construction and the w_n schedule fixed per step (graph capture records it).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import warp as wp

from alloy._vendor.newton import ParticleFlags

INITIAL_GUESSES = ("inertial", "adaptive")
FULL_SHARE = 0.999          # a realised share at or above this keeps Newton's start exactly (free fall, bit for bit)


@dataclass(frozen=True)
class AccelConfig:
    """``initial_guess``: "inertial" (Newton's) or "adaptive"; ``chebyshev_rho``: None (off) or in (0, 1)."""

    initial_guess: str = "inertial"
    chebyshev_rho: float | None = None
    hold_self_contact: bool = False         # particles of a self-contact pair keep the plain iterate (module doc)

    def __post_init__(self) -> None:
        if self.initial_guess not in INITIAL_GUESSES:
            raise ValueError(f"alloy_initial_guess={self.initial_guess!r}, expected one of {INITIAL_GUESSES}")
        rho = self.chebyshev_rho
        if rho is not None and not 0.0 < float(rho) < 1.0:
            raise ValueError(f"alloy_chebyshev_rho={rho!r}: None (off) or a spectral radius estimate in (0, 1) "
                             "(<= 0.95 advised: accel.py)")

    @property
    def adaptive(self) -> bool:
        return self.initial_guess == "adaptive"


def chebyshev_weight(n: int, rho: float, prev: float) -> float:
    """w_n of the Chebyshev schedule (module docstring), n >= 1, ``prev`` = w_(n-1)."""
    if n == 1:
        return 1.0
    if n == 2:
        return 2.0 / (2.0 - rho * rho)
    return 4.0 / (4.0 - rho * rho * prev)


@wp.kernel
def shift_history(v_in: wp.array(dtype=wp.vec3), dt: float, v_cur: wp.array(dtype=wp.vec3),
                  a_last: wp.array(dtype=wp.vec3), hist: wp.array(dtype=wp.int32), h_cur: wp.array(dtype=float)):
    i = wp.tid()
    v, c, h = v_in[i], v_cur[i], hist[i]
    if h == 0 or v[0] != c[0] or v[1] != c[1] or v[2] != c[2]:
        if h > 0 and h_cur[i] > 0.0:
            a_last[i] = (v - c) / h_cur[i]
        v_cur[i] = v
        hist[i] = wp.min(h + 1, 2)
    h_cur[i] = dt                                    # this step's dt: the next shift's divisor


@wp.kernel
def adaptive_start(dt: float, gravity: wp.array(dtype=wp.vec3), particle_world: wp.array(dtype=wp.int32),
                   inv_mass: wp.array(dtype=float), external_force: wp.array(dtype=wp.vec3),
                   flags: wp.array(dtype=wp.int32), pinned_start: int, piece: wp.array(dtype=wp.int32),
                   piece_pinned: wp.array(dtype=wp.int32), v_cur: wp.array(dtype=wp.vec3),
                   a_last: wp.array(dtype=wp.vec3), hist: wp.array(dtype=wp.int32), lo: wp.array(dtype=wp.vec3),
                   disp: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    if hist[i] < 2 or (flags[i] & ParticleFlags.ACTIVE) == 0 or inv_mass[i] == 0.0:
        return
    if pinned_start != 0:
        if piece_pinned[piece[i]] != 0:
            return                                   # Newton started this pinned piece at x + h v
    a = gravity[particle_world[i]] + external_force[i] * inv_mass[i]
    a2 = wp.dot(a, a)
    if a2 == 0.0:
        return
    s = wp.clamp(wp.dot(a_last[i], a) / a2, 0.0, 1.0)
    if s >= FULL_SHARE:
        return                                       # free fall: Newton's start, bit for bit
    disp[i] = lo[i] + (v_cur[i] + a * (s * dt)) * dt     # from the true position: pos + its remainder (SCR-WP11-F32)


@wp.kernel
def clear_history(world_mask: wp.array(dtype=wp.bool), particle_world: wp.array(dtype=wp.int32),
                  hist: wp.array(dtype=wp.int32)):
    i = wp.tid()
    w = particle_world[i]
    if w < 0:
        w = world_mask.shape[0] - 1                  # Newton's mask: world_count + 1 entries, the last for world -1
    if world_mask[w]:
        hist[i] = 0


@wp.kernel
def mark_contact_records(count: wp.array(dtype=wp.int32), indices: wp.array(dtype=wp.vec3i),
                         held: wp.array(dtype=wp.int32)):
    k = wp.tid()
    if k >= count[0]:
        return
    ids = indices[k]
    for a in range(3):
        if ids[a] >= 0:
            held[ids[a]] = 1


@wp.kernel
def mark_attached(particle: wp.array(dtype=wp.int32), enabled: wp.array(dtype=wp.bool), held: wp.array(dtype=wp.int32)):
    k = wp.tid()
    if enabled[k]:
        held[particle[k]] = 1


@wp.kernel
def mark_counted(count: wp.array(dtype=wp.int32), held: wp.array(dtype=wp.int32)):
    """A vertex with a self-contact partner (its count of colliding triangles) is held."""
    v = wp.tid()
    if count[v] > 0:
        held[v] = 1


@wp.kernel
def mark_counted_rows(count: wp.array(dtype=wp.int32), rows: wp.array2d(dtype=wp.int32), first: int,
                      held: wp.array(dtype=wp.int32)):
    """An edge or triangle with a self-contact partner holds its vertices (row columns ``first``..)."""
    k = wp.tid()
    if count[k] > 0:
        for a in range(first, rows.shape[1]):
            held[rows[k, a]] = 1


@wp.kernel
def chebyshev_mix(omega: float, d1: wp.array(dtype=wp.vec3), d2: wp.array(dtype=wp.vec3),
                  pos_cd: wp.array(dtype=wp.vec3), cd_ref: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
                  flags: wp.array(dtype=wp.int32), held: wp.array(dtype=wp.int32), disp: wp.array(dtype=wp.vec3)):
    """disp = d(n-2) + omega (disp - d(n-2)), with the kept iterates first rebased to the current x_cd."""
    i = wp.tid()
    shift = cd_ref[i] - pos_cd[i]
    d1[i] = d1[i] + shift
    d2[i] = d2[i] + shift
    cd_ref[i] = pos_cd[i]
    if (flags[i] & ParticleFlags.ACTIVE) == 0 or inv_mass[i] == 0.0 or held[i] != 0:
        return
    disp[i] = d2[i] + omega * (disp[i] - d2[i])


@wp.kernel
def chebyshev_roll(disp: wp.array(dtype=wp.vec3), pos_cd: wp.array(dtype=wp.vec3), d1: wp.array(dtype=wp.vec3),
                   d2: wp.array(dtype=wp.vec3), cd_ref: wp.array(dtype=wp.vec3)):
    """d(n-2) <- d(n-1) (rebased to the current x_cd), d(n-1) <- this iteration's displacement."""
    i = wp.tid()
    d2[i] = d1[i] + (cd_ref[i] - pos_cd[i])
    d1[i] = disp[i]
    cd_ref[i] = pos_cd[i]


class AccelState:
    """The arrays and per-step counters of one solver's accelerators (module docstring)."""

    def __init__(self, cfg: AccelConfig, n: int, device) -> None:
        self.cfg, self.n, self.device = cfg, n, device
        z = (lambda: wp.zeros(n, dtype=wp.vec3, device=device)) if n else (lambda: None)
        self.v_cur, self.a_last = (z(), z()) if cfg.adaptive else (None, None)
        self.h_cur = wp.zeros(n, dtype=float, device=device) if cfg.adaptive and n else None   # its step's dt
        self.hist = wp.zeros(n, dtype=wp.int32, device=device) if cfg.adaptive and n else None
        cheb = cfg.chebyshev_rho is not None
        self.x1, self.x2, self.cd_ref = (z(), z(), z()) if cheb else (None, None, None)   # d(n-1), d(n-2), x_cd
        self.held = wp.zeros(n, dtype=wp.int32, device=device) if self.x1 is not None else None
        self.pending_dt: float | None = None     # set from the step's _initialize_particles to its first truncation
        self.it, self.w = 0, 1.0

    def set_rho(self, rho: float) -> None:
        """Chebyshev rho for the following steps (WP10's coupled-step scaling, alloy.coupling.vbd_iterations; validated
        by AccelConfig). Chebyshev must have been on at build: its scratch arrays exist only then. The w_n weights are host
        constants baked into a captured graph, so a caller must make rho a pure function of what the graph key carries.
        Between steps only: start_iterations resets the schedule (it, w), and a change inside a step would mix two."""
        if self.x1 is None:
            raise ValueError("Chebyshev was off when this solver was built (alloy_chebyshev_rho=None); it cannot be "
                             "turned on later: build it with a rho")
        self.cfg = dataclasses.replace(self.cfg, chebyshev_rho=float(rho))

    def begin_step(self, state_in, dt: float) -> None:
        """Shift the history from the step's input velocity; arm the adaptive start for the first truncation."""
        if self.hist is None:
            return
        wp.launch(shift_history, dim=self.n, inputs=[state_in.particle_qd, float(dt), self.v_cur, self.a_last,
                                                     self.hist, self.h_cur],
                  device=self.device)
        self.pending_dt = dt

    def apply_start(self, solver, state) -> None:
        """Overwrite the start displacement of particles with history (once per step, before its truncation)."""
        dt, self.pending_dt = self.pending_dt, None
        m = solver.model
        piece = getattr(solver, "particle_cloth_piece", None)
        pinned = getattr(solver, "piece_pinned", None)
        flag = bool(getattr(solver, "particle_pinned_piece_start", False))
        on = int(flag and piece is not None and pinned is not None)
        if not on:
            piece = pinned = self.hist                # unread when the flag is off (any int32 array)
        wp.launch(adaptive_start, dim=self.n,
                  inputs=[float(dt), m.gravity, m.particle_world, m.particle_inv_mass, state.particle_f,
                          m.particle_flags, on, piece, pinned, self.v_cur, self.a_last, self.hist,
                          solver._f32_lo_in],
                  outputs=[solver.particle_displacements], device=self.device)

    def start_iterations(self, solver, contacts) -> None:
        if self.x1 is None:
            return
        wp.copy(self.x1, solver.particle_displacements)   # d(0): the step's truncated start, from x_cd
        wp.copy(self.cd_ref, solver.pos_prev_collision_detection)
        self.it, self.w = 0, 1.0
        self.mark_held(solver.model, contacts)
        self.mark_self_contact(solver)

    def mark_held(self, model, contacts) -> None:
        """Particles of a body/shape contact record or an enabled body attachment keep the plain iterate: their
        friction is lagged and smoothed, and extrapolating it lets a clamp slip (WP4's pinch drag, 3 mm: 0.3-0.6 N
        sliding instead of mu N 12.6 N at rho 0.9-0.95)."""
        self.held.zero_()
        if contacts is not None and contacts.soft_contact_max > 0:
            wp.launch(mark_contact_records, dim=contacts.soft_contact_max,
                      inputs=[contacts.soft_contact_count, contacts.soft_contact_indices], outputs=[self.held],
                      device=self.device)
        if model.attachment_particle_body_count > 0:
            wp.launch(mark_attached, dim=model.attachment_particle_body_count,
                      inputs=[model.attachment_particle_body_particle, model.attachment_particle_body_enabled],
                      outputs=[self.held], device=self.device)

    def mark_self_contact(self, solver) -> None:
        """With ``hold_self_contact``: every particle of a self-contact pair from the solver's last detection is held
        too (adds to the step's held set; ``start_iterations`` clears it). Fixed launch sizes, no atomics."""
        if self.held is None or not self.cfg.hold_self_contact or not solver.particle_enable_self_contact:
            return
        info, m = solver.trimesh_collision_info, solver.model
        c = info.vertex_colliding_triangles_count
        wp.launch(mark_counted, dim=min(c.shape[0], self.n), inputs=[c], outputs=[self.held], device=self.device)
        for c, rows, first in ((info.edge_colliding_edges_count, m.edge_indices, 2),
                               (getattr(info, "triangle_colliding_vertices_count", None), m.tri_indices, 0)):
            if c is not None and rows is not None and rows.shape[0] > 0:
                wp.launch(mark_counted_rows, dim=min(c.shape[0], rows.shape[0]), inputs=[c, rows, first],
                          outputs=[self.held], device=self.device)

    def mix(self, solver, state) -> bool:
        """After one particle iteration: the extrapolated displacement, or False when w_n = 1 (nothing to do)."""
        self.it += 1
        self.w = chebyshev_weight(self.it, float(self.cfg.chebyshev_rho), self.w)
        mixed = self.it >= 2
        if mixed:
            m = solver.model
            wp.launch(chebyshev_mix, dim=self.n,
                      inputs=[self.w, self.x1, self.x2, solver.pos_prev_collision_detection, self.cd_ref,
                              m.particle_inv_mass, m.particle_flags, self.held],
                      outputs=[solver.particle_displacements], device=self.device)
        return mixed

    def roll(self, solver) -> None:
        wp.launch(chebyshev_roll, dim=self.n,
                  inputs=[solver.particle_displacements, solver.pos_prev_collision_detection, self.x1, self.x2,
                          self.cd_ref], device=self.device)

    def clear(self, model, world_mask) -> None:
        if self.hist is None:
            return
        if world_mask is None:
            self.hist.zero_()
            return
        wp.launch(clear_history, dim=self.n, inputs=[world_mask, model.particle_world, self.hist], device=self.device)

    def arrays(self) -> dict[str, wp.array]:
        """The per-particle history for snapshots (empty when the adaptive start is off)."""
        if self.hist is None:
            return {}
        return {"_accel_v_cur": self.v_cur, "_accel_a_last": self.a_last, "_accel_hist": self.hist,
                "_accel_h_cur": self.h_cur}


__all__ = ["FULL_SHARE", "INITIAL_GUESSES", "AccelConfig", "AccelState", "chebyshev_weight"]
