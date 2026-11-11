# SPDX-License-Identifier: Apache-2.0
"""The film's contact skin limit at static shapes (docs/known_gaps/film-thin-sheet-box-edge.md).

Two stages pushed a thin film draped over a static box's edge into the box, each measured on the contact records' own
planes (the distance along the record's normal, as health measures it):
- **VBD.** Newton's VBD gives a particle-shape contact no Hessian while the particle is outside its skin (distance
  above its contact radius). A film vertex just outside took its Newton step with the membrane's and inertia's small
  Hessian alone, overshot into the shape, and the contact's full stiffness threw it back in the next iteration; the
  step ended wherever the last iteration left it. 50 um LDPE over a box at ke 2e6: 0.23 mm into the 0.5 mm skin at 10
  iterations, 0.16 mm at 200, non-monotone in the count.
- **The strain projection** moves vertices after VBD and ignored contacts with no force yet: it pulled the taut library
  sheet over the cube's edge from 0.001 mm (VBD, limited here) to 0.52 mm deep (PEN_FAIL).

``SkinLimit.apply`` runs in each colour's truncation (before the static-shape DAT and Newton's own truncation): a film
vertex that starts the iteration outside its skin against a static shape's record may end it no deeper than
``(1 - entry)`` of the skin, so the next iteration has the contact in its Hessian. A vertex inside the skin (one this
limit placed there, or one loaded deeper) is left to VBD: the limit holds no load. The
truncations that follow scale the displacement toward the step's reference, which keeps the limit wherever the
reference is outside it. ``SkinLimit.guard`` runs after the projection: a contact vertex VBD left at the limit or above
(no load through contact depth yet) ends no deeper than GUARD_DEPTH (10 %) of the skin; a loaded vertex is left as
before. 1 um for the guard too narrowed a roller's contact patch on a tube 6 % (the Bonds C-1 law: a lightly pressed
patch edge is the projection's to set); 10 % keeps it (and the library sheet at 0.05 mm, PEN_FAIL is at 0.5 mm).
Both push the excess back out along the record's normal. A particle has one record per static shape in
reach: of its records only the deepest pushes, since two coplanar records (a seam between two boxes) would push it out
twice; the guard runs twice for a vertex deep in two planes at once (a box's floor and wall). Dynamic bodies are not
covered (their planes move with the body; grip.py guards contacts that carry force).

Cost: once per step ``begin_step`` links each film particle's static records into a list (one launch over the records;
the list order is atomic but the deepest-record choice is order independent, ties to the lowest record); each colour's
truncation is then one launch over that colour's particles, and the guard one per pass over the film particles. Fixed
launch sizes, no float atomics. ``reaction``: the static shapes' force on each particle for the guard's pushes
(m d / dt^2, N), the share a support's load read from contact depths misses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy._vendor.newton._src.geometry.soft_contacts_mesh import (_make_soft_contact_filter_data,
                                                                   _soft_contact_force_enabled, _SoftContactFilterData)

GUARD_PASSES = 2            # after the projection: a vertex deep in two orthogonal planes (a box's floor and wall)
GUARD_DEPTH = 0.1           # after the projection: no deeper than this fraction of the skin (VBD's own depth kept)
HELD_ENTRIES = 2.0          # the guard covers vertices VBD left within this many entry depths of the skin


@wp.kernel(enable_backward=False)
def link_records(count: wp.array[wp.int32], max_count: int, corners: wp.array[wp.vec3i], shape: wp.array[wp.int32],
                 shape_body: wp.array[wp.int32], film: wp.array[wp.int32], flt: _SoftContactFilterData,
                 head: wp.array[wp.int32], nxt: wp.array[wp.int32]):
    """Per record: a static-shape particle record with force of a film particle joins that particle's list."""
    c = wp.tid()
    if c >= wp.min(count[0], max_count):
        return
    ids = corners[c]
    p = ids[0]
    if ids[1] >= 0 or p < 0 or film[p] == 0:
        return
    if shape_body[shape[c]] >= 0 or not _soft_contact_force_enabled(c, flt):
        return
    nxt[c] = wp.atomic_exch(head, p, c)  # alloy: det-ok list order only; readers take the deepest, ties to lowest c


@wp.func
def _skin(p: int, s: int, radius: wp.array[float], shape_margin: wp.array[float]) -> float:
    """The contact's skin: VBD evaluates it (force and Hessian) only below this distance."""
    margin = float(0.0)
    if shape_margin.shape[0] > 0:
        margin = shape_margin[s]
    return radius[p] + margin


@wp.func
def _limit(p: int, s: int, radius: wp.array[float], shape_margin: wp.array[float], entry: float) -> float:
    return _skin(p, s, radius, shape_margin) * (1.0 - entry)


@wp.kernel(enable_backward=False)
def skin_colour(group: wp.array[wp.int32], head: wp.array[wp.int32], nxt: wp.array[wp.int32],
                normal: wp.array[wp.vec3], body_pos: wp.array[wp.vec3], shape: wp.array[wp.int32],
                radius: wp.array[float], shape_margin: wp.array[float], x: wp.array[wp.vec3], ref: wp.array[wp.vec3],
                disp: wp.array[wp.vec3], entry: float):
    """Per particle of the colour: a vertex outside its skin ends no deeper than the limit of its deepest record. A
    vertex inside (at the limit after an earlier push, or loaded deeper) is VBD's, with the contact in its Hessian."""
    p = group[wp.tid()]
    c = head[p]
    best = float(0.0)
    pick = int(-1)
    while c >= 0:
        n = normal[c]
        if wp.dot(n, x[p] - body_pos[c]) >= _skin(p, shape[c], radius, shape_margin):   # outside: no Hessian yet
            e = _limit(p, shape[c], radius, shape_margin, entry) - wp.dot(n, ref[p] + disp[p] - body_pos[c])
            if e > best or (e == best and pick >= 0 and c < pick):
                best = e
                pick = c
        c = nxt[c]
    if pick >= 0:
        disp[p] = disp[p] + normal[pick] * best


@wp.kernel(enable_backward=False)
def guard_particles(particles: wp.array[wp.int32], head: wp.array[wp.int32], nxt: wp.array[wp.int32],
                    normal: wp.array[wp.vec3], body_pos: wp.array[wp.vec3], shape: wp.array[wp.int32],
                    radius: wp.array[float], shape_margin: wp.array[float], entry: float, mass: wp.array[float],
                    inv_dt2: float, x: wp.array[wp.vec3], dx: wp.array[wp.vec3], reaction: wp.array[wp.vec3]):
    """Per film particle after the projection: a vertex VBD left at the entry limit or above (within HELD_ENTRIES
    entries of the skin: it carries no load through contact depth yet) ends no deeper than GUARD_DEPTH of the skin at
    its deepest record (the projection may press it in that far, as under a roller's light patch edge).
    A loaded vertex VBD left deeper is VBD's and grip.py's, as without the limit (a roller's or
    a support's load reads as contact depth)."""
    p = particles[wp.tid()]
    c = head[p]
    best = float(0.0)
    pick = int(-1)
    while c >= 0:
        n = normal[c]
        d1 = wp.dot(n, x[p] - body_pos[c])
        dv = d1 - wp.dot(n, dx[p])                          # where VBD left it
        skin = _skin(p, shape[c], radius, shape_margin)
        e = float(0.0)
        if dv >= skin * (1.0 - HELD_ENTRIES * entry):
            e = wp.min(dv, skin * (1.0 - GUARD_DEPTH)) - d1
        if e > best or (e == best and pick >= 0 and c < pick):
            best = e
            pick = c
        c = nxt[c]
    if pick >= 0:
        d = normal[pick] * best
        x[p] = x[p] + d
        dx[p] = dx[p] + d
        reaction[p] = reaction[p] + d * (mass[p] * inv_dt2)


class SkinLimit:
    """The film particles, the per-step record lists and the guard's reaction (module docstring)."""

    def __init__(self, model: Any, film_particles: np.ndarray, entry: float) -> None:
        P, dev = int(model.particle_count), model.device
        mask = np.zeros(P, np.int32)
        idx = np.asarray(film_particles, np.int64)
        mask[idx] = 1
        self.entry = float(entry)
        self.film = wp.array(mask, dtype=wp.int32, device=dev)
        self.particles = wp.array(idx.astype(np.int32), dtype=wp.int32, device=dev)
        self.head = wp.full(P, -1, dtype=wp.int32, device=dev)
        self.nxt = wp.zeros(0, dtype=wp.int32, device=dev)
        self.reaction = wp.zeros(P, dtype=wp.vec3, device=dev)      # N, the last step's guard
        self.contacts = None                                        # the step's contacts once linked

    def begin_step(self, model: Any, contacts: Any, x: wp.array, body_q: Any = None) -> None:
        """Before VBD: the step's static records per film particle."""
        self.reaction.zero_()
        self.contacts = None
        if contacts is None or contacts.soft_contact_max == 0:
            return
        if self.nxt.shape[0] < contacts.soft_contact_max:
            self.nxt = wp.zeros(contacts.soft_contact_max, dtype=wp.int32, device=model.device)
        flt = _make_soft_contact_filter_data(model, contacts, x, body_q if model.body_count else None)
        self.head.fill_(-1)
        wp.launch(link_records, dim=contacts.soft_contact_max,
                  inputs=[contacts.soft_contact_count, contacts.soft_contact_max, contacts.soft_contact_indices,
                          contacts.soft_contact_shape, model.shape_body, self.film, flt],
                  outputs=[self.head, self.nxt], device=model.device)
        self.contacts = contacts

    def _geometry(self, model: Any) -> list:
        c = self.contacts
        return [self.head, self.nxt, c.soft_contact_normal, c.soft_contact_body_pos, c.soft_contact_shape,
                model.particle_radius, model.shape_margin]

    def apply(self, solver: Any, state: Any, group: Any = None) -> None:
        """In a colour's truncation: ``solver.particle_displacements`` (from ``solver.pos_prev_collision_detection``)
        of ``group``'s particles (all film particles when None); ``state`` holds the iterate before the step."""
        if self.contacts is None:
            return
        group = self.particles if group is None else group
        wp.launch(skin_colour, dim=group.shape[0],
                  inputs=[group, *self._geometry(solver.model), state.particle_q, solver.pos_prev_collision_detection,
                          solver.particle_displacements, self.entry], device=solver.model.device)

    def guard(self, model: Any, x: wp.array, dx: wp.array, dt: float) -> None:
        """After the projection: ``x`` and its shift ``dx`` pushed back out where the projection took a contact vertex
        deeper than VBD left it, past the limit; the push as a force into ``reaction``."""
        if self.contacts is None:
            return
        for _ in range(GUARD_PASSES):
            wp.launch(guard_particles, dim=self.particles.shape[0],
                      inputs=[self.particles, *self._geometry(model), self.entry, model.particle_mass,
                              1.0 / (float(dt) * float(dt)), x, dx, self.reaction], device=model.device)


__all__ = ["GUARD_DEPTH", "GUARD_PASSES", "HELD_ENTRIES", "SkinLimit", "guard_particles", "link_records", "skin_colour"]
