# SPDX-License-Identifier: Apache-2.0
"""VBD self contact leaves the liquid's and other contact-free entries' particles out of detection.

What went wrong. A coupled VBD entry runs on a model view that keeps every particle of the scene: the entry's own
(ACTIVE in the view) and the other entries' (the liquid's, cleared ACTIVE). Newton's vertex-triangle self-contact
detection (``TriMeshCollisionDetector.vertex_triangle_collision_detection``) launches over ``model.particle_count``
and assumes every vertex lies on the entry's triangles. So it pairs SPH liquid particles with cloth triangles. SPH
rests its water about 1.15 mm above the cloth (its soft wall has zero thickness), inside VBD's 2 mm self-contact
margin. VBD's penalty (ke 2e6 N/m) then pushes the cloth off a zero-mass particle that VBD does not move: 16,780 N
on a 0.28 g vertex at the stream's first contact. The sheet leaves at the DAT cap, and the world goes INVALID
(ENERGY_JUMP, often SPEED_FAIL) 0.02-0.05 s after the stream lands. The self-contact stiffness cap of the default
cloth accuracy delays that failure; it does not stop it. A dry sheet, a pour landing beside it, and
``self_contact=False`` all stay OK; so does the same run with only the self-contact force kernel skipped. That last
run matches ``self_contact=False`` to 4 digits.

The fix. Self-contact detection takes positions in which every foreign particle stands at one far point (``FAR``).
Foreign means: a particle of a contact-free entry (SPH or MPM: ``CONTACT_FREE_DESTINATIONS``, the set the fallback
contacts already give no soft candidates), not ACTIVE in the entry's view, and on none of the view's triangles or
edges. ``make_entries`` attaches those entries' particle ids to the vbd entry's solver kwargs (``FREE_KEY``, as id
ranges), and ``make_vbd`` passes them here. The set is fixed at install, as one static int8 array. Particles of
another VBD-type entry (a film sheet beside a cloth: two entries) are never masked, so whatever contact they make
through self contact today is kept. A non-ACTIVE vertex that is on a triangle or edge keeps its position: moved
to FAR, its triangles would span 1e7 m and list wrong pairs. Today VBD pins by zero mass, not by the ACTIVE flag, so
in a coupled view the foreign set is exactly the liquid's or MPM's particles. At FAR a particle meets no triangle and
lists no pair. The detector's ``refit`` is wrapped
per solver instance, so every caller sees the masked positions: Newton's detections, the rows census
(self_contact_rows.py) and the crossing watcher (alloy.precision.self_contact). The entry's own particles keep
their positions bit for bit. The solve, the truncation reference (``pos_prev_collision_detection``) and the state
are untouched. The masking is one kernel launch into a preallocated array, so it is graph-safe: no host read
inside a step.

Scope. The filter installs only on a solver with particle self contact on, Newton's trimesh detector (no
``collision_pipeline``) and at least one foreign particle. Without contact-free ids (a solver built outside the
scene assembler) it is a no-op. Cloth-only scenes and every other solver
are unchanged. ``ENV`` set to ``"off"`` keeps the old detection, for A/B runs.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor.newton import ParticleFlags

log = logging.getLogger("alloy.coupling")

FAR = 1.0e7  # [m] where foreign particles stand for self-contact detection: beyond any scene, finite in float32
ENV = "ALLOY_VBD_FOREIGN_SELF_CONTACT"  # "off": the old detection over every particle of the view (A/B only)
FREE_KEY = "alloy_contact_free_particles"  # vbd solver kwarg: ((start, stop), ...) id ranges of SPH / MPM particles
_ATTR = "_alloy_foreign_filter"


@wp.kernel(enable_backward=False)
def mask_foreign_kernel(q: wp.array[wp.vec3], foreign: wp.array[wp.int8], far: float, out: wp.array[wp.vec3]):
    """out[i] = the far point for a foreign particle (module doc), q[i] for every other particle."""
    i = wp.tid()
    if foreign[i] != 0:
        out[i] = wp.vec3(far, far, far)
    else:
        out[i] = q[i]


class ForeignParticleFilter:
    """Wraps one solver's ``trimesh_collision_detector.refit`` so detection sees masked positions (module doc)."""

    def __init__(self, solver: Any, mask: np.ndarray) -> None:
        self.solver = solver
        self.foreign = int(np.count_nonzero(mask))
        self.mask = wp.array(mask.astype(np.int8), dtype=wp.int8, device=solver.model.particle_q.device)
        self.detector = solver.trimesh_collision_detector
        self.scratch = wp.empty_like(solver.model.particle_q)
        self._refit = self.detector.refit
        self.detector.refit = self.refit

    def masked(self, particle_q: wp.array) -> wp.array:
        """``particle_q`` with the foreign particles at the far point (into the preallocated scratch array)."""
        wp.launch(mask_foreign_kernel, dim=self.solver.model.particle_count, inputs=[particle_q, self.mask, FAR],
                  outputs=[self.scratch], device=self.scratch.device)
        return self.scratch

    def refit(self, new_pos: wp.array | None = None) -> None:
        self._refit(None if new_pos is None else self.masked(new_pos))


def _mesh_vertices(m: Any) -> np.ndarray:
    """(particle_count,) bool: the particle is a vertex of one of the model's triangles or edges."""
    on = np.zeros(m.particle_count, bool)
    for name in ("tri_indices", "edge_indices"):
        idx = getattr(m, name, None)
        if idx is not None and getattr(idx, "shape", (0,))[0] > 0:
            v = idx.numpy().ravel()
            on[v[v >= 0]] = True
    return on


def id_ranges(ids: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Sorted unique ids as half-open (start, stop) runs: a compact, hashable solver kwarg."""
    ids = np.unique(np.asarray(ids, np.int64))
    if ids.size == 0:
        return ()
    cut = np.flatnonzero(np.diff(ids) != 1) + 1
    return tuple((int(r[0]), int(r[-1]) + 1) for r in np.split(ids, cut))


def attach_contact_free(specs: list) -> None:
    """Give the vbd entry's solver kwargs the particle ids of the scene's contact-free entries (SPH, MPM)."""
    from alloy.coupling.proxy_contacts import CONTACT_FREE_DESTINATIONS  # noqa: PLC0415

    free = [np.asarray(e.particles, np.int64) for e in specs
            if e.capability in CONTACT_FREE_DESTINATIONS and len(e.particles)]
    if not free:
        return
    ranges = id_ranges(np.concatenate(free))
    for e in specs:
        if e.capability == "vbd":
            e.solver_kwargs[FREE_KEY] = ranges


def foreign_mask(solver: Any, free: tuple[tuple[int, int], ...] | None) -> np.ndarray:
    """(particle_count,) bool: foreign particles of the solver's model (module doc): in a contact-free entry's id
    ranges, not ACTIVE, on no triangle or edge. Non-ACTIVE mesh vertices among them are logged and kept."""
    m = solver.model
    out = np.zeros(m.particle_count, bool)
    if not free or m.particle_count == 0 or m.particle_flags is None:
        return out
    for a, b in free:
        out[max(int(a), 0):min(int(b), m.particle_count)] = True
    out &= (m.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) == 0
    on_mesh = _mesh_vertices(m)
    kept = int(np.count_nonzero(out & on_mesh))
    if kept:
        log.warning("VBD self contact: %d contact-free particles lie on the entry's triangles or edges and stay in "
                    "detection", kept)
    return out & ~on_mesh


def foreign_count(solver: Any, free: tuple[tuple[int, int], ...] | None) -> int:
    """How many particles the filter would leave out of the solver's self-contact detection."""
    return int(np.count_nonzero(foreign_mask(solver, free)))


def install_foreign_filter(solver: Any,
                           free: tuple[tuple[int, int], ...] | None = None) -> ForeignParticleFilter | None:
    """Install the filter on a VBD solver when it applies (module doc, Scope); return it, or None. ``free``: the
    contact-free entries' particle id ranges (``FREE_KEY``); None or empty: no-op."""
    if getattr(solver, _ATTR, None) is not None:
        return getattr(solver, _ATTR)
    if os.environ.get(ENV, "").strip().lower() == "off":
        return None
    if not getattr(solver, "particle_enable_self_contact", False) or getattr(solver, "collision_pipeline", None):
        return None
    if getattr(solver, "trimesh_collision_detector", None) is None:
        return None
    mask = foreign_mask(solver, free)
    n = int(np.count_nonzero(mask))
    if n == 0:
        return None
    filt = ForeignParticleFilter(solver, mask)
    setattr(solver, _ATTR, filt)
    log.info("VBD self contact: %d liquid / MPM particles left out of detection", n)
    return filt


__all__ = ["ENV", "FAR", "FREE_KEY", "ForeignParticleFilter", "attach_contact_free", "foreign_count", "foreign_mask",
           "id_ranges", "install_foreign_filter", "mask_foreign_kernel"]
