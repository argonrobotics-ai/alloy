# SPDX-License-Identifier: Apache-2.0
"""SELF_CONTACT_OVERFLOW: VBD self-contact pairs dropped over their fixed buffers (known gap
``docs/known_gaps/vbd-self-contact-buffer-overflow.md``).

Newton's VBD self-contact detection (``TriMeshCollisionDetector``) keeps each vertex's colliding triangles and each
edge's colliding edges in fixed rows (``particle_vertex_contact_buffer_size`` 32, ``particle_edge_contact_buffer_size``
64; triangle rows when recorded) for every pair in the entry, within one cloth or soft body and between two. It
counts every pair it finds but writes only up to the row's size, and a pair beyond it is never resolved, so the two
surfaces can pass into each other (``TriMeshCollisionInfo``: "counts may exceed a row's capacity").

After every detection (``SolverVBD._collision_detection_penetration_free``, wrapped per solver instance) one kernel
per row family compares each row's count with its size and keeps, per world, the most pairs one row dropped
(``dropped``, int32 [W]). Health reads it as a ``nonzero`` capacity source with the SELF_CONTACT_OVERFLOW bit
(DEGRADED). The value is sticky for the run, as CLOTH_STRETCH's peak, until ``reset(world_mask)`` (the source's
``reset``, for HealthMonitor's reset of those worlds; a rollback restore keeps it). Rows of global particles (world
-1) cannot be attributed and flag every world (fail closed, as Newton's global contact counts in
``health_adapters``). The launches use arrays fixed at attach time, so graph capture records them.

Dropped pairs are often harmless (a calm stack with rows of one pair stays apart). On request the same pass also
marks the particles of every row that dropped (``marked``), and after each detection counts, per world, the triangle
pairs that then intersect with a marked vertex on one of them, plus the marked vertices that changed sides of a
triangle since the detection before (``crossed``; the law and the tests are in ``self_contact_cross.py``).
``crossed > 0`` is a real pass-through. ``CheckPolicy.self_contact_cross`` (or ``ALLOY_SELF_CONTACT_CROSS`` at build)
says what to do: ``"off"`` (the default: no meter at all, every step as before), ``"report"`` (counted and shown in
the health row, the world stays DEGRADED) or ``"invalid"`` (such a world is INVALID).
"""

from __future__ import annotations

import logging
import os
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.precision.self_contact_cross import CrossMeter, row_particles
from alloy.precision.types import HealthFlags

log = logging.getLogger("alloy.precision")

WATCH_ATTR = "_alloy_self_contact_watch"
CROSS_ENV = "ALLOY_SELF_CONTACT_CROSS"      # off | report | invalid: overrides CheckPolicy.self_contact_cross
CROSS_MODES = ("off", "report", "invalid")
# (count, buffer sizes, the element array that names each row's particles, the column holding a row's own particle)
ROW_FAMILIES: tuple[tuple[str, str, str | None, int], ...] = (
    ("vertex_colliding_triangles_count", "vertex_colliding_triangles_buffer_sizes", None, 0),
    ("edge_colliding_edges_count", "edge_colliding_edges_buffer_sizes", "edge_indices", 2),     # (o0, o1, v0, v1)
    ("triangle_colliding_vertices_count", "triangle_colliding_vertices_buffer_sizes", "tri_indices", 0),
)
# per element array: the columns holding the particles a dropped pair of that row marks (an edge's two ends)
MARK_COLUMNS: dict[str, tuple[int, ...]] = {"edge_indices": (2, 3), "tri_indices": (0, 1, 2)}


@wp.kernel
def _dropped_pairs(count: wp.array[wp.int32], size: wp.array[wp.int32], row_world: wp.array[wp.int32],
                   world_count: int, dropped: wp.array[wp.int32]):
    i = wp.tid()
    d = count[i] - size[i]
    if d <= 0:
        return
    w = row_world[i]
    if w >= 0 and w < world_count:
        wp.atomic_max(dropped, w, d)
    else:                                           # a global particle's row: no world to blame, so every world
        for k in range(world_count):
            wp.atomic_max(dropped, k, d)


@wp.kernel
def _clear_masked(mask: wp.array[wp.int32], dropped: wp.array[wp.int32]):
    w = wp.tid()
    if mask[w] != 0:
        dropped[w] = 0


@dataclass(frozen=True)
class _Rows:
    count: str
    size: str
    world: wp.array            # int32 [rows]: the world of each row
    particles: wp.array        # int32 [rows, 3]: the particles a dropped pair of the row marks (-1: none)


class SelfContactWatch:
    """Per-world dropped self-contact pairs of one VBD solver (module docstring). ``dropped[w]``: the most pairs one
    row of world ``w`` dropped at any detection so far (0: none). With the pass-through meter on (``enable_cross``),
    ``marked`` (int32 [particles]) and ``crossed`` (int32 [W]): the particles of rows that dropped, and the most
    crossings one detection then counted; None while it is off."""

    cross: CrossMeter | None = None                 # the pass-through meter, only on request (enable_cross)
    invalid = False                                 # crossed > 0 makes the world INVALID (set by health at build)

    def __init__(self, solver: newton.solvers.SolverVBD, world_count: int) -> None:
        self.solver, self.world_count = solver, int(world_count)
        self.device = solver.model.device
        self.dropped = wp.zeros(max(self.world_count, 1), dtype=wp.int32, device=self.device)
        self._rows = [r for r in (self._row_family(*fam) for fam in ROW_FAMILIES) if r is not None]

    def _row_family(self, count: str, size: str, elements: str | None, column: int) -> _Rows | None:
        info = self.solver.trimesh_collision_info
        c, s = getattr(info, count, None), getattr(info, size, None)
        if c is None or s is None or c.shape[0] == 0 or c.shape[0] != s.shape[0]:
            return None
        pworld = _particle_world(self.solver.model)
        table = None
        if elements is not None:
            idx = getattr(self.solver.model, elements, None)
            if idx is None or idx.shape[0] != c.shape[0]:
                log.warning("self_contact: %s has %d rows but the model has no matching %s; rows not watched",
                            count, c.shape[0], elements)
                return None
            table = idx.numpy()
        rows = pworld if table is None else pworld[table[:, column]]
        marks = row_particles(c.shape[0], table, MARK_COLUMNS.get(elements or "", ()))
        return _Rows(count, size, wp.array(rows.astype(np.int32), dtype=wp.int32, device=self.device),
                     wp.array(marks, dtype=wp.int32, device=self.device))

    @property
    def marked(self) -> wp.array | None:
        return None if self.cross is None else self.cross.marked

    @property
    def crossed(self) -> wp.array | None:
        return None if self.cross is None else self.cross.crossed

    def enable_cross(self, invalid: bool) -> None:
        """Turn the pass-through meter on (health, at build, before any capture); ``invalid``: crossings make the
        world INVALID, else they are only reported."""
        if self.cross is None:
            model = self.solver.model
            self.cross = CrossMeter(_triangles(model), _particle_world(model), self.world_count, self.device,
                                    start=getattr(model, "particle_q", None))
        self.invalid = self.invalid or bool(invalid)

    def state_arrays(self) -> dict[str, wp.array]:
        """What an engine rollback restores per world with health's own arrays (empty with the meter off)."""
        return {} if self.cross is None else self.cross.state_arrays()

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Forget the dropped pairs of the masked worlds (all when None; int32 [>= W], nonzero = reset): a reset
        starts a new run (WP5, gap #13 review). A rollback restore keeps them (a sticky warning). The crossings
        and the marks of those worlds' particles go with them."""
        if self.cross is not None:
            self.cross.reset(world_mask)
        if world_mask is None:
            self.dropped.zero_()
            return
        wp.launch(_clear_masked, dim=min(int(world_mask.shape[0]), self.dropped.shape[0]),
                  inputs=[world_mask], outputs=[self.dropped], device=self.device)

    def keep_period(self) -> None:
        """The marks and crossings as a period starts (``row_growth`` keeps ``dropped`` itself), for
        ``restore_period`` when the period re-runs on grown rows: the abandoned attempt is no period's."""
        if self.cross is not None:
            self.cross.keep()

    def restore_period(self) -> None:
        if self.cross is not None:
            self.cross.restore()

    def after_detection(self) -> None:
        """Fold the latest detection's rows into ``dropped`` (and, with the meter on, ``marked``, then its crossings
        into ``crossed``); launches only, no host read."""
        info = self.solver.trimesh_collision_info   # read each call: a collision pipeline may rebind it
        cross = self.cross
        if cross is not None:
            cross.begin()                           # worlds an engine rollback restored since the last detection
        for r in self._rows:
            count, size = getattr(info, r.count), getattr(info, r.size)
            wp.launch(_dropped_pairs, dim=count.shape[0], inputs=[count, size, r.world, self.world_count],
                      outputs=[self.dropped], device=self.device)
            if cross is not None:
                cross.mark(count, size, r.particles)
        det = detector(self.solver) if cross is not None else None   # its triangle BVH was refit just before
        if det is not None:
            cross.count(det.bvh_tris.id, det.bvh_tris_group_roots, det.vertex_positions,
                             pad=float(getattr(self.solver, "_self_contact_query_radius", 0.0)))


def _particle_world(model: Any) -> np.ndarray:
    pw = getattr(model, "particle_world", None)
    if pw is None:
        return np.zeros(int(model.particle_count), dtype=np.int32)
    return pw.numpy().astype(np.int32)


def _triangles(model: Any) -> wp.array:
    tri = getattr(model, "tri_indices", None)
    return tri if tri is not None else wp.zeros((0, 3), dtype=wp.int32, device=model.device)


def detector(solver: Any) -> Any:
    """The ``TriMeshCollisionDetector`` the solver's detection runs: its own, or its collision pipeline's."""
    pipeline = getattr(solver, "collision_pipeline", None)
    if pipeline is not None:
        return getattr(pipeline, "_soft_self_contact_detector", None)
    return getattr(solver, "trimesh_collision_detector", None)


def cross_mode(policy: Any = None) -> str:
    """The pass-through meter's mode: ``"off"`` (no meter), ``"report"`` or ``"invalid"``.
    ``CheckPolicy.self_contact_cross``, unless ``CROSS_ENV`` names one of the three."""
    raw = os.environ.get(CROSS_ENV, "").strip().lower()
    if raw in CROSS_MODES:
        return raw
    if raw:
        log.warning("self_contact: %s=%r is none of %s; ignored", CROSS_ENV, raw, "/".join(CROSS_MODES))
    mode = str(getattr(policy, "self_contact_cross", "off")).strip().lower()
    return mode if mode in CROSS_MODES else "off"


def _meters(solvers: Any) -> list[SelfContactWatch]:
    found = (getattr(s, WATCH_ATTR, None) for s in solvers)
    return [w for w in found if w is not None and w.cross is not None]


def state_arrays(solvers: Any) -> dict[str, wp.array]:
    """For ``HealthMonitor.state_arrays``: each meter's per-world arrays, so a world's rollback restores them."""
    return {f"health.self_contact_{i}_{k}": a
            for i, w in enumerate(_meters(solvers)) for k, a in w.state_arrays().items()}


def report_fields(solvers: Any) -> dict[str, Any]:
    """For ``HealthReport``: per world, the most crossings one detection counted over the solvers' meters, and
    whether they make the world INVALID. Empty with no meter on."""
    watches = _meters(solvers)
    if not watches:
        return {}
    crossed = np.max([w.crossed.numpy() for w in watches], axis=0).astype(np.int64)
    return {"self_contact_crossed": crossed, "self_contact_cross_invalid": any(w.invalid for w in watches)}


# What health and sim.check() say for a world whose surfaces crossed (api/health.py reads these).
CROSS_WHAT = ("more contact pairs than the contact rows hold were dropped, and the surfaces then passed through each "
              "other")
CROSS_FIX = ("lower the stack or fold depth, or the speed; lower Scene(dt=...); at the engine layer, give the contact "
             "rows more room (alloy_self_contact_row_bytes in the VBD entry's kwargs)")


def cross_row(report: Any, w: int) -> dict[str, Any] | None:
    """A health row's ``self_contact``: the most triangle pairs one detection found crossed in world ``w`` after
    their contact rows dropped pairs (None: none crossed, or no meter). ``level`` "fail": the world is INVALID."""
    n = getattr(report, "self_contact_crossed", None)
    if n is None or w >= len(n) or int(n[w]) <= 0:
        return None
    return {"crossed_pairs": int(n[w]), "level": "fail" if report.self_contact_cross_invalid else "warn"}


def cross_text(c: dict[str, Any]) -> str:
    return f"{CROSS_WHAT} ({c['crossed_pairs']} crossing triangle pairs)"


def cross_problems(h: dict[str, Any], listed: Any = ()) -> list[dict[str, Any]]:
    """``sim.check()`` problems for worlds whose surfaces passed through each other after their self-contact rows
    dropped pairs (``self_contact`` level "fail"), except worlds a problem in ``listed`` already says it for.

    Example: ``cross_problems(sim.health())`` -> ``[{"world": 1, "check": "self_contact", "level": "fail",
    "what": "more contact pairs ... passed through each other (12 crossing triangle pairs)", ...}]``."""
    out = []
    for w, row in enumerate(h.get("worlds", [])):
        c = row.get("self_contact")
        if c is None or c["level"] != "fail" or any(
                p.get("world") == w and "passed through" in str(p.get("what", "")) for p in listed):
            continue
        out.append({"world": w, "level": "fail", "check": "self_contact", "severity": row["severity"],
                    "what": cross_text(c), "fix": CROSS_FIX})
    return out


def _detect_and_watch(self: Any, current_state: Any, *, reset_reference: bool = True) -> None:
    type(self)._collision_detection_penetration_free(self, current_state, reset_reference=reset_reference)
    getattr(self, WATCH_ATTR).after_detection()


def watch(solver: Any, world_count: int | None = None) -> SelfContactWatch | None:
    """The solver's watch, attached on first call (idempotent); None for a solver without VBD self contact. The
    wrapper is set on the instance only: other solvers and the class stay stock."""
    if not isinstance(solver, newton.solvers.SolverVBD) or not getattr(solver, "particle_enable_self_contact", False):
        return None
    if getattr(solver, "trimesh_collision_info", None) is None:
        return None
    existing = getattr(solver, WATCH_ATTR, None)
    if existing is not None:
        return existing
    w = int(world_count if world_count is not None else getattr(solver.model, "world_count", 1) or 1)
    sc = SelfContactWatch(solver, w)
    setattr(solver, WATCH_ATTR, sc)
    solver._collision_detection_penetration_free = types.MethodType(_detect_and_watch, solver)
    log.info("self_contact: watching %d VBD self-contact row families over %d worlds", len(sc._rows), w)
    return sc


def flag() -> int:
    return int(HealthFlags.SELF_CONTACT_OVERFLOW)


__all__ = ["CROSS_ENV", "CROSS_FIX", "CROSS_MODES", "CROSS_WHAT", "ROW_FAMILIES", "SelfContactWatch", "WATCH_ATTR",
           "cross_mode", "cross_problems", "cross_row", "cross_text", "detector", "flag", "report_fields",
           "state_arrays", "watch"]
