# SPDX-License-Identifier: Apache-2.0
"""VBD particle contact for one entry (SCR-OSS22-4, SCR-SOFT-3-1): when it is on, and at what distance.

Newton's VBD self-contact is one switch per solver: every triangle and edge of every cloth and soft solid in the entry
then collides with every other, within the object and across objects. It is on when

- a cloth card asks for it (``features.self_contact``: a folded towel rests as two layers, SCR-OSS22-4), or
- the entry holds two or more cloth or soft-solid objects (SCR-SOFT-3-1: a skin pad placed on a fat pad rests on it
  instead of falling through).

One object without the feature leaves the entry unchanged, bit for bit.

Distance rule: each object's contact thickness is twice its particle contact radius, the distance at which it rests
on a rigid surface: a sheet's thickness h (radius h/2, ``emit_cloth``), a soft solid's 2r (``soft_particle_radius``,
0.2 of its shortest tet edge). Two objects touch when their surfaces are half of each one's thickness apart, so the
interaction distance (Newton's ``particle_self_contact_margin``) is the thinnest object's thickness. Thicker objects
then rest a little closer than their own thickness (they still never cross); one warning names a spread over 2x. A
finely meshed soft solid can set a small margin for the whole entry: a cloth beside it then folds onto itself at that
distance.

Detection reach (``particle_self_contact_gap``, detection only) comes from a motion bound, not the thickness
(coordinator ruling 2026-11-08 08:40Z (b)): the entry passes ``alloy_self_contact_speed`` and the solver sets the gap
from each step's dt so nothing moving slower than the bound is truncated (`alloy.solvers.vbd.reach`). The gap was the
thickness before: a 0.6 mm towel then fell at a constant 0.82 m/s at the 1.25 ms step.

Pre-split objects (tearable cloth, cuttable solids) start with seam twins coincident and joined only by bond springs:
contact would push every seam open (WP11 review). With one in the entry, pairs closer than 0.5 of the margin in the
rest shape (``model.particle_q``) never collide (``particle_rest_shape_contact_exclusion_radius``); objects placed
closer than that at the start then never collide either. Without one, nothing is excluded, so objects placed exactly
touching still collide.

Small bodies (audit GAPS-11): cloth and film sheets touch rigid shapes at their vertices only (each vertex a sphere
of half the sheet's thickness); nothing collides with the inside of a triangle. A rigid body that fits between the
vertices passes through the sheet with no health flag (known gap cloth-small-body-passes-through). The largest ball
that fits between the four vertices of a square cell has a radius of cell / sqrt(2) less half the thickness (WP2's
review), which is the measured edge (R / cell 0.65 passed, 0.76 pressed). The build warns once per sheet, naming the
rigid primitives whose smallest dimension is under ``SMALL_BODY_CELLS`` of its median rest edge
(`small_body_warnings`: by size alone, wherever the bodies are; mesh-shaped bodies and robot links are not checked).

Mesh check: a cloth whose shortest rest edge is under the contact distance starts with neighbours outside the
one-ring filter inside each other's contact, and self contact pushes the sheet apart. The build warns, naming the
sheet (WP2, 2026-11-08: a jittered-hex disc with 1.1 mm edges at h 2 mm stretched 89 % in 0.5 s, 3.6 % without self
contact, 3.9 % on a grid disc).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

import numpy as np

from alloy.route.types import Route
from alloy.scene.template import TemplateObject
from alloy.solvers.vbd.reach import SELF_CONTACT_SPEED, SPEED_KEY

log = logging.getLogger("alloy.scene")

CONTACT_ROUTES = (Route.CLOTH_VBD, Route.SOFT_VBD)
SHEET_ROUTES = (Route.CLOTH_VBD, Route.CLOTH_FILM)
RIGID_ROUTES = (Route.STATIC, Route.RIGID, Route.RIGID_COMPLIANT, Route.RIGID_PILE)
SMALL_BODY_CELLS = 2.0   # warn under this many cells across: mid-cell balls of R / cell <= 0.65 passed through,
#                          0.76-0.87 pressed (docs/known_gaps/cloth-small-body-passes-through.md)
SMALL_BODY_NAMES = 4     # bodies named per sheet in the warning; the rest are counted
REST_EXCLUSION = 0.5   # x margin: with a pre-split object, pairs closer than this at rest never collide (seam twins)
SPREAD_WARN = 2.0      # warn when the thickest object is more than this many times the thinnest
EDGE_TOL = 1.0e-6      # relative: a rest edge exactly at the margin (a folded towel's fold strip) does not warn


def vbd_self_contact(objs: Sequence[TemplateObject], radii: Mapping[str, float] | None = None) -> dict[str, object]:
    """Newton VBD kwargs for the entry's objects (module docstring); {} leaves the entry unchanged. ``radii``: each
    object's particle contact radius as emitted (the assembler reads it from the template builder); without it, the
    radius comes from the card (a soft solid's tet mesh is then rebuilt)."""
    soft = [t for t in objs if t.obj is not None and t.decision.route in CONTACT_ROUTES]
    asked = any(t.decision.route == Route.CLOTH_VBD and t.obj.card.features.self_contact for t in soft)
    if not asked and len(soft) < 2:
        return {}
    thick = {t.name: contact_thickness(t, radii) for t in soft}
    h = min(thick.values())
    _warn_spread(thick, h)
    _warn_short_edges(soft, h)
    kw: dict[str, object] = {"particle_enable_self_contact": True, "particle_self_contact_margin": h,
                             SPEED_KEY: SELF_CONTACT_SPEED}
    if any(_presplit(t) for t in soft):
        kw["particle_rest_shape_contact_exclusion_radius"] = REST_EXCLUSION * h
    return kw


def contact_thickness(t: TemplateObject, radii: Mapping[str, float] | None = None) -> float:
    """Twice the object's particle contact radius: a sheet's thickness, a soft solid's 2r."""
    from alloy.scene.emitters.deformable import sheet_thickness, soft_mesh, soft_particle_radius

    if t.decision.route == Route.CLOTH_VBD:
        return sheet_thickness(t.obj)
    if radii is not None and t.name in radii:
        return 2.0 * float(radii[t.name])
    verts, tets, _ = soft_mesh(t.obj)
    return 2.0 * soft_particle_radius(verts, tets)


def emitted_radii(objs: Sequence[TemplateObject], particle_radius: Sequence[float]) -> dict[str, float]:
    """Each soft solid's particle contact radius from the template builder's ``particle_radius``: the largest over its
    particles (topology spares are parked at 1e-4 m, below any contact radius)."""
    out = {}
    for t in objs:
        rng = t.ranges.particle
        if t.obj is not None and t.decision.route == Route.SOFT_VBD and len(rng):
            out[t.name] = float(max(particle_radius[rng.start:rng.stop]))
    return out


def _presplit(t: TemplateObject) -> bool:
    from alloy.scene.emitters.topology import wants_presplit

    return wants_presplit(t.obj)


def _warn_spread(thick: Mapping[str, float], h: float) -> None:
    thicker = sorted(name for name, v in thick.items() if v > SPREAD_WARN * h)
    if thicker:
        log.warning("VBD contact at the thinnest object's thickness %.3g m (%s): %s rest(s) closer than their own "
                    "thickness to what they touch (no crossing)", h, min(thick, key=thick.get), ", ".join(thicker))


def shortest_edge(vertices: np.ndarray, indices: np.ndarray) -> float:
    """The shortest edge of a triangle mesh (inf without triangles)."""
    f = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if not len(f):
        return math.inf
    v = np.asarray(vertices, dtype=np.float64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    return float(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1).min())


def _warn_short_edges(soft: Sequence[TemplateObject], h: float) -> None:
    """Warn for each cloth whose shortest rest edge is under the contact distance ``h`` (module docstring)."""
    from alloy.api._say import warn  # noqa: PLC0415 - the API's user-warning helper
    from alloy.scene.emitters.deformable import cloth_mesh  # noqa: PLC0415 - as contact_thickness

    for t in soft:
        if t.decision.route != Route.CLOTH_VBD:
            continue
        mesh, _ = cloth_mesh(t.obj)
        edge = shortest_edge(mesh.vertices, mesh.indices)
        if edge < h * (1.0 - EDGE_TOL):
            warn("%s: a mesh edge is %.3g mm, under the self-contact distance %.3g mm: the sheet starts in contact "
                 "with itself and pushes apart. Use edges of at least %.3g mm (a coarser resolution or mesh), a "
                 "thinner sheet, or self_contact=False", t.name, edge * 1e3, h * 1e3, h * 1e3)


def median_edge(vertices: np.ndarray, indices: np.ndarray) -> float:
    """The median edge length of a triangle mesh (inf without triangles): a grid sheet's cell size."""
    f = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    if not len(f):
        return math.inf
    v = np.asarray(vertices, dtype=np.float64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    return float(np.median(np.linalg.norm(v[e[:, 0]] - v[e[:, 1]], axis=1)))


def smallest_extent(shape: object) -> float:
    """The smallest solid dimension [m] of a primitive shape card (inf for a plane, a mesh or anything else)."""
    from alloy.cards.geometry import primitive_min_member  # noqa: PLC0415 - the thin-member check's measure
    from alloy.cards.types import Primitive  # noqa: PLC0415 - as contact_thickness

    return float(primitive_min_member(shape)) if isinstance(shape, Primitive) and shape.size else math.inf


def small_body_warnings(objs: Sequence[TemplateObject]) -> list[str]:
    """One sentence per sheet naming the rigid primitives whose smallest dimension is under ``SMALL_BODY_CELLS`` of
    its cell, the smallest first, up to ``SMALL_BODY_NAMES`` of them (module docstring, Small bodies). By size
    alone: wherever the bodies are."""
    from alloy.core.errors import AssemblyError  # noqa: PLC0415
    from alloy.scene.emitters.deformable import cloth_mesh  # noqa: PLC0415 - as contact_thickness

    bodies = sorted((smallest_extent(t.obj.card.shape), t.name) for t in objs
                    if t.obj is not None and t.decision.route in RIGID_ROUTES)
    out: list[str] = []
    for t in objs:
        if not bodies or t.obj is None or t.decision.route not in SHEET_ROUTES:
            continue
        try:
            mesh, _ = cloth_mesh(t.obj)
        except AssemblyError:
            continue
        cell = median_edge(mesh.vertices, mesh.indices)
        small = [(e, n) for e, n in bodies if e < SMALL_BODY_CELLS * cell]
        if not small:
            continue
        names = ", ".join(f"{n} ({e * 1e3:.3g} mm across)" for e, n in small[:SMALL_BODY_NAMES])
        more = f" and {len(small) - SMALL_BODY_NAMES} more" if len(small) > SMALL_BODY_NAMES else ""
        out.append(f"{t.name} and {names}{more} can pass through each other: a sheet touches rigid shapes at its "
                   f"vertices only, and these are under {SMALL_BODY_CELLS:g} of its cells ({cell * 1e3:.3g} mm each). "
                   f"Use a finer resolution for {t.name} (cells of {small[0][0] / SMALL_BODY_CELLS * 1e3:.3g} mm or "
                   "less) or larger bodies")
    return out


def warn_small_bodies(objs: Sequence[TemplateObject]) -> None:
    """Say each `small_body_warnings` sentence to the user (the API's warning helper)."""
    from alloy.api._say import warn  # noqa: PLC0415 - the API's user-warning helper

    for line in small_body_warnings(objs):
        warn("%s", line)


__all__ = ["CONTACT_ROUTES", "REST_EXCLUSION", "RIGID_ROUTES", "SHEET_ROUTES", "SMALL_BODY_CELLS",
           "SMALL_BODY_NAMES", "contact_thickness", "emitted_radii", "median_edge", "shortest_edge",
           "small_body_warnings", "smallest_extent", "vbd_self_contact", "warn_small_bodies"]
