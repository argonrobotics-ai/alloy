# SPDX-License-Identifier: Apache-2.0
"""Surface films are one-way with what they do not declare (``docs/known_gaps/surface-film-one-way.md``).

The film couples only to the bodies its config names: the bodies its charts ride, its tools (``Applicator``, whose
underside ploughs the film and whose pad presses back), its lumped films' bodies and, with ``sph=True``, the SPH
liquid its drops join. Any other colliding body (a dropped block, a robot's links) and any other particles (cloth,
paper, paste, soft bodies) pass through the film as if it were not there, and feel nothing from it; the film keeps its
depth under them. :func:`warn_one_way` says so once at setup, naming them.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("alloy.plugins.surface_flow")

GAP_DOC = "docs/known_gaps/surface-film-one-way.md"
SHOWN = 4                               # body labels named in the warning


def _ids(arr: Any) -> set[int]:
    """Body indices in an int array (warp or numpy, any shape), without -1 (the world)."""
    if arr is None:
        return set()
    a = np.asarray(arr.numpy() if hasattr(arr, "numpy") else arr).ravel()
    return {int(b) for b in a if b >= 0}


def declared_bodies(flow: Any) -> set[int]:
    """The bodies the film couples to: chart bodies, tools and lumped films' bodies."""
    out = _ids(flow.tables.cbody)
    if getattr(flow, "tools", None) is not None:
        out |= _ids(flow.tools.body)
    if getattr(flow, "lumped", None) is not None:
        out |= _ids(flow.lumped.bodies)
    return out


def undeclared_bodies(model: Any, declared: set[int]) -> list[str]:
    """Labels of the bodies with a colliding shape that are not in ``declared``."""
    from alloy._vendor import newton  # noqa: PLC0415

    if model.shape_count == 0:
        return []
    bodies = model.shape_body.numpy()
    flags = model.shape_flags.numpy()
    collide = (flags & (int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES))) != 0
    ids = sorted({int(b) for b, c in zip(bodies, collide, strict=True) if c and b >= 0} - declared)
    labels = list(model.body_label) if model.body_count else []
    return [str(labels[b]).rsplit("/", 1)[-1] if b < len(labels) else f"body {b}" for b in ids]


def one_way_note(flow: Any, model: Any) -> str | None:
    """The warning for ``flow`` in ``model``, or None when nothing passes through its film."""
    bodies = undeclared_bodies(model, declared_bodies(flow))
    particles = int(model.particle_count)
    sph = bool(flow.config.sph)
    parts = []
    if bodies:
        names = ", ".join(bodies[:SHOWN]) + (f" and {len(bodies) - SHOWN} more" if len(bodies) > SHOWN else "")
        parts.append(f"{len(bodies)} bodies not declared as its tools ({names})")
    if particles:
        parts.append(f"{particles} particles" + (" besides the SPH liquid its drops join" if sph else
                                                 " (cloth, paper, paste, liquid or soft bodies)"))
    if not parts:
        return None
    return (f"surface_flow {flow.label}: the film is one-way with {' and '.join(parts)}: they pass through it and feel "
            "nothing from it, and it keeps its depth under them. To couple a rigid body, declare it as a tool "
            f"(Applicator(body=...)); see {GAP_DOC}")


def warn_one_way(flow: Any, ctx: Any) -> str | None:
    """Log :func:`one_way_note` once (at setup); returns it."""
    note = one_way_note(flow, ctx.model)
    if note is not None:
        log.warning(note)
    return note


__all__ = ["GAP_DOC", "declared_bodies", "one_way_note", "undeclared_bodies", "warn_one_way"]
