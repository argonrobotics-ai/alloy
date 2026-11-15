# SPDX-License-Identifier: Apache-2.0
"""The limits of a convex mesh's ``hull_vertices`` (``shape.mesh.hull_vertices``): the user's opt-out from
``alloy.contact.hull_cap``'s rule, a fixed cap on the mesh's collision hull (64: Newton's own, the fast hull of earlier
releases). Plain Python, so the card checks and the scene API can import it without the hull machinery."""

from __future__ import annotations

from typing import Any

from alloy.core.errors import AssemblyError

HULL_VERTICES_MAX = 2048        # alloy.contact.hull_cap.CEILING: the most hull vertices Alloy gives a mesh


def hull_vertices_problem(value: object, collision: str) -> str | None:
    """Why ``hull_vertices`` is not valid for a mesh with ``collision`` (None: it is): an int from 4 (MuJoCo's least)
    to ``HULL_VERTICES_MAX``, on a "convex" mesh only. Example: ``hull_vertices_problem(64, "convex")`` is None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return f"hull_vertices must be an int, got {value!r}"
    if not 4 <= value <= HULL_VERTICES_MAX:
        return f"hull_vertices must be 4 to {HULL_VERTICES_MAX}, got {value}"
    if collision != "convex":
        return f"hull_vertices caps a convex hull, so it applies to collision='convex' only, not {collision!r}"
    return None


def check(src: Any, name: str) -> None:
    """Refuse a mesh source whose ``hull_vertices`` is not valid for its ``collision`` (AssemblyError naming it)."""
    if problem := hull_vertices_problem(getattr(src, "hull_vertices", None), getattr(src, "collision", "convex")):
        raise AssemblyError(f"{name}: shape.hull_vertices: {problem}")


__all__ = ["HULL_VERTICES_MAX", "check", "hull_vertices_problem"]
