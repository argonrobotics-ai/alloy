# SPDX-License-Identifier: Apache-2.0
"""Explicit-stability numbers of the drill, worst over worlds, read once at setup (no device reads during the step).

The drill's front spring and damper, its two hole wall springs and the viscous end of its cutting torque act
explicitly between the tool and the workpiece, at points away from both bodies' centres of mass. Each bound uses the
point mobility (the velocity a unit force at that point gives, translation plus rotation) of both bodies, as the
removal pad's do after WP2's review of WP15 (``alloy.plugins.removal.stability``):

- the front acts along the face normal at the tip: tool ``1/m + r_perp²/I_min`` (``r_perp`` the tip's offset from the
  COM across the drill axis);
- the wall springs act across the axis at the tip (the farthest of their two points): tool ``1/m + r²/I_perp``
  (``I_perp`` the smallest inertia about an axis perpendicular to the drill);
- a floating workpiece adds ``1/m_f + reach²/I_f,min``, with ``reach`` bounding the patch from its COM (the formula
  of ``removal.stability.world_response``);
- the torque's viscous slope acts on the reduced inertia about the drill axis; the kernel raises ``spin_eps`` to
  ``2·M·dt / I``.

The numbers are the worst over worlds, so per-world randomized masses stay covered. Re-create the plug-in if masses
change later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.removal.stability import Body, body_of, in_plane_inertia, smallest_inertia
from alloy.plugins.removal.surface import Patch


@dataclass(frozen=True)
class DrillResponse:
    """Worst-case (over worlds) response the drill's explicit forces act on."""

    front_mobility: float     # tool + workpiece point mobility along the face normal at the tip [1/kg]
    wall_mobility: float      # tool + workpiece point mobility across the drill axis at the tip [1/kg]
    spin_inertia: float       # reduced inertia about the drill axis [kg·m^2]


def face_mobility(face: Body, patch: Patch) -> float:
    """Point mobility of a floating workpiece anywhere on its patch: 1/m + reach²/I_min."""
    surface = np.asarray(patch.local[:3], np.float64)
    reach = float(np.linalg.norm(surface - face.com)) + 0.5 * float(np.hypot(*patch.size))
    return 1.0 / face.mass + reach**2 / smallest_inertia(face.inertia)


def world_response(tool: Body, face: Body | None, patch: Patch, tip: np.ndarray, a: np.ndarray) -> tuple:
    """(front mobility, wall mobility, reduced axial inertia) of one world; ``face`` is None when static."""
    off = tip - tool.com
    perp = off - float(np.dot(off, a)) * a
    front = 1.0 / tool.mass + float(perp @ perp) / smallest_inertia(tool.inertia)
    wall = 1.0 / tool.mass + float(off @ off) / in_plane_inertia(tool.inertia, a)
    inv_spin = 1.0 / max(float(a @ tool.inertia @ a), 1e-12)
    if face is not None:
        extra = face_mobility(face, patch)
        front, wall = front + extra, wall + extra
        inv_spin += 1.0 / smallest_inertia(face.inertia)
    return front, wall, 1.0 / inv_spin


def drill_response(model: newton.Model, tools: np.ndarray, patches: list[Patch], c: Any) -> DrillResponse:
    """The worst world's numbers: ``c`` is the drill's config (``tip``, ``axis``)."""
    a = np.asarray(c.axis, np.float64) / np.linalg.norm(c.axis)
    tip = np.asarray(c.tip, np.float64)
    rows = []
    for t, p in zip(tools, patches, strict=True):
        face = body_of(model, p.body, "workpiece") if p.body >= 0 else None
        rows.append(world_response(body_of(model, int(t), "tool"), face, p, tip, a))
    r = np.asarray(rows, np.float64)
    return DrillResponse(float(r[:, 0].max()), float(r[:, 1].max()), float(r[:, 2].min()))


__all__ = ["DrillResponse", "drill_response", "face_mobility", "world_response"]
