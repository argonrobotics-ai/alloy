# SPDX-License-Identifier: Apache-2.0
"""Explicit-stability numbers of the removal pad, per world, read once at setup (no device reads during the step).

The pad's spring, damper and viscous friction act explicitly between the tool and the workpiece. A floating
workpiece (a shape on a body) responds too, so every bound uses both bodies:

- spring and damper along the normal act on the normal mobility at the pad, ``w_n = 1/m_tool + d⊥²/I_tool,plane +
  1/m_face + reach_face²/I_face,min``: the load can sit anywhere on a floating face, which then rotates as well as
  translates (WP2 review N1); ``d⊥`` is the pad centre's in-plane offset from the tool's COM (0 for the usual tool);
- the tilt spring and damper act on the reduced tilt inertia (tool: its smallest inertia about an axis in the pad
  plane, i.e. perpendicular to the pad normal; workpiece: its smallest principal inertia). Splitting the pad load
  into its centroid and the spread about it bounds the explicit step by ``ω² <= stiffness·w_n + 2·tilt/I_red`` (the
  2 covers a pad only partly on the face, whose load centroid leaves the pad axis), and the same for the damper;
- viscous friction at a pad point acts on that point's mobility, ``1/m + |r|^2 / I_min`` for each body, with ``r``
  the farthest pad point from that body's centre of mass. The force kernel raises ``slip_eps`` so that
  ``mu·F/eps·dt·mobility <= 1/2``, which also bounds the spin damping of a spinning pad (WP2 review K1, K2).

The numbers are the worst over worlds, so per-world randomized masses stay covered (WP2 review C3). They are read at
setup: re-create the plug-in if masses change later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.removal.surface import Patch

TINY = 1.0e-12


@dataclass(frozen=True)
class PadResponse:
    """Worst-case (over worlds) body response the pad's explicit forces act on."""

    normal_mobility: float    # w_n along the normal at the pad, tool + floating face [1/kg] (largest over worlds)
    tilt_inertia: float       # reduced inertia about an axis in the pad plane [kg·m^2]
    mobility: np.ndarray      # [W] float32: summed point mobility of tool + workpiece at the pad [1/kg]


@dataclass(frozen=True)
class Body:
    """One body's mass properties (body frame, inertia about the COM)."""

    mass: float
    inertia: np.ndarray       # (3, 3)
    com: np.ndarray           # (3,)


def body_of(model: newton.Model, b: int, what: str) -> Body:
    mass = float(model.body_mass.numpy()[b])
    if mass <= 0.0:
        raise PluginError(f"{what} body {b} has no mass: the pad force needs dynamic bodies (density > 0)")
    inertia = np.asarray(model.body_inertia.numpy()[b], np.float64).reshape(3, 3)
    return Body(mass, inertia, np.asarray(model.body_com.numpy()[b], np.float64))


def in_plane_inertia(inertia: np.ndarray, normal: np.ndarray) -> float:
    """Smallest moment of inertia about an axis perpendicular to ``normal``."""
    a = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
    a /= np.linalg.norm(a)
    basis = np.stack([a, np.cross(normal, a)])                 # (2, 3), orthonormal, both perpendicular to normal
    return max(float(np.linalg.eigvalsh(basis @ inertia @ basis.T).min()), TINY)


def smallest_inertia(inertia: np.ndarray) -> float:
    return max(float(np.linalg.eigvalsh(inertia).min()), TINY)


def point_mobility(body: Body, reach_sq: float) -> float:
    """Upper bound of a body's inverse mass at any point within ``sqrt(reach_sq)`` of its COM: ``1/m + r²/I_min``.
    Example: a floating plate's normal response anywhere on its face (``patch_reach_sq``)."""
    return 1.0 / body.mass + reach_sq / smallest_inertia(body.inertia)


def patch_reach_sq(face: Body, patch: Patch) -> float:
    """Squared bound of the distance from the face body's COM to any point of the patch."""
    surface = np.asarray(patch.local[:3], np.float64)
    return (float(np.linalg.norm(surface - face.com)) + 0.5 * float(np.hypot(*patch.size))) ** 2


def world_response(tool: Body, face: Body | None, center: np.ndarray, normal: np.ndarray, radius: float,
                   patch: Patch) -> tuple[float, float, float]:
    """(normal mobility w_n, reduced tilt inertia, friction mobility) of one world. ``center``/``normal``: the pad in
    the tool frame; ``face`` is None for a static workpiece."""
    arm = center - tool.com
    offset = arm - float(np.dot(arm, normal)) * normal                       # pad centre's in-plane offset
    plane = in_plane_inertia(tool.inertia, normal)
    normal_mobility = 1.0 / tool.mass + float(np.dot(offset, offset)) / plane
    inv_tilt = 1.0 / plane
    mobility = point_mobility(tool, (float(np.linalg.norm(arm)) + radius) ** 2)   # farthest pad point from the COM
    if face is not None:
        face_mobility = point_mobility(face, patch_reach_sq(face, patch))
        normal_mobility += face_mobility
        inv_tilt += 1.0 / smallest_inertia(face.inertia)
        mobility += face_mobility
    return normal_mobility, 1.0 / inv_tilt, mobility


def pad_response(model: newton.Model, tools: np.ndarray, patches: list[Patch], center: tuple[float, float, float],
                 normal: tuple[float, float, float], radius: float) -> PadResponse:
    """The worst world's numbers for ``check_dt`` plus every world's pad mobility for the force kernel."""
    c = np.asarray(center, np.float64)
    n = np.asarray(normal, np.float64) / np.linalg.norm(normal)
    rows = []
    for t, p in zip(tools, patches, strict=True):
        face = body_of(model, p.body, "workpiece") if p.body >= 0 else None
        rows.append(world_response(body_of(model, int(t), "tool"), face, c, n, radius, p))
    r = np.asarray(rows, np.float64)
    return PadResponse(float(r[:, 0].max()), float(r[:, 1].min()), r[:, 2].astype(np.float32))


__all__ = ["Body", "PadResponse", "body_of", "in_plane_inertia", "pad_response", "patch_reach_sq", "point_mobility",
           "smallest_inertia", "world_response"]
