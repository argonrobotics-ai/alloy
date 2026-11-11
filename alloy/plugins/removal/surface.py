# SPDX-License-Identifier: Apache-2.0
"""Surface patches: a flat face of a shape (static or on a body) divided into a per-world grid of cells.

This generalizes the sheet geometry of the ``ink`` plug-in so any contact-driven surface plug-in can share it. A patch
is a face of a shape at local height ``surface_offset`` (a box's top face by default) spanning ``size`` in the shape's
local x and y. Its frame (``patch_frame``) puts the nominal surface at local z = 0 with +z pointing out of the
material, and cell ``(i, j)`` covers local x in ``[-hx + i·dx, -hx + (i + 1)·dx]`` (and y alike).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.errors import PluginError


@dataclass(frozen=True)
class Patch:
    """Host description of a patch: the shape, its owning body (-1: static) and the face's frame on that body."""

    shape: int
    body: int
    local: tuple[float, ...]          # shape-to-body transform with the surface offset folded in (x, y, z, qx..qw)
    size: tuple[float, float]         # face extent along local x, y [m]


def patch_of(model: newton.Model, shape: int, label: str, size: tuple[float, float] | None = None,
             surface_offset: float | None = None) -> Patch:
    """The patch on ``shape``: the face size and height come from the config, or from the shape if it is a box.

    Example: ``patch_of(model, s, "plate")`` is the top face of the box shape ``s``."""
    is_box = int(model.shape_type.numpy()[shape]) == int(newton.GeoType.BOX)
    half = model.shape_scale.numpy()[shape]
    if (size is None or surface_offset is None) and not is_box:
        raise PluginError(f"shape {label!r} is not a box: set its size=(x, y) and surface_offset=height")
    sx, sy = size if size is not None else (2.0 * float(half[0]), 2.0 * float(half[1]))
    top = surface_offset if surface_offset is not None else float(half[2])
    if sx <= 0.0 or sy <= 0.0:
        raise PluginError(f"patch {label!r} needs a positive size, got {(sx, sy)}")
    xf = model.shape_transform.numpy()[shape].astype(np.float64)
    offset = wp.transform(wp.vec3(0.0, 0.0, float(top)), wp.quat_identity())
    local = wp.transform_multiply(wp.transform(wp.vec3(*xf[:3]), wp.quat(*xf[3:])), offset)
    return Patch(shape, int(model.shape_body.numpy()[shape]), tuple(float(v) for v in local), (float(sx), float(sy)))


def patch_tables(patches: list[Patch], device: object) -> tuple[wp.array, wp.array]:
    """Per-world device tables: the owning body ``int32 [W]`` and the face frame on it ``transform [W]``."""
    body = np.asarray([p.body for p in patches], np.int32)
    local = np.asarray([p.local for p in patches], np.float32)
    return wp.array(body, dtype=wp.int32, device=device), wp.array(local, dtype=wp.transform, device=device)


@wp.func
def patch_frame(body_q: wp.array[wp.transform], body: wp.int32, local: wp.transform) -> wp.transform:
    """World frame of a patch: its surface at local z = 0, +z out of the material."""
    if body >= 0:
        return wp.transform_multiply(body_q[body], local)
    return local


@wp.func
def moved_point(q_in: wp.array[wp.transform], q_out: wp.array[wp.transform], body: wp.int32, p: wp.vec3) -> wp.vec3:
    """Where the material point at world ``p`` of ``body`` (at ``q_in``) is at ``q_out``; static (-1) stays put."""
    if body < 0:
        return p
    return wp.transform_point(q_out[body], wp.transform_point(wp.transform_inverse(q_in[body]), p))


@wp.func
def cell_window(u: wp.float32, v: wp.float32, r: wp.float32, cell: wp.vec2, nx: int, ny: int) -> wp.vec4i:
    """Cells ``(i0, i1, j0, j1)`` (inclusive) whose centres may lie within ``r`` of grid point ``(u, v)``.

    ``(u, v)`` is measured from the grid's corner (local x + hx, y + hy); an empty window has ``i0 > i1``."""
    i0 = wp.max(int(wp.floor((u - r) / cell[0])), 0)
    i1 = wp.min(int(wp.floor((u + r) / cell[0])), nx - 1)
    j0 = wp.max(int(wp.floor((v - r) / cell[1])), 0)
    j1 = wp.min(int(wp.floor((v + r) / cell[1])), ny - 1)
    return wp.vec4i(i0, i1, j0, j1)


__all__ = ["Patch", "cell_window", "moved_point", "patch_frame", "patch_of", "patch_tables"]
