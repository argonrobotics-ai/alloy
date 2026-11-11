# SPDX-License-Identifier: Apache-2.0
"""PANELS emitter (SPEC §3.3 step 5): a creased sheet as thin rigid boxes joined by revolute hinges on the creases.

v1 handles creases that span the whole sheet and are parallel to one in-plane axis (cartons, folders, box lids);
other crease patterns raise AssemblyError. Hinge stiffness is the crease's N·m/rad (MuJoCo passive stiffness, else
a joint drive toward 0). The thin axis of the box is the sheet normal; the first panel carries the free joint.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.cards.types import Primitive, ResolvedObject
from alloy.core.errors import AssemblyError
from alloy.route.types import RouteDecision
from alloy.scene.emitters.common import shape_config

PARALLEL_TOL = 1e-6


def crease_offsets(prim: Primitive, creases: tuple) -> tuple[int, int, list[tuple[float, float]]]:
    """(cut axis, crease-direction axis, sorted [(offset along the cut axis, stiffness)]) for parallel creases."""
    half = np.asarray(prim.size[:3], np.float64)
    thin = int(np.argmin(half))
    plane = [k for k in range(3) if k != thin]
    out, along = [], None
    for c in creases:
        a, b = np.asarray(c.a, np.float64), np.asarray(c.b, np.float64)
        d = b - a
        ax = plane[int(np.argmax(np.abs(d[plane])))]
        cut = plane[0] if ax == plane[1] else plane[1]
        if along is None:
            along = ax
        if ax != along or abs(d[cut]) > PARALLEL_TOL * max(1.0, abs(d[ax])):
            raise AssemblyError("PANELS v1 needs parallel, axis-aligned creases")
        if abs(abs(d[ax]) - 2.0 * half[ax]) > 1e-6 * max(1.0, 2.0 * half[ax]):
            raise AssemblyError("PANELS v1 needs creases that span the whole sheet")
        out.append((float(a[cut]), float(c.stiffness.value)))
    if along is None:
        raise AssemblyError("PANELS route without creases")
    cut_axis = plane[0] if along == plane[1] else plane[1]
    return cut_axis, along, sorted(out)


def emit_panels(b: newton.ModelBuilder, obj: ResolvedObject, d: RouteDecision) -> None:
    prim = obj.card.shape
    if not isinstance(prim, Primitive) or prim.kind != "box":
        raise AssemblyError(f"{obj.card.name}: PANELS needs a thin box primitive")
    cut, along, creases = crease_offsets(prim, obj.card.features.creases)
    half = np.asarray(prim.size[:3], np.float64)
    edges = [-half[cut]] + [o for o, _ in creases] + [half[cut]]
    cfg = shape_config(obj.material)
    mujoco = d.entry == "mujoco"
    parent, joints = -1, []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        center = np.zeros(3)
        center[cut] = 0.5 * (lo + hi)
        hx = half.copy()
        hx[cut] = 0.5 * (hi - lo)
        body = b.add_link(xform=wp.transform(wp.vec3(*center), wp.quat_identity()), label=f"{obj.card.name}/{i}")
        b.add_shape_box(body, hx=float(hx[0]), hy=float(hx[1]), hz=float(hx[2]), cfg=cfg)
        if parent < 0:
            joints.append(b.add_joint_free(body, parent_xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                                           child_xform=wp.transform_identity()))
        else:
            joints.append(_hinge(b, parent, body, (cut, along), (edges[i - 1], lo, hi), creases[i - 1][1], mujoco))
        parent = body
    b.add_articulation(joints, label=obj.card.name)


def _hinge(b: newton.ModelBuilder, parent: int, child: int, axes: tuple[int, int], span: tuple[float, float, float],
           k: float, mujoco: bool) -> int:
    cut, along = axes
    prev_lo, lo, hi = span
    p_off, c_off, axis = np.zeros(3), np.zeros(3), np.zeros(3)
    p_off[cut] = lo - 0.5 * (prev_lo + lo)       # crease line in the parent panel frame
    c_off[cut] = lo - 0.5 * (lo + hi)            # and in the child panel frame
    axis[along] = 1.0
    kw = {"custom_attributes": {"mujoco:dof_passive_stiffness": [k]}} if mujoco else {"target_ke": k,
                                                                                      "target_pos": 0.0}
    return b.add_joint_revolute(parent, child, parent_xform=wp.transform(wp.vec3(*p_off), wp.quat_identity()),
                                child_xform=wp.transform(wp.vec3(*c_off), wp.quat_identity()),
                                axis=tuple(axis), **kw)
