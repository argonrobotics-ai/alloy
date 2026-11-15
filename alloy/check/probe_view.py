# SPDX-License-Identifier: Apache-2.0
"""The model the check's contact probe collides: the shapes as the simulation collides them.

MuJoCo's own contacts (``native="mujoco"``, the default) collide every mesh as its convex hull, while Newton's model
keeps an MJCF mesh geom (a robot link, a RoboCasa object's ``hulls/*.msh`` collider) as a triangle mesh. Collided as
triangle meshes, 1,024 worlds of RoboCasa kitchen objects overflowed the probe's triangle-pair and contact buffers
(315,906 candidates dropped), where the same slots holding convex scans never did, and the depths measured were not
the ones the simulation's contacts see. ``collider_view`` gives the probe a view of the model (Newton's
``ModelView``: the shared model is untouched) on which every triangle-mesh shape MuJoCo collides is a convex mesh of
the same vertices (Newton's support of a convex mesh is over its vertices, so a non-convex mesh collides as its hull,
as in MuJoCo). Scenes on Newton's contacts (``native="newton"``) and scenes with particles keep the model as it is:
Newton collides their meshes as triangles, and particle contacts need the triangle surface.

Example::

    >>> view = collider_view(built, "mujoco")                       # doctest: +SKIP
    >>> hulled(view)                                                # doctest: +SKIP
    512
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("alloy.check")

VIEW_NAME = "check probe"


def hull_shapes(built: Any, native: str | None) -> np.ndarray:
    """Shape ids MuJoCo collides as convex hulls that Newton's model holds as triangle meshes: colliding mesh shapes
    on the MuJoCo entry's bodies or static, when the simulation runs MuJoCo's own contacts and has no particles."""
    from alloy._vendor import newton  # noqa: PLC0415

    model = built.model
    mujoco = [e for e in getattr(built, "entries", ()) if getattr(e, "capability", None) == "mujoco"]
    if native != "mujoco" or not mujoco or int(model.particle_count or 0) or not int(model.shape_count or 0):
        return np.zeros(0, np.int64)
    kind = model.shape_type.numpy()
    collides = (model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    body = model.shape_body.numpy()
    bodies = np.unique(np.concatenate([np.asarray(e.bodies, np.int64) for e in mujoco]))
    on_mujoco = (body < 0) | np.isin(body, bodies)
    return np.flatnonzero((kind == int(newton.GeoType.MESH)) & collides & on_mujoco)


def collider_view(built: Any, native: str | None) -> Any:
    """``built.model``, or a view of it on which the shapes of ``hull_shapes`` are convex meshes."""
    import warp as wp  # noqa: PLC0415

    from alloy._vendor import newton  # noqa: PLC0415
    from alloy.core.newton_compat import ModelView  # noqa: PLC0415

    model = built.model
    shapes = hull_shapes(built, native)
    if shapes.size == 0:
        return model
    kind = model.shape_type.numpy().copy()
    kind[shapes] = int(newton.GeoType.CONVEX_MESH)
    view = ModelView(model, VIEW_NAME)
    view.shape_type = wp.array(kind, dtype=model.shape_type.dtype, device=model.device)
    log.info("check probe: %d triangle-mesh shapes collide as convex hulls, as MuJoCo's contacts collide them",
             shapes.size)
    return view


def hulled(model: Any) -> int:
    """How many shapes ``model`` (a probe's view) collides as convex hulls in place of triangle meshes."""
    if getattr(model, "_name", None) != VIEW_NAME:
        return 0
    from alloy._vendor import newton  # noqa: PLC0415

    parent = object.__getattribute__(model, "_parent")
    return int(np.count_nonzero((model.shape_type.numpy() == int(newton.GeoType.CONVEX_MESH))
                                & (parent.shape_type.numpy() == int(newton.GeoType.MESH))))


__all__ = ["collider_view", "hull_shapes", "hulled"]
