# SPDX-License-Identifier: Apache-2.0
"""Newton model -> :class:`~alloy.camera.scene.SceneData`: one instance per visible shape, in its own world.

Shapes are tessellated like Newton's viewers (``newton.Mesh.create_*``) and coloured like them: the model's
``shape_color`` when present, else the mesh's own colour, else a per-shape palette colour. A shape's ``shape_color``
that is only a builder palette colour (what a shape added with no colour gets: scenes assembled object by object give
every shape the palette's first colour) counts as unset, as does one colour for every visible shape: such shapes
take palette colours by their index in their world (the same object has the same colour in every world), also beside
shapes that carry their own colours (a robot's rgba), and a ground plane with no colour of its own draws neutral grey.
The full-material tracer's shapes without a look take the same colours (``default_rgb``; before, they
drew the raw palette blue, ground included). Instances on a body follow ``state.body_q``; shapes with
``shape_world == -1`` are shared by every world.

Example::

    from alloy._vendor import newton
    from alloy.camera.from_newton import scene_from_model
    b = newton.ModelBuilder()
    b.add_ground_plane()
    body = b.add_body()
    b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    sd = scene_from_model(b.finalize(device="cpu"))
    assert sd.num_instances == 2
"""
from __future__ import annotations

import logging

import numpy as np

from .card_looks import look_colors
from .errors import CameraError
from .scene import Lighting, MeshGeom, SceneData

log = logging.getLogger(__name__)
PLANE_EXTENT = 20.0  # metres: size of an infinite ground plane (ends well outside any useful camera view)


def _primitive_mesh(newton, geo_type: int, scale: np.ndarray):
    """The viewer's tessellation of one primitive (Z-up), or None for unsupported types."""
    gt, M = newton.GeoType, newton.Mesh
    s = [float(v) for v in scale]
    if geo_type == gt.PLANE:
        w = s[0] if s[0] > 0 else PLANE_EXTENT
        h = s[1] if s[1] > 0 else PLANE_EXTENT
        return M.create_plane(w, h, compute_inertia=False)
    if geo_type == gt.SPHERE:
        return M.create_sphere(s[0], compute_inertia=False)
    if geo_type == gt.CAPSULE:
        return M.create_capsule(s[0], s[1], up_axis=newton.Axis.Z, compute_inertia=False)
    if geo_type == gt.CYLINDER:
        return M.create_cylinder(s[0], s[1], up_axis=newton.Axis.Z, compute_inertia=False)
    if geo_type == gt.CONE:
        return M.create_cone(s[0], s[1], up_axis=newton.Axis.Z, compute_inertia=False)
    if geo_type == gt.BOX:
        return M.create_box(s[0], s[1], s[2], compute_inertia=False)
    if geo_type == gt.ELLIPSOID:
        return M.create_ellipsoid(s[0], s[1], s[2], compute_inertia=False)
    return None


def _shape_mesh(newton, geo_type: int, scale: np.ndarray, src) -> tuple[np.ndarray, np.ndarray] | None:
    """(points, tris) of one shape in its shape frame, scale applied."""
    gt = newton.GeoType
    if geo_type in (gt.MESH, gt.CONVEX_MESH):
        if src is None:
            return None
        pts = np.asarray(src.vertices, np.float64) * np.asarray(scale, np.float64)
        tris = np.asarray(src.indices, np.int32).reshape(-1, 3)
        if np.prod(scale) < 0:  # mirrored: keep triangles counter-clockwise from outside
            tris = tris[:, [0, 2, 1]]
        return pts.astype(np.float32), tris
    mesh = _primitive_mesh(newton, int(geo_type), scale)
    if mesh is None:
        return None
    return np.asarray(mesh.vertices, np.float32), np.asarray(mesh.indices, np.int32).reshape(-1, 3)


def _palette(newton, i: int) -> tuple[float, float, float]:
    """Newton viewer's per-shape colour map (shape index -> RGB)."""
    pal = getattr(newton.ModelBuilder, "_SHAPE_COLOR_PALETTE", None) or [(128, 128, 128)]
    return tuple(c / 255.0 for c in pal[i % len(pal)][:3])


def _shape_color(newton, s: int, colors, flags: int, src, index: int | None = None) -> tuple[float, float, float]:
    """Shape ``s``'s own colour (not a builder palette colour), else its mesh's, else the palette's by ``index``."""
    if colors is not None and not _palette_only(newton, colors[s]):
        return tuple(float(c) for c in colors[s][:3])
    if src is not None and getattr(src, "color", None) is not None:
        return tuple(float(c) for c in src.color[:3])
    if (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) == 0:
        return (0.5, 0.5, 0.5)
    return _palette(newton, s if index is None else index)


def default_rgb(newton, s: int, colors, geo_type: int, flags: int, src, index: int | None = None) -> tuple:
    """The colour shape ``s`` is drawn in when no look covers it: neutral grey for a ground plane with no colour of
    its own, else :func:`_shape_color`. ``colors``: :func:`_model_colors`; ``index``: its palette index (default s)."""
    if _grey_ground(newton, geo_type, colors, s):
        return GROUND_RGB
    return _shape_color(newton, s, colors, flags, src, index)


def palette_index(model) -> np.ndarray:
    """(shape_count,) int: each shape's index as the matching shape of the first world (shared shapes: their own), so
    the palette gives an object one colour in every world of a replicated scene."""
    world = model.shape_world.numpy()
    idx = np.arange(len(world))
    own = world >= 0
    if not own.any():
        return idx
    first = {int(w): int(np.flatnonzero(world == w)[0]) for w in np.unique(world[own])}
    base = first[min(first)]
    return np.where(own, idx - np.array([first.get(int(w), base) for w in world]) + base, idx)


GROUND_RGB = (0.5, 0.5, 0.5)  # a ground plane without a colour of its own


def _palette_only(newton, rgb) -> bool:
    """True when ``rgb`` is one of the builder's palette colours: what a shape added with no colour gets."""
    pal = np.asarray(getattr(newton.ModelBuilder, "_SHAPE_COLOR_PALETTE", None) or [(128, 128, 128)], np.float64)
    return bool(np.any(np.all(np.abs(pal[:, :3] / 255.0 - np.asarray(rgb, np.float64)[:3]) < 1e-5, axis=1)))


def _grey_ground(newton, geo_type: int, colors, s: int) -> bool:
    """A ground plane with no colour of its own: the model's colours are unset, or its own is a palette colour."""
    return geo_type == int(newton.GeoType.PLANE) and (colors is None or _palette_only(newton, colors[s]))


def drawn_shapes(model) -> np.ndarray:
    """(shape_count,) bool: the shapes the camera draws, the VISIBLE ones plus the colliders of every moving body that
    has no VISIBLE shape at all.

    Newton's MJCF import gives no collider the VISIBLE flag once the file has a single visual-only geom
    (contype = conaffinity = 0), so a hand-written robot with one such marker lost every other part: the gantry
    gripper of the Gym tasks drew only its 5 mm carriage spheres, not its palm and fingers (TA-7 finding 3). A body
    with a display mesh keeps its collision hulls hidden, and so do world shapes and bodies welded in place (a task
    file's floor or table without a display mesh): those keep the importer's choice."""
    from alloy._vendor import newton
    flags, body = model.shape_flags.numpy(), model.shape_body.numpy()
    vis = (flags & int(newton.ShapeFlags.VISIBLE)) != 0
    collider = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    joint_type, child = model.joint_type.numpy(), model.joint_child.numpy()
    moving = np.unique(child[joint_type != int(newton.JointType.FIXED)])
    lone = collider & ~vis & np.isin(body, moving) & ~np.isin(body, np.unique(body[vis & (body >= 0)]))
    return vis | lone


def _model_colors(model, drawn: np.ndarray) -> np.ndarray | None:
    """The model's ``shape_color``, or None when it is unset (absent, or one colour for every drawn shape)."""
    colors = model.shape_color.numpy() if getattr(model, "shape_color", None) is not None else None
    if colors is None:
        return None
    shown = colors[drawn]
    if len(shown) > 1 and np.all(shown == shown[0]):
        return None
    return colors


def _geom_key(geo_type: int, scale: np.ndarray, src) -> tuple:
    return int(geo_type), tuple(np.round(np.asarray(scale, np.float64), 9)), id(src) if src is not None else 0


def scene_from_model(model, lighting: Lighting | None = None, include=None) -> SceneData:
    """Every visible shape of a Newton model as a camera scene (meshes shared between equal shapes).

    ``include``: optional predicate ``f(shape_index, label) -> bool`` to keep only some shapes.
    Example: ``scene_from_model(model, include=lambda s, label: "collision" not in label)``.
    """
    from alloy._vendor import newton
    n = int(model.shape_count)
    if n == 0:
        raise CameraError("the model has no shapes", "add shapes (add_shape_box, add_mjcf, ...) before the camera")
    ty, sc = model.shape_type.numpy(), model.shape_scale.numpy()
    body, world = model.shape_body.numpy(), model.shape_world.numpy()
    xf, flags = model.shape_transform.numpy(), model.shape_flags.numpy()
    drawn = drawn_shapes(model)
    colors, pal = _model_colors(model, drawn), palette_index(model)
    labels = list(getattr(model, "shape_label", None) or getattr(model, "shape_key", None) or [""] * n)
    card_rgb = look_colors(labels)  # library objects: their card's look colour instead of the palette
    meshes, keys, rows = [], {}, []
    for s in range(n):
        if not drawn[s] or (include and not include(s, labels[s])):
            continue
        src = model.shape_source[s]
        key = _geom_key(ty[s], sc[s], src)
        if key not in keys:
            geom = _shape_mesh(newton, int(ty[s]), sc[s], src)
            if geom is None:
                log.warning("camera: shape %d (%s) of type %d is not drawn", s, labels[s], int(ty[s]))
                continue
            keys[key] = len(meshes)
            meshes.append(MeshGeom(f"{labels[s] or 'shape'}_{s}", *geom))
        rgb = card_rgb.get(s) or default_rgb(newton, s, colors, int(ty[s]), int(flags[s]), src, int(pal[s]))
        rows.append((keys[key], int(world[s]), int(body[s]), xf[s], rgb, labels[s] or f"shape_{s}"))
    if not rows:
        raise CameraError("no visible shapes", "set ShapeFlags.VISIBLE on the shapes to draw, or loosen include")
    num_worlds = max(1, int(getattr(model, "world_count", 0) or getattr(model, "num_worlds", 0) or 1))
    m, g, b, x, c, nm = zip(*rows)
    return SceneData(meshes, np.array(m), np.array(g), np.array(b), np.stack(x), np.array(c), num_worlds, list(nm),
                     lighting or Lighting.studio())
