# SPDX-License-Identifier: Apache-2.0
"""Mesh datablocks from numpy arrays (foreach_set throughout) and per-frame attribute updates."""

from __future__ import annotations

import math

import bpy
import numpy as np


def make_mesh(name: str, verts: np.ndarray, corners: np.ndarray, sizes: np.ndarray,
              smooth_angle_deg: float | None = 30.0, uv: np.ndarray | None = None,
              corner_uv: np.ndarray | None = None) -> bpy.types.Mesh:
    """Polygon mesh; smooth shaded with edges sharper than smooth_angle_deg kept sharp (None = all smooth).

    `uv` is per vertex, `corner_uv` per face corner (OBJ vt); either becomes the UV map "UVMap"."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", np.ascontiguousarray(verts, np.float32).ravel())
    me.loops.add(len(corners))
    me.loops.foreach_set("vertex_index", np.ascontiguousarray(corners, np.int32))
    me.polygons.add(len(sizes))
    starts = np.zeros(len(sizes), np.int32)
    if len(sizes) > 1:
        starts[1:] = np.cumsum(sizes[:-1])
    me.polygons.foreach_set("loop_start", starts)
    if corner_uv is None and uv is not None and len(uv) == len(verts):
        corner_uv = uv[corners]
    if corner_uv is not None and len(corner_uv) == len(corners):
        layer = me.uv_layers.new(name="UVMap")
        layer.data.foreach_set("uv", np.ascontiguousarray(corner_uv, np.float32).ravel())
    me.update(calc_edges=True)
    if len(sizes):
        me.shade_smooth()
        if smooth_angle_deg is not None:
            me.set_sharp_from_angle(angle=math.radians(smooth_angle_deg))
    return me


def point_mesh(name: str, count: int) -> bpy.types.Mesh:
    """Vertices only (a point cloud carrier for Geometry Nodes)."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(count)
    return me


def chain_mesh(name: str, count: int) -> bpy.types.Mesh:
    """`count` vertices joined in order by edges (a curve centreline carrier)."""
    me = bpy.data.meshes.new(name)
    me.vertices.add(count)
    if count > 1:
        me.edges.add(count - 1)
        idx = np.stack([np.arange(count - 1), np.arange(1, count)], axis=1).astype(np.int32)
        me.edges.foreach_set("vertices", idx.ravel())
    me.update()
    return me


def set_positions(me: bpy.types.Mesh, pos: np.ndarray) -> None:
    me.vertices.foreach_set("co", np.ascontiguousarray(pos, np.float32).ravel())
    me.update()


def set_attr(me: bpy.types.Mesh, name: str, values: np.ndarray, typ: str = "FLOAT_VECTOR",
             domain: str = "POINT") -> None:
    """Create (if needed) and fill a generic attribute; QUATERNION values are wxyz."""
    attr = me.attributes.get(name) or me.attributes.new(name, typ, domain)
    key = {"FLOAT": "value", "INT": "value", "BOOLEAN": "value", "QUATERNION": "value",
           "FLOAT_VECTOR": "vector", "FLOAT_COLOR": "color"}[typ]
    dtype = {"INT": np.int32, "BOOLEAN": bool}.get(typ, np.float32)
    attr.data.foreach_set(key, np.ascontiguousarray(values, dtype).ravel())


def new_object(name: str, data, collection: bpy.types.Collection) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    return obj


def set_material(obj: bpy.types.Object, materials: list) -> None:
    """Material slots on the object's data, in order (face material_index picks the slot)."""
    data = obj.data
    if data is None or not hasattr(data, "materials"):
        return
    data.materials.clear()
    for m in materials:
        data.materials.append(m)
