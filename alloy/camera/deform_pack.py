# SPDX-License-Identifier: Apache-2.0
"""Deformables as full-material pack meshes (``model_pack.pack_from_model(..., deform=layout)``): one pack mesh per
deformable and world, in world coordinates (identity object transform), drawn only in its own world.

Materials: the look's or the scene items' ``assign`` rule matching the object's name, else a matte material of the
kind's colour (light cloth, brown soil). Corner normals: smooth over the whole surface (no angle limit, as the device
recomputes them every render); grains carry their icosahedron's vertex directions, which never change.

Example::

    meshes, mats, tex, present = deform_meshes(layout, look={"assign": [{"match": "towel", "material": "linen"}]})
"""
from __future__ import annotations

import numpy as np

from .card_looks import assigned
from .deform import ICO_F, ICO_V, DeformLayout, DeformObject
from .pbr.pack_io import Mesh
from .smooth import corner_normals

DEFAULT_PRESET = "plastic_matte"
_EYE = np.eye(4).tolist()


def _soft(spec: dict | str) -> dict:
    """A look spec without the hard-surface Bevel (a deformable has no machined edges; its Bevel mesh would be stale)."""
    return {**({"preset": spec} if isinstance(spec, str) else spec), "hard": False}


def _material(o: DeformObject, look: dict, items: dict | None) -> tuple[str, dict]:
    """(material name, look spec) of a deformable: look rule, item rule, else a matte colour of its kind."""
    name = assigned(look, [o.name]) if look else None
    if name is not None:
        return f"{name}~soft", _soft(look.get("materials", {}).get(name, name))
    name = assigned(items, [o.name]) if items else None
    if name is not None:
        return f"{name}~soft", _soft(items["materials"][name])
    rgb = [float(c) for c in o.color]
    return f"{o.kind}_{'%02x%02x%02x' % tuple(int(round(255 * c)) for c in rgb)}", _soft(
        {"preset": DEFAULT_PRESET, "base_color": rgb, "roughness": 0.8})


def _normals(o: DeformObject) -> np.ndarray:
    """(T, 3, 3) corner normals of the rest mesh."""
    if o.kind == "grains":
        return np.ascontiguousarray(np.broadcast_to(ICO_V[ICO_F], (o.count, 20, 3, 3)).reshape(-1, 3, 3))
    return corner_normals(o.rest, o.tris, max_angle_deg=180.0)


def deform_meshes(layout: DeformLayout, look: dict | None = None, items: dict | None = None):
    """(meshes, {material: pack material}, {texture key: entry}, present (len(meshes), worlds) bool): world w's
    copy of object o is ``meshes[o * worlds + w]``, present in world w only."""
    from .looks import material_tree
    look = look or {}
    meshes, specs = [], {}
    for o in layout.objects:
        mat, spec = _material(o, look, items)
        specs.setdefault(mat, spec)
        n, ids = _normals(o), np.zeros(len(o.tris), np.int32)
        for w in range(layout.worlds):
            meta = {"name": f"{o.name}@{w}", "object_name": o.name, "materials": [mat], "object_to_world": _EYE,
                    "dynamic": True, "kind": o.kind}
            meshes.append(Mesh(meta, o.rest, o.tris, n, ids))
    present = np.zeros((len(meshes), layout.worlds), bool)
    present[np.arange(len(meshes)), np.tile(np.arange(layout.worlds), len(layout.objects))] = True
    mats, tex = {}, {}
    for name, spec in specs.items():
        mats[name] = material_tree(name, spec)
        tex.update(mats[name].pop("textures", {}))
    return meshes, mats, tex, present
