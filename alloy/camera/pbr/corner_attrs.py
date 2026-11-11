# SPDX-License-Identifier: Apache-2.0
"""Per-corner mesh attributes for the generated materials: the UV layer and one colour attribute, flattened over the
scene's unique meshes into the ``ShadeData`` arrays ``uv`` / ``uv_off`` and ``col`` / ``col_off``.

``uv`` holds 3 entries per triangle of every mesh that has UVs (Blender convention, origin bottom-left; ``uv_off[m]``
= first triangle of mesh m in it, -1 without UVs); ``col`` likewise for the colour attribute (rgb; alpha is dropped).
A mesh uses its ``UVMap`` layer, else its first layer (sorted by name). The colour attribute is the one name an
Attribute node of the scene's materials asks for that some mesh carries; two such names raise CameraError.

Example::

    uv_names, col_name = select_attributes(frame, pack.static, mat_names)
    fill_corner_attrs(sd, frame, col_name, "cuda:0")
"""
from __future__ import annotations

import numpy as np
import warp as wp

from ..errors import CameraError

BUILTIN_ATTRIBUTES = frozenset({"rest_p", "rest_n", "has_rest", "UVMap"})


def uv_layer(mesh) -> tuple[str, np.ndarray] | None:
    """(name, (T, 3, 2) corner UVs) of the layer a mesh renders with, or None without UVs."""
    if not mesh.uv:
        return None
    name = "UVMap" if "UVMap" in mesh.uv else sorted(mesh.uv)[0]
    return name, np.asarray(mesh.uv[name], np.float32).reshape(-1, 3, 2)


def attribute_names(static: dict, mat_names: list[str]) -> set[str]:
    """Names of the Attribute nodes in the materials' trees that are not UVMap / rest_*."""
    out = set()
    for nm in mat_names:
        tree = static["materials"][nm].get("tree")
        for node in (tree or {}).get("nodes", []):
            if node["type"] == "ShaderNodeAttribute":
                out.add(node["props"].get("attribute_name", ""))
    return out - BUILTIN_ATTRIBUTES - {""}


def select_attributes(frame, static: dict, mat_names: list[str]) -> tuple[list[str], list[str]]:
    """(UV layer names in use, colour attribute names in use: zero or one) for generating the materials."""
    uv_names = sorted({u[0] for u in map(uv_layer, frame.meshes) if u})
    wanted = attribute_names(static, mat_names)
    have = sorted(n for n in wanted if any(n in m.attr for m in frame.meshes))
    if len(have) > 1:
        raise CameraError(f"materials read the colour attributes {have}",
                          "use one colour attribute per scene (rename the others or drop their Attribute nodes)")
    return uv_names, have


def _flat(frame, pick) -> tuple[np.ndarray, np.ndarray]:
    """(concatenated corner values (3 * triangles with data, k), per-mesh first triangle or -1)."""
    parts, off, n = [], [], 0
    for m in frame.meshes:
        a = pick(m)
        if a is None:
            off.append(-1)
            continue
        off.append(n)
        n += len(a)
        parts.append(a.reshape(-1, a.shape[-1]))
    return (np.concatenate(parts) if parts else None), np.asarray(off, np.int32)


def _colour(name: str):
    def pick(m):
        a = m.attr.get(name)
        return None if a is None else np.asarray(a, np.float32).reshape(len(m.tri), 3, -1)[..., :3]
    return pick


def fill_corner_attrs(sd, frame, col_name: str | None, device: str) -> None:
    """Set ``sd.uv / uv_off / col / col_off`` (one dummy entry when no mesh has data)."""
    uv, uv_off = _flat(frame, lambda m: (uv_layer(m) or (None, None))[1])
    sd.uv = wp.array(uv if uv is not None else np.zeros((1, 2), np.float32), dtype=wp.vec2, device=device)
    sd.uv_off = wp.array(uv_off, dtype=wp.int32, device=device)
    col, col_off = _flat(frame, _colour(col_name) if col_name else (lambda m: None))
    sd.col = wp.array(col if col is not None else np.zeros((1, 3), np.float32), dtype=wp.vec3, device=device)
    sd.col_off = wp.array(col_off, dtype=wp.int32, device=device)
