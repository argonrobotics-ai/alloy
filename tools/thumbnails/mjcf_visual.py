# SPDX-License-Identifier: Apache-2.0
"""MJCF visual geoms (MuJoCo's compiler: meshes, texcoords, materials, 2-D textures) as textured ``Part``s.

Visual geoms are the ones that do not collide (``contype == conaffinity == 0``); when a model has none, every mesh
geom is used. Poses are MuJoCo's at ``qpos0`` (``mj_forward``), so a multi-body fixture renders assembled.

Example::

    parts = load_mjcf(Path("mjcf/basket/robocasa_basket_001/model.xml"))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from visuals import DEFAULT_COLOR, Part, VisualError, _srgb

MESH = 7   # mjGEOM_MESH
TEXROLE_RGB = 1


def _visual_geoms(m) -> list[int]:
    meshes = [g for g in range(m.ngeom) if m.geom_type[g] == MESH and m.geom_dataid[g] >= 0]
    vis = [g for g in meshes if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0]
    return vis or meshes


def _texture(m, mat: int) -> np.ndarray | None:
    if mat < 0:
        return None
    tid = int(m.mat_texid[mat][TEXROLE_RGB]) if np.ndim(m.mat_texid) == 2 else int(m.mat_texid[mat])
    if tid < 0 or m.tex_type[tid] != 0:                     # 0 = mjTEXTURE_2D
        return None
    h, w, c = int(m.tex_height[tid]), int(m.tex_width[tid]), int(m.tex_nchannel[tid])
    a = int(m.tex_adr[tid])
    return np.asarray(m.tex_data[a:a + h * w * c], np.uint8).reshape(h, w, c)


def _uv(m, mesh: int, nface: int) -> np.ndarray | None:
    if m.mesh_texcoordnum[mesh] <= 0:
        return None
    tc = m.mesh_texcoord[m.mesh_texcoordadr[mesh]: m.mesh_texcoordadr[mesh] + m.mesh_texcoordnum[mesh]]
    ftc = m.mesh_facetexcoord[m.mesh_faceadr[mesh]: m.mesh_faceadr[mesh] + nface]
    uv = np.asarray(tc, np.float32)[np.asarray(ftc, np.int64)].reshape(-1, 2)
    uv[:, 1] = 1.0 - uv[:, 1]          # MuJoCo stores image rows top-down; Part UVs are OpenGL (origin bottom-left)
    return uv


def _part(m, d, g: int) -> Part:
    mesh = int(m.geom_dataid[g])
    va, vn = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
    fa, fn = int(m.mesh_faceadr[mesh]), int(m.mesh_facenum[mesh])
    v = np.asarray(m.mesh_vert[va:va + vn], np.float64)
    f = np.asarray(m.mesh_face[fa:fa + fn], np.int64)
    R, p = np.asarray(d.geom_xmat[g]).reshape(3, 3), np.asarray(d.geom_xpos[g])
    pts = ((v @ R.T) + p).astype(np.float32)[f].reshape(-1, 3)
    mat = int(m.geom_matid[g])
    rgba = np.asarray(m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[g], float)
    tex, uv = _texture(m, mat), _uv(m, mesh, fn)
    color, note = tuple(float(c) for c in rgba[:3]), ""
    if tex is not None and uv is None:          # a texture the mesh has no UVs for: its mean colour stands in
        color = tuple(float(c) for c in rgba[:3] * tex[..., :3].reshape(-1, 3).mean(0) / 255.0)
        tex, note = None, "texture without UVs: its mean colour"
    elif tex is None:
        uv = None
    shin = float(m.mat_shininess[mat]) if mat >= 0 else 0.5
    metal = float(m.mat_reflectance[mat]) if mat >= 0 else 0.0
    return Part(f"geom{g}", pts, np.arange(len(pts), dtype=np.int32).reshape(-1, 3), uv, tex,
                color if mat >= 0 or rgba[:3].max() > 0 else DEFAULT_COLOR, roughness=float(np.clip(1.0 - shin, 0.05, 1)),
                metallic=float(np.clip(metal, 0, 1)), glass=bool(rgba[3] < 0.5), note=note)


def load_mjcf(path: Path) -> list[Part]:
    """Visual parts of an MJCF model at its reference pose (MuJoCo's colours are already sRGB)."""
    import mujoco  # noqa: PLC0415

    try:
        m = mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:  # noqa: BLE001 - MuJoCo raises ValueError with the compiler message
        raise VisualError(f"{path}: MuJoCo cannot compile it: {exc}") from exc
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    parts = [_part(m, d, g) for g in _visual_geoms(m)]
    if not parts:
        raise VisualError(f"{path}: no mesh geoms")
    return parts


__all__ = ["load_mjcf", "_srgb"]
