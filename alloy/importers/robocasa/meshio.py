# SPDX-License-Identifier: Apache-2.0
"""OBJ (with UVs) in, MuJoCo ``.msh`` out: mesh files that MuJoCo and Newton both read without extra packages.

Newton's MJCF importer reads OBJ files through ``trimesh`` (an optional package) but reads MuJoCo's binary ``.msh``
natively, so imported models use ``.msh`` for every mesh. A ``.msh`` file holds four int32 counts (vertices,
normals, texture coordinates, faces), then float32 positions, normals, UVs and int32 triangles; UVs are per vertex,
so OBJ corners with different UVs become separate vertices, and the v coordinate is stored top-origin (1 - v) as
MuJoCo does when it loads an OBJ.

Example::

    >>> import numpy as np, tempfile, pathlib
    >>> from alloy.importers.robocasa.meshio import read_msh, write_msh
    >>> v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    >>> f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    >>> p = pathlib.Path(tempfile.mkdtemp()) / "tet.msh"
    >>> write_msh(p, v, f)
    >>> read_msh(p)[0].shape
    (4, 3)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("alloy.importers.robocasa")


def _index(token: str, count: int) -> int:
    i = int(token)
    return i - 1 if i > 0 else count + i


def _corners(tokens: list[str], nv: int, nt: int, nn: int) -> list[tuple[int, int, int]]:
    out = []
    for tok in tokens:
        parts = tok.split("/")
        vi = _index(parts[0], nv)
        ti = _index(parts[1], nt) if len(parts) > 1 and parts[1] else -1
        ni = _index(parts[2], nn) if len(parts) > 2 and parts[2] else -1
        out.append((vi, ti, ni))
    return out


def _mapped_triangles(tris: list, nuv: int) -> list:
    """The triangles to keep, with MuJoCo's reading of a partly mapped OBJ (which RoboCasa's look comes from): if at
    least half the triangles carry UVs, the ones without are dropped (in the RoboCasa meshes they are coplanar
    overlays that z-fight); otherwise every triangle is kept and the UVs are discarded (empty list = no UVs)."""
    if not nuv:
        return []
    mapped = [t for t in tris if all(c[1] >= 0 for c in t)]
    if len(mapped) == len(tris):
        return mapped
    if 2 * len(mapped) >= len(tris):
        log.info("OBJ: %d of %d triangles have no UVs and are dropped", len(tris) - len(mapped), len(tris))
        return mapped
    return []


def _parse_obj(path: Path) -> tuple[list, list, list, list]:
    verts, uvs, normals, tris = [], [], [], []
    for line in Path(path).read_text(errors="replace").splitlines():
        head, _, rest = line.strip().partition(" ")
        if head == "v":
            verts.append([float(x) for x in rest.split()[:3]])
        elif head == "vt":
            uvs.append([float(x) for x in rest.split()[:2]])
        elif head == "vn":
            normals.append([float(x) for x in rest.split()[:3]])
        elif head == "f":
            c = _corners(rest.split(), len(verts), len(uvs), len(normals))
            tris += [(c[0], c[i], c[i + 1]) for i in range(1, len(c) - 1)]
    if not tris:
        raise ValueError(f"{path}: no faces; not a usable mesh")
    return verts, uvs, normals, tris


def read_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """(vertices, per-vertex UVs or None, per-vertex normals or None, triangles) of an OBJ. Polygons are
    fan-triangulated; corners that differ in UV or normal become separate vertices. A partly mapped mesh follows
    ``_mapped_triangles``. Normals are kept when every kept corner has one (MuJoCo shades with the file's normals;
    recomputing them from inconsistent winding shows as dark patches)."""
    verts, uvs, normals, tris = _parse_obj(path)
    mapped = _mapped_triangles(tris, len(uvs))
    corners = np.array([c for tri in (mapped or tris) for c in tri], dtype=np.int64)
    has_uv, has_n = bool(mapped), bool(normals) and bool((corners[:, 2] >= 0).all())
    cols = [0] + ([1] if has_uv else []) + ([2] if has_n else [])
    uniq, inv = np.unique(corners[:, cols], axis=0, return_inverse=True)
    v = np.asarray(verts, np.float64)[uniq[:, 0]]
    uv = np.asarray(uvs, np.float64)[uniq[:, 1]] if has_uv else None
    n = np.asarray(normals, np.float64)[uniq[:, -1]] if has_n else None
    return v, uv, n, inv.reshape(-1, 3)


def _quat_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _compiled_mesh(path: Path):
    import mujoco  # noqa: PLC0415 - heavy import, only when converting meshes

    for attrs in ("", ' inertia="shell"'):
        try:
            return mujoco.MjModel.from_xml_string(
                f'<mujoco><asset><mesh name="m" file="{Path(path).resolve()}"{attrs}/></asset></mujoco>')
        except ValueError as exc:
            err = exc
    raise ValueError(f"MuJoCo cannot read {path}: {err}")


def read_obj_mujoco(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """(vertices, UVs or None, normals or None, triangles) of a mesh file exactly as MuJoCo reads it (the look
    RoboCasa's models were made for: MuJoCo drops or keeps faces and UVs of partly mapped OBJs by its own rules),
    mapped back from MuJoCo's inertial frame to the file's frame. UVs are returned bottom-origin."""
    m = _compiled_mesh(path)
    va, vn = int(m.mesh_vertadr[0]), int(m.mesh_vertnum[0])
    fa, fn = int(m.mesh_faceadr[0]), int(m.mesh_facenum[0])
    rot, pos = _quat_matrix(m.mesh_quat[0]), np.asarray(m.mesh_pos[0], np.float64)
    verts = np.asarray(m.mesh_vert[va: va + vn], np.float64) @ rot.T + pos
    cols = [np.asarray(m.mesh_face[fa: fa + fn])]
    tc = nrm = None
    if m.mesh_texcoordnum[0]:
        ta, tn = int(m.mesh_texcoordadr[0]), int(m.mesh_texcoordnum[0])
        tc = np.asarray(m.mesh_texcoord[ta: ta + tn], np.float64)
        cols.append(np.asarray(m.mesh_facetexcoord[fa: fa + fn]))
    if m.mesh_normalnum[0]:
        na, nn = int(m.mesh_normaladr[0]), int(m.mesh_normalnum[0])
        nrm = np.asarray(m.mesh_normal[na: na + nn], np.float64) @ rot.T
        cols.append(np.asarray(m.mesh_facenormal[fa: fa + fn]))
    corners = np.stack([c.reshape(-1) for c in cols], axis=1)
    uniq, inv = np.unique(corners, axis=0, return_inverse=True)
    uv = None if tc is None else np.c_[tc[uniq[:, 1], 0], 1.0 - tc[uniq[:, 1], 1]]
    normals = None if nrm is None else nrm[uniq[:, -1]]
    return verts[uniq[:, 0]], uv, normals, inv.reshape(-1, 3)


def read_obj_uv(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """(vertices, per-vertex UVs or None, triangles) of an OBJ (``read_obj_mesh`` without the normals)."""
    v, uv, _, f = read_obj_mesh(path)
    return v, uv, f


def write_msh(path: Path, vertices: np.ndarray, faces: np.ndarray, uvs: np.ndarray | None = None,
              normals: np.ndarray | None = None) -> None:
    """Write a MuJoCo binary mesh; ``uvs`` (bottom-origin, per vertex) are stored top-origin; ``normals`` per
    vertex (unit length) or None to let the reader compute them."""
    v = np.ascontiguousarray(vertices, dtype=np.float32).reshape(-1, 3)
    f = np.ascontiguousarray(faces, dtype=np.int32).reshape(-1, 3)
    if len(v) < 4:
        raise ValueError(f"{path}: MuJoCo meshes need at least 4 vertices, got {len(v)}")
    t = np.zeros((0, 2), np.float32)
    if uvs is not None:
        t = np.ascontiguousarray(uvs, dtype=np.float32).reshape(-1, 2).copy()
        t[:, 1] = 1.0 - t[:, 1]
    nrm = np.zeros((0, 3), np.float32)
    if normals is not None:
        nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        nrm = (nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    head = np.array([len(v), len(nrm), len(t), len(f)], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        for arr in (head, v, nrm, t, f):
            fh.write(arr.tobytes())


def read_msh(path: Path) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """(vertices, bottom-origin UVs or None, triangles) of a ``.msh`` written by ``write_msh``."""
    data = np.fromfile(path, dtype=np.uint8)
    nv, nn, nt, nf = (int(x) for x in data[:16].view(np.int32))
    off = 16
    v = data[off: off + 12 * nv].view(np.float32).reshape(-1, 3)
    off += 12 * nv + 12 * nn
    uv = data[off: off + 8 * nt].view(np.float32).reshape(-1, 2).copy() if nt else None
    off += 8 * nt
    f = data[off: off + 12 * nf].view(np.int32).reshape(-1, 3)
    if uv is not None:
        uv[:, 1] = 1.0 - uv[:, 1]
    return v.astype(np.float64), uv, f.astype(np.int64)


__all__ = ["read_msh", "read_obj_mesh", "read_obj_mujoco", "read_obj_uv", "write_msh"]
