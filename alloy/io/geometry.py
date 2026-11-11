# SPDX-License-Identifier: Apache-2.0
"""Geometry helpers for the exporters and importers: poses, volumes, mesh files, primitive tessellation.

Poses are ``(x, y, z, qx, qy, qz, qw)`` (Warp order). All values float64, SI.

Example::

    from alloy.io.geometry import compose, shape_volume
    from alloy.io.model import IOShape
    compose((1, 0, 0, 0, 0, 0, 1), (0, 1, 0, 0, 0, 0, 1))   # -> (1, 1, 0, 0, 0, 0, 1)
    shape_volume(IOShape("s", "sphere", (0.1,)))           # -> 0.00418879...
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from alloy.io.model import IDENTITY, IOMesh, IOShape, Pose


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product of (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def quat_rotate(q, v) -> np.ndarray:
    """Rotate vector ``v`` by unit quaternion ``q`` (x, y, z, w)."""
    u, w = np.asarray(q[:3], np.float64), float(q[3])
    v = np.asarray(v, np.float64)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def quat_matrix(q) -> np.ndarray:
    """3x3 rotation matrix of a unit (x, y, z, w) quaternion."""
    return np.stack([quat_rotate(q, e) for e in np.eye(3)], axis=1)


def matrix_quat(r: np.ndarray) -> np.ndarray:
    """Unit (x, y, z, w) quaternion of a rotation matrix (Shepperd's method)."""
    r = np.asarray(r, np.float64)
    t = np.trace(r)
    if t > 0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k])
        q = [0.0, 0.0, 0.0, (r[k, j] - r[j, k]) / s]
        q[i], q[j], q[k] = 0.25 * s, (r[j, i] + r[i, j]) / s, (r[k, i] + r[i, k]) / s
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def compose(a: Pose, b: Pose) -> Pose:
    """Pose ``a * b`` (b expressed in a's frame)."""
    p = np.asarray(a[:3]) + quat_rotate(a[3:], b[:3])
    q = quat_mul(a[3:], b[3:])
    return tuple(float(x) for x in (*p, *(q / np.linalg.norm(q))))


def inverse(a: Pose) -> Pose:
    """Inverse pose."""
    qi = np.array([-a[3], -a[4], -a[5], a[6]], np.float64)
    p = -quat_rotate(qi, a[:3])
    return tuple(float(x) for x in (*p, *qi))


def is_identity(p: Pose, tol: float = 1e-9) -> bool:
    """True if ``p`` is the identity pose within ``tol`` (q and -q are the same rotation)."""
    a = np.asarray(p, np.float64)
    return bool(np.all(np.abs(a[:3]) < tol) and min(np.abs(a[3:] - IDENTITY[3:]).max(),
                                                       np.abs(a[3:] + IDENTITY[3:]).max()) < tol)


def mesh_volume(mesh: IOMesh) -> float:
    """Enclosed volume of a closed, outward-wound triangle mesh (divergence theorem); 0 for open sheets."""
    v = mesh.vertices[mesh.faces]
    return float(abs(np.einsum("ij,ij->i", v[:, 0], np.cross(v[:, 1], v[:, 2])).sum()) / 6.0)


def shape_volume(shape: IOShape) -> float:
    """Solid volume [m^3] of a shape; planes and open meshes give 0."""
    s = shape.size
    k = shape.kind
    if k == "box":
        return 8.0 * s[0] * s[1] * s[2]
    if k == "sphere":
        return 4.0 / 3.0 * math.pi * s[0] ** 3
    if k == "ellipsoid":
        return 4.0 / 3.0 * math.pi * s[0] * s[1] * s[2]
    if k == "capsule":
        return math.pi * s[0] ** 2 * 2.0 * s[1] + 4.0 / 3.0 * math.pi * s[0] ** 3
    if k == "cylinder":
        return math.pi * s[0] ** 2 * 2.0 * s[1]
    if k == "cone":
        return math.pi * s[0] ** 2 * 2.0 * s[1] / 3.0
    if k == "mesh" and shape.mesh is not None:
        return mesh_volume(shape.mesh)
    return 0.0


def tessellate(shape: IOShape, segments: int = 32) -> IOMesh:
    """A closed triangle mesh of a primitive in its shape frame (for formats without that primitive)."""
    from alloy._vendor import newton  # noqa: PLC0415 - heavy import, only when a format needs a mesh

    s, k = shape.size, shape.kind
    flat = {"compute_normals": False, "compute_uvs": False, "compute_inertia": False}
    round_ = {"num_latitudes": segments, "num_longitudes": segments, **flat}
    axial = {"up_axis": newton.Axis.Z, "segments": segments, **flat}   # Alloy primitives extend along +Z
    makers = {
        "box": lambda: newton.Mesh.create_box(s[0], s[1], s[2], duplicate_vertices=False, **flat),
        "sphere": lambda: newton.Mesh.create_sphere(s[0], **round_),
        "ellipsoid": lambda: newton.Mesh.create_ellipsoid(s[0], s[1], s[2], **round_),
        "capsule": lambda: newton.Mesh.create_capsule(s[0], s[1], **axial),
        "cylinder": lambda: newton.Mesh.create_cylinder(s[0], s[1], **axial),
        "cone": lambda: newton.Mesh.create_cone(s[0], s[1], **axial),
    }
    if k == "mesh" and shape.mesh is not None:
        return shape.mesh
    if k not in makers:
        raise ValueError(f"cannot tessellate a {k!r} shape; give it a size or use a box")
    m = makers[k]()
    return IOMesh(np.asarray(m.vertices, np.float64), np.asarray(m.indices, np.int32))


def obj_text(mesh: IOMesh, digits: int = 9) -> str:
    """Wavefront OBJ text of a mesh (vertices, then 1-based triangle faces)."""
    fmt = f"v %.{digits}g %.{digits}g %.{digits}g"
    lines = [fmt % tuple(v) for v in mesh.vertices]
    lines += ["f %d %d %d" % tuple(f + 1) for f in mesh.faces]
    return "\n".join(lines) + "\n"


def mesh_digest(mesh: IOMesh) -> str:
    """Short content hash of a mesh (names written mesh files so equal meshes share one file)."""
    h = hashlib.sha256(np.ascontiguousarray(mesh.vertices).tobytes() + np.ascontiguousarray(mesh.faces).tobytes())
    return h.hexdigest()[:12]


def write_obj(mesh: IOMesh, path: Path) -> str:
    """Write ``mesh`` as OBJ at ``path`` (parents created) and return the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = obj_text(mesh).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_obj(path: Path) -> IOMesh:
    """Triangles of an OBJ file (polygons fan-triangulated; normals, UVs and groups ignored)."""
    verts, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            verts.append([float(x) for x in parts[1:4]])
        elif parts[0] == "f":
            idx = [int(p.split("/")[0]) for p in parts[1:]]
            idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
            faces.extend([idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1))
    return IOMesh(np.asarray(verts, np.float64), np.asarray(faces, np.int32))
