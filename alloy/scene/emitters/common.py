# SPDX-License-Identifier: Apache-2.0
"""Shared emitter helpers: shape configs from cards, primitive shapes, meshes, geometry (public builder API only).

Emitters build each object in its own frame on a fresh sub-builder; the assembler merges it at the placement pose
with `ModelBuilder.add_builder(sub, xform=pose)` (newton@ac7f7830:newton/_src/sim/builder.py:5650), which moves
bodies, world-fixed shapes and particles alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.cards.types import MaterialCard, MeshSource, Primitive, ResolvedObject
from alloy.core.errors import AssemblyError

RIGID_GAP = 5.0e-3  # m, contact gap for ordinary rigid shapes (SPEC §4.2; never Newton's rigid_gap = 0.1 m)
DEFAULT_CONTACT_STIFFNESS = 2.0e6  # N/m per material (Surface.contact_stiffness default, D22)
# Newton capsules, cylinders and cones extend along their local Z axis (builder.py:8992).
Z_TO_X = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)


@dataclass(frozen=True)
class TriMesh:
    vertices: np.ndarray  # [V, 3] float64, m
    indices: np.ndarray  # [T, 3] int32


def surface_values(mat: MaterialCard) -> dict[str, float]:
    """Per-shape μ (dynamic), restitution, levers and stiffness of a material, SI."""
    s = mat.surface
    k = s.contact_stiffness.value if s.contact_stiffness is not None else DEFAULT_CONTACT_STIFFNESS
    return {
        "mu": float(s.mu_dynamic.value),
        "restitution": float(s.restitution.value),
        "mu_torsional": float(s.torsional_lever.value) if s.torsional_lever is not None else 0.0,
        "mu_rolling": float(s.rolling_lever.value) if s.rolling_lever is not None else 0.0,
        "ke": float(k),
    }


def shape_config(mat: MaterialCard, *, density: float | None = None, gap: float = RIGID_GAP,
                 collide: bool = True, torsional_lever: float | None = None) -> newton.ModelBuilder.ShapeConfig:
    """ShapeConfig carrying the card's surface values; the pair table overrides per contact (D13, D22)."""
    v = surface_values(mat)
    cfg = newton.ModelBuilder.ShapeConfig(
        density=float(mat.density.value) if density is None else density,
        mu=v["mu"], restitution=v["restitution"],
        mu_torsional=v["mu_torsional"] if torsional_lever is None else torsional_lever,
        mu_rolling=v["mu_rolling"], ke=v["ke"], kd=0.0, margin=0.0, gap=gap)
    if not collide:
        cfg.has_shape_collision = False
        cfg.has_particle_collision = False
    return cfg


def add_primitive(b: newton.ModelBuilder, body: int, prim: Primitive, cfg: newton.ModelBuilder.ShapeConfig,
                  xform: wp.transform | None = None, label: str | None = None) -> int:
    """One Newton shape for a card primitive; capsules/cylinders/cones keep Newton's Z axis."""
    s, kw = prim.size, {"xform": xform, "cfg": cfg, "label": label}
    if prim.kind == "box":
        return b.add_shape_box(body, hx=s[0], hy=s[1], hz=s[2], **kw)
    if prim.kind == "sphere":
        return b.add_shape_sphere(body, radius=s[0], **kw)
    if prim.kind == "capsule":
        return b.add_shape_capsule(body, radius=s[0], half_height=s[1], **kw)
    if prim.kind == "cylinder":
        return b.add_shape_cylinder(body, radius=s[0], half_height=s[1], **kw)
    if prim.kind == "cone":
        return b.add_shape_cone(body, radius=s[0], half_height=s[1], **kw)
    if prim.kind == "ellipsoid":
        return b.add_shape_ellipsoid(body, rx=s[0], ry=s[1], rz=s[2], **kw)
    if prim.kind == "plane":
        return b.add_shape_plane(body=body, width=0.0, length=0.0, xform=xform, cfg=cfg, label=label)
    raise AssemblyError(f"unknown primitive kind {prim.kind!r}")


def primitive_volume(prim: Primitive) -> float:
    """Solid volume [m³] (planes: 0)."""
    s = prim.size
    if prim.kind == "box":
        return 8.0 * s[0] * s[1] * s[2]
    if prim.kind == "sphere":
        return 4.0 / 3.0 * math.pi * s[0] ** 3
    if prim.kind == "capsule":
        return math.pi * s[0] ** 2 * (2.0 * s[1]) + 4.0 / 3.0 * math.pi * s[0] ** 3
    if prim.kind == "cylinder":
        return math.pi * s[0] ** 2 * 2.0 * s[1]
    if prim.kind == "cone":
        return math.pi * s[0] ** 2 * 2.0 * s[1] / 3.0
    if prim.kind == "ellipsoid":
        return 4.0 / 3.0 * math.pi * s[0] * s[1] * s[2]
    return 0.0


def load_mesh(src: MeshSource, root: Path | None = None) -> TriMesh:
    """Triangle mesh of a MeshSource (.obj, .stl, or .npz with `vertices`/`indices`), scaled; sha256-checked."""
    from alloy.core.ids import sha256_hex

    path = _mesh_file(src, root)
    if not path.exists():
        raise AssemblyError(f"mesh {path} not found")
    digest = sha256_hex(path.read_bytes())
    if src.sha256 and digest != src.sha256:
        raise AssemblyError(f"mesh {path}: sha256 {digest} != card {src.sha256}")
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path)
        verts, tris = np.asarray(data["vertices"], np.float64), np.asarray(data["indices"], np.int32)
    elif suffix == ".obj":                     # the card layer's reader: relative (negative) indices, range check
        from alloy.cards.geometry import read_obj

        try:
            obj = read_obj(path)
        except ValueError as e:
            raise AssemblyError(f"mesh {path}: {e}") from None
        verts, tris = obj.vertices, obj.faces.astype(np.int32)
    elif suffix == ".stl":                     # binary or ASCII; coincident corners merged (the card layer's reader)
        from alloy.cards.geometry import read_stl

        stl = read_stl(path)
        verts, tris = stl.vertices, stl.faces.astype(np.int32)
    else:
        raise AssemblyError(f"mesh {path}: unsupported format {path.suffix} (use .obj, .stl or .npz)")
    return TriMesh(verts * np.asarray(src.scale, np.float64), tris.reshape(-1, 3))


HULLS_SUFFIX = ".hulls.obj"   # SCR-OSS-10: precomputed convex hulls next to a `collision: decompose` card mesh
MAX_HULL_VERTS = 64
_HULL_FIX = ("regenerate it with the importer that made the card (python -m alloy.importers.<library> hulls <card>); "
             "CoACD never runs inside assembly")


def _mesh_file(src: MeshSource, root: Path | None) -> Path:
    return Path(src.path) if root is None or Path(src.path).is_absolute() else root / src.path


def hulls_file(src: MeshSource, root: Path | None = None) -> Path:
    """The sidecar `<stem>.hulls.obj` next to the card mesh `<stem>.obj` (SCR-OSS-10)."""
    path = _mesh_file(src, root)
    return path.with_name(path.stem + HULLS_SUFFIX)


def _hull_source_sha(side: Path) -> str:
    first = side.read_text(encoding="utf-8").split("\n", 1)[0].split()
    return first[2] if len(first) == 3 and first[:2] == ["#", "source_sha256:"] else ""


def _hull_groups(side: Path, data) -> list[str]:
    """Hull group names in file order; every `g`/`o` must be `hull_<k>`, non-empty, and own every face once."""
    import re

    declared = [ln.split()[1] for ln in side.read_text(encoding="utf-8").splitlines()
                if ln.split()[:1] in (["g"], ["o"]) and len(ln.split()) > 1]
    bad = [g for g in declared if not re.fullmatch(r"hull_\d+", g)]
    empty = [g for g in declared if g not in data.groups]
    owned = sum(len(v) for v in data.groups.values())
    if bad or empty or owned != len(data.faces) or len(set(declared)) != len(declared):
        raise AssemblyError(f"hull file {side}: groups must be unique non-empty 'o hull_<k>' blocks owning every face "
                            f"(bad names {bad[:3]}, empty {empty[:3]}, {len(data.faces) - owned} faces outside a "
                            "hull); " + _HULL_FIX)
    return sorted(data.groups, key=lambda g: int(data.groups[g].min()))


def load_hulls(src: MeshSource, root: Path | None = None) -> list[TriMesh]:
    """Convex hulls of a `collision: decompose` card, from the sidecar `<stem>.hulls.obj` (SCR-OSS-10), scaled.

    The sidecar's first line is `# source_sha256: <sha256 of the card mesh file>`; it holds one `o hull_<k>` block
    per hull, at most `src.max_parts` hulls of at most 64 vertices. A missing, stale or oversized sidecar raises.
    """
    from alloy.cards.geometry import read_obj
    from alloy.core.ids import sha256_hex

    side = hulls_file(src, root)
    if not side.exists():
        raise AssemblyError(f"mesh {_mesh_file(src, root)}: collision='decompose' needs its hull file {side}; "
                            + _HULL_FIX)
    want = sha256_hex(_mesh_file(src, root).read_bytes())
    if _hull_source_sha(side) != want:
        raise AssemblyError(f"hull file {side} is stale or has no '# source_sha256:' first line (card mesh sha256 "
                            f"{want}); " + _HULL_FIX)
    data = read_obj(side)
    names = _hull_groups(side, data)
    if not names or len(names) > src.max_parts:
        raise AssemblyError(f"hull file {side}: {len(names)} hulls, need 1..max_parts={src.max_parts}; " + _HULL_FIX)
    hulls = []
    for name in names:
        tris = data.faces[data.groups[name]]
        used, local = np.unique(tris, return_inverse=True)
        if len(used) > MAX_HULL_VERTS:
            raise AssemblyError(f"hull file {side}: {name} has {len(used)} vertices > {MAX_HULL_VERTS}; " + _HULL_FIX)
        verts = data.vertices[used] * np.asarray(src.scale, np.float64)
        hulls.append(TriMesh(verts, local.reshape(-1, 3).astype(np.int32)))
    return hulls


def newton_mesh(mesh: TriMesh) -> newton.Mesh:
    return newton.Mesh(mesh.vertices.astype(np.float32), mesh.indices.reshape(-1).astype(np.int32))


def grid_mesh(lx: float, ly: float, cell: float) -> TriMesh:
    """Rectangular sheet [−lx/2, lx/2]×[−ly/2, ly/2] in the z = 0 plane, alternating diagonals."""
    nx, ny = max(1, round(lx / cell)), max(1, round(ly / cell))
    xs, ys = np.linspace(-0.5 * lx, 0.5 * lx, nx + 1), np.linspace(-0.5 * ly, 0.5 * ly, ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    tris = []
    for i in range(nx):
        for j in range(ny):
            a, b2, c, d = i * (ny + 1) + j, (i + 1) * (ny + 1) + j, (i + 1) * (ny + 1) + j + 1, i * (ny + 1) + j + 1
            tris += [[a, b2, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b2, d], [b2, c, d]]
    return TriMesh(verts, np.asarray(tris, np.int32))


def sheet_axes(prim: Primitive) -> tuple[float, float, float, wp.quat]:
    """(l_a, l_b, thickness, rotation taking the grid's z = 0 plane onto the box's thin axis) of a thin box."""
    full = [2.0 * x for x in prim.size[:3]]
    thin = int(np.argmin(full))
    rot = {2: wp.quat_identity(), 0: wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi),
           1: wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)}[thin]
    others = [full[i] for i in range(3) if i != thin]
    return others[0], others[1], full[thin], rot


def object_volume(obj: ResolvedObject) -> float:
    """Volume of the object's shape source [m³] (primitives exact, particle regions as boxes, else the OBB)."""
    shape = obj.card.shape
    if isinstance(shape, Primitive):
        return primitive_volume(shape)
    region = getattr(shape, "region", None)
    if region is not None:
        lo, hi = np.asarray(region[0]), np.asarray(region[1])
        return float(np.prod(hi - lo))
    l1, l2, l3 = obj.extents
    return l1 * l2 * l3
