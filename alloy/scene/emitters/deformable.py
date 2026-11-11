# SPDX-License-Identifier: Apache-2.0
"""Deformable emitters: CLOTH_VBD / CLOTH_STYLE3D (add_cloth_mesh), SOFT_VBD (add_soft_mesh) and
GRAIN_MPM / PASTE_MPM / LIQUID_MPM (add_particle_grid + `mpm:*` per-particle attributes).

Mappings (object-model §4.4, derive.py): cloth tri_ke = E·h/(1−ν²), tri_ka = 0, edge_ke = E·h³/(12(1−ν²)) (Newton's
USD importer, CARD-4); areal density ρ·h; particle radius h/2; damping kd = β·k (Rayleigh β; VBD damps Green-strain
rate, newton@ac7f7830:newton/_src/solvers/vbd/particle_vbd_kernels.py:313, 651). Tets: (k_mu, k_lambda) = Lamé,
k_damp = β·μ (provisional, OM:291-304). MPM: 2 particles per voxel per axis, m = ρ·spacing³; granular friction
tan φ; a material with no friction angle (a paste or liquid) gets friction 0, a pressure-independent yield.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.cards.types import MaterialCard, MeshSource, ParticleSource, Primitive, ResolvedObject, TetSource
from alloy.core.errors import AssemblyError
from alloy.route.checks import GRAIN_PACKING
from alloy.route.types import Route, RouteDecision
from alloy.scene import derive, ortho_membrane
from alloy.solvers.vbd import cloth_profile
from alloy.scene.emitters import topology
from alloy.scene.emitters.common import TriMesh, grid_mesh, load_mesh, primitive_volume, sheet_axes

log = logging.getLogger("alloy.scene")

CLOTH_VERTS_TARGET = 1600
SOFT_TETS_PER_CELL = 6  # Kuhn split of each hexahedral cell
SOFT_RADIUS_FRACTION = 0.2  # SOFT_VBD contact radius / the mesh's shortest tet edge (median over tets), SCR-XB4-5-1


def _elastic(mat: MaterialCard, card: str) -> tuple[float, float, float]:
    if mat.elastic is None:
        raise AssemblyError(f"{card}: a deformable route needs material.elastic ({mat.name} has none)")
    beta = mat.elastic.rayleigh_beta.value if mat.elastic.rayleigh_beta is not None else 0.0
    return float(mat.elastic.youngs.value), float(mat.elastic.poisson.value), float(beta)


def sheet_thickness(obj: ResolvedObject) -> float:
    t = obj.card.features.thickness
    return float(t.value) if t is not None else float(obj.extents[2])


def cloth_mesh(obj: ResolvedObject) -> tuple[TriMesh, wp.quat]:
    """Triangle mesh of the sheet in the object frame and the rotation to apply to grid sheets."""
    shape = obj.card.shape
    if isinstance(shape, MeshSource):
        return load_mesh(shape), wp.quat_identity()
    if isinstance(shape, Primitive) and shape.kind == "box":
        la, lb, _, rot = sheet_axes(shape)
        n_target = min(obj.card.budget.particles, CLOTH_VERTS_TARGET)
        return grid_mesh(la, lb, math.sqrt(la * lb / n_target)), rot
    raise AssemblyError(f"{obj.card.name}: cloth needs a thin box primitive or a mesh")


def emit_cloth(b: newton.ModelBuilder, obj: ResolvedObject, d: RouteDecision) -> None:
    E, nu, beta = _elastic(obj.material, obj.card.name)
    h = sheet_thickness(obj)
    tri_ke, tri_ka, edge_ke = derive.cloth_stiffness(E, nu, h)     # VBD's Lame membrane pair (derive)
    if d.route == Route.CLOTH_STYLE3D:      # Style3D stretches by its ClothAniso values: its area / damping inputs
        tri_ke, tri_ka = E * h / (1.0 - nu * nu), 0.0                # as before the VBD Lame map (2026-11-08)
    edge_ke = derive.sheet_bending(obj.material, E, nu, h)            # the card's bending, not E·h³ at h
    mesh, rot = cloth_mesh(obj)
    aniso = obj.material.elastic.cloth
    ortho = d.route == Route.CLOTH_VBD and aniso is not None      # woven: the orthotropic membrane (ortho_membrane)
    if ortho and obj.card.route is None:                           # opt-in only: a card route override to cloth_vbd
        raise AssemblyError(f"{obj.card.name}: an anisotropic cloth (ClothAniso) on CLOTH_VBD needs the opt-in "
                            "orthotropic membrane: give the card route {route: cloth_vbd, reason: ...} "
                            "(alloy.scene.ortho_membrane), or keep it alone in its scene for CLOTH_STYLE3D")
    if ortho:
        tri_ke = tri_ka = 0.0                                      # Newton's isotropic membrane off on these triangles
    tri_start, e0, p0 = b.tri_count, len(b.edge_indices), len(b.particle_q)
    kw = dict(pos=wp.vec3(), rot=rot, scale=1.0, vel=wp.vec3(), density=float(obj.material.density.value) * h,
              tri_ke=tri_ke, tri_ka=tri_ka, tri_kd=beta * tri_ke, edge_ke=edge_ke, edge_kd=beta * edge_ke,
              particle_radius=0.5 * h, label=obj.card.name)
    if d.route == Route.CLOTH_STYLE3D and aniso is not None:
        _style3d_cloth(b, mesh, kw, aniso)
    elif topology.wants_presplit(obj):
        if ortho:
            raise AssemblyError(f"{obj.card.name}: a pre-split (tearable) cloth has no orthotropic membrane yet")
        topology.add_cloth(b, obj, mesh.vertices, mesh.indices, kw, h)
    else:
        b.add_cloth_mesh(vertices=mesh.vertices.astype(np.float32), indices=mesh.indices.reshape(-1).tolist(), **kw)
    if d.route == Route.CLOTH_VBD:          # VBD's hinge energy is ½·edge_ke·L·θ²: the plate map, not D (CARD-8)
        ke = cloth_profile.rewrite_hinges(b, e0, p0, edge_ke, beta)
        _mark_hysteresis(b, obj, e0)
        log.debug("%s: plate hinge map %g·D·L/(A1+A2): median %.4g N (D %.4g N·m)", obj.card.name,
                  cloth_profile.HINGE_C, float(np.median(ke[ke > 0])) if np.any(ke > 0) else 0.0, edge_ke)
    if ortho:
        ortho_membrane.emit(b, aniso, beta, mesh.vertices, (tri_start, e0, p0), edge_ke)
    log.debug("%s: %d cloth triangles (%s)", obj.card.name, b.tri_count - tri_start, d.route)


def _style3d_cloth(b: newton.ModelBuilder, mesh: TriMesh, kw: dict, aniso) -> None:
    """CLOTH_STYLE3D through Newton's Style3D helper (newton@ac7f7830 solvers/style3d/cloth.py:181), which writes the
    panel rest data the solver divides by (`style3d:edge_rest_area`, `edge_bending_cot`): the generic
    `add_cloth_mesh` leaves them 0 and the first step is NaN (CARD-1, napkin_cotton_aniso). Panel = the mesh x-y
    plane (warp along x). Stretch (weft, warp, shear) as the helper documents; bending per edge = a·sin¹² + b·cos¹² +
    c·4sin²cos² of the panel edge angle (cloth.py:144-155), so (bend_weft, bend_warp, √(bend_warp·bend_weft)) with
    the twist term from Huber's orthotropic plate."""
    from alloy._vendor.newton.solvers import style3d

    if "style3d:tri_aniso_ke" not in b.custom_attributes:
        raise AssemblyError("CLOTH_STYLE3D needs SolverStyle3D custom attributes registered on the builder")
    bw, bf = float(aniso.bend_warp.value), float(aniso.bend_weft.value)
    style3d.add_cloth_mesh(
        b, pos=kw["pos"], rot=kw["rot"], vel=kw["vel"], vertices=mesh.vertices.astype(np.float64).tolist(),
        indices=mesh.indices.reshape(-1).tolist(), density=kw["density"],
        tri_aniso_ke=wp.vec3(float(aniso.stretch_weft.value), float(aniso.stretch_warp.value), float(aniso.shear.value)),
        edge_aniso_ke=wp.vec3(bf, bw, math.sqrt(bw * bf)), tri_ka=kw["tri_ka"], tri_kd=kw["tri_kd"],
        edge_kd=kw["edge_kd"], particle_radius=kw["particle_radius"], label=kw["label"])


def emit_soft(b: newton.ModelBuilder, obj: ResolvedObject, d: RouteDecision) -> None:
    E, nu, beta = _elastic(obj.material, obj.card.name)
    k_mu, k_lambda = derive.lame(E, nu)
    verts, tets, rho = soft_mesh(obj)
    kw = dict(pos=wp.vec3(), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(), label=obj.card.name, density=rho,
              k_mu=k_mu, k_lambda=k_lambda, k_damp=beta * k_mu, particle_radius=soft_particle_radius(verts, tets))
    t0 = b.tet_count
    if topology.wants_presplit(obj):
        topology.add_soft(b, obj, verts, tets, kw)
    else:
        b.add_soft_mesh(vertices=verts.astype(np.float32).tolist(), indices=tets.reshape(-1).tolist(), **kw)
    _mark_j2(b, obj, t0)


def _mark_hysteresis(b: newton.ModelBuilder, obj: ResolvedObject, e0: int) -> None:
    """A card with KES 2HB (``elastic.bending_hysteresis``): its hinges hold a fold by friction (vbd/hysteresis.py)."""
    from alloy.solvers.vbd import hysteresis  # noqa: PLC0415

    hb = hysteresis.card_hb(obj.material)
    if hb > 0.0:
        n = hysteresis.mark_edges(b, e0, len(b.edge_indices), hb)
        log.debug("%s: %d hysteretic hinges, hb %.4g N", obj.card.name, n, hb)


def _mark_j2(b: newton.ModelBuilder, obj: ResolvedObject, t0: int) -> None:
    """A `plastic.model: j2` card yields: its tets get σ_y0 and H for the J2 hook (alloy.solvers.plastic.tets)."""
    p = obj.material.plastic
    if p is None or p.model != "j2":
        return
    from alloy.solvers.plastic.tets import mark_tets  # noqa: PLC0415

    hmod = float(p.hardening_modulus.value) if p.hardening_modulus is not None else 0.0
    mark_tets(b, t0, b.tet_count, float(p.yield_stress.value), hmod)


def soft_mesh(obj: ResolvedObject) -> tuple[np.ndarray, np.ndarray, float]:
    """The SOFT_VBD tet mesh (vertices [V, 3], tets [T, 4], object frame) and the density that gives the card's mass
    (a round primitive's voxel body is rescaled to ρ·V of the primitive)."""
    rho, shape = float(obj.material.density.value), obj.card.shape
    if isinstance(shape, Primitive) and shape.kind == "box":
        verts, tets = box_mesh(shape, obj.card.budget.tets)
        return verts, tets, rho
    if isinstance(shape, Primitive) and shape.kind in ROUND_PRIMITIVES:
        verts, tets = round_mesh(shape, obj.card.budget.tets)
        return verts, tets, rho * primitive_volume(shape) / tet_volume(verts, tets)
    verts, tets = _tet_mesh(obj)
    return verts, tets, rho


TET_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def soft_particle_radius(verts: np.ndarray, tets: np.ndarray) -> float:
    """SOFT_VBD particle contact radius (SCR-XB4-5-1; add_soft_mesh's default is Newton's 0.1 m):
    `SOFT_RADIUS_FRACTION` x the median over tets of each tet's shortest edge, which is the shortest cell edge of a Kuhn
    grid (DC5's pad radius, 1/5 of its 0.49 mm cell). A soft solid rests one radius above what it touches, less the
    contact compression F / soft_contact_ke; VBD's 0.4·r/|v| substep bound (`ParticleEstimator`) scales with it.
    Particle-shape contact is vertex-only: a rigid feature narrower than the gaps between the particle spheres
    (~0.6 of a cell) can pass through a face between vertices (a probe tip, a finger edge); a platen or a table
    cannot."""
    v, t = np.asarray(verts, np.float64).reshape(-1, 3), np.asarray(tets, np.int64).reshape(-1, 4)
    if t.shape[0] == 0:
        raise AssemblyError("SOFT_VBD: the tet mesh is empty, no particle radius")
    edges = np.stack([np.linalg.norm(v[t[:, a]] - v[t[:, b]], axis=1) for a, b in TET_EDGES], axis=1)
    return float(SOFT_RADIUS_FRACTION * np.median(edges.min(axis=1)))


def box_mesh(prim: Primitive, tet_budget: int) -> tuple[np.ndarray, np.ndarray]:
    """Box of Kuhn-split hexahedra (6 conforming tets per cell) for add_soft_mesh, which lumps ρ·V_tet/4 per
    vertex. Not add_soft_grid: it gives every vertex ρ·V_cell (builder.py:11995), so a box of n³ cells weighs
    (n+1)³/n³ × ρ·V (CARD-3 readback caught it)."""
    full = np.asarray([2.0 * x for x in prim.size[:3]])
    cell = (np.prod(full) * SOFT_TETS_PER_CELL / max(tet_budget, SOFT_TETS_PER_CELL)) ** (1.0 / 3.0)
    dims = np.maximum(1, np.floor(full / cell)).astype(int)
    dims -= ((dims > 3) & (dims % 2 == 1)).astype(int)    # even: a body slides along an odd axis (test_soft_box_rest)
    for k in np.nonzero(dims == 3)[0]:                    # a thin axis keeps 3 layers, 4 when the budget allows
        dims[k] = 4 if SOFT_TETS_PER_CELL * int(dims.prod()) // 3 * 4 <= tet_budget else 3
    return box_tets(full, dims)


KUHN = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def box_tets(full: np.ndarray, dims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertices [V, 3] (centred) and positively oriented tets [T, 4] of a dims-cell box of size `full`."""
    nx, ny, nz = (int(d) for d in dims)
    axes = [np.linspace(-0.5 * full[k], 0.5 * full[k], int(dims[k]) + 1) for k in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    vid = lambda c: (c[0] * (ny + 1) + c[1]) * (nz + 1) + c[2]  # noqa: E731
    tets = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):    # each cell mirrored along its odd-index axes: one diagonal everywhere shears
                base, flip = np.array([i, j, k]), np.array([i % 2, j % 2, k % 2])
                for perm in KUHN:
                    local, path = np.zeros(3, dtype=int), [vid(base + flip)]
                    for ax in perm:
                        local[ax] = 1
                        path.append(vid(base + (local ^ flip)))
                    tets.append(_oriented(verts, path))
    return verts, np.asarray(tets, np.int32)


def round_mesh(prim: Primitive, tet_budget: int) -> tuple[np.ndarray, np.ndarray]:
    """A sphere/ellipsoid/cylinder/capsule as the Kuhn tets of its bounding box whose centroids lie inside it (a
    voxelised body; the emitter rescales density so the mass is ρ·V of the primitive). rubber_ball_40mm did not
    assemble before (CARD-1)."""
    full = 2.0 * np.asarray(_half_extents(prim), np.float64)
    cell = (np.prod(full) * SOFT_TETS_PER_CELL / max(tet_budget, SOFT_TETS_PER_CELL)) ** (1.0 / 3.0)
    verts, tets = box_tets(full, np.maximum(2, np.round(full / cell)).astype(int))
    tets = tets[_inside(prim, verts[tets].mean(axis=1))]
    used = np.unique(tets)
    remap = np.full(len(verts), -1, np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[tets].astype(np.int32)


def tet_volume(verts: np.ndarray, tets: np.ndarray) -> float:
    a, b, c, d = (verts[tets[:, i]] for i in range(4))
    return float(np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)).sum() / 6.0)


def _oriented(verts: np.ndarray, t: list[int]) -> list[int]:
    a, b2, c, d = (verts[i] for i in t)
    return t if np.dot(np.cross(b2 - a, c - a), d - a) > 0 else [t[0], t[2], t[1], t[3]]


def _tet_mesh(obj: ResolvedObject) -> tuple[np.ndarray, np.ndarray]:
    shape = obj.card.shape
    if isinstance(shape, TetSource) and shape.path is not None and shape.path.endswith(".npz"):
        data = np.load(shape.path)
        return np.asarray(data["vertices"], np.float64), np.asarray(data["indices"], np.int32).reshape(-1, 4)
    if isinstance(shape, (TetSource, MeshSource)):
        try:
            import pytetwild  # noqa: F401  (tetrahedralize uses it)
        except ImportError as exc:
            raise AssemblyError(f"{obj.card.name}: tetrahedralising a mesh needs the optional [mesh] extra "
                                "(pytetwild), not installed") from exc
        src = load_mesh(shape) if isinstance(shape, MeshSource) else load_mesh(MeshSource(shape.from_mesh, ""))
        edge = ({"edge_length_abs": float(shape.edge_length)} if isinstance(shape, TetSource) else
                {"edge_length_fac": TETWILD_EDGE_FAC})
        return tetrahedralize(src.vertices, src.indices, edge)
    raise AssemblyError(f"{obj.card.name}: SOFT_VBD needs a box/round primitive, a TetSource (.npz) or a mesh")


TETWILD_EDGE_FAC = 0.05     # fTetWild target edge for a mesh card: this fraction of the bounding-box diagonal
TETWILD_THREADS = 4         # a first build on a shared host does not take every core (repeatability is lost anyway)
TETWILD_MAX_CELLS = 50_000  # bounding-box volume / edge^3 cap (8-28 tets a cell measured): past it fTetWild runs long
_TETS: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def tetrahedralize(vertices: np.ndarray, faces: np.ndarray, edge: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """fTetWild's tets of a closed surface (vertices [V, 3], positively oriented tets [T, 4]), cached by content.

    pytetwild returns tets. Newton's ``remesh_ftetwild``, used here before, returns the surface of its tet mesh, and
    its triangles were reshaped into "tets" (SCR OSS-9-5): a reshape error, or nonsense tets that Newton dropped as
    inverted. fTetWild's mesh optimisation is on: without it 73 % of a dough ball's tets have quality < 0.1. fTetWild
    is not repeatable (two runs of one mesh give different tet counts, one thread too), so the result is cached in
    the process and in ``$ALLOY_CACHE_DIR/tets/`` (default ``~/.cache/alloy``), keyed by the surface, the edge setting
    and the pytetwild version, and the first process to write a key wins: every build of a mesh on one host gets the
    same tets. A card that must build the same tets on every host ships them (a TetSource ``.npz``)."""
    import pytetwild  # noqa: PLC0415 (optional [mesh] extra; _tet_mesh checked the import)

    v = np.ascontiguousarray(vertices, np.float64).reshape(-1, 3)
    f = np.ascontiguousarray(faces, np.int32).reshape(-1, 3)
    key = _tet_key(v, f, edge, getattr(pytetwild, "__version__", "?"))
    if key not in _TETS:
        _TETS[key] = _cached_tets(key) or _run_tetwild(pytetwild, v, f, edge, key)
    tv, tt = _TETS[key]
    return tv.copy(), tt.copy()


def _tet_key(v: np.ndarray, f: np.ndarray, edge: dict[str, float], version: str) -> str:
    h = hashlib.sha256(v.tobytes())
    h.update(f.tobytes())
    h.update(json.dumps({"edge": edge, "optimize": True, "pytetwild": version}, sort_keys=True).encode())
    return h.hexdigest()[:32]


def _tet_cache_file(key: str) -> Path:
    return Path(os.environ.get("ALLOY_CACHE_DIR", Path.home() / ".cache" / "alloy")) / "tets" / f"{key}.npz"


def _cached_tets(key: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = _tet_cache_file(key)
    try:
        with np.load(path) as data:
            return np.asarray(data["vertices"], np.float64), np.asarray(data["tets"], np.int32)
    except (OSError, KeyError, ValueError):
        return None


def _check_tet_size(v: np.ndarray, edge: dict[str, float]) -> None:
    ext = v.max(axis=0) - v.min(axis=0)
    e = edge.get("edge_length_abs") or edge.get("edge_length_fac", TETWILD_EDGE_FAC) * float(np.linalg.norm(ext))
    cells = float(np.prod(ext)) / e**3
    if cells > TETWILD_MAX_CELLS:
        raise AssemblyError(f"fTetWild at a {e:.3g} m edge: {cells:.3g} edge cubes in the bounding box (cap "
                            f"{TETWILD_MAX_CELLS}, 8-28 tets each); raise the TetSource's edge_length")


def _run_tetwild(pytetwild, v: np.ndarray, f: np.ndarray, edge: dict[str, float],
                 key: str) -> tuple[np.ndarray, np.ndarray]:
    _check_tet_size(v, edge)
    with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):   # fTetWild writes __tracked_surface.stl
        tv, tt = pytetwild.tetrahedralize(v, f, optimize=True, quiet=True, num_threads=TETWILD_THREADS, **edge)
    tv, tt = _compact_oriented(np.asarray(tv, np.float64), np.asarray(tt, np.int64))
    if tt.shape[0] == 0:
        raise AssemblyError(f"fTetWild made no tets from a {len(f)}-face surface (is it closed?)")
    log.info("fTetWild: %d tets, %d vertices (cache key %s)", tt.shape[0], tv.shape[0], key)
    return _store_tets(key, tv, tt)


def _store_tets(key: str, tv: np.ndarray, tt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Write the tets to the disk cache. The first writer wins: a process that finds the key already written (a
    concurrent first build) drops its own tets and uses that file's."""
    path, tmp = _tet_cache_file(key), None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp.npz")
        with os.fdopen(fd, "wb") as fh:
            np.savez(fh, vertices=tv, tets=tt)
        os.link(tmp, path)
    except FileExistsError:
        theirs = _cached_tets(key)
        if theirs is not None:
            return theirs
    except OSError as exc:
        log.warning("tet mesh cache %s not written (%s): later builds re-run fTetWild", path, exc)
    finally:
        if tmp is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
    return tv, tt


def _compact_oriented(v: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tets with positive volume (``_oriented``'s sign; fTetWild's are negative) over the vertices they use."""
    a, b, c, d = (v[t[:, i]] for i in range(4))
    neg = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) < 0.0
    t = t.copy()
    t[neg] = t[neg][:, [0, 2, 1, 3]]
    used = np.unique(t)
    remap = np.full(len(v), -1, np.int64)
    remap[used] = np.arange(len(used))
    return v[used], remap[t].astype(np.int32)


# --- MPM ----------------------------------------------------------------------------------------------------------

def mpm_values(mat: MaterialCard) -> dict[str, float]:
    """`mpm:*` per-particle values from a material card (names: solvers/implicit_mpm/solver_implicit_mpm.py:1379).

    ``friction`` is tan φ for a granular material (a friction angle on the card) and 0 for everything else: a paste or
    liquid yields at ``yield_stress`` whatever its pressure. Newton's per-particle default (0.5) would pile honey and
    ketchup like sand (docs/notes/viscoplastic-design.md §1).
    A granular card's Mohr-Coulomb cohesion (``plastic.cohesion``, soil) is Newton's pressure-independent
    ``yield_stress``, s = c with no factor: a free-standing column stands at 0.85-0.89 H_c and slumps at 1.18-1.3 H_c
    (H_c = 2 c tan(45° + φ/2) / (ρ_bulk g)) for φ 0-40°; at φ 0 its boundary is 1.03 ± 0.03 H_c, a strip footing's
    1.00 ± 0.05 of Prandtl's (π + 2) c (CPU; docs/notes/soil-design.md). A card that sets both raises (two meanings
    for one parameter). A j2 card gives ``mpm_solid.j2_mpm_values`` (sigma_y / sqrt(3), GAP #6)."""
    from alloy.integration import mpm_solve  # noqa: PLC0415 (the friction cone, gap #5b)
    from alloy.solvers.plastic import mpm_solid  # noqa: PLC0415 (a j2 solid: von Mises, GAP #6)

    p = mat.plastic
    if p is not None and p.model == "j2":
        return mpm_solid.j2_mpm_values(mat)
    out: dict[str, float] = {}
    if mat.elastic is not None:
        out["young_modulus"] = float(mat.elastic.youngs.value)
        out["poisson_ratio"] = float(mat.elastic.poisson.value)
    angle = p.friction_angle if p is not None else None
    out["friction"] = mpm_solve.cone_friction(float(angle.value)) if angle is not None else 0.0   # tan(phi) default
    out.update({n: float(q.value) for n in ("yield_pressure", "yield_stress", "hardening", "dilatancy")
                if (q := getattr(p, n, None)) is not None})
    if p is not None and p.cohesion is not None:
        if p.yield_stress is not None:
            raise AssemblyError(f"material {mat.name}: plastic.cohesion and plastic.yield_stress both set; a "
                                "granular card gives its Mohr-Coulomb cohesion only")
        out["yield_stress"] = float(p.cohesion.value)
    if mat.fluid is not None:
        out["viscosity"] = float(mat.fluid.viscosity.value)
    return out


def mpm_density(obj: ResolvedObject, d: RouteDecision) -> float:
    """Continuum density of an MPM object [kg/m³]. A granular card holds its grain (solid) density (WP1's card
    ruling, as GRAIN_RIGID and ``grain_count`` use it), so GRAIN_MPM fills at the bulk density GRAIN_PACKING x
    density; pastes and liquids are continua at their card density."""
    rho = float(obj.material.density.value)
    return GRAIN_PACKING * rho if d.route == Route.GRAIN_MPM else rho


def emit_mpm(b: newton.ModelBuilder, obj: ResolvedObject, d: RouteDecision) -> None:
    voxel = float(d.numbers.get("voxel", 0.0))
    if voxel <= 0.0:
        raise AssemblyError(f"{obj.card.name}: MPM route without a voxel size in the decision")
    lo, hi = _region(obj)
    spacing = 0.5 * voxel
    n = np.maximum(1, np.floor((hi - lo) / spacing + 1e-6).astype(int))    # 0.04 / 0.0025 = 15.999999999999996 (WP1)
    if int(np.prod(n)) > obj.card.budget.particles:
        from alloy.core.errors import CapacityError

        raise CapacityError(f"{obj.card.name}: {int(np.prod(n))} MPM particles > budget {obj.card.budget.particles}")
    rho = mpm_density(obj, d)
    start = len(b.particle_q)
    shape = obj.card.shape
    if isinstance(shape, ParticleSource) or shape.kind == "box":
        b.add_particle_grid(pos=wp.vec3(*(lo + 0.5 * spacing)), rot=wp.quat_identity(), vel=wp.vec3(),
                            dim_x=int(n[0]), dim_y=int(n[1]), dim_z=int(n[2]), cell_x=spacing, cell_y=spacing,
                            cell_z=spacing, mass=rho * spacing**3, jitter=0.0, radius_mean=0.5 * spacing)
    else:                                   # the lattice points inside a round primitive (a dough ball)
        axes = [lo[i] + (np.arange(n[i]) + 0.5) * spacing for i in range(3)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        pts = pts[_inside(shape, pts)]
        b.add_particles(pos=[tuple(map(float, p)) for p in pts], vel=[(0.0, 0.0, 0.0)] * len(pts),
                        mass=[rho * spacing**3] * len(pts), radius=[0.5 * spacing] * len(pts))
    parts = range(start, len(b.particle_q))
    _set_particle_attrs(b, "mpm", mpm_values(obj.material), parts)
    hb = rheology_values(obj.material)
    if hb:
        _set_particle_attrs(b, "rheo", hb, parts)


def rheology_values(mat: MaterialCard) -> dict[str, float]:
    """``rheo:*`` per-particle Herschel-Bulkley values (consistency K, flow index n), empty for a material without
    them; SolverRheologyMPM turns them into a per-step viscosity (alloy.solvers.mpm_rheology)."""
    f = mat.fluid
    if f is None or f.consistency is None or f.flow_index is None:
        return {}
    return {"consistency": float(f.consistency.value), "flow_index": float(f.flow_index.value)}


ROUND_PRIMITIVES = ("sphere", "ellipsoid", "cylinder", "capsule")


def _region(obj: ResolvedObject) -> tuple[np.ndarray, np.ndarray]:
    shape = obj.card.shape
    if isinstance(shape, ParticleSource):
        return np.asarray(shape.region[0], np.float64), np.asarray(shape.region[1], np.float64)
    if isinstance(shape, Primitive) and shape.kind in ("box", *ROUND_PRIMITIVES):
        h = np.asarray(_half_extents(shape), np.float64)
        return -h, h
    raise AssemblyError(f"{obj.card.name}: MPM needs a ParticleSource region or a box/sphere/ellipsoid/cylinder/"
                        "capsule primitive")


def _half_extents(p: Primitive) -> tuple[float, float, float]:
    s = p.size
    if p.kind in ("box", "ellipsoid"):
        return s[0], s[1], s[2]
    if p.kind == "sphere":
        return s[0], s[0], s[0]
    return s[0], s[0], s[1] + (s[0] if p.kind == "capsule" else 0.0)


def _inside(p: Primitive, x: np.ndarray) -> np.ndarray:
    s = p.size
    if p.kind == "sphere":
        return np.linalg.norm(x, axis=1) <= s[0]
    if p.kind == "ellipsoid":
        return np.sum((x / np.asarray(s[:3])) ** 2, axis=1) <= 1.0
    if p.kind == "cylinder":
        return (np.hypot(x[:, 0], x[:, 1]) <= s[0]) & (np.abs(x[:, 2]) <= s[1])
    z = np.clip(x[:, 2], -s[1], s[1])                                    # capsule: distance to the axis segment
    return np.linalg.norm(x - np.stack([0 * z, 0 * z, z], 1), axis=1) <= s[0]


def _set_particle_attrs(b: newton.ModelBuilder, ns: str, values: dict[str, float], parts: range) -> None:
    for name, v in values.items():
        key = f"{ns}:{name}"
        if key not in b.custom_attributes:
            raise AssemblyError(f"custom attribute {key} not registered (register the MPM capability first)")
        store = b.custom_attributes[key].values
        for i in parts:
            store[i] = v
