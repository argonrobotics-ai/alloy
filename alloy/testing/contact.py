# SPDX-License-Identifier: Apache-2.0
"""Contact test geometry (WP3): meshed peg/hole, rings, thin plates.

Every builder returns float32 vertices and int32 triangle indices with outward-facing CCW winding (Newton's
mesh-primitive narrow phase back-face culls, newton@ac7f7830:docs/concepts/collisions.rst:1068-1078), and a
``newton.Mesh`` wrapper. Units are metres; the axis of every round part is +Z through the shape origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PegHoleSpec",
    "segments_for_chord",
    "chord_error",
    "cylinder_tris",
    "blind_hole_tris",
    "ring_tris",
    "box_tris",
    "signed_volume",
    "to_newton_mesh",
    "peg_hole_mesh",
]


@dataclass(frozen=True)
class PegHoleSpec:
    """The CON-46 peg-in-hole: a 12.0 mm peg in a 12.2 mm blind bore (0.1 mm radial clearance)."""

    peg_d: float = 12.0e-3
    peg_len: float = 30.0e-3
    bore_d: float = 12.2e-3
    outer_d: float = 30.0e-3
    height: float = 25.0e-3
    depth: float = 20.0e-3          # bore depth from the top face; bottom at z = height - depth
    segments: int = 96              # 12.2 mm bore -> chord error 3.3 um (CON-46)

    @property
    def radial_clearance(self) -> float:
        return 0.5 * (self.bore_d - self.peg_d)

    @property
    def bottom_z(self) -> float:
        return self.height - self.depth


def chord_error(radius: float, segments: int) -> float:
    """Max distance between a circle and its inscribed regular N-gon: r(1 - cos(pi/N)) [m]."""
    return radius * (1.0 - math.cos(math.pi / segments))


def segments_for_chord(radius: float, chord_tol: float) -> int:
    """Smallest N with r(1 - cos(pi/N)) <= chord_tol (contact §4.4 tessellation rule)."""
    if radius <= 0.0 or chord_tol <= 0.0:
        raise ValueError(f"radius and chord_tol must be positive, got {radius}, {chord_tol}")
    if chord_tol >= radius:
        return 3
    n = math.ceil(math.pi / math.acos(1.0 - chord_tol / radius))
    return max(3, n)


def _ring(radius: float, z: float, n: int) -> np.ndarray:
    th = np.arange(n) * (2.0 * math.pi / n)
    return np.stack([radius * np.cos(th), radius * np.sin(th), np.full(n, z)], axis=1)


def _orient(verts: np.ndarray, tris: list[tuple[int, int, int]], dirs: list[np.ndarray]) -> np.ndarray:
    """Flip each triangle so its normal points along its expected outward direction."""
    out = np.empty((len(tris), 3), dtype=np.int32)
    for k, ((a, b, c), d) in enumerate(zip(tris, dirs)):
        nrm = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        out[k] = (a, b, c) if float(np.dot(nrm, d)) >= 0.0 else (a, c, b)
    return out


def _wall(lo: int, hi: int, n: int, outward: bool, tris: list, dirs: list) -> None:
    """Quad strip between rings ``lo`` and ``hi`` (vertex offsets), normals radial out (or in)."""
    th = np.arange(n) * (2.0 * math.pi / n)
    for i in range(n):
        j = (i + 1) % n
        rad = np.array([math.cos(th[i]) + math.cos(th[j]), math.sin(th[i]) + math.sin(th[j]), 0.0])
        d = rad if outward else -rad
        tris += [(lo + i, lo + j, hi + j), (lo + i, hi + j, hi + i)]
        dirs += [d, d]


def _annulus(inner: int, outer: int, n: int, up: bool, tris: list, dirs: list) -> None:
    d = np.array([0.0, 0.0, 1.0 if up else -1.0])
    for i in range(n):
        j = (i + 1) % n
        tris += [(outer + i, outer + j, inner + j), (outer + i, inner + j, inner + i)]
        dirs += [d, d]


def _fan(center: int, ring: int, n: int, up: bool, tris: list, dirs: list) -> None:
    d = np.array([0.0, 0.0, 1.0 if up else -1.0])
    for i in range(n):
        tris.append((center, ring + i, ring + (i + 1) % n))
        dirs.append(d)


def cylinder_tris(radius: float, half_height: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Closed cylinder centred at the origin, axis +Z."""
    n = segments
    verts = np.concatenate([_ring(radius, -half_height, n), _ring(radius, half_height, n),
                            [[0.0, 0.0, -half_height], [0.0, 0.0, half_height]]])
    tris: list = []
    dirs: list = []
    _wall(0, n, n, True, tris, dirs)
    _fan(2 * n, 0, n, False, tris, dirs)
    _fan(2 * n + 1, n, n, True, tris, dirs)
    return verts.astype(np.float32), _orient(verts, tris, dirs)


def blind_hole_tris(spec: PegHoleSpec) -> tuple[np.ndarray, np.ndarray]:
    """Solid cylinder (outer_d, height) on z in [0, height] minus a bore (bore_d, depth) from the top face."""
    n = spec.segments
    ro, rb, zb = 0.5 * spec.outer_d, 0.5 * spec.bore_d, spec.bottom_z
    if not 0.0 < zb < spec.height or rb >= ro:
        raise ValueError(f"invalid blind hole: bottom_z={zb}, bore_r={rb}, outer_r={ro}")
    verts = np.concatenate([_ring(ro, 0.0, n), _ring(ro, spec.height, n), _ring(rb, spec.height, n),
                            _ring(rb, zb, n), [[0.0, 0.0, 0.0], [0.0, 0.0, zb]]])
    o0, o1, b1, b0, c0, cb = 0, n, 2 * n, 3 * n, 4 * n, 4 * n + 1
    tris: list = []
    dirs: list = []
    _wall(o0, o1, n, True, tris, dirs)
    _annulus(b1, o1, n, True, tris, dirs)
    _wall(b0, b1, n, False, tris, dirs)
    _fan(cb, b0, n, True, tris, dirs)
    _fan(c0, o0, n, False, tris, dirs)
    return verts.astype(np.float32), _orient(verts, tris, dirs)


def ring_tris(inner_r: float, outer_r: float, height: float, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Through-hole ring (tube) on z in [0, height]."""
    if not 0.0 < inner_r < outer_r or height <= 0.0:
        raise ValueError(f"invalid ring: inner_r={inner_r}, outer_r={outer_r}, height={height}")
    n = segments
    verts = np.concatenate([_ring(outer_r, 0.0, n), _ring(outer_r, height, n), _ring(inner_r, height, n),
                            _ring(inner_r, 0.0, n)])
    o0, o1, i1, i0 = 0, n, 2 * n, 3 * n
    tris: list = []
    dirs: list = []
    _wall(o0, o1, n, True, tris, dirs)
    _annulus(i1, o1, n, True, tris, dirs)
    _wall(i0, i1, n, False, tris, dirs)
    _annulus(i0, o0, n, False, tris, dirs)
    return verts.astype(np.float32), _orient(verts, tris, dirs)


def box_tris(hx: float, hy: float, hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned box centred at the origin (thin plates use a small hz)."""
    s = np.array([[x, y, z] for z in (-hz, hz) for y in (-hy, hy) for x in (-hx, hx)], dtype=np.float64)
    quads = [((0, 2, 3, 1), (0, 0, -1)), ((4, 5, 7, 6), (0, 0, 1)), ((0, 1, 5, 4), (0, -1, 0)),
             ((2, 6, 7, 3), (0, 1, 0)), ((0, 4, 6, 2), (-1, 0, 0)), ((1, 3, 7, 5), (1, 0, 0))]
    tris: list = []
    dirs: list = []
    for (a, b, c, d), nrm in quads:
        tris += [(a, b, c), (a, c, d)]
        dirs += [np.array(nrm, float)] * 2
    return s.astype(np.float32), _orient(s, tris, dirs)


def signed_volume(verts: np.ndarray, tris: np.ndarray) -> float:
    """Divergence-theorem volume; positive for a closed mesh with outward winding [m^3]."""
    v = verts.astype(np.float64)
    a, b, c = v[tris[:, 0]], v[tris[:, 1]], v[tris[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def to_newton_mesh(verts: np.ndarray, tris: np.ndarray, *, compute_inertia: bool = True):
    """Wrap (vertices, triangles) as a ``newton.Mesh`` (imported lazily so geometry stays numpy-only)."""
    from alloy._vendor import newton  # noqa: PLC0415

    return newton.Mesh(verts, tris.reshape(-1).astype(np.int32), compute_inertia=compute_inertia)


def peg_hole_mesh(spec: PegHoleSpec | None = None):
    """Meshed peg and blind hole for the precision-pair tests: returns (peg_mesh, hole_mesh, spec)."""
    spec = spec or PegHoleSpec()
    peg = to_newton_mesh(*cylinder_tris(0.5 * spec.peg_d, 0.5 * spec.peg_len, spec.segments))
    hole = to_newton_mesh(*blind_hole_tris(spec), compute_inertia=False)
    return peg, hole, spec


@dataclass
class PegHoleScene:
    """A built peg-in-hole pack: ``model``, the precision ``spec`` (template ids), per-world peg bodies, SDF infos."""

    model: object
    spec: object
    geometry: PegHoleSpec
    peg_bodies: list[int]
    peg_mass: float
    sdf: list = None


HYDRO_GAP_M = 5e-6     # per shape: gap_a + gap_b + sqrt(3)*voxel < 0.1 mm radial clearance at voxel 50 µm


def peg_hole_scene(worlds: int, *, device: str, mode: str = "hydroelastic", geometry: PegHoleSpec | None = None,
                   voxel: float = 5e-5, kh: float = 1e11, pair_stiffness: float | None = 2e6,
                   start_gap: float = 2e-4, tilt_deg: float = 0.0, lateral: float = 0.0, density: float = 7800.0,
                   mu: float = 0.3, cache_dir=None, ccd: str = "speculative", gap: float | None = None,
                   narrow_band: tuple[float, float] = (-5e-4, 5e-4)) -> PegHoleScene:
    """Meshed peg standing ``start_gap`` above the bore bottom (axis tilted by ``tilt_deg`` about X, shifted by
    ``lateral`` in x), static blind hole, precision pair (hole, peg) filtered from native collision, flat imp set on
    both shapes, SDFs baked on CUDA (``mode`` "hydroelastic" or "sdf"). ``gap`` defaults to 0.5 mm for sdf pairs
    and ``HYDRO_GAP_M`` for hydroelastic ones (Newton has no speculative hydroelastic contacts, and a gap band that
    spans the clearance doubles the iso voxels; ``alloy.contact.pipeline.hydro_band_spans_clearance``)."""
    import warp as wp  # noqa: PLC0415

    from alloy._vendor import newton  # noqa: PLC0415
    from alloy.contact.sdf_cache import build_feature_sdf  # noqa: PLC0415
    from alloy.contact.shadow import set_flat_solimp  # noqa: PLC0415
    from alloy.contact.types import PrecisionPairSpec  # noqa: PLC0415

    g = geometry or PegHoleSpec()
    gap = gap if gap is not None else (HYDRO_GAP_M if mode == "hydroelastic" else 5e-4)
    peg_mesh = to_newton_mesh(*cylinder_tris(0.5 * g.peg_d, 0.5 * g.peg_len, g.segments))
    hole_mesh = to_newton_mesh(*blind_hole_tris(g), compute_inertia=False)
    spec = PrecisionPairSpec(0, 1, mode=mode, voxel_size=voxel, kh=kh, pair_stiffness=pair_stiffness,
                             clearance=g.bore_d - g.peg_d, ccd=ccd, gap=gap, narrow_band=tuple(narrow_band))
    infos = [build_feature_sdf(m, spec, device=device, cache_dir=cache_dir) for m in (hole_mesh, peg_mesh)]
    cfg = newton.ModelBuilder.ShapeConfig(density=density, mu=mu, gap=spec.gap, margin=0.0,
                                          is_hydroelastic=(mode == "hydroelastic"), kh=kh)
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    hole = t.add_shape_mesh(-1, mesh=hole_mesh, cfg=cfg, label="hole")
    q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.radians(tilt_deg))
    body = t.add_body(xform=wp.transform((lateral, 0.0, g.bottom_z + start_gap + 0.5 * g.peg_len), q), label="peg")
    peg = t.add_shape_mesh(body, mesh=peg_mesh, cfg=cfg, label="peg")
    t.add_shape_collision_filter_pair(hole, peg)
    for shape in (hole, peg):
        set_flat_solimp(t, shape, spec.mj_imp)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, worlds)
    model = b.finalize(device=device)
    nb = t.body_count
    mass = float(model.body_mass.numpy()[0])
    return PegHoleScene(model, PrecisionPairSpec(hole, peg, **{k: getattr(spec, k) for k in (
        "mode", "voxel_size", "kh", "pair_stiffness", "clearance", "ccd", "gap", "narrow_band")}), g,
        [w * nb for w in range(worlds)],
        mass, infos)
