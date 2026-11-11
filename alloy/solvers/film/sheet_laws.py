# SPDX-License-Identifier: Apache-2.0
"""Laws of J2 metal sheets on the film route (SCR-WP7-15 P3; the hinge: sheet_j2.py, the model: plastic/j2.py).

- **S1, pure bending.** A strip of a sourced metal card is laid on arcs of rising curvature κ (all particles pinned,
  an isometric bend, so the membrane carries nothing) and stepped once per κ. Each bending hinge's moment per unit
  width, M = ke (θ - θ̄), follows the plane-strain section's closed form (``closed_forms.rect_moment`` with E', σ_y',
  H'): first yield at σ_y' t²/6, toward σ_y' t²/4. The curvature its rest angle keeps, (θ̄ - θ0)/l_h, is the
  elastic spring-back κ0 - M(κ0)/D (Gardiner's R0/Rf = 4a³ - 3a + 1 for a card without hardening).
- **S1 off the grid's axes** (``s1_off_axis``, a measured row, WP11's review). The same isometric bend with its axis at
  an angle to the grid: then the diagonal and x-edge hinges bend too, each on its own normal curvature, and each yields
  on its own. The network's moment about the bend axis is its virtual work, Σ ke |e| (θ - θ̄) ∂θ/∂κ over the area, and
  the curvature it keeps is the least-squares cylinder through the kept rest angles. A plate's M(κ) does not depend on
  the angle; the network's does, which is the main risk of one section per hinge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy.solvers.plastic import closed_forms as CF
from alloy.solvers.plastic.j2 import plane_strain

from .emit import add_film_mesh
from .materials import FilmSpec
from .rigs import FilmRig, grid_mesh, pin


def metal_spec(material: str, thickness: float) -> FilmSpec:
    """A film spec from a library metal card with ``plastic.model: j2`` (E, ν, ρ, σ_y0, H as sourced)."""
    from alloy.cards.library import CardLibrary

    m = CardLibrary().material(material)
    if m.plastic is None or m.plastic.model != "j2":
        raise ValueError(f"{material}: not a j2 card")
    h = m.plastic.hardening_modulus
    sy = float(m.plastic.yield_stress.value)
    return FilmSpec(youngs=float(m.elastic.youngs.value), poisson=float(m.elastic.poisson.value),
                    density=float(m.density.value), thickness=thickness, yield_stress=sy, j2_yield=sy,
                    j2_hardening=float(h.value) if h is not None else 0.0)


def section(spec: FilmSpec) -> tuple[float, float, float, float]:
    """(E', σ_y', H', D) of the spec's plane-strain section."""
    Ep, syp, hp = plane_strain(spec.youngs, spec.poisson, spec.j2_yield, spec.j2_hardening)
    return Ep, syp, hp, Ep * spec.thickness**3 / 12.0


def arc(v: np.ndarray, kappa: float) -> np.ndarray:
    """The flat strip ``v`` (x along its length) wrapped isometrically onto a cylinder of curvature κ about y."""
    if kappa == 0.0:
        return v.copy()
    r, s = 1.0 / kappa, v[:, 0]
    return np.stack([r * np.sin(s / r), v[:, 1], r * (1.0 - np.cos(s / r))], 1)


@dataclass
class BendRow:
    kappa: float           # 1/m, imposed
    moment: float          # N·m/m, mean over the bending hinges
    moment_cf: float       # closed form
    spread: float          # relative spread of the hinges' moments
    kept: float            # 1/m, the curvature the rest angles keep (mean)
    kept_cf: float         # κ - M_cf/D


def s1_pure_bending(material: str = "aluminium_6061_t6", thickness: float = 5.0e-4, k_max: float = 20.0,
                    steps: int = 40, nx: int = 24, ny: int = 4, dx: float = 2.0e-3,
                    device: str = "cpu") -> list[BendRow]:
    """S1 (module doc): ``steps`` curvatures up to ``k_max`` times the first-yield curvature, monotonic."""
    spec = metal_spec(material, thickness)
    Ep, syp, hp, D = section(spec)
    k_y = syp / (Ep * 0.5 * thickness)
    v, tris = grid_mesh(nx, ny, dx)

    def build(b):
        p0, _ = add_film_mesh(b, v, tris, spec)
        pin(b, range(p0, p0 + len(v)))
        return {"p0": [p0]}

    rig = FilmRig(build=build, device=device, gravity=0.0)
    m = rig.model
    ei = m.edge_indices.numpy()
    bend = _bending_hinges(v, ei, nx, dx)
    ke = m.edge_bending_properties.numpy()[bend, 0]
    lh = m.film.hinge_length.numpy()[bend]
    rest0 = m.edge_rest_angle.numpy()[bend].copy()
    rows = []
    for k in np.linspace(0.0, k_max * k_y, steps + 1)[1:]:
        x = wp.array(arc(v, k).astype(np.float32), dtype=wp.vec3, device=device)
        wp.copy(rig.s0.particle_q, x)
        wp.copy(rig.s1.particle_q, x)
        rig.s0.particle_qd.zero_()
        rig.step(1)
        q = rig.q()
        th = np.array([_angle(q, ei[e]) for e in bend])
        rest = m.edge_rest_angle.numpy()[bend]
        M = np.abs(ke * (th - rest))
        mcf = CF.rect_moment(float(k), thickness, Ep, syp, hp)
        kept = np.abs(rest - rest0) / lh
        rows.append(BendRow(kappa=float(k), moment=float(M.mean()), moment_cf=mcf,
                            spread=float((M.max() - M.min()) / max(M.mean(), 1e-30)), kept=float(kept.mean()),
                            kept_cf=CF.springback_curvature(float(k), mcf, D)))
    return rows


def gardiner_kept(material: str, thickness: float, kappa0: float) -> float:
    """The curvature a perfectly plastic plane-strain strip keeps after release from κ0 >= its first yield
    (Gardiner: R0/Rf = 4a³ - 3a + 1, so κ_f = κ0 R0/Rf)."""
    spec = metal_spec(material, thickness)
    Ep, syp, _, _ = section(spec)
    return kappa0 * CF.gardiner_ratio(1.0 / kappa0, thickness, Ep, syp)


def _bending_hinges(v: np.ndarray, ei: np.ndarray, nx: int, dx: float) -> np.ndarray:
    """Interior hinges whose edge runs across the strip (along y), two cells clear of its ends."""
    out = []
    for e, (o0, o1, a, c) in enumerate(ei):
        if o0 < 0 or o1 < 0 or abs(v[a, 0] - v[c, 0]) > 1e-12:
            continue
        if 2 * dx - 1e-12 <= v[a, 0] <= (nx - 2) * dx + 1e-12:
            out.append(e)
    return np.asarray(out, np.int64)


def _angle(q: np.ndarray, row) -> float:
    o0, o1, a, c = (int(i) for i in row)
    n1 = np.cross(q[a] - q[o0], q[c] - q[o0])
    n2 = np.cross(q[c] - q[o1], q[a] - q[o1])
    e = q[c] - q[a]
    n1, n2 = n1 / np.linalg.norm(n1), n2 / np.linalg.norm(n2)
    return math.atan2(float(np.dot(np.cross(n1, n2), e / np.linalg.norm(e))), float(np.dot(n1, n2)))


def hinge_angles(q: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Dihedral angles of hinge rows (o0, o1, a, c) at positions ``q`` (the sign convention of the VBD hinge;
    boundary rows, o < 0, come out as garbage: select interior rows)."""
    o0, o1, a, c = (rows[:, i] for i in range(4))
    n1 = np.cross(q[a] - q[o0], q[c] - q[o0])
    n2 = np.cross(q[c] - q[o1], q[a] - q[o1])
    e = q[c] - q[a]
    n1 /= np.maximum(np.linalg.norm(n1, axis=1, keepdims=True), 1e-30)
    n2 /= np.maximum(np.linalg.norm(n2, axis=1, keepdims=True), 1e-30)
    e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-30)
    return np.arctan2(np.sum(np.cross(n1, n2) * e, axis=1), np.sum(n1 * n2, axis=1))


@dataclass
class OffAxisRow:
    angle: float           # deg, the bend direction from the grid's x axis
    kappa: float           # 1/m, imposed
    moment: float          # N·m/m, the network's moment about the bend axis (virtual work over the window)
    moment_cf: float       # the plane-strain plate's
    kept: float            # 1/m, the least-squares cylinder through the kept rest angles
    kept_cf: float         # κ - M_cf/D
    stiffness: float       # the network's elastic bending stiffness over the plate's D


def s1_off_axis(angle: float, material: str = "aluminium_6061_t6", thickness: float = 5.0e-4, k_max: float = 8.0,
                steps: int = 16, n: int = 16, dx: float = 2.0e-3, device: str = "cpu") -> list[OffAxisRow]:
    """S1 with the bend direction at ``angle`` degrees to the grid's x axis on an n x n patch (module doc); sums over a
    window two cells clear of the edges, each hinge weighted by the share of its two wings inside it."""
    spec = metal_spec(material, thickness)
    Ep, syp, hp, D = section(spec)
    k_y = syp / (Ep * 0.5 * thickness)
    v, tris = grid_mesh(n, n, dx)
    v = v - np.array([0.5 * n * dx, 0.5 * n * dx, 0.0])
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])               # world -> bend frame (x' along it)

    def build(b):
        p0, _ = add_film_mesh(b, v, tris, spec)
        pin(b, range(p0, p0 + len(v)))
        return {"p0": [p0]}

    rig = FilmRig(build=build, device=device, gravity=0.0)
    m = rig.model
    ei = m.edge_indices.numpy().astype(np.int64)
    rows = np.flatnonzero((ei[:, 0] >= 0) & (ei[:, 1] >= 0))
    ei = ei[rows]
    half = (0.5 * n - 2.0) * dx
    inside = lambda o: np.all(np.abs((v[ei[:, 2], :2] + v[ei[:, 3], :2] + v[o, :2]) / 3.0) < half, axis=1)  # noqa: E731
    f = 0.5 * (inside(ei[:, 0]).astype(float) + inside(ei[:, 1]).astype(float))
    area = (2.0 * half) ** 2
    kL = m.edge_bending_properties.numpy()[rows, 0] * np.linalg.norm(v[ei[:, 3]] - v[ei[:, 2]], axis=1) * f
    rest0 = m.edge_rest_angle.numpy()[rows].copy()
    out, g = [], None
    for k in np.linspace(0.0, k_max * k_y, steps + 1)[1:]:
        x = (arc(v @ rot.T, k) @ rot).astype(np.float32)
        wp.copy(rig.s0.particle_q, wp.array(x, dtype=wp.vec3, device=device))
        wp.copy(rig.s1.particle_q, wp.array(x, dtype=wp.vec3, device=device))
        rig.s0.particle_qd.zero_()
        rig.step(1)
        th = hinge_angles(rig.q().astype(np.float64), ei)
        g = th / k if g is None else g                                          # dθ/dκ from the first (elastic) bend
        rest = m.edge_rest_angle.numpy()[rows]
        mcf = CF.rect_moment(float(k), thickness, Ep, syp, hp)
        out.append(OffAxisRow(angle=angle, kappa=float(k), moment=float(np.sum(kL * (th - rest) * g) / area),
                              moment_cf=mcf, kept=float(np.sum(kL * g * (rest - rest0)) / np.sum(kL * g * g)),
                              kept_cf=CF.springback_curvature(float(k), mcf, D),
                              stiffness=float(np.sum(kL * g * g) / (D * area))))
    return out
