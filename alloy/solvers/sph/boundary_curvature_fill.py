# SPDX-License-Identifier: Apache-2.0
"""Conforming liquid fill for round bores (SCR-WP9-10, coordinator rulings 2026-11-04 20:19Z / 20:53Z).

Why the fill matters as much as the wall volume.
- A square lattice clipped to a circle leaves gaps at a round wall.
- Rings at (k + 1/2) spacings (``ring_layer`` in SC10's pipette_geom) hold too few particles at small R/h: 28 per
  layer at R/h 1.67, against the bore area's 35. The column then starts at 0.8 of rest density and sags into the
  gaps, which is WP9's measured near-wall ratio of 0.81.
- Rings at R - r - k spacing hold the right count. Their inner rings, though, sit up to 2.5 % above rest density at
  R/h 1.67. The unilateral density solve removes that error in one step, with a velocity of about
  error x column / dt, so at the C18 step of 1.95e-5 s a 5 mm column leaves the bore at 2.3 m/s.

``round_bore_layer`` therefore solves the ring radii.
- Ring k holds round(2 pi r_k / spacing) points, starting from rings at R - r - k spacing.
- A centre point is added when it brings the count closer to pi R^2 / spacing^2.
- When those counts cannot reach rest density (the curved wall's larger volume leaves room for one point fewer at
  R/h 1.67), each ring's count +-1 is tried and the closest result is kept.
- The radii are solved (damped Newton on the mean density of each ring) so that every ring sits at rest density,
  counting the layers above and below (interlocked: every other layer turned by half a step) and the wall's boundary
  volume.
- The wall volume is the solver's: the fitted flat volume with the taper, times ``curved_volume``'s curvature ratio
  for a concave cylinder of radius R (boundary_curvature.py). It is evaluated by the same Warp functions.

Result, cubic kernel at h = 2 spacings: every ring at R/h 1.67 and 3.5 sits within 1e-4 of rest density
(tests/solvers/sph/test_boundary_curvature.py).

``round_bore_points`` stacks the layers at spacing pitch from z0 + spacing/2. For a polygon mesh bore (a lathe ring
with ``segments`` sides) it fills the polygon's equal-area radius.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from .boundary_curvature import curved_volume
from .kernels import KERNEL_IDS, boundary_fit, clamp_principal_curvature, halfspace_volume, kernel_w_np, lattice_sum

log = logging.getLogger("alloy.sph")


@dataclass(frozen=True)
class BoreModel:
    """What the fill must match: the solver's spacing, support, kernel and curvature order (DFSPHConfig)."""

    spacing: float
    h: float
    kernel: str = "cubic"
    curvature: int = 1

    @staticmethod
    def of(cfg) -> BoreModel:
        return BoreModel(float(cfg.spacing), float(cfg.h), str(cfg.kernel), int(cfg.boundary_curvature_order))


@wp.kernel(enable_backward=False)
def _bore_psi_kernel(d: wp.array[wp.float32], H: float, offset: float, amplitude: float, h: float, kind: int,
                     taper0: float, taper1: float, curvature: int, out: wp.array[wp.float32]):
    """boundary._fill_one's volume for one mesh slot at distance d[i] from a wall of mean curvature H."""
    i = wp.tid()
    phi = d[i]
    v = halfspace_volume(phi + offset, h, 0.0, kind) * amplitude
    if curvature > 0 and phi < taper1:
        v = curved_volume(phi, offset, amplitude, h, H, kind)
    if phi > taper0:
        t = wp.min((phi - taper0) / (taper1 - taper0), 1.0)
        v = wp.vec2(v[0] * (1.0 - t * t * (3.0 - 2.0 * t)), 0.0)
    out[i] = v[0]


@wp.kernel(enable_backward=False)
def _clamped_h_kernel(k: float, h: float, out: wp.array[wp.float32]):
    out[0] = clamp_principal_curvature(k, h)


def wall_psi(model: BoreModel, radius: float, d: np.ndarray) -> np.ndarray:
    """The solver's boundary volume of a concave round wall of ``radius`` at wall distances ``d`` [m] (one slot)."""
    kind = KERNEL_IDS[model.kernel]
    offset, amplitude = boundary_fit(model.spacing, model.h, model.kernel)
    hk = wp.zeros(1, dtype=wp.float32, device="cpu")
    wp.launch(_clamped_h_kernel, dim=1, inputs=[1.0 / radius, model.h, hk], device="cpu")
    dd = wp.array(np.asarray(d, np.float32), dtype=wp.float32, device="cpu")
    out = wp.zeros(len(dd), dtype=wp.float32, device="cpu")
    wp.launch(_bore_psi_kernel, dim=len(dd),
              inputs=[dd, float(hk.numpy()[0]), float(offset), float(amplitude), model.h, kind,
                      0.5 * model.spacing, 1.5 * model.spacing, model.curvature, out], device="cpu")
    return out.numpy().astype(np.float64)


def _ring_counts(radius: float, spacing: float) -> tuple[list[float], list[int], bool]:
    """Starting radii (R - r - k spacing), points per ring and whether a centre point is added."""
    radii, counts = [], []
    rk = radius - 0.5 * spacing
    while rk >= 0.45 * spacing:
        radii.append(rk)
        counts.append(max(1, int(round(2.0 * math.pi * rk / spacing))))
        rk -= spacing
    target = math.pi * radius * radius / (spacing * spacing)
    centre = abs(sum(counts) + 1 - target) < abs(sum(counts) - target)
    return radii, counts, centre


def _layer(radii, counts, centre: bool, phase: float) -> np.ndarray:
    pts = []
    for rk, n in zip(radii, counts):
        a = 2.0 * math.pi * (np.arange(n) + phase) / n
        pts.append(np.stack([rk * np.cos(a), rk * np.sin(a)], axis=1))
    if centre:
        pts.append(np.zeros((1, 2)))
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def ring_densities(model: BoreModel, radius: float, radii, counts, centre: bool) -> np.ndarray:
    """Mean density ratio of each ring of the middle layer (fluid neighbours + wall volume), interlocked layers."""
    s, h = model.spacing, model.h
    S = lattice_sum(s, h, model.kernel)
    m = int(math.ceil(h / s))
    layers = [np.column_stack([_layer(radii, counts, centre, 0.5 * (j % 2)), np.full(sum(counts) + centre, j * s)])
              for j in range(-m, m + 1)]
    allp = np.concatenate(layers)
    mid = layers[m]
    out = np.zeros(len(radii))
    start = 0
    psi = wall_psi(model, radius, radius - np.asarray(radii, np.float64))
    for k, n in enumerate(counts):
        pk = mid[start:start + n]
        dist = np.linalg.norm(allp[None, :, :] - pk[:, None, :], axis=2)
        out[k] = float(kernel_w_np(dist, h, model.kernel).sum(axis=1).mean()) / S + psi[k]
        start += n
    return out


def solve_ring_radii(model: BoreModel, radius: float, tol: float = 1.0e-5, iters: int = 40):
    """(radii, counts, centre, densities): ring radii with every ring at rest density. The starting counts first;
    when they cannot reach it (too many or too few points for the bore's free volume), each ring's count +-1, and
    the closest result is kept."""
    radii, counts, centre = _ring_counts(radius, model.spacing)
    best = _solve_radii(model, radius, radii, counts, centre, tol, iters)
    if _err(best) < 10.0 * tol:
        return best
    for k in range(len(counts)):
        for dn in (-1, 1):
            c = list(counts)
            c[k] += dn
            if c[k] < 1:
                continue
            trial = _solve_radii(model, radius, radii, c, centre, tol, iters)
            if _err(trial) < _err(best):
                best = trial
    if _err(best) >= 10.0 * tol:
        log.warning("round bore fill R = %.4g m: ring densities off rest by up to %.2g", radius, _err(best))
    return best


def _err(sol) -> float:
    return float(np.max(np.abs(sol[3] - 1.0))) if len(sol[3]) else 0.0


def _solve_radii(model: BoreModel, radius: float, radii, counts, centre: bool, tol: float, iters: int):
    """Damped Newton on the ring radii (counts fixed) for every ring at rest density."""
    x = np.asarray(radii, np.float64)
    if len(x) == 0:
        return x, counts, centre, np.zeros(0)
    f = ring_densities(model, radius, x, counts, centre) - 1.0
    eps = 1.0e-4 * model.spacing
    for _ in range(iters):
        if np.max(np.abs(f)) < tol:
            break
        J = np.empty((len(x), len(x)))
        for k in range(len(x)):
            xp = x.copy()
            xp[k] += eps
            J[:, k] = (ring_densities(model, radius, xp, counts, centre) - 1.0 - f) / eps
        step = np.linalg.lstsq(J, -f, rcond=None)[0]
        step = np.clip(step, -0.1 * model.spacing, 0.1 * model.spacing)
        x_new = np.clip(x + step, 0.3 * model.spacing, radius - 0.2 * model.spacing)
        f_new = ring_densities(model, radius, x_new, counts, centre) - 1.0
        if np.max(np.abs(f_new)) > np.max(np.abs(f)):  # backtrack once
            x_new = np.clip(x + 0.5 * step, 0.3 * model.spacing, radius - 0.2 * model.spacing)
            f_new = ring_densities(model, radius, x_new, counts, centre) - 1.0
        x, f = x_new, f_new
    return x, list(counts), centre, f + 1.0


def round_bore_layer(model: BoreModel, radius: float, phase: float = 0.0) -> np.ndarray:
    """(x, y) of one layer of a round bore of ``radius`` [m] centred on the axis, every ring at rest density."""
    if not (radius > 0.0 and model.spacing > 0.0):
        raise ValueError(f"round_bore_layer: need radius > 0 and spacing > 0, got {radius}, {model.spacing}")
    radii, counts, centre, _ = solve_ring_radii(model, radius)
    return _layer(radii, counts, centre, phase)


def polygon_radius(radius: float, segments: int | None) -> float:
    """Radius of the circle with the area of a regular ``segments``-gon inscribed in ``radius`` (None: radius).
    A lathe mesh's wall lies inside its nominal circle, by 1.3 um (0.017 r) at R = 1.05 mm and 64 segments, which
    is +0.7 % density at the wall for a fill made for the circle."""
    if segments is None:
        return float(radius)
    if segments < 3:
        raise ValueError(f"polygon_radius: need segments >= 3, got {segments}")
    return float(radius * math.sqrt(segments * math.sin(2.0 * math.pi / segments) / (2.0 * math.pi)))


def round_bore_points(model: BoreModel, radius: float, z0: float, z1: float,
                      center: tuple[float, float] = (0.0, 0.0), segments: int | None = None) -> np.ndarray:
    """(N, 3) points filling a round bore along z between z0 and z1 [m]. Layers sit at z0 + (k + 1/2) spacing while
    they stay half a spacing below z1, and every other layer is turned by half a step. ``segments``: the bore is a
    regular polygon mesh with that many sides (a lathe ring), filled for its equal-area radius."""
    if z1 <= z0:
        raise ValueError(f"round_bore_points: need z1 > z0, got {z0}, {z1}")
    s = model.spacing
    radius = polygon_radius(radius, segments)
    radii, counts, centre, dens = solve_ring_radii(model, radius)
    log.info("round bore fill R = %.4g m: %d rings, %d per layer, ring densities %.5f..%.5f", radius, len(radii),
             sum(counts) + centre, float(dens.min()) if len(dens) else 1.0, float(dens.max()) if len(dens) else 1.0)
    zs = np.arange(z0 + 0.5 * s, z1 - 0.5 * s + 1.0e-9 * s, s)
    out = []
    for i, z in enumerate(zs):
        xy = _layer(radii, counts, centre, 0.5 * (i % 2))
        out.append(np.column_stack([xy[:, 0] + center[0], xy[:, 1] + center[1], np.full(len(xy), z)]))
    return np.concatenate(out) if out else np.zeros((0, 3))


__all__ = ["BoreModel", "polygon_radius", "ring_densities", "round_bore_layer", "round_bore_points", "solve_ring_radii", "wall_psi"]
