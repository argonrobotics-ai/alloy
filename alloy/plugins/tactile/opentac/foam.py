# SPDX-License-Identifier: Apache-2.0
"""OpenTac's foam contact model for a plain skin, ported as written: a rigid indenter pressed into foam.

Port of ``opentac/sim/foam.py`` (github.com/opentac/opentac, commit e302861), MIT licence, Copyright (c) 2026 The
OpenTac authors (``LICENSE-OPENTAC`` beside this file). Pressure is a Winkler foundation,
``p = k * max(delta - z(x, y), 0)``, with delta solved so the pressure integrates to the force; it is blurred by a
Gaussian of sigma 3.0 mm on a 0.5 mm grid and summed over a square around each taxel. sigma = 3.00 mm and
k = 0.00866 N/mm^3 were fitted once on one skin (robotiq_long) in the OpenTac paper. Outputs are not renormalised.

In Alloy the pressure comes from the solver's contacts instead (``model.foam_loads``, ``kernels.py``); this file is
the reference those are checked against, through ``winkler_points`` (the same pressure as point loads), and it
reproduces their fixture ``foam_reference.json`` (``tests/plugins/test_tactile_opentac.py``).
"""

from __future__ import annotations

import numpy as np

from alloy.plugins.tactile.opentac.model import FOAM_GRID_MM, FOAM_SIGMA_MM

K_N_PER_MM3 = 0.008659643233600654    # fitted on robotiq_long (paper)


def indenter_height(geom: dict, dx: np.ndarray, dy: np.ndarray, angle_deg: float = 0.0) -> np.ndarray:
    """Indenter surface height above the skin at offsets (mm): sphere, punch (flat disc) or bar; inf off it."""
    kind = geom["kind"]
    if kind == "sphere":
        r, r2 = geom["R"], dx ** 2 + dy ** 2
        return np.where(r2 < r ** 2, r - np.sqrt(np.clip(r ** 2 - r2, 0, None)), np.inf)
    if kind == "punch":
        return np.where(dx ** 2 + dy ** 2 < geom["R"] ** 2, 0.0, np.inf)
    if kind == "bar":
        th = np.deg2rad(angle_deg)
        s = dx * np.cos(th) + dy * np.sin(th)
        u = -dx * np.sin(th) + dy * np.cos(th)
        r, length = geom["R"], geom["L"]
        z = np.where(np.abs(u) < r, r - np.sqrt(np.clip(r ** 2 - u ** 2, 0, None)), np.inf)
        return np.where(np.abs(s) <= length / 2, z, np.inf)
    raise ValueError(f"indenter kind {kind!r}: sphere, punch or bar")


def gaussian_blur(field: np.ndarray, sigma_px: float) -> np.ndarray:
    """Their separable Gaussian: truncated at 3 sigma, normalised, ``np.convolve(mode="same")``."""
    if sigma_px <= 0:
        return field
    k = int(np.ceil(3 * sigma_px))
    x = np.arange(-k, k + 1)
    g = np.exp(-0.5 * (x / sigma_px) ** 2)
    g /= g.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 0, field)
    return np.apply_along_axis(lambda m: np.convolve(m, g, mode="same"), 1, out)


def _pressure(xy_mm, pitch_mm, geom, centre_mm, total_n, sigma_mm, angle_deg, k):
    """(gx, gy, p): the Winkler pressure field (N/mm^2) on their grid, delta solved to ``total_n``."""
    xy, pitch = np.asarray(xy_mm, dtype=float), float(pitch_mm)
    lo = xy.min(0) - 3 * pitch - 2 * sigma_mm
    hi = xy.max(0) + 3 * pitch + 2 * sigma_mm
    gx = np.arange(lo[0], hi[0] + FOAM_GRID_MM, FOAM_GRID_MM)
    gy = np.arange(lo[1], hi[1] + FOAM_GRID_MM, FOAM_GRID_MM)
    gxx, gyy = np.meshgrid(gx, gy)
    z = indenter_height(geom, gxx - centre_mm[0], gyy - centre_mm[1], angle_deg)
    cell = FOAM_GRID_MM ** 2

    def force_at(delta: float) -> float:
        return k * np.clip(delta - z, 0, None).sum() * cell

    a, b = 0.0, 0.05
    while force_at(b) < total_n and b < 200.0:
        b *= 2
    for _ in range(60):
        mid = 0.5 * (a + b)
        a, b = (mid, b) if force_at(mid) < total_n else (a, mid)
    return gx, gy, k * np.clip(0.5 * (a + b) - z, 0, None)


def taxel_forces(xy_mm, pitch_mm, geom, centre_mm, total_n, sigma_mm=FOAM_SIGMA_MM, angle_deg=0.0,
                 k_n_per_mm3=K_N_PER_MM3) -> np.ndarray:
    """N per taxel for a rigid indenter pressed into foam at a total force (their ``taxel_forces``)."""
    xy = np.asarray(xy_mm, dtype=float)
    gx, gy, p = _pressure(xy, pitch_mm, geom, centre_mm, total_n, sigma_mm, angle_deg, k_n_per_mm3)
    p = gaussian_blur(p, sigma_mm / FOAM_GRID_MM)
    if p.sum() <= 0:
        return np.zeros(len(xy))
    ix = np.clip(np.round((xy[:, 0] - gx[0]) / FOAM_GRID_MM).astype(int), 0, len(gx) - 1)
    iy = np.clip(np.round((xy[:, 1] - gy[0]) / FOAM_GRID_MM).astype(int), 0, len(gy) - 1)
    half = max(1, int(round(float(pitch_mm) / 2 / FOAM_GRID_MM)))
    cum = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    y0, y1 = np.clip(iy - half, 0, len(gy)), np.clip(iy + half + 1, 0, len(gy))
    x0, x1 = np.clip(ix - half, 0, len(gx)), np.clip(ix + half + 1, 0, len(gx))
    box = cum[y1, x1] - cum[y0, x1] - cum[y1, x0] + cum[y0, x0]
    return np.clip(box, 0, None) * FOAM_GRID_MM ** 2


def winkler_points(xy_mm, pitch_mm, geom, centre_mm, total_n, sigma_mm=FOAM_SIGMA_MM, angle_deg=0.0,
                   k_n_per_mm3=K_N_PER_MM3) -> tuple[np.ndarray, np.ndarray]:
    """The same indenter's pressure before the blur, as point loads: (points [N, 2] mm, forces [N] N)."""
    gx, gy, p = _pressure(xy_mm, pitch_mm, geom, centre_mm, total_n, sigma_mm, angle_deg, k_n_per_mm3)
    iy, ix = np.nonzero(p > 0)
    return np.column_stack([gx[ix], gy[iy]]), p[iy, ix] * FOAM_GRID_MM ** 2
