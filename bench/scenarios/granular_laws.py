# SPDX-License-Identifier: Apache-2.0
"""Granular family laws and host-side measures (pure numpy, no Newton or Warp), shared by bench/scenarios/granular.py
and the offline Genesis reference (bench/scenarios/_granular_genesis.py, which runs in a venv without Newton)."""
from __future__ import annotations

import math

import numpy as np

G = 9.81

# Martin & Moyce 1952 (Phil. Trans. R. Soc. A 244, DOI 10.1098/rsta.1952.0006), n^2 = 2, a = 2.25 in: surge front
# Z = x/a at T = t sqrt(2g/a). UNVERIFIED transcription: the table as commonly reproduced in SPH/MPS validation papers
# (Z at 1/2-inch gauge steps); check against the 1952 figure before quoting the NRMSE as final.
MM_T = np.array([0.41, 0.84, 1.19, 1.43, 1.63, 1.83, 1.98, 2.20, 2.32, 2.51, 2.65, 2.83, 2.98, 3.11])
MM_Z = np.array([1.11, 1.22, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56, 2.78, 3.00, 3.22, 3.44, 3.67, 3.89])


# ---- laws (pure functions) ---------------------------------------------------------------------------------
def lube_runout(a: float) -> float:
    """Planar column collapse runout (L_inf - L0)/L0 (Lube et al., Phys. Rev. E 72, 041301, 2005)."""
    return 1.2 * a if a < 2.3 else 1.9 * a ** (2.0 / 3.0)


def ritter_front(t: np.ndarray, a: float, h: float) -> np.ndarray:
    """Ideal-fluid dam-break front x_f = a + 2 t sqrt(g h) (Ritter 1892); no real front may run faster."""
    return a + 2.0 * np.asarray(t) * math.sqrt(G * h)


def huppert_radius(t: np.ndarray, volume: float, nu: float) -> np.ndarray:
    """Axisymmetric constant-volume viscous gravity current r = 0.894 (g V^3 t / (3 nu))^(1/8) (Huppert 1982)."""
    return 0.894 * (G * volume**3 * np.asarray(t) / (3.0 * nu)) ** 0.125


def rankine_force(rho: float, depth: float, width: float, phi_deg: float) -> float:
    """Rankine passive force on a smooth vertical wall, F = 1/2 gamma d^2 w tan^2(45 + phi/2)."""
    kp = math.tan(math.radians(45.0 + 0.5 * phi_deg)) ** 2
    return 0.5 * rho * G * depth**2 * width * kp


def fit_power(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares y = c x^n in log space; returns (n, c)."""
    n, logc = np.polyfit(np.log(x), np.log(y), 1)
    return float(n), float(math.exp(logc))


def fit_power_offset(t: np.ndarray, r: np.ndarray, frac_max: float = 0.9) -> tuple[float, float, float]:
    """Fit r = c (t - t0)^n with a virtual time origin t0 (grid search); returns (n, c, t0)."""
    best = (math.inf, 0.0, 0.0, 0.0)
    for t0 in np.linspace(-t[0], frac_max * t[0], 61):
        n, c = fit_power(t - t0, r)
        res = float(np.sum((np.log(r) - np.log(c) - n * np.log(t - t0)) ** 2))
        if res < best[0]:
            best = (res, n, c, float(t0))
    return best[1], best[2], best[3]


def nrmse(sim: np.ndarray, ref: np.ndarray) -> float:
    """RMS error normalised by the reference range."""
    return float(np.sqrt(np.mean((sim - ref) ** 2)) / (ref.max() - ref.min()))


def heap_profile(r: np.ndarray, z: np.ndarray, bin_w: float) -> tuple[np.ndarray, np.ndarray]:
    """Surface profile of an axisymmetric heap: (bin centre radius, 95th percentile of particle height) per radial
    bin holding at least 5 particles."""
    edges = np.arange(0.0, r.max() + bin_w, bin_w)
    idx = np.digitize(r, edges) - 1
    rc, zs = [], []
    for k in range(len(edges) - 1):
        sel = z[idx == k]
        if sel.size >= 5:
            rc.append(0.5 * (edges[k] + edges[k + 1]))
            zs.append(np.percentile(sel, 95))
    return np.array(rc), np.array(zs)


def heap_angle_deg(r: np.ndarray, z: np.ndarray, bin_w: float) -> tuple[float, int]:
    """Repose angle of an axisymmetric heap: line fit of the surface profile z_s(r) (heap_profile) on the flank,
    i.e. the bins beyond the peak whose surface lies between 20 % and 80 % of the peak height (excludes the
    rounded top and the toe). Returns (angle [deg], number of bins fitted); NaN when fewer than 3 bins qualify.
    """
    rc, zs = heap_profile(r, z, bin_w)
    if rc.size < 3:
        return math.nan, 0
    peak = int(np.argmax(zs))
    flank = (np.arange(rc.size) >= peak) & (zs > 0.2 * zs[peak]) & (zs < 0.8 * zs[peak])
    if flank.sum() < 3:
        return math.nan, int(flank.sum())
    slope = np.polyfit(rc[flank], zs[flank], 1)[0]
    return float(math.degrees(math.atan(-slope))), int(flank.sum())


def deposit_front(x: np.ndarray, max_gap: float) -> tuple[float, int]:
    """Front of a connected deposit of discrete grains: walking the sorted x from the back wall, the last grain
    before a gap wider than max_gap. Returns (front x, number of detached grains beyond it)."""
    xs = np.sort(np.asarray(x, float))
    cut = np.nonzero(np.diff(xs) > max_gap)[0]
    k = int(cut[0]) if cut.size else len(xs) - 1
    return float(xs[k]), len(xs) - 1 - k


def occupancy_volume(x: np.ndarray, cell: float, n_full: float) -> float:
    """Occupied volume: sum over grid cells of min(count / n_full, 1) * cell^3 (cells aligned to the local origin).

    With cell = voxel and n_full = ppc^3 this is the solver's own volume-fraction picture: a compacted cell counts
    as one full cell, so bunching (the implicit-MPM volume loss) shows as lost volume. Exact at t = 0 for the
    cell-centred lattice used here (jitter stays inside the cell). Thin layers (< 2 cells) read low.
    """
    idx = np.floor(x / cell).astype(np.int64)
    _, counts = np.unique(idx, axis=0, return_counts=True)
    return float(np.minimum(counts / n_full, 1.0).sum() * cell**3)


def surface_volume(x: np.ndarray, col: float, spacing: float, mirror_xy: bool = False,
                   bias: float = 0.0) -> tuple[float, float]:
    """Volume under the free surface of a layer resting on z = 0; returns (volume, surface excess).

    Sum over xy columns of width col: interior columns (all 8 neighbours occupied) count
    (max z + spacing / 2 - bias) col^2; edge columns count their particle volume n spacing^3 (their covered area is
    unknown). Compaction lowers the surface, so it shows as lost volume, while particle disorder inside the layer
    does not (unlike occupancy_volume). The max height of a column reads high (jitter of its top particles): the
    returned surface excess is the mean of (max z + spacing / 2 - n spacing^3 / col^2) over interior columns, zero
    for a perfect lattice; measure it at t = 0 and pass it back as bias. mirror_xy reflects the points across
    x = 0 and y = 0 first (quarter-symmetric scenes) and returns the quarter volume.
    """
    if mirror_xy:
        x = np.concatenate([x * np.array(m) for m in ((1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1))])
    ij = np.floor(x[:, :2] / col).astype(np.int64)
    keys, inv, counts = np.unique(ij, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    top = np.full(len(keys), -np.inf)
    np.maximum.at(top, inv, x[:, 2])
    occupied = {(int(i), int(j)) for i, j in keys}
    interior = np.array([all((int(i) + di, int(j) + dj) in occupied for di in (-1, 0, 1) for dj in (-1, 0, 1))
                         for i, j in keys])
    height = top + 0.5 * spacing
    excess = float(np.mean(height[interior] - counts[interior] * spacing**3 / col**2)) if interior.any() else 0.0
    vol = np.where(interior, (height - bias) * col**2, counts * spacing**3)
    return float(vol.sum()) / (4.0 if mirror_xy else 1.0), excess
