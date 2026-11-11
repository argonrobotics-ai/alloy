# SPDX-License-Identifier: Apache-2.0
"""Heat v2 surface areas (WP13 review of step 4): the exposed area a particle convects through.

Convection acts on each particle's exposed area A_i = s_i² · min(6, F_i, C_i), with s_i = (m_i/ρ)^(1/3) its spacing,
s_i² its footprint, and two estimates of its open faces, each calibrated so a flat lattice face reads one footprint
per face column (at setup, for the run's support / spacing ratio):
- F_i = max(0, −log2(f_i) − ``FLOOR``) / κ_F, with f_i = tr(M_i)/3 the filled fraction of its neighbourhood (M_i the
  second-moment tensor that conduction's free-surface correction computes, the identity in the bulk). Exact on ordered
  bodies and rotation-invariant, but it reads density gaps as surface;
- C_i = −log2(c_i) / κ_C, with c_i the closed fraction of its view: of 62 directions with the cube's symmetry (the
  48 signed permutations of (1, 0.3, 0.6), the 6 face and the 8 corner directions), the share whose 35° cone holds a
  neighbour within the support. One neighbour per cone closes it, so it ignores density gaps, but its wide cones blur
  edges and corners;
- 6 footprints, a lone cube's area, caps a particle with no neighbours.
Each estimate over-reads where the other does not, so the smaller one is taken. −log2 counts open faces: a half
space halves f and c, a quarter space quarters them.

Measured on a numpy copy of the kernel (support 2.2 spacings; 12³ bodies, total as a share of the box area):
lattice 1.01; rotated 23, 37, 51° about x, y, z 1.03; jittered ±0.2 / ±0.35 spacings 1.00 / 1.01; a uniformly random
cloud 1.04. Thin 30 × 30 puddles two and three layers deep, jittered ±0.35 or uniformly random (a spread melt's
shape): 0.98-1.01. Lattice faces 1.00 per column on every axis (the set's cubic symmetry), edge 2.08, corner 3.35
footprints (sharp: 1, 2, 3); a one-layer sheet over-reads its area by 18 % (1.18); a lone particle 6. The set sits in
a plateau: (1, a, b) with a 0.27-0.33 and b 0.55-0.62 read alike; at b ≥ 0.65 the lattice calibration catches a cone
edge and disordered bodies read 0.6. History: the earlier 32 Fibonacci directions and their opposites read y faces
0.95 and thin disordered puddles 0.90-0.92 (WP13 re-review); F alone read 1.57 of the random cloud and 3x a melted
MPM chocolate puddle's area; C alone read corners 3.77.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

FLOOR = 0.03  # F's noise floor, in log2 units of the filled fraction
ETA = 0.1  # the kernels' softening: d² + (ETA·hs)²
COUNT = 62  # view directions: the cube's 48 signed permutations of GENERIC, 6 faces, 8 corners (≤ 64: two bit words)
GENERIC = (1.0, 0.3, 0.6)
CONE_DEG = 35.0  # a direction is closed by a neighbour within this angle of it
CAP = 6.0  # footprints: a lone cube's six faces


def _orbit(v: tuple[float, float, float]) -> np.ndarray:
    """Unit vectors of every signed permutation of v (the cube's symmetry group acting on it), without repeats."""
    out = {tuple(round(a * b, 12) for a, b in zip(p, sgn)) for p in itertools.permutations(v)
           for sgn in itertools.product((1.0, -1.0), repeat=3)}
    a = np.asarray(sorted(out), np.float64)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def directions() -> np.ndarray:
    """[COUNT, 3] unit view directions with the cube's symmetry, so faces on every axis (and opposite faces) read
    alike."""
    d = np.concatenate([_orbit(GENERIC), _orbit((1.0, 0.0, 0.0)), _orbit((1.0, 1.0, 1.0))])
    assert len(d) == COUNT <= 64, len(d)   # the kernel keeps the closed cones in two 32-bit words
    return d


def open_faces(points: np.ndarray, centre: np.ndarray, support: float) -> float:
    """−log2 of the closed fraction at ``centre`` over ``points`` (the kernel's measure, before κ and the cap)."""
    e = points - centre
    d = np.linalg.norm(e, axis=1)
    keep = (d > 0.0) & (d < support)
    if not keep.any():
        return math.inf
    e = e[keep] / d[keep, None]
    closed = (e @ directions().T >= math.cos(math.radians(CONE_DEG))).any(axis=0).mean()
    return math.inf if closed == 0.0 else -math.log2(closed)


def _dwdr(r: np.ndarray, hs: float) -> np.ndarray:
    """dW/dr of the 3-D cubic spline (``particle_kernels.cubic_dwdr``), smoothing length hs."""
    q, sig = r / hs, 1.0 / (math.pi * hs ** 4)
    return np.where(q < 1.0, sig * (-3.0 * q + 2.25 * q * q), np.where(q < 2.0, -sig * 0.75 * (2.0 - q) ** 2, 0.0))


def filled_faces(points: np.ndarray, centre: np.ndarray, support: float) -> float:
    """max(0, −log2(tr M / 3) − FLOOR) at ``centre`` over unit-volume ``points`` (``surface_moment``'s M)."""
    hs = 0.5 * support
    r = centre - points
    d = np.linalg.norm(r, axis=1)
    keep = (d > 0.0) & (d < support)
    r, d = r[keep], d[keep]
    f = -d * _dwdr(d, hs) / (d * d + (ETA * hs) ** 2)
    fill = float((f[:, None] * r * r).sum() / 3.0)
    return math.inf if fill <= 0.0 else max(0.0, -math.log2(fill) - FLOOR)


def fill_calibration(ratio: float) -> float:
    """κ_F: one z face column's summed ``filled_faces`` (F is rotation-invariant), halved over a two-faced slab."""
    pts, layers = _slab(ratio)
    return 0.5 * sum(filled_faces(pts, np.array([0.0, 0.0, float(k)]), ratio) for k in range(layers))


def _slab(ratio: float) -> tuple[np.ndarray, int]:
    n = int(math.ceil(ratio)) + 1
    layers = 2 * n + 2
    g = np.arange(-n, n + 1, dtype=float)
    x, y, z = np.meshgrid(g, g, np.arange(layers, dtype=float), indexing="ij")
    return np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1), layers


def face_calibration(ratio: float) -> float:
    """κ_C: a face column's summed ``open_faces``, averaged over x, y and z faces of a two-faced slab and halved (the
    64 cones are not exactly isotropic: faces differ by up to 8 %)."""
    pts, layers = _slab(ratio)
    total = 0.0
    for axis in range(3):
        perm = [(axis + 1) % 3, (axis + 2) % 3, axis]               # the slab's thin axis becomes ``axis``
        p = np.empty_like(pts)
        p[:, perm] = pts
        for k in range(layers):
            c = np.zeros(3)
            c[axis] = float(k)
            total += open_faces(p, c, ratio)
    return total / 6.0


__all__ = ["CAP", "CONE_DEG", "COUNT", "FLOOR", "GENERIC", "directions", "face_calibration", "fill_calibration",
           "filled_faces", "open_faces"]
