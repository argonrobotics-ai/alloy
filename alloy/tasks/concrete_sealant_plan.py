# SPDX-License-Identifier: Apache-2.0
"""tool-sealant's stroke plan: where each stroke ends so the nose lifts off with at most a budget of sealant.

A nose touching no sealant keeps only what its yield holds on it (GA's 0.52 ml on a vertical wall) and drops the rest
off the region. The bead's cut varies along the joint: about 0.05 ml per mm over the plain bead, nearly twice that over
a gun stop (a 3 mm bump), and little over the sagged stretch. So the strokes are planned from the start fields, as the
privileged carriage is: the nose's underside as the film sees it (WP15's roller pad: a cylinder resting on both
arrises, square at its ends) is swept along each stroke's path, and the stroke ends at the last row whose cut, run-in
and run-out included, stays within the budget. The engine's cut per stroke matched the plan within about 3 % in four
CPU windows (gun stops, the top end, the sagged stretch); the profile fill into sagged cells only lowers the load.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Nose:
    """The nose's pad: radius, half length along the joint [m], its axis over the faces when it rests on the
    arrises [m]."""

    r: float
    half: float
    rest: float


@dataclass(frozen=True)
class Path:
    """A stroke's nose path along the joint (downward): it comes down at ``run_in`` above the stroke's top a and slides
    to a at rest, slides to the stroke's bottom b, then runs out ``run_out`` further while rising ``rise``, then lifts
    straight up (which cuts nothing more)."""

    run_in: float
    run_out: float
    rise: float
    step: float = 5.0e-4                  # m between the swept centres


def underside(x: np.ndarray, dz: np.ndarray, h: float, nose: Nose) -> np.ndarray:
    """The pad's underside height over the faces [m] above cells at across x and along-offset dz from its centre, its
    axis at h; +inf where it is not over the cell (a cell is under it when its centre is: the store's whole-cell
    sole)."""
    q = nose.r**2 - x**2
    return np.where((q > 0.0) & (np.abs(dz) <= nose.half), h - np.sqrt(np.maximum(q, 0.0)), np.inf)


def centres(a: float, b: float, path: Path, nose: Nose) -> np.ndarray:
    """The nose's centre (z, axis height) along a stroke from a down to b, with its run-in and run-out."""
    n_in = max(1, int(np.ceil(path.run_in / path.step)))
    n_mid = max(1, int(np.ceil((a - b) / path.step)))
    n_out = max(1, int(np.ceil(path.run_out / path.step)))
    z_in = np.linspace(a + path.run_in, a, n_in + 1)
    z_mid = np.linspace(a, b, n_mid + 1)[1:]
    s = np.linspace(0.0, 1.0, n_out + 1)[1:]
    z = np.concatenate([z_in, z_mid, b - s * path.run_out])
    h = np.concatenate([np.full(n_in + 1 + n_mid, nose.rest), nose.rest + s * path.rise])
    return np.stack([z, h], axis=1)


@dataclass
class Field:
    """The sealant surface over the joint cells [m over the faces], rows along z, and the cells' across offsets."""

    surf: np.ndarray                      # [cols, rows] (+-inf off the joint: never cut)
    x: np.ndarray                         # [cols, 1] across offset from the joint's centre line
    z: np.ndarray                         # [rows]
    cell: float


def sweep(f: Field, cs: np.ndarray, nose: Nose) -> tuple[float, np.ndarray, slice]:
    """What the nose cuts along centres ``cs`` [m³], the rows' surface after it and the rows it reached."""
    reach = nose.half
    i0 = int(np.searchsorted(f.z, cs[:, 0].min() - reach))
    i1 = int(np.searchsorted(f.z, cs[:, 0].max() + reach, side="right"))
    rows = slice(i0, i1)
    z = f.z[rows][None, :]
    swept = np.full((f.x.shape[0], i1 - i0), np.inf)
    for zc, hc in cs:
        np.minimum(swept, underside(f.x, z - zc, hc, nose), out=swept)
    s = f.surf[:, rows]
    cut = float(np.sum(np.clip(s - swept, 0.0, None)[np.isfinite(s)])) * f.cell**2
    return cut, np.minimum(s, swept), rows


def plan(f: Field, lo: float, hi: float, budget: float, max_len: float, path: Path, nose: Nose) -> list[float]:
    """Stroke edges from hi down to lo [m along z]: each stroke as long as its cut stays within ``budget`` [m³] (at
    least one row, at most ``max_len``)."""
    edges, a = [hi], hi
    while a > lo + 1e-9:
        b, best = a, None
        while True:
            b = max(b - f.cell, lo)
            cut = sweep(f, centres(a, b, path, nose), nose)[0]
            if best is not None and cut > budget:
                break
            best = b
            if b <= lo or a - b >= max_len - 1e-12:
                break
        _, surf, rows = sweep(f, centres(a, best, path, nose), nose)
        f.surf[:, rows] = surf
        edges.append(best)
        a = best
    return edges
