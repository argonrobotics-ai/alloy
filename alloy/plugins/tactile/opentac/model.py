# SPDX-License-Identifier: Apache-2.0
"""The board model in NumPy: contact forces to per-taxel load (bump split or foam spread), and the decoder.

This is the reference the Warp kernels (``kernels.py``) are checked against, and the decoder a user runs on the
``counts`` reading. Ported from OpenTac (github.com/opentac/opentac, commit e302861), MIT licence, Copyright (c) 2026
The OpenTac authors (``LICENSE-OPENTAC`` beside this file):

- **Bump split** (``split_wrench``): ``TactileSensor.taxel_forces_from_wrench`` + ``_apply_moments`` in
  ``opentac/sim/mujoco/loader.py``. Each bump's base force is shared evenly over its member taxels, its base moment
  is added as a couple with minimum-norm weights about the group centroid, taxels are clipped at zero (a taxel
  cannot pull) and the group is rescaled to the bump's base force.
- **Base moment** (``base_moments``): by default the guided-cantilever term ``F_t * H / 2`` of the shear, as in
  ``Surrogate.base_moments`` (``opentac/sim/mujoco/surrogate.py``). With ``tip`` (``Tactile.Config.opentac_tip``)
  it adds the tip hinges' moment of their full scene (``SensorScene.base_moments``, ``scene.py``) at its static
  value, the first moment of the normal load about the bump centre (sum of ``F_n * (x - x_bump)``), since the forces
  here come from the solver's contacts, not from bump joints. Off by default: MuJoCo puts a flat face's contacts at
  the corners of the overlap, anywhere on a bump's block, and lets a held object creep, so the first moment of a flat
  grip decodes as shear that is not there (``docs/notes/opentac.md``). It is right for a curved object touching one
  spot of a bump.
- **Foam spread** (``foam_loads``): ``opentac/sim/foam.py``: a Gaussian of sigma 3.0 mm (fitted on one skin in the
  OpenTac paper) integrated over each taxel's square, not renormalised. Here the pressure is the contacts' point
  loads, and the Gaussian and the square are integrated exactly (erf), not on their 0.5 mm grid.
- **Decoder** (``decode``): ``Readout.forces`` + ``TactileSensor.decode_shear``: counts to newtons, then each
  group's first moment about its centroid gives the base moment and ``F_t = 2 M / H``.

Their bump model is marked "Not validated against hardware" by its authors, and nobody has compared this port
with a real skin.
"""

from __future__ import annotations

import math

import numpy as np

from alloy.plugins.tactile.opentac import readout as R
from alloy.plugins.tactile.opentac.board import NO_BUMP, Board, centred_offsets

# opentac/sim/foam.py (MIT, The OpenTac authors): fitted once on robotiq_long (96 taxels at 4 mm) in the paper.
FOAM_SIGMA_MM = 3.0
FOAM_GRID_MM = 0.5          # their pressure-field sampling; it sets the width of their taxel squares


def foam_half_width(pitch: float, cells: str = "opentac") -> float:
    """Half width (m) of the square each taxel integrates. "opentac": their square of 2 * half + 1 grid cells,
    half = max(1, round(pitch / 2 / 0.5 mm)), which is wider than the pitch unless the pitch is an odd multiple of
    0.5 mm (2.5 mm: equal; 4 mm: 4.5 mm; 2 mm: 2.5 mm); "pitch": exactly one pitch, so loads add up to the force."""
    if cells == "pitch":
        return 0.5 * pitch
    if cells != "opentac":
        raise ValueError(f"foam cells must be 'opentac' or 'pitch', not {cells!r}")
    half = max(1, int(np.round(pitch / 1e-3 / 2.0 / FOAM_GRID_MM)))
    return (2 * half + 1) * FOAM_GRID_MM * 1e-3 / 2.0


def foam_box_factor(pitch: float, cells: str = "opentac") -> float:
    """Sum of taxel loads over the force, for a contact deep inside a dense board: (square width / pitch)^2."""
    return (2.0 * foam_half_width(pitch, cells) / pitch) ** 2


_ERF = np.vectorize(math.erf)


def box_share(d: np.ndarray, hw: float, sigma: float) -> np.ndarray:
    """Share of a unit point load at offset ``d`` from a taxel centre that a Gaussian of width ``sigma`` puts
    inside the taxel's [-hw, hw] (one axis); ``sigma`` = 0 is no spread."""
    if sigma <= 0.0:
        return (np.abs(d) <= hw).astype(float)
    r = 1.0 / (sigma * math.sqrt(2.0))
    return 0.5 * (_ERF((hw - d) * r) - _ERF((-hw - d) * r))


def foam_loads(board: Board, points: np.ndarray, fn: np.ndarray, cells: str = "opentac",
               sigma_mm: float = FOAM_SIGMA_MM) -> np.ndarray:
    """[T] N per taxel of a plain skin for normal point loads ``fn`` [N] at pad points ``points`` [N, 2] (m)."""
    hw, s = foam_half_width(board.pitch[0], cells), sigma_mm * 1e-3
    pts, fn = np.asarray(points, float).reshape(-1, 2), np.asarray(fn, float).reshape(-1)
    out = np.zeros(board.n_taxels)
    for (x, y), f in zip(pts, fn):
        out += f * box_share(board.xy[:, 0] - x, hw, s) * box_share(board.xy[:, 1] - y, hw, s)
    return out


def nearest_taxel(board: Board, points: np.ndarray) -> np.ndarray:
    """[N] nearest taxel per point, -1 beyond ``board.reach`` (the same capture as ``Tactile``'s taxels)."""
    d = np.linalg.norm(np.asarray(points, float)[:, None] - board.xy[None], axis=-1)
    j = d.argmin(1)
    return np.where(d[np.arange(len(j)), j] <= board.reach, j, -1)


def bump_wrench(board: Board, points: np.ndarray, forces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-bump contact load and the load on bump-less taxels.

    ``points`` [N, 2] pad-frame contact points (m), ``forces`` [N, 3] force on the pad as (normal, shear x, shear y).
    Returns (``wrench`` [B, 5]: F_n, F_x, F_y and the first moments sum F_n dx, sum F_n dy about the bump centre;
    ``direct`` [T]: normal load on taxels with no bump above them).
    """
    pts, f = np.asarray(points, float).reshape(-1, 2), np.asarray(forces, float).reshape(-1, 3)
    wrench, direct = np.zeros((board.n_bumps, 5)), np.zeros(board.n_taxels)
    for p, fc, j in zip(pts, f, nearest_taxel(board, pts)):
        if j < 0:
            continue
        b = board.taxel_bump[j]
        if b == NO_BUMP:
            direct[j] += fc[0]
            continue
        d = p - board.bump_xy[b]
        wrench[b] += (fc[0], fc[1], fc[2], fc[0] * d[0], fc[0] * d[1])
    return wrench, direct


def base_moments(wrench: np.ndarray, height: float, tip: bool = False) -> np.ndarray:
    """[B, 2] (Mx, My) at each bump base: shear x H / 2 (guided cantilever), plus with ``tip`` the normal load's
    first moment about the bump centre (the tip hinges of their full scene, statically)."""
    w = np.asarray(wrench, float)
    t = 1.0 if tip else 0.0
    my = w[:, 1] * height / 2.0 + t * w[:, 3]
    mx = -w[:, 2] * height / 2.0 - t * w[:, 4]
    return np.column_stack([mx, my])


def split_wrench(board: Board, fz: np.ndarray, moments: np.ndarray) -> np.ndarray:
    """[T] taxel loads from per-bump base force [B] and moments [B, 2] (taxel_forces_from_wrench + _apply_moments)."""
    fz = np.clip(np.asarray(fz, float), 0.0, None)
    out = np.zeros(board.n_taxels)
    has = board.taxel_bump != NO_BUMP
    b = board.taxel_bump[has]
    out[has] = fz[b] / np.maximum(board.group_sizes()[b], 1)
    w = board.moment_weights[has]
    m = np.asarray(moments, float)
    out[has] = np.clip(out[has] + w[:, 0] * m[b, 1] - w[:, 1] * m[b, 0], 0.0, None)
    total = np.bincount(b, weights=out[has], minlength=board.n_bumps)
    scale = np.divide(fz, total, out=np.zeros(board.n_bumps), where=total > 0)
    out[has] *= scale[b]
    return out


def board_loads(board: Board, points: np.ndarray, forces: np.ndarray, cells: str = "opentac",
                tip: bool = False) -> np.ndarray:
    """[T] N per taxel as the board's model has it, for contacts on the pad (the ``board_load`` reading)."""
    f = np.asarray(forces, float).reshape(-1, 3)
    if not board.bumps:
        return foam_loads(board, points, f[:, 0], cells)
    wrench, direct = bump_wrench(board, points, f)
    return split_wrench(board, wrench[:, 0], base_moments(wrench, board.bump_height, tip)) + direct


def degenerate_axes(board: Board) -> np.ndarray:
    """[B, 2] True where a group has all its taxels on one side of its centroid: that shear axis is unobservable."""
    d = centred_offsets(board.xy, board.bump_xy, board.taxel_bump)
    has = board.taxel_bump != NO_BUMP
    spread = np.zeros((board.n_bumps, 2))
    np.add.at(spread, board.taxel_bump[has], np.abs(d[has]))
    return spread < 1e-9


def decode_shear(board: Board, per_taxel: np.ndarray) -> np.ndarray:
    """[..., B, 2] per-bump shear (N) from per-taxel normal loads [..., T]: F_t = 2 M / H (decode_shear)."""
    f = np.asarray(per_taxel, float)
    has = np.flatnonzero(board.taxel_bump != NO_BUMP)
    d = centred_offsets(board.xy, board.bump_xy, board.taxel_bump)[has]
    onehot = np.zeros((len(has), board.n_bumps))
    onehot[np.arange(len(has)), board.taxel_bump[has]] = 1.0
    fx = (f[..., has] * d[:, 0]) @ onehot
    fy = (f[..., has] * d[:, 1]) @ onehot
    return np.stack([2.0 * fx, 2.0 * fy], -1) / board.bump_height


def decode(board: Board, counts: np.ndarray, readout: str | R.Profile = "v3", gain: float = 1.0) -> dict:
    """What the firmware can compute from ``counts`` [..., T]: per-taxel N, and per bump the normal load and the
    shear (zero on a plain skin). ``degenerate`` [B, 2] marks shear axes a group cannot observe."""
    p = readout if isinstance(readout, R.Profile) else R.profile(readout)
    taxel_n = R.forces(counts, board.pitch, p, gain)
    out = {"taxel_n": taxel_n, "degenerate": degenerate_axes(board) if board.bumps else np.zeros((0, 2), bool)}
    if not board.bumps:
        out["bump_normal"] = np.zeros(taxel_n.shape[:-1] + (0,))
        out["bump_shear"] = np.zeros(taxel_n.shape[:-1] + (0, 2))
        return out
    onehot = np.zeros((board.n_taxels, board.n_bumps))
    has = np.flatnonzero(board.taxel_bump != NO_BUMP)
    onehot[has, board.taxel_bump[has]] = 1.0
    out["bump_normal"] = taxel_n @ onehot
    out["bump_shear"] = decode_shear(board, taxel_n)
    return out
