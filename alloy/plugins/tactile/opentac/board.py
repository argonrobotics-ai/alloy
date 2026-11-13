# SPDX-License-Identifier: Apache-2.0
"""An OpenTac sensor board from its ``manifest.json``: taxel points in metres, the reach, and the bump groups.

The manifest is OpenTac-CAD's export (github.com/opentac/opentac, read at commit e302861). Lengths are in mm, Y up,
in one frame shared by ``outline``, ``edge_cuts`` and ``pixels``. The fields read here:
- ``pixels``: ``[x, y, row, col]`` per active taxel (the authority on where taxels sit);
- ``r``, ``c``: the scan grid; ``w``, ``h``: the taxel size;
- ``shear.bumps``: bump centres; ``shear.b``: the bump above each taxel (-1 = none), index-parallel to ``pixels``;
- ``bump_params`` (optional): bump-sheet fields chosen after export (only ``bump_height_mm`` and ``bump_base_frac``
  matter here).

The checks are a port of OpenTac's ``opentac/sim/mujoco/loader.py`` (``TactileSensor.__init__``), MIT licence,
Copyright (c) 2026 The OpenTac authors (``LICENSE-OPENTAC`` beside this file). The JSON is parsed here: there is no
dependency on their package. A sensor wrapped on a curved part (a ``sim3d`` block with ``taxels3d``) is refused, as
theirs is.

The pad frame of ``Tactile`` is centred: ``Board.xy`` is the manifest's taxel positions less the centre of their
bounding box (``Board.centre_mm``), in metres. The manifest's x, y become the pad's x, y, and the pad's z is the
sensing side's outward normal.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MM = 1e-3
NO_BUMP = -1
# OpenTac-CAD's bump-sheet defaults (opentac/sim/mujoco/params.py, BumpParams; MIT, The OpenTac authors).
DEFAULT_BUMP_HEIGHT_MM = 2.0
DEFAULT_BUMP_BASE_FRAC = 0.8
# loader.py FOOTPRINT_TOLERANCE_M: footprints may touch; larger than the pitch by more than this overlaps.
FOOTPRINT_TOLERANCE_M = 1e-5
_BUMP_KEYS = ("bump_height_mm", "bump_base_frac", "bump_top_ratio", "sheet_thickness_mm", "e_eff_mpa", "poisson",
              "density", "mu", "omega_max", "zeta", "damping_mode")


class ManifestError(ValueError):
    """The manifest cannot describe a flat OpenTac board (the message names the field)."""


@dataclass(frozen=True)
class Board:
    """One flat OpenTac board, in SI, pad frame centred on the taxels' bounding box.

    ``taxel_bump`` [T] holds each taxel's bump (-1 = none). ``members``/``member_start`` list each bump's taxels
    (CSR), and ``moment_weights`` [T, 2] are OpenTac's minimum-norm couple weights about each group's centroid.
    ``bumps`` is False for a plain skin (no bump sheet): the foam route applies.
    """

    name: str
    xy: np.ndarray                    # [T, 2] m, pad frame
    rc: np.ndarray                    # [T, 2] scan row, column
    scan_shape: tuple[int, int]       # (r, c)
    taxel_size: tuple[float, float]   # m, (w, h)
    pitch: tuple[float, float]        # m, x and y
    reach: float                      # m, nearest-taxel capture radius
    centre_mm: tuple[float, float]    # manifest point at the pad origin
    bumps: bool
    bump_xy: np.ndarray               # [B, 2] m, pad frame (B = 0 on a plain skin)
    taxel_bump: np.ndarray            # [T] int
    member_start: np.ndarray          # [B + 1] int
    members: np.ndarray               # [sum of group sizes] taxel index
    moment_weights: np.ndarray        # [T, 2]
    bump_height: float                # m

    @property
    def n_taxels(self) -> int:
        return len(self.xy)

    @property
    def n_bumps(self) -> int:
        return len(self.bump_xy)

    def group_sizes(self) -> np.ndarray:
        """Taxels under each bump: 4 on a full 2x2 block, fewer where the outline cut it."""
        return np.diff(self.member_start)

    def describe(self) -> str:
        sizes = self.group_sizes()
        kind = (f"{self.n_bumps} bumps ({int((sizes < 4).sum())} on partial blocks), height "
                f"{self.bump_height / MM:.2f} mm" if self.bumps else "no bump sheet (foam route)")
        return (f"OpenTac board {self.name}: {self.n_taxels} taxels on a {self.scan_shape[0]}x{self.scan_shape[1]} "
                f"scan grid, pitch {self.pitch[0] / MM:.3f} x {self.pitch[1] / MM:.3f} mm, {kind}")


def read_manifest(source: Any) -> tuple[dict, str]:
    """(dict, name) from a manifest dict or a path to a ``manifest.json``."""
    if isinstance(source, dict):
        return source, str(source.get("name", "manifest"))
    path = Path(source)
    if not path.is_file():
        raise ManifestError(f"{path}: no such manifest file")
    try:
        return json.loads(path.read_text(encoding="utf-8")), path.stem
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path.name} is not JSON: {exc}") from None


def _pixels(m: dict) -> np.ndarray:
    pixels = m.get("pixels")
    if not pixels:
        raise ManifestError("manifest has no 'pixels'; nothing to simulate")
    if m.get("units", "mm") != "mm":
        raise ManifestError("manifest lengths must be in mm ('units')")
    try:
        px = np.asarray([p[:4] for p in pixels], dtype=float)
    except (TypeError, ValueError):
        raise ManifestError("pixels must be [x, y, row, col] numbers") from None
    if px.ndim != 2 or px.shape[1] != 4 or not np.isfinite(px).all():
        raise ManifestError("pixels must contain finite x, y, row, col values")
    if np.any(px[:, 2:4] != np.floor(px[:, 2:4])):
        raise ManifestError("pixel row and column indices must be integers")
    if len(np.unique(px[:, :2], axis=0)) != len(px) or len(np.unique(px[:, 2:4], axis=0)) != len(px):
        raise ManifestError("pixel positions and scan indices must be unique")
    return px


def _reject_curved(m: dict) -> None:
    sim = m.get("sim3d")
    if sim and sim.get("taxels3d"):
        raise ManifestError("this sensor is mounted on a curved surface (the manifest has a 'sim3d' block). Curved "
                            "placement is not supported: a wrapped sensor modelled as a flat pad has every contact "
                            "normal in the wrong direction (OpenTac's loader refuses it too)")


def _scan_shape(m: dict, rc: np.ndarray) -> tuple[int, int]:
    """(r, c) from the manifest, or from the indices when a design has no scan fields (a plain test grid)."""
    rows = int(m.get("r", rc[:, 0].max() + 1))
    cols = int(m.get("c", rc[:, 1].max() + 1))
    if rows <= 0 or cols <= 0 or np.any(rc < 0) or np.any(rc >= [rows, cols]):
        raise ManifestError("pixel scan indices are outside the sensor dimensions (r, c)")
    return rows, cols


def geometric_rc(xy: np.ndarray) -> np.ndarray:
    """Each taxel's (row, col) by position, top row first (port of loader.geometric_rc; not the scan order)."""
    ux, uy = np.unique(np.round(xy[:, 0], 9)), np.unique(np.round(xy[:, 1], 9))
    col = np.searchsorted(ux, np.round(xy[:, 0], 9))
    row = np.searchsorted(uy, np.round(xy[:, 1], 9))
    return np.column_stack([len(uy) - 1 - row, col])


def neighbour_pitch(xy: np.ndarray) -> tuple[float, float]:
    """Median spacing of geometric neighbours (port of loader._derive_pitch): the bump route's pitch."""
    by_rc = {(int(r), int(c)): i for i, (r, c) in enumerate(geometric_rc(xy))}
    dx, dy = [], []
    for (r, c), i in by_rc.items():
        if (j := by_rc.get((r, c + 1))) is not None:
            dx.append(abs(xy[j, 0] - xy[i, 0]))
        if (j := by_rc.get((r + 1, c))) is not None:
            dy.append(abs(xy[j, 1] - xy[i, 1]))
    if not dx or not dy:
        raise ManifestError("cannot derive the taxel pitch: no two taxels are adjacent in a row and in a column")
    return float(np.median(dx)), float(np.median(dy))


def square_pitch_mm(xy_mm: np.ndarray) -> float:
    """The smallest gap between distinct x (and y), rounded to 1 um (port of transfer.pixel_pitch_mm and
    square_pitch_mm): the foam route's pitch. Square spacing only, as in OpenTac."""
    out = []
    for axis in (0, 1):
        v = np.unique(np.round(xy_mm[:, axis], 6))
        if len(v) < 2:
            raise ManifestError(f"cannot derive the {'xy'[axis]} pitch: one distinct {'xy'[axis]} in pixels")
        out.append(round(float(np.diff(v).min()) * 1000.0) / 1000.0)
    if out[0] != out[1]:
        raise ManifestError(f"taxel pitch differs in x and y ({out[0]} / {out[1]} mm); OpenTac's foam model "
                            "supports square taxel spacing only")
    return out[0]


def _taxel_size(m: dict, pitch: tuple[float, float]) -> tuple[float, float]:
    size = (float(m.get("w", pitch[0] / MM)) * MM, float(m.get("h", pitch[1] / MM)) * MM)
    if not np.isfinite(size).all() or min(size) <= 0:
        raise ManifestError("taxel dimensions 'w', 'h' must be positive and finite")
    if np.any(np.asarray(size) > np.asarray(pitch) + FOOTPRINT_TOLERANCE_M):
        raise ManifestError("taxel footprints overlap at the derived pitch")
    return size


def _bump_params(m: dict) -> dict:
    extra = dict(m.get("bump_params") or {})
    if "taxel_pitch_mm" in extra:
        raise ManifestError("bump_params cannot set taxel_pitch_mm; the pitch comes from the taxel positions")
    unknown = sorted(set(extra) - set(_BUMP_KEYS))
    if unknown:
        raise ManifestError(f"bump_params has unknown fields {unknown}; known: {list(_BUMP_KEYS)}")
    h = float(extra.get("bump_height_mm", DEFAULT_BUMP_HEIGHT_MM))
    frac = float(extra.get("bump_base_frac", DEFAULT_BUMP_BASE_FRAC))
    if not (np.isfinite(h) and h > 0 and np.isfinite(frac) and frac > 0):
        raise ManifestError("bump_params: bump_height_mm and bump_base_frac must be positive and finite")
    return {"height_mm": h, "base_frac": frac}


def _bump_groups(m: dict, n_taxels: int) -> tuple[np.ndarray, np.ndarray]:
    """(bump centres [B, 2] mm, taxel_bump [T]) from the ``shear`` block, with loader.py's checks."""
    shear = m.get("shear") or {}
    bumps = shear.get("bumps")
    if not bumps:
        raise ManifestError(f"manifest (v{m.get('v', '?')}) has no usable 'shear' block, so no taxel can be put "
                            "under a bump; OpenTac-CAD added it in v5: re-export, or use bumps=False (plain skin)")
    if shear.get("b") is None:
        raise ManifestError("'shear' has bump centres but no 'b', so no taxel can be assigned to a bump")
    bxy = np.asarray(bumps, dtype=float)
    if bxy.ndim != 2 or bxy.shape[1] != 2 or not np.isfinite(bxy).all():
        raise ManifestError("shear.bumps must contain finite XY centres")
    if len(np.unique(bxy, axis=0)) != len(bxy):
        raise ManifestError("bump centres must be unique")
    raw_b = np.asarray(shear["b"], dtype=float)
    if raw_b.ndim != 1 or not np.isfinite(raw_b).all() or np.any(raw_b != np.floor(raw_b)):
        raise ManifestError("shear.b must contain integer bump indices")
    if np.any(raw_b < NO_BUMP) or np.any(raw_b >= len(bxy)):
        raise ManifestError("shear.b index is outside [-1, number of bumps)")
    if len(raw_b) != n_taxels:
        raise ManifestError(f"'shear.b' has {len(raw_b)} entries but there are {n_taxels} taxels; they are "
                            "index-parallel")
    return bxy, raw_b.astype(np.int64)


def _check_bump_bases(bxy_m: np.ndarray, pitch: tuple[float, float], base_frac: float) -> None:
    """Refuse overlapping bump bases (loader._check_bump_bases): equal ellipses of semi-axes r_base overlap when
    their scaled centres are closer than 2. r_base = base_frac x bump pitch / 2, bump pitch = 2 x taxel pitch."""
    if len(bxy_m) < 2:
        return
    r_base = np.asarray(pitch) * base_frac
    s = bxy_m / r_base
    d = np.linalg.norm(s[:, None] - s[None], axis=-1) + np.eye(len(s)) * 1e30
    if d.min() < 2.0 - 1e-9:
        raise ManifestError("neighbouring bump bases overlap; reduce bump_params.bump_base_frac")


def _groups(taxel_bump: np.ndarray, n_bumps: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR (member_start [B + 1], members) of the taxels under each bump."""
    has = np.flatnonzero(taxel_bump != NO_BUMP)
    order = has[np.argsort(taxel_bump[has], kind="stable")]
    sizes = np.bincount(taxel_bump[has], minlength=n_bumps)
    if np.any(sizes == 0):
        raise ManifestError("every bump must have at least one assigned taxel")
    return np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64), order.astype(np.int64)


def moment_weights(xy: np.ndarray, bump_xy: np.ndarray, taxel_bump: np.ndarray) -> np.ndarray:
    """Minimum-norm couple weights about each group's centroid (loader.__init__ with _centred_offsets): pinv of the
    centred offsets, so asymmetric and three-taxel groups work and a rank-deficient group reports only the moment
    component its geometry observes."""
    d = centred_offsets(xy, bump_xy, taxel_bump)
    w = np.zeros((len(xy), 2))
    for k in range(len(bump_xy)):
        idx = np.flatnonzero(taxel_bump == k)
        w[idx] = np.linalg.pinv(d[idx].T)
    return w


def centred_offsets(xy: np.ndarray, bump_xy: np.ndarray, taxel_bump: np.ndarray) -> np.ndarray:
    """Taxel offsets from their group's centroid (loader._centred_offsets); zero for a taxel with no bump."""
    d = np.zeros_like(xy)
    has = taxel_bump != NO_BUMP
    if not has.any():
        return d
    b = taxel_bump[has]
    d[has] = xy[has] - bump_xy[b]
    n = np.maximum(np.bincount(b, minlength=len(bump_xy)), 1)
    for axis in (0, 1):
        c = np.bincount(b, weights=d[has, axis], minlength=len(bump_xy)) / n
        d[has, axis] -= c[b]
    return d


def load_board(source: Any, bumps: bool = True, name: str | None = None) -> Board:
    """A ``Board`` from a manifest dict or path. ``bumps=False`` reads it as a plain skin (no bump sheet)."""
    m, default_name = read_manifest(source)
    px = _pixels(m)
    _reject_curved(m)
    xy_mm, rc = px[:, :2], px[:, 2:4].astype(np.int64)
    shape = _scan_shape(m, rc)
    lo, hi = xy_mm.min(0), xy_mm.max(0)
    centre = 0.5 * (lo + hi)
    xy = (xy_mm - centre) * MM
    T = len(xy)
    if bumps:
        pitch = neighbour_pitch(xy)
        bxy_mm, taxel_bump = _bump_groups(m, T)
        bp = _bump_params(m)
        bump_xy = (bxy_mm - centre) * MM
        _check_bump_bases(bump_xy, pitch, bp["base_frac"])
        start, members = _groups(taxel_bump, len(bump_xy))
        weights, height = moment_weights(xy, bump_xy, taxel_bump), bp["height_mm"] * MM
    else:
        p = square_pitch_mm(xy_mm) * MM
        pitch, bump_xy, taxel_bump = (p, p), np.zeros((0, 2)), np.full(T, NO_BUMP, np.int64)
        start, members, weights, height = np.zeros(1, np.int64), np.zeros(0, np.int64), np.zeros((T, 2)), 0.0
    if not np.isfinite(pitch).all() or min(pitch) <= 0:
        raise ManifestError("taxel pitch must be positive and finite")
    return Board(name=name or default_name, xy=xy, rc=rc, scan_shape=shape, taxel_size=_taxel_size(m, pitch),
                 pitch=(float(pitch[0]), float(pitch[1])), reach=float(0.5 * np.hypot(*pitch)),
                 centre_mm=(float(centre[0]), float(centre[1])), bumps=bool(bumps), bump_xy=bump_xy,
                 taxel_bump=taxel_bump, member_start=start, members=members, moment_weights=weights,
                 bump_height=float(height))
