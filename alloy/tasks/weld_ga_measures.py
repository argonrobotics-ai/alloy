# SPDX-License-Identifier: Apache-2.0
"""GA's weld sections on GridThermal (SCR-WP8-14): a port of GA's ``sections.tee_section``. A slice with a cap (the
pool's free surface, ``grid_cap``) is read on that surface as GA reads a slice with a pool profile
(``weld_ga_profile``); the others on their cells as below.

One row per 0.5 mm slice along x, with GA's keys (lengths in m). In the weld frame the root is the x axis, the plate's
top is z = 0 and the bracket's weld face y = 0; the bead lies in y > 0, z > 0.
- legs: contiguous weld metal from the root up the bracket's first column and out along the plate's top row;
- face: air cells of the quadrant beside weld metal; throat: the least root-to-face distance (less half a cell);
  convexity: the face's largest distance beyond the toe-to-toe line (less half a cell); size: min(legs, throat / 0.707)
  (AWS D1.1 Annex B, the largest inscribed isosceles right triangle);
- fusion: parent metal whose peak reached T_half (GA: peak liquid fraction >= 0.5); root fused: the plate cell under
  the root and the bracket cell beside it; penetration: the deepest fused cell below the plate's top (y >= 0) and behind
  the bracket's face (z >= 0);
- overlap (cold lap): per plate column, its top cell unfused with weld metal anywhere above it; per bracket row, its
  face cell unfused with weld metal anywhere in the row (sum of widths);
- freeze_t: when the slice last froze (GridThermal's ``liquid_s``; None if it is liquid now or never held a pool);
- undercut and porosity are 0: parent metal never leaves (heat v3) and there is no shielding gas to lose.
On the surface, the legs are the toes and size, throat and convexity are read on the cap (sub-cell); on cells a leg
steps in 0.25 mm, so a 6.4 mm fillet reads 6.25.
"""

from __future__ import annotations

import math

import numpy as np

from alloy.tasks.weld_ga_profile import Cap, read

X_READ = (-2e-3, 0.152)                    # m: the joint and 2 mm past each end (GA task.X_READ)
GEOMETRY = ("leg_vertical_m", "leg_horizontal_m", "throat_m", "size_m", "convexity_m", "concavity_m", "face_width_m")


def _centres(e: np.ndarray) -> np.ndarray:
    return 0.5 * (e[1:] + e[:-1])


def _run(mask: np.ndarray) -> int:
    """Length of the leading run of True."""
    off = np.flatnonzero(~mask)
    return int(off[0]) if off.size else int(mask.size)


def _face(weld: np.ndarray, metal: np.ndarray, yc: np.ndarray, zc: np.ndarray, dy, dz, legs) -> dict:
    """Throat, convexity, concavity and face width from the quadrant's air cells beside weld metal (GA's cells)."""
    leg_v, leg_h = legs
    out = dict(throat_m=0.0, convexity_m=0.0, concavity_m=0.0, face_width_m=0.0)
    q = (yc[:, None] > 0) & (zc[None, :] > 0)
    nb = np.zeros_like(weld)
    nb[1:, :] |= weld[:-1, :]
    nb[:-1, :] |= weld[1:, :]
    nb[:, 1:] |= weld[:, :-1]
    nb[:, :-1] |= weld[:, 1:]
    face = ~metal & q & nb
    if not (face.any() and leg_v > 0 and leg_h > 0):
        return out
    Y, Z = np.meshgrid(yc, zc, indexing="ij")
    half = 0.5 * np.minimum(dy[:, None], dz[None, :])
    inside = face & (Y <= leg_h + 2e-3) & (Z <= leg_v + 2e-3)
    out["throat_m"] = float((np.sqrt(Y**2 + Z**2) - half)[inside].min()) if inside.any() else 0.0
    a, b = np.array([0.0, leg_v]), np.array([leg_h, 0.0])
    t = (b - a) / np.linalg.norm(b - a)
    n = np.array([t[1], -t[0]])
    n = -n if n @ np.array([1.0, 1.0]) < 0 else n
    s = (Y - a[0]) * n[0] + (Z - a[1]) * n[1] - half
    u = (Y - a[0]) * t[0] + (Z - a[1]) * t[1]
    span = inside & (u > 0) & (u < np.linalg.norm(b - a))
    if span.any():
        out["convexity_m"] = float(max(0.0, s[span].max()))
        out["concavity_m"] = float(max(0.0, -s[span].min()))
    out["face_width_m"] = float(np.linalg.norm(b - a))
    return out


def section(cell: np.ndarray, peak: np.ndarray, ey: np.ndarray, ez: np.ndarray, weld: int, t_half: float,
            cap: Cap | None = None) -> dict:
    """GA's measures of one slice (``cell``, ``peak``: [ny, nz]; ``ey``, ``ez``: the cell edges); with ``cap``, the
    shape on its surface (GA keeps the cells' reading as <key>_cells)."""
    yc, zc = _centres(ey), _centres(ez)
    dy, dz = np.diff(ey), np.diff(ez)
    jb, kb = int(np.searchsorted(yc, 0.0)), int(np.searchsorted(zc, 0.0))
    w = cell == weld
    metal = cell != 0
    orig = metal & ~w                          # parent metal: heat v3 never turns it into weld metal
    fused = orig & (peak >= t_half)
    nv, nh = _run(w[jb, kb:]), _run(w[jb:, kb])
    leg_v = float(ez[kb + nv]) if nv else 0.0
    leg_h = float(ey[jb + nh]) if nh else 0.0
    r = dict(weld_area_m2=float((w * np.outer(dy, dz)).sum()), leg_vertical_m=leg_v, leg_horizontal_m=leg_h)
    r.update(_face(w, metal, yc, zc, dy, dz, (leg_v, leg_h)))
    r["size_m"] = float(min(leg_v, leg_h, r["throat_m"] / math.sqrt(0.5))) if leg_v > 0 and leg_h > 0 else 0.0
    base_open = orig[jb:, kb - 1] & ~fused[jb:, kb - 1]
    r["overlap_base_m"] = float(((base_open & w[jb:, kb:].any(1)) * dy[jb:]).sum())
    face_open = orig[jb - 1, kb:] & ~fused[jb - 1, kb:]
    r["overlap_bracket_m"] = float(((face_open & w[jb:, kb:].any(0)) * dz[kb:]).sum())
    r["root_fused"] = bool(fused[jb, kb - 1] and fused[jb - 1, kb])
    fb = fused & (zc[None, :] < 0) & (yc[:, None] > -1e-9)
    r["penetration_base_m"] = float((-(ez[:-1])[None, :] * fb).max()) if fb.any() else 0.0
    fk = fused & (yc[:, None] < 0) & (zc[None, :] > -1e-9)
    r["penetration_bracket_m"] = float((-(ey[:-1])[:, None] * fk).max()) if fk.any() else 0.0
    r.update(undercut_base_m=0.0, undercut_bracket_m=0.0, porosity=0.0, geometry="cells")
    if cap is not None and r["weld_area_m2"] > 0:
        for k in GEOMETRY:
            r[k[:-2] + "_cells_m"] = r[k]
        r.update(read(cap, w, orig, ey, ez))
    return r


def measure(grid, world: int = 0, span: tuple[float, float] = X_READ, weld: int = 3) -> list[dict]:
    """GA's rows (one per slice whose centre lies in ``span``) of a ``GridThermal`` with a fixed-root cap model, with
    x_m, dx_m, liquid (metal near the root at or above T_half at the last exchange) and freeze_t; a slice the cap
    shaped (fed, both toes set) is read on its surface (area: its weld metal and the wire still owed)."""
    ex, ey, ez = (np.asarray(e, np.float64) for e in (grid.config.x, grid.config.y, grid.config.z))
    m = grid._mat
    t_half = 0.5 * (m.t_solidus + m.t_liquidus)
    f = grid.fields(world)
    liquid_s = grid.liquid_s.numpy()[world]
    now = float(grid.clock_s.numpy()[world])
    toes, fed = grid.toes.numpy()[world], grid.slice_fed_m3.numpy()[world]
    owed, dyz = grid.deposit_budget.numpy()[world], np.outer(np.diff(ey), np.diff(ez))
    cx = _centres(ex)
    rows = []
    for i in np.flatnonzero((cx >= span[0]) & (cx <= span[1])):
        area = float(((f["cell"][i] == weld) * dyz).sum()) + float(owed[i]) / float(ex[i + 1] - ex[i])
        cap = Cap(float(toes[i][0]), float(toes[i][1]), area) if fed[i] > 0 and min(toes[i]) > 0 else None
        r = section(f["cell"][i], f["peak"][i], ey, ez, weld, t_half, cap)
        liquid = bool(liquid_s[i] > 0 and liquid_s[i] >= now - 1e-9)
        r.update(x_m=float(cx[i]), dx_m=float(ex[i + 1] - ex[i]), liquid=liquid,
                 freeze_t=None if liquid or liquid_s[i] < 0 else float(liquid_s[i]))
        rows.append(r)
    return rows


__all__ = ["X_READ", "measure", "section"]
