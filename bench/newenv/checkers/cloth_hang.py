# SPDX-License-Identifier: Apache-2.0
"""E11 cloth hang: a 40 cm, 100 g cloth held by two adjacent corners 1 m up; corners driven at up to 0.2 m/s.

Derivations (from the request sentence only; nothing from an engine's frames, vertex order or counts):
- Rest, hang depth: held by two adjacent corners 0.4 m apart, the sheet hangs as a curtain whose side is 0.40 m; its
  weight is 0.98 N over a 0.4 m edge (2.5 N/m), a strain of well under 1 % for cotton, so the lowest point is about
  0.40 m below the corners. If the corners are held closer than a side the sheet bulges and the bottom edge rides
  a few cm higher. The request's band 0.30-0.45 m covers both (it is also the "not stretched" bound: 0.40 x 1.1).
- Corners 1 m above the ground (+-2 cm); they hold within 2 mm at rest (a held corner does not sag).
- Stretch: an in-plane sheet's longest extent is its diagonal 0.4 sqrt(2) = 0.566 m; <= 1.1 of it at every row, and the
  adjacent corners are at most 0.40 x 1.1 apart (and, in the scripted phases, not closer than 0.1 m: they hold one edge; CHECK-CHANGE: the first draft also applied this to "random", where independent random corner motion closes them to 1.6 cm). A cloth that is far
  smaller than that (extent < 0.8 of the diagonal) is not a 40 cm cloth.
- Act, speed: at action 1 each corner moves 0.2 m/s: over 1 s its displacement is 0.2 m (+-10 %), along the
  commanded axis, with under 1 cm off-axis. The cloth follows: after sliding 0.2 m sideways its centroid moved
  at least half of that (0.1 m; a hanging sheet trails its edge like a pendulum), at most the corner's own travel;
  after lifting 0.2 m its lowest point rose between 0.10 and 0.22 m.
- Nothing below the ground (z >= -2 mm) in any phase.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

SIDE = 0.40
DIAG = SIDE * np.sqrt(2.0)
CORNER_SPEED = 0.2
MIN_POINTS = 100


def _corner_mid(r: c.Rollout, name: str) -> np.ndarray:
    return r.phase(name).probe["corner_pos"].mean(axis=2)          # (T, N, 3)


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("rest").probe
    corners, pts = p["corner_pos"], p["cloth_points"]
    drift = np.linalg.norm(corners - corners[0], axis=-1).max(axis=(0, 2))
    depth = corners[-1].mean(axis=1)[:, 2] - pts[-1, :, :, 2].min(axis=1)
    return [c.within("rest_corner_hold", drift, 0.0, 0.002, 2),
            c.within("rest_corner_height", corners[-1].mean(axis=1)[:, 2], 0.98, 1.02, 2),
            c.within("rest_hang_depth", depth, 0.30, 0.45, 2),
            c.at_least("cloth_points", float(pts.shape[2]), MIN_POINTS, 2)]


def _all_phase_checks(r: c.Rollout) -> list[c.Check]:
    low, ext, cdist = np.inf, 0.0, []
    for ph in r.phases.values():
        if "cloth_points" not in ph.probe:
            continue
        pts = ph.probe["cloth_points"]
        low = min(low, float(np.nanmin(pts[..., 2])))
        ext = max(ext, float(np.nanmax(c.extent(pts))))
        if ph.name != "random":      # random actions move the corners independently: they may legitimately close in
            cdist.append(np.linalg.norm(np.diff(ph.probe["corner_pos"], axis=2)[:, :, 0], axis=-1))
    rest_ext = c.extent(r.phase("rest").probe["cloth_points"][-1])
    d = np.concatenate(cdist)
    return [c.at_least("above_ground", low, -0.002, 2),
            c.at_most("cloth_stretch", ext, 1.1 * DIAG, 2),
            c.within("cloth_size", rest_ext, 0.8 * DIAG, 1.1 * DIAG, 2),
            c.at_most("corner_distance_max", float(d.max()), SIDE * 1.1, 2),
            c.at_least("corner_distance_min", float(d.min()), 0.1, 2)]


def _speed_checks(r: c.Rollout, name: str, axis: int) -> list[c.Check]:
    ph = r.phase(name)
    corners = ph.probe["corner_pos"]
    d = corners[-1] - corners[0]                                   # (N, 2, 3)
    secs = float(ph.t[-1] - ph.t[0])
    off = np.delete(d, axis, axis=-1)
    return [c.within(f"{name}_corner_travel", d[..., axis].ravel(), 0.9 * CORNER_SPEED * secs, 1.1 * CORNER_SPEED * secs, 2),
            c.at_most(f"{name}_off_axis", float(np.abs(off).max()), 0.01, 2)]


def _follow_checks(r: c.Rollout) -> list[c.Check]:
    mv, lf = r.phase("act_move").probe["cloth_points"], r.phase("act_lift").probe["cloth_points"]
    dx = np.nanmean(mv[-1, :, :, 0], axis=1) - np.nanmean(mv[0, :, :, 0], axis=1)
    rise = lf[-1, :, :, 2].min(axis=1) - lf[0, :, :, 2].min(axis=1)
    return [c.within("act_move_cloth_follows", dx, 0.5 * CORNER_SPEED, 1.1 * CORNER_SPEED, 2),
            c.within("act_lift_cloth_follows", rise, 0.10, 0.22, 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request)
    checks += _rest_checks(rollout) + _all_phase_checks(rollout)
    checks += _speed_checks(rollout, "act_move", 0) + _speed_checks(rollout, "act_lift", 2)
    checks += _follow_checks(rollout)
    return c.finish(checks)
