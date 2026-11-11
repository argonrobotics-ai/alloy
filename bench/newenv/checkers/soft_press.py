# SPDX-License-Identifier: Apache-2.0
"""E13 soft press: a 10 cm silicone cube (E ~ 100 kPa, 1,100 kg/m3) under a plate moving at up to 2 cm/s.

Derivations (from the sentence only; ``soft_points`` are surface or node points of the cube, so a point-based height
can differ from the exact one by the engine's contact radius; checks use point extents as the request does):
- Rest: self weight compresses a column by rho g h / (2 E) = 1100 x 9.81 x 0.1 / 2e5 = 0.54 % (0.5 mm), so the height
  is 0.0995 m minus whatever the resting plate adds; band 0.095-0.101 m (the request's). The base is on the ground:
  its lowest point within [-4, +6] mm of z = 0. Width (x extent) 0.095-0.105 m.
- Press: the plate moves 2 cm in 1 s at 2 cm/s (+-10 %); the cube top follows it: the cube's height at the end is
  its rest height minus 0.02 m (+-4 mm), i.e. 20 % strain.
- Force at 20 % strain: linear Hooke, frictionless: E eps A = 1e5 x 0.2 x 0.01 m2 = 200 N (the request's ideal). A
  neo-Hookean incompressible block (lambda = 0.8, stress E/3 (lambda - lambda^-2) on the unloaded area) gives 254 N, and
  friction at the plate and the ground (a bonded block, shape factor S = a / 4h = 0.25, compression modulus E (1 + 2 S^2) =
  1.125 E) up to about 1.1x more: so 50-400 N holds the whole range, and the force rises monotonically (smoothed
  over 5 rows, no drop of more than 5 % of the final force).
- Bulge: an incompressible block compressed to 0.8 widens by 1/sqrt(0.8) = 1.118 (a compressible nu = 0.3 one by 6 %):
  the x extent grows 3-15 % over rest.
- Release: after the plate rises 2 cm (1 s), the cube's height is back within 3 mm of its rest height.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

PLATE_SPEED = 0.02
SMOOTH = 5


def _height(pts: np.ndarray) -> np.ndarray:
    return np.nanmax(pts[..., 2], axis=-1) - np.nanmin(pts[..., 2], axis=-1)


def _width(pts: np.ndarray) -> np.ndarray:
    return np.nanmax(pts[..., 0], axis=-1) - np.nanmin(pts[..., 0], axis=-1)


def _smooth(f: np.ndarray) -> np.ndarray:
    kernel = np.ones(SMOOTH) / SMOOTH
    return np.stack([np.convolve(f[:, k], kernel, mode="valid") for k in range(f.shape[1])], axis=1)


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    pts = r.phase("rest").probe["soft_points"]
    last = pts[-1]
    return [c.within("rest_height", _height(last), 0.095, 0.101, 2),
            c.within("rest_base_on_ground", np.nanmin(last[..., 2], axis=-1), -0.004, 0.006, 2),
            c.within("rest_width", _width(last), 0.095, 0.105, 2),
            c.at_least("soft_points", float(pts.shape[2]), 100, 2)]


def _press_checks(r: c.Rollout) -> list[c.Check]:
    ph, rest = r.phase("act_down"), r.phase("rest").probe["soft_points"][-1]
    pts, plate = ph.probe["soft_points"], ph.probe["plate_z"]
    force = np.abs(ph.probe["plate_force"])
    secs = float(ph.t[-1] - ph.t[0])
    travel = plate[0] - plate[-1]
    squeeze = _height(rest) - _height(pts[-1])
    grow = _width(pts[-1]) / _width(rest) - 1.0
    f_end = force[-SMOOTH:].mean(axis=0)
    sm = _smooth(force)
    drop = (np.maximum.accumulate(sm, axis=0) - sm).max(axis=0)
    return [c.within("press_plate_travel", travel, 0.9 * PLATE_SPEED * secs, 1.1 * PLATE_SPEED * secs, 2),
            c.within("press_cube_follows", squeeze - travel, -0.004, 0.004, 2),
            c.within("press_bulge", grow, 0.03, 0.15, 2),
            c.within("press_force_end", f_end, 50.0, 400.0, 2),
            c.at_most("press_force_monotonic", float((drop / np.maximum(f_end, 1e-9)).max()), 0.05, 2)]


def _release_checks(r: c.Rollout) -> list[c.Check]:
    rest = _height(r.phase("rest").probe["soft_points"][-1])
    end = _height(r.phase("act_settle").probe["soft_points"][-1])
    up = r.phase("act_up").probe["plate_z"]
    return [c.within("release_height", end - rest, -0.003, 0.003, 2),
            c.within("release_plate_travel", up[-1] - up[0], 0.9 * PLATE_SPEED, 1.1 * PLATE_SPEED, 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request)
    checks += _rest_checks(rollout) + _press_checks(rollout) + _release_checks(rollout)
    return c.finish(checks)
