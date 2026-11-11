# SPDX-License-Identifier: Apache-2.0
"""E10 wheeled cart: rest, straight drive (acceleration, drift), turn in place.

Derivation. Four wheels of radius r = 0.04 m, torque tau = 0.2 N m each (action +1 on both sides), cart mass m (body
2 kg plus the wheels, whose mass the request does not give: 0 to 0.2 kg each, so m in [2.0, 2.8] kg). With no slip the
traction force is 4 tau / r = 20 N, so a = 4 tau / (r m) = 2 x 2 x tau / (r m_total): 10 m/s2 for m = 2 kg, 7.1 m/s2
for 2.8 kg (wheel rotational inertia adds to m a little, covered by the tolerance). The request allows 35 %: the
accepted range is [0.65 x 7.14, 1.35 x 10] = [4.6, 13.5] m/s2. Acceleration is the least-squares a of s = a t^2 / 2
for the displacement along the initial heading (cart_yaw at the phase's first row) over the first 0.3 s, before the
cart moves far; rows from a world's first done on are not used. Lateral drift: the displacement perpendicular to that
heading over the first 0.5 s (or until done) must be < 10 % of the distance travelled along it.
Turn in place: left +1, right -1 torques; the cart's yaw (unwrapped) must change by more than 0.5 rad within the
phase (1 s) and the position must stay within 5 cm of the start over it. The sign (which way it turns) is not tested.
Rest: no action, position within 5 mm and yaw within 0.01 rad over the 2 s rest.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

TAU, RADIUS = 0.2, 0.04
M_LOW, M_HIGH, ACC_TOL = 2.0, 2.8, 0.35
ACC_WINDOW_S, LATERAL_WINDOW_S = 0.3, 0.5
MIN_WORLDS = 14


def acc_range() -> tuple[float, float]:
    a = lambda m: 4 * TAU / (RADIUS * m)
    return (1 - ACC_TOL) * a(M_HIGH), (1 + ACC_TOL) * a(M_LOW)


def first_done(done: np.ndarray) -> np.ndarray:
    """(N,) row of each world's first done, or len(done) when none."""
    return np.where(done.any(axis=0), done.argmax(axis=0), len(done))


def usable(ph: c.Phase, seconds: float) -> np.ndarray:
    """(T, N) rows with t <= seconds and before the world's first done."""
    rows = np.arange(len(ph.t))[:, None]
    return (ph.t[:, None] <= seconds + 1e-9) & (rows < first_done(ph.done)[None, :])


def along_lateral(ph: c.Phase, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Displacement from row 0 along / across the initial heading, (T, N); rows outside ``mask`` are NaN."""
    pos, yaw = ph.probe["cart_pos"], ph.probe["cart_yaw"][0]
    d = (pos - pos[0:1])[..., :2]
    head = np.stack([np.cos(yaw), np.sin(yaw)], axis=-1)
    along = (d * head[None]).sum(-1)
    lateral = d[..., 0] * -head[None, :, 1] + d[..., 1] * head[None, :, 0]
    return np.where(mask, along, np.nan), np.where(mask, lateral, np.nan)


def fitted_acceleration(t: np.ndarray, along: np.ndarray) -> np.ndarray:
    """(N,) a minimizing sum (s - a t^2 / 2)^2 over the non-NaN rows."""
    basis = 0.5 * t[:, None] ** 2 * np.isfinite(along)
    s = np.nan_to_num(along)
    den = (basis ** 2).sum(0)
    return np.where(den > 0, (basis * s).sum(0) / np.maximum(den, 1e-12), np.nan)


def _unwrapped_yaw_change(ph: c.Phase, mask: np.ndarray) -> np.ndarray:
    yaw = np.unwrap(ph.probe["cart_yaw"].astype(np.float64), axis=0)
    last = np.where(mask.any(0), (mask * np.arange(len(ph.t))[:, None]).max(0), 0)
    return yaw[last, np.arange(yaw.shape[1])] - yaw[0]


def _forward_checks(r: c.Rollout) -> list[c.Check]:
    ph = r.phase("act_forward")
    a_mask, l_mask = usable(ph, ACC_WINDOW_S), usable(ph, LATERAL_WINDOW_S)
    along, _ = along_lateral(ph, a_mask)
    acc = fitted_acceleration(ph.t, along)
    along_l, lateral = along_lateral(ph, l_mask)
    dist = np.nanmax(np.where(l_mask, along_l, -np.inf), axis=0)
    drift = np.nanmax(np.abs(lateral), axis=0) / np.maximum(np.abs(dist), 1e-9)
    lo, hi = acc_range()
    return [c.within("forward_acceleration", acc, lo, hi, 2, MIN_WORLDS),
            c.within("forward_lateral_drift", drift, 0.0, 0.10, 2, MIN_WORLDS)]


def _spin_checks(r: c.Rollout) -> list[c.Check]:
    ph = r.phase("act_spin")
    mask = usable(ph, ph.t[-1])
    yaw = np.abs(_unwrapped_yaw_change(ph, mask))
    pos = ph.probe["cart_pos"]
    moved = np.where(mask, np.linalg.norm((pos - pos[0:1])[..., :2], axis=-1), 0.0).max(0)
    return [c.within("spin_yaw_change", yaw, 0.5, np.inf, 2, MIN_WORLDS),
            c.within("spin_position_change", moved, 0.0, 0.05, 2, MIN_WORLDS)]


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    pos, yaw = rest.probe["cart_pos"], np.unwrap(rest.probe["cart_yaw"].astype(np.float64), axis=0)
    moved = np.linalg.norm(pos - pos[0:1], axis=-1).max(0)
    return [c.within("rest_position", moved, 0.0, 0.005, 2), c.within("rest_yaw", np.abs(yaw - yaw[0]).max(0), 0.0, 0.01, 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request) + _rest_checks(rollout) + _forward_checks(rollout)
    return c.finish(checks + _spin_checks(rollout))
