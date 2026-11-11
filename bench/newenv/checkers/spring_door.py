# SPDX-License-Identifier: Apache-2.0
"""E08 spring door: rest, step responses against the linear model, joint limit, release from 1 rad.

Model (the request): I theta'' = tau - k theta - c theta', k = 2 N m/rad, c = 1 N m s/rad, tau = 3 N m x action,
theta limited to [0, 1.5] (the limit stops the door: theta clamped, rate zeroed when it pushes outward).
Inertia of a uniform 10 kg slab 0.8 m wide hinged on its vertical edge: I = m w^2 / 3 = 2.133 kg m2 (thickness adds
m t^2 / 12 = 0.0005, negligible; the brief's 2.13).
Underdamped: sigma = c / 2I = 0.2344 /s, omega_n = sqrt(k / I) = 0.9682 /s, zeta = 0.242 -> omega_d = 0.940 /s.
Step response to tau0 from (theta0, w0): theta = tau0/k + e^{-sigma t} (A cos wd t + B sin wd t); settling to 5 % needs
ln(20)/sigma = 12.8 s, longer than an episode, so "settles at 0.75 rad within 5 %" cannot be observed: the check is
that the 4 s response follows this model (initial state from the probe) within 0.01 rad + 5 % of the steady value
(0.75 rad for action 0.5). Action 1.0 would settle at tau/k = 1.5 rad = the limit, but the response overshoots to 2.5
rad without a stop: with the stop at 1.5 the door reaches it at about 1.93 s: max angle in [1.45, 1.52].
Release from ~1 rad (the harness law in scripts/spring_door.py drives it there with the door's torque): the model
from the measured start gives the time to fall to half the start angle (about 1.3 s from 1 rad); the measured time
must be within 30 % (the time-constant check), and the angle never goes below -0.02 (the 0 limit; "no overshoot").
The door started below 0.8 rad (the torque is not 3 N m) is a failure of the earlier law, so it fails this check.
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

I_DOOR, K, C, MAX_TORQUE, LIMIT = 2.1333, 2.0, 1.0, 3.0, 1.5
STEP_TOL_ABS, STEP_TOL_REL = 0.01, 0.05
HALF_TIME_TOL = 0.30
MIN_WORLDS = 15
SUB = 40


def simulate(theta0: float, rate0: float, torque: float, times: np.ndarray) -> np.ndarray:
    """Door angle at ``times`` (s, ascending from 0) for a constant torque (N m); semi-implicit steps, limit stops."""
    theta, rate, now, out = theta0, rate0, 0.0, []
    for target in times:
        n = max(1, int(np.ceil((target - now) * 2000)))
        h = (target - now) / n
        for _ in range(n):
            rate += (torque - K * theta - C * rate) / I_DOOR * h
            theta += rate * h
            if theta < 0.0 or theta > LIMIT:
                theta, rate = float(np.clip(theta, 0.0, LIMIT)), 0.0
        now = target
        out.append(theta)
    return np.array(out)


def _step_error(phase: c.Phase, action: float) -> np.ndarray:
    """(N,) worst |measured - model| over the phase's rows."""
    angle, rate = phase.probe["door_angle"], phase.probe["door_rate"]
    errs = np.zeros(angle.shape[1])
    for w in range(angle.shape[1]):
        model = simulate(float(angle[0, w]), float(rate[0, w]), MAX_TORQUE * action, phase.t - phase.t[0])
        errs[w] = np.abs(model - angle[:, w]).max()
    return errs


def _half_time(t: np.ndarray, angle: np.ndarray, level: float) -> float:
    below = np.nonzero(angle <= level)[0]
    return float(t[below[0]]) if below.size else float("inf")


def _release_ratio(ph: c.Phase, op: c.Phase) -> np.ndarray:
    """(N,) measured / model time for the door to fall to half its start angle; NaN when the start is unusable."""
    angle, rate = ph.probe["door_angle"], ph.probe["door_rate"]
    out = np.full(angle.shape[1], np.nan)
    for w in range(angle.shape[1]):
        theta0 = float(angle[0, w])
        if theta0 < 0.8 or op.done[:-1, w].any() or ph.done[:-1, w].any():
            continue
        model = simulate(theta0, float(rate[0, w]), 0.0, ph.t - ph.t[0])
        t_model = _half_time(ph.t, model, theta0 / 2)
        out[w] = _half_time(ph.t, angle[:, w], theta0 / 2) / t_model if np.isfinite(t_model) else np.nan
    return out


def _checks(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest").probe["door_angle"]
    half, full = r.phase("act_half"), r.phase("act_full")
    ratio = _release_ratio(r.phase("release"), r.phase("act_open"))
    err = _step_error(half, 0.5)
    return [c.at_most("rest_closed", float(np.abs(rest).max()), 0.01, 2),
            c.within("act_half_follows_model", err, 0.0, STEP_TOL_ABS + STEP_TOL_REL * 0.75, 2, MIN_WORLDS),
            c.within("act_full_reaches_limit", full.probe["door_angle"].max(axis=0), 1.45, 1.52, 2, MIN_WORLDS),
            c.within("release_half_time_ratio", ratio, 1 - HALF_TIME_TOL, 1 + HALF_TIME_TOL, 2, MIN_WORLDS),
            c.within("release_no_overshoot", -r.phase("release").probe["door_angle"].min(axis=0), -np.inf, 0.02, 2,
                     MIN_WORLDS)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    return c.finish(c.generic_checks(rollout, request) + _checks(rollout))
