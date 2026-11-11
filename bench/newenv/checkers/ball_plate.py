# SPDX-License-Identifier: Apache-2.0
"""E02 ball on a tilting plate: rest, rolling acceleration on a held tilt, plate rate and the 0.3 rad clamp.

Derivations (from the request sentence only):
- Rest: level plate (|rpy| < 5 mrad) and the ball stays within 1 cm of its start for the whole rest phase.
- Act (pitch law: +1 rate until pitch >= 0.1 rad, then hold): a solid ball rolling without slipping on a plane tilted by
  theta accelerates down it at a = g sin(theta) / (1 + I/(m r^2)) = (5/7) g sin(theta) (I = 2/5 m r^2). At 0.1 rad
  that is 0.7 * 9.81 * 0.0998 = 0.70 m/s2 (the request's number). The tilt is about +y, which lowers +x (right-hand
  rule on the pitch of ``plate_rpy``), so the ball accelerates toward +x with a horizontal share cos(theta) ~ 0.995.
  The plate holds at 0.1 + up to half a control step of rate (bang law), so the expected value uses the measured held
  pitch; its own range is checked. Acceleration is the quadratic fit of x(t) after the tilt stopped (2 rows later).
  A start within ~0.1 m of the +x edge can leave the plate within the 0.55 s phase, so 10 of 16 worlds must agree.
- Rate: pitch (act_tilt) and roll (act_clamp) at action 1 turn at 0.5 rad/s: (angle(0.15 s) - angle(0)) / 0.15 within 10 %.
- Clamp: held +1 roll for 1.4 s needs 0.6 s to reach 0.3 rad; the roll must end in [0.295, 0.305] and never exceed 0.305.
- Starts: the plate is 0.4 m square and the ball 0.03 m radius, so ball starts span at most 0.34 m per axis, and must
  vary across worlds (std > 1 cm).
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

G = 9.81
HOLD_PITCH = 0.1
RATE = 0.5
RATE_WINDOW_S = 0.15
MIN_WORLDS = 10


def _fit_accel(t: np.ndarray, x: np.ndarray) -> float:
    return 2.0 * float(np.polyfit(t - t[0], x, 2)[0])


def _rest_checks(r: c.Rollout) -> list[c.Check]:
    rest = r.phase("rest")
    ball = rest.probe["ball_pos"].astype(np.float64)
    drift = np.linalg.norm(ball - ball[0], axis=-1).max(axis=0)
    return [c.within("rest_ball_drift", drift, 0.0, 0.01, 2),
            c.within("rest_plate_level", np.abs(rest.probe["plate_rpy"]).max(axis=(0, 2)), 0.0, 0.005, 2)]


def _rate_of(angle: np.ndarray, t: np.ndarray) -> np.ndarray:
    k = int(np.searchsorted(t, RATE_WINDOW_S + 1e-6, side="right")) - 1
    return (angle[k] - angle[0]) / max(float(t[k] - t[0]), 1e-9)


def _tilt_accel(phase: c.Phase) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per world: fitted x-acceleration / cos(theta), held mean pitch, fitted y acceleration."""
    t = phase.t.astype(np.float64)
    pitch = phase.probe["plate_rpy"][:, :, 1].astype(np.float64)
    ball = phase.probe["ball_pos"].astype(np.float64)
    ax, th, ay = [], [], []
    for w in range(pitch.shape[1]):
        hit = np.flatnonzero(pitch[:, w] >= HOLD_PITCH - 0.003)
        lo = hit[0] + 2 if hit.size else len(t)
        if len(t) - lo < 6:
            ax.append(np.nan), th.append(np.nan), ay.append(np.nan)
            continue
        theta = float(pitch[lo:, w].mean())
        ax.append(_fit_accel(t[lo:], ball[lo:, w, 0]) / np.cos(theta))
        th.append(theta)
        ay.append(_fit_accel(t[lo:], ball[lo:, w, 1]))
    return np.array(ax), np.array(th), np.array(ay)


def _tilt_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("act_tilt")
    ax, theta, ay = _tilt_accel(p)
    expected = 5.0 / 7.0 * G * np.sin(theta)
    dt = float(r.meta.get("dt", 0.02))
    return [c.within("tilt_hold_angle", theta, HOLD_PITCH, HOLD_PITCH + 0.5 * dt + 0.005, 2, min_worlds=MIN_WORLDS),
            c.within("tilt_accel_ratio", ax / expected, 0.75, 1.25, 2, min_worlds=MIN_WORLDS),
            c.within("tilt_accel_sideways", ay, -0.15, 0.15, 2, min_worlds=MIN_WORLDS),
            c.within("pitch_rate", _rate_of(p.probe["plate_rpy"][:, :, 1], p.t), 0.9 * RATE, 1.1 * RATE, 2),
            c.within("tilt_roll_still", np.abs(p.probe["plate_rpy"][:, :, 0]).max(axis=0), 0.0, 0.01, 2)]


def _clamp_checks(r: c.Rollout) -> list[c.Check]:
    p = r.phase("act_clamp")
    roll = p.probe["plate_rpy"][:, :, 0].astype(np.float64)
    other = np.abs(p.probe["plate_rpy"][:, :, 1:]).max(axis=(0, 2))
    return [c.within("roll_rate", _rate_of(roll, p.t), 0.9 * RATE, 1.1 * RATE, 2),
            c.within("roll_clamp_final", roll[-1], 0.295, 0.305, 2),
            c.within("roll_never_over", roll.max(axis=0), 0.0, 0.305, 2),
            c.within("clamp_other_axes_still", other, 0.0, 0.01, 2)]


def _start_checks(r: c.Rollout) -> list[c.Check]:
    ball = r.phase("random").probe["ball_pos"][0].astype(np.float64)
    return [c.at_least("start_spread", float(ball[:, :2].std(axis=0).min()), 0.01, 2),
            c.at_most("start_extent", float(np.ptp(ball[:, :2], axis=0).max()), 0.34, 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request)
    checks += _rest_checks(rollout) + _tilt_checks(rollout) + _clamp_checks(rollout) + _start_checks(rollout)
    return c.finish(checks)
