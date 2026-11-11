# SPDX-License-Identifier: Apache-2.0
"""E02 checker on synthetic rollouts: a good one, a frozen scene, a sliding (not rolling) ball, a slow plate, no clamp."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import ball_plate
from bench.newenv.checkers import common as c

N, DT, G = 16, 0.02, 9.81


def _plate(ramp_rate: float, final: float, axis: int, steps: int) -> np.ndarray:
    t = np.arange(steps) * DT
    ang = np.minimum(ramp_rate * t, final)
    rpy = np.zeros((steps, N, 3))
    rpy[:, :, axis] = ang[:, None]
    return rpy


def _ball(accel_factor: float, rpy: np.ndarray, x0: np.ndarray) -> np.ndarray:
    steps = rpy.shape[0]
    pitch = rpy[:, 0, 1]
    a = accel_factor * G * np.sin(pitch)
    v = np.cumsum(a) * DT
    x = np.cumsum(v) * DT
    pos = np.zeros((steps, N, 3))
    pos[:, :, 0] = x0[None, :] + x[:, None]
    pos[:, :, 1] = np.linspace(-0.1, 0.1, N)[None, :]
    pos[:, :, 2] = 0.03
    return pos


def _rollout(accel_factor: float = 5 / 7, rate: float = 0.5, clamp: float = 0.3, frozen: bool = False) -> c.Rollout:
    x0 = np.linspace(-0.15, 0.0, N)
    rest_n, tilt_n, clamp_n = 101, 28, 71
    zero = np.zeros((rest_n, N, 3))
    ballrest = _ball(0.0, zero, x0)
    rest = c.make_phase("rest", np.arange(rest_n) * DT, {"ball_pos": ballrest, "plate_rpy": zero})
    tilt_rpy = _plate(rate, 0.1, 1, tilt_n)
    ball = _ball(0.0 if frozen else accel_factor, tilt_rpy, x0)
    tilt = c.make_phase("act_tilt", np.arange(tilt_n) * DT, {"ball_pos": ball, "plate_rpy": tilt_rpy})
    clamp_rpy = _plate(rate, clamp, 0, clamp_n)
    cl = c.make_phase("act_clamp", np.arange(clamp_n) * DT, {"ball_pos": _ball(0.0, clamp_rpy, x0), "plate_rpy": clamp_rpy})
    rng = np.random.default_rng(0)
    start = np.concatenate([rng.uniform(-0.15, 0.15, (3, N, 2)), np.full((3, N, 1), 0.03)], axis=-1)
    done = np.zeros((3, N), bool)
    done[1] = True
    rand = c.make_phase("random", np.arange(3) * DT, {"ball_pos": start, "plate_rpy": np.zeros((3, N, 3))}, done=done)
    return c.synthetic([rest, tilt, cl, rand], dt=DT, episode_steps=250)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in ball_plate.check(r, {}).checks if not x.ok and x.level == 2]


def test_good_rollout_checks_out() -> None:
    assert _failed(_rollout()) == []


def test_frozen_ball_fails() -> None:
    assert "tilt_accel_ratio" in _failed(_rollout(frozen=True))


def test_sliding_ball_fails() -> None:
    assert "tilt_accel_ratio" in _failed(_rollout(accel_factor=1.0))


def test_slow_plate_fails_rate() -> None:
    assert "pitch_rate" in _failed(_rollout(rate=0.25))


def test_no_clamp_fails() -> None:
    assert "roll_never_over" in _failed(_rollout(clamp=0.6))
