# SPDX-License-Identifier: Apache-2.0
"""E07 checker on synthetic rollouts from a numpy double pendulum: right physics passes, wrong ones fail."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import double_pendulum as dp

N, DT = 16, 0.01
L, M = 0.3, 0.5


def _accel(t1, t2, w1, w2, tau, g, damping):
    a, d, h = 4 / 3 * M * L ** 2, M * L ** 2 / 3, 0.5 * M * L ** 2
    delta = t1 - t2
    mat = np.stack([np.stack([np.full_like(t1, a), h * np.cos(delta)], -1),
                    np.stack([h * np.cos(delta), np.full_like(t1, d)], -1)], -2)
    rhs = np.stack([-tau - h * np.sin(delta) * w2 ** 2 - 1.5 * M * g * L * np.sin(t1) - damping * w1,
                    tau + h * np.sin(delta) * w1 ** 2 - 0.5 * M * g * L * np.sin(t2) - damping * w2], -1)
    return np.linalg.solve(mat, rhs[..., None])[..., 0]


def _rk4(t1, t2, w1, w2, tau, g, damping, h):
    def f(s):
        k = _accel(*s, tau, g, damping)
        return np.stack([s[2], s[3], k[:, 0], k[:, 1]])
    y = np.stack([t1, t2, w1, w2])
    k1 = f(y)
    k2 = f(y + h / 2 * k1)
    k3 = f(y + h / 2 * k2)
    k4 = f(y + h * k3)
    return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _rows(steps: int, tau: float, g: float = 9.81, damping: float = 0.0, tip_scale: float = 1.0):
    rng = np.random.default_rng(1)
    t1, t2 = rng.uniform(-0.5, 0.5, N), rng.uniform(-0.5, 0.5, N)           # absolute angles
    w1, w2 = np.zeros(N), np.zeros(N)
    q, qd, tip = [], [], []
    for _ in range(steps + 1):
        q.append(np.stack([t1, t2 - t1], 1))
        qd.append(np.stack([w1, w2 - w1], 1))
        x = L * (np.sin(t1) + np.sin(t2)) * tip_scale
        z = 1.0 - L * (np.cos(t1) + np.cos(t2)) * tip_scale
        tip.append(np.stack([x, np.zeros(N), z], 1))
        for _ in range(10):
            t1, t2, w1, w2 = _rk4(t1, t2, w1, w2, tau, g, damping, DT / 10)
    return {"joint_pos": np.array(q), "joint_vel": np.array(qd), "tip_pos": np.array(tip)}


def _rollout(torque_scale: float = 1.0, **kw) -> c.Rollout:
    rest = c.make_phase("rest", np.arange(201) * DT, _rows(200, 0.0, **kw))
    act = c.make_phase("act_torque", np.arange(31) * DT, _rows(30, 2.0 * torque_scale, **kw))
    return c.synthetic([rest, act], dt=DT, episode_steps=500)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in dp.check(r, {}).checks if not x.ok and x.level == 2]


def test_correct_physics_passes() -> None:
    assert _failed(_rollout()) == []


def test_damped_hinges_fail_energy_drift() -> None:
    assert "rest_energy_drift" in _failed(_rollout(damping=0.01))


def test_wrong_torque_fails_work() -> None:
    # a rollout driven with a quarter of the torque: energy gain is not 2 N m x the elbow angle
    rest = c.make_phase("rest", np.arange(201) * DT, _rows(200, 0.0))
    probe = _rows(30, 2.0)
    probe["joint_vel"] = probe["joint_vel"] * 1.0
    act = c.make_phase("act_torque", np.arange(31) * DT, _rows(30, 0.5))
    r = c.synthetic([rest, act], dt=DT, episode_steps=500)
    assert "act_work_equals_torque_x_elbow_angle" in _failed(r)


def test_scaled_tip_fails_fk() -> None:
    assert "tip_matches_fk" in _failed(_rollout(tip_scale=1.2))
