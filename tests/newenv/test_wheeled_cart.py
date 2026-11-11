# SPDX-License-Identifier: Apache-2.0
"""E10 checker on synthetic kinematic carts: the right response passes, wrong ones fail at the right check."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import wheeled_cart as wc

N, DT = 16, 0.01
YAW0 = np.linspace(-3.0, 3.0, N)


def _phase(name: str, seconds: float, acc: float = 0.0, side: float = 0.0, spin: float = 0.0,
           wobble: float = 0.0) -> c.Phase:
    t = np.arange(int(round(seconds / DT)) + 1) * DT
    head = np.stack([np.cos(YAW0), np.sin(YAW0)], 1)
    left = np.stack([-np.sin(YAW0), np.cos(YAW0)], 1)
    s = 0.5 * acc * t[:, None] ** 2
    pos = np.zeros((len(t), N, 3))
    pos[..., :2] = (s * 1.0)[..., None] * head[None] + (side * s)[..., None] * left[None] + wobble * np.sin(40 * t)[:, None, None]
    pos[..., 2] = 0.05
    yaw = YAW0[None] + spin * t[:, None]
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return c.make_phase(name, t, {"cart_pos": pos, "cart_yaw": yaw, "goal_pos": np.zeros((len(t), N, 3))})


def _rollout(acc: float = 8.3, side: float = 0.0, spin: float = 2.0, spin_wobble: float = 0.0) -> c.Rollout:
    return c.synthetic([_phase("rest", 2.0), _phase("act_forward", 1.0, acc=acc, side=side),
                        _phase("act_spin", 1.0, spin=spin, wobble=spin_wobble)], dt=DT, episode_steps=500)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in wc.check(r, {}).checks if not x.ok and x.level == 2]


def test_correct_response_passes() -> None:
    assert _failed(_rollout()) == []


def test_double_torque_fails_acceleration() -> None:
    assert "forward_acceleration" in _failed(_rollout(acc=20.0))


def test_crabbing_fails_lateral_drift() -> None:
    assert "forward_lateral_drift" in _failed(_rollout(side=0.2))


def test_no_turn_fails_spin() -> None:
    assert "spin_yaw_change" in _failed(_rollout(spin=0.2))


def test_turn_with_translation_fails_position() -> None:
    assert "spin_position_change" in _failed(_rollout(spin_wobble=0.1))
