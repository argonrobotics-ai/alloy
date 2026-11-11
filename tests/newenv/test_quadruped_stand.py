# SPDX-License-Identifier: Apache-2.0
"""E09 checker on synthetic Go2 rollouts: standing, offsets, falls; wrong ones fail at the right check."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c
from bench.newenv.checkers import quadruped_stand as qs

N, DT, EP = 16, 0.02, 500


def _quat(tilt: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(tilt / 2), np.sin(tilt / 2), np.zeros_like(tilt), np.zeros_like(tilt)], -1)


def _phase(name: str, seconds: float, height=0.27, tilt=0.02, shift=0.0, sag=0.0, fall_at=None, reset_height=None,
           drift=0.0) -> c.Phase:
    steps = int(round(seconds / DT))
    t = np.arange(steps + 1) * DT
    z = np.full((steps + 1, N), height) - sag * t[:, None]
    done = np.zeros((steps + 1, N), bool)
    if fall_at is not None:
        z[fall_at:, 0] = 0.1
        z[fall_at, 0] = reset_height if reset_height is not None else 0.27
        z[fall_at + 1:, 0] = 0.27
        done[fall_at, 0] = True
    pos = np.zeros((steps + 1, N, 3))
    pos[..., 0], pos[..., 2] = drift * t[:, None], z
    q = np.zeros((steps + 1, N, 12)) + shift * np.minimum(t / 0.5, 1.0)[:, None, None]
    quat = _quat(np.full((steps + 1, N), tilt))
    return c.make_phase(name, t, {"base_pos": pos, "base_quat": quat, "joint_pos": q}, done=done)


def _rollout(shift=0.28, sag=0.0, fall_at=300, reset_height=None, drift=0.0) -> c.Rollout:
    ph = [_phase("rest", 2.0), _phase("stand", 3.0, sag=sag, drift=drift), _phase("act_plus", 1.0, shift=shift),
          _phase("random", 20.0, fall_at=fall_at, reset_height=reset_height)]
    ph[3].done[EP - 1 + 1, 1:] = True                       # the other worlds' time limit
    return c.synthetic(ph, dt=DT, episode_steps=EP)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in qs.check(r, {}).checks if not x.ok and x.level == 2]


def test_correct_behaviour_passes() -> None:
    assert _failed(_rollout()) == []


def test_collapsing_robot_fails_stand() -> None:
    assert "stand_standing_rows" in _failed(_rollout(sag=0.05))


def test_sliding_robot_fails_drift() -> None:
    assert "stand_drift" in _failed(_rollout(drift=0.03))


def test_weak_offset_fails_act() -> None:
    assert "act_joint_change_median" in _failed(_rollout(shift=0.05))


def test_lying_robot_without_done_fails_random() -> None:
    r = _rollout()
    r.phase("random").probe["base_pos"][400:420, 3, 2] = 0.1          # trunk on the ground, no done
    assert "random_unterminated_fall_rows" in _failed(r)


def test_reset_into_a_crouch_fails() -> None:
    assert "random_resets_standing" in _failed(_rollout(reset_height=0.15))
