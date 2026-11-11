# SPDX-License-Identifier: Apache-2.0
"""E21 checker on synthetic rollouts: a good pull passes; a drawer that never opens, one that opens with the hand far
away, a handle that does not follow, and a drawer past its rail fail at the right check."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import arm_drawer
from bench.newenv.checkers import common as c

N, DT = 4, 0.02
HANDLE0 = np.array([0.56, 0.0, 0.10])


def _arrays(steps: int, opening, hand_far=False, handle_scale=1.0, joint1=None, width=None) -> dict[str, np.ndarray]:
    d = np.broadcast_to(np.asarray(opening, float).reshape(-1, 1), (steps, N))
    handle = HANDLE0 + np.stack([-handle_scale * d, np.zeros_like(d), np.zeros_like(d)], axis=-1)
    ee = handle + (np.array([0.0, 0.6, 0.0]) if hand_far else 0.0)
    q = np.zeros((steps, N, 7))
    if joint1 is not None:
        q[:, :, 0] = np.asarray(joint1).reshape(-1, 1)
    w = np.full((steps, N), 0.08) if width is None else np.broadcast_to(np.asarray(width).reshape(-1, 1), (steps, N))
    return {"drawer_open": d, "handle_pos": handle, "ee_pos": ee, "joint_pos": q, "gripper_width": w}


def _phase(name: str, steps: int, **kw) -> c.Phase:
    return c.make_phase(name, np.arange(steps) * DT, _arrays(steps, **kw))


def _rollout(opens: bool = True, turn: float = 0.5, hand_far: bool = False, handle_scale: float = 1.0,
             rest_open: float = 0.0, overshoot: float = 0.25) -> c.Rollout:
    steps = 250
    t = np.arange(steps) * DT
    opening = np.clip((t - 1.0) * 0.1, 0.0, overshoot) if opens else np.zeros(steps)
    pull = c.make_phase("policy", t, _arrays(steps, opening, hand_far=hand_far, handle_scale=handle_scale))
    j = np.minimum(np.arange(51) * 0.05, turn)
    tg = np.arange(151) * DT
    grip = np.where((tg > 1.5) & (tg < 3.0), 0.0, 0.08)
    phases = [_phase("rest", 101, opening=np.full(101, rest_open)),
              c.make_phase("act_joint", np.arange(51) * DT, _arrays(51, np.zeros(51), joint1=j)),
              c.make_phase("act_gripper", tg, _arrays(151, np.zeros(151), width=grip)),
              _phase("random", 3, opening=np.zeros(3)), pull]
    return c.synthetic(phases, dt=DT, episode_steps=250)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in arm_drawer.check(r, {"policy": False}).checks if not x.ok and x.level >= 2]


def test_good_run_passes() -> None:
    assert _failed(_rollout()) == []


def test_drawer_that_never_opens_fails_policy_only() -> None:
    assert _failed(_rollout(opens=False)) == ["handle_moves_with_drawer", "policy_drawer_open"]


def test_drawer_opening_with_hand_far_fails_contact_law() -> None:
    failed = _failed(_rollout(hand_far=True))
    assert "drawer_moves_only_near_hand" in failed and "drawer_moves_only_with_arm_near" in failed


def test_handle_that_does_not_follow_fails() -> None:
    assert "handle_one_to_one" in _failed(_rollout(handle_scale=0.5))


def test_drawer_past_its_rail_fails_travel() -> None:
    assert "drawer_travel" in _failed(_rollout(overshoot=0.4))


def test_drawer_open_at_rest_fails() -> None:
    assert "rest_drawer_closed" in _failed(_rollout(rest_open=0.01))
