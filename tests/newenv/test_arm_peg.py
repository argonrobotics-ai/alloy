# SPDX-License-Identifier: Apache-2.0
"""E22 checker on synthetic rollouts: a good insertion passes; a peg that never goes in, one pushed through the block's
top, a peg that slips out of the hand, a frozen arm and one that ends off the axis fail at the right check."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import arm_peg
from bench.newenv.checkers import common as c

N, DT = 4, 0.02
HOLE = np.array([[0.48, -0.01, 0.08], [0.52, 0.01, 0.08], [0.5, 0.02, 0.08], [0.5, -0.02, 0.08]])
TOP = 0.08


def _arrays(steps: int, tip_z, tip_off=0.0, slip=0.0, joint1=None) -> dict[str, np.ndarray]:
    """Peg vertical with identity quaternion (length along local z); the tip at tip_z (rows) and tip_off from the axis."""
    z = np.broadcast_to(np.asarray(tip_z, float).reshape(-1, 1), (steps, N)) + arm_peg.HALF_LEN
    xy = HOLE[None, :, :2] + np.array([tip_off, 0.0])
    peg = np.concatenate([np.broadcast_to(xy, (steps, N, 2)), z[..., None]], axis=-1)
    ee = peg + np.array([0.0, 0.0, 0.01]) + np.linspace(0.0, slip, steps)[:, None, None] * np.array([1.0, 0, 0])
    q = np.zeros((steps, N, 7))
    if joint1 is not None:
        q[:, :, 0] = np.asarray(joint1).reshape(-1, 1)
    quat = np.broadcast_to(np.array([1.0, 0, 0, 0]), (steps, N, 4))
    return {"peg_pos": peg, "peg_quat": quat, "hole_pos": np.broadcast_to(HOLE, (steps, N, 3)), "ee_pos": ee,
            "joint_pos": q}


def _phase(name: str, steps: int, **kw) -> c.Phase:
    return c.make_phase(name, np.arange(steps) * DT, _arrays(steps, **kw))


def _rollout(insert: bool = True, tip_off: float = 0.0, turn: float = 0.2, slip: float = 0.0,
             through: bool = False, rest_slip: float = 0.0) -> c.Rollout:
    steps = 300
    z = np.concatenate([np.full(100, TOP + 0.05), np.linspace(TOP + 0.05, TOP - 0.04, 150), np.full(50, TOP - 0.04)])
    tip_z = z if insert else np.full(steps, TOP + 0.05)
    off = 0.02 if through else tip_off                       # off the hole axis, over the block, deep: through the top
    policy = c.make_phase("policy", np.arange(steps) * DT, _arrays(steps, tip_z=tip_z, tip_off=off))
    j = np.minimum(np.arange(51) * 0.02, turn)
    phases = [_phase("rest", 101, tip_z=np.full(101, TOP + 0.05), slip=rest_slip),
              _phase("act_joint", 51, tip_z=np.full(51, TOP + 0.05), joint1=j, slip=slip),
              _phase("random", 3, tip_z=np.full(3, TOP + 0.05)), policy]
    return c.synthetic(phases, dt=DT, episode_steps=300)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in arm_peg.check(r, {"policy": False}).checks if not x.ok and x.level >= 2]


def test_good_run_passes() -> None:
    assert _failed(_rollout()) == []


def test_peg_that_never_goes_in_fails_policy_only() -> None:
    assert _failed(_rollout(insert=False)) == ["policy_peg_inserted"]


def test_peg_through_the_block_fails() -> None:
    failed = _failed(_rollout(through=True))
    assert "peg_not_through_block" in failed and "policy_peg_inserted" in failed


def test_peg_off_axis_at_the_end_fails_policy() -> None:
    assert "policy_peg_inserted" in _failed(_rollout(tip_off=0.005))


def test_peg_slipping_in_the_hand_fails_rest() -> None:
    assert "rest_peg_in_hand" in _failed(_rollout(rest_slip=0.01))


def test_frozen_joint_fails_response() -> None:
    assert "joint1_turn" in _failed(_rollout(turn=0.0))


def test_peg_slipping_during_the_move_fails() -> None:
    assert "peg_follows_hand" in _failed(_rollout(slip=0.05))
