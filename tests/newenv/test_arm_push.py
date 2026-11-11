# SPDX-License-Identifier: Apache-2.0
"""E20 checker on synthetic rollouts: a good push passes; frozen joints, no push, a teleported cube, a cube shoved with
no hand near it and a cube left tilted fail at the right check."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import arm_push
from bench.newenv.checkers import common as c

N, DT = 4, 0.02
START = np.array([[0.45, 0.0], [0.5, 0.05], [0.55, -0.05], [0.5, -0.08]])
TARGET = START + np.array([0.2, 0.0])


def _arrays(steps: int, cube_xy=None, cube_z=0.03, ee=None, joint=None, target=TARGET) -> dict[str, np.ndarray]:
    xy = np.broadcast_to(START, (steps, N, 2)) if cube_xy is None else cube_xy
    cube = np.concatenate([xy, np.broadcast_to(np.asarray(cube_z, float).reshape(-1, 1, 1), (steps, N, 1))], axis=-1)
    tgt = np.concatenate([target, np.zeros((N, 1))], axis=1)
    q = np.zeros((steps, N, 7)) if joint is None else joint
    return {"cube_pos": cube, "target_pos": np.broadcast_to(tgt, (steps, N, 3)), "ee_pos": cube.copy() if ee is None else ee,
            "joint_pos": q, "table_z": np.zeros((steps, N))}


def _turn(steps: int, col: int, amount: float) -> np.ndarray:
    q = np.zeros((steps, N, 7))
    q[:, :, col] = np.minimum(np.arange(steps) * 0.05, abs(amount))[:, None] * np.sign(amount)
    return q


def _policy(steps: int, push: bool, shove_far: bool, tilt: bool) -> c.Phase:
    s = np.minimum(np.arange(steps) / 40.0, 1.0)[:, None, None]      # pushed in 40 steps: 5 mm per step
    xy = START + (TARGET - START) * s if push else np.broadcast_to(START, (steps, N, 2)).copy()
    arr = _arrays(steps, cube_xy=xy, cube_z=0.04 if tilt else 0.03)
    if shove_far:                                   # the cube slides though the hand never leaves 0.5 m away
        arr["ee_pos"] = arr["cube_pos"] + np.array([0.5, 0.0, 0.0])
    return c.make_phase("policy", np.arange(steps) * DT, arr)


def _rollout(push=True, turn=0.5, shove_far=False, tilt=False, teleport=False) -> c.Rollout:
    rest = c.make_phase("rest", np.arange(101) * DT, _arrays(101))
    if teleport:                                    # the cube jumps 10 cm in the middle of the rest, hand far away
        cube = rest.probe["cube_pos"].copy()
        cube[50:, :, 0] += 0.1
        rest.probe["cube_pos"] = cube
        rest.probe["ee_pos"] = cube + np.array([0.5, 0.0, 0.0], np.float32)
    j1 = c.make_phase("act_joint", np.arange(51) * DT, _arrays(51, joint=_turn(51, 0, turn)))
    j4 = c.make_phase("act_joint4", np.arange(51) * DT, _arrays(51, joint=_turn(51, 3, -turn)))
    rand = c.make_phase("random", np.arange(5) * DT, _arrays(5))
    return c.synthetic([rest, j1, j4, rand, _policy(251, push, shove_far, tilt)], dt=DT, episode_steps=250)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in arm_push.check(r, {"policy": False}).checks if not x.ok and x.level >= 2]


def test_good_run_passes() -> None:
    # the policy phase pushes with the hand next to the cube (ee == cube centre in the synthetic data)
    assert _failed(_rollout()) == []


def test_no_push_fails_policy_only() -> None:
    assert _failed(_rollout(push=False)) == ["policy_push_to_target"]


def test_frozen_joints_fail_response() -> None:
    assert set(_failed(_rollout(turn=0.0))) >= {"joint1_turn", "joint4_turn"}


def test_cube_moved_with_no_hand_near_fails_contact_law() -> None:
    assert "cube_moves_only_near_hand" in _failed(_rollout(shove_far=True))


def test_teleported_cube_fails() -> None:
    failed = _failed(_rollout(teleport=True))
    assert "cube_moves_only_near_hand" in failed and "rest_cube_still" in failed


def test_tilted_cube_fails_policy() -> None:
    assert "policy_push_to_target" in _failed(_rollout(tilt=True))
