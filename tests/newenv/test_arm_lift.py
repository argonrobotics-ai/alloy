# SPDX-License-Identifier: Apache-2.0
"""E19 checker on synthetic rollouts: a good lift passes; a frozen arm, a teleported cube, no lift fail."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import arm_lift
from bench.newenv.checkers import common as c

N, DT = 4, 0.02


def _arrays(steps: int, cube_z, joint1=None, width=None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    xy = rng.uniform(-0.05, 0.05, (1, N, 2)) + np.array([0.5, 0.0])
    z = np.broadcast_to(np.asarray(cube_z, float).reshape(-1, 1), (steps, N))
    cube = np.concatenate([np.broadcast_to(xy, (steps, N, 2)), z[..., None]], axis=-1)
    q = np.zeros((steps, N, 7))
    if joint1 is not None:
        q[:, :, 0] = np.asarray(joint1).reshape(-1, 1)
    w = np.full((steps, N), 0.08) if width is None else np.broadcast_to(np.asarray(width).reshape(-1, 1), (steps, N))
    return {"cube_pos": cube, "ee_pos": cube.copy(), "joint_pos": q, "gripper_width": w, "table_z": np.zeros((steps, N))}


def _phase(name: str, steps: int, **kw) -> c.Phase:
    return c.make_phase(name, np.arange(steps) * DT, _arrays(steps, **kw))


def _rollout(lift: bool = True, turn: float = 0.5, cube_rest: float = 0.02, grip: bool = True) -> c.Rollout:
    steps = 250
    t = np.arange(steps) * DT
    z = np.where(t > 1.0, 0.2, 0.02) if lift else np.full(steps, 0.02)
    width = np.where((t > 1.5) & (t < 3.0), 0.0, 0.08) if grip else None
    policy = c.make_phase("policy", t, _arrays(steps, cube_z=z))
    j = np.minimum(np.arange(51) * 0.05 * 0.5 / 0.5, turn)
    phases = [_phase("rest", 101, cube_z=np.full(101, cube_rest)),
              c.make_phase("act_joint", np.arange(51) * DT, _arrays(51, np.full(51, 0.02), joint1=j)),
              c.make_phase("act_gripper", np.arange(151) * DT,
                           _arrays(151, np.full(151, 0.02), width=np.where((np.arange(151) * DT > 1.5) & (np.arange(151) * DT < 3.0), 0.0, 0.08) if grip else None)),
              _phase("random", 3, cube_z=np.full(3, 0.02)), policy]
    return c.synthetic(phases, dt=DT, episode_steps=250)


def _failed(r: c.Rollout) -> list[str]:
    res = arm_lift.check(r, {})
    return [x.name for x in res.checks if not x.ok and x.level >= 2]


def test_good_run_passes() -> None:
    assert _failed(_rollout()) == []


def test_no_lift_fails_policy() -> None:
    assert _failed(_rollout(lift=False)) == ["policy_lift_hold"]


def test_frozen_joint_fails_response() -> None:
    assert "joint1_turn" in _failed(_rollout(turn=0.0))


def test_cube_in_the_air_fails_rest() -> None:
    assert "rest_cube_height" in _failed(_rollout(cube_rest=0.2))


def test_gripper_that_never_moves_fails() -> None:
    assert "gripper_close" in _failed(_rollout(grip=False))
