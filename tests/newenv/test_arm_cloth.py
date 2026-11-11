# SPDX-License-Identifier: Apache-2.0
"""E24 checker on synthetic cloth: a clean pull passes; a frozen scene, a dragged cube, a 2x stretched sheet and a cloth
that never leaves the cube fail at the right check."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import arm_cloth, common as c

N, G, DT = 4, 21, 0.02
CUBE_C = np.array([0.45, 0.0, 0.03])


def _sheet(cx: float, cy: float, stretch: float = 1.0, draped: bool = True) -> np.ndarray:
    """(V, 3) 30 cm sheet centred (cx, cy): over the cube's top when draped, else flat on the table."""
    x, y = np.meshgrid(np.linspace(-0.15, 0.15, G) * stretch, np.linspace(-0.15, 0.15, G) * stretch)
    pts = np.stack([cx + x.ravel(), cy + y.ravel(), np.full(x.size, 0.002)], axis=-1)
    if draped:
        on_top = (np.abs(pts[:, 0] - CUBE_C[0]) < 0.0301) & (np.abs(pts[:, 1] - CUBE_C[1]) < 0.0301)
        pts[on_top, 2] = 0.062
        near = (np.abs(pts[:, 0] - CUBE_C[0]) < 0.04) & (np.abs(pts[:, 1] - CUBE_C[1]) < 0.04) & ~on_top
        pts[near, 0] += np.sign(pts[near, 0] - CUBE_C[0]) * 0.01
    return pts


def _probe(cloth: np.ndarray, cube: np.ndarray, q1: np.ndarray | float = 0.0, width: np.ndarray | float = 0.08) -> dict:
    steps = len(cloth)
    q = np.zeros((steps, N, 7))
    q[..., 0] = q1
    return {"cloth_points": cloth, "cube_pos": cube, "ee_pos": np.zeros((steps, N, 3)), "joint_pos": q,
            "gripper_width": np.broadcast_to(np.asarray(width, np.float64).reshape(-1, 1), (steps, N)).copy(),
            "table_z": np.zeros((steps, N))}


def _tile(arr: np.ndarray, steps: int) -> np.ndarray:
    return np.broadcast_to(arr, (steps, N) + arr.shape).copy()


def _rollout(frozen: bool = False, drag: bool = False, stretch: float = 1.0, teleport: bool = False,
             flat_rest: bool = False) -> c.Rollout:
    cube = np.broadcast_to(CUBE_C, (101, N, 3)).copy()
    rest = c.make_phase("rest", np.arange(101) * DT, _probe(_tile(_sheet(0.45, 0.0, stretch, not flat_rest), 101), cube))
    ramp = np.minimum(np.arange(51) * 0.05, 0.5)[:, None]
    jpts = _tile(_sheet(0.45, 0.0, stretch), 51)
    joint = c.make_phase("act_joint", np.arange(51) * DT,
                         _probe(jpts, np.broadcast_to(CUBE_C, (51, N, 3)).copy(), 0.0 if frozen else ramp))
    wid = np.where(np.arange(151) * DT < 1.5, 0.0, 0.08)
    grip = c.make_phase("act_gripper", np.arange(151) * DT, _probe(_tile(_sheet(0.45, 0.0, stretch), 151),
                        np.broadcast_to(CUBE_C, (151, N, 3)).copy(), width=np.concatenate([[0.08], wid[1:]])))
    steps = 201
    cy = np.linspace(0.0, -0.30, steps)
    cloth = np.stack([_sheet(0.45, 0.0 if frozen else y, stretch, draped=frozen) for y in cy])
    cloth = np.broadcast_to(cloth[:, None], (steps, N) + cloth.shape[1:]).copy()
    cube_p = np.broadcast_to(CUBE_C, (steps, N, 3)).copy()
    if drag:
        cube_p[:, :, 1] = np.linspace(0.0, 0.05, steps)[:, None]
    if teleport:
        cloth[:, :, :, 2] = 5.0
    done = np.zeros((steps, N), bool)
    done[-1] = True
    pol = c.make_phase("policy", np.arange(steps) * DT, _probe(cloth, cube_p), done=done)
    return c.synthetic([rest, joint, grip, pol], dt=DT, episode_steps=steps - 1)


def _failed(r: c.Rollout) -> list[str]:
    generic = {"reward_success", "policy_success_flag"}
    return [x.name for x in arm_cloth.check(r, {"policy": True}).checks if not x.ok and x.level >= 2
            and x.name not in generic]


def test_good_pull_passes() -> None:
    assert _failed(_rollout()) == []


def test_cloth_that_never_leaves_the_cube_fails_policy() -> None:
    assert "policy_cloth_clear" in _failed(_rollout(frozen=True))


def test_dragged_cube_fails_policy() -> None:
    assert "policy_cube_still" in _failed(_rollout(drag=True))


def test_stretched_cloth_fails_stretch() -> None:
    assert "cloth_stretch" in _failed(_rollout(stretch=2.0))


def test_flat_cloth_at_rest_does_not_cover() -> None:
    assert "rest_covers_cube" in _failed(_rollout(flat_rest=True))


def test_frozen_arm_fails_joint_response() -> None:
    assert "joint1_turn" in _failed(_rollout(frozen=True))


def test_teleported_cloth_fails_policy() -> None:
    assert "policy_sheet_intact" in _failed(_rollout(teleport=True))
