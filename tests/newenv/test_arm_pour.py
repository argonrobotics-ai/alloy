# SPDX-License-Identifier: Apache-2.0
"""E23 checker on synthetic rollouts: a pour passes; a frozen scene, a slipped cup, a miss and a leak fail."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import arm_pour, common as c

N, P, DT = 4, 300, 0.02
PV = 150e-6 / P
BOWL = np.array([0.40, 0.0, 0.0])
EE0 = np.array([0.336, 0.0, 0.215])


def _water(rng: np.random.Generator, centre: np.ndarray, radius: float, z0: float, dz: float) -> np.ndarray:
    ang, rr = rng.uniform(0, 2 * np.pi, P), radius * np.sqrt(rng.uniform(0, 1, P))
    pts = np.stack([centre[0] + rr * np.cos(ang), centre[1] + rr * np.sin(ang), z0 + dz * rng.uniform(0, 1, P)], axis=1)
    return np.broadcast_to(pts, (N, P, 3)).copy()


def _probe(points: np.ndarray, ee: np.ndarray, cup_shift: np.ndarray | float = 0.0, pv: float = PV,
           q0: np.ndarray | None = None) -> dict[str, np.ndarray]:
    steps = len(points)
    cup = ee + np.array([0.0, 0.0, -0.037]) + np.asarray(cup_shift).reshape(-1, 1, 1) * np.array([1.0, 0.0, 0.0])
    q = np.zeros((steps, N, 7)) if q0 is None else q0
    return {"liquid_points": points, "particle_volume": np.full((steps, N), pv), "cup_pos": cup, "ee_pos": ee,
            "bowl_pos": np.broadcast_to(BOWL, (steps, N, 3)), "bowl_radius": np.full((steps, N), 0.09), "joint_pos": q}


def _ee(steps: int) -> np.ndarray:
    return np.broadcast_to(EE0, (steps, N, 3)).copy()


def _rollout(volume_scale: float = 1.0, frozen: bool = False, slip: bool = False, miss: bool = False,
             leak: bool = False, gravity_off: bool = False) -> c.Rollout:
    rng = np.random.default_rng(0)
    cup = _water(rng, EE0 * [1, 1, 0], 0.03, 0.14, 0.05)
    rest_pts = np.broadcast_to(cup, (61, N, P, 3)).copy()
    if leak:
        rest_pts[30:, :, : P // 4, 0] += 0.3
    rest = c.make_phase("rest", np.arange(61) * DT, _probe(rest_pts, _ee(61), pv=PV * volume_scale))
    q = np.zeros((51, N, 7))
    q[:, :, 0] = 0 if frozen else np.minimum(np.arange(51) * 0.05, 0.5)[:, None]
    joint_pts = np.broadcast_to(cup, (51, N, P, 3)).copy()
    act = c.make_phase("act_joint", np.arange(51) * DT, _probe(joint_pts, _ee(51), q0=q))
    steps = 201
    out = _water(rng, BOWL if not miss else BOWL + 0.4, 0.07, 0.01, 0.03)
    if gravity_off:
        out = cup
    held = np.arange(steps) < 100
    frames = np.where(held[:, None, None, None], cup[None], out[None])
    shift = np.where(held, 0.0, 0.0) if not slip else np.linspace(0.0, 0.08, steps)
    done = np.zeros((steps, N), bool)
    done[-1] = True
    pol = c.make_phase("policy", np.arange(steps) * DT, _probe(frames, _ee(steps), shift), done=done)
    return c.synthetic([rest, act, pol], dt=DT, episode_steps=steps - 1)


def _failed(r: c.Rollout) -> list[str]:
    return [x.name for x in arm_pour.check(r, {"policy": True}).checks if not x.ok and x.name.split("_")[0] in
            ("volume", "rest", "joint1", "act", "policy") and x.name != "policy_success_flag"]


def test_good_pour_passes() -> None:
    assert _failed(_rollout()) == []


def test_wrong_volume_fails() -> None:
    assert "volume_ml" in _failed(_rollout(volume_scale=2.0))


def test_leak_at_rest_fails() -> None:
    assert "rest_in_cup" in _failed(_rollout(leak=True))


def test_frozen_arm_fails_joint_response() -> None:
    assert "joint1_turn" in _failed(_rollout(frozen=True))


def test_slipping_cup_fails() -> None:
    assert "policy_cup_no_slide" in _failed(_rollout(slip=True))


def test_missing_the_bowl_fails_policy() -> None:
    assert "policy_water_in_bowl" in _failed(_rollout(miss=True))


def test_water_never_leaving_the_cup_fails_policy() -> None:
    assert "policy_water_in_bowl" in _failed(_rollout(gravity_off=True))
