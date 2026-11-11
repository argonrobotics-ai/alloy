# SPDX-License-Identifier: Apache-2.0
"""bench/newenv/checkers/common.py: loader, helpers, levels and the generic checks on synthetic rollouts."""
from __future__ import annotations

import json

import numpy as np
import pytest

from bench.newenv.checkers import common as c

N, T = 4, 41


def _rollout(**meta) -> c.Rollout:
    """Two phases of a slider that starts at a per-world x and resets every 20 steps in 'random'."""
    t = np.arange(T) * 0.01
    start = np.linspace(-0.1, 0.1, N)
    x = start[None, :] + 0.01 * (np.arange(T) % 20)[:, None] * (1 + np.arange(N))[None, :]
    done = np.zeros((T, N), bool)
    done[[20, 40]] = True
    obs = x[:, :, None].astype(np.float32)
    reward = np.broadcast_to(np.linspace(0.0, 1.0, T)[:, None], (T, N)).copy()
    success = np.zeros((T, N), bool)
    success[-1] = True
    rest = c.make_phase("rest", t, {"x": np.broadcast_to(start, (T, N))}, obs=np.broadcast_to(start, (T, N))[:, :, None])
    rand = c.make_phase("random", t, {"x": x}, obs=obs, done=done, reward=reward, success=success)
    return c.synthetic([rest, rand], **meta)


def _failed(rollout: c.Rollout, request: dict | None = None) -> list[str]:
    return c.finish(c.generic_checks(rollout, request or {"probe": {"x": []}})).failed


def test_good_rollout_passes_generic_checks():
    result = c.finish(c.generic_checks(_rollout(), {"probe": {"x": []}}))
    assert result.failed == [] and result.level == 2
    assert json.dumps(result.to_dict())                   # plain types only


def test_generic_checks_catch_each_fault():
    r = _rollout()
    r.phase("random").done[:] = False
    r.meta["phases"]["random"]["done_count"] = [0] * N
    assert {"done_fires", "resets"} <= set(_failed(r))

    r = _rollout()
    r.phase("random").probe["x"][20] = 5.0                # "reset" lands far outside the start range
    assert _failed(r) == ["resets"]

    r = _rollout()
    r.phase("random").obs[:] = 0.3                        # every world identical
    assert _failed(r) == ["worlds_differ"]

    r = _rollout()
    r.phase("random").obs[7, 1, 0] = np.inf
    assert _failed(r) == ["finite"]

    r = _rollout()
    r.phase("random").reward[-1] = -1.0                   # success pays less than the start
    assert _failed(r) == ["reward_success"] and c.finish(c.generic_checks(r, {})).level == 1

    assert _failed(_rollout(seeds={"seed0_repeat_max_err": 0.02, "start_world_spread": 0.1,
                                   "seed1_differs": True})) == ["seed_repeat"]
    assert _failed(_rollout(seeds={"seed0_repeat_max_err": 0.0, "start_world_spread": 0.1,
                                   "seed1_differs": False})) == ["seed_differs"]
    assert _failed(_rollout(seeds={"seed0_repeat_max_err": 0.0, "start_world_spread": 0.0,
                                   "seed1_differs": False})) == []   # a fixed start has nothing to vary
    assert _failed(_rollout(violations=["reward: shape (4, 1)"])) == ["shapes"]
    assert "completed" in _failed(_rollout(completed=False))
    assert "policy_success_flag" in _failed(_rollout(policy={"success_rate": 0.5}), {"policy": True})


def test_empty_rollout_is_level_0_without_raising():
    result = c.finish(c.generic_checks(c.Rollout({}, {}), {"probe": {"x": []}}))
    assert result.level == 0 and "completed" in result.failed


def test_levels():
    mk = lambda ok, level: c.Check(f"l{level}", ok, 0, "", level)  # noqa: E731
    assert c.finish([mk(True, 1), mk(True, 2)]).level == 2         # no level-3 check: 2 is the top
    assert c.finish([mk(True, 1), mk(True, 2), mk(True, 3)]).level == 3
    assert c.finish([mk(True, 1), mk(False, 2), mk(True, 3)]).level == 1
    assert c.finish([mk(False, 1), mk(True, 2)]).level == 0


def test_load_round_trip_and_run_checker(tmp_path):
    r = _rollout()
    flat = {}
    for p in r.phases.values():
        flat.update({f"{p.name}/{f}": getattr(p, f) for f in c.FIELDS})
        flat.update({f"{p.name}/probe/{k}": v for k, v in p.probe.items()})
    np.savez(tmp_path / "rollout.npz", **flat)
    (tmp_path / "result.json").write_text(json.dumps(r.meta))
    back = c.Rollout.load(tmp_path)
    assert list(back.phases) == ["rest", "random"] and back.phase("random").probe["x"].shape == (T, N)
    with pytest.raises(KeyError, match="rest, random"):
        back.phase("act")

    class Good:
        check = staticmethod(lambda rollout, request: c.finish(c.generic_checks(rollout, request)))

    class Broken:
        check = staticmethod(lambda rollout, request: rollout.phase("act"))

    request = {"probe": {"x": []}}
    assert c.run_checker(Good, tmp_path, request)["level"] == 2
    broken = c.run_checker(Broken, tmp_path, request)
    assert broken["level"] == 1 and "KeyError" in broken["checker_error"]
    (tmp_path / "result.json").write_text(json.dumps({"supported": False, "reason": "no liquids"}))
    assert c.run_checker(Good, tmp_path, request)["declined"]


def test_helpers():
    half = np.sqrt(0.5)
    quats = np.array([[1, 0, 0, 0], [half, half, 0, 0], [0, 1, 0, 0], [half, 0, 0, half]], float)
    assert np.allclose(c.tilt_from_quat_wxyz(quats), [0, np.pi / 2, np.pi, 0], atol=1e-6)

    t = np.arange(30) * 0.1
    pos = np.zeros((30, 2, 3))
    pos[:, 0, 0] = 2.0 * t                                # world 0 moves at 2 m/s, world 1 is still
    v = c.speed(pos, t)
    assert v.shape == (30, 2) and np.allclose(v[:, 0], 2.0)
    assert c.settled(v, 0.01).tolist() == [False, True]

    grid = np.stack(np.meshgrid(np.linspace(0, 0.4, 9), np.linspace(0, 0.4, 9), [0.0]), -1).reshape(-1, 3)
    sheet = np.stack([grid, grid * 2.0])[:, None]         # (T=2, N=1, V, 3): second row stretched 2x
    assert np.allclose(c.extent(sheet)[0], 0.4 * np.sqrt(2), rtol=1e-6)
    assert np.allclose(c.stretch(sheet)[:, 0], [1.0, 2.0])
    assert np.allclose(c.stretch(sheet, rest_extent=0.4 * np.sqrt(2))[1], 2.0)
    assert np.isclose(c.polyline_length(np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1]], float)), 2.0)

    pts = np.array([[0.0, 0.0, 0.05], [0.2, 0.0, 0.05], [0.0, 0.0, -0.01], [0.0, 0.029, 0.119]])
    assert c.inside_box(pts, [-0.1, -0.1, 0.0], [0.1, 0.1, 0.1]).tolist() == [True, False, False, False]
    assert c.inside_cylinder(pts, [0, 0, 0], 0.03, 0.12).tolist() == [True, False, False, True]
    assert c.inside_cylinder(pts, [0, 0, 0], 0.03, 0.12, margin=0.02).tolist() == [True, False, True, True]

    liquid = np.zeros((2, 1, 5, 3))
    liquid[0, 0, 3:] = np.nan                             # two particles not emitted yet in row 0
    vol = c.volume(liquid, np.array([1e-6]))
    assert np.allclose(vol[:, 0], [3e-6, 5e-6], atol=0)
    assert np.allclose(c.volume(liquid, 1e-6, mask=np.zeros((2, 1, 5), bool)), 0.0, atol=0)

    assert c.within("z", np.array([0.5, 0.52, 0.9]), 0.4, 0.6, level=2, min_worlds=2).ok
    bad = c.within("z", np.array([0.5, np.nan]), 0.4, 0.6, level=2)
    assert not bad.ok and bad.level == 2
    assert c.at_least("a", 1.0, 0.5, 1).ok and not c.at_most("b", np.nan, 0.5, 1).ok
