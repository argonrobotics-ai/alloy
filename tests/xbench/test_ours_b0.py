# SPDX-License-Identifier: Apache-2.0
"""P3-2: the engine-side B0 generator (bench/xbench/ours/b0.py) gives Alloy's SB-7 traces bit for bit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
OURS = REPO / "bench" / "xbench" / "ours"
sys.path.insert(0, str(OURS))

import b0  # noqa: E402

from alloy.bully.actions import legalize  # noqa: E402
from alloy.bully.stress import random_legal_traces  # noqa: E402
from alloy.bully.types import ActionSpace, BullyTarget  # noqa: E402


def _space() -> dict:
    return json.loads((OURS / "packs" / "t12" / "t12.json").read_text())["action_space"]


def _target(space: dict) -> BullyTarget:
    actions = ActionSpace(low=np.asarray(space["low"]), high=np.asarray(space["high"]), rate_hz=space["rate_hz"],
                          max_delta_per_tick=np.asarray(space["max_delta_per_tick"]),
                          initial=np.asarray(space["initial"]))
    return BullyTarget(name="t12_pack", scene=None, make_sim=None, actions=actions, action_kind="mujoco.ctrl",
                       action_index=tuple(range(len(space["low"]))), robot_bodies=())


@pytest.mark.parametrize("seed", [100_000, 100_147])
def test_b0_traces_equal_alloy_bitwise(seed):
    space = _space()
    ours = b0.random_legal_traces(space, 64, seed)
    alloy = random_legal_traces(_target(space), 64, 2.0, seed=seed)
    assert ours.dtype == np.float32 and ours.shape == (64, 100, 14)
    assert np.array_equal(ours.view(np.uint32), alloy.view(np.uint32))


def test_legalize_matches_alloy_and_is_idempotent():
    space = _space()
    rng = np.random.default_rng(3)
    lo, hi = np.asarray(space["low"]), np.asarray(space["high"])
    raw = lo + (hi - lo) * rng.uniform(-0.2, 1.2, size=(8, 30, lo.size))
    md, init = np.asarray(space["max_delta_per_tick"]), np.asarray(space["initial"])
    ours, clips = b0.legalize(raw, lo, hi, md, init)
    ref = legalize(raw, lo, hi, md, init)
    assert np.array_equal(ours, ref.trace) and np.array_equal(clips, ref.clips)
    again, clips2 = b0.legalize(ours, lo, hi, md, init)
    assert np.allclose(again, ours, rtol=0, atol=1e-6) and not clips2.any()   # no clip, but not bitwise


def test_applied_traces_equal_what_the_episode_runner_plays():
    """EpisodeRunner._legal re-legalises the float32 trace (alloy/bully/replay.py); engines must play that."""
    space = _space()
    sp = _target(space).actions
    first = random_legal_traces(_target(space), 32, 2.0, seed=100_000)
    played = legalize(first, sp.low, sp.high, sp.max_delta_per_tick, sp.initial).trace
    assert np.array_equal(b0.applied_traces(space, 32, 100_000).view(np.uint32), played.view(np.uint32))


def test_batch_traces_rows_are_seed_world_episodes():
    space = _space()
    seeds = b0.campaign_seeds(3 * b0.SEED_BATCH)
    assert seeds == [100_000, 100_001, 100_002]
    rows = b0.batch_traces(space, seeds[1:2])
    assert np.array_equal(rows[5], b0.applied_traces(space, b0.SEED_BATCH, 100_001)[5])
    assert len(b0.campaign_seeds(303_104)) == 148
