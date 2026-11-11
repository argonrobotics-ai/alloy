# SPDX-License-Identifier: Apache-2.0
"""A pour starts over at a reset (rc2 red fix; ``alloy.solvers.sph.emitters.rewind``). CPU.

Before the fix ``sim.reset()`` put a pour's particles back at the nozzle but not its emitter's place in line (cursor)
and layer clock: the build's start snapshot is taken before the pours' emitters exist. A finished pour then never
poured again (0 of 294 particles), a pour reset part-way delivered only its unpoured rest, and ``Pour-v0`` poured
water in its first episode only (0 of 448 particles in every later one).
- a finished pour, reset, pours the same particles at the same steps, bit for bit, and the mass ledger balances;
- a pour reset part-way pours its whole reservoir again;
- a reset of one world (a mask) re-pours that world as a fresh build's and leaves the other bit for bit;
- ``Pour-v0``: episode 2 (after the autoreset) pours the same water as episode 1 (``ALLOY_DEVICE=cuda:0`` runs this
  law on a GPU, within 1 %: the emitter repeats, but a splash does not, and a particle that leaves the world bounds
  is retired, so ``"water.active"`` can read a few particles less in one episode).

    PYTHONPATH=. python -m pytest tests/integration/test_pour_reset.py -q
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.api._liquid import liquid_solvers

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1800)]


def cup(worlds: int = 1, volume: float = 20e-6) -> alloy.Scene:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_open_box("cup", inner=(0.08, 0.08, 0.06), wall=0.005, pos=(0, 0, 0))
    s.add_pour("water", volume, pos=(0, 0, 0.12), rate=2e-4)        # 200 ml/s: 20 ml is out in 0.1 s
    return s


def active(sim: alloy.Sim) -> np.ndarray:
    return np.asarray(sim.state()["water.active"]).reshape(sim.worlds, -1).sum(axis=1)


def run(sim: alloy.Sim, steps: int) -> tuple[list[np.ndarray], bytes]:
    """(poured particles per world after each step, the water's points at the end)."""
    counts = []
    for _ in range(steps):
        sim.step(1)
        counts.append(active(sim))
    return counts, np.ascontiguousarray(sim.state()["water.points"]).tobytes()


def balanced(sim: alloy.Sim) -> bool:
    led = liquid_solvers(sim.engine)[0].mass_ledger()
    return bool(np.array_equal(led["active"] + led["out"], led["initial"] + led["in"]))


def mask(sim: alloy.Sim, *worlds: int) -> wp.array:
    m = np.zeros(sim.worlds, np.int32)
    m[list(worlds)] = 1
    return wp.array(m, dtype=wp.int32, device=sim.engine.model.device)


def test_a_finished_pour_pours_again_after_a_reset() -> None:
    sim = alloy.build(cup(), device="cpu", seed=0)
    first, points = run(sim, 20)
    assert first[-1][0] == first[9][0] > 200, first                  # done by 0.1 s, all of it out
    sim.reset()
    assert active(sim)[0] == 0 and balanced(sim)
    again, points_again = run(sim, 20)
    assert [int(c[0]) for c in again] == [int(c[0]) for c in first]  # before the fix: 0 at every step
    assert points_again == points and balanced(sim)


def test_a_pour_reset_part_way_pours_its_whole_reservoir() -> None:
    sim = alloy.build(cup(volume=40e-6), device="cpu", seed=0)
    full, _ = run(sim, 30)
    sim.reset()
    sim.step(8)                                                       # part-way: about a third poured
    assert 0 < active(sim)[0] < full[-1][0]
    sim.reset()
    again, _ = run(sim, 30)
    assert [int(c[0]) for c in again] == [int(c[0]) for c in full]  # before the fix: 348 of 648
    assert balanced(sim)


def test_a_masked_reset_re_pours_that_world_only() -> None:
    ref = alloy.build(cup(worlds=2), device="cpu", seed=0)
    ref_first, _ = run(ref, 15)
    ref_rest, ref_points = run(ref, 15)
    sim = alloy.build(cup(worlds=2), device="cpu", seed=0)
    run(sim, 15)
    sim.engine.reset(mask(sim, 1))
    assert active(sim).tolist() == [ref_first[-1][0], 0]
    rest, points = run(sim, 15)
    assert [int(c[1]) for c in rest] == [int(c[1]) for c in ref_first]       # world 1 pours as a fresh build
    assert [int(c[0]) for c in rest] == [int(c[0]) for c in ref_rest]        # world 0 goes on as if no reset
    w0 = lambda b: np.frombuffer(b, np.float32).reshape(2, -1, 3)[0]  # noqa: E731
    assert np.array_equal(w0(points), w0(ref_points)) and balanced(sim)


def test_pour_env_pours_in_every_episode() -> None:
    from alloy.envs import make  # noqa: PLC0415

    env = make("Pour-v0", num_envs=2, device=os.environ.get("ALLOY_DEVICE", "cpu"), seed=0, max_steps=30,
               volume=30e-6, vary=False)
    env.reset(seed=0)
    act = np.zeros((2,) + tuple(env.single_action_space.shape), np.float32)
    counts, ends = [], []
    for k in range(60):
        _, _, _, trunc, _ = env.step(act)
        counts.append(active(env.sim).tolist())
        if np.asarray(trunc).any():
            ends.append(k)
    assert ends == [29, 59], ends
    assert counts[29] == [0, 0] and counts[59] == [0, 0]              # each autoreset puts the water back
    first, second = np.array(counts[0:29]), np.array(counts[30:59])
    assert first[-1].min() > 400, counts                              # before the fix: episode 2 poured 0
    if str(env.sim.device).startswith("cpu"):
        assert np.array_equal(second, first), counts
    else:   # a GPU splash does not repeat: on an L4 2 of 448 left the world bounds (retired) in one of two worlds
        assert np.all(np.abs(second - first) <= np.ceil(0.01 * first)), counts
