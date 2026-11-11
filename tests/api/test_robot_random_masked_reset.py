# SPDX-License-Identifier: Apache-2.0
"""A reset of some worlds leaves the other worlds alone in a scene with ``scene.randomize`` rows on objects
(TRAIN-FIX 3a, TA-8 F1), on both CPU routes.

Law: 8 worlds, a cube dropped from 0.3 m beside a slider robot, a friction, density or start-x row on the cube.
After 0.5 s every cube rests on the ground; a reset of world 0 puts world 0 back at its start state, bit for bit, and
worlds 1-7 keep their state, bit for bit. The per-world drawn values and the model arrays they set do not change.

Before the fix, any object row made the engine's reset copy the model's start joint coordinates into every world and
run forward kinematics over all of them: every cube went back to its start, reset or not.

A world whose draws changed since the last reset (a caller redrew it at another seed) is the one case where a reset
moves a start: there the reset world goes to its new start, again without touching the others.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import pytest
import warp as wp

import alloy.api as alloy

pytestmark = [pytest.mark.compile_heavy, pytest.mark.timeout(900)]     # a cold MuJoCo Warp compile passes 180 s

W = 8
SLIDER = """<mujoco><worldbody><body name="b" pos="0.5 0 0.05"><joint name="x" type="slide" axis="1 0 0"
 range="-0.1 0.1"/><geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body></worldbody>
 <actuator><position joint="x" kp="100"/></actuator></mujoco>"""
ROWS = {"none": None, "friction": ("friction", 0.3, 1.0), "density": ("density", 800.0, 2400.0),
        "x": ("x", -0.2, 0.2)}


@pytest.fixture(params=[pytest.param("mjc", marks=pytest.mark.cpu_route), "warp"])
def route(request):
    """Each CPU route in turn (the suite's conftest pins ``ALLOY_CPU_ROUTE=warp``; "mjc" is the users' default)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ALLOY_CPU_ROUTE", request.param)
        yield request.param


def _sim(row: str, worlds: int = W) -> alloy.Sim:
    scene = alloy.Scene(worlds=worlds, dt=0.02)
    scene.add_ground()
    scene.add_robot("slider", mjcf=SLIDER)
    scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.3), density=1600.0)   # falls 27.5 cm
    if ROWS[row] is not None:
        scene.randomize("cube", *ROWS[row])
    return alloy.build(scene, device="cpu", seed=5)


def _mask(*worlds: int) -> wp.array:
    m = np.zeros(W, np.int32)
    m[list(worlds)] = 1
    return wp.array(m, dtype=wp.int32, device="cpu")


def _model(sim: alloy.Sim) -> dict[str, np.ndarray]:
    """The model arrays a randomize row writes, and the drawn values."""
    m, out = sim.engine.model, {}
    for name in ("body_mass", "body_inv_mass", "body_inertia", "shape_material_mu", "joint_q"):
        out[name] = getattr(m, name).numpy().copy()
    table = sim.engine.built.pair_table
    if getattr(table, "world_table", None) is not None:
        out["world_table"] = table.world_table.numpy().copy()
    for key, vals in getattr(sim.engine.built, "randomized", {}).items():
        out[".".join(key)] = np.asarray(vals).copy()
    return out


def _same(a: dict, b: dict, worlds) -> list[str]:
    """Keys whose values differ, bit for bit, in ``worlds`` (state dicts: world axis first)."""
    return [k for k in a if k != "t" and not np.array_equal(np.asarray(a[k])[worlds], np.asarray(b[k])[worlds])]


@pytest.mark.parametrize("row", list(ROWS))
def test_a_masked_reset_leaves_the_other_worlds_alone(route, row):
    sim = _sim(row)
    start, model = sim.state(), _model(sim)
    sim.step(25)                                                         # 0.5 s: every cube rests on the ground
    before = sim.state()
    assert np.all(np.abs(before["cube.pos"][:, 2] - 0.025) < 2e-3), before["cube.pos"][:, 2]
    sim.engine.reset(_mask(0))
    after = sim.state()
    assert _same(before, after, slice(1, W)) == [], f"worlds 1-7 moved on a reset of world 0 ({row})"
    assert _same(start, after, [0]) == [], f"world 0 is not back at its start ({row})"
    now = _model(sim)
    assert [k for k in model if not np.array_equal(model[k], now[k])] == [], f"a reset changed drawn values ({row})"
    if row == "x":                                                       # each world has its own start x
        assert len(np.unique(start["cube.pos"][:, 0])) == W


def test_a_redrawn_world_goes_to_its_new_start_and_the_others_stay(route):
    from alloy.scene.randomize import apply_randomization

    sim = _sim("x")
    start = sim.state()
    sim.step(25)
    apply_randomization(sim.engine.built, 7, wp.array([3], dtype=wp.int32))   # world 3's draw changes
    before = sim.state()
    sim.engine.reset(_mask(3))
    after = sim.state()
    assert _same(before, after, [w for w in range(W) if w != 3]) == []
    np.testing.assert_allclose(after["cube.pos"][3], start["cube.pos"][3], atol=1e-6)   # back at the seed's draw
    np.testing.assert_allclose(after["cube.vel"][3], 0.0, atol=1e-9)


def test_a_vector_env_reset_of_one_world_leaves_the_others_alone():
    """The repro's path (TA-8 F1): ``VectorEnv`` with ``reset_mask``, NumPy backend, friction row."""
    from alloy.envs import Task, VectorEnv

    scene = alloy.Scene(dt=0.02)
    scene.add_ground()
    scene.add_robot("slider", mjcf=SLIDER)
    scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.3), mass=0.2)
    scene.randomize("cube", "friction", 0.3, 1.0)
    env = VectorEnv(scene, Task(obs=["cube"], max_steps=1000), worlds=W, seed=0, device="cpu", backend="numpy",
                    autoreset="disabled")
    env.reset(seed=0)
    for _ in range(25):
        env.step(np.zeros((W, 1), np.float32))
    z = np.asarray(env.state["cube.pos"][:, 2]).copy()
    mask = np.zeros(W, bool)
    mask[0] = True
    env.reset(options={"reset_mask": mask})
    z_after = np.asarray(env.state["cube.pos"][:, 2])
    assert np.array_equal(z[1:], z_after[1:]), (z, z_after)
    assert abs(float(z_after[0]) - 0.3) < 1e-6
    env.close()


def _mask_of(worlds: int, ids) -> wp.array:
    m = np.zeros(worlds, np.int32)
    m[list(ids)] = 1
    return wp.array(m, dtype=wp.int32, device="cpu")


def test_a_plain_reset_writes_nothing_and_tells_no_solver(monkeypatch):
    """TRAIN-FIX 3b (TA-8 F3): a reset with build-time object rows draws the values the model holds already, so it
    rewrites none and notifies no solver (on a GPU a notice drops the captured step graphs); each row is drawn once
    for every world, at the first reset, and never again."""
    import alloy.scene.randomize as R

    sim = _sim("density")
    eng, writes, told, draws = sim.engine, [], [], []
    real_draw = R.draw
    monkeypatch.setattr(R, "apply_randomization", lambda *a, **k: writes.append(a))
    monkeypatch.setattr(R, "draw", lambda *a: draws.append(a) or real_draw(*a))
    monkeypatch.setattr(eng, "_notify", lambda flags: told.append(flags))
    model = _model(sim)
    sim.step(2)
    for worlds in ([0], [3, 5], list(range(W))):
        eng.reset(_mask(*worlds))
    eng.reset(None)
    assert writes == [] and told == []
    assert len(draws) == W
    now = _model(sim)
    assert [k for k in model if not np.array_equal(model[k], now[k])] == []


@pytest.mark.parametrize("row", ["friction", "density"])
def test_a_reset_with_a_build_time_row_costs_at_most_a_tenth_more_than_one_without(row):
    """TRAIN-FIX 3b's law on the CPU: masked resets of 32 of 1,024 worlds, interleaved with the same resets of the scene
    without the row so that both see the same host load; the median reset takes at most 1.10x as long. Before the fix
    (TA-8 F3) the same reset took 39 ms (friction) and 2,162 ms (density) against 23 ms without a row."""
    n = 1024
    sims = {k: _sim(k, worlds=n) for k in ("none", row)}
    rng, times = np.random.default_rng(0), {k: [] for k in sims}
    for i in range(45):
        mask = _mask_of(n, rng.choice(n, 32, replace=False))
        for k in (("none", row) if i % 2 else (row, "none")):
            t = time.perf_counter()
            sims[k].engine.reset(mask)
            if i >= 5:
                times[k].append(time.perf_counter() - t)
    base, cost = float(np.median(times["none"])), float(np.median(times[row]))
    assert cost <= 1.10 * base, f"{row}: {cost * 1e3:.3f} ms against {base * 1e3:.3f} ms without the row"


@pytest.mark.gpu
def test_on_a_gpu_a_masked_reset_keeps_the_other_worlds_and_the_captured_graphs():
    """WP2's review ask: the 3a law on cuda:0 with the period graphs on. 64 worlds, a friction row: after a reset of
    world 0, worlds 1-63 are unchanged bit for bit and world 0 is at its start; two masked resets keep the graphs.
    The simulation is closed and collected before the test returns: left to the collector, its device arrays could
    be freed while a later test captures a CUDA graph."""
    if not wp.is_cuda_available():
        pytest.skip("needs a CUDA GPU")
    sim = _gpu_law("cuda:0")
    sim.close()
    del sim
    gc.collect()
    wp.synchronize_device("cuda:0")


def _gpu_law(dev: str) -> alloy.Sim:
    n = 64
    scene = alloy.Scene(worlds=n, dt=0.02)
    scene.add_ground()
    scene.add_robot("slider", mjcf=SLIDER)
    scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.3), density=1600.0)
    scene.randomize("cube", "friction", 0.3, 1.0)
    sim = alloy.build(scene, device=dev, seed=5)
    assert sim.engine.use_graph
    start = sim.state()
    sim.step(25)
    before = sim.state()
    m = np.zeros(n, np.int32)
    m[0] = 1
    sim.engine.reset(wp.array(m, dtype=wp.int32, device=dev))
    after = sim.state()
    assert _same(before, after, slice(1, n)) == []
    assert _same(start, after, [0]) == []
    sim.step(3)
    m[:] = 0
    m[[5, 9]] = 1
    sim.engine.reset(wp.array(m, dtype=wp.int32, device=dev))
    sim.step(1)
    stats = sim.engine.graph_stats()
    assert (stats["reset_kept"], stats["reset_dropped"]) == (2, 0), stats
    return sim
