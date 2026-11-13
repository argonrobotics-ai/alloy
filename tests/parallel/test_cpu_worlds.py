# SPDX-License-Identifier: Apache-2.0
"""CPU mode A (``alloy.build(..., device="cpu", processes=N)``): the law that a world's trajectory does not depend on
which process steps it, and the split sim's calls.

Law: 5 worlds (per-world starts, drawn friction and start x, an arm driven by per-world target ramps) in 2 processes
(blocks 3 + 2) and in 3 (2 + 2 + 1) record, bit for bit, the trajectories of the same scene in one process, on the
MuJoCo C route and on the Warp route, with drawn density too (the C route derives each world's contact softness from
its own masses, ``alloy.route.mjc_core_consts``; before, a worker's block took its first world's); an MPM paste the
same.

The law holds while every process keeps the same clock. An adaptive time step (the coupler's escalation, its rungs,
a liquid's CFL step) is chosen from one model's worlds, so a block can take another clock than one process of all the
worlds (as a world's clock in one process depends on all its batch-mates): the split sim then warns and reports each
worker's ``worker_physics_steps``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import alloy
from alloy.api.errors import FeatureUnavailableError, SimulationError, UnknownNameError
from alloy.parallel.cpu_worlds import SplitSim

pytestmark = [pytest.mark.cpu_route]

W = 5
SECONDS = 0.3
ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def _scene(density: bool) -> alloy.Scene:
    scene = alloy.Scene(worlds=W, dt=0.01)
    scene.add_ground()
    xs = np.stack([np.linspace(-0.2, 0.2, W), np.full(W, 0.4), np.full(W, 0.3)], axis=1)
    scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=xs, mass=0.2, rpy=(0.3, 0.2, 0.1))
    scene.add_box("puck", size=(0.06, 0.06, 0.02), pos=(0.0, -0.4, 0.2), mass=0.1, rpy=(0.0, 0.4, 0.0))
    scene.randomize("cube", "friction", 0.3, 1.0)
    scene.randomize("puck", "x", -0.05, 0.05)
    if density:
        scene.randomize("cube", "density", 500.0, 2000.0)
    scene.add_robot("arm", mjcf=ARM, pos=(0.6, 0.0, 0.3))
    return scene


def _drive(sim) -> alloy.Trajectory:
    sim.set_targets({"shoulder": np.linspace(-0.2, 0.2, W)})
    sim.step(3)
    return sim.run(SECONDS, targets={"shoulder": (np.linspace(-0.2, 0.2, W), np.linspace(0.6, -0.6, W))})


def _same(a: alloy.Trajectory, b: alloy.Trajectory) -> list[str]:
    assert sorted(a) == sorted(b)
    return [k for k in a if not np.array_equal(a[k], b[k])]


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.parametrize(("route", "density"), [("mjc", True), ("warp", True)])
def test_a_world_steps_the_same_in_any_process(monkeypatch, route, density):
    """Same worlds, bit for bit, in 1, 2 and 3 processes; uneven blocks; the split sim's joins in world order."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    scene = _scene(density)
    one = alloy.build(scene, device="cpu", seed=7)
    ref = _drive(one)
    for procs in (2, 3):
        with alloy.build(scene, device="cpu", seed=7, processes=procs) as split:
            assert isinstance(split, SplitSim) and split.processes == procs and split.worlds == W
            got = _drive(split)
            assert _same(ref, got) == [], f"{procs} processes differ from one in {_same(ref, got)}"
            assert got.meta["worlds"] == W and got.meta["processes"] == procs
            for w in (0, 2, 4):
                s1, s2 = one.state(world=w), split.state(world=w)
                assert all(np.array_equal(s1[k], s2[k]) for k in s1), w
            assert split.health()["ok"] and len(split.health()["worlds"]) == W
    one.close()


@pytest.mark.law
@pytest.mark.compile_heavy
def test_a_paste_world_steps_the_same_in_any_process():
    """MPM paste (mayonnaise, a dense grid of separate worlds on the CPU): 3 worlds in 2 processes (2 + 1) and in 3
    (one world each) match one process bit for bit. The workers hold the whole scene's grid box and a one-world
    worker builds the multi-world MPM entry (dense, separate worlds), as one process of all the worlds does."""
    scene = alloy.Scene(worlds=3, dt=0.01)
    scene.add_ground()
    scene.add_open_box("tub", inner=(0.06, 0.06, 0.04), pos=(0, 0, 0))
    starts = np.array([[-0.008, 0.0, 0.02], [0.0, 0.004, 0.02], [0.008, 0.0, 0.021]])
    scene.add_liquid("paste", size=(0.04, 0.04, 0.02), pos=starts, material="mayonnaise")
    one = alloy.build(scene, device="cpu")
    assert one.routes()["paste"]["route"] == "paste_mpm"
    ref = one.run(0.02, record=["paste"])
    one.close()
    for procs in (2, 3):
        with alloy.build(scene, device="cpu", processes=procs) as split:
            got = split.run(0.02, record=["paste"])
            assert _same(ref, got) == [], f"{procs} processes differ from one in {_same(ref, got)}"


def _ball_in_water(worlds: int, tank: float, water: float, ball: float, z: tuple[float, float]) -> alloy.Scene:
    """A light ball (500 kg/m³) dropped into water (SPH + MuJoCo: a staggered strong pair), z drawn per world."""
    sc = alloy.Scene(dt=0.01, worlds=worlds)
    sc.add_ground()
    sc.add_open_box("tank", inner=(tank, tank, 0.06), wall=0.006)
    sc.add_liquid("water", size=(tank, tank, water), pos=(0.0, 0.0, 0.006 + water / 2))
    sc.add_sphere("ball", ball, pos=(0.0, 0.0, 0.05), density=500.0)
    sc.randomize("ball", "z", *z)
    return sc


def _split_matches_one(scene: alloy.Scene, seconds: float, procs: int = 2) -> None:
    one = alloy.build(scene, device="cpu", seed=3)
    ref = one.run(seconds, record=["ball", "water"])
    one.close()
    with alloy.build(scene, device="cpu", seed=3, processes=procs) as split:
        got = split.run(seconds, record=["ball", "water"])
        assert got.meta["worker_physics_steps"] == [ref.meta["physics_steps"]] * procs
    assert _same(ref, got) == [], f"{procs} processes differ from one in {_same(ref, got)}"


@pytest.mark.law
@pytest.mark.compile_heavy
def test_a_coupled_scene_builds_and_steps_the_same_in_two_processes(monkeypatch):
    """A ball dropped 4-14 mm onto water (two solver entries, a Scheduler; WP9's review: the workers looked for MPM
    grids on the stepper and every coupled scene failed to build): 4 worlds, 0.06 s, 1 vs 2 processes, bit for bit.
    Worlds converge in different passes, so restart passes re-step frozen worlds: without the C route's window
    restore (``mjc_core_window``) worlds 0 and 1 differ by 1e-5 m (same clock)."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")              # the suite runs on Warp; the window is the C route's
    _split_matches_one(_ball_in_water(4, 0.04, 0.012, 0.008, (0.022, 0.032)), 0.06)


def test_growing_the_held_box_puts_the_margin_on_once() -> None:
    """Each growth unions the particles' box with the held core and adds one margin: growing from the margined box
    stacked a margin per growth until the dense guard raised (WP12's 100k red <gpu-vm>, L4, 64 worlds)."""
    import warp as wp  # noqa: PLC0415

    from alloy.integration import mpm_worlds  # noqa: PLC0415

    solver = SimpleNamespace(model=SimpleNamespace(particle_count=100_000), _environment_count=2)
    m, x = mpm_worlds.MARGIN, np.zeros((2, 3), dtype=np.float32)
    x[1] = (8.0, 8.0, 3.0)                                               # 8 x 8 x 3 voxels of 1 m
    for step in range(5):
        lo, hi = mpm_worlds._dense_bounds(solver, wp.array(x, dtype=wp.vec3, device="cpu"), None, 1.0, 0)
        assert np.array_equal(lo, [-m, -m, -m]) and np.array_equal(hi, [8 + m + step * (m + 1), 8 + m, 3 + m])
        x[1, 0] = hi[0] + 1.0                                            # one voxel past the held box: it grows
    held = SimpleNamespace(model=solver.model, _environment_count=2)
    mpm_worlds.hold_box(held, np.zeros(3, np.float32), np.array([8.0, 8.0, 3.0], np.float32), 2, 100_000)
    x[1, 0] = 8.0 + m + 1.0
    lo, hi = mpm_worlds._dense_bounds(held, wp.array(x, dtype=wp.vec3, device="cpu"), None, 1.0, 0)
    assert np.array_equal(lo, [-m, -m, -m]) and np.array_equal(hi, [8 + 2 * m + 1, 8 + m, 3 + m])


@pytest.mark.law
@pytest.mark.slow
@pytest.mark.timeout(900)
def test_restart_passes_step_a_world_the_same_in_any_block(monkeypatch):
    """WP9's liquid_ball (a 10 mm ball dropped 40-70 mm into 1,125 particles of water, 4 worlds): restart passes of
    the staggered pair re-step converged worlds frozen while a batch-mate iterates. The C route now continues them
    from the window start's float64 state (``mjc_core_window``); before, world 0 departed at 50 ms in 2 processes.
    0.1 s, 1 vs 2 processes, bit for bit (the clocks agree over this window)."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    _split_matches_one(_ball_in_water(4, 0.06, 0.02, 0.01, (0.04, 0.07)), 0.1)


def test_the_split_sim_answers_like_one_sim():
    """Names, routes, per-world teleports, errors of the right class, a closed sim, an engine that lives elsewhere."""
    scene = _scene(False)
    one = alloy.build(scene, device="cpu", seed=1)
    split = alloy.build(scene, device="cpu", seed=1, processes=2)
    try:
        assert split.names() == one.names() and split.routes() == one.routes() and split.dt == one.dt
        split.set_pose("cube", pos=(0.0, 0.4, 0.5), world=3)                # world 3 is in the second block
        split.step(1)
        z = split.state(["cube"])["cube.pos"][:, 2]
        assert z[3] > 0.45 and np.all(z[[0, 1, 2, 4]] < 0.35), z
        split.set_pose("puck", pos=np.tile([0.0, -0.4, 0.4], (W, 1)))       # one row per world, cut per block
        assert np.all(split.state(["puck"])["puck.pos"][:, 2] > 0.39)
        with pytest.raises(UnknownNameError):
            split.state(["nope"])
        with pytest.raises(SimulationError, match="for 5 worlds"):
            split.set_targets({"shoulder": np.zeros(3)})
        with pytest.raises(FeatureUnavailableError, match="processes=None"):
            split.engine  # noqa: B018
        assert split.check()["ok"] and split.t == pytest.approx(0.01)
        split.reset()
        one.step(1).reset()
        assert split.t == one.t and np.array_equal(split.state(world=4)["cube.pos"], one.state(world=4)["cube.pos"])
    finally:
        split.close()
        one.close()
    with pytest.raises(SimulationError, match="closed"):
        split.step(1)


def test_a_worker_that_dies_is_named():
    """A worker killed mid-run (its pid): the next call says which worker and worlds stopped, not a hang."""
    split = alloy.build(_scene(False), device="cpu", seed=0, processes=2)
    try:
        victim = split._workers[1]
        victim.proc.kill()
        victim.proc.wait()
        with pytest.raises(SimulationError, match=r"worker 1 \(worlds 3..4\) stopped"):
            split.step(1)
    finally:
        split.close()


def _cube_scene():
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 0.1))
    return scene


def same_as_one(split, one) -> None:
    np.testing.assert_array_equal(split[:2], one[:2])                  # the first process's block: worlds 0 and 1
    a, b = split[2:].astype(np.float64), one[2:].astype(np.float64)
    hit = np.isfinite(a) & np.isfinite(b)                                # depth is inf where a ray meets nothing
    assert (np.isfinite(a) == np.isfinite(b)).mean() >= 0.99
    diff = np.abs(a[hit] - b[hit])
    assert diff.mean() <= 0.01 * max(float(np.abs(b[hit]).max()), 1e-9), diff.mean()


def _camera(name: str):
    return alloy.Camera(name, look_at=((0.5, -0.5, 0.4), (0.0, 0.0, 0.05)), width=32, height=24, spp=1)


@pytest.fixture(scope="module")
def sims():
    """The same 4 worlds in one process and in two (blocks 2 + 2)."""
    scene = _cube_scene()
    one = alloy.build(scene, worlds=4, device="cpu", processes=1)
    split = alloy.build(scene, worlds=4, device="cpu", processes=2)
    try:
        yield one, split
    finally:
        one.close()
        split.close()


def test_a_split_sim_renders_several_outputs(sims):
    """An ``output=`` tuple comes back as one process's does, a dict joined per output (it raised "zero-dimensional
    arrays cannot be concatenated"). The first process's worlds render bit for bit as in one process; the second's are
    its local worlds 0 and 1, whose 1-sample noise is drawn from another world index, so they match within it."""
    one, split = sims
    rgbd_one = one.render(_camera("a"), output=("rgb", "depth"))
    rgbd_split = split.render(_camera("b"), output=("rgb", "depth"))
    assert set(rgbd_split) == {"rgb", "depth"}
    for key in rgbd_one:
        assert rgbd_split[key].shape == rgbd_one[key].shape
        same_as_one(rgbd_split[key], rgbd_one[key])


def test_a_split_sim_renders_with_a_camera_used_elsewhere(sims):
    """A camera that already rendered in this process renders in a split sim too: its renderer (ctypes handles) does
    not pickle, so the copy sent leaves it here (it raised AttributeError on CDLL._FuncPtr). The camera that stays here
    keeps its renderer and renders as before, bit for bit; the copy binds again in the worker."""
    one, split = sims
    here = _camera("here")
    first = one.render(here)
    bound = here._bound
    same_as_one(split.render(here), first)
    assert here._bound is bound
    np.testing.assert_array_equal(one.render(here), first)
    assert here._bound is bound
    import pickle  # noqa: PLC0415

    from alloy.parallel.cpu_worlds import portable  # noqa: PLC0415

    assert pickle.loads(pickle.dumps(portable(here)))._bound is None and here._bound is bound
