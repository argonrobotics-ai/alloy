# SPDX-License-Identifier: Apache-2.0
"""Timed pushes (``scene.add_push``) and random starts at reset (``start_x/_y/_z/_yaw``), on both CPU routes.

Laws, each per world, from the motion:

- push: a 2 kg mass on a free slide along x gains speed F dt / m in every period a force F is held; the speed at the
  end equals the sum of the held forces (read from ``sim.pushes()`` after each step) x dt / m, to 1e-4 of the largest
  speed. Every world is pushed at its own times with its own force.
- start: after a reset, a free-base robot's base is at its start + the drawn x, y, z and turned by the drawn yaw
  about z through the base; a link 0.3 m out along the base's x is where that turn puts it (1e-5 m), and it starts
  at rest there (it then falls straight down: the rig has no floor). A masked reset moves the reset worlds only.

The other tests are about the draws and the wiring: the same seed gives the same pushes and starts for any world
count, a reset drops a push, no rows means no pusher and no hook, and the argument checks.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import warp as wp

import alloy.api as alloy

pytestmark = [pytest.mark.compile_heavy, pytest.mark.timeout(900)]     # a cold MuJoCo Warp compile passes 180 s

W, M, DT = 4, 2.0, 0.01
BOX = '<geom type="box" size="0.05 0.05 0.05" mass="{m}" contype="0" conaffinity="0"/>'
SLIDER = f"""<mujoco><worldbody><body name="mass" pos="0 0 0.5"><joint name="j" type="slide" axis="1 0 0"/>
 {BOX.format(m=M)}</body></worldbody></mujoco>"""
FLOATER = f"""<mujoco><worldbody><body name="base" pos="0.1 0.2 0.5"><freejoint/>
 {BOX.format(m=M)}<body name="tip" pos="0.3 0 0"><joint name="hinge" type="hinge" axis="0 1 0"/>{BOX.format(m=0.5)}
 </body></body></worldbody></mujoco>"""
WELDED = f"""<mujoco><worldbody><body name="mass" pos="0 0 0.5"><joint name="j" type="hinge" axis="0 1 0"/>
 {BOX.format(m=M)}</body></worldbody></mujoco>"""
SOLVERS = {"mjc": "MjcCoreSolver", "warp": "SolverMuJoCo"}


@pytest.fixture(params=[pytest.param("mjc", marks=pytest.mark.cpu_route), "warp"])
def route(request):
    """Each CPU route in turn (the suite's conftest pins ``ALLOY_CPU_ROUTE=warp``; "mjc" is the users' default)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ALLOY_CPU_ROUTE", request.param)
        yield request.param


def _pushed(worlds: int = W, seed: int = 3, every=(0.05, 0.2)) -> alloy.Scene:
    scene = alloy.Scene(worlds=worlds, dt=DT)
    scene.add_robot("rig", mjcf=SLIDER)
    scene.add_push("rig/mass", force=(10.0, 0.0, 0.0), every=every, duration=0.05)
    return scene


def _floating(worlds: int = W) -> alloy.Scene:
    scene = alloy.Scene(worlds=worlds, dt=DT)
    scene.add_robot("bot", mjcf=FLOATER)
    scene.randomize("bot", "start_x", -0.5, 0.5, when="reset")
    scene.randomize("bot", "start_y", -0.5, 0.5, when="reset")
    scene.randomize("bot", "start_z", 0.0, 0.2, when="reset")
    scene.randomize("bot", "start_yaw", -3.0, 3.0, when="reset")
    return scene


def _yaw_of(quat_wxyz: np.ndarray) -> np.ndarray:
    return 2.0 * np.arctan2(quat_wxyz[:, 3], quat_wxyz[:, 0])


def _wrapped(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# --- laws ---------------------------------------------------------------------------------------------------------


def test_each_world_gains_the_speed_its_own_pushes_give(route):
    sim = alloy.build(_pushed(), seed=3, device="cpu")
    assert [type(s).__name__ for s in sim.engine.stepper.solvers()] == [SOLVERS[route]]
    impulse, seen, starts = np.zeros(W), [], []
    for k in range(60):
        sim.step(1)
        f = sim.pushes()["rig/mass"]
        assert not f[:, 1:].any()
        impulse += f[:, 0] * DT
        seen.append(f[:, 0].copy())
    seen = np.asarray(seen)
    speed = sim.state(["rig/mass"])["rig/mass.vel"][:, 0]
    assert np.abs(speed).max() > 0.05                                       # the pushes moved the masses
    np.testing.assert_allclose(speed, impulse / M, rtol=0, atol=1e-4 * np.abs(speed).max())
    for w in range(W):
        on = np.flatnonzero(seen[:, w])
        starts.append(int(on[0]))
        assert len(on) >= 5 and np.abs(seen[on, w]).max() <= 10.0           # held 5 periods (0.05 s), inside +-10 N
    assert len(set(starts)) > 1 and len({round(float(v), 6) for v in seen[seen != 0]}) > W   # own times, own forces
    assert sim.check()["ok"] and all(not h["flags"] for h in sim.health()["worlds"])


def test_a_reset_puts_each_world_at_its_drawn_start_and_a_masked_one_moves_only_its_worlds(route):
    sim = alloy.build(_floating(), seed=5, device="cpu")
    assert [type(s).__name__ for s in sim.engine.stepper.solvers()] == [SOLVERS[route]]
    names = ["bot/base", "bot/tip"]
    nominal = sim.state(names)
    assert not any(v.any() for v in sim.randomized().values())              # nothing is drawn before a reset
    sim.step(3)
    sim.reset()
    first, drawn = sim.state(names), sim.randomized()
    off = np.stack([drawn[f"bot.start_{a}"] for a in "xyz"], axis=1)
    yaw = drawn["bot.start_yaw"]
    assert np.ptp(yaw) > 0.5 and np.ptp(off[:, 0]) > 0.05
    np.testing.assert_allclose(first["bot/base.pos"], nominal["bot/base.pos"] + off, rtol=0, atol=1e-5)
    np.testing.assert_allclose(_wrapped(_yaw_of(first["bot/base.quat_wxyz"]) - yaw), 0.0, atol=1e-5)
    arm = 0.3 * np.stack([np.cos(yaw), np.sin(yaw), np.zeros(W)], axis=1)   # the tip, 0.3 m along the base's x
    np.testing.assert_allclose(first["bot/tip.pos"], first["bot/base.pos"] + arm, rtol=0, atol=1e-5)
    sim.step(20)
    held = sim.state(names)
    np.testing.assert_allclose(held["bot/base.pos"][:, :2], first["bot/base.pos"][:, :2], rtol=0, atol=1e-5)
    assert (held["bot/base.pos"][:, 2] < first["bot/base.pos"][:, 2] - 0.1).all()     # it falls straight from there
    mask = np.zeros(W, np.int32)
    mask[[0, 2]] = 1
    with wp.ScopedDevice("cpu"):
        sim.engine.reset(wp.array(mask, dtype=wp.int32, device="cpu"))
    second, again = sim.state(names), sim.randomized()
    for key in drawn:
        assert (again[key][[1, 3]] == drawn[key][[1, 3]]).all() and (again[key][[0, 2]] != drawn[key][[0, 2]]).all()
    np.testing.assert_allclose(second["bot/base.pos"][[1, 3]], held["bot/base.pos"][[1, 3]], rtol=0, atol=0)
    off2 = np.stack([again[f"bot.start_{a}"] for a in "xyz"], axis=1)
    np.testing.assert_allclose(second["bot/base.pos"][[0, 2]], (nominal["bot/base.pos"] + off2)[[0, 2]], rtol=0,
                               atol=1e-5)
    sim.step(5)
    assert sim.check()["ok"] and all(not h["flags"] for h in sim.health()["worlds"])


# --- draws and wiring ---------------------------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_pushes_and_starts_for_any_world_count():
    def forces(worlds: int) -> np.ndarray:
        sim = alloy.build(_pushed(worlds), seed=3, device="cpu")
        out = []
        for _ in range(30):
            sim.step(1)
            out.append(sim.pushes()["rig/mass"][:2].copy())
        return np.asarray(out)

    small, big = forces(2), forces(5)
    assert small.any() and (small == big).all()
    other = alloy.build(_pushed(2), seed=4, device="cpu")
    other.step(30)
    starts = []
    for worlds in (2, 5):
        sim = alloy.build(_floating(worlds), seed=5, device="cpu")
        sim.reset()
        starts.append(sim.randomized()["bot.start_yaw"][:2])
    assert (starts[0] == starts[1]).all()


def test_a_reset_drops_the_push_and_starts_a_new_wait():
    sim = alloy.build(_pushed(every=(0.03, 0.03)), seed=3, device="cpu")
    sim.step(4)                                                            # 3 periods of wait, then pushing
    assert sim.pushes()["rig/mass"][:, 0].all()
    sim.reset()
    assert not sim.pushes()["rig/mass"].any() and not sim.engine.state.body_f.numpy().any()
    assert not sim.state(["rig/mass"])["rig/mass.vel"].any()
    sim.step(3)
    assert not sim.pushes()["rig/mass"].any()                              # waiting again
    sim.step(1)
    assert sim.pushes()["rig/mass"][:, 0].all()
    before = sim.pushes()["rig/mass"].copy()
    mask = np.asarray([1, 0, 0, 0], np.int32)
    with wp.ScopedDevice("cpu"):
        sim.engine.reset(wp.array(mask, dtype=wp.int32, device="cpu"))
    now = sim.pushes()["rig/mass"]
    assert not now[0].any() and (now[1:] == before[1:]).all()


def test_no_rows_means_no_pusher_and_the_same_steps():
    def run(push: bool) -> np.ndarray:
        scene = alloy.Scene(worlds=2, dt=DT)
        scene.add_robot("rig", mjcf=SLIDER.replace('axis="1 0 0"', 'axis="0 0 1"'))
        if push:
            scene.add_push("rig/mass", force=(1e-30, 0.0, 0.0), every=(100.0, 100.0))
        sim = alloy.build(scene, seed=0, device="cpu")
        assert (sim._pushes is not None) == push and sim.pushes().keys() == ({"rig/mass"} if push else set())
        sim.step(20)
        return sim.state(["rig/mass"])["rig/mass.pos"]

    plain = run(False)
    assert plain[0, 2] < 0.49 and (plain == run(True)).all()               # a push that never comes changes no bit
    scene = alloy.Scene(worlds=2, dt=DT)
    scene.add_robot("rig", mjcf=SLIDER)
    assert "pushes" not in scene.to_dict()
    assert alloy.build(scene, device="cpu").engine.reset_hooks == []


def test_rows_survive_json():
    scene = _pushed()
    scene.randomize("rig", "mass_scale", 0.9, 1.1)
    back = alloy.Scene.from_dict(json.loads(scene.to_json()))
    assert back.pushes == scene.pushes == [{"target": "rig/mass", "force": [10.0, 0.0, 0.0], "every": [0.05, 0.2],
                                            "duration": 0.05}]
    floating = alloy.Scene.from_dict(json.loads(_floating().to_json()))
    assert [r["field"] for r in floating.robot_random] == ["start_x", "start_y", "start_z", "start_yaw"]


def test_argument_checks_name_the_fix():
    scene = alloy.Scene(worlds=2, dt=DT)
    scene.add_robot("rig", mjcf=SLIDER)
    for kwargs, field in (({"force": (0, 0, 0)}, "force"), ({"force": (1, -1, 0)}, "force"),
                          ({"force": (1, 0)}, "force"), ({"force": (1, 0, 0), "every": (0, 1)}, "every"),
                          ({"force": (1, 0, 0), "every": (3, 1)}, "every"),
                          ({"force": (1, 0, 0), "duration": 0}, "every")):
        with pytest.raises(alloy.SceneError) as err:
            scene.add_push("rig/mass", **kwargs)
        assert err.value.field == field and err.value.fix
    scene.add_push("rig/mass", force=(1, 0, 0))
    with pytest.raises(alloy.SceneError, match="already has a push"):
        scene.add_push("rig/mass", force=(1, 0, 0))
    with pytest.raises(alloy.SceneError, match="drawn at every reset") as err:
        scene.randomize("rig", "start_yaw", -1, 1)
    assert "when='reset'" in err.value.fix
    with pytest.raises(alloy.SceneError, match="moves a whole robot"):
        scene.randomize("rig/mass", "start_x", -1, 1, when="reset")
    with pytest.raises(alloy.SceneError, match="start_yaw"):
        scene.randomize("rig", "yaw", -1, 1)


def test_build_refuses_what_it_cannot_do():
    scene = alloy.Scene(worlds=2, dt=DT)
    scene.add_robot("rig", mjcf=SLIDER)
    scene.add_push("rig/nolink", force=(1, 0, 0))
    with pytest.raises(alloy.BuildError, match="is not a body") as err:
        alloy.build(scene, device="cpu")
    assert "rig/mass" in err.value.fix
    for kwargs, word in (({"processes": 2}, "processes"), ({"rerun_worlds": 1}, "rerun_worlds")):
        with pytest.raises(alloy.BuildError, match=word):
            alloy.build(_pushed(), device="cpu", **kwargs)
    welded = alloy.Scene(worlds=2, dt=DT)
    welded.add_robot("rig", mjcf=WELDED)
    welded.randomize("rig", "start_x", -1, 1, when="reset")
    with pytest.raises(alloy.BuildError, match="no free base") as err:
        alloy.build(welded, device="cpu")
    assert "freejoint" in err.value.fix
