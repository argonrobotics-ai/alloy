# SPDX-License-Identifier: Apache-2.0
"""Task checkers, observations and spaces on hand-made batched state (no engine): fast, CPU, NumPy."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import checks as C
from alloy.envs import spaces
from alloy.envs.backend import NumpyBackend, pick_backend
from alloy.envs.errors import EnvError
from alloy.envs.layout import JointRows, Layout
from alloy.envs.state import BatchState, Rows
from alloy.envs.task import Observer, Rewarder, Task


def _layout() -> Layout:
    lay = Layout(worlds=2, bodies={"cube": np.array([0, 3]), "g/left": np.array([1, 4]), "g/right": np.array([2, 5])})
    z = np.zeros((1,))
    lay.joints["g/lift"] = JointRows(q=np.array([[0], [1]]), qd=np.array([[0], [1]]), target_q=np.array([[0], [1]]),
                                     lower=z - 0.1, upper=z + 0.2, actuated=True)
    lay.robots["g"], lay.objects = ["g/lift"], ["cube"]
    lay.particles["water"] = np.array([[0, 1, 2], [3, 4, 5]])
    return lay


def _state(cube_z: tuple[float, float] = (0.13, 0.03), tilt_world1: bool = False) -> BatchState:
    bq = np.zeros((6, 7), np.float32)
    bq[:, 6] = 1.0                                              # identity quaternions (x, y, z, w)
    bq[[0, 3], 2] = cube_z                                       # cube heights
    bq[[1, 4], 1], bq[[2, 5], 1] = 0.04, -0.04                   # fingers either side of the cube in y
    bq[[1, 2, 4, 5], 2] = cube_z[0], cube_z[0], cube_z[1], cube_z[1]
    if tilt_world1:
        bq[3, 3:7] = [np.sin(np.pi / 4), 0, 0, np.cos(np.pi / 4)]  # 90 degrees about x
    bqd = np.zeros((6, 6), np.float32)
    bqd[0, 0] = 0.5
    pts = np.zeros((6, 3), np.float32)
    pts[:3, 2], pts[3:, 2] = 0.05, 0.5                           # world 0's water in the cup box, world 1's above it
    arrays = {"body_q": bq, "body_qd": bqd, "joint_q": np.array([0.1, 0.0], np.float32),
              "joint_qd": np.zeros(2, np.float32), "particle_q": pts}
    rows = Rows(_layout(), NumpyBackend("cpu"))
    start_q = bq.copy()
    start_q[[0, 3], 2] = 0.03
    start = BatchState({**arrays, "body_q": start_q}.__getitem__, rows)
    extra = {"steps": np.array([1, 1]), "action": np.array([[1.0, 0.0], [0.5, 0.5]], np.float32)}
    return BatchState(arrays.__getitem__, rows, extra=extra, start=start)


def test_measurements():
    s = _state()
    assert np.allclose(C.height("cube")(s), [0.13, 0.03])
    assert np.allclose(C.rise("cube")(s), [0.10, 0.0], atol=1e-6)
    assert np.allclose(C.distance("cube", (0, 0, 0))(s), [0.13, 0.03], atol=1e-6)
    assert np.allclose(C.speed("cube")(s), [0.5, 0.0])
    assert np.allclose(C.joint("g/lift")(s), [0.1, 0.0])
    assert np.allclose(C.fraction_inside("water", (-1, -1, 0), (1, 1, 0.1))(s), [1.0, 0.0])
    assert np.allclose(C.action_size()(s), [1.0, 0.5])
    assert np.allclose(C.tilt("cube")(_state(tilt_world1=True)), [0.0, 90.0], atol=1e-3)


def test_tests_and_logic():
    s = _state()
    lifted, held = C.lifted("cube", by=0.05), C.between("cube", "g/left", "g/right")
    assert lifted(s).tolist() == [True, False]
    assert held(s).tolist() == [True, True]
    assert (lifted & held)(s).tolist() == [True, False]
    assert (lifted | ~held)(s).tolist() == [True, False]
    assert (C.height("cube") > 0.1)(s).tolist() == [True, False]
    assert C.inside("cube", (-1, -1, 0.1), (1, 1, 1))(s).tolist() == [True, False]
    assert C.near("cube", (0, 0, 0.03), 0.01)(s).tolist() == [False, True]
    assert C.at_rest("cube", 0.1)(s).tolist() == [False, True]
    assert C.upright("cube", 10)(_state(tilt_world1=True)).tolist() == [True, False]
    assert C.all_of(lifted, held).name == "(lifted(cube, 0.05) & between(cube, g/left, g/right))"


def test_state_keys_and_errors():
    s = _state()
    assert s["cube.pos"].shape == (2, 3) and s["cube.quat_wxyz"][0].tolist() == [1, 0, 0, 0]
    assert s["water.center"].shape == (2, 3) and s["water.points"].shape == (2, 3, 3)
    assert "g/lift.qd" in s.keys() and "steps" in s
    with pytest.raises(EnvError, match="cube.speed"):
        s["cube.speed"]
    with pytest.raises(EnvError, match="no body, joint, robot or particle object named 'cub'") as e:
        s["cub.pos"]
    assert "cube" in e.value.fix


def test_observer_rewarder():
    s = _state()
    lay = _layout()
    obs = Observer(["g", "cube", "action", ("x", lambda st: st["cube.pos"][:, 0]), C.height("cube")], lay)
    vec = obs(s)
    assert vec.shape == (2, 2 + 13 + 2 + 1 + 1) and vec.dtype == np.float32
    assert obs.names()[:3] == ["g/lift.q", "g/lift.qd", "cube.pos[0]"] and obs.names()[-1] == "height(cube)"
    assert Observer(None, lay)(s).shape == (2, 2 + 13 + 3)
    with pytest.raises(EnvError, match="observation"):
        Observer(["nope"], lay)
    r, terms = Rewarder(Task(reward={"up": (2.0, C.rise("cube")), "held": (1.0, C.lifted("cube", 0.05))}))(s)
    assert np.allclose(r, [1.2, 0.0], atol=1e-6) and set(terms) == {"up", "held"}


def test_task_validation():
    with pytest.raises(EnvError, match="max_steps"):
        Task(max_steps=0).validate()
    with pytest.raises(EnvError, match="on_invalid"):
        Task(on_invalid="explode").validate()
    assert Task(actions=["g/lift"]).action_spec().joints == ["g/lift"]


def test_spaces_and_backend():
    box = spaces.Box(-1.0, 1.0, (3,))
    box.seed(0)
    assert box.contains(box.sample()) and not box.contains(np.full(3, 2.0))
    batched = spaces.batch(spaces.box(-1.0, 1.0, (3,)), 4)
    assert batched.shape == (4, 3)
    d = spaces.Dict({"state": box})
    assert d.contains({"state": np.zeros(3, np.float32)})
    with pytest.raises(EnvError, match="backend"):
        pick_backend("jax", "cpu")
    xp = pick_backend("numpy", "cpu")
    assert xp.clip(np.array([-2.0, 0.5, 2.0]), -1, 1).tolist() == [-1.0, 0.5, 1.0]
