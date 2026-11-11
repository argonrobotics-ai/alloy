# SPDX-License-Identifier: Apache-2.0
"""Task-space (end-effector) actions: the Jacobian against MuJoCo, tracking, per-world motion, limits, errors.

CPU, a few worlds; Panda environments with the NumPy backend, one with torch."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.envs import ActionSpec, EnvError
from alloy.envs.actions import ActionWriter
from alloy.envs.diffik import EEActionSpec, EEWriter, ee_pose, jacobian, panda_reach_ee
from alloy.envs.tasks import panda_reach_task, panda_scene
from alloy.envs.vector import VectorEnv, _make_writer

ARM = [f"panda/actuator{k}" for k in range(1, 8)]
WORLDS = 3
# Measured on CPU: along the move up to 5 mm off; off the axis up to 8 mm (the Panda's servos have no gravity
# compensation, so the sag changes as the arm reaches). Limits are about twice those.
AXIAL, OFF_AXIS = 8e-3, 15e-3
NO_WINDOW = float("inf")     # the solve itself, open loop: the default window keeps targets near the measured joints


def _env(spec: EEActionSpec, worlds: int = WORLDS) -> VectorEnv:
    task = dataclasses.replace(panda_reach_task(max_steps=1000), actions=spec, name="PandaReachEE")
    return VectorEnv(panda_scene(worlds, 0, props=False), task, worlds=worlds, device="cpu", backend="numpy")


@pytest.fixture(scope="module")
def env():
    e = _env(EEActionSpec(body="panda/hand", joints=ARM, window=NO_WINDOW))
    yield e
    e.close()


@pytest.fixture(scope="module")
def full_env():
    spec = EEActionSpec(body="panda/hand", joints=ARM, orientation=True, extra=["panda/hand_finger"],
                        extra_delta=0.01, window=NO_WINDOW)
    e = _env(spec, worlds=2)
    yield e
    e.close()


def _hand(e: VectorEnv) -> np.ndarray:
    return np.array(e.state["panda/hand.pos"], np.float64)


def _run(e: VectorEnv, action: np.ndarray, steps: int) -> None:
    for _ in range(steps):
        e.step(np.broadcast_to(np.asarray(action, np.float32), (e.num_envs, e.writer.size)).copy())


def _angle_between(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angle (degrees) of the rotation taking quaternion q2 to q1 (xyzw, (W, 4))."""
    dot = np.abs(np.sum(q1 * q2, axis=1)) / (np.linalg.norm(q1, axis=1) * np.linalg.norm(q2, axis=1))
    return np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))


def _mujoco_jacobian(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import mujoco
    from alloy.cards.library import default_root

    model = mujoco.MjModel.from_xml_path(str(default_root() / "robots" / "emb_e2_panda" / "scene.xml"))
    data = mujoco.MjData(model)
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{k}") for k in range(1, 8)]
    data.qpos[[model.jnt_qposadr[j] for j in joints]] = q
    mujoco.mj_forward(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jp, jr, body)
    dofs = [model.jnt_dofadr[j] for j in joints]
    return np.concatenate([jp[:, dofs], jr[:, dofs]]), data.xpos[body].copy()


@pytest.mark.compile_heavy
def test_jacobian_matches_mujoco_at_three_configurations_per_world(env):
    rng = np.random.default_rng(3)
    env.reset()
    worst = 0.0
    for _ in range(3):
        _run(env, rng.uniform(-1, 1, (WORLDS, 3)), 12)
        jac, pos = np.array(jacobian(env), np.float64), _hand(env)
        for w in range(WORLDS):
            q = np.array([float(env.state[f"panda/joint{k}.q"][w]) for k in range(1, 8)])
            ref, ref_pos = _mujoco_jacobian(q)
            worst = max(worst, float(np.abs(jac[w] - ref).max()))
            assert np.abs(pos[w] - ref_pos).max() < 5e-3                    # same frame: the base sits at the origin
    print(f"max Jacobian error vs MuJoCo: {worst:.2e}")
    assert worst < 1e-4


@pytest.mark.compile_heavy
def test_straight_moves_track_the_command(env):
    env.reset()
    _run(env, [0.0, 0.0, 0.0], 40)                      # the arm sags to its servo equilibrium first
    start = _hand(env)
    _run(env, [1.0, 0.0, 0.0], 5)                       # 5 steps x 2 cm = 10 cm
    _run(env, [0.0, 0.0, 0.0], 40)                      # let the servos settle
    moved = _hand(env) - start
    print("x move error (m):", np.abs(moved - [0.1, 0, 0]).max(axis=0))
    assert np.abs(moved[:, 0] - 0.1).max() < AXIAL and np.abs(moved[:, 1:]).max() < OFF_AXIS
    mid = _hand(env)
    _run(env, [0.0, 0.0, 1.0], 5)
    _run(env, [0.0, 0.0, 0.0], 40)
    moved = _hand(env) - mid
    print("z move error (m):", np.abs(moved - [0, 0, 0.1]).max(axis=0))
    assert np.abs(moved[:, 2] - 0.1).max() < AXIAL and np.abs(moved[:, :2]).max() < OFF_AXIS


@pytest.mark.compile_heavy
def test_with_the_default_window_an_open_loop_move_falls_short():
    e = _env(EEActionSpec(body="panda/hand", joints=ARM), worlds=2)
    e.reset()
    _run(e, [0.0, 0.0, 0.0], 40)
    start = _hand(e)
    _run(e, [1.0, 0.0, 0.0], 5)                       # 10 cm asked in 0.1 s; the arm's servos lag about 0.1 s
    _run(e, [0.0, 0.0, 0.0], 40)
    moved = (_hand(e) - start)[:, 0]
    e.close()
    print("default window, 10 cm open-loop command, x moved (m):", moved)
    assert np.all((moved > 0.02) & (moved < 0.07))   # measured 4.1-4.2 cm


@pytest.mark.compile_heavy
def test_each_world_moves_its_own_way(env):
    env.reset()
    _run(env, [0.0, 0.0, 0.0], 40)
    start = _hand(env)
    _run(env, np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]]), 4)
    _run(env, [0.0, 0.0, 0.0], 40)
    moved = _hand(env) - start
    for w, axis in enumerate([0, 1, 2]):
        other = np.delete(np.abs(moved[w]), axis)
        assert abs(moved[w, axis]) > 0.06 and other.max() < OFF_AXIS
    assert moved[1, 1] < 0 and moved[0, 0] > 0 and moved[2, 2] > 0


@pytest.mark.compile_heavy
def test_orientation_is_held_and_rotation_columns_turn_the_hand(full_env):
    e = full_env
    e.reset()
    _, q0 = ee_pose(e)
    _run(e, [1, 0, 0, 0, 0, 0, 0], 5)
    _run(e, np.zeros(7), 30)
    _, q1 = ee_pose(e)
    held = _angle_between(np.array(q1), np.array(q0))
    print("orientation drift over a 10 cm move (deg):", held)
    assert held.max() < 2.0
    _run(e, [0, 0, 0, 0, 0, 1, 0], 5)                   # 5 x 0.1 rad about world z
    _run(e, np.zeros(7), 30)
    _, q2 = ee_pose(e)
    turned = _angle_between(np.array(q2), np.array(q1))
    print("turn about z for 0.5 rad commanded (deg):", turned)
    assert np.all(np.abs(turned - np.degrees(0.5)) < 4.0)


@pytest.mark.compile_heavy
def test_extra_columns_drive_the_gripper_and_columns_are_named(full_env):
    e = full_env
    assert e.action_names() == ["ee_dx", "ee_dy", "ee_dz", "ee_rx", "ee_ry", "ee_rz", "panda/hand_finger"]
    e.reset()
    before = np.array(e.xp.view(e.writer.inner.target)[e.writer.inner.rows][:, -1])
    targets = e.writer.apply(np.array([[0, 0, 0, 0, 0, 0, 1.0], [0, 0, 0, 0, 0, 0, -1.0]], np.float32))
    assert targets[0, -1] > before[0] and targets[1, -1] <= before[1]
    assert targets[0, -1] <= e.writer.inner.high_np[-1] + 1e-6


@pytest.mark.compile_heavy
def test_zero_action_holds_targets_and_limits_are_respected(env):
    env.reset()
    first = env.writer.apply(np.zeros((WORLDS, 3), np.float32)).copy()
    assert np.allclose(env.writer.apply(np.zeros((WORLDS, 3), np.float32)), first, atol=1e-6)
    _run(env, [1.0, 1.0, 1.0], 80)                      # far past the reach: joints pile up at their limits
    t = np.array(env.writer.apply(np.array([[1, 1, 1.0]] * WORLDS, np.float32)))
    inner = env.writer.inner
    assert np.isfinite(t).all() and np.isfinite(_hand(env)).all()
    assert (t >= inner.low_np - 1e-6).all() and (t <= inner.high_np + 1e-6).all()


@pytest.mark.compile_heavy
def test_default_spec_still_gets_a_plain_writer_and_bad_specs_raise(env):
    plain = _make_writer(ActionSpec(ARM, mode="delta"), env.layout, env.sim.engine, env.xp)
    assert type(plain) is ActionWriter and isinstance(env.writer, EEWriter)
    for bad in (EEActionSpec("panda/hand", []), EEActionSpec("panda/hand", ARM, damping=0.0),
                EEActionSpec("panda/hand", ARM, frame="tool"), EEActionSpec("panda/hand", ARM, kind="effort"),
                EEActionSpec("panda/hand", ARM, extra=["a", "b"], extra_delta=[0.1, 0.1, 0.1])):
        with pytest.raises(EnvError) as err:
            bad.validate()
        assert err.value.fix
    with pytest.raises(EnvError):
        EEActionSpec("panda/nothing", ARM).make_writer(env.layout, env.sim.engine, env.xp)
    with pytest.raises(EnvError):
        EEActionSpec("panda/hand", ["panda/hand"]).make_writer(env.layout, env.sim.engine, env.xp)
    with pytest.raises(EnvError):
        env.writer._check(np.zeros((WORLDS, 2)))


@pytest.mark.compile_heavy
def test_panda_reach_ee_with_torch_steps_and_resets_by_mask():
    torch = pytest.importorskip("torch")
    e = panda_reach_ee(num_envs=2, device="cpu", seed=1)
    try:
        assert e.xp.name == "torch"
        obs, _ = e.reset()
        assert isinstance(obs, torch.Tensor)
        for _ in range(3):
            obs, reward, term, trunc, info = e.step(torch.tensor([[1.0, 0, 0], [0, 0, -1.0]]))
        assert torch.isfinite(obs).all() and info["targets"].shape == (2, 7)
        assert float(jacobian(e).abs().max()) > 0.1
        e.reset(options={"reset_mask": torch.tensor([True, False])})
        assert e.state["steps"].tolist() == [0, 3]
    finally:
        e.close()
