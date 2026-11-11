# SPDX-License-Identifier: Apache-2.0
"""Task ports as environments (alloy.envs.ports): the coroutine run, the env over a synthetic port, and the painting
family on its port (CPU).

The synthetic port is a kinematic tool in a one-body harness scene whose script places it 1 cm further each period
and whose verdict is "the tool ended at least 2.5 cm out": the expert's actions pass, holding it at 0 fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import make
from alloy.envs.errors import EnvError
from alloy.envs.ports import PORTS, PortEnv, PortSpec
from alloy.envs.ports.movers import KinematicBody, Mover
from alloy.envs.ports.run import PortRun

PERIODS = 4


def _scene(worlds: int):  # noqa: ANN202 - a harness TestScene
    from alloy.plugins.harness.scene import TestScene, body_xform  # noqa: PLC0415

    def build(b) -> None:  # noqa: ANN001 - newton.ModelBuilder
        tool = b.add_body(xform=body_xform(0.0, 0.0, 0.5), is_kinematic=True, label="tool")
        b.add_shape_sphere(tool, radius=0.02)
        dummy = b.add_body(xform=body_xform(2.0, -2.0, 1.0), label="dummy")
        b.add_shape_sphere(dummy, radius=0.01)

    return TestScene(build=build, init=None, worlds=worlds, periods=1, control_dt=0.01, dt=0.01, solver="xpbd",
                     ground=False)


def slide_port(worlds: int = 1, device: str = "cpu", seed: int = 0, periods: int = PERIODS) -> dict:  # noqa: ARG001
    """The synthetic port: place the tool at x = 1 cm x period, then judge where it ended."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415

    sim = build_simulation(_scene(worlds), [], device, worlds)
    idx = [i for i, lbl in enumerate(sim.model.body_label) if str(lbl).endswith("tool")]
    for k in range(periods):
        q = sim.state.body_q.numpy().copy()
        q[idx, 0] = 0.01 * k
        sim.state.body_q.assign(q)
        sim.step_period()
    x = sim.state.body_q.numpy()[idx, 0]
    return {"x": x.tolist(), "succeeded": bool((x >= 0.025).all())}


SLIDE = PortSpec("Slide-v0", "synthetic/slide", slide_port, lambda: Mover(KinematicBody("tool")),
                 lambda r, w: r["x"][w] >= 0.025, batched=True, look_at=((1.0, -1.0, 1.0), (0.0, 0.0, 0.5)))


def test_a_port_run_pauses_every_period_and_returns_its_verdict() -> None:
    run = PortRun(lambda: slide_port(1))
    alive, pauses = run.start(), 0
    while alive:
        pauses += 1
        alive = run.resume()
    assert pauses == PERIODS and run.steps == PERIODS
    assert run.result["succeeded"] and run.result["x"] == pytest.approx([0.03])


def test_an_aborted_run_unwinds() -> None:
    run = PortRun(lambda: slide_port(1))
    assert run.start()
    run.abort()
    assert run.finished and run.result is None


def test_a_port_error_reaches_the_caller() -> None:
    def broken() -> None:
        raise ValueError("the port broke")

    with pytest.raises(ValueError, match="the port broke"):
        PortRun(broken).start()


def test_the_expert_passes_and_a_held_tool_fails() -> None:
    env = PortEnv(SLIDE, num_envs=2, device="cpu", autoreset="disabled")
    assert env.single_action_space.shape == (13,) and env.observation_space.shape == (2, 14)
    obs, info = env.reset(seed=0)
    for k in range(PERIODS):
        a = info["expert_action"]
        assert a[:, 0] == pytest.approx([0.01 * k] * 2)
        if k == PERIODS - 1:
            a = a.copy()
            a[1, 0] = 0.0                                  # world 1: the policy keeps the tool home
        obs, reward, terminated, truncated, info = env.step(a)
    assert terminated.all() and not truncated.any()
    assert info["success"].tolist() == [True, False] and reward.tolist() == [1.0, 0.0]
    assert info["episode"]["l"].tolist() == [PERIODS, PERIODS]
    env.close()


def test_a_finished_run_starts_again_in_the_same_step() -> None:
    env = PortEnv(SLIDE, num_envs=1, device="cpu", camera=(16, 12), spp=1)
    obs, info = env.reset(seed=0)
    assert obs["image"].shape == (1, 12, 16, 3) and obs["image"].dtype == np.uint8
    for _ in range(PERIODS):
        obs, reward, terminated, truncated, info = env.step(info["expert_action"])
    assert terminated.all() and info["_final_obs"].all()
    assert info["final_obs"]["state"][0, 0] == pytest.approx(0.03)      # the last placement, before the restart
    assert obs["state"][0, 0] == pytest.approx(0.0)                     # the new run's start
    env.close()


def test_wrong_action_sizes_and_unknown_names_are_refused() -> None:
    env = PortEnv(SLIDE, num_envs=1, device="cpu")
    with pytest.raises(EnvError, match="13"):
        env.step(np.zeros((1, 3), np.float32))
    env.close()
    with pytest.raises(EnvError, match="PaintRollStrip-v0"):
        make("PaintNoSuchTask-v0")


def test_the_painting_family_is_registered() -> None:
    painting = sorted(n for n, s in PORTS.items() if s.ga_task.startswith("painting/"))
    assert painting == sorted(["PaintBehindRadiator-v0", "PaintBrushStroke-v0", "PaintClearCoat-v0",
                               "PaintCutInTrim-v0", "PaintLoadRoller-v0", "PaintRollStrip-v0",
                               "PaintSprayStencil-v0", "PaintSwatchCard-v0"])


def _film_state(sim) -> list[np.ndarray]:  # noqa: ANN001
    flow = next(p for p in sim.plugins.plugins.values() if hasattr(p, "depth"))
    return [sim.state.joint_q.numpy(), sim.state.body_q.numpy(), flow.depth(0)]


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_the_brush_stroke_env_steps_as_its_port() -> None:
    """PaintBrushStroke-v0 (an FR3 and the round brush on paper): 6 periods with the expert's actions land on the
    port's own state bit for bit; another action moves the arm elsewhere."""
    from alloy.envs.ports.painting import run_brush_stroke  # noqa: PLC0415

    ref = PortRun(lambda: run_brush_stroke(1, "cpu", 0))
    assert ref.start()
    for _ in range(6):
        assert ref.resume()
    env = make("PaintBrushStroke-v0", device="cpu", camera=(16, 16), spp=1)
    assert env.single_action_space.shape == (7,) and env.single_observation_space["state"].shape == (15,)
    obs, info = env.reset(seed=0)
    for _ in range(6):
        obs, reward, terminated, truncated, info = env.step(info["expert_action"])
    sim = env._slots[0].run.sim
    for a, b in zip(_film_state(ref.sim), _film_state(sim), strict=True):
        assert np.array_equal(a, b)
    for _ in range(5):
        assert ref.resume()
        obs, *_ = env.step(info["expert_action"] + np.float32(0.05))
    assert np.abs(sim.state.joint_q.numpy() - ref.sim.state.joint_q.numpy()).max() > 1e-3
    assert obs["image"].shape == (1, 16, 16, 3)
    ref.abort()
    env.close()
