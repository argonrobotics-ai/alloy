# SPDX-License-Identifier: Apache-2.0
"""Regression tests for _void NaN hardening, next_step voiding, SkrlEnv reset/bootstrap, and ActionDelay."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.envs import actuators, realism, vector as vec
from alloy.envs.backend import NumpyBackend


def test_void_replaces_nan_and_inf_in_masked_worlds():
    """_void must replace NaN/Inf in invalid worlds with 0 rather than leaving NaN * 0 = NaN."""
    mask = np.array([False, True, False])
    obs = {"policy": np.array([[1.0, 2.0], [np.nan, np.inf], [3.0, 4.0]], dtype=np.float32)}
    out = vec._void(obs, mask)
    assert out["policy"].dtype == np.float32 and np.isfinite(out["policy"]).all()
    np.testing.assert_array_equal(out["policy"][1], [0.0, 0.0])
    np.testing.assert_array_equal(out["policy"][0], [1.0, 2.0])


def test_finish_pending_voids_observations_of_still_invalid_worlds(monkeypatch):
    """next_step autoreset must still void observations of worlds that remain invalid after pending resets."""
    xp = NumpyBackend("cpu")
    env = object.__new__(vec.VectorEnv)
    env.xp = xp
    env._pending = np.array([True, False])
    monkeypatch.setattr(vec.VectorEnv, "state", property(lambda self: object()))
    env.writer = object()
    env.reset_worlds = lambda mask: None
    env._observe = lambda state, only=None: np.array([[1.0, 2.0], [np.nan, 9.0]], dtype=np.float32)

    info = {
        "success": np.array([False, False]),
        "invalid": np.array([False, True]),
        "terms": {"r": np.array([1.0, 2.0], dtype=np.float32)},
    }
    obs, rew, term, trunc = env._finish_pending(
        None,
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([False, True]),
        np.array([False, False]),
        info,
    )
    assert np.isfinite(obs).all()
    np.testing.assert_array_equal(obs[1], [0.0, 0.0])
    np.testing.assert_array_equal(obs[0], [1.0, 2.0])
    assert float(rew[0]) == 0.0 and bool(term[1]) and not bool(trunc[1])


def test_skrl_just_restarted_requires_all_worlds_done():
    """A partial episode end must not set _just_restarted=True and skip a subsequent full reset()."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("skrl")
    from alloy.envs.wrappers import for_skrl

    reset_calls = [0]

    class _StubEnv:
        num_envs = 2
        obs_groups = ()

        def reset(self):
            reset_calls[0] += 1
            return torch.zeros((2, 3), dtype=torch.float32), {}

        def step(self, actions):
            return (
                torch.ones((2, 3), dtype=torch.float32),
                torch.zeros(2, dtype=torch.float32),
                torch.tensor([True, False]),
                torch.tensor([False, False]),
                {"invalid": torch.tensor([False, False])},
            )

    env = object.__new__(for_skrl.SkrlEnv)
    env._env = _StubEnv()
    env._unwrapped = env._env
    env._tdevice = torch.device("cpu")
    env._obs = torch.zeros((2, 3), dtype=torch.float32)
    env._just_restarted = False
    env._count = lambda info: None

    env.step(torch.zeros((2, 1), dtype=torch.float32))
    assert env._just_restarted is False
    env.reset()
    assert reset_calls[0] == 1


def test_skrl_with_bootstrap_handles_distinct_critic_rank(monkeypatch):
    """_with_bootstrap must reshape done_s using now_state.dim() when obs is 3-D and critic state is 2-D."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("skrl")
    from alloy.envs.wrappers import for_skrl

    env = object.__new__(for_skrl.SkrlEnv)
    env._env = SimpleNamespace(num_envs=2, obs_groups=("policy", for_skrl.CRITIC))
    env._unwrapped = env._env
    now_obs = torch.zeros((2, 4, 3), dtype=torch.float32)       # 3-D policy observation
    now_state = torch.ones((2, 5), dtype=torch.float32)         # 2-D critic state
    env._current = lambda: (now_obs, now_state)
    monkeypatch.setattr(for_skrl, "state_groups", lambda _e, raw: raw)

    info = {
        "_final_obs": torch.tensor([True, False]),
        "final_obs": {
            "policy": torch.full((2, 4, 3), 9.0, dtype=torch.float32),
            for_skrl.CRITIC: torch.full((2, 5), 7.0, dtype=torch.float32),
        },
    }
    out = env._with_bootstrap(info)
    assert out[for_skrl.BOOTSTRAP_STATES].shape == (2, 5)
    assert out[for_skrl.BOOTSTRAP_STATES][0, 0].item() == pytest.approx(7.0)
    assert out[for_skrl.BOOTSTRAP_STATES][1, 0].item() == pytest.approx(1.0)


def test_action_delay_attach_and_on_action_work_with_actuator_writer():
    """ActionDelay.attach() + on_action() must hold ActuatorWriter's initial command without touching .target."""
    xp = NumpyBackend("cpu")
    writer = object.__new__(actuators.ActuatorWriter)
    writer.xp = xp
    writer.joints = ["j0", "j1"]
    writer.spec = SimpleNamespace(mode="absolute")
    writer.command = np.array([[0.5, -0.25], [0.1, 0.2]], dtype=np.float32)

    stub_env = SimpleNamespace(xp=xp, num_envs=2, writer=writer)
    layer = realism.ActionDelay(1)
    layer.attach(stub_env)
    a1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    a2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    out1 = layer.on_action(a1)
    np.testing.assert_allclose(out1, [[0.5, -0.25], [0.1, 0.2]])
    writer.command[0, 0] = 99.0  # the layer must hold a copy, not a view of the live command
    np.testing.assert_allclose(layer.first, [[0.5, -0.25], [0.1, 0.2]])
    out2 = layer.on_action(a2)
    np.testing.assert_allclose(out2, a1)
