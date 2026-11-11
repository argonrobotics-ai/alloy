# SPDX-License-Identifier: Apache-2.0
"""One test per port family (CPU): the family's quickest env, stepped with ``info["expert_action"]``, lands on the
port's own run bit for bit after a few periods (particles too), and another action takes it elsewhere. Painting's is in test_ports.py (brush-stroke)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from alloy.envs import make
from alloy.envs.ports import PORTS
from alloy.envs.ports.run import PortRun


def _state(sim) -> list[np.ndarray]:  # noqa: ANN001
    st = sim.state
    out = [st.joint_q.numpy(), st.body_q.numpy()]
    if getattr(st, "particle_q", None) is not None:
        out.append(st.particle_q.numpy())
    return out


def _nudge(a: np.ndarray) -> np.ndarray:
    return a + np.float32(0.05)


def _drive(a: np.ndarray) -> np.ndarray:
    """Forklift: key and engine on, forward, full throttle, brake off."""
    a = a.copy()
    a[:, [0, 1, 4, 8, 9]] = [1.0, 0.0, 1.0, 1.0, 1.0]
    return a


def _open_levers(a: np.ndarray) -> np.ndarray:
    """Drain: grip every lever and pull it 0.5 rad."""
    a = a.copy()
    a[:, :3] = 0.5
    a[:, 3:6] = 1.0
    return a


FAMILIES = [
    ("RinseDrainWash-v0", 4, 10, _open_levers),
    ("GiftOpenFlaps-v0", 4, 20, _nudge),
    ("CookReheatPorridge-v0", 3, 5, _nudge),
    ("ForkliftStartAndPark-v0", 4, 25, _drive),
    ("ConcretePatchFlush-v0", 3, 10, _nudge),
]


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
@pytest.mark.parametrize(("name", "steps", "more", "other"), FAMILIES, ids=[f[0] for f in FAMILIES])
def test_the_env_steps_as_its_port(name: str, steps: int, more: int,
                                   other: Callable[[np.ndarray], np.ndarray]) -> None:
    spec = PORTS[name]
    ref = PortRun(lambda: spec.run(worlds=1, device="cpu", seed=0))
    assert ref.start()
    env = make(name, device="cpu", camera=(16, 16), spp=1)
    obs, info = env.reset(seed=0)
    assert obs["state"].shape == (1, env.single_observation_space["state"].shape[0])
    assert info["expert_action"].shape == (1, env.single_action_space.shape[0])
    for _ in range(steps):
        assert ref.resume()
        obs, reward, terminated, truncated, info = env.step(info["expert_action"])
        assert not terminated.any()
    sim = env._slots[0].run.sim
    for a, b in zip(_state(ref.sim), _state(sim), strict=True):
        assert np.array_equal(a, b)
    for _ in range(more):
        assert ref.resume()
        env.step(other(info["expert_action"]))
    assert max(float(np.abs(a - b).max()) for a, b in zip(_state(ref.sim), _state(sim), strict=True)) > 1e-4
    ref.abort()
    env.close()


def test_every_family_is_registered() -> None:
    families = {}
    for spec in PORTS.values():
        families.setdefault(spec.ga_task.split("/")[0], []).append(spec.name)
    assert {k: len(v) for k, v in families.items()} == {"painting": 8, "surface_rinsing": 7, "gift_bags": 2,
                                                         "ingredient_mixing": 1, "forklift": 5,
                                                         "concrete_finishing": 5}
