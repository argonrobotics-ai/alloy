# SPDX-License-Identifier: Apache-2.0
"""A plug-in input's expert action is the script's command, not the policy's last write echoed back (audit T1-020).

``PluginInput.read`` (``info["expert_action"]``) read the input as it stood, and the policy's write had replaced the
script's value there: a script that sets a hob once (the porridge's 1156 W) then reported 0 W as the expert's hob
power for the rest of the episode after the policy turned it off for one period, and a valve the script opened was
reported closed. The script's writes are now kept apart from the policy's.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.envs import make
from alloy.envs.ports.movers import PluginArray, PluginInput


class _Valves:
    """A plug-in with one input per world and a plain array, as a port's script writes them."""

    def __init__(self) -> None:
        self.valve = SimpleNamespace(v=np.zeros(2, np.float32))
        self.release = _Array(np.zeros(2, np.float32))

    def inputs(self) -> dict:
        return {"valve": _Array(self.valve.v, self.valve)}

    def set_input(self, name: str, values) -> None:  # noqa: ANN001
        self.valve.v = np.broadcast_to(np.asarray(values, np.float32), (2,)).copy()


class _Array:
    def __init__(self, v: np.ndarray, owner=None) -> None:  # noqa: ANN001
        self.v, self.owner = v, owner

    def numpy(self) -> np.ndarray:
        return self.v if self.owner is None else self.owner.v

    def assign(self, values) -> None:  # noqa: ANN001
        self.v[:] = values


def _sim(plug: _Valves) -> SimpleNamespace:
    return SimpleNamespace(plugins=SimpleNamespace(plugins={"tap": plug}), model=SimpleNamespace(world_count=2))


@pytest.mark.parametrize("kind", [PluginInput, PluginArray])
def test_the_expert_is_the_scripts_last_write(kind):
    plug = _Valves()
    name = "valve" if kind is PluginInput else "release"
    part = kind(name)
    sim = _sim(plug)
    script = (lambda v: plug.set_input("valve", v)) if kind is PluginInput else (lambda v: plug.release.assign(v))
    script(np.array([1.0, 1.0]))                                      # the script opens both before the first pause
    part.bind(sim)
    part.write(sim, np.array([[0.0], [1.0]], np.float32))             # the policy closes world 0
    assert part.read(sim).ravel().tolist() == [1.0, 1.0]              # the script's command, not the policy's
    assert part._array().numpy().tolist() == [0.0, 1.0]               # what the plant gets this period
    script(np.array([0.0, 1.0]))                                      # the script itself closes world 0 later
    assert part.read(sim).ravel().tolist() == [0.0, 1.0]


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_the_porridge_hob_stays_the_scripts_after_the_policy_turns_it_off():
    env = make("CookReheatPorridge-v0", device="cpu")
    _obs, info = env.reset(seed=0)
    hob = info["expert_action"][:, -1].copy()
    assert np.all(hob > 100.0)                                         # watts, the script's draw
    a = info["expert_action"].copy()
    a[:, -1] = 0.0                                                     # the policy turns the hob off for one period
    _obs, _r, _d, _t, info = env.step(a)
    assert np.allclose(info["expert_action"][:, -1], hob)
    _obs, _r, _d, _t, info = env.step(info["expert_action"])
    assert np.allclose(info["expert_action"][:, -1], hob)
    env.close()
