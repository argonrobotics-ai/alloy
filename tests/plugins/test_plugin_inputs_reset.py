# SPDX-License-Identifier: Apache-2.0
"""Gemini 0.1.2 audit S1-019: ``sim.reset()`` returns plug-in inputs (``add_input``) to their value at the build, or
at the last ``sim.settle``, as joint targets: a heater set to 500 K and an ambient of 450 K in one episode no longer
drive the next (they leaked 3.5 K into 5 periods after the reset). A masked reset restores only its worlds; a
plug-in's state keys are unchanged. CPU, a pan of water on a heater, 2 worlds."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.plugins.host import PREFIX


def _pan() -> alloy.Sim:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("pan", inner=(0.08, 0.08, 0.04), wall=0.005, pos=(0.0, 0.0, 0.01))
    scene.add_liquid("water", size=(0.04, 0.04, 0.012), pos=(0.0, 0.0, 0.022), temperature=293.15)
    scene.set_heater("pan_floor", temperature=300.0)
    scene.set_heat(time_scale=60.0, ambient=293.15, convection=20.0)
    return alloy.build(scene, worlds=2, device="cpu")


def _inputs(sim: alloy.Sim) -> tuple[np.ndarray, np.ndarray]:
    inputs = sim.plugins["particle_thermal"].inputs()
    return inputs["heater_values"].numpy().copy(), inputs["ambient"].numpy().copy()


def _crank(sim: alloy.Sim) -> None:
    sim.set_heater("pan_floor", temperature=500.0)
    sim.plugins["particle_thermal"].set_input("ambient", np.array([450.0, 450.0]))


@pytest.fixture(scope="module")
def sim():
    s = _pan()
    yield s
    s.close()


def test_reset_returns_heater_and_ambient_to_the_build(sim):
    sim.reset()
    built = _inputs(sim)
    sim.step(5)
    first = sim.temperature("water").copy()
    sim.reset()
    _crank(sim)
    sim.step(5)
    assert not np.array_equal(sim.temperature("water"), first)
    sim.reset()
    for now, then in zip(_inputs(sim), built):
        np.testing.assert_array_equal(now, then)
    sim.step(5)
    np.testing.assert_array_equal(sim.temperature("water"), first)      # the episode repeats bit for bit


def test_a_masked_reset_restores_only_its_worlds(sim):
    sim.reset()
    built = _inputs(sim)
    _crank(sim)
    mask = wp.array(np.array([0, 1], dtype=np.int32), dtype=wp.int32, device=sim.engine.model.device)
    sim.engine.reset(mask)
    heater, ambient = _inputs(sim)
    np.testing.assert_array_equal(heater[1], built[0][1])
    assert ambient[1] == built[1][1] and ambient[0] == 450.0 and np.all(heater[0] == 500.0)


def test_settle_keeps_the_inputs_it_ran_with():
    s = _pan()
    try:
        s.set_heater("pan_floor", temperature=400.0)
        s.settle(0.02)
        _crank(s)
        s.reset()
        heater, ambient = _inputs(s)
        assert np.all(heater == 400.0) and np.all(ambient == pytest.approx(293.15))
    finally:
        s.close()


def test_state_keys_are_unchanged_and_inputs_listed_beside_them(sim):
    host = sim.plugins
    keys = set(host.state_arrays())
    p = host["particle_thermal"]
    state = {f"{PREFIX}/time"} | {f"{PREFIX}/particle_thermal/{k}" for k in p.state_arrays()}
    assert {k for k in keys if "/input:" not in k} == state
    assert {k for k in keys if "/input:" in k} == {f"{PREFIX}/particle_thermal/input:{k}" for k in p.inputs()}
