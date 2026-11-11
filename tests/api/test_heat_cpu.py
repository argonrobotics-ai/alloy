# SPDX-License-Identifier: Apache-2.0
"""Heat in the Scene API: heaters, start temperatures and read-outs on the heat v2 plug-in (CPU).

The quick start of the heat guide (docs/notes/heat-v2-guide.md) through ``alloy.Scene`` / ``alloy.build`` gives the
same numbers as the plug-in's own path (``particle_scene.for_scene`` + ``alloy.plugins.simulation``): mean 307.548 K
after 90 s of heating at time scale 60, heater contact 7.02e-3 m² (CPU, 2026-11-06).
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy.api as alloy

T0, T_PAN = 293.15, 353.15
FLOOR = 0.08 * 0.08


def pan_scene(**heat) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("pan", inner=(0.08, 0.08, 0.04), wall=0.005, pos=(0.0, 0.0, 0.02))
    scene.add_liquid("water", size=(0.07, 0.07, 0.02), pos=(0.0, 0.0, 0.0355), temperature=T0)
    scene.set_heater("pan_floor", **(heat or {"temperature": T_PAN}))
    scene.set_heat(time_scale=60.0)
    return scene


@pytest.mark.compile_heavy
def test_guide_quick_start_through_the_api_heats_records_and_resets():
    sim = alloy.build(pan_scene(), device="cpu")
    traj = sim.run(1.5, record=["water"], every=50)
    s = sim.heat("water")
    assert s["mean_K"][0] == pytest.approx(307.548, abs=0.01)               # the plug-in's own path: 307.5482
    assert T0 <= s["min_K"][0] < s["mean_K"][0] < s["max_K"][0] <= T_PAN
    area = sim.plugins.read()["particle_thermal"]["heater_area_m2"]
    assert area[0, 0] > 0.9 * FLOOR
    t = traj["water.temperature"]                                            # (frames, worlds, particles)
    assert t.shape == traj["water.points"].shape[:3] and np.all(np.diff(t[:, 0].mean(axis=1)) > 0)
    assert np.array_equal(sim.state(["water"])["water.temperature"], sim.temperature("water"))
    assert sim.check()["ok"]
    sim.reset()
    assert np.allclose(sim.temperature("water"), T0)                          # plug-in state is in the snapshots
    sim.close()


@pytest.mark.compile_heavy
def test_power_heater_per_world_and_runtime_change():
    sim = alloy.build(pan_scene(power=40.0), worlds=2, device="cpu")
    sim.set_heater("pan_floor", power=np.array([40.0, 0.0]))                 # world 1: heater off
    sim.run(0.3)
    mean = sim.heat("water")["mean_K"]
    assert mean[0] > T0 + 0.5 and mean[1] == pytest.approx(T0, abs=1e-3)
    with pytest.raises(alloy.SimulationError, match="power heater"):
        sim.set_heater("pan_floor", temperature=T_PAN)
    with pytest.raises(alloy.SimulationError, match="not a heater"):
        sim.set_heater("pan_wall0", power=1.0)
    sim.close()


def test_scene_heat_checks_say_what_to_change():
    scene = alloy.Scene()
    scene.add_box("block", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.05))
    with pytest.raises(alloy.SceneError, match="kelvin") as err:
        scene.add_liquid("tea", size=(0.03, 0.03, 0.02), pos=(0, 0, 0.2), temperature=80)
    assert "353.15" in err.value.fix and "tea" not in scene.items           # nothing half-added
    with pytest.raises(alloy.SceneError, match="liquids and particle objects"):
        scene.set_temperature("block", T0)
    with pytest.raises(alloy.SceneError, match="static box"):
        scene.set_heater("block", temperature=T_PAN)
    with pytest.raises(alloy.SceneError, match="exactly one"):
        scene.add_heater("hob", size=(0.1, 0.1, 0.01), pos=(0, 0, 0.3), temperature=T_PAN, power=10.0)
    assert "hob" not in scene.items
    with pytest.raises(alloy.SceneError, match="no heat data"):
        scene.add_liquid("oil", size=(0.03, 0.03, 0.02), pos=(0, 0, 0.2), temperature=T0, heat_material="olive")
    with pytest.raises(alloy.SceneError, match="unknown heat setting"):
        scene.set_heat(timescale=60)
    scene.add_heater("hob", size=(0.1, 0.1, 0.01), pos=(0, 0, 0.3), temperature=T_PAN)
    with pytest.raises(alloy.SceneError, match="nothing to heat"):
        alloy.build(scene, device="cpu")
    back = alloy.Scene.from_dict(pan_scene().to_dict())
    assert back.thermal["heaters"] == {"pan_floor": ["temperature", T_PAN]}
    assert back.thermal["objects"]["water"] == {"temperature": T0, "material": "water"}
    assert back.thermal["config"] == {"time_scale": 60.0}


def test_add_pour_carries_its_temperature():
    scene = alloy.Scene()
    scene.add_pour("hot", 50e-6, pos=(0, 0, 0.2), temperature=363.15)
    assert scene.thermal["objects"]["hot"] == {"temperature": 363.15, "material": "water"}


@pytest.mark.compile_heavy
def test_build_runs_your_own_plugins_inside_the_api_sim():
    from alloy.plugins.library.magnet import Magnet

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_sphere("a", 0.01, pos=(-0.025, 0, 0.01), mass=0.05)
    scene.add_sphere("b", 0.01, pos=(0.025, 0, 0.01), mass=0.05)
    pair = lambda built: Magnet(bodies=("a", "b"), moments=(1.0, 1.0), axes=((1, 0, 0), (1, 0, 0)))  # noqa: E731
    sim = alloy.build(scene, device="cpu", plugins=[pair])
    gap = sim.run(0.3, record=["a", "b"]).world(0)
    d = gap["b.pos"][:, 0] - gap["a.pos"][:, 0]
    assert d[0] == pytest.approx(0.05) and d[-1] < 0.045                    # coaxial dipoles pull together
    assert sorted(sim.plugins.plugins) == ["magnet"] and sim.check()["ok"]
    sim.close()
    with pytest.raises(alloy.BuildError, match="not a plug-in"):
        alloy.build(scene, device="cpu", plugins=["magnet"])


def test_code_that_makes_its_own_plugin_host_can_attach_it():
    """``alloy.plugins.species.build`` installs its own host and assigns ``sim.plugins`` (the getter reads it back)."""
    from types import SimpleNamespace

    fake = SimpleNamespace(engine=SimpleNamespace())
    alloy.Sim.plugins.fset(fake, "host")
    assert fake.engine.plugins == "host" and alloy.Sim.plugins.fget(fake) == "host"
