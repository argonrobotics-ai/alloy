# SPDX-License-Identifier: Apache-2.0
"""``scene.set_joint_yield`` (SCR-WP7-15 P2): arguments, Scene JSON, build-time checks and the
``"<joint>.plastic_strain"`` readout. The physics laws are in tests/plugins/test_joint_yield.py."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import alloy
from alloy.plugins.joint_yield import laws


def _scene(rig: laws.Rig = laws.J2_RIG, **yield_kw) -> alloy.Scene:
    scene = alloy.Scene(dt=laws.STEP_S)
    scene.add_robot("rig", mjcf=rig.mjcf(), pos=(0, 0, 0.5))
    if yield_kw:
        scene.set_joint_yield("rig/j", **yield_kw)
    return scene


@pytest.mark.parametrize("kw, field", [({}, "torque"), ({"torque": 1.0, "force": 1.0}, "torque"),
                                       ({"torque": 0.0}, "torque"), ({"force": -1.0}, "force"),
                                       ({"torque": math.nan}, "torque"), ({"torque": "x"}, "torque"),
                                       ({"torque": 1.0, "hardening": -0.1}, "hardening")])
def test_bad_arguments_name_the_field(kw, field):
    scene = alloy.Scene()
    with pytest.raises(alloy.SceneError) as e:
        scene.set_joint_yield("arm/hinge", **kw)
    assert e.value.field == field and e.value.fix


def test_scene_json_keeps_the_yield():
    plain = _scene()
    assert "joint_yield" not in plain.to_dict()                    # scenes without it serialize as before
    scene = _scene(force=2.0, hardening=10.0)
    back = alloy.Scene.from_dict(json.loads(scene.to_json()))
    assert back.joint_yield == {"rig/j": {"kind": "force", "limit": 2.0, "hardening": 10.0}}


def test_build_checks_the_joint():
    with pytest.raises(alloy.UnknownNameError):
        alloy.build(_scene_named("rig/nope"), device="cpu")
    with pytest.raises(alloy.BuildError, match="slide"):
        alloy.build(_scene(torque=2.0), device="cpu")              # a slide takes force=
    springless = laws.Rig("hinge", k=0.0, limit=0.1, hardening=0.0, kv=0.5, mass=0.5, segments=())
    with pytest.raises(alloy.BuildError, match="stiffness"):
        alloy.build(_scene(springless, torque=0.1), device="cpu")


def _scene_named(joint: str) -> alloy.Scene:
    scene = _scene()
    scene.set_joint_yield(joint, force=2.0)
    return scene


def test_plastic_strain_readout():
    sim = alloy.build(_scene(force=2.0), device="cpu")
    st = sim.state()
    assert st["rig/j.plastic_strain"].shape == (1,) and st["rig/j.plastic_strain"][0] == 0.0
    sim.set_targets({"rig/j": 0.1}, kind="velocity")
    traj = sim.run(0.1, record=["j"])                               # the bare joint name works too
    strain = traj["j.plastic_strain"][:, 0]
    assert strain[-1] > 1e-4 and (np.diff(strain) >= 0).all()
    no_yield = alloy.build(_scene(), device="cpu")
    assert "rig/j.plastic_strain" not in no_yield.state()


def test_coupled_scene_is_refused():
    """Beside cloth the robot steps on a coupled stepper, which joint yield does not support yet: the build says so."""
    scene = _scene(force=2.0)
    scene.add_cloth("sheet", size=(0.2, 0.2), pos=(1.0, 0, 0.3), resolution=4, mass=0.02)
    with pytest.raises(alloy.BuildError, match="plasticity-metal.md"):
        alloy.build(scene, device="cpu")


def test_rest_after_a_reversed_load():
    """``"<joint>.rest"`` is where the spring rests now (springref + set); after a reversed load it differs from the
    accumulated |set| (``plastic_strain``)."""
    rig = laws.J2_RIG
    sim = alloy.build(_scene(force=rig.limit), device="cpu")
    for v in (0.1, -0.1):
        sim.set_targets({"rig/j": v}, kind="velocity")
        sim.run(0.2)
    st = sim.state(["rig/j"])
    q_p = sim.engine.plugins["joint_yield"].q_p.numpy()[:, 0]
    np.testing.assert_allclose(st["rig/j.rest"], rig.springref + q_p, rtol=0, atol=1e-9)
    assert st["rig/j.plastic_strain"][0] > abs(q_p[0]) + 1e-3                  # yielded both ways
