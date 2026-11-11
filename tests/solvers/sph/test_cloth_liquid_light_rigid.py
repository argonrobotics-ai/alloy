# SPDX-License-Identifier: Apache-2.0
"""Cloth, liquid and a light free rigid body (audit WORLDS-3, 2026-11-09), CPU.

The rigid / cloth pair is strong when the cloth weighs more than half the free rigid bodies, the strong pair runs on
the ADMM coupler, and ADMM cannot step a liquid. Such a scene is refused at build: the error used to name solver
entries only ("mujoco/vbd is strong (D5) and needs ADMM, which has no one-way stage for SPH") with the fix "check the
object named in the message". It now names the objects, their masses and the lightest rigid mass that builds, and the
scene builds at that mass (docs/known_gaps/liquid-cloth-light-rigid.md).
"""

from __future__ import annotations

import pytest

import alloy


def _scene(ball_mass: float | None, **ball) -> alloy.Scene:
    sc = alloy.Scene()
    sc.add_ground()
    sc.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, 0.3), resolution=8, mass=0.02)
    sc.add_open_box("tub", inner=(0.06, 0.06, 0.06), wall=0.01, pos=(0.5, 0, 0))
    sc.add_liquid("water", size=(0.04, 0.04, 0.04), pos=(0.5, 0, 0.03))
    sc.add_sphere("ball", 0.02, pos=(1, 1, 0.02), **({} if ball_mass is None else {"mass": ball_mass}), **ball)
    return sc


def test_a_light_free_body_beside_cloth_and_liquid_is_refused_with_the_mass_to_use():
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(_scene(None), device="cpu")              # the ball: 34.9 g by its default material
    e = err.value
    assert "['ball']" in e.message and "['sheet']" in e.message and "0.02 kg" in e.message
    assert "at least 0.0401 kg" in e.fix and "static=True or kinematic=True" in e.fix
    assert "liquid-cloth-light-rigid.md" in e.fix
    assert "check the object named" not in e.fix
    assert not [w for w in ("ADMM", "D5", "m_a/m_s", "SCR-", "mujoco/vbd") if w in e.message + e.fix], e.message


@pytest.mark.compile_heavy
def test_the_scene_builds_at_the_mass_the_error_names():
    sim = alloy.build(_scene(0.0401), device="cpu")
    assert {r["route"] for r in sim.routes().values()} >= {"rigid", "cloth_vbd", "liquid_sph"}
    sim.close()


@pytest.mark.compile_heavy
@pytest.mark.parametrize("how", ["static", "kinematic"])
def test_the_scene_builds_with_the_light_body_static_or_kinematic(how):
    sim = alloy.build(_scene(None, **{how: True}), device="cpu")
    entries = sorted(sim.engine.stepper._entries)
    assert entries == (["kinematic", "sph", "vbd"] if how == "kinematic" else ["sph", "vbd"])
    sim.close()
