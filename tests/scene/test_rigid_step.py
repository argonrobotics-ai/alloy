# SPDX-License-Identifier: Apache-2.0
"""SCR-WP0-RIGIDDT: the API picks the rigid step per scene (alloy.scene.rigid_step); every other scene keeps the fixed
step, bit for bit (CPU)."""

from __future__ import annotations

import logging

import numpy as np
import pytest

import alloy
from alloy.api.errors import SceneError
from alloy.scene import rigid_step as RS
from alloy.scene import rigid_step_bodies as RB

TODAY = 1.25e-3
ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def built_of(sim):
    return sim.engine.parts.built


def stack(worlds: int = 2, layers: int = 3) -> alloy.Scene:
    """Resting columns of 1 m, 10 kg cubes (XB4-9's box stack, 1-2 columns)."""
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground(friction=1.0)
    for i in range(2):
        for k in range(layers):
            z = np.full(worlds, 0.5 + k * 1.0 + 0.002 * k)
            pos = np.stack([np.full(worlds, 2.0 * i), np.zeros(worlds), z], axis=1)
            s.add_box(f"c{i}{k}", size=(1.0, 1.0, 1.0), pos=pos, density=10.0, friction=1.0)
    return s


def one_box(z: float = 0.426, size: float = 0.05) -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    s.add_box("block", size=(size, size, size), pos=(0, 0, z), mass=0.2)
    return s


def test_box_stack_steps_coarser_bound_by_rest_sag() -> None:
    sim = alloy.build(stack(), device="cpu")
    step = built_of(sim).report.rigid_step
    assert step.today == TODAY and step.dt > TODAY and step.dt in (2.5e-3, 5e-3, 1e-2)
    assert step.reason.startswith("sag cap, c0") or step.dt == 1e-2, step
    assert built_of(sim).rigid_dt == step.dt
    sim.step(2)
    assert sim.physics_steps == 2 * round(0.01 / step.dt) and sim.check()["ok"]
    mj = next(e for e in built_of(sim).entries if e.name == "mujoco")
    assert mj.target_dt == step.dt


def test_route_report_names_the_step() -> None:
    sim = alloy.build(stack(), device="cpu")
    reasons = sim.routes()["c00"]["reasons"]
    assert any("rigid step" in r and "scene.rigid_step = 'fixed'" in r for r in reasons), reasons


@pytest.mark.parametrize("case", ["robot", "kinematic", "precision", "debug", "fixed", "cloth"])
def test_ineligible_scenes_keep_todays_step_bitwise(case: str) -> None:
    """The default build of an ineligible scene equals its rigid_step="fixed" build: same step and model digest."""
    def scene() -> alloy.Scene:
        s = one_box()
        if case == "robot":
            s.add_robot("arm", mjcf=ARM, pos=(0.5, 0, 0.5))
        if case == "kinematic":
            s.add_box("pusher", size=(0.05, 0.2, 0.1), pos=(-0.2, 0, 0.45), kinematic=True)
        if case == "cloth":
            s.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, 0.7), resolution=4, mass=0.01)
        return s

    profile = case if case in ("precision", "debug") else "standard"
    auto = scene()
    if case == "fixed":
        auto.rigid_step = "fixed"
    pinned = scene()
    pinned.rigid_step = "fixed"
    a = built_of(alloy.build(auto, device="cpu", profile=profile))
    b = built_of(alloy.build(pinned, device="cpu", profile=profile))
    assert a.rigid_dt == b.rigid_dt == TODAY
    assert a.model_digest == b.model_digest
    assert [e.target_dt for e in a.entries] == [e.target_dt for e in b.entries]


def test_pinned_step_and_its_warning(caplog: pytest.LogCaptureFixture) -> None:
    s = one_box(z=0.6)
    s.rigid_step = 0.01
    with caplog.at_level(logging.WARNING, logger="alloy.scene"):
        built = built_of(alloy.build(s, device="cpu"))
    assert built.rigid_dt == 0.01
    assert any("rigid_step pinned at 10 ms breaks" in r.message for r in caplog.records)


def test_scene_json_keeps_the_setting() -> None:
    s = one_box()
    assert s.rigid_step == "auto" and "rigid_step" not in s.to_dict()
    s.rigid_step = 0.0025
    back = alloy.Scene.from_dict(s.to_dict())
    assert back.rigid_step == 0.0025
    s.rigid_step = "today"                  # the name before 0.1.0 still reads, as "fixed"
    assert s.rigid_step == "fixed"
    with pytest.raises(SceneError):
        s.rigid_step = "quick"              # "fast" is a mode (tests/scene/test_rigid_fast.py)
    with pytest.raises(SceneError):
        s.rigid_step = -1.0


@pytest.mark.parametrize(("make", "cap"), [
    (lambda: _heavy_drop(), "gap"),                        # a 17.5 cm drop: 1.85 m/s
    (lambda: _plate(), "tunnel"),                          # a 2 mm plate
    (lambda: _bead(), "spin"),                             # a 6 mm bead
])
def test_each_cap_binds(make, cap: str) -> None:
    step = built_of(alloy.build(make(), device="cpu")).report.rigid_step
    assert dict(step.caps)[cap] == step.dt < 0.01, step         # this cap binds (others may tie)


def _heavy_drop() -> alloy.Scene:
    """A 30 cm, 20 kg box 17.5 cm over a table: its material sets its stiffness, so cap 5 does not bind first."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True)
    s.add_box("crate", size=(0.3, 0.3, 0.3), pos=(0, 0, 0.725), mass=20.0)
    return s


def test_light_cube_keeps_todays_step_heavy_stack_does_not() -> None:
    """Cap 5: contacts never get softer than at the fixed step. A light cube's stiffness is the stability cap's
    (1/dt²): it keeps the fixed step (test_box_diagonal's 20 / 50 um); the stack's 10 kg cubes are material-bound
    and step coarser."""
    light = built_of(alloy.build(one_box(), device="cpu")).report.rigid_step
    assert light.dt == TODAY and dict(light.caps)["sag"] == TODAY and light.reason.startswith("sag cap"), light
    heavy = built_of(alloy.build(stack(), device="cpu")).report.rigid_step
    assert heavy.dt > TODAY, heavy


def _plate() -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("plate", size=(0.2, 0.2, 0.002), pos=(0, 0, 0.001), mass=0.05, rigid=True)
    return s


def _bead() -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_sphere("bead", 0.003, pos=(0, 0, 0.003), mass=0.001)
    return s


def test_slender_body_keeps_the_floors_step() -> None:
    """Cap 4 (WP3): a slender light body keeps the fixed step until the floor is swept above 1.5625 ms."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("marker", size=(0.12, 0.012, 0.012), pos=(0, 0, 0.006), mass=0.016, rigid=True)   # r_gyr 4.9 mm
    step = built_of(alloy.build(s, device="cpu")).report.rigid_step
    assert step.dt == TODAY and dict(step.caps)["floor"] == TODAY, step


def test_drops_and_loads() -> None:
    """Three cubes in a column on a plane, a fourth hanging 0.1 m over the plane beside them."""
    P = np.array([[[0, 0, 0.5], [0, 0, 1.5], [0, 0, 2.5], [3, 0, 0.6]]], float)
    half = np.full((4, 3), 0.5)
    mass = np.full(4, 10.0)
    drop, load = RB.drops_and_loads(P, np.zeros((4, 3)), half, mass, (np.zeros((0, 3)), np.zeros((0, 3)), 0.0))
    assert np.allclose(drop, [0, 0, 0, 0.1], atol=0, rtol=1e-12)
    assert np.allclose(load, [3, 2, 1, 1], atol=0, rtol=1e-12)


def test_randomized_height_counts_its_top() -> None:
    """A randomized z is taken at its highest for the body's own drop."""
    class Row:
        field, lo, hi = "pose.z", 0.5, 0.7

    class Placement:
        randomize = (Row(),)

    P, U = RB.start_positions(["a"], np.array([[0.0, 0.0, 0.5]]), None, {"a": Placement()})
    drop, _ = RB.drops_and_loads(P, U, np.full((1, 3), 0.5), np.ones(1), (np.zeros((0, 3)), np.zeros((0, 3)), 0.0))
    assert drop[0] == pytest.approx(0.2, abs=0, rel=1e-12)


def test_today_step_and_candidates() -> None:
    assert RS.today_step(0.01) == TODAY and RS.today_step(0.02) == TODAY
    assert RS.candidates(0.01, TODAY) == [0.01, 0.005, 0.0025, 0.00125]
