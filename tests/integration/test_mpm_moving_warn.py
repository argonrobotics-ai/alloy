# SPDX-License-Identifier: Apache-2.0
"""The build warns for sand or paste with a body that can move above 64 GPU worlds, the range the integer cell
lookup's law did not reach (``alloy.integration.mpm_moving``; known gap
docs/known_gaps/mpm-moving-container-many-worlds.md, fixed for 64 worlds and below; physics rulings 2026-11-11). CPU:
the GPU check is forced on where a law needs it, so the scene builds here.
- the known gap's scene (sand in a kinematic open box) at 65 worlds on a GPU: one warning, naming the page, the
  sparse grid's capacity and sim.check();
- a robot with a servo beside sand, and a free box resting on sand, at 65 worlds: warns;
- silent at 64 worlds, on the CPU, and with only static shapes (a static open box; a slab on the ground alone).

    PYTHONPATH=. python -m pytest tests/integration/test_mpm_moving_warn.py -q
"""

from __future__ import annotations

import warnings

import pytest

import alloy
from alloy.integration import mpm_moving as MM

pytestmark = pytest.mark.integration

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.2"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="20" kv="1"/></actuator></mujoco>"""


def bin_scene(worlds: int, kinematic: bool = True) -> alloy.Scene:
    s = alloy.Scene(dt=0.01, worlds=worlds)
    s.add_ground()
    move = {"kinematic": True} if kinematic else {"static": True}
    s.add_open_box("bin", inner=(0.03, 0.03, 0.03), wall=0.006, pos=(0.3, 0, 0.1), **move)
    s.add_box("sand", size=(0.02, 0.02, 0.01), pos=(0.3, 0, 0.112), material="sand_silica_dry")
    return s


def slab_scene(worlds: int, box: bool) -> alloy.Scene:
    s = alloy.Scene(dt=0.01, worlds=worlds)
    s.add_ground()
    s.add_box("sand", size=(0.04, 0.04, 0.02), pos=(0, 0, 0.01), material="sand_silica_dry")
    if box:
        s.add_box("box", size=(0.03, 0.03, 0.03), pos=(0, 0, 0.036), mass=0.3)
    return s


def robot_scene(worlds: int) -> alloy.Scene:
    s = alloy.Scene(dt=0.01, worlds=worlds)
    s.add_ground()
    s.add_box("sand", size=(0.03, 0.03, 0.01), pos=(0.05, 0, 0.005), material="sand_silica_dry")
    s.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.0))
    return s


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(MM, "_is_cuda", lambda built: True)


def _build(scene: alloy.Scene) -> tuple[alloy.Sim, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sim = alloy.build(scene, device="cpu", seed=0)
    return sim, [str(w.message) for w in caught if "not validated" in str(w.message)]


def test_the_known_gaps_scene_warns_above_64_gpu_worlds(gpu) -> None:
    _, said = _build(bin_scene(65))
    assert len(said) == 1, said
    assert MM.PAGE in said[0] and "status 8" in said[0] and "bin" in said[0]
    assert "lowest grain" in said[0] and "sim.check() does not flag" in said[0] and "64 worlds or fewer" in said[0]


def test_a_servo_robot_beside_sand_warns(gpu) -> None:
    _, said = _build(robot_scene(65))
    assert len(said) == 1 and "link1" in said[0], said


def test_a_free_box_resting_on_sand_warns(gpu) -> None:
    """AUDIT-STACK's static check: a free box resting on a slab, nothing driven (45 of 64 L4 worlds took grains
    before the fix)."""
    _, said = _build(slab_scene(65, box=True))
    assert len(said) == 1 and "box" in said[0], said


@pytest.mark.parametrize("case", ["64 worlds", "cpu", "static box", "slab alone"])
def test_silent_at_64_worlds_on_the_cpu_and_with_only_static_shapes(case, monkeypatch) -> None:
    if case != "cpu":
        monkeypatch.setattr(MM, "_is_cuda", lambda built: True)
    scene = {"64 worlds": bin_scene(64), "cpu": bin_scene(65), "static box": bin_scene(65, kinematic=False),
             "slab alone": slab_scene(65, box=False)}[case]
    _, said = _build(scene)
    assert said == []
