# SPDX-License-Identifier: Apache-2.0
"""SPEED-NOW S1, ruling 2026-11-09 09:18Z condition (3): ``alloy.estimate(...)["rigid_step"]`` names the rigid fast
mode's step and what it gives up for a scene of free bodies (its "fast" row), and has no such row otherwise (CPU)."""

from __future__ import annotations

import alloy
from alloy.scene.rigid_step import FAST_COST

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def stack() -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground(friction=1.0)
    for k in range(3):
        s.add_box(f"c{k}", size=(1.0,) * 3, pos=(0.0, 0.0, 0.5 + k + 1e-4 * k), density=10.0, friction=1.0)
    return s


def test_a_free_body_scene_gets_the_fast_row() -> None:
    rs = alloy.estimate(stack(), 4, 1.0, device="cpu")["rigid_step"]
    # "auto"'s own step for 3 bodies depends on F6 R2 (off for the release: rk4 keeps them at 2.5 ms; on: 5 ms)
    assert rs["mode"] == "auto" and rs["dt_s"] in (2.5e-3, 5e-3) and rs["eligible"]
    fast = rs["fast"]
    assert fast["dt_s"] == 1e-2 and fast["reason"].startswith("fast mode") and fast["gives_up"] == FAST_COST
    assert fast["physics_steps_per_world"] == 100


def test_a_scene_already_in_fast_mode_repeats_its_own_row() -> None:
    s = stack()
    s.rigid_step = "fast"
    out = alloy.estimate(s, 4, 1.0, device="cpu")
    assert out["rigid_step"]["mode"] == "fast" and out["rigid_step"]["fast"]["wall_time_s"] == out["wall_time_s"]


def test_a_fixed_scene_still_gets_the_fast_row() -> None:
    """Eligibility is the scene's (rigid_step.why_not), not the mode it was set to."""
    s = stack()
    s.rigid_step = "fixed"
    rs = alloy.estimate(s, 4, 1.0, device="cpu")["rigid_step"]
    assert rs["mode"] == "fixed" and rs["dt_s"] == 1.25e-3 and rs["eligible"] and rs["fast"]["dt_s"] == 1e-2


def test_a_robot_scene_has_no_fast_row() -> None:
    s = stack()
    s.add_robot("arm", mjcf=ARM, pos=(3.0, 0, 0.5))
    rs = alloy.estimate(s, 4, 1.0, device="cpu")["rigid_step"]
    assert not rs["eligible"] and "fast" not in rs
