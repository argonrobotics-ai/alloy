# SPDX-License-Identifier: Apache-2.0
"""ERR-2: a build warns when a body's start drop breaks the contact-gap cap at the chosen rigid step (v dt > 5 mm), in
every mode and for scenes the rigid-step scan does not cover (alloy.scene.rigid_step_warn). The warning names the body,
the impact speed, the travel per step, what it can miss and a fix that, followed, removes it. Build only (CPU)."""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest

import alloy

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""
MARK = "start drop breaks the travel-per-step limit"


def drop_scene(heights, dt: float = 0.01, arm: bool = False, plate: float = 0.01) -> alloy.Scene:
    """A 2 cm steel ball above a static ``plate`` (default 1 cm; top z 0.305 for it), one world per height [m] (the
    audit's scene)."""
    s = alloy.Scene(worlds=len(heights), dt=dt)
    s.add_ground()
    s.add_box("plate", size=(0.5, 0.5, plate), pos=(0, 0, 0.3 + (plate - 0.01) / 2), static=True)
    pos = np.array([[0.0, 0.0, 0.3 + plate / 2 + 0.01 + h] for h in heights])
    s.add_sphere("ball", 0.01, pos=pos if len(heights) > 1 else pos[0], density=7800.0)
    if arm:
        s.add_robot("arm", mjcf=ARM, pos=(2.0, 0.0, 0.5))
    return s


def warnings_of(scene: alloy.Scene, caplog: pytest.LogCaptureFixture) -> list[str]:
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        alloy.build(scene, device="cpu")
    return [r.getMessage() for r in caplog.records if MARK in r.getMessage()]


def test_audit_scene_warns_with_speed_and_fix(caplog):
    (text,) = warnings_of(drop_scene([5.0]), caplog)
    assert "'ball'" in text and "9.9 m/s" in text and "5 m" in text
    assert "a static part 10 mm thick" in text and "against" not in text and "12.4 mm per rigid step (1.25 ms)" in text     # 9.9 m/s x 1.25 ms
    assert "thinner than about 24.8 mm" in text                                      # 2 v dt
    assert "scene.rigid_step = 0.0003125" in text and "0.815 m" in text


def test_low_drop_does_not_warn(caplog):
    assert warnings_of(drop_scene([0.5]), caplog) == []


def test_per_world_starts_warn_once_and_count_worlds(caplog):
    (text,) = warnings_of(drop_scene([0.5, 4.0, 8.0, 0.2, 12.0]), caplog)
    assert "in 3 of 5 worlds" in text and "15.3 m/s" in text                         # worst: 12 m


def test_robot_scene_warns_and_names_a_working_dt(caplog):
    (text,) = warnings_of(drop_scene([8.0], arm=True), caplog)
    assert "'ball'" in text and "12.5 m/s" in text and "scene.rigid_step is ignored" in text
    dt = float(re.search(r"alloy\.Scene\(dt=([0-9.e-]+)\)", text).group(1))
    assert warnings_of(drop_scene([8.0], dt=dt, arm=True), caplog) == []


@pytest.mark.parametrize("mode", ["auto", "fixed"])
def test_following_the_printed_pin_removes_the_warning(caplog, mode):
    s = drop_scene([5.0])
    s.rigid_step = mode
    (text,) = warnings_of(s, caplog)
    pin = float(re.search(r"scene\.rigid_step = ([0-9.e-]+)", text).group(1))
    s.rigid_step = pin
    assert warnings_of(s, caplog) == []


def test_estimate_carries_the_same_text(caplog):
    s = drop_scene([5.0])
    (text,) = warnings_of(s, caplog)
    assert alloy.estimate(s, 1, 1.0)["rigid_step"]["warnings"] == [text]


def test_cubes_over_bare_ground_do_not_warn(caplog):
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    for i in range(3):
        s.add_box(f"c{i}", size=(1.0,) * 3, pos=(2.0 * i, 0, 3.5), density=10.0)
    assert warnings_of(s, caplog) == []


def test_thick_support_does_not_warn_and_thin_one_does(caplog):
    assert warnings_of(drop_scene([5.0], plate=0.4), caplog) == []                  # 2 v dt 24.8 mm < 400 mm
    (text,) = warnings_of(drop_scene([5.0], plate=0.02), caplog)                    # 24.8 mm > 20 mm
    assert "a static part 20 mm thick" in text


@pytest.mark.parametrize("mode", ["auto", "fast"])
def test_many_bodies_warn_once_each(caplog, mode):
    """More than MAX_NAMED breakers: the named ones and one closing line, each text once, in the build's report and on
    the user's channel, and estimate does not repeat them."""
    from alloy.scene.rigid_step_warn import MAX_NAMED

    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("plate", size=(8.0, 8.0, 0.01), pos=(0, 0, 0.3), static=True)
    for i in range(10):
        s.add_sphere(f"ball{i}", 0.01, pos=(0.5 * i - 2.0, 0, 5.3), density=7800.0)
    s.rigid_step = mode
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        sim = alloy.build(s, device="cpu")
        alloy.estimate(s, 1, 1.0)
    shown = [r.getMessage() for r in caplog.records if MARK in r.getMessage()]
    kept = [w for w in sim.built.report.warnings if MARK in w]
    assert len(shown) == len(kept) == MAX_NAMED + 1
    assert len(set(shown)) == len(shown) and sorted(shown) == sorted(kept)
