# SPDX-License-Identifier: Apache-2.0
"""GAP-R1 API laws: the opt-in rigid step for robot scenes (``scene.rigid_step = <seconds>`` or an explicit "auto";
``alloy.scene.rigid_step_robots``), on CPU, one world. A scene that never sets ``rigid_step``, the precision and
debug profiles, and robots beside cloth keep the fixed 1.25 ms step. The physics behind the caps:
tests/scene/test_rigid_step_robots_laws.py; derivations and tables: [internal note].

Servo numbers of the arm below (I_eff = 0.030 kg m^2 about the hinge; b = kv + joint damping; the damping cap's
ratio is (1 + dt b / I) / (1 + 1.25 ms b / I) <= 1.25):
- gentle (kp 200, kv 1, damping 0.2): omega = 81 rad/s, omega dt = 0.41 and damping ratio 1.14 at 5 ms: every cap
  passes up to the period;
- soft (kp 200, kv 10, damping 2; AGENTS.md's arm): the servo cap passes, the damping ratio is 1.33 at 2.5 ms and
  2.0 at 5 ms;
- stiff (kp 8000, kv 5, damping 2): omega = 515 rad/s, zeta 0.23: omega dt = 0.64 at 1.25 ms, 1.29 at 2.5 ms
  (damping ratio 1.23 there, so the servo cap is the one that binds).
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

import alloy
from alloy.core.defaults import RIGID_GAP_M

TODAY = 1.25e-3
DT = 5.0e-3                         # the control period, and the step the laws pin
HINGE_Z, REACH, RADIUS = 0.6, 0.3, 0.02
GENTLE = {"kp": 200.0, "kv": 1.0, "damping": 0.2}
SOFT = {"kp": 200.0, "kv": 10.0}
STIFF = {"kp": 8000.0, "kv": 5.0}
LINK_KG, GRAVITY = 1.0, 9.81
CAP_NAMES = ("servo", "solref", "link_gap", "damping")
ROBOT_TAU, D0, DMAX = 0.02, 0.9, 0.95   # a robot geom's contact: MuJoCo's default solref and solimp


def arm(kp: float | None = 200.0, kv: float = 10.0, damping: float = 2.0) -> str:
    """AGENTS.md section 5's one-hinge arm; ``kp=None``: no actuator (a pendulum)."""
    servo = "" if kp is None else f'<actuator><position joint="shoulder" kp="{kp}" kv="{kv}"/></actuator>'
    return f"""<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="{damping}"/>
  <geom type="capsule" fromto="0 0 0 {REACH} 0 0" size="{RADIUS}" mass="{LINK_KG}"/>
  </body></worldbody>{servo}</mujoco>"""


def arm_scene(step: str | float | None = None, **servo: float) -> alloy.Scene:
    """The arm alone (nothing to touch), control period DT; ``step``: ``scene.rigid_step``, None to never set it."""
    s = alloy.Scene(dt=DT)
    s.add_robot("arm", mjcf=arm(**servo), pos=(0, 0, 0.5))
    if step is not None:
        s.rigid_step = step
    return s


def built_of(sim):
    return sim.engine.parts.built


def step_of(sim):
    return built_of(sim).report.rigid_step


def build(scene: alloy.Scene, profile: str = "standard"):
    return alloy.build(scene, device="cpu", profile=profile)


def ramp(sim, end: float = 0.4, seconds: float = 0.5) -> np.ndarray:
    return sim.run(seconds, record=["shoulder"], targets={"shoulder": (0.0, end)}).world(0)["shoulder.q"].ravel()


def warnings_of(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name.startswith("alloy.scene") and r.levelno >= logging.WARNING]


# --- D1: a pinned step is taken -----------------------------------------------------------------------------------

def test_pinned_step_is_taken_and_reported() -> None:
    """D1: one physics step per control period, and the build, the route report and the estimate all say 5 ms."""
    scene = arm_scene(DT, **GENTLE)
    sim = build(scene)
    assert built_of(sim).rigid_dt == DT and step_of(sim).dt == DT and step_of(sim).today == TODAY
    sim.step(10)
    assert sim.physics_steps == 10 and sim.check()["ok"]
    reasons = sim.routes()["arm"]["reasons"]
    assert any("rigid step 5 ms" in r for r in reasons), reasons
    est = alloy.estimate(scene, 1, 1.0, device="cpu")["rigid_step"]
    assert est["dt_s"] == DT and "5 ms" in est["reason"], est


def test_pinned_step_tracks_the_ramp_as_the_fixed_step_does() -> None:
    """D1: a 0.8 rad/s ramp at 5 ms against 1.25 ms. Law A: a coarser step lags by half a step, so the two differ
    by at most 0.5 rate (dt - fixed), plus half of omega dt of the 7.4 mrad gravity sag taken up at the start:
    1.5 + 1.5 mrad (1.50 mrad measured with the fixed step forced to 5 ms)."""
    rate, omega = 0.8, math.sqrt(SOFT["kp"] / (LINK_KG * REACH**2 / 3.0))
    coarse, fine = build(arm_scene(DT, **SOFT)), build(arm_scene("fixed", **SOFT))
    assert built_of(coarse).rigid_dt == DT and built_of(fine).rigid_dt == TODAY
    sag = LINK_KG * GRAVITY * 0.5 * REACH / SOFT["kp"]
    bound = 0.5 * rate * (DT - TODAY) + 0.5 * omega * DT * sag
    differ = float(np.max(np.abs(ramp(coarse) - ramp(fine))))
    assert 0.0 < differ <= bound, (differ, bound)
    assert coarse.check()["ok"]


# --- D2: never set ------------------------------------------------------------------------------------------------

def test_never_set_keeps_the_fixed_step_bit_for_bit() -> None:
    """D2: without ``scene.rigid_step`` a robot scene is the "fixed" scene: same step, model and trajectory."""
    default, fixed = build(arm_scene(**SOFT)), build(arm_scene("fixed", **SOFT))
    assert built_of(default).rigid_dt == built_of(fixed).rigid_dt == TODAY
    assert built_of(default).model_digest == built_of(fixed).model_digest
    a, b = ramp(default, seconds=0.2), ramp(fixed, seconds=0.2)
    assert np.array_equal(a, b) and default.physics_steps == fixed.physics_steps == 160


# --- D3: explicit "auto" ------------------------------------------------------------------------------------------

def test_auto_keeps_the_fixed_step_for_a_stiff_servo_and_names_it() -> None:
    """D3: omega dt = 1.29 at 2.5 ms breaks the servo cap, so the scene stays at 1.25 ms and says which servo."""
    scene = arm_scene("auto", **STIFF)
    step = step_of(build(scene))
    assert step.dt == TODAY
    assert "servo" in step.reason and "shoulder" in step.reason, step.reason
    est = alloy.estimate(scene, 1, 1.0, device="cpu")["rigid_step"]
    assert est["dt_s"] == TODAY and "servo" in est["reason"] and "shoulder" in est["reason"], est


def test_auto_keeps_the_fixed_step_for_a_damped_joint_and_names_it() -> None:
    """D3: AGENTS.md's arm (b = 12) has a damping ratio of 1.33 at 2.5 ms: it stays at 1.25 ms and says why."""
    scene = arm_scene("auto", **SOFT)
    step = step_of(build(scene))
    assert step.dt == TODAY
    assert "damping" in step.reason and "shoulder" in step.reason, step.reason
    est = alloy.estimate(scene, 1, 1.0, device="cpu")["rigid_step"]
    assert est["dt_s"] == TODAY and "damping" in est["reason"] and "shoulder" in est["reason"], est


def test_auto_takes_the_control_period_when_every_cap_passes() -> None:
    """D3: the gentle servo (omega dt 0.41, damping ratio 1.14 at 5 ms), nothing to touch: the control period."""
    sim = build(arm_scene("auto", **GENTLE))
    assert step_of(sim).dt == built_of(sim).rigid_dt == DT, step_of(sim)
    sim.step(4)
    assert sim.physics_steps == 4 and sim.check()["ok"]


# --- D4: a pin that breaks a cap ----------------------------------------------------------------------------------

def test_pin_past_the_servo_cap_is_taken_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """D4: 5 ms on the stiff servo (omega dt 2.6) is honoured; the build warns, naming the cap and the joint."""
    with caplog.at_level(logging.WARNING, logger="alloy.scene"):
        sim = build(arm_scene(DT, **STIFF))
    assert built_of(sim).rigid_dt == DT
    said = warnings_of(caplog)
    assert any("servo" in m and "shoulder" in m for m in said), said


def test_pin_past_the_damping_cap_is_taken_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """D4: 5 ms on AGENTS.md's arm (damping ratio 2.0) is honoured; the build warns, naming the cap and the joint."""
    with caplog.at_level(logging.WARNING, logger="alloy.scene"):
        sim = build(arm_scene(DT, **SOFT))
    assert built_of(sim).rigid_dt == DT
    said = warnings_of(caplog)
    assert any("damping" in m and "shoulder" in m for m in said), said
    assert not [m for m in said if "servo" in m], said


def test_pin_within_the_caps_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    """D4's other side: the gentle servo at 5 ms breaks no cap, so no warning names one."""
    with caplog.at_level(logging.WARNING, logger="alloy.scene"):
        sim = build(arm_scene(DT, **GENTLE))
    assert built_of(sim).rigid_dt == DT
    assert not [m for m in warnings_of(caplog) if any(cap in m for cap in CAP_NAMES)], warnings_of(caplog)


# --- D5 / D6: who keeps the fixed step ----------------------------------------------------------------------------

@pytest.mark.parametrize("profile", ["precision", "debug"])
def test_precision_and_debug_profiles_take_a_pin_but_not_auto(profile: str) -> None:
    """D5, as for free bodies: under these profiles an explicit "auto" is the "fixed" scene (same step and model),
    and a step in seconds is honoured."""
    auto = built_of(build(arm_scene("auto", **GENTLE), profile))
    fixed = built_of(build(arm_scene("fixed", **GENTLE), profile))
    assert auto.rigid_dt == fixed.rigid_dt == TODAY
    assert auto.model_digest == fixed.model_digest
    assert [e.target_dt for e in auto.entries] == [e.target_dt for e in fixed.entries]
    assert built_of(build(arm_scene(DT, **GENTLE), profile)).rigid_dt == DT


def test_robot_beside_cloth_keeps_the_fixed_step_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    """D6: robots beside a non-MuJoCo entry are out of scope: the pin is ignored and the build says so."""
    def scene(step: str | float) -> alloy.Scene:
        s = arm_scene(step, **SOFT)
        s.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, 0.3), resolution=4, mass=0.01)
        return s

    with caplog.at_level(logging.WARNING, logger="alloy.scene"):
        pinned = built_of(build(scene(DT)))
    fixed = built_of(build(scene("fixed")))
    assert pinned.rigid_dt == fixed.rigid_dt == TODAY and pinned.model_digest == fixed.model_digest
    assert any("rigid_step" in m and "ignored" in m for m in warnings_of(caplog)), warnings_of(caplog)


def test_robot_beside_an_sdf_mesh_keeps_the_fixed_step() -> None:
    """A scene on Newton's contact pipeline (an sdf mesh) keeps the fixed step under a pin too: a robot's contact
    rows there are not the file's, which the solref cap reads."""
    s = arm_scene(DT, **GENTLE)
    v = 0.05 * np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                  [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    s.add_mesh("part", vertices=v, faces=f, collision="sdf", pos=(0.6, 0, 0.05), mass=0.2)
    step = step_of(build(s))
    assert step.dt == TODAY and "Newton's contact pipeline" in step.reason, step.reason


def test_robot_beside_a_chain_rod_keeps_the_fixed_step() -> None:
    """A rod on the chain route is not a free rigid body: its grip numbers were fitted at the fixed step, so the
    scene stays there under a pin."""
    s = arm_scene(DT, **GENTLE)
    s.add_ground()
    s.add_rod("cable", length=0.3, radius=0.004, pos=(0.6, 0, 0.02), direction=(0, 1, 0))
    step = step_of(build(s))
    assert step.dt == TODAY and "keeps the fixed step" in step.reason and "cable" in step.reason, step.reason


# --- law C: a link meets a plate ----------------------------------------------------------------------------------

TOUCH = 0.105                       # rad below horizontal where the tip meets the plate: 0.96 m/s, v dt = 4.8 mm
PLATE_TOP = HINGE_Z - REACH * math.sin(TOUCH) - RADIUS
PLATE_THICK = 0.01


def pendulum_scene(step: str | float) -> alloy.Scene:
    """The arm without a servo or damping, falling from horizontal onto a static 10 mm plate under its tip."""
    s = alloy.Scene(dt=DT)
    s.add_robot("arm", mjcf=arm(kp=None, damping=0.0), pos=(0, 0, 0.5))
    s.add_box("plate", size=(0.2, 0.2, PLATE_THICK), pos=(REACH * math.cos(TOUCH), 0, PLATE_TOP - 0.5 * PLATE_THICK),
              static=True)
    s.rigid_step = step
    return s


def tip_depth(sim, seconds: float = 0.4) -> np.ndarray:
    """How far the tip's lowest point is under the plate's top [m], every control period."""
    q = sim.run(seconds, record=["shoulder"]).world(0)["shoulder.q"].ravel()
    return PLATE_TOP - (HINGE_Z - REACH * np.sin(q) - RADIUS)


def test_link_within_the_gap_cap_is_stopped_by_the_plate() -> None:
    """Law C: the tip arrives at v = sqrt(3 g L sin(touch)) = 0.96 m/s, so v dt = 4.8 mm <= the 5 mm contact gap and
    the contact exists before the surfaces meet. The plate then stops it as a robot contact does at any step: a
    critically damped row of time constant tau = 0.02 s lets it in by at most v tau dmax / (e d0) = 7.5 mm (under
    the plate's 10 mm), and it comes to rest as deep as at the fixed step (0.06 mm, within 10 %).
    Measured with the fixed step forced to 5 ms: peak 3.45 mm, rest 0.061 mm; at 1.25 ms: 6.36 mm, 0.060 mm."""
    speed = math.sqrt(3.0 * GRAVITY * REACH * math.sin(TOUCH))
    assert speed * DT <= RIGID_GAP_M
    coarse, fine = build(pendulum_scene(DT)), build(pendulum_scene("fixed"))
    assert built_of(coarse).rigid_dt == DT and built_of(fine).rigid_dt == TODAY
    bound = speed * ROBOT_TAU * DMAX / (math.e * D0)
    deep, ref = tip_depth(coarse), tip_depth(fine)
    assert coarse.physics_steps == 80 and coarse.check()["ok"]
    assert 0.0 < ref.max() <= bound < PLATE_THICK, (ref.max(), bound)
    assert 0.0 < deep.max() <= bound, (deep.max(), bound)
    assert deep[-1] == pytest.approx(ref[-1], rel=0.1, abs=0), (deep[-1], ref[-1])
