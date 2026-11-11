# SPDX-License-Identifier: Apache-2.0
"""Robots under the rigid step (alloy.scene.rigid_step_robots): opt-in only, the three robot caps, and the pinned
step's warning (CPU). A scene that never set ``rigid_step`` steps as before, bit for bit."""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import numpy as np
import pytest

import alloy
from alloy.route.types import Route
from alloy.scene import assembler
from alloy.scene import rigid_step as RS
from alloy.scene import rigid_step_robots as RR
from alloy.scene import rigid_step_robots_facts as RF

TODAY = 1.25e-3
REL = 1e-5                      # the builder keeps file values in float32

# A 0.3 m, 1 kg box link on a hinge about y (armature 0.01), a <position> servo of kp 200.
ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2" armature="0.01"/>
  <geom name="bar" type="box" pos="0.15 0 0" size="0.15 0.02 0.02" mass="1" {solref}/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""
# A 2 kg block on a slide along x, a Menagerie-style <general> servo of kp 300; beside it a loose 5 kg crate.
SLIDER = """<mujoco><worldbody><body name="block" pos="0 0 0.1">
  <joint name="rail" type="slide" axis="1 0 0" range="-0.2 0.2"/>
  <geom name="pad" type="box" size="0.05 0.05 0.05" mass="2"/>
  </body>
  <body name="prop" pos="0 0.5 0.1"><freejoint/><geom name="crate" type="box" size="0.05 0.05 0.05" mass="5"/></body>
  </worldbody>
  <actuator><general joint="rail" gainprm="300" biasprm="0 -300 -30" biastype="affine"/></actuator></mujoco>"""


def arm_scene(dt: float = 0.01, solref: str = "") -> alloy.Scene:
    s = alloy.Scene(dt=dt)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM.format(solref=solref), pos=(0, 0, 0.5))
    return s


def built_of(sim):
    return sim.engine.parts.built


@pytest.fixture(scope="module")
def template():
    """(builder, objects, scene card, integrator) of a hinge arm beside a slider, as ``choose_rigid_step`` gets
    them from the assembler."""
    seen: dict = {}
    real = assembler.choose_rigid_step

    def spy(b, objs, scene, integrator, request, report=None):
        seen.update(b=b, objs=objs, scene=scene, integrator=integrator)
        return real(b, objs, scene, integrator, request, report)

    s = arm_scene(solref='solref="0.004 1"')
    s.add_robot("slider", mjcf=SLIDER, pos=(1.0, 0, 0.5), contact_gap=0.002)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(assembler, "choose_rigid_step", spy)
        alloy.build(s, device="cpu")
    return SimpleNamespace(**seen)


# --- the caps, by hand ------------------------------------------------------------------------------------------

def test_servo_cap() -> None:
    f = RR.RobotFacts(TODAY, servos=(RR.Servo("arm/a", 100.0, 1.0), RR.Servo("arm/b", 25.0, 1.0)))   # omega 10, 5
    assert RR.cap_servo(f, 0.1) is None                             # omega dt = 1: met exactly
    slow = RR.cap_servo(f, 0.15)
    assert slow.startswith("arm/a: servo omega dt 1.5 > 1 (") and "stable below 2)" in slow, slow
    both = RR.cap_servo(f, 0.25)
    assert both.startswith("arm/a (and 1 more): servo omega dt 2.5 > 1 (") and "unstable" in both, both


def test_servo_cap_lets_an_overdamped_servo_follow_its_slow_pole() -> None:
    """kp 100 on inertia 1 with kv 60: zeta 3, so omega dt may reach 3 (dt = kv / (2 kp) = 0.3 s)."""
    servo = RR.Servo("arm/a", 100.0, 1.0, damping=60.0)
    assert servo.zeta == 3.0 and RR.stability_limit(1.0) == pytest.approx(2.0 + 2.0 * math.sqrt(2.0), rel=1e-12)
    f = RR.RobotFacts(TODAY, servos=(servo,))
    assert RR.cap_servo(f, 0.3) is None
    why = RR.cap_servo(f, 0.4)
    assert why.startswith("arm/a: servo omega dt 4 > 3 (") and "zeta 3" in why and "unstable" not in why, why


def test_damping_cap() -> None:
    """b / I = 400 /s: 1 + dt b / I is 1.5 at 1.25 ms, 1.75 at 1.875 ms (x1.17), 2 at 2.5 ms (x1.33), 3 at 5 ms."""
    f = RR.RobotFacts(TODAY, damped=(RR.Damped("arm/shoulder", 12.0, 0.03), RR.Damped("arm/wrist", 0.03, 0.03)))
    assert RR.cap_damping(f, TODAY) is None and RR.cap_damping(f, 1.875e-3) is None
    assert RR.cap_damping(f, 2.5e-3).endswith("contacts on arm/shoulder's links sink x1.33 deeper than at 1.25 ms")
    why = RR.cap_damping(f, 5e-3)
    assert why.startswith("arm/shoulder: joint damping dilutes contacts (dt b / I 2)") and "sink x2 deeper" in why, why
    light = RR.RobotFacts(TODAY, damped=(RR.Damped("arm/wrist", 0.03, 0.03),))          # b / I = 1 /s
    assert RR.cap_damping(light, 0.01) is None


def test_solref_cap() -> None:
    tau = float(np.float32(0.0025))                                 # as the builder keeps it: 2.49999994 ms
    f = RR.RobotFacts(TODAY, rows=(RR.Row("arm/pad", "contact", tau), RR.Row("arm/elbow", "joint limit", 0.02)))
    assert RR.cap_solref(f, 1.25e-3) is None                        # tau = 2 dt: met exactly
    why = RR.cap_solref(f, 2.5e-3)
    assert why.startswith("arm/pad: contact time constant 2.5 ms < 2 dt, so it acts as 5 ms"), why
    assert why.endswith("contacts 0.25x as stiff as at 1.25 ms"), why
    assert RR.cap_solref(f, 5e-3).endswith("contacts 0.0625x as stiff as at 1.25 ms")
    assert RR.cap_solref(f, 0.02).startswith("arm/pad (and 1 more)")
    assert RR.softening(0.02, 0.01, TODAY) == 1.0 and RR.softening(0.004, 0.004, TODAY) == 0.25


def test_link_gap_cap() -> None:
    f = RR.RobotFacts(TODAY, links=(RR.Link("arm/elbow", 1.0, 0.005, reach=0.5), RR.Link("arm/rail", 0.5, 0.005)))
    assert RR.cap_link_gap(f, 0.005) is None                        # 1 m/s x 5 ms = the 5 mm gap
    why = RR.cap_link_gap(f, 0.0075)
    assert why.startswith("arm/elbow: v dt 7.5 mm > contact gap 5 mm") and "start 2.5 mm deep" in why, why
    assert RR.cap_link_gap(f, 0.02).startswith("arm/elbow (and 1 more)")


def test_loose_body_cap() -> None:
    """The light-body rule on a default 20 ms row (zeta 1, dmax 0.95): tau = sqrt(m / (K (1 - dmax))), K 1e6 N/m,
    never under 2 dt. A 5 kg crate's 10 ms holds up to 5 ms steps; a 0.1 kg box's 1.4 ms is 2.5 ms at 1.25 ms."""
    crate, box = RR.Loose("arm/crate", 0.02, 1.0, 0.95, 5.0), RR.Loose("arm/box", 0.02, 1.0, 0.95, 0.1)
    assert crate.acting(TODAY) == pytest.approx(0.01, rel=1e-9) and box.acting(TODAY) == pytest.approx(2.5e-3, rel=1e-9)
    heavy = RR.RobotFacts(TODAY, loose=(crate,))
    assert RR.cap_loose_body(heavy, 0.005) is None                  # 2 dt = its 10 ms: met exactly
    why = RR.cap_loose_body(heavy, 0.01)
    assert why == ("arm/crate: a loose body's contact acts at 20 ms against 10 ms at the fixed step: contacts 0.25x "
                   "as stiff as at 1.25 ms"), why
    both = RR.cap_loose_body(RR.RobotFacts(TODAY, loose=(crate, box)), 0.01)
    assert both.startswith("arm/box (and 1 more)") and both.endswith("contacts 0.0156x as stiff as at 1.25 ms"), both
    assert RR.cap_loose_body(RR.RobotFacts(TODAY, loose=(box,)), 2.5e-3).endswith("0.25x as stiff as at 1.25 ms")


def test_no_robot_facts_no_refusal() -> None:
    empty = RR.RobotFacts(TODAY)
    assert not empty and all(cap(empty, 1.0) is None for cap in RR.ROBOT_CAPS.values())


# --- the facts, from the builder --------------------------------------------------------------------------------

def test_facts_of_a_hinge_and_a_slide(template) -> None:
    f = RF.collect(template.b, template.objs, template.scene, TODAY)
    servo = {s.name: s for s in f.servos}
    # the bar about the hinge: m (a^2 + c^2) / 12 about its centre + m d^2, + armature
    inertia = (0.3**2 + 0.04**2) / 12.0 + 0.15**2 + 0.01
    assert servo["arm/shoulder"].kp == 200.0 and servo["arm/shoulder"].inertia == pytest.approx(inertia, rel=REL)
    assert servo["arm/shoulder"].omega == pytest.approx(math.sqrt(200.0 / inertia), rel=REL)
    assert servo["arm/shoulder"].damping == 12.0                    # kv 10 + the joint's damping 2
    assert servo["slider/rail"].kp == 300.0 and servo["slider/rail"].inertia == pytest.approx(2.0, rel=REL)
    assert servo["slider/rail"].damping == 30.0                     # biasprm's kv
    assert {(j.name, j.damping) for j in f.damped} == {("arm/shoulder", 12.0), ("slider/rail", 30.0)}
    link = {k.name: k for k in f.links}
    reach = math.hypot(0.3, 0.02)                                   # the bar's far corners, from the y axis
    assert link["arm/shoulder"].reach == pytest.approx(reach, rel=REL)
    assert link["arm/shoulder"].speed == pytest.approx(RR.LINK_SPEED_RAD_S * reach, rel=REL)
    assert link["arm/shoulder"].gap == 0.005 and link["slider/rail"].gap == 0.002      # add_robot(contact_gap=)
    assert link["slider/rail"].speed == RR.LINK_SPEED_M_S and link["slider/rail"].reach == 0.0
    rows = {(r.name, r.kind): r.tau for r in f.rows}
    assert rows[("arm/bar", "contact")] == pytest.approx(0.004, rel=REL)               # the file's solref
    assert rows[("slider/pad", "contact")] == pytest.approx(0.02, rel=REL)             # MuJoCo's default
    assert rows[("arm/shoulder", "joint limit")] == rows[("slider/rail", "joint limit")] == 0.02
    assert ("slider/crate", "contact") not in rows                  # the loose crate: the light-body rule's row
    (crate,) = f.loose
    assert (crate.name, crate.zeta) == ("slider/crate", 1.0) and crate.mass == pytest.approx(5.0, rel=REL)
    assert crate.tau == pytest.approx(0.02, rel=REL) and crate.dmax == pytest.approx(0.95, rel=REL)


def test_auto_takes_the_largest_step_the_robot_caps_allow(template) -> None:
    """Explicit "auto": the arm's damping (b / I = 299 /s: x1.27 at 2.5 ms) and its 4 ms contact time constant
    (2 ms) hold the grid's 1.25 ms; the slider's 2 mm gap at 0.5 m/s allows 4 ms (2.5 ms on the grid) and the arm's
    servo 14 ms."""
    ask = RS.RigidStepRequest("auto", explicit=True)
    step = RS.choose_rigid_step(template.b, template.objs, template.scene, template.integrator, ask)
    caps = dict(step.caps)
    assert step.eligible and step.dt == TODAY and caps["solref"] == TODAY, step
    assert caps["link_gap"] == 2.5e-3 and caps["servo"] == 0.01 and caps["loose_body"] == 0.005, step
    assert caps["damping"] == TODAY, step
    assert step.reason.startswith("damping cap, arm/shoulder: joint damping dilutes contacts"), step.reason
    assert set(RS.CAPS) | set(RR.ROBOT_CAPS) == set(caps)


# --- eligibility ------------------------------------------------------------------------------------------------

def _thing(name: str, *, motion: str = "dynamic", entry: str = "mujoco", route: Route = Route.RIGID,
           precision: tuple = ()) -> SimpleNamespace:
    card = SimpleNamespace(motion=motion, precision_partners=precision, precision_features=())
    decision = SimpleNamespace(entry=entry, route=route)
    return SimpleNamespace(name=name, obj=SimpleNamespace(card=card), decision=decision)


def _robot(entry: str = "mujoco") -> SimpleNamespace:
    return SimpleNamespace(name="arm", obj=None, decision=SimpleNamespace(entry=entry, route=Route.RIGID))


def _scene(**kw) -> SimpleNamespace:
    return SimpleNamespace(robots=("arm",), attachments=(), **kw)


PIN = RS.RigidStepRequest(0.005)


@pytest.mark.parametrize(("request_", "why"), [
    (None, RR.KEEPS),                                               # the engine layers' default
    (RS.RigidStepRequest("auto"), RR.KEEPS),                        # never set
    (RS.RigidStepRequest("fixed"), RR.KEEPS),
    (RS.RigidStepRequest("fast", from_profile=True), RR.KEEPS),     # profile="fast" on a never-set scene
    (RS.RigidStepRequest("fast"), RR.NO_FAST),                      # the user's own "fast"
    (RS.RigidStepRequest("auto", explicit=True), None),
    (PIN, None),
])
def test_robots_step_coarser_only_when_asked(request_, why) -> None:
    assert RS.why_not(None, [_robot()], _scene(), request_) == why
    assert RR.refusal(request_) == why


def test_a_robot_beside_a_pour_keeps_the_fixed_step() -> None:
    """A pour steps on the liquid's exchange clock, which never reads the rigid step."""
    pour = SimpleNamespace(robots=("arm",), attachments=(), emitters=("pour",))
    assert RS.why_not(None, [_robot()], pour, PIN) == RR.POUR
    assert RS.why_not(None, [_robot()], pour, RS.RigidStepRequest("auto")) == RR.KEEPS


@pytest.mark.parametrize(("objs", "scene", "why"), [
    ([_robot()], SimpleNamespace(robots=("arm",), attachments=("weld",)), "attachments keep the fixed step"),
    ([_robot("featherstone")], _scene(), "arm: a robot on the featherstone entry keeps the fixed step"),
    ([_robot(), _thing("pusher", motion="kinematic")], _scene(), "pusher: kinematic objects keep the fixed step"),
    ([_robot(), _thing("sheet", entry="vbd", route=Route.CLOTH_VBD)], _scene(),
     "sheet: route cloth_vbd on vbd keeps the fixed step"),
    ([_robot(), _thing("peg", precision=("hole",))], _scene(), "peg: precision pairs keep the fixed step"),
])
def test_a_pinned_robot_scene_is_still_refused(objs, scene, why) -> None:
    assert RS.why_not(None, objs, scene, PIN) == why


def test_never_set_robot_scene_steps_as_the_fixed_one_bitwise() -> None:
    fixed = arm_scene()
    fixed.rigid_step = "fixed"
    sims = [alloy.build(s, device="cpu") for s in (arm_scene(), fixed)]
    a, b = (built_of(sim) for sim in sims)
    assert a.rigid_dt == b.rigid_dt == TODAY and a.model_digest == b.model_digest
    assert [e.target_dt for e in a.entries] == [e.target_dt for e in b.entries]
    assert not a.report.rigid_step.eligible and a.report.rigid_step.reason == RR.KEEPS
    for sim in sims:
        sim.set_targets({"shoulder": 0.5})
        sim.step(10)
    qa, qb = (sim.state(world=0)["arm/shoulder.q"] for sim in sims)
    assert sims[0].physics_steps == sims[1].physics_steps == 80
    assert np.asarray(qa).tobytes() == np.asarray(qb).tobytes() and float(np.ravel(qa)[0]) != 0.0


# --- a pinned step ----------------------------------------------------------------------------------------------

def test_pinned_step_reaches_the_robot_and_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    s = arm_scene(dt=0.005, solref='solref="0.0025 1"')
    s.rigid_step = 0.005
    with caplog.at_level(logging.WARNING, logger="alloy.scene"):
        sim = alloy.build(s, device="cpu")
    said = [r.getMessage() for r in caplog.records if "rigid_step pinned" in r.getMessage()]
    assert said == ["rigid_step pinned at 5 ms breaks 2 cap(s): damping: arm/shoulder: joint damping dilutes contacts "
                    "(dt b / I 1.5): contacts on arm/shoulder's links sink x1.82 deeper than at 1.25 ms; solref: "
                    "arm/bar: contact time constant 2.5 ms < 2 dt, so it acts as 10 ms: contacts 0.0625x as stiff "
                    "as at 1.25 ms"], said
    built = built_of(sim)
    assert built.rigid_dt == 0.005 and built.report.rigid_step.eligible
    assert next(e for e in built.entries if e.name == "mujoco").target_dt == 0.005
    sim.step(4)
    assert sim.physics_steps == 4 and sim.check()["ok"]             # one physics step per control period
    assert any("rigid step 5 ms" in r and "pinned at 5 ms" in r for r in sim.routes()["arm"]["reasons"])
