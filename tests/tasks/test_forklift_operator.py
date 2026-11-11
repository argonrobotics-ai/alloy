# SPDX-License-Identifier: Apache-2.0
"""The scripted forklift operator on a kinematic plant: each task's phase list runs to done and ends on its pose."""

from __future__ import annotations

import math

import pytest

from alloy.tasks.forklift_control import REST_SPEED, Controls, TruckSpec, kmh, speed_commands
from alloy.tasks.forklift_operator import Operator, Plan
from alloy.tasks.forklift_path import Path, segment
from alloy.tasks.forklift_plant import KinematicPlant, ramp_along_x

DT = 0.02
SPEC = TruckSpec()
PI = math.pi


def wrap(a: float) -> float:
    return (a + PI) % (2 * PI) - PI


class Run:
    """Closed loop of an operator and a plant with the invariants checked every step."""

    def __init__(self, plan: Plan, plant: KinematicPlant) -> None:
        self.plan, self.plant, self.op = plan, plant, Operator(plan, SPEC)
        self.obs, self.ctrl = plant.observe(), Controls()
        self.over_cap, self.park_moving, self.ramp_over, self.ramp_steps, self.cross = 0, 0, 0, 0, {}
        self.paths = {k: Path.from_segment(s) for k, s in plan.segments.items()}

    def step(self) -> None:
        cap = self.op.speed_cap
        name = self.op.phase
        self.ctrl = self.op.step(self.obs)
        self.obs = self.plant.step(self.ctrl, DT)
        speed = abs(self.obs.speed)
        self.over_cap += cap is not None and speed > cap + 1e-3
        self.ramp_steps += self.obs.on_ramp
        self.ramp_over += self.obs.on_ramp and speed > kmh(6.0) + 1e-3
        self.park_moving += self.ctrl.park and abs(self.plant.speed) >= REST_SPEED
        if name in self.paths and cap is not None:
            self.cross[name] = max(self.cross.get(name, 0.0), self.paths[name].cross_track(self.obs.x, self.obs.y))

    def until(self, limit: float, stop=lambda r: False) -> "Run":
        while self.plant.t < limit and not self.op.done and not stop(self):
            self.step()
        return self

    def finish_obs(self, phase: str):
        return next(o for n, o in self.op.history if n == phase)


def make_plant(x: float, y: float, yaw: float, **kw) -> KinematicPlant:
    return KinematicPlant(SPEC, x=x, y=y, yaw=yaw, **kw)


def assert_pose(obs, wp, pos_tol: float = 0.02, yaw_tol_deg: float = 1.0) -> None:
    assert math.hypot(obs.x - wp[0], obs.y - wp[1]) <= pos_tol, (obs.x, obs.y, wp)
    assert abs(math.degrees(wrap(obs.yaw - wp[2]))) <= yaw_tol_deg, (math.degrees(obs.yaw), wp)


def assert_stops_on_segments(run: Run, skip=()) -> None:
    for name, obs in run.op.history:
        if name in run.plan.segments and name not in skip:
            assert_pose(obs, run.plan.segments[name].end)


def assert_clean(run: Run) -> None:
    assert run.op.done, run.op.phase
    assert (run.over_cap, run.ramp_over, run.park_moving) == (0, 0, 0)


def start_park_plan() -> Plan:
    to_bay = segment([(-8.0, 2.0, 0.0, 1), (-2.0, 2.0, 0.0, 1), (2.0, -1.0, -PI / 2, 1), (2.0, -4.0, -PI / 2, 1)], 8.0)
    return Plan("start_park", {"to_bay": to_bay})


def test_start_and_park() -> None:
    run = Run(start_park_plan(), make_plant(-8.0, 2.0, 0.0, fork_height=0.4, tilt=-0.05)).until(120)
    assert_clean(run)
    assert run.op.history[0][0] == "key_on" and run.op.history[-1][0] == "key_off"
    assert_stops_on_segments(run)
    obs = run.plant.observe()                   # tips on the floor, blades sloping down to them
    assert run.plant.fork_height <= 0.005 and obs.tip_height <= 0.003 and obs.blade_pitch <= 0.0
    assert run.plant.tilt < SPEC.tilt_limit     # stopped there, short of the tilt stop
    assert (run.ctrl.key_on, run.ctrl.engine_on, run.ctrl.park) == (False, False, True)
    assert run.cross["to_bay"] <= 0.05          # the 90 degree turn tracks within 5 cm


def test_reverse_segment_tracks_and_stops() -> None:
    back = segment([(0.0, 0.0, 0.0, -1), (-4.0, 1.0, 0.0, -1), (-8.0, 1.0, 0.0, -1)], 5.0)
    run = Run(Plan("start_park", {"to_bay": back}), make_plant(0.0, 0.0, 0.0))
    run.until(60, lambda r: r.op.phase == "neutral")
    assert run.cross["to_bay"] <= 0.05
    assert_pose(run.finish_obs("to_bay"), back.end)


def ramp_plan() -> Plan:
    segs = {
        "out": segment([(14.0, 0.0, PI, 1), (1.0, 0.0, PI, 1)], 8.0),
        "ramp": segment([(1.0, 0.0, PI, 1), (-8.0, 0.0, PI, 1)], 6.0),
        "to_drop": segment([(-8.0, 0.0, PI, 1), (-11.0, 0.0, PI, 1), (-15.0, -3.0, -PI / 2, 1),
                            (-15.0, -6.0, -PI / 2, 1)], 8.0),
        "back_out": segment([(-15.0, -6.0, -PI / 2, -1), (-15.0, -3.0, -PI / 2, -1)], 5.0)}
    return Plan("ramp_down", segs, {"carry": 0.15, "set": 0.0, "rest": 0.0})


def test_ramp_down_loaded() -> None:
    plant = make_plant(14.0, 0.0, PI, load_on_forks=True, terrain=ramp_along_x(-8.0, 1.0))
    run = Run(ramp_plan(), plant).until(150)
    assert_clean(run)
    assert_stops_on_segments(run, skip=("out", "ramp"))
    assert run.plant.fork_height <= 0.005 and abs(run.plant.tilt) <= math.radians(0.2)
    assert run.ramp_steps > 300                              # it really drove the ramp, under 6 km/h throughout
    assert abs(run.finish_obs("ramp").x - (-8.0)) < 0.2     # handed over at the foot of the ramp


@pytest.mark.parametrize("beam", [1.60, 3.20])
def test_put_away_rack(beam: float) -> None:
    segs = {"to_face": segment([(-8.0, 0.5, 0.0, 1), (-4.0, 0.0, 0.0, 1), (0.0, 0.0, 0.0, 1)], 8.0),
            "creep": segment([(0.0, 0.0, 0.0, 1), (1.2, 0.0, 0.0, 1)], 2.0),
            "back_out": segment([(1.2, 0.0, 0.0, -1), (-0.2, 0.0, 0.0, -1)], 4.0)}
    plan = Plan("put_away", segs, {"beam": beam})
    run = Run(plan, make_plant(-8.0, 0.5, 0.0, load_on_forks=True, fork_height=0.2)).until(200)
    assert_clean(run)
    assert_stops_on_segments(run)
    up = run.finish_obs("raise_above_beams")
    assert abs(up.fork_height - (beam + 0.10)) <= 0.005 and abs(up.tilt) <= math.radians(0.2)
    assert abs(run.finish_obs("creep").fork_height - up.fork_height) <= 0.005
    assert abs(run.finish_obs("lower_onto_beams").fork_height - (beam - 0.05)) <= 0.005
    assert run.finish_obs("creep").x - run.finish_obs("back_out").x >= 1.0
    assert abs(run.plant.fork_height - 0.2) <= 0.005 and run.plant.fork_height <= 0.3


def trailer_segments(n: str = "", pick_y: float = 0.0, pick_x: float = -4.0, start=(-10.0, 0.4)) -> dict:
    return {
        "pick" + n: segment([(start[0], start[1], 0.0, 1), (pick_x - 3.0, pick_y, 0.0, 1),
                             (pick_x, pick_y, 0.0, 1)], 8.0),
        "dock" + n: segment([(pick_x, pick_y, 0.0, 1), (-1.5, 0.0, 0.0, 1)], 8.0),
        "in" + n: segment([(-1.5, 0.0, 0.0, 1), (3.0, 0.0, 0.0, 1), (7.0, -0.8, 0.0, 1), (10.0, -0.8, 0.0, 1)], 8.0),
        "back" + n: segment([(10.0, -0.8, 0.0, -1), (6.0, -0.8, 0.0, -1), (2.0, 0.0, 0.0, -1),
                             (-1.5, 0.0, 0.0, -1)], 5.0)}


TRAILER_LEVELS = {"approach": 0.05, "carry": 0.15, "set": 0.0, "rest": 0.0, "side": 0.08}


def test_load_trailer_green() -> None:
    plan = Plan("load_trailer", trailer_segments(), TRAILER_LEVELS)
    run = Run(plan, make_plant(-10.0, 0.4, 0.0, restraint=lambda t: True)).until(200)
    assert_clean(run)
    assert_stops_on_segments(run)
    assert abs(run.finish_obs("set_down").sideshift - 0.08) <= 0.003
    assert abs(run.plant.sideshift) <= 0.003 and run.plant.fork_height <= 0.005


def test_load_trailer_red_restraint_waits_short_of_leveler() -> None:
    plan = Plan("load_trailer", trailer_segments(), TRAILER_LEVELS)
    run = Run(plan, make_plant(-10.0, 0.4, 0.0, restraint=lambda t: False)).until(100)
    assert run.op.phase == "restraint_check" and not run.op.done
    assert run.ctrl.park and abs(run.plant.speed) < REST_SPEED and run.plant.x < -1.0
    assert_pose(run.obs, plan.segments["dock"].end)
    run.plant.restraint = lambda t: True
    run.until(300)
    assert_clean(run)
    assert_stops_on_segments(run)


def test_restraint_absent_does_not_block() -> None:
    plan = Plan("load_trailer", trailer_segments(), TRAILER_LEVELS)
    assert Run(plan, make_plant(-10.0, 0.4, 0.0)).until(200).op.done


def test_load_two_bales_side_by_side() -> None:
    segs = trailer_segments("1", pick_y=0.0) | trailer_segments("2", pick_y=-1.0, pick_x=-6.0, start=(-12.0, 0.0))
    segs["back1"] = segment(segs["back1"].points + ((-12.0, 0.0, 0.0, -1),), 5.0)
    levels = dict(TRAILER_LEVELS, side1=0.08, side2=-0.08, bales=2)
    run = Run(Plan("load_bales", segs, levels), make_plant(-10.0, 0.4, 0.0)).until(400)
    assert_clean(run)
    assert_stops_on_segments(run)
    assert abs(run.finish_obs("set_down1").sideshift - 0.08) <= 0.003
    assert abs(run.finish_obs("set_down2").sideshift + 0.08) <= 0.003


def test_speed_controller_feed_forward_and_brake_split() -> None:
    down = -math.atan(0.136)
    throttle, brake = speed_commands(1.4, 1.4, 1, down, 0.0, SPEC)       # holding speed down a ramp: brake, no throttle
    assert throttle == 0.0 and 0.3 < brake < 0.6
    throttle, brake = speed_commands(1.4, 1.4, 1, 0.0, 0.0, SPEC)         # level: nothing
    assert (throttle, brake) == (0.0, 0.0)
    throttle, brake = speed_commands(1.0, -0.2, -1, 0.0, 0.0, SPEC)       # reverse start: negative throttle
    assert throttle < 0 and brake == 0.0


def test_park_only_at_rest_and_phase_names() -> None:
    op = Operator(start_park_plan(), SPEC)
    assert op.phase_list[:4] == ["key_on", "engine_start", "release_brake", "to_bay"]
    assert op.phase == "key_on" and not op.done
