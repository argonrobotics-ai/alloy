# SPDX-License-Identifier: Apache-2.0
"""The rest of ``alloy.Scene`` and ``alloy.Sim`` without Python: recipes take every ``Scene.add_*`` kind (pour,
heater, push included) and ``Scene.set_*`` rows, and drive, pose, wrench, settle, heater and controls act on an open
simulation (one real CPU run of a kinematic belt and a cube)."""

from __future__ import annotations

import pytest

import alloy
from alloy.cli.registry import OPS, respond
from alloy.cli.scenes import ADDERS, SETTERS, from_recipe, to_scene
from alloy.cli.session import Session

KITCHEN = {"dt": 0.01, "objects": [
    {"add": "ground"},
    {"add": "open_box", "name": "pan", "inner": [0.1, 0.1, 0.03], "wall": 0.005, "pos": [0.3, 0, 0], "static": True},
    {"add": "heater", "name": "burner", "size": [0.1, 0.1, 0.01], "pos": [-0.3, 0, 0.005], "temperature": 370.0},
    {"add": "pour", "name": "water", "volume": 2e-5, "pos": [0.3, 0, 0.1], "temperature": 293.15},
    {"add": "box", "name": "cube", "size": [0.05, 0.05, 0.05], "pos": [0, 0.3, 0.025], "mass": 0.2}],
    "set": [{"set": "heat", "time_scale": 60}, {"set": "heater", "name": "pan_floor", "temperature": 353.15}]}
BELT = {"dt": 0.01, "objects": [
    {"add": "ground"},
    {"add": "box", "name": "belt", "size": [0.4, 0.2, 0.04], "pos": [0, 0, 0.02], "kinematic": True},
    {"add": "box", "name": "cube", "size": [0.05, 0.05, 0.05], "pos": [0.5, 0, 0.025], "mass": 0.2}]}


def test_recipes_take_every_scene_add_and_set_method():
    adds = {m[4:] for m in dir(alloy.Scene) if m.startswith("add_") and m != "add_material"}
    sets = {m[4:] for m in dir(alloy.Scene) if m.startswith("set_")}
    assert set(ADDERS) == adds and {"pour", "heater", "push"} <= adds
    assert set(SETTERS) == sets and {"heater", "heat", "temperature"} <= sets
    scene = from_recipe(KITCHEN)
    assert {"burner", "water", "pan_floor"} <= set(scene.names())
    assert to_scene(KITCHEN, worlds=3).to_dict() == {**scene.to_dict(), "worlds": 3}    # settings survive


@pytest.mark.parametrize(("row", "field"), [
    ({"set": "colour", "name": "cube"}, "set[0].set"),
    ({"set": "heater", "name": "pan_floor", "temperature": 353.15, "watts": 3}, "set[0].watts"),
])
def test_a_bad_set_row_names_the_field(row, field):
    reply = respond(Session(), {"op": "scene", "scene": {"objects": [{"add": "ground"}], "set": [row]}})
    assert not reply["ok"] and reply["error"]["field"] == field and reply["error"]["fix"], reply


@pytest.mark.parametrize(("request_", "field"), [
    ({"op": "drive", "velocity": [1, 0, 0]}, "name"),
    ({"op": "drive", "name": "belt", "stop": True}, "name"),
    ({"op": "pose", "name": "cube"}, "pos"),
    ({"op": "heater", "name": "burner"}, "temperature"),
    ({"op": "heater", "name": "burner", "temperature": 300, "power": 5}, "temperature"),
    ({"op": "heater", "name": "burner", "power": "hot"}, "power"),
    ({"op": "controls", "controls": {"arm/actuator1": "up"}}, "controls.arm/actuator1"),
    ({"op": "wrench", "name": "cube", "force": "up"}, "force"),
])
def test_control_arguments_are_checked_before_a_scene_is_built(request_, field):
    built = []
    session = Session(builder=lambda scene, **kw: built.append(scene))
    reply = respond(session, {**request_, "scene": BELT})
    assert not reply["ok"] and reply["error"]["field"] == field and reply["error"]["fix"], reply
    assert built == [] and session.sims == {}


def test_drive_pose_wrench_settle_on_cpu():
    assert {"drive", "pose", "wrench", "settle", "heater", "controls"} <= set(OPS)
    session = Session()
    try:
        assert respond(session, {"op": "build", "scene": BELT, "device": "cpu"})["ok"]
        driving = respond(session, {"op": "drive", "name": "belt", "velocity": [0.25, 0, 0], "duration": 0.2})
        assert driving["ok"], driving
        moved = respond(session, {"op": "step", "steps": 10, "names": ["belt"]})["result"]["state"]["belt.pos"]
        assert moved[0] == pytest.approx(0.025, abs=1e-3)
        assert respond(session, {"op": "drive", "stop": True})["result"]["stopped"]
        posed = respond(session, {"op": "pose", "name": "cube", "pos": [0.5, 0.0, 0.3]})["result"]
        assert posed["state"]["cube.pos"] == pytest.approx([0.5, 0.0, 0.3], abs=1e-6)
        assert respond(session, {"op": "wrench", "name": "cube", "force": [0, 0, 1.0]})["ok"]
        settled = respond(session, {"op": "settle", "seconds": 0.2})["result"]
        assert settled["t"] == pytest.approx(0.3) and settled["health"]["ok"]
        unknown = respond(session, {"op": "pose", "name": "zz", "pos": [0, 0, 1]})
        assert unknown["error"]["type"] == "UnknownNameError"
        no_heat = respond(session, {"op": "heater", "name": "belt", "temperature": 300})
        assert not no_heat["ok"] and no_heat["error"]["fix"]
    finally:
        session.close_all()
