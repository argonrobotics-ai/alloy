# SPDX-License-Identifier: Apache-2.0
"""What ``alloy scene`` accepts, a build accepts: a second randomize of one object field replaces the first (with a
warning naming both ranges when both came from randomize calls; silently when it overrides the object's own card),
and a worlds override that does not match an object's per-world starts is refused before the build."""

from __future__ import annotations

import logging

import pytest

import alloy
from alloy.cli.registry import passed, respond
from alloy.cli.session import Session


def test_a_second_randomize_replaces_the_first_and_warns(caplog):
    scene = alloy.Scene(worlds=4)
    scene.add_ground()
    scene.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.1), mass=0.1)
    scene.randomize("cube", "x", -0.01, 0.01)
    with caplog.at_level(logging.WARNING, logger="alloy.api"):
        scene.randomize("cube", "x", -0.02, 0.02)
        scene.randomize("cube", "y", -0.01, 0.01)               # another field: no warning
    assert scene.items["cube"]["randomize"] == [{"field": "pose.0", "lo": -0.02, "hi": 0.02},
                                                {"field": "pose.1", "lo": -0.01, "hi": 0.01}]
    warned = [r.getMessage() for r in caplog.records if "replaces" in r.getMessage()]
    assert len(warned) == 1 and "-0.02" in warned[0] and "-0.01" in warned[0], warned
    sim = alloy.build(scene, device="cpu")                     # was CardError: 'pose.0' randomized twice
    assert sim.worlds == 4


def test_a_recipe_overrides_its_base_cards_range_without_a_warning(caplog):
    recipe = {"base": "demo_peg_d12", "randomize": [{"name": "peg_d12", "field": "x", "lo": -0.005, "hi": 0.005}]}
    with caplog.at_level(logging.WARNING, logger="alloy.api"):
        reply = respond(Session(), {"op": "check", "scene": recipe, "quick": True, "device": "cpu"})
    assert passed(reply), reply["result"]["findings"][:2]
    assert not [r for r in caplog.records if "replaces" in r.getMessage()]


@pytest.fixture()
def per_world(tmp_path):
    """A Scene JSON file whose box has one start per world, for 2 worlds."""
    scene = alloy.Scene(worlds=2)
    scene.add_ground()
    scene.add_box("b", size=(0.05, 0.05, 0.05), pos=[[0, 0, 0.1], [0.2, 0, 0.1]], mass=0.1)
    path = tmp_path / "per_world.json"
    path.write_text(scene.to_json(), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("request_", [
    {"op": "scene", "scene": {"base": "PATH", "worlds": 4}},
    {"op": "scene", "scene": "PATH", "worlds": 4},
    {"op": "estimate", "scene": "PATH", "worlds": 4, "device": "cpu"},
    {"op": "build", "scene": {"base": "PATH", "worlds": 4}, "device": "cpu"},     # was a raw IndexError
])
def test_a_worlds_override_must_match_per_world_starts(per_world, request_):
    request_ = {k: (per_world if v == "PATH" else {**v, "base": per_world} if isinstance(v, dict) else v)
                for k, v in request_.items()}
    reply = respond(Session(), request_)
    assert not reply["ok"] and reply["error"]["type"] == "SceneError" and reply["error"]["field"] == "worlds"
    assert "worlds=2" in reply["error"]["fix"], reply


def test_matching_worlds_and_a_smaller_estimate_still_pass(per_world):
    assert respond(Session(), {"op": "scene", "scene": {"base": per_world, "worlds": 2}})["ok"]
    estimate = respond(Session(), {"op": "estimate", "scene": per_world, "worlds": 1, "device": "cpu"})
    assert estimate["ok"], estimate
