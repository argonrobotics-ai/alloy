# SPDX-License-Identifier: Apache-2.0
"""``--liquid-accuracy`` on the command line and in MCP (SCR-WP9-LIQRES): estimate shows both settings and names the
scene's, build routes at it, a recipe or Scene JSON takes it, and check refuses it (saved cards keep no setting)."""

from __future__ import annotations

import io
import json

from alloy.cli import main
from alloy.cli.registry import OPS
from alloy.cli.registry import respond
from alloy.cli.scenes import to_scene
from alloy.cli.session import Session
from alloy.cli.testing import fake_build
from alloy.mcp.tools import tool_list

RECIPE = json.dumps({"objects": [{"add": "ground"},
                                 {"add": "open_cylinder", "name": "cup", "inner_radius": 0.02, "depth": 0.04},
                                 {"add": "liquid", "name": "water", "size": [0.03, 0.03, 0.02],
                                  "pos": [0, 0, 0.015], "shape": "cylinder"}]})


def _run(*argv: str) -> dict:
    out = io.StringIO()
    main(list(argv), stdout=out)
    return json.loads(out.getvalue())


def test_the_flag_is_on_every_building_op_and_in_mcp():
    for name in ("build", "run", "step", "state", "estimate"):
        assert "liquid_accuracy" in {p.name for p in OPS[name].params}, name
    tools = {t["name"]: t for t in tool_list()}
    prop = tools["estimate"]["inputSchema"]["properties"]["liquid_accuracy"]
    assert prop["enum"] == ["default", "super"]


def test_to_scene_sets_it_and_keeps_the_default_out_of_json():
    assert to_scene(json.loads(RECIPE)).liquid_accuracy == "default"
    sc = to_scene(json.loads(RECIPE), liquid_accuracy="super")
    assert sc.liquid_accuracy == "super" and sc.to_dict()["liquid_accuracy"] == "super"
    assert to_scene(sc.to_dict()).liquid_accuracy == "super"                  # Scene JSON carries it


def test_estimate_names_the_setting_and_both_rows():
    reply = _run("estimate", RECIPE, "--liquid-accuracy", "super", "--device", "cpu", "--seconds", "0.2")
    assert reply["ok"], reply
    res = reply["result"]
    assert res["liquid_accuracy"] == "super" and set(res["liquid"]) == {"default", "super"}
    assert res["liquid"]["super"]["particles"] > 6 * res["liquid"]["default"]["particles"]


def test_check_refuses_the_setting_it_cannot_carry():
    reply = _run("check", RECIPE, "--liquid-accuracy", "super", "--device", "cpu", "--quick")
    assert not reply["ok"] and reply["error"]["field"] == "liquid_accuracy"
    assert "sim=" in reply["error"]["fix"]


def test_build_uses_it_and_the_reply_names_it():
    seen = []

    def builder(scene, **kw):
        seen.append(scene.liquid_accuracy)
        sim = fake_build(scene, **kw)
        sim.scene = scene
        return sim

    reply = respond(Session(builder=builder), {"op": "build", "scene": json.loads(RECIPE), "liquid_accuracy": "super"})
    assert reply["ok"] and seen == ["super"] and reply["result"]["liquid_accuracy"] == "super", reply


def test_a_recipe_takes_it_as_a_key():
    recipe = {**json.loads(RECIPE), "liquid_accuracy": "super"}
    assert to_scene(recipe).liquid_accuracy == "super"
    assert to_scene({"base": to_scene(recipe).to_dict(), "worlds": 2}).liquid_accuracy == "super"
    reply = _run("scene", json.dumps({**recipe, "liquid_accuracy": "fine"}))
    assert not reply["ok"] and reply["error"]["fix"], reply


def test_check_refuses_a_scene_that_carries_super_without_the_flag(tmp_path):
    path = tmp_path / "super.json"
    path.write_text(json.dumps(to_scene(json.loads(RECIPE), liquid_accuracy="super").to_dict()))
    reply = _run("check", str(path), "--device", "cpu", "--quick")
    assert not reply["ok"] and reply["error"]["field"] == "liquid_accuracy", reply
