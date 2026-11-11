# SPDX-License-Identifier: Apache-2.0
"""Operation specs: JSON Schema, request checking, command-line flag parsing, and every example request."""

from __future__ import annotations

import json

import pytest

from alloy.cli.errors import RequestError
from alloy.cli.registry import OPS, SCENE
from alloy.cli.scenes import RECIPE_KEYS
from alloy.cli.spec import check_request, describe, json_schema, parse_flag


def test_every_example_is_a_valid_request():
    for op in OPS.values():
        args = check_request(op, op.example)                    # raises on drift between spec and example
        assert set(args) == {p.name for p in op.params}, op.name
        assert set(op.example) <= set(json_schema(op)["properties"]), op.name


def test_schemas_are_json_and_closed():
    for op in OPS.values():
        schema = json.loads(json.dumps(json_schema(op)))
        assert schema["type"] == "object" and schema["additionalProperties"] is False
        for name, prop in schema["properties"].items():
            assert prop["description"], (op.name, name)
    assert set(OPS["run"].param("seconds").json_schema()) >= {"type", "minimum", "description"}
    assert json_schema(OPS["run"])["required"] == ["seconds"]
    recipe = SCENE.json_schema()["anyOf"][1]
    assert set(recipe["properties"]) | {"schema"} == set(RECIPE_KEYS)


def test_unknown_missing_and_wrong_arguments_say_what_to_change():
    with pytest.raises(RequestError, match="unknown argument") as info:
        check_request(OPS["step"], {"stpes": 3})
    assert "steps" in info.value.fix and info.value.field == "stpes"
    with pytest.raises(RequestError, match="missing argument 'seconds'"):
        check_request(OPS["run"], {})
    with pytest.raises(RequestError, match="must be integer") as info:
        check_request(OPS["step"], {"steps": 1.5})
    assert info.value.fix.startswith("pass steps=")
    with pytest.raises(RequestError, match=">= 0"):
        check_request(OPS["step"], {"steps": -1})
    with pytest.raises(RequestError, match="one of"):
        check_request(OPS["step"], {"points": "some"})
    with pytest.raises(RequestError, match="must be boolean"):
        check_request(OPS["check"], {"quick": "yes"})
    with pytest.raises(RequestError, match="JSON object"):
        check_request(OPS["step"], [1, 2])


def test_defaults_and_world_kind():
    args = check_request(OPS["step"], {"world": "all"})
    assert args["steps"] == 1 and args["world"] == "all" and args["points"] == "summary" and args["sim"] is None
    assert check_request(OPS["run"], {"seconds": 1})["seconds"] == 1.0           # ints accepted as numbers
    with pytest.raises(RequestError):
        check_request(OPS["step"], {"world": "first"})
    with pytest.raises(RequestError):
        check_request(OPS["step"], {"world": True})


def test_flag_parsing():
    p = OPS["step"].param
    assert parse_flag(p("steps"), "12") == 12
    assert parse_flag(p("names"), "cube, peg") == ["cube", "peg"]
    assert parse_flag(p("names"), '["a"]') == ["a"]
    assert parse_flag(p("targets"), '{"lift": 0.1}') == {"lift": 0.1}
    assert parse_flag(p("world"), "all") == "all" and parse_flag(p("world"), "3") == 3
    assert parse_flag(p("scene"), "demo_peg_d12") == "demo_peg_d12"
    assert parse_flag(p("scene"), '{"objects": []}') == {"objects": []}
    assert parse_flag(OPS["render"].param("look_at"), "1,-1,0.5,0,0,0") == [1.0, -1.0, 0.5, 0.0, 0.0, 0.0]
    with pytest.raises(RequestError, match="not valid JSON"):
        parse_flag(p("targets"), "{lift: 1}")
    with pytest.raises(RequestError, match="must be number"):
        parse_flag(OPS["run"].param("seconds"), "two")


def test_describe_lists_every_operation():
    rows = describe(OPS.values())
    assert [r["name"] for r in rows] == list(OPS)
    assert all(r["example"]["op"] == r["name"] and r["input_schema"]["type"] == "object" for r in rows)
    assert {"build", "step", "check", "render", "estimate", "import", "export"} <= set(OPS)
