# SPDX-License-Identifier: Apache-2.0
"""Numbers must be real and finite: NaN, Infinity and booleans in a number list (CLI and MCP) and in a compose
spec (area, worlds, count, robot pose) are refused with the field and a fix that works, never a raw crash; and
``python -m alloy.compose`` answers JSON on stdout with exit code 2."""

from __future__ import annotations

import json
import math

import pytest

from alloy.cli.registry import respond
from alloy.cli.session import Session
from alloy.compose.errors import ComposeError
from alloy.compose.spec import RobotSpec, SceneSpec

NAN, INF = math.nan, math.inf


@pytest.mark.parametrize("area", ["bad", [NAN, 0, 0.2, 0.2], [-INF, -INF, INF, INF], [True, 0, 1, 1]])
def test_compose_area_must_be_finite_numbers(area):
    reply = respond(Session(), {"op": "compose", "sentence": "a wooden cube on a table", "area": area,
                                "check": "none"})
    assert not reply["ok"] and reply["error"]["type"] == "RequestError" and reply["error"]["field"] == "area"
    assert "x0 y0 x1 y1" in reply["error"]["fix"] and "[0.1, 0.2]" not in reply["error"]["fix"]
    json.dumps(reply, allow_nan=False)                               # the reply itself is valid JSON


@pytest.mark.parametrize(("op", "args", "field"), [
    ("render", {"look_at": [1, 1, 1, 0, 0, NAN]}, "look_at"),
    ("pose", {"name": "cube", "pos": [0, 0, INF]}, "pos"),
    ("drive", {"name": "belt", "velocity": [[NAN, 0, 0]]}, "velocity"),
])
def test_cli_number_lists_refuse_nan_and_infinity(op, args, field):
    reply = respond(Session(builder=lambda *a, **k: pytest.fail("built")), {"op": op, "scene": "x", **args})
    assert not reply["ok"] and reply["error"]["field"] == field and "finite" in reply["error"]["message"], reply


@pytest.mark.parametrize(("data", "field"), [
    ({"items": [{"query": "cube"}], "worlds": True}, "worlds"),
    ({"items": [{"query": "cube", "count": True}]}, "items.count"),
    ({"items": [{"query": "cube"}], "area": [NAN, 0, 0.2, 0.2]}, "area"),
    ({"items": [{"query": "cube"}], "area": [-INF, -INF, INF, INF]}, "area"),
    ({"items": [{"query": "cube"}], "robots": [{"path": "arm.xml", "pose": [0, 0, 0, INF]}]}, "robots.pose"),
    ({"items": [{"query": "cube"}], "robots": [{"path": "arm.xml", "reach": INF}]}, "robots.reach"),
])
def test_compose_spec_refuses_bools_and_non_finite_numbers(data, field):
    with pytest.raises(ComposeError) as info:
        SceneSpec.from_dict(data)
    assert info.value.field.startswith(field.split(".")[0]) and info.value.fix


def test_a_finite_spec_still_parses():
    spec = SceneSpec.from_dict({"items": [{"query": "cube", "count": 2}], "worlds": 3, "area": [-0.2, -0.2, 0.2, 0.2],
                                "robots": [{"path": "arm.xml", "pose": [0, -0.4, 0, 1.57]}]})
    assert spec.worlds == 3 and spec.items[0].count == 2 and isinstance(spec.robots[0], RobotSpec)


@pytest.mark.parametrize("area", [["nan", "0", "0.2", "0.2"], ["-inf", "-inf", "inf", "inf"]])
def test_python_m_alloy_compose_answers_json_with_exit_2(area, capfd):
    from alloy.compose.cli import main

    code = main(["build", "a wooden cube on a table", "--area", *area, "--check", "none"])
    reply = json.loads(capfd.readouterr().out)
    assert code == 2 and reply["error"] == "ComposeError" and reply["field"] == "area", reply
