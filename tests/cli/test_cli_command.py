# SPDX-License-Identifier: Apache-2.0
"""The ``alloy`` command in-process: JSON out, exit codes, --json input, call, ops, version, and the console
entry through ``python -m alloy.cli``."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest

from alloy.cli import main


def run(argv, stdin=None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    out = io.StringIO()
    code = main(argv, stdout=out)
    text = out.getvalue()
    assert text.endswith("\n") and text.count("\n") == 1, text          # exactly one JSON line
    return code, json.loads(text)


def test_list_and_match():
    code, reply = run(["list", "scenes", "--match", "PEG"])
    assert code == 0 and reply == {"ok": True, "op": "list", "result": {"kind": "scenes", "count": 1,
                                                                         "items": ["demo_peg_d12"], "truncated": False}}
    code, reply = run(["list", "materials", "--match", "water", "--limit", "1"])
    assert reply["result"]["items"][0]["name"] == "water" and reply["result"]["items"][0]["kind"] == "fluid"
    code, reply = run(["list", "adds"])
    assert "box" in [r["name"] for r in reply["result"]["items"]]
    code, reply = run(["list", "ops"])
    assert {"build", "step", "render"} <= {r["name"] for r in reply["result"]["items"]}


def test_scene_flags_json_and_saving(tmp_path, monkeypatch):
    recipe = {"objects": [{"add": "ground"}, {"add": "box", "name": "cube", "size": [0.1, 0.1, 0.1]}]}
    code, reply = run(["scene", json.dumps(recipe), "--save", str(tmp_path / "s.json"), "--cards", str(tmp_path / "c")])
    assert code == 0 and reply["result"]["names"] == ["ground", "cube"]
    assert json.loads((tmp_path / "s.json").read_text())["schema"] == "alloy.api.scene/1"
    assert reply["result"]["scene_card"].endswith(".yaml")
    code, reply = run(["scene", str(tmp_path / "s.json"), "--worlds", "3"])
    assert reply["result"]["summary"]["worlds"] == 3
    code, reply = run(["scene", "--json", "-"], stdin=json.dumps({"scene": recipe, "worlds": 4}),
                      monkeypatch=monkeypatch)
    assert code == 0 and reply["result"]["summary"]["worlds"] == 4
    (tmp_path / "args.json").write_text(json.dumps({"scene": recipe, "worlds": 5}))
    code, reply = run(["scene", "--json", f"@{tmp_path / 'args.json'}", "--worlds", "6"])   # flags win
    assert reply["result"]["summary"]["worlds"] == 6


@pytest.mark.parametrize("argv,kind,field", [
    (["scene", '{"objects": [{"add": "box"}]}'], "SceneError", "objects[0].name"),
    (["scene", "no_such_scene_zz"], "SceneError", "source"),
    (["list", "nope"], "RequestError", "kind"),
    (["step", "--steps", "two"], "RequestError", "steps"),
    (["scene", "--json", "{bad"], "RequestError", "--json"),
    (["stpe"], "RequestError", "argv"),
    (["step", "--frobnicate", "1"], "RequestError", "argv"),
    (["ops", "nope"], "RequestError", "name"),
    (["call", '{"op": "zz"}'], "RequestError", "op"),
    (["batch", "/no/such/file.json"], "RequestError", "source"),
])
def test_errors_are_json_with_a_fix_and_exit_2(argv, kind, field):
    code, reply = run(argv)
    assert code == 2 and reply["ok"] is False
    assert reply["error"]["type"] == kind and reply["error"]["field"] == field and reply["error"]["fix"]


def test_call_and_ops(monkeypatch):
    code, reply = run(["call", "-"], stdin='{"op": "list", "kind": "scenes", "match": "cloth"}',
                      monkeypatch=monkeypatch)
    assert code == 0 and reply["result"]["items"] == ["demo_cloth_drop"]
    code, reply = run(["ops", "step"])
    (op,) = reply["result"]["ops"]
    assert op["name"] == "step" and "steps" in op["input_schema"]["properties"] and op["example"]["op"] == "step"
    code, reply = run(["ops"])
    assert len(reply["result"]["ops"]) >= 14


def test_python_dash_m_alloy_cli_is_the_command(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    res = subprocess.run([sys.executable, "-m", "alloy.cli", "list", "scenes", "--match", "peg", "--pretty"],
                         capture_output=True, text=True, timeout=120, env=env, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["result"]["items"] == ["demo_peg_d12"] and "\n  " in res.stdout
    res = subprocess.run([sys.executable, "-m", "alloy.cli", "--version"], capture_output=True, text=True,
                         timeout=120, env=env)
    assert res.returncode == 0 and res.stdout.startswith("alloy ")


@pytest.mark.parametrize("argv", [["--pretty", "list", "scenes"], ["list", "scenes", "--pretty"]])
def test_pretty_before_or_after_the_command(argv):
    out = io.StringIO()
    assert main(argv, stdout=out) == 0
    assert out.getvalue().startswith("{\n  ")
