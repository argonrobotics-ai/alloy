# SPDX-License-Identifier: Apache-2.0
"""python -m alloy.plugins: JSON on stdout only, JSON requests, exit codes."""

from __future__ import annotations

import io
import json
import subprocess
import sys

import pytest

from alloy.plugins.cli import main


def _run(argv, capfd, stdin=None, monkeypatch=None):
    if stdin is not None:
        monkeypatch.setattr(sys, "stdin", io.StringIO(stdin))
    code = main(argv)
    out = capfd.readouterr().out
    return code, json.loads(out)


def test_list_and_describe(capfd):
    code, rows = _run(["list", "--kind", "force"], capfd)
    assert code == 0 and {"magnet", "ink", "hydraulic"} <= {r["name"] for r in rows}
    code, desc = _run(["describe", "hydraulic"], capfd)
    assert code == 0 and desc[0]["config"][0]["name"] == "barrel"


def test_errors_are_json_with_exit_2(capfd, monkeypatch):
    code, err = _run(["describe", "no_such_plugin"], capfd)
    assert code == 2 and err["error"] == "PluginError" and "available" in err["message"]
    code, err = _run(["--request", "-"], capfd, stdin="not json", monkeypatch=monkeypatch)
    assert code == 2 and "JSON" in err["message"]


def test_request_and_new(capfd, monkeypatch, tmp_path):
    req = json.dumps({"command": "new", "name": "gust", "kind": "sensor", "dir": str(tmp_path)})
    code, out = _run(["--request", "-"], capfd, stdin=req, monkeypatch=monkeypatch)
    assert code == 0 and (tmp_path / "gust" / "gust" / "plugin.py").exists()
    assert "check" in out["next"]


def test_stdout_is_pure_json_in_a_subprocess():
    res = subprocess.run([sys.executable, "-m", "alloy.plugins", "list"], capture_output=True, text=True, check=True)
    assert isinstance(json.loads(res.stdout), list)


@pytest.mark.parametrize("kind", ["force", "sensor", "object", "solver"])
def test_new_then_the_folder_is_a_target(kind, capfd, monkeypatch, tmp_path):
    """AUDIT AGENTS-7 / GUIDE-2 / TOOLS-2: right after ``new my_force``, ``my_force`` (the project folder ``new``
    made), its package folder and its plugin.py all name the plug-in, in a fresh directory, without --dir."""
    from alloy.plugins.registry import load_target

    monkeypatch.chdir(tmp_path)
    name = f"my_{kind}"
    code, made = _run(["new", name, "--kind", kind], capfd)
    assert code == 0 and made["next"] == f"python -m alloy.plugins check {name}"
    code, desc = _run(["describe", name], capfd)
    assert code == 0 and desc[0]["name"] == name and desc[0]["kind"] == kind
    for target in (name, f"{name}/{name}", f"{name}/{name}/plugin.py"):
        (cls,) = load_target(target)
        assert cls.name == name


@pytest.mark.compile_heavy       # a quick check: 10-25 s, more on a loaded gate host
def test_new_then_check_folder_passes(capfd, monkeypatch, tmp_path):
    """The documented pair, quick: ``new my_force --kind force`` then ``check my_force`` (the full, exact pair runs
    from a wheel in test_plugins_wheel.py)."""
    monkeypatch.chdir(tmp_path)
    assert _run(["new", "my_force", "--kind", "force"], capfd)[0] == 0
    code, report = _run(["check", "my_force", "--quick"], capfd)
    assert code == 0 and report["passed"] and report["plugin"] == "my_force", report


def test_folder_target_errors_carry_a_fix(capfd, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "empty").mkdir()
    code, err = _run(["check", "empty"], capfd)
    assert code == 2 and "not a plug-in folder" in err["message"] and "plugin.py" in err["fix"]
    code, err = _run(["check", "nowhere"], capfd)
    assert code == 2 and str(tmp_path) in err["message"] and "project folder" in err["fix"]
    code, err = _run(["new", "json"], capfd)                       # would hide the stdlib module
    assert code == 2 and "my_json" in err["fix"]
    (tmp_path / "magnet").mkdir()                                  # a folder that only shares a built-in's name
    code, desc = _run(["describe", "magnet"], capfd)
    assert code == 0 and desc[0]["target"].startswith("alloy.plugins.")


def test_same_named_packages_from_two_folders(monkeypatch, tmp_path):
    """A long-lived process (a tool server) can check ``my_force`` from one folder, then another."""
    from alloy.plugins.registry import load_target
    from alloy.plugins.scaffold import new_plugin

    for sub, kind in (("a", "force"), ("b", "sensor")):
        new_plugin("my_force", kind=kind, dest=tmp_path / sub)
        (cls,) = load_target(str(tmp_path / sub / "my_force"))
        assert cls.kind == kind
