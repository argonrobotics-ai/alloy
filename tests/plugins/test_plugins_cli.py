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


@pytest.mark.parametrize("argv", [
    [],                                                          # no command
    ["check", "magnet", "--bogus"],                              # an unknown flag
    ["--request", '{"command": "bogus"}'],                       # an unknown command in a request
    ["--request", '{"command": "list", "unknown": 1}'],          # an unknown request option
])
def test_bad_requests_still_reply_json_with_exit_2(argv, capfd):
    """Audit S2-042: argparse's own errors and a missing command printed usage text only (stdout empty)."""
    code, err = _run(argv, capfd)
    assert code == 2 and err["error"] == "PluginError" and err["fix"], err


def test_request_speed_matches_the_python_api():
    """Audit S2-042: ``"speed": false`` was dropped (the speed check still ran) and ``"speed": true`` was refused."""
    from alloy.plugins.cli import _args_from_request, _parser

    for speed in (False, True):
        req = json.dumps({"command": "check", "target": "magnet", "quick": True, "speed": speed})
        assert _args_from_request(_parser(), req).no_speed is (not speed)


def test_out_in_a_new_folder_is_written(capfd, monkeypatch, tmp_path):
    """Audit S2-042: ``--out new_dir/report.json`` ran the whole check, then crashed (FileNotFoundError, exit 1)."""
    import importlib

    from alloy.plugins.harness.report import CheckReport, CheckResult

    def fake_check(plugin, **_):
        report = CheckReport(plugin.name, plugin.kind, "magnet", "cpu")
        report.add(CheckResult("contract", "contract", True))
        return report

    harness = importlib.import_module("alloy.plugins.harness.check")   # the module (the package re-exports check())
    monkeypatch.setattr(harness, "check", fake_check)
    out = tmp_path / "new_dir" / "deeper" / "report.json"
    code, report = _run(["check", "magnet", "--quick", "--out", str(out)], capfd)
    assert code == 0 and report["passed"] and json.loads(out.read_text()) == report


def _guide_drag_file(dest):
    """GUIDE.md's force plug-in as one file: the code of section 2 (Anatomy), then section 6's scene and law."""
    import re
    from pathlib import Path

    import alloy.plugins

    guide = (Path(alloy.plugins.__file__).parent / "GUIDE.md").read_text()
    blocks = []
    for heading in ("## 2. Anatomy", "## 6. Test scenes and laws"):
        after = guide[guide.index(heading):]
        blocks.append(re.search(r"```python\n(.*?)```", after, re.S).group(1))
    dest.write_text("\n\n".join(blocks))
    return dest


def test_a_plugin_importing_top_level_newton_gets_the_import_to_write(capfd, monkeypatch, tmp_path):
    """Audit S2-048: ``import newton`` (no such package in a pip install) crashed the CLI with a bare
    ModuleNotFoundError and an empty stdout; it is now a JSON error naming Alloy's copy."""
    monkeypatch.setitem(sys.modules, "newton", None)               # as in a pip install: no top-level newton
    path = tmp_path / "drag.py"
    path.write_text("import newton\n")
    code, err = _run(["check", str(path), "--quick"], capfd)
    assert code == 2 and err["error"] == "PluginError" and "from alloy._vendor import newton" in err["fix"], err
    path.write_text("raise RuntimeError('a bug in the plug-in')\n")
    code, err = _run(["describe", str(path)], capfd)                # any other failure: JSON too
    assert code == 2 and err["error"] == "RuntimeError" and "traceback" in err["fix"], err


@pytest.mark.compile_heavy       # a quick check: 10-25 s, more on a loaded gate host
def test_guide_force_plugin_passes_as_written(capfd, monkeypatch, tmp_path):
    """Audit S2-048: GUIDE.md's worked Drag plug-in imported a top-level ``newton``, which a pip install lacks."""
    monkeypatch.setitem(sys.modules, "newton", None)
    path = _guide_drag_file(tmp_path / "drag.py")
    code, report = _run(["check", str(path), "--quick"], capfd)
    assert code == 0 and report["passed"] and report["plugin"] == "drag", report
