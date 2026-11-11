# SPDX-License-Identifier: Apache-2.0
"""The OS sandbox: Landlock confinement, scrubbed environment, and a hostile plug-in that cannot get out."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from alloy.plugins.sandbox import read_paths, sandboxed_check, scrubbed_env
from alloy.plugins.sandbox.landlock import abi_version

needs_landlock = pytest.mark.skipif(abi_version() < 1, reason="kernel without Landlock")


def test_scrubbed_env_drops_credentials(monkeypatch, tmp_path):
    secrets = {"MY_API_TOKEN", "AWS_REGION", "GOOGLE_APPLICATION_CREDENTIALS", "ALLOY_HIDDEN_SET", "OPENAI_X"}
    for k in secrets:
        monkeypatch.setenv(k, "x")
    env = scrubbed_env(tmp_path / "h", tmp_path / "t", tmp_path / "c")
    assert not secrets & set(env)
    assert env["HOME"] == str(tmp_path / "h") and env["PATH"] == os.environ["PATH"]


def test_child_imports_the_parents_alloy_without_pythonpath(monkeypatch, tmp_path):
    """rc1 nightly red (2026-11-10): pytest run from the tree with no PYTHONPATH and no install; the sandbox child,
    started in its own folder, failed with "No module named 'alloy'". The child now gets the parent's import path."""
    import alloy

    monkeypatch.delenv("PYTHONPATH", raising=False)
    env = scrubbed_env(tmp_path / "h", tmp_path / "t", tmp_path / "c")
    root = str(Path(alloy.__file__).resolve().parents[1])
    assert root in env["PYTHONPATH"].split(os.pathsep)
    res = subprocess.run([sys.executable, "-c", "import alloy.plugins.sandbox.child as c, alloy; print(alloy.__file__)"],
                         cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-1500:]
    assert Path(res.stdout.strip().splitlines()[-1]).resolve() == Path(alloy.__file__).resolve()


def test_read_paths_never_include_home_or_its_parents():
    home = Path.home().resolve()
    for p in read_paths("magnet", ()):
        assert not home.is_relative_to(Path(p)), p


@needs_landlock
def test_landlock_blocks_reads_writes_and_tcp(tmp_path):
    inside, outside = tmp_path / "in", tmp_path / "out"
    inside.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    code = textwrap.dedent(f'''
        import socket, sys
        from alloy.plugins.sandbox.landlock import restrict_self
        restrict_self(read=[sys.prefix, sys.base_prefix, "/usr", "/lib", "/lib64", "/etc", "/proc"],
                      write=["{inside}"])
        open("{inside}/ok.txt", "w").write("ok")
        results = []
        for probe in (lambda: open("{outside}/secret.txt").read(), lambda: open("{outside}/new.txt", "w"),
                      lambda: socket.create_connection(("127.0.0.1", 9), timeout=2)):
            try:
                probe(); results.append("allowed")
            except PermissionError:
                results.append("blocked")
            except OSError:
                results.append("other")
        print(results)
    ''')
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    expect_tcp = "blocked" if abi_version() >= 4 else "other"
    assert res.stdout.strip() == str(["blocked", "blocked", expect_tcp])
    assert (inside / "ok.txt").read_text() == "ok" and not (outside / "new.txt").exists()


HOSTILE = '''
import os, socket
from pathlib import Path
from alloy.plugins import ForcePlugin, PluginError, TestScene


class Hostile(ForcePlugin):
    """Tries to leave the sandbox."""
    name = "hostile_test"
    summary = "probe"

    def update(self, state_in, state_out, dt):
        pass

    def setup(self, ctx):
        Path("{target}").write_text("escaped")

    def test_scene(self):
        return TestScene(build=lambda b: b.add_shape_box(b.add_body(label="x"), hx=0.1, hy=0.1, hz=0.1))
'''


@needs_landlock
@pytest.mark.slow
def test_hostile_plugin_cannot_write_outside(tmp_path):
    target = Path.home() / f".alloy-sandbox-probe-{os.getpid()}"
    src = tmp_path / "hostile.py"
    src.write_text(HOSTILE.format(target=target))
    try:
        report = sandboxed_check(str(src), quick=True, timeout_s=600)
        assert not target.exists()
        assert not report.passed and "PermissionError" in report.summary()
        assert any(c.group == "sandbox" and "landlock" in c.detail for c in report.checks)
    finally:
        target.unlink(missing_ok=True)
