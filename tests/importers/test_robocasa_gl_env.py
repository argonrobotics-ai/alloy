# SPDX-License-Identifier: Apache-2.0
"""The RoboCasa importer picks MuJoCo's GL backend only for its own render: the caller's environment comes back as it was.

MuJoCo reads ``MUJOCO_GL`` once, at the first ``import mujoco`` in a process, and its OSMesa and EGL backends then set
``PYOPENGL_PLATFORM`` to match. A thumbnail used to leave ``MUJOCO_GL=osmesa`` (and so ``PYOPENGL_PLATFORM``) set in the
caller's process, and a later subprocess's ``import mujoco`` failed where PyOpenGL finds no OSMesa (the GitHub runner,
a python:3.12 container). Laws: after a thumbnail, in a fresh process whose first ``import mujoco`` is the render's
own, both variables are as the caller had them and a subprocess's ``import mujoco`` sees the caller's values, while
the render itself still gets OSMesa where its library exists; the command line's backend choice holds while its
command runs (batch workers inherit it) and is gone when ``main`` returns.
"""

from __future__ import annotations

import ctypes.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from alloy.importers.robocasa import cli

GL_VARS = ("MUJOCO_GL", "PYOPENGL_PLATFORM")
REPO = Path(__file__).resolve().parents[2]
MODEL = """<mujoco><asset><mesh name="tet" vertex="0 0 0  0.05 0 0  0 0.05 0  0 0 0.05"/></asset>
<worldbody><body name="b"><geom type="mesh" mesh="tet" group="2"/></body></worldbody></mujoco>"""
CHILD = """
import json, os, subprocess, sys
from pathlib import Path
from alloy.importers.robocasa.thumbs import render_thumbnail
VARS = ("MUJOCO_GL", "PYOPENGL_PLATFORM")
try:
    render_thumbnail(Path(sys.argv[1]), Path(sys.argv[2]))
    err = None
except Exception as exc:
    err = f"{type(exc).__name__}: {exc}"
probe = subprocess.run([sys.executable, "-c", "import json, os, mujoco; "
                        "print(json.dumps({k: os.environ.get(k) for k in %r}))" % (VARS,)],
                       capture_output=True, text=True, timeout=300)
print(json.dumps({"err": err, "after": {k: os.environ.get(k) for k in VARS}, "rc": probe.returncode,
                  "seen": probe.stdout.strip(), "stderr": probe.stderr[-2000:]}))
"""


def test_a_thumbnail_leaves_the_callers_gl_variables_for_later_subprocesses(tmp_path):
    model, out = tmp_path / "model.xml", tmp_path / "thumb.png"
    model.write_text(MODEL)
    env = {k: v for k, v in os.environ.items() if k not in (*GL_VARS, "DISPLAY")}
    run = subprocess.run([sys.executable, "-c", CHILD, str(model), str(out)], cwd=REPO, env=env, capture_output=True,
                         text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    res = json.loads(run.stdout.strip().splitlines()[-1])
    assert res["after"] == {"MUJOCO_GL": None, "PYOPENGL_PLATFORM": None}, res       # the caller's: unset
    assert res["rc"] == 0, res["stderr"]                                               # a later `import mujoco` works
    assert json.loads(res["seen"]) == {"MUJOCO_GL": None, "PYOPENGL_PLATFORM": None}, res
    if ctypes.util.find_library("OSMesa"):                                             # the render still got OSMesa
        assert res["err"] is None and out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", res


@pytest.mark.parametrize("before", [None, "egl"])
def test_the_command_lines_gl_choice_lasts_for_its_command_only(monkeypatch, capsys, before):
    for k in (*GL_VARS, "DISPLAY"):
        monkeypatch.delenv(k, raising=False)
    if before is not None:
        monkeypatch.setenv("MUJOCO_GL", before)
    seen = []
    monkeypatch.setattr(cli, "_packs", lambda _args: seen.append(os.environ.get("MUJOCO_GL")) or 0)
    assert cli.main(["packs"]) == 0
    picked = before or ("osmesa" if ctypes.util.find_library("OSMesa") else "disable")
    assert seen == [picked]                                       # in effect while the command runs (batch workers)
    assert {k: os.environ.get(k) for k in GL_VARS} == {"MUJOCO_GL": before, "PYOPENGL_PLATFORM": None}
