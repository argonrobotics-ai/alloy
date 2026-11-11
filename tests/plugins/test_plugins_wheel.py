# SPDX-License-Identifier: Apache-2.0
"""From an installed wheel, the way users get Alloy (agent friction trial 2026-11-07, F2 and F3).

A wheel is built from this tree (``uv build --wheel``, as tools/deps/wheels.py builds the release) and unpacked into
a site folder; everything then runs from an empty folder with only that folder on PYTHONPATH and ALLOY_CPU_ROUTE
unset, i.e. on the users' default CPU route (Alloy's MuJoCo C):
- F2: ``python -m alloy.plugins new`` finds its templates in the installed package;
- F3: the untouched template passes the full check, graph_replay included;
- AUDIT AGENTS-7 / GUIDE-2 / TOOLS-2: the two-line quick start of AGENTS.md section 10 and docs/guide/plugins.md,
  ``python -m alloy.plugins new my_force --kind force`` then ``python -m alloy.plugins check my_force`` (and the
  guide's ``my_wind``), run exactly so in a fresh directory (it failed: "my_force does not exist", exit 2);
- F4: a force plug-in on a light body (a 10 g ball in a 10 m/s wind) in an engine built as ``alloy.api`` builds it,
  with health on, gets no ENERGY_JUMP: health books the plug-in forces' work (WP5's ForceWork, f4066e922). Before,
  the world froze at x = 0.08 m within 0.35 s.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build the wheel")]


@pytest.fixture(scope="module")
def site(tmp_path_factory) -> Path:
    """This tree's alloy wheel, unpacked (a pure-Python wheel's files are its installed layout)."""
    work = tmp_path_factory.mktemp("wheel")
    src = work / "src"
    src.mkdir()
    shutil.copy2(ROOT / "pyproject.toml", src / "pyproject.toml")
    shutil.copytree(ROOT / "alloy", src / "alloy", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.so"))
    subprocess.run(["uv", "build", "--wheel", "--no-sources", "--out-dir", str(work / "dist"), str(src)],
                   check=True, capture_output=True, text=True, timeout=600)
    (wheel,) = (work / "dist").glob("*.whl")
    out = work / "site"
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(out)
    return out


def _run(site: Path, cwd: Path, *args: str, timeout: float = 1500) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "ALLOY_CPU_ROUTE"}
    env["PYTHONPATH"] = str(site)
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)


def test_installed_package_is_the_wheel(site, tmp_path):
    res = _run(site, tmp_path, "-c", "import alloy.plugins.templates as t; print(t.__file__)")
    assert res.returncode == 0, res.stderr[-2000:]
    assert res.stdout.strip().startswith(str(site)), res.stdout


def test_new_and_full_check_from_the_wheel(site, tmp_path):
    made = _run(site, tmp_path, "-m", "alloy.plugins", "new", "wind", "--kind", "force", "--dir", str(tmp_path))
    assert made.returncode == 0, made.stdout + made.stderr[-2000:]
    plugin = tmp_path / "wind" / "wind" / "plugin.py"
    assert plugin.exists() and "class Wind(ForcePlugin)" in plugin.read_text()
    out = tmp_path / "report.json"
    res = _run(site, tmp_path, "-m", "alloy.plugins", "check", str(plugin), "--out", str(out))
    report = json.loads(out.read_text())
    checks = {c["name"]: c for c in report["checks"]}
    assert res.returncode == 0 and report["passed"], json.dumps(report["failed"]) + res.stderr[-2000:]
    assert checks["graph_replay"]["passed"] and not checks["graph_replay"]["skipped"]
    assert "MuJoCo Warp" in checks["graph_replay"]["detail"]          # the default CPU route cannot be captured


@pytest.mark.parametrize("name", ["my_force", "my_wind"])       # AGENTS.md section 10; docs/guide/plugins.md
def test_documented_quick_start_from_the_wheel(site, tmp_path, name):
    made = _run(site, tmp_path, "-m", "alloy.plugins", "new", name, "--kind", "force")
    assert made.returncode == 0, made.stdout + made.stderr[-2000:]
    assert json.loads(made.stdout)["next"] == f"python -m alloy.plugins check {name}"
    res = _run(site, tmp_path, "-m", "alloy.plugins", "check", name)
    assert res.returncode == 0, res.stdout[-1500:] + res.stderr[-2000:]
    report = json.loads(res.stdout)
    assert report["passed"] and report["plugin"] == name and report["failed"] == []
    ran = {c["name"] for c in report["checks"] if not c["skipped"]}
    assert {"crash", "determinism", "graph_replay", "terminal_velocity", "speed"} <= ran


WIND = """
import json, sys
from dataclasses import dataclass
import numpy as np, warp as wp, newton
from alloy.api._plugins import build_engine
from alloy.plugins import ForcePlugin, TestScene, body_xform
from alloy.plugins.funcs import add_wrench
from alloy.plugins.harness.scene import build_model
from alloy.precision.types import HealthFlags
from alloy.scene.builtin_plugins import register_builtin_solvers
from alloy.testing.integration import built_from_model, entry_spec


@wp.kernel
def _wind(body_qd: wp.array[wp.spatial_vector], body: wp.array[wp.int32], wind: wp.vec3, k: wp.float32,
          body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    b = body[w]
    qd = body_qd[b]
    rel = wind - wp.vec3(qd[0], qd[1], qd[2])
    f = k * wp.length(rel) * rel
    add_wrench(body_f, b, wp.spatial_vector(f[0], f[1], f[2], 0.0, 0.0, 0.0))


class Wind(ForcePlugin):
    \"\"\"Quadratic drag of a 10 m/s wind on one body (the trial's plug-in, k = rho Cd A / 2).\"\"\"
    name = "trial_wind"
    summary = "wind"

    @dataclass
    class Config:
        body: str = "ball"
        k: float = 2.0e-4

    def setup(self, ctx):
        self.body = ctx.bodies(self.config.body)

    def apply(self, state, forces, dt):
        wp.launch(_wind, dim=self.ctx.world_count, inputs=[state.body_qd, self.body, wp.vec3(10.0, 0.0, 0.0),
                  float(self.config.k)], outputs=[forces.body_f], device=self.ctx.device)


def ball(b):
    body = b.add_body(xform=body_xform(0.0, 0.0, 0.0151), label="ball")
    rho = 0.010 / (4.0 / 3.0 * np.pi * 0.015 ** 3)
    b.add_shape_sphere(body, radius=0.015, cfg=newton.ModelBuilder.ShapeConfig(density=rho, mu=0.5))


register_builtin_solvers()
model = build_model(TestScene(build=ball, worlds=2), "cpu")
built = built_from_model(model, [entry_spec(model, "mujoco", 2e-3)], control_dt=0.02)
sim = build_engine(built, "standard", seed=0, plugins=[Wind()])
for _ in range(25):
    sim.step_period()
rep = sim.parts.health.poll()
flags = [[f.name for f in HealthFlags if int(rep.flags[w]) & int(f)] for w in range(len(rep.flags))]
json.dump({"severity": rep.severity.tolist(), "flags": flags, "x": sim.state.body_q.numpy()[:, 0].tolist()},
          sys.stdout)
"""


def test_wind_on_a_light_body_keeps_health_from_the_wheel(site, tmp_path):
    script = tmp_path / "wind.py"
    script.write_text(WIND)
    res = _run(site, tmp_path, str(script), timeout=900)
    assert res.returncode == 0, res.stderr[-3000:]
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["flags"] == [[], []] and max(out["severity"]) == 0, out
    assert min(out["x"]) > 0.1, out                    # the ball rolls downwind (froze at 0.08 m with ENERGY_JUMP)
