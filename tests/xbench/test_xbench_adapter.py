# SPDX-License-Identifier: Apache-2.0
"""alloy.xbench (CPU): the class swap, bitwise-equal physics, health records, and the engine shim end to end."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
import alloy.xbench as xb
from alloy.xbench.solvers import is_alloy_solver

REPO = Path(__file__).resolve().parents[2]
SHIM = REPO / "bench" / "xbench" / "scenes" / "engine_shim.py"
STOCK = {n: getattr(newton.solvers, n) for n in ("SolverMuJoCo", "SolverXPBD", "SolverFeatherstone", "SolverVBD",
                                                  "SolverKamino", "SolverSemiImplicit")}


@pytest.fixture
def alloy_solvers():
    info = xb.install(protocol="test")
    try:
        yield info
    finally:
        xb.uninstall()


def boxes(cls: type, worlds: int = 2) -> newton.Model:
    """``worlds`` copies of a free box 0.5 m above a ground plane."""
    robot = _builder(cls)
    body = robot.add_link(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    robot.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    robot.add_articulation([robot.add_joint_free(body)])
    builder = _builder(cls)
    builder.replicate(robot, worlds)
    builder.add_ground_plane()
    return builder.finalize(device="cpu")


def _builder(cls: type) -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    if hasattr(cls, "register_custom_attributes"):
        cls.register_custom_attributes(b)
    return b


def roll(name: str, steps: int = 30, newton_contacts: bool = False, **kw):
    """Body poses after ``steps`` steps of ``newton.solvers.<name>`` (whatever is installed), and the solver."""
    cls = getattr(newton.solvers, name)
    model = boxes(cls)
    solver = cls(model, **kw)
    s0, s1, control = model.state(), model.state(), model.control()
    pipe = newton.CollisionPipeline(model) if newton_contacts else None
    contacts = pipe.contacts() if pipe is not None else None
    for _ in range(steps):
        if pipe is not None:
            pipe.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, 2e-3)
        s0, s1 = s1, s0
    return s0.body_q.numpy(), solver


def test_install_swaps_every_routed_solver_and_uninstall_restores(alloy_solvers):
    json.dumps(alloy_solvers)
    assert alloy_solvers["engine"] == "alloy" and alloy_solvers["protocol"] == "test"
    for name, stock in STOCK.items():
        cls = getattr(newton.solvers, name)
        assert cls is not stock and issubclass(cls, stock) and cls.__name__ == name, name
        assert name in alloy_solvers["solver_map"] and cls.__module__ == "alloy.xbench.solvers"
        assert ("reset" in vars(cls)) == ("reset" in vars(stock)), name   # newton_newton.py probes vars()
    assert newton.solvers.SolverKamino.Config is STOCK["SolverKamino"].Config
    xb.uninstall()
    assert all(getattr(newton.solvers, n) is c for n, c in STOCK.items()) and xb.installed() == {}


def test_registry_keeps_the_stock_classes(alloy_solvers):
    from alloy.core.registry import capability_for, solver

    assert solver("mujoco").solver_cls is STOCK["SolverMuJoCo"]
    _, s = roll("SolverMuJoCo", steps=1)
    assert capability_for(s).name == "mujoco"


@pytest.mark.parametrize("name,kw,newton_contacts", [
    ("SolverMuJoCo", {}, False),
    ("SolverMuJoCo", {"use_mujoco_contacts": False}, True),
    ("SolverXPBD", {"iterations": 4}, True),
    ("SolverFeatherstone", {}, True),
])
def test_alloy_solver_matches_stock_bitwise(name, kw, newton_contacts):
    q_stock, _ = roll(name, newton_contacts=newton_contacts, **kw)
    xb.install(protocol="test")
    try:
        q_alloy, solver = roll(name, newton_contacts=newton_contacts, **kw)
    finally:
        xb.uninstall()
    assert is_alloy_solver(solver) and type(solver).__name__ == name
    assert np.array_equal(q_alloy, q_stock)
    summary = solver.alloy_ledger.summary()
    assert summary["host_steps"] == 30 and summary["ok"] == 2 and summary["invalid"] == 0


def test_mujoco_internals_are_exposed(alloy_solvers):
    _, s = roll("SolverMuJoCo", steps=2)
    assert s.mjw_model is not None and s.mjw_data is not None and s.mj_model is not None
    assert int(s.mjw_data.nworld) == 2


def test_health_flags_a_nonfinite_world(alloy_solvers):
    cls = newton.solvers.SolverXPBD
    model = boxes(cls)
    solver = cls(model)
    s0, s1, control = model.state(), model.state(), model.control()
    q = s0.body_qd.numpy()
    q[1, 0] = np.nan                       # world 1's box
    s0.body_qd.assign(q)
    solver.step(s0, s1, control, None, 2e-3)
    summary = solver.alloy_ledger.summary()
    assert summary["invalid"] == 1 and summary["ok"] == 1 and "NONFINITE_QD" in summary["flags"]


def test_fast_profile_checks_every_tenth_step(monkeypatch):
    monkeypatch.setenv(xb.PROFILE_ENV, "fast")
    info = xb.install(protocol="test")
    try:
        _, s = roll("SolverMuJoCo", steps=3)
    finally:
        xb.uninstall()
    assert info["profile"] == "fast" and info["health"]["every"] == 10
    # "full" since audit ERR-13 (2026-11-10): "cheap" never measured joints or welds, so JOINT_FAIL could not fire
    # under fast; still every 10th step (one joint + weld measurement per check: +0.35 % of a CPU robot step)
    assert s.alloy_ledger.checks == "full" and s.alloy_ledger.every == 10


def test_engine_shim_runs_a_scene_through_alloy(tmp_path):
    scene = tmp_path / "scene_newton.py"
    scene.write_text(textwrap.dedent("""
        import sys
        import newton
        from newton.solvers import SolverXPBD
        sys.path.insert(0, sys.argv[1])
        from test_xbench_adapter import boxes
        model = boxes(SolverXPBD)
        solver = SolverXPBD(model, iterations=2)
        s0, s1, c = model.state(), model.state(), model.control()
        for _ in range(5):
            solver.step(s0, s1, c, None, 1e-3)
            s0, s1 = s1, s0
        print("SCENE_DONE", type(solver).__module__, flush=True)
    """))
    cmd = [sys.executable, str(SHIM), "--engine", "alloy", "--protocol", "unit", str(scene), "--",
           str(Path(__file__).parent)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=tmp_path,
                         env=_env())
    assert out.returncode == 0, out.stderr[-2000:]
    lines = out.stdout.splitlines()
    engine = json.loads(next(ln for ln in lines if ln.startswith("XB_ENGINE:")).split(":", 1)[1])
    assert engine["engine_under_test"] == "alloy" and "SolverXPBD" in engine["alloy"]["solver_map"]
    assert "SCENE_DONE alloy.xbench.solvers" in out.stdout
    health = json.loads(next(ln for ln in lines if ln.startswith(xb.HEALTH_MARKER)).split(":", 1)[1])
    assert health["protocol"] == "unit" and health["solvers"][0]["host_steps"] == 5
    assert health["solvers"][0]["invalid"] == 0


def _env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env.setdefault("OMP_NUM_THREADS", "4")
    return env


def test_install_does_not_initialise_warp():
    """GPUSimBench reads its NVML memory baseline before any CUDA context: install must not run wp.init."""
    code = ("import warp._src.context as c, alloy.xbench as xb; xb.install(protocol='gpusimbench'); "
            "print('RUNTIME', c.runtime is not None)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, env=_env())
    assert out.returncode == 0, out.stderr[-2000:]
    assert "RUNTIME False" in out.stdout


def test_child_of_an_alloy_scene_runs_alloy(tmp_path):
    """A scene that spawns child interpreters (newton_newton.py, one per benchmark) keeps the engine under test."""
    scene = tmp_path / "parent_newton.py"
    scene.write_text(textwrap.dedent("""
        import subprocess, sys
        from engine_shim import child_command
        if len(sys.argv) > 1 and sys.argv[1] == "child":
            import newton
            print("CHILD_SOLVER", newton.solvers.SolverXPBD.__module__, flush=True)
        else:
            sys.exit(subprocess.call(child_command(__file__, ["child"])))
    """))
    env = _env()
    env["PYTHONPATH"] = f"{REPO}:{SHIM.parent}"
    out = subprocess.run([sys.executable, str(SHIM), "--engine", "alloy", "--protocol", "unit", str(scene)],
                         capture_output=True, text=True, timeout=600, cwd=tmp_path, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "CHILD_SOLVER alloy.xbench.solvers" in out.stdout
    assert out.stdout.count("XB_ENGINE:") == 2      # the child went through the shim again
