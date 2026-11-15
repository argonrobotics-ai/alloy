# SPDX-License-Identifier: Apache-2.0
"""The one-pass free-body guard (alloy.contact.mjw_free_gyro; physics order 2026-11-14 after the chain-cost profile).

Laws:
- contract: MuJoCo Warp calls the launcher by module name (forward's step, inverse), and the two replaced functions
  still hash as pinned, by the same hash as ``alloy.precision.mjw_pin``;
- order: in a fresh process, ``pin_ok`` and ``install`` give the same answer, with no warning, whether
  ``alloy.precision`` (whose import installs the replacement) is imported before them, after them or not at all;
- the guard: on the same state, the replacement and MuJoCo Warp's own kernel write the same bodies and the same
  values, bit for bit (``out`` pre-filled with NaN: a skipped body keeps it), for the forward and the inverse kernel,
  over random scenes with a chain cable (the DISCRETE integrator), clutter at rest and in flight, an actuated arm,
  contacts and a weld's equality rows; both outcomes occur in each; CPU Warp, and cuda;
- qacc: whole runs on the CPU Warp route are bit for bit the same with the replacement and without.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

import alloy
from alloy._vendor.mujoco_warp._src import forward
from alloy._vendor.mujoco_warp._src import inverse
from alloy._vendor.mujoco_warp._src.types import IntegratorType
from alloy.contact import mjw_free_gyro as FG

pytestmark = pytest.mark.timeout(1800)
ARM = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="b1" pos="0 0 0.05"><joint name="j1" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="5"/>
  <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.015" mass="0.5"/></body></worldbody>
  <actuator><position joint="j1" kp="100" kv="10"/></actuator></mujoco>"""
WELDED = """<mujoco><worldbody>
  <body name="p" pos="0.6 0.4 0.3"><freejoint/><geom type="box" size="0.03 0.03 0.03" mass="0.3"/></body>
  <body name="q" pos="0.68 0.4 0.3"><freejoint/><geom type="box" size="0.03 0.03 0.03" mass="0.3"/></body>
  </worldbody><equality><weld body1="p" body2="q"/></equality></mujoco>"""


def test_mujoco_warp_reaches_the_launcher_through_the_module():
    assert inspect.getsource(forward).count("_launch_discrete_free_gyro(m, d") >= 2          # the step paths
    assert "forward._launch_discrete_free_gyro" in inspect.getsource(inverse)
    assert FG.pin_ok(), "mujoco_warp moved: re-copy _discrete_free_gyro and update PINNED"
    from alloy.precision import mjw_pin

    assert FG._source_hash(FG.stock()) == FG.PINNED["forward._launch_discrete_free_gyro"]
    assert mjw_pin.source_hash("forward._discrete_free_gyro_kernel") == FG.PINNED["forward._discrete_free_gyro_kernel"]
    was = forward._launch_discrete_free_gyro is FG.launch
    try:
        FG.uninstall()
        assert mjw_pin.source_hash("forward._launch_discrete_free_gyro") == FG._source_hash(FG.stock())
    finally:
        if was:
            FG.install()


ORDER = '''import json, sys
from alloy._vendor.mujoco_warp._src import forward
first = sys.argv[1] == "precision_first"
if first:
    import alloy.precision
from alloy.contact import mjw_free_gyro as FG
before = "alloy.precision" in sys.modules
ok, inst = FG.pin_ok(), FG.install()
if sys.argv[1] == "precision_after":
    import alloy.precision
print(json.dumps({"precision_before": before, "pin_ok": ok, "install": inst, "pin_ok_again": FG.pin_ok(),
                  "replaced": forward._launch_discrete_free_gyro is FG.launch,
                  "stock_kept": FG.stock() is not FG.launch}))
'''


@pytest.mark.parametrize("order", ["precision_first", "precision_after", "no_precision"])
def test_the_pin_and_install_do_not_depend_on_import_order(order):
    env = {**os.environ, "PYTHONPATH": str(Path(alloy.__file__).resolve().parents[1])}
    env.pop("ALLOY_MJW_FREE_GYRO", None)
    p = subprocess.run([sys.executable, "-c", ORDER, order], env=env, capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stderr[-2000:]
    row = json.loads(p.stdout.strip().splitlines()[-1])
    assert row == {"precision_before": order == "precision_first", "pin_ok": True, "install": True,
                   "pin_ok_again": True, "replaced": True, "stock_kept": True}, row
    assert "differs" not in p.stderr, p.stderr[-2000:]


def _scene(seed: int, worlds: int) -> alloy.Scene:
    """A clamped chain (the DISCRETE integrator), an actuated arm, boxes and balls resting or dropped per world, and
    two welded boxes, placed at random (``seed``)."""
    rng = np.random.default_rng(seed)
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM, pos=(-0.45, 0.0, 0.0))
    s.add_robot("pair", mjcf=WELDED)
    s.add_rod("cable", length=0.2, radius=0.004, pos=(-0.45, 0.25, 0.02), direction=(1, 0, 0), segments=6,
              clamp="start", route="chain")
    for i in range(12):
        x, y = rng.uniform(-0.3, 0.3, size=2)
        z = np.where(rng.random(worlds) < 0.5, 0.0, rng.uniform(0.05, 0.4, size=worlds))   # resting or in flight
        pos = np.column_stack([np.full(worlds, x), np.full(worlds, y), 0.03 + 1e-4 + z])
        if i % 2:
            s.add_box(f"b{i}", size=(0.04,) * 3, pos=pos, mass=0.1)
        else:
            s.add_sphere(f"s{i}", 0.02, pos=pos, mass=0.1)
    return s


def _solver(sim):
    st = sim.engine.stepper
    return next(s for s in (getattr(e, "solver", e) for e in (getattr(st, "entries", None) or [st]))
                if hasattr(s, "mjw_model"))


def _compare(device: str, seed: int, monkeypatch) -> tuple[int, int]:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    sim = alloy.build(_scene(seed, 4), device=device, seed=seed)
    try:
        sim.set_targets({"j1": 1.0})
        sim.step(3)
        solver = _solver(sim)
        m, d = solver.mjw_model, solver.mjw_data
        assert int(m.opt.integrator) == int(IntegratorType.DISCRETE)
        written = skipped = 0
        for is_inverse in (False, True):
            outs = []
            for fn in (FG.stock(), FG.launch):
                out = wp.full(d.qacc.shape, np.nan, dtype=float, device=d.qacc.device)
                with wp.ScopedDevice(d.qacc.device):    # MuJoCo Warp's launches name no device: the solver scopes it
                    fn(m, d, out, d.qacc, is_inverse)
                outs.append(out.numpy())
            a, b = outs
            assert np.array_equal(np.isnan(a), np.isnan(b)), (seed, is_inverse)
            assert np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)]), (seed, is_inverse)
            free = np.asarray(m.body_dofadr.numpy())[np.asarray(m.body_freeadr.numpy())]
            done = ~np.isnan(a[:, free])
            written, skipped = written + int(done.sum()), skipped + int((~done).sum())
        assert sim.check()["ok"], sim.check()["problems"]
        return written, skipped
    finally:
        sim.close()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_guard_skips_and_solves_the_same_bodies_bit_for_bit(seed, monkeypatch):
    written, skipped = _compare("cpu", seed, monkeypatch)
    assert written > 0 and skipped > 0, (written, skipped)


@pytest.mark.gpu
@pytest.mark.parametrize("seed", [0, 1])
def test_the_guard_skips_and_solves_the_same_bodies_bit_for_bit_on_cuda(seed, monkeypatch):
    written, skipped = _compare("cuda:0", seed, monkeypatch)
    assert written > 0 and skipped > 0, (written, skipped)


def _run_digest(seed: int) -> str:
    sim = alloy.build(_scene(seed, 2), device="cpu", seed=seed)
    try:
        sim.set_targets({"j1": 1.0})
        h = hashlib.sha256()
        for _ in range(40):
            sim.step(1)
            st = sim.state()
            for k in sorted(st):
                h.update(np.ascontiguousarray(st[k]).tobytes())
        assert sim.check()["ok"], sim.check()["problems"]
        return h.hexdigest()
    finally:
        sim.close()


def test_runs_are_bit_for_bit_with_and_without_the_replacement(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    was = forward._launch_discrete_free_gyro is FG.launch
    try:
        FG.uninstall()
        stock = _run_digest(5)
        assert FG.install() and forward._launch_discrete_free_gyro is FG.launch
        assert _run_digest(5) == stock
    finally:
        if not was:
            FG.uninstall()
