# SPDX-License-Identifier: Apache-2.0
"""Gap #10b P2 S2a, the controls law (WP4's review of S2a item 3, WP12's "controls have no law"): with
``alloy.build(scene, clock_worlds=n)`` a resident world's inputs reach its fine slot only through
``WorldTransfer.copy_controls`` (alloy/coupling/fine_pack.py), every period, by name prefix ("control:" /
"extra:control:"). The resident gets exactly its own rows, and no other world's.

Rig: test_world_clock_run.py's (4 worlds, a 3 cm box on a sheet on a table, the box's density drawn per world, so one
world resides from the first period) plus a two-hinge arm beside the table, clear of everything: a position servo on
the shoulder driven by ``set_targets`` (a per-world ramp) and a motor on the elbow driven by ``set_controls`` (a
per-world torque). Every world gets different values. CPU, MuJoCo Warp + VBD."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import alloy
from alloy.core.snapshot import _state_arrays
from alloy.coupling.fine_pack import WorldTransfer, _control_world

W, SLOTS, SEED, PERIODS = 4, 1, 0, 5
LIGHT, HEAVY = 500.0, 8000.0
TARGET = 0.1 * (1.0 + np.arange(W))       # [rad] each world's shoulder target at the last period (ramped up to it)
TORQUE = 0.002 * (1.0 + np.arange(W))     # [N m] each world's elbow motor torque
NAMES = ["box", "cloth", "arm/shoulder", "arm/elbow"]
ARM = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="upper" pos="0 0 0"><joint name="shoulder" type="hinge" axis="0 0 1" range="-2 2" damping="0.5"/>
    <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.01" mass="0.2"/>
    <body name="fore" pos="0.12 0 0"><joint name="elbow" type="hinge" axis="0 0 1" range="-1.5 1.5" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.1"/></body></body></worldbody>
  <actuator><position name="servo" joint="shoulder" kp="20" kv="1"/>
    <motor name="motor" joint="elbow" gear="1" ctrlrange="-1 1"/></actuator></mujoco>"""


def _scene() -> alloy.Scene:
    side = 0.03
    scene = alloy.Scene(worlds=W, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.4, 0.4, 0.1), pos=(0, 0, 0.05), static=True, friction=0.5)
    scene.add_cloth("cloth", size=(0.2, 0.2), pos=(0, 0, 0.102), resolution=(9, 9), mass=0.03, thickness=0.004,
                    self_contact=False)
    scene.add_box("box", size=(side,) * 3, pos=(0, 0, 0.104 + 0.001 + side / 2), density=HEAVY)
    scene.randomize("box", "density", LIGHT, HEAVY)
    scene.add_robot("arm", mjcf=ARM, pos=(0.6, 0.0, 0.3))      # reaches x >= 0.38: clear of the table (x <= 0.2)
    return scene


def _commands(p: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Period ``p``'s per-world commands: shoulder targets (a ramp to TARGET) and elbow torques."""
    return {"arm/shoulder": TARGET * (p + 1) / PERIODS}, {"arm/motor": TORQUE}


def _ctrl(sim: Any) -> np.ndarray:
    """Every Control array's rows of each world side by side, (worlds, values): the servo's joint target and the
    motor's MuJoCo ``ctrl`` among them (the item -> world maps copy_controls moves rows by)."""
    eng, cols = sim.engine, []
    for _, path, arr in _state_arrays(eng.control):
        world = _control_world(eng.model, path, arr)
        if world is None or not arr.shape[0]:
            continue
        w, vals = world.numpy(), arr.numpy().reshape(arr.shape[0], -1).astype(np.float64)
        cols.append(np.stack([vals[w == i].reshape(-1) for i in range(int(sim.worlds))]))
    return np.concatenate(cols, axis=1)


def _run(sim: Any, by_hand: int | None = None) -> list[dict[str, Any]]:
    """PERIODS periods, the commands given to every world each period. ``by_hand``: also give world ``by_hand``'s own
    commands to the fine pack's slot 0 directly (the reference run, with copy_controls switched off by the caller).
    Per period: the state, the residents, the coarse level and both packs' ctrl rows."""
    clock = getattr(sim.engine, "world_clock", None)
    rows = []
    for p in range(PERIODS):
        targets, controls = _commands(p)
        sim.set_targets(targets)
        sim.set_controls(controls)
        if by_hand is not None:
            clock.fine.set_targets({k: float(v[by_hand]) for k, v in targets.items()})
            clock.fine.set_controls({k: float(v[by_hand]) for k, v in controls.items()})
        sim.step(1)
        rows.append({"state": {k: np.array(v, copy=True) for k, v in sim.state(NAMES).items() if k != "t"},
                     "level": int(round(np.log2(sim.dt / float(sim.engine._last_coupling.dt_x)))),
                     "resident": None if clock is None else clock.resident.copy(),
                     "ctrl": _ctrl(sim), "fine_ctrl": None if clock is None else _ctrl(clock.fine),
                     "ok": sim.check()["ok"]})
    return rows


@pytest.fixture(scope="module")
def on() -> Any:
    """The run with the option: (rows, the resident world)."""
    sim = alloy.build(_scene(), device="cpu", seed=SEED, clock_worlds=SLOTS)
    try:
        rows = _run(sim)
    finally:
        sim.close()
    k = np.flatnonzero(rows[0]["resident"])
    assert k.size == 1, rows[0]["resident"]
    return rows, int(k[0])


@pytest.mark.law
@pytest.mark.timeout(1800)
def test_the_resident_gets_its_own_control_rows_and_no_other_worlds(on: Any) -> None:
    """(b) Every period the fine slot's ctrl row is the resident's coarse row bit for bit (the new ramp value each
    period), never another world's (every world's commands differ); the resident's joints follow its own commands
    (the shoulder angles keep the targets' order across all worlds, the resident's included)."""
    rows, k = on
    for p, row in enumerate(rows):
        assert row["resident"].tolist() == [w == k for w in range(W)], p
        np.testing.assert_array_equal(row["fine_ctrl"][0], row["ctrl"][k], err_msg=f"period {p}")
        for j in range(W):
            if j != k:
                assert not np.array_equal(row["fine_ctrl"][0], row["ctrl"][j]), (p, j, row["fine_ctrl"][0])
        if p:
            assert np.any(row["fine_ctrl"][0] != rows[p - 1]["fine_ctrl"][0]), p    # the ramp moved this period
        assert row["ok"], p
    q = rows[-1]["state"]["arm/shoulder.q"]
    assert np.all(np.diff(q) > 0.0), q                     # TARGET rises with the world index


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_the_resident_follows_bit_for_bit_as_with_its_slot_driven_by_hand(on: Any, monkeypatch: Any) -> None:
    """(a) The same run with copy_controls switched off and the fine slot given the resident's commands by hand
    (``fine.set_targets`` / ``set_controls``) is the run with the option, bit for bit, every world, every period."""
    rows, k = on
    monkeypatch.setattr(WorldTransfer, "copy_controls", lambda self, e, s: None)
    sim = alloy.build(_scene(), device="cpu", seed=SEED, clock_worlds=SLOTS)
    try:
        ref = _run(sim, by_hand=k)
    finally:
        sim.close()
    for p, (a, b) in enumerate(zip(rows, ref, strict=True)):
        assert b["resident"].tolist() == a["resident"].tolist() and b["level"] == a["level"], p
        np.testing.assert_array_equal(b["fine_ctrl"], a["fine_ctrl"], err_msg=f"period {p}")
        for key, v in a["state"].items():
            np.testing.assert_array_equal(b["state"][key], v, err_msg=f"period {p}: {key}")


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_the_coarse_worlds_take_their_commands_as_with_that_slot_parked(on: Any) -> None:
    """F1 with commands: the worlds left in the coarse pack are bit for bit a one-pack run (no option) with the
    resident's slot parked by hand from the first period and the same per-world commands; so is the coarse level."""
    rows, k = on
    sim = alloy.build(_scene(), device="cpu", seed=SEED)
    try:
        sim.engine.stepper.park_worlds([k], sim.engine.state)
        ref = _run(sim)
    finally:
        sim.close()
    rest = [w for w in range(W) if w != k]
    for p, (a, b) in enumerate(zip(rows, ref, strict=True)):
        assert b["level"] == a["level"], p
        for key, v in a["state"].items():
            np.testing.assert_array_equal(b["state"][key][rest], v[rest], err_msg=f"period {p}: {key}")
