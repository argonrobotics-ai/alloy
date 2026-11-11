# SPDX-License-Identifier: Apache-2.0
"""``sim.reset()`` puts the coupled exchange clock back (alloy/coupling/clock_memory.py).

Through the public API: a fresh run, the run after a reset and the run after a second reset agree bit for bit, every
period, in every state array and in the clock (dt_x and the entries' substeps). Each scene must also make the clock
adapt, or the law proves nothing (``_adapts``).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.coupling.clock_memory import capture, restore
from alloy.coupling.contact_gate import gate_for

pytestmark = pytest.mark.timeout(600)       # 18-70 s each on a loaded 4-core host, 20-220 s under 14 workers

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.3">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


def _box_on_cloth(drop: float = 0.05, worlds: int = 2) -> alloy.Scene:
    """A 1 kg block ``drop`` m over a sheet on a table."""
    sc = alloy.Scene(worlds=worlds, dt=0.01)
    sc.add_ground()
    sc.add_box("table", size=(0.3, 0.3, 0.1), pos=(0, 0, 0.05), static=True)
    sc.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, 0.13), resolution=6, mass=0.02)
    sc.add_box("block", size=(0.04, 0.04, 0.04), pos=(0, 0, 0.15 + drop), mass=1.0)
    return sc


def _cloth_liquid() -> alloy.Scene:
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_open_box("jug", inner=(0.05, 0.05, 0.05), wall=0.004, pos=(0, 0, 0.1), static=True)
    sc.add_liquid("water", size=(0.044, 0.044, 0.016), pos=(0, 0, 0.113))
    sc.add_box("tray", size=(0.2, 0.2, 0.01), pos=(0.4, 0, 0.2), static=True)
    sc.add_cloth("sheet", size=(0.1, 0.1), pos=(0.4, 0, 0.215), resolution=6, mass=0.01)
    return sc


def _arm_beside_cloth() -> alloy.Scene:
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_box("table", size=(0.3, 0.3, 0.1), pos=(0.3, 0, 0.05), static=True)
    sc.add_cloth("sheet", size=(0.12, 0.12), pos=(0.3, 0, 0.115), resolution=5, mass=0.02)
    sc.add_robot("arm", mjcf=ARM, pos=(0, 0, 0))
    return sc


def _trace(sim: alloy.Sim, periods: int, targets: dict | None = None) -> list[tuple]:
    """Per period: (dt_x, substeps, {state key: bytes})."""
    st, out = sim.engine.stepper, []
    if targets:
        sim.set_targets(targets)
    for _ in range(periods):
        sim.step(1)
        out.append((st._dt_x, tuple(sorted(st._substeps_now.items())),
                    {k: np.asarray(v).tobytes() for k, v in sim.state().items() if k != "t"}))   # t keeps counting
    return out


def _first_difference(a: list[tuple], b: list[tuple]) -> str | None:
    for p, (x, y) in enumerate(zip(a, b, strict=True)):
        if x[:2] != y[:2]:
            return f"period {p}: clock {x[:2]} vs {y[:2]}"
        bad = [k for k in x[2] if x[2][k] != y[2][k]]
        if bad:
            return f"period {p}: state {bad}"
    return None


def _adapts(trace: list[tuple]) -> bool:
    return len({t[:2] for t in trace}) > 1


def _plain(x: object) -> object:
    """A ``ClockMemory`` (or any part of one) as plain data, to compare two of them exactly."""
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(str(v) for v in x)
    return _plain(vars(x)) if hasattr(x, "__dict__") else x


def _law(scene: alloy.Scene, periods: int, targets: dict | None = None) -> list[tuple]:
    """fresh == after one reset == after a second reset, and the clock's whole host state after a reset is the built
    simulation's (as period 0 finds it: one ``choose_clock`` made, the wiring's flags set); returns the fresh trace."""
    sim = alloy.build(scene, device="cpu")
    built = _plain(capture(sim.engine.stepper))
    fresh = _trace(sim, periods, targets)
    for label in ("one reset", "two resets"):
        sim.reset()
        assert _plain(capture(sim.engine.stepper)) == built, label
        again = _trace(sim, periods, targets)
        assert _first_difference(fresh, again) is None, f"{label}: {_first_difference(fresh, again)}"
    return fresh


def _law_moved(scene: alloy.Scene, periods: int, targets: dict | None = None) -> None:
    """For a scene whose clock does not move by itself in a few periods: the law moves it, the sticky escalation floor
    one level up, as a watchdog trip does. The next period steps at half the dt_x; a reset must put the floor back:
    the clock's host state is the built one's and the run after it (and after a second reset) is the fresh run."""
    sim = alloy.build(scene, device="cpu")
    st = sim.engine.stepper
    built = _plain(capture(st))
    fresh = _trace(sim, periods, targets)
    assert st._clock.escalate()
    moved = _trace(sim, 1)
    assert moved[0][0] == pytest.approx(0.5 * fresh[-1][0], rel=1e-12)      # the moved clock is really in use
    for label in ("one reset", "two resets"):
        sim.reset()
        assert _plain(capture(st)) == built, label
        again = _trace(sim, periods, targets)
        assert _first_difference(fresh, again) is None, f"{label}: {_first_difference(fresh, again)}"


def test_cloth_and_rigid_box_repeat_after_reset_from_a_moved_clock() -> None:
    _law_moved(_box_on_cloth(drop=0.0005, worlds=1), 2)     # the block is on the sheet within the first period


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_cloth_and_rigid_box_from_5_cm_repeat_after_reset() -> None:
    assert _adapts(_law(_box_on_cloth(), 10))               # the clock moves at the landing; 235 s on a loaded host


def test_cloth_and_liquid_repeat_after_reset() -> None:
    assert _adapts(_law(_cloth_liquid(), 6))


def test_robot_beside_cloth_repeats_after_reset_from_a_moved_clock() -> None:
    _law_moved(_arm_beside_cloth(), 3, {"shoulder": 0.6})   # dt_x 3.1e-4, rungs 1/1 over 8 periods by itself


def test_settle_makes_the_settled_clock_the_reset_target() -> None:
    sim = alloy.build(_cloth_liquid(), device="cpu")
    sim.settle(0.02)
    a = _trace(sim, 4)
    sim.reset()
    assert _first_difference(a, _trace(sim, 4)) is None


def test_masked_reset_keeps_the_shared_clock_and_puts_the_worlds_rows_back() -> None:
    sim = alloy.build(_box_on_cloth(), device="cpu")
    st = sim.engine.stepper
    sim.step(6)
    st._onset.last = np.full(2, 5, dtype=np.int64)       # rows are sized lazily: set them, world 1's must survive
    st._ladder.last, st._ladder.per_world = np.full(2, 5, dtype=np.int64), np.full(2, 3, dtype=np.int64)
    shared = (st._clock.j, st._clock.j_floor, dict(st._substeps_now), st._band_bound, st._dt_x)
    sim.engine.reset(wp.array([1, 0], dtype=wp.int32, device=sim.device))
    assert st._onset.last.tolist()[1] == 5 and st._ladder.last.tolist()[1] == 5          # world 1 kept running
    assert st._ladder.per_world.tolist() == [0, 3]                             # world 0's count is the start's
    assert shared == (st._clock.j, st._clock.j_floor, dict(st._substeps_now), st._band_bound, st._dt_x)


def _gate_on(sim: alloy.Sim) -> object:
    """The contact gate (``ClockPolicy.contact_gate``, opt-in) on a built simulation, before its first period."""
    st = sim.engine.stepper
    st._alloy_policy = dataclasses.replace(st._alloy_policy,
                                           clock=dataclasses.replace(st._alloy_policy.clock, contact_gate=True))
    st._gate = gate_for(st)
    return st._gate


def test_with_the_contact_gate_a_reset_puts_its_rows_back() -> None:
    """The box starts 5 cm over the sheet, so the gate lets its pair go after the hold: the gate's rows at the end of
    a run are not a fresh build's. fresh == after a reset with the gate on; then the rows themselves, by world."""
    sim = alloy.build(_box_on_cloth(), device="cpu")
    st, gate = sim.engine.stepper, _gate_on(sim)
    start = (gate.armed_host.copy(), gate.hold.copy())
    fresh = _trace(sim, 4)
    assert not gate.armed_host.all()                                        # the run changed the gate
    sim.reset()
    assert (gate.armed_host == start[0]).all() and (gate.hold == start[1]).all()
    assert _first_difference(fresh, _trace(sim, 4)) is None
    memory, ptr = capture(st), gate.armed.ptr
    saved = {n: getattr(gate, n).numpy().copy() for n in ("armed", "_vmax", "_count")}
    gate.hold[:] = -7
    gate.armed.fill_(5)
    gate._vmax.fill_(9.0)
    gate._count.fill_(9)
    ends = np.asarray(gate.ends.world)
    restore(st, memory, np.array([True, False]))                            # world 0 only
    assert (gate.hold[ends == 0] == memory.gate["hold"][ends == 0]).all() and (gate.hold[ends == 1] == -7).all()
    armed, vmax = gate.armed.numpy(), gate._vmax.numpy()
    assert (armed[ends == 0] == saved["armed"][ends == 0]).all() and (armed[ends == 1] == 5).all()
    assert (vmax[:, 0] == saved["_vmax"][:, 0]).all() and (vmax[:, 1] == 9.0).all()
    assert (gate._count.numpy()[:, 1] == 9).all()
    restore(st, memory, None)                                               # every world, in place
    assert gate.armed.ptr == ptr and (gate.hold == memory.gate["hold"]).all()
    assert all((getattr(gate, n).numpy() == saved[n]).all() for n in saved)
