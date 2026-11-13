# SPDX-License-Identifier: Apache-2.0
"""Sample-sweep F2 (WP11): a health-latched world leaves a coupled scene's exchange clock (opt-in
``ALLOY_PARK_LATCHED``; known gap coupling-frozen-world-sets-clock). CPU.

- ``parked.band_over``: the band bound for the next clock, taken again over the worlds left when one is parked
  between periods (before, the bound read with the old set held the next period at the parked world's need).
- ``park_latched``: what it parks (latched now, once), and when it leaves the stepper alone.
- The clock after a park: the substep rungs go straight to what the worlds left want (``jump_next(rungs=True)``).
- The law (slow, about 5 + 6 min): two worlds of a sheet over a small open box and a free 1 kg ball; world 0's sheet
  starts 5 mm inside the box walls and freezes at the first check. Without the option the frozen world, restored
  every exchange, holds the clock at its finest step; with it the healthy world steps on its own clock."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.coupling import contact_band as CB
from alloy.coupling.parked import band_over
from alloy.integration import park_latched as PL
from alloy.precision.types import HealthFlags

SPEEDS = (1.0, 9.0)                      # m/s: world 0's contact, world 1's
GAP = 0.003                              # m: the pipeline's detection band


def _band(rows: bool) -> CB.ContactBandBound:
    """Two worlds' vertex records read once (world 0's particle at 1 m/s, world 1's at 9), with or without rows."""
    contacts = SimpleNamespace(
        soft_contact_max=2, soft_contact_count=wp.array([2], dtype=wp.int32, device="cpu"),
        soft_contact_indices=wp.array(np.array([(0, -1, -1), (1, -1, -1)], np.int32), dtype=wp.vec3i, device="cpu"),
        soft_contact_body_vel=wp.zeros(2, dtype=wp.vec3, device="cpu"))
    view = SimpleNamespace(particle_world=wp.array([0, 1], dtype=wp.int32, device="cpu"))
    factory = SimpleNamespace(built=[(view, SimpleNamespace(soft_contact_gap=GAP), contacts, None)])
    model = SimpleNamespace(particle_mass=wp.array(np.zeros(2, np.float32), dtype=float, device="cpu"))
    qd = wp.array(np.array([(s, 0, 0) for s in SPEEDS], np.float32), dtype=wp.vec3, device="cpu")
    entries = {"cloth": SimpleNamespace(state_1=SimpleNamespace(particle_qd=qd), solver=SimpleNamespace(model=model))}
    band = CB.ContactBandBound([(factory, "cloth")], "cpu", 2)
    band.keep_world_rows(rows)
    band.ensure_worlds(2)
    band.accumulate(entries)
    band.end_period()
    return band


def test_a_world_parked_between_periods_leaves_the_band_bound() -> None:
    """The shared bound read at the period's end is world 1's (9 m/s); parked now, the next clock takes world 0's
    (1 m/s, 9x coarser) from the rows. Unparked, or with no rows, it is the shared value unchanged."""
    band = _band(rows=True)
    shared = CB.band_bound(GAP, SPEEDS[1], band.last_travel)
    assert band.world_bounds(need=False)[1] == pytest.approx(shared, rel=1e-6)
    alone = band_over(band, np.array([False, True]), shared)
    assert alone == pytest.approx(CB.band_bound(GAP, SPEEDS[0], band.last_travel), rel=1e-6)
    assert alone == pytest.approx(9.0 * shared, rel=1e-5)
    assert band_over(band, None, shared) == pytest.approx(shared, rel=1e-6)
    assert band_over(band, np.array([True, True]), shared) == math.inf
    assert band_over(_band(rows=False), np.array([False, True]), shared) == shared


CONTROL = 0.01
PAIR = CONTROL / 128                     # s: the ball's interface bound on the grid (78 us)


@pytest.mark.parametrize("rungs", [False, True], ids=["plain", "rungs"])
def test_a_park_drops_the_rung_a_parked_world_raised(rungs: bool) -> None:
    """VBD's rung is at 4 for a frozen world's stable step (50 us at dt_x 78 us, safety 0.5). The park re-takes the
    stable step over the others (1 ms): ``jump_next(rungs=True)`` takes the next period to rung 1 at once (the L4's
    4 worlds: 10.1 s a period at rung 4 against 2.5 s at rung 1); plain ``jump_next`` waits a quiet stretch per rung."""
    from alloy.coupling.clocks import ClockController  # noqa: PLC0415
    from alloy.coupling.types import ClockPolicy  # noqa: PLC0415

    c = ClockController(ClockPolicy(), {"mujoco": 1, "vbd": 1}, {"mujoco": True, "vbd": True})
    c.safety_for = {"mujoco": 0.5, "vbd": 0.5}
    for _ in range(2):
        assert c.decide(CONTROL, PAIR, {"mujoco": 1.0e-3, "vbd": 50.0e-6}).substeps["vbd"] == 4
    c.jump_next(rungs=rungs)
    after = [c.decide(CONTROL, PAIR, {"mujoco": 1.0e-3, "vbd": 1.0e-3}) for _ in range(3)]
    assert [d.dt_x for d in after] == [pytest.approx(PAIR)] * 3
    assert [d.substeps["vbd"] for d in after] == ([1, 1, 1] if rungs else [4, 4, 4])


@pytest.mark.parametrize("follow", [False, True], ids=["driver", "latch"])
def test_only_a_latch_park_retakes_the_band_bound_and_the_rungs(monkeypatch, follow: bool) -> None:
    """A park without ``follow`` (``clock_worlds=``'s driver, every caller but ``park_latched``) keeps the old clock
    rule: the level jumps, the rungs keep their hysteresis, the band bound read with the old set stands (AUDIT-STACK,
    2026-11-11: the off path is the old code by construction). With ``follow`` both are re-taken. A real Scheduler
    over fake solvers (``alloy.testing.coupling_parked``)."""
    import alloy.coupling.parked as PK  # noqa: PLC0415
    from alloy.testing.coupling import PeriodRunner  # noqa: PLC0415
    from alloy.testing.coupling_parked import fake_mass_model, fake_mass_scheduler  # noqa: PLC0415

    model = fake_mass_model((0.2, 0.2, 0.2, 0.002), (0.0,) * 4, (0.5,) * 4)
    sched = fake_mass_scheduler(model)
    runner = PeriodRunner(model, sched, CONTROL)
    runner.run_period()
    jumps: list[bool] = []
    retaken: list[list[bool]] = []
    monkeypatch.setattr(sched._clock, "jump_next", lambda rungs=False: jumps.append(rungs))
    monkeypatch.setattr(PK, "band_over", lambda band, mask, bound: retaken.append(mask.tolist()) or bound)
    sched.park_worlds([3], runner.state, follow=follow)
    assert jumps == [follow]
    assert retaken == ([[False, False, False, True]] if follow else [])
    sched.unpark_worlds([3])
    assert jumps == [follow, False] and len(retaken) == int(follow)


class _Stepper:
    def __init__(self, park: bool = True, worlds: int = 4) -> None:
        self.parked = np.zeros(worlds, dtype=bool)
        self.calls: list[list[int]] = []
        self.rows: list[bool] = []
        if park:
            self.park_worlds = self._park
            self.keep_world_rows = lambda on=True: self.rows.append(bool(on))

    def _park(self, worlds, state, *, follow: bool = False) -> None:
        assert follow                    # park_latched's parks take the band bound and rungs over the worlds left
        self.calls.append(np.asarray(worlds).tolist())
        self.parked[np.asarray(worlds)] = True


class _Record:
    """A health record's host read: ``flags`` per world, LATCHED on the given worlds (and another bit on world 0)."""
    def __init__(self, latched: list[int], worlds: int = 4) -> None:
        self.flags = np.full(worlds, int(HealthFlags.PEN_WARN), dtype=np.uint32)
        self.flags[latched] |= np.uint32(int(HealthFlags.LATCHED))

    def numpy(self) -> np.ndarray:
        rec = np.zeros(self.flags.size, dtype=[("flags", np.uint32), ("first_bad_step", np.int32)])
        rec["flags"] = self.flags
        return rec


def _sim(latched: list[int], **kw) -> SimpleNamespace:
    return SimpleNamespace(stepper=_Stepper(**kw), state=object(), world_clock=None,
                           latch_log=SimpleNamespace(record=_Record(latched)))


@pytest.mark.parametrize("env", ["", "0", "1"])
def test_park_latched_parks_each_latched_world_once_when_asked(monkeypatch, env: str) -> None:
    """Worlds 1 and 3 latched: parked at the first call (from the record's flags as they are now, not a period-old
    copy), not again at the next. Off: nothing is parked and no band rows are asked for."""
    monkeypatch.setenv(PL.ENV, env)
    sim = _sim([3, 1])
    PL.prepare(sim)
    on = env == "1"
    assert PL.park_latched(sim) == ([1, 3] if on else [])
    assert PL.park_latched(sim) == []
    sim.latch_log.record.flags[2] |= np.uint32(int(HealthFlags.LATCHED))
    assert PL.park_latched(sim) == ([2] if on else [])
    assert sim.stepper.calls == ([[1, 3], [2]] if on else [])
    assert sim.stepper.rows == ([True] if on else [])


def test_park_latched_leaves_clock_worlds_and_plain_steppers_alone(monkeypatch) -> None:
    monkeypatch.setenv(PL.ENV, "1")
    driven = _sim([0])
    driven.world_clock = object()
    plain = _sim([0], park=False)
    assert PL.park_latched(driven) == [] and driven.stepper.calls == []
    PL.prepare(plain)
    assert PL.park_latched(plain) == []


# ------------------------------------------------------------------------------------------------ the law (slow)

def _cup_scene(monkeypatch, on: bool, periods: int) -> tuple[list[tuple[int, list[bool]]], float, list[bool]]:
    """Per period: the physics steps (exchanges) and which worlds are INVALID; the exchange bound [s] the clock
    read after period 0 (the frozen world's or the healthy one's); and the parked worlds before and after a
    ``sim.reset()`` at the end. The option ``on`` or off."""
    import alloy  # noqa: PLC0415 - heavy

    monkeypatch.setenv(PL.ENV, "1" if on else "0")
    sc = alloy.Scene(dt=0.01, worlds=2)
    sc.add_ground()
    sc.add_open_box("cup", inner=(0.06, 0.06, 0.06), wall=0.008)                  # rim at z 0.068
    sc.add_cloth("sheet", size=(0.09, 0.09), pos=np.array([[0, 0, 0.063], [0, 0, 0.075]]), resolution=8,
                 mass=0.005, self_contact=False)
    sc.add_sphere("ball", 0.03, pos=(0, 0, 0.4), mass=1.0)
    sim = alloy.build(sc, device="cpu", seed=0)
    out, bound = [], math.nan
    for p in range(periods):
        p0 = sim.physics_steps
        sim.step(1)
        out.append((sim.physics_steps - p0, [w["severity"] == "INVALID" for w in sim.health()["worlds"]]))
        if p == 0:
            bound = sim.engine.stepper._interface_bound()
    parked = [bool(sim.engine.stepper.parked.any())]
    sim.reset()
    return out, bound, parked + [bool(sim.engine.stepper.parked.any())]


FINEST = 0.01 / 1024                     # s: the clock's finest exchange step at dt 10 ms


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
def test_a_frozen_world_leaves_the_healthy_worlds_clock(monkeypatch, on: bool) -> None:
    """World 0 freezes in period 0 (its sheet starts inside the walls), world 1 stays healthy; period 0 takes 512
    exchanges either way (a band re-run). Off: the bound the next clock reads is the frozen world's (vertices leaving
    the walls at about 1,000 m/s, under the finest step: 1,024 exchanges from then on, about 4.5 min a period here and
    climbing). On: world 0 is parked and the bound is the healthy world's own (the 1 kg ball's interface bound,
    3.5e-4 s), so periods 1-2 take 32 exchanges each (16 s). A reset unparks it."""
    rows, bound, parked = _cup_scene(monkeypatch, on, 3 if on else 1)
    assert [r[1] for r in rows] == [[True, False]] * len(rows)
    assert parked == [on, False]
    if not on:
        assert bound < FINEST
        return
    assert bound == pytest.approx(0.5 * math.sqrt(1.0 / 2.0e6), rel=1e-3)
    assert [r[0] for r in rows[1:]] == [32, 32]
