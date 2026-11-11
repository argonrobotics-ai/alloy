# SPDX-License-Identifier: Apache-2.0
"""#10b P1 S2 pieces (alloy.coupling.fine_rerun; [internal note] notes/wp4-perworld-clock/DESIGN.md sections 13-14)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.coupling.capture_passes import PER_WORLD_REPORT_FIELDS, merge_reports
from alloy.coupling.fine_rerun import widen_report
from alloy.coupling.types import CouplingReport


def _report(worlds: int, base: float, dt_x: float, **notes) -> CouplingReport:
    row = base + np.arange(worlds, dtype=np.float64)
    fields = {n: row.copy() for n in PER_WORLD_REPORT_FIELDS}
    fields["watch_state"] = np.zeros(worlds, np.int32) + int(base)
    return CouplingReport(dt_x=dt_x, substeps={"vbd": 2}, rollbacks=0, rung_changes=0, strong_pairs=(),
                          notes=dict(notes), **fields)


def test_the_fine_report_lands_on_its_worlds_and_merges_as_today() -> None:
    """Coarse worlds 1 and 3 re-ran in fine slots 0 and 1: their rows (fields and per-world notes, also one dict deep)
    come from the fine report, every other world's from the first; scalars are the re-run's; an array of another
    shape stays the first's and is named."""
    W, W_f = 5, 2
    first = _report(W, 100.0, 0.01, ladder=np.arange(W) + 100, onset={"flags": np.arange(W) + 100},
                    pairs=np.zeros(3), level=0)
    fine = _report(W_f, 200.0, 0.0025, ladder=np.arange(W_f) + 200, onset={"flags": np.arange(W_f) + 200},
                   pairs=np.ones(4), level=1)
    wide = widen_report(first, fine, [1, 3], [0, 1])
    want = np.array([100.0, 200.0, 102.0, 201.0, 104.0])
    for n in PER_WORLD_REPORT_FIELDS:
        if n != "watch_state":
            assert np.array_equal(getattr(wide, n), want), n
    assert np.array_equal(wide.watch_state, [100, 200, 100, 200, 100])
    assert np.array_equal(wide.notes["ladder"], [100, 200, 102, 201, 104])
    assert np.array_equal(wide.notes["onset"]["flags"], [100, 200, 102, 201, 104])
    assert wide.dt_x == 0.0025 and wide.notes["level"] == 1
    assert np.array_equal(wide.notes["pairs"], np.zeros(3)) and wide.notes["fine_pack"]["unmerged"] == ["pairs"]
    keep = np.array([1, 0, 1, 0, 1], np.int32)
    merged = merge_reports(first, wide, keep)
    assert np.array_equal(merged.res_vel, want) and merged.dt_x == 0.0025


def _scene(worlds: int, mass: float = 0.2, drop: float = 0.0):
    """test_period_ladder's three boxes (a box resting on a 4 mm sheet on a table, per world), here 0.2 kg: at 5 kg
    the scene is not batch invariant (identical worlds part at rounding level within one pack: VBD's contact sums),
    so no re-run of a subset can match today's bit for bit there (``test_a_heavy_box...``)."""
    import alloy

    side = (mass / 8000.0) ** (1.0 / 3.0)
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.4, 0.4, 0.1), pos=(0, 0, 0.05), static=True, friction=0.5)
    scene.add_cloth("cloth", size=(0.2, 0.2), pos=(0, 0, 0.102), resolution=(9, 9), mass=0.03, thickness=0.004,
                    self_contact=False)
    z = 0.104 + 0.001 + side / 2 + np.where(np.arange(worlds) == 0, drop, 0.0)     # world 0's box from ``drop`` higher
    scene.add_box("box", size=(side,) * 3, pos=np.stack([np.zeros(worlds), np.zeros(worlds), z], 1), mass=mass)
    return alloy.build(scene, device="cpu")


def _run(fine: bool, trip_period: int | tuple[int, ...] = 2, periods: int = 5, world: int | tuple[int, ...] = 1,
         mass: float = 0.2, drop: float = 0.0):
    """Each period's clock, every world's state, the last report and health records; world ``world`` trips once in
    ``trip_period`` (the Scheduler's first-attempt report says it escalated, as test_period_ladder)."""
    import dataclasses

    from alloy.coupling.fine_rerun import FineRerun
    from alloy.coupling.types import WatchState

    sim = _scene(3, mass, drop)
    eng, st = sim.engine, sim.engine.stepper
    if fine:
        eng.fine_rerun = FineRerun(lambda n: _scene(n, mass, drop), 1)
    report, seen = st.report, {}

    def tripped():
        rep = report()
        if eng.periods in np.atleast_1d(trip_period) and eng.periods not in seen:
            seen[eng.periods] = True
            ws = np.asarray(rep.watch_state).copy()
            ws[list(np.atleast_1d(world))] = int(WatchState.ESCALATED)
            rep = dataclasses.replace(rep, watch_state=ws)
        return rep

    st.report = tripped
    out = []
    for _ in range(periods):
        sim.step(1)
        rep = eng._last_coupling
        s = sim.state(["box", "cloth"])
        out.append({"clock": (rep.dt_x, dict(rep.substeps)), "rows": {n: np.array(getattr(rep, n)) for n in
                    PER_WORLD_REPORT_FIELDS}, "state": {k: np.array(v, copy=True) for k, v in s.items() if k != "t"},
                    "t": float(sim.t), "steps": eng.steps, "record": eng.health.record.numpy().copy(),
                    "band": float(getattr(st, "_band_bound", math.inf))})
    return out, sim


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_the_fine_pack_rerun_equals_todays_whole_rerun() -> None:
    """Law (CPU; DESIGN 13 S2, S3's independence brought forward): world 1 trips once in period 2. Re-run alone in a
    1-world fine pack (pinned to the coarse decisions), every period ends bitwise as today's whole re-run: every
    world's state and health record, the clock, the step count, and the merged report's rows; the fine pack took the
    re-run (no fallback) and the merged report asks for no rollback."""
    base, a = _run(False)
    got, b = _run(True)
    try:
        fr = b.engine.fine_rerun
        assert fr.reruns == 1 and fr.fallbacks == {}, fr.fallbacks
        assert a.engine.rollbacks == b.engine.rollbacks == 1
        for p, (x, y) in enumerate(zip(base, got, strict=True)):
            assert x["clock"] == y["clock"] and x["t"] == y["t"] and x["steps"] == y["steps"], p
            for k in x["state"]:
                np.testing.assert_array_equal(x["state"][k], y["state"][k], err_msg=f"period {p} {k}")
            for n in x["rows"]:
                np.testing.assert_array_equal(x["rows"][n], y["rows"][n], err_msg=f"period {p} report {n}")
            for f in x["record"].dtype.names:          # field by field: the struct's padding bytes are not state
                assert np.array_equal(x["record"][f], y["record"][f], equal_nan=True), (p, f)
        assert not b.engine.stepper.rollback_requested(b.engine._last_coupling)
    finally:
        a.close()
        b.engine.fine_rerun.close()
        b.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_a_heavy_box_reruns_alone_within_its_batch_spread() -> None:
    """At 5 kg the scene is not batch invariant: three identical worlds part at rounding level within one pack, so
    world 1's re-run depends on its batch today too. The kept worlds are still bitwise today's and the clock is
    today's; the re-run world stays within 0.1 mm and 1 mm/s of today's over the run (the rounding grows, the
    physics does not part)."""
    base, a = _run(False, mass=5.0)
    got, b = _run(True, mass=5.0)
    try:
        assert b.engine.fine_rerun.reruns == 1, b.engine.fine_rerun.fallbacks
        for p, (x, y) in enumerate(zip(base, got, strict=True)):
            assert x["clock"] == y["clock"] and x["steps"] == y["steps"], p
            for k in x["state"]:
                for w in (0, 2):
                    np.testing.assert_array_equal(x["state"][k][w], y["state"][k][w], err_msg=f"{p} {k} {w}")
                tol = 1e-3 if k.endswith(".vel") else 1e-4      # m/s, m: a resting box, rounding grown chaotically
                np.testing.assert_allclose(x["state"][k][1], y["state"][k][1], rtol=0, atol=tol, err_msg=f"{p} {k}")
    finally:
        a.close()
        b.engine.fine_rerun.close()
        b.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_a_refused_rerun_is_todays_bit_for_bit() -> None:
    """Two worlds trip and the fine pack has one slot: refused before any side effect (WP12 14.1), so the run is
    today's bit for bit and the reason is counted."""
    base, a = _run(False, world=(0, 1))
    got, b = _run(True, world=(0, 1))
    try:
        assert b.engine.fine_rerun.reruns == 0 and b.engine.fine_rerun.fallbacks == {"2 worlds over 1 slots": 1}
        for p, (x, y) in enumerate(zip(base, got, strict=True)):
            assert x["clock"] == y["clock"] and x["steps"] == y["steps"], p
            for k in x["state"]:
                np.testing.assert_array_equal(x["state"][k], y["state"][k], err_msg=f"period {p} {k}")
    finally:
        a.close()
        b.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_the_band_after_a_fine_rerun_keeps_the_kept_worlds_speeds() -> None:
    """WP10's (b) 1: in world 0 a 5 kg box lands on the cloth at ~1 m/s in period 9, the first period whose band binds
    (period 10 runs at dt_x / 2); world 1 (at rest) trips in that period. The fine path's band keeps the first
    attempt's speeds (world 0's landing): its band bound after the trip period is today's within 2x (from the re-run
    world alone it would be ~30x looser; the clock cannot show that, its hysteresis holds the finer rung the re-run
    attempt took), every period's clock is today's and the kept worlds are bitwise."""
    base, a = _run(False, trip_period=9, periods=11, mass=5.0, drop=0.05)
    got, b = _run(True, trip_period=9, periods=11, mass=5.0, drop=0.05)
    try:
        assert b.engine.fine_rerun.reruns == 1, b.engine.fine_rerun.fallbacks
        assert base[10]["clock"][0] < base[8]["clock"][0]          # the band binds after the landing
        today, fine = base[9]["band"], got[9]["band"]               # [s] the band bound after the trip period
        assert today < 1e-3 and 0.5 < fine / today < 2.0, (today, fine)
        assert [x["clock"] for x in base] == [y["clock"] for y in got]
        for p, (x, y) in enumerate(zip(base, got, strict=True)):
            for k in x["state"]:
                for w in (0, 2):
                    np.testing.assert_array_equal(x["state"][k][w], y["state"][k][w], err_msg=f"{p} {k} {w}")
    finally:
        a.close()
        b.engine.fine_rerun.close()
        b.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_the_sticky_ladder_moves_with_the_fine_pack() -> None:
    """World 1 trips in periods 2, 3 and 4: the second trip within the ladder's window halves dt_x for good (the
    clock's floor), the third switches the coupling to staggered + Aitken (the sticky ladder, pack-wide). The fine pack
    takes every re-run with that state mirrored, and every period is today's bit for bit (states, clocks, records)."""
    base, a = _run(False, trip_period=(2, 3, 4), periods=7)
    got, b = _run(True, trip_period=(2, 3, 4), periods=7)
    try:
        fr, st = b.engine.fine_rerun, b.engine.stepper
        assert fr.reruns == 3 and fr.fallbacks == {}, fr.fallbacks
        assert st._clock.j_floor > 0 and getattr(st, "_trips", 0) == 2 and all(es.staggered for es in st._sets)
        for p, (x, y) in enumerate(zip(base, got, strict=True)):
            assert x["clock"] == y["clock"] and x["steps"] == y["steps"], p
            for k in x["state"]:
                np.testing.assert_array_equal(x["state"][k], y["state"][k], err_msg=f"period {p} {k}")
            for f in x["record"].dtype.names:
                assert np.array_equal(x["record"][f], y["record"][f], equal_nan=True), (p, f)
    finally:
        a.close()
        fr.close()
        b.close()


@pytest.mark.law
@pytest.mark.timeout(600)
def test_a_held_wrench_keeps_todays_rerun() -> None:
    """sim.apply_wrench's work is booked around the coarse stepper only (health_work.WrenchWork): while a wrench is
    held, a re-run stays whole and says why (OSS-3), and health stays OK."""
    from alloy.coupling.fine_rerun import FineRerun

    sim = _scene(3)
    try:
        sim.engine.fine_rerun = FineRerun(lambda n: _scene(n), 1)
        sim.apply_wrench("box", force=(0.0, 0.0, 0.2))
        report = sim.engine.stepper.report
        tripped = {}

        def trip():
            rep = report()
            if sim.engine.periods == 1 and not tripped:
                tripped[1] = True
                import dataclasses

                from alloy.coupling.types import WatchState
                ws = np.asarray(rep.watch_state).copy()
                ws[1] = int(WatchState.ESCALATED)
                rep = dataclasses.replace(rep, watch_state=ws)
            return rep

        sim.engine.stepper.report = trip
        sim.step(3)
        fr = sim.engine.fine_rerun
        assert fr.reruns == 0 and [k for k in fr.fallbacks if k.startswith("a held wrench")], fr.fallbacks
        assert sim.check()["ok"]
    finally:
        sim.engine.fine_rerun.close()
        sim.close()
