# SPDX-License-Identifier: Apache-2.0
"""Gap #10b P2 slice 1 (alloy/coupling/parked.py): a parked world holds no clock bound, asks for no re-run, is held
bitwise and comes back with unpark; with no world parked nothing changes. CPU.

Fast laws drive a real Scheduler over fake solvers (4 worlds: a 0.2 kg box above a pad per world; world 3 is the one
that needs a finer clock, in a different way per law). The bitwise laws run MuJoCo + VBD on the box-on-cloth scene
(``alloy.testing.coupling``, 1 kg boxes pressed into the sheet at 0.3 m/s; world 3's box is 0.05 kg)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.coupling import contact_band as CB
from alloy.coupling.capture_passes import PER_WORLD_REPORT_FIELDS
from alloy.coupling.clock_census import PARKED_RUNG
from alloy.coupling.parked import quiet
from alloy.coupling.types import WatchState
from alloy.testing.coupling import ClothSceneSpec, PeriodRunner, cloth_scheduler, cloth_seed_model
from alloy.testing.coupling_parked import STATE, fake_mass_model, fake_mass_scheduler
from alloy.testing.coupling_parked import world_rows as _world

CONTROL_DT = 0.01
W, P = 4, 3                              # worlds; the parked one
HEAVY, LIGHT = 0.2, 0.002                # kg: fake boxes; at k_c 2,500 N/m the clock picks j 3 (dt_x_max) and j 5
DT_HEAVY, DT_LIGHT = CONTROL_DT / 8, CONTROL_DT / 32
FAR, NEAR = 0.0, 0.9                     # box x [m]: the destination's pad is at x = 1 (the contact gate's "near")
HIGH, REST = 0.5, 0.1                    # box z [m]: falling, or on the destination's floor (its proxy carries a force)
MIXED = (HEAVY, HEAVY, HEAVY, LIGHT)


def _fake(masses=MIXED, *, park=(), **kw):
    """(scheduler, runner) on the fake scene, ``park`` parked before the first period."""
    model = fake_mass_model(masses, kw.pop("xs", (FAR,) * W), kw.pop("heights", (HIGH,) * W))
    sched = fake_mass_scheduler(model, **kw)
    runner = PeriodRunner(model, sched, CONTROL_DT)
    if len(park):
        sched.park_worlds(list(park), runner.state)
    return sched, runner


def _periods(sched, runner, periods: int) -> list[dict]:
    """Run ``periods``: each one's clock, report rows and notes, and every world's state rows."""
    out = []
    for _ in range(periods):
        rep = runner.run_period()
        out.append({"clock": (rep.dt_x, dict(rep.substeps)), "census": rep.notes["census"],
                    "rows": {n: np.array(getattr(rep, n)) for n in PER_WORLD_REPORT_FIELDS},
                    "parked": rep.notes.get("parked"),
                    "worlds": [_world(runner.model, runner.state, sched, w) for w in range(W)]})
    return out


def _same(a: list[dict], b: list[dict], worlds=range(W), rows: bool = True) -> None:
    """Two runs agree period by period: the clock, and bit for bit the report rows and state of ``worlds``."""
    assert [x["clock"] for x in a] == [y["clock"] for y in b]
    for p, (x, y) in enumerate(zip(a, b, strict=True)):
        for w in worlds:
            assert x["worlds"][w] == y["worlds"][w], f"period {p} world {w}"
            for n in PER_WORLD_REPORT_FIELDS if rows else ():
                assert x["rows"][n][w].tobytes() == y["rows"][n][w].tobytes(), f"period {p} world {w} report {n}"


def _sticky(sched) -> tuple:
    """The host state a re-run or an escalation of some world leaves behind."""
    return (sched._clock.j_floor, getattr(sched, "_trips", 0), sched._ladder.refines, sched._onset.reruns,
            sched._ladder.last.tolist(), sched._onset.last.tolist(), sched._rollbacks)


# ---------------------------------------------------------------------------------------------- the clock (law b)

def test_a_parked_light_world_holds_no_interface_bound() -> None:
    """World 3's 2 g box sets the shared clock (j 5). Parked, every period's dt_x and substeps are those of four
    identical heavy worlds, worlds 0-2 end bitwise as in that batch, and the census marks world 3. Parked or
    unparked mid-run, the next period's clock is at the new bound at once."""
    loud, loud_runner = _fake()
    assert loud_runner.run_period().dt_x == DT_LIGHT
    loud.park_worlds([P], loud_runner.state)
    assert loud_runner.run_period().dt_x == DT_HEAVY
    loud.unpark_worlds()
    assert loud_runner.run_period().dt_x == DT_LIGHT
    _, ref = _fake((HEAVY,) * W)
    sched, runner = _fake(park=[P])
    want, got = _periods(ref.stepper, ref, 3), _periods(sched, runner, 3)
    _same(want, got, worlds=range(P))
    assert got[0]["clock"][0] == DT_HEAVY
    for row in got:
        assert row["census"]["j_world"] == [3, 3, 3, PARKED_RUNG] and row["census"]["drift"] is False
        assert row["census"]["per_world_exchanges"] == row["census"]["world_exchanges"] == W * 8
        assert row["parked"].tolist() == [False, False, False, True]


def test_a_parked_worlds_stable_step_moves_no_rung() -> None:
    """World 3's destination reports a stable step 1/16 of the others': the shared rung is its 16. Parked, the rung
    is the identical batch's every period (its NaN does not reach the host minimum either) and the report states no
    estimate for it (the precision rung pin reads those rows); the first period after unpark takes its rung."""
    def run(stable, park, then_unpark=False):
        sched, runner = _fake((HEAVY,) * W, adaptive=True, park=park)
        values = np.asarray(stable, dtype=np.float32)
        sched._estimators["dst"] = SimpleNamespace(alloy_stable_dt=lambda state, contacts, out: out.assign(values))
        reports = [runner.run_period() for _ in range(3)]
        assert all(np.isinf(r.notes["stable_dt"]["dst"][P]) == bool(len(park)) for r in reports)
        rungs = [r.substeps["dst"] for r in reports]
        if then_unpark:
            sched.unpark_worlds()
            rungs.append(runner.run_period().substeps["dst"])
        return rungs

    slow, fast = [1.0e-3] * W, [1.0e-3] * P + [1.0e-3 / 16]
    base = run(slow, ())
    assert run(fast, ())[-1] == 16 * base[-1]
    assert run(fast, [P], then_unpark=True) == [*base, 16 * base[-1]]
    assert run([1.0e-3] * P + [np.nan], [P]) == base


def test_a_parked_world_arms_no_pair_and_its_touch_reruns_nothing() -> None:
    """The contact gate: world 3's light box rests on the destination's floor beside the pad, the others fall far
    away. Unparked it arms the pair (the light clock). Parked, the pair stays released as in a batch with no box
    near, and the force on its unarmed proxy asks for no re-run."""
    xs, heights = (FAR, FAR, FAR, NEAR), (HIGH, HIGH, HIGH, REST)
    kw = {"masses": (LIGHT,) * W, "contact_gate": True}
    assert [r["clock"][0] for r in _periods(*_fake(xs=xs, heights=heights, **kw), 3)][-1] == DT_LIGHT
    want = _periods(*_fake(**kw), 3)
    sched, runner = _fake(xs=xs, heights=heights, park=[P], **kw)
    got = _periods(sched, runner, 3)
    assert [x["clock"] for x in want] == [y["clock"] for y in got] and got[-1]["clock"][0] == DT_HEAVY
    gate = sched._gate
    assert gate.touched.numpy()[P] == 1 and not gate.touched_unarmed().any()     # it did touch, unarmed
    assert gate.reruns == 0 and runner.rollbacks == runner.onset_reruns == 0 and not gate.armed_host.any()


def test_a_parked_worlds_armed_proxies_hold_no_gate_bound() -> None:
    """The contact gate with every box near its pad (the pair armed by the others): the bound is the lightest armed
    proxy's, world 3's. Parked, it is the heavy ones'."""
    kw = {"xs": (NEAR,) * W, "heights": (REST,) * W, "contact_gate": True}
    assert _periods(*_fake(**kw), 3)[-1]["clock"][0] == DT_LIGHT
    sched, runner = _fake(park=[P], **kw)
    want = _periods(*_fake((HEAVY,) * W, **kw), 3)
    got = _periods(sched, runner, 3)
    _same(want, got, worlds=range(P))
    assert sched._gate.armed_host.all() and got[0]["census"]["j_world"] == [3, 3, 3, PARKED_RUNG]


def test_a_parked_world_holds_no_always_on_bound() -> None:
    """A pair with a stated interface stiffness is never gated: its bound is always on, per world."""
    kw = {"contact_gate": True, "stiffness": 2500.0}
    assert _fake(**kw)[1].run_period().dt_x == DT_LIGHT
    sched, runner = _fake(park=[P], **kw)
    assert runner.run_period().dt_x == DT_HEAVY
    sched.unpark_worlds()
    assert runner.run_period().dt_x == DT_LIGHT


def _band(parked: bool, mass: float) -> float:
    """The band's speed [m/s] from two worlds' vertex records: world 0's particle at 1 m/s, world 1's at 9."""
    qd = np.array([(1.0, 0, 0), (9.0, 0, 0)], np.float32)
    contacts = SimpleNamespace(
        soft_contact_max=2, soft_contact_count=wp.array([2], dtype=wp.int32, device="cpu"),
        soft_contact_indices=wp.array(np.array([(0, -1, -1), (1, -1, -1)], np.int32), dtype=wp.vec3i, device="cpu"),
        soft_contact_body_vel=wp.zeros(2, dtype=wp.vec3, device="cpu"))
    view = SimpleNamespace(particle_world=wp.array([0, 1], dtype=wp.int32, device="cpu"))
    factory = SimpleNamespace(built=[(view, SimpleNamespace(soft_contact_gap=0.003), contacts, None)])
    model = SimpleNamespace(particle_mass=wp.array(np.full(2, mass, np.float32), dtype=float, device="cpu"))
    entries = {"cloth": SimpleNamespace(state_1=SimpleNamespace(particle_qd=wp.array(qd, dtype=wp.vec3, device="cpu")),
                                        solver=SimpleNamespace(model=model))}
    band = CB.ContactBandBound([(factory, "cloth")], "cpu", 2)
    band.park(np.array([False, True]) if parked else None)
    band.accumulate(entries)
    band.end_period()
    return float(band.last_speed[0])


@pytest.mark.parametrize("mass", [0.0, 1.0], ids=["records", "sheet"])
def test_a_parked_worlds_contacts_set_no_band_bound(mass: float) -> None:
    """The contact band's speed is the fastest contact of any world: world 1's 9 m/s. Parked, its records (massless
    particles: the records alone) and its sheet's particles (with mass) are skipped, and unparked they count again."""
    assert _band(False, mass) == pytest.approx(9.0, rel=1e-6, abs=0)
    assert _band(True, mass) == pytest.approx(1.0, rel=1e-6, abs=0)


# ---------------------------------------------------------------------------------------------- re-runs (law c)

def _tripping(sched, runner, world: int, period: int):
    """As the fine-pack laws trip a world: the first report of ``period`` says ``world`` escalated."""
    report, seen = sched.report, []

    def tripped():
        rep = report()
        if runner.periods != period or seen:
            return rep
        seen.append(True)
        ws = np.asarray(rep.watch_state).copy()
        ws[world] = int(WatchState.ESCALATED)
        return dataclasses.replace(rep, watch_state=ws)

    sched.report = tripped


def test_a_trip_in_a_parked_world_reruns_and_escalates_nothing() -> None:
    """World 3's report says ESCALATED in period 1. Parked: no rollback, no ladder or onset state changes (in that
    period or when the next one reads the report), and the run is the untripped one bit for bit. Unparked, the same
    trip re-runs the period."""
    calm, calm_runner = _fake((HEAVY,) * W, park=[P])
    want = _periods(calm, calm_runner, 4)
    sched, runner = _fake((HEAVY,) * W, park=[P])
    _tripping(sched, runner, P, 1)
    got = _periods(sched, runner, 4)
    assert runner.rollbacks == 0 and _sticky(sched) == _sticky(calm)
    _same(want, got, rows=False)
    loud, loud_runner = _fake((HEAVY,) * W)
    _tripping(loud, loud_runner, P, 1)
    _periods(loud, loud_runner, 2)
    assert loud_runner.rollbacks == 1 and loud._ladder.refines == 1


def _jumping(sched, runner, world: int, period: int) -> None:
    """One exchange of ``period`` books 10x the force-jump limit into ``world``'s power bond."""
    wd, seen = sched._watchdog, []
    bond = wd.accumulate_bond

    def jump(*args) -> None:
        bond(*args)
        if runner.periods != period or seen:
            return
        seen.append(True)
        work = wd.interface_work.numpy().copy()
        work[world] += 10.0 * sched._alloy_policy.watchdog.onset_gain_per_exchange * wd.e_scale.numpy()[world]
        wd.interface_work.assign(work)

    wd.accumulate_bond = jump


def test_a_force_jump_in_a_parked_world_is_no_event() -> None:
    """A jump of energy in one of world 3's exchanges. Unparked it is a force jump: the period re-runs finer and the
    world's cooldown starts. Parked: no re-run, its cooldown and watch state untouched, the booked work gone with the
    hold, and the report marks it."""
    loud, loud_runner = _fake((HEAVY,) * W)
    _jumping(loud, loud_runner, P, 1)
    _periods(loud, loud_runner, 2)
    assert loud_runner.onset_reruns >= 1 and loud._onset.last[P] == 2
    sched, runner = _fake((HEAVY,) * W, park=[P])
    work = sched._watchdog.interface_work.numpy()[P].copy()
    _jumping(sched, runner, P, 1)
    got = _periods(sched, runner, 2)
    assert runner.onset_reruns == runner.rollbacks == 0 and sched._onset.reruns == 0
    assert sched._onset.last[P] < 0 and not sched._watchdog.onset_hit.numpy()[P]
    assert sched._watchdog.interface_work.numpy()[P] == work
    assert all(row["rows"]["watch_state"][P] == int(WatchState.OK) for row in got)


def test_a_parked_world_short_of_passes_asks_for_no_rerun() -> None:
    """Lever #1: a capped capture leaves world 3 short of coupling passes (the flag is set by hand: the CPU never
    caps). Unparked that asks for the short re-run; parked it does not, and an escalated row of the report the next
    clock reads trips no ladder."""
    sched, runner = _fake((HEAVY,) * W, mode="staggered", capture_while=False)
    sched.allow_capped_captures()
    rep = runner.run_period()
    short = dataclasses.replace(rep, notes={**rep.notes, "short": np.arange(W) == P})
    assert sched.rollback_requested(short)
    sched.park_worlds([P], runner.state)
    assert not sched.rollback_requested(short)
    escalated = dataclasses.replace(rep, watch_state=np.where(np.arange(W) == P, int(WatchState.ESCALATED), 0))
    sched.choose_clock(CONTROL_DT, rep)
    before = _sticky(sched)
    sched.choose_clock(CONTROL_DT, escalated)
    assert _sticky(sched) == before
    assert quiet(escalated, None) is escalated and quiet(None, sched.parked) is None


# ---------------------------------------------------------------------------------------------- park and unpark

def test_unpark_resumes_the_cooldowns_and_a_reset_unparks() -> None:
    """A world parked one period after a force jump and a ladder trip is, 3 periods later, still one period after
    them; the other worlds' cooldowns are not shifted. Its held watch state is OK whatever the period before the park
    left. A reset world is no longer held."""
    sched, runner = _fake((HEAVY,) * W)
    _periods(sched, runner, 2)
    for rows in (sched._onset, sched._ladder):
        rows.last = np.full(W, sched._onset.period - 1, dtype=np.int64)
    watch = sched._watchdog.watch_state
    watch.assign(np.full(W, int(WatchState.ESCALATED), dtype=np.int32))      # the period before the park ended so
    sched.park_worlds(np.arange(W) == P, runner.state)
    assert sched.parked.tolist() == [False, False, False, True] and sched.graph_key()[-1] == "parked"
    _periods(sched, runner, 3)
    assert watch.numpy()[P] == int(WatchState.OK)                           # held calm, not put back escalated
    sched.unpark_worlds([P])
    now = sched._onset.period
    assert not sched.parked.any() and sched.graph_key() == () and sched._onset.parked is None
    assert sched._onset.last.tolist() == sched._ladder.last.tolist() == [now - 4, now - 4, now - 4, now - 1]
    sched.park_worlds([0, P], runner.state)
    sched.reset_worlds(wp.array([0, 0, 0, 1], dtype=wp.int32, device="cpu"))
    assert sched.parked.tolist() == [True, False, False, False]
    with pytest.raises(Exception, match="bool mask of length 4 or world ids"):
        sched.park_worlds(np.zeros(3, dtype=bool), runner.state)


# ---------------------------------------------------------------------------------------------- MuJoCo + VBD

SPEC = ClothSceneSpec(cells=8)
M_HEAVY, M_LIGHT = 1.0, 0.05             # kg: the clock's j 3 (dt_x_max) and j 5 at the cloth's 1e5 N/m


def _cloth(masses):
    """(scheduler, runner): the box-on-cloth law per world, each box pressed into the sheet at 0.3 m/s."""
    model = cloth_seed_model(masses, [0.3] * W, SPEC.contact_ke, SPEC, device="cpu", dz=-0.002)
    sched = cloth_scheduler(model, SPEC)
    return sched, PeriodRunner(model, sched, CONTROL_DT)


@pytest.fixture(scope="module")
def identical():
    """Four identical heavy worlds, the parking calls never made: 5 periods."""
    return _periods(*_cloth([M_HEAVY] * W), 5)


@pytest.fixture(scope="module")
def parked_run():
    """The light world 3 parked before the first period, 5 periods; then unparked, 1 more."""
    sched, runner = _cloth([M_HEAVY] * P + [M_LIGHT])
    out = {"bound": sched._interface_bound()}
    sched.park_worlds([P], runner.state)
    out["parked_bound"] = sched._interface_bound()
    out["held"] = _world(runner.model, runner.state, sched, P)
    out["periods"] = _periods(sched, runner, 5)
    sched.unpark_worlds([P])
    out["after"] = _periods(sched, runner, 1)
    return out


@pytest.mark.compile_heavy
@pytest.mark.law
@pytest.mark.timeout(600)
def test_no_world_parked_is_a_bitwise_no_op(identical) -> None:
    """Law a: a run whose parking calls park nobody, and that parks and unparks a world before its first period, is
    the untouched run bit for bit: every world's State and stepper arrays, the report rows, the census and the clock."""
    sched, runner = _cloth([M_HEAVY] * W)
    sched.park_worlds([], runner.state)
    sched.park_worlds([1], runner.state)
    sched.unpark_worlds()
    got = _periods(sched, runner, 5)
    _same(identical, got)
    assert [x["census"] for x in identical] == [y["census"] for y in got]
    assert all(row["parked"] is None for row in got) and sched.graph_key() == ()


@pytest.mark.compile_heavy
@pytest.mark.law
@pytest.mark.timeout(600)
def test_the_other_worlds_run_as_if_the_parked_world_were_heavy(identical, parked_run) -> None:
    """Law b: unparked, world 3's 50 g box binds the clock 4x finer. Parked, every period's dt_x and substeps are the
    identical heavy batch's and worlds 0-2 end every period bitwise as in it (State, solver adapters, report rows)."""
    assert parked_run["bound"] < CONTROL_DT / 16 < CONTROL_DT / 8 <= parked_run["parked_bound"]
    _same(identical, parked_run["periods"], worlds=range(P))
    assert parked_run["periods"][0]["clock"][0] == CONTROL_DT / 8
    assert {k for k in parked_run["held"] if k.startswith(("mjc.", "vbd."))}, "no solver adapter array was compared"


@pytest.mark.compile_heavy
@pytest.mark.law
@pytest.mark.timeout(600)
def test_a_parked_world_is_held_bitwise_and_moves_again_after_unpark(parked_run) -> None:
    """Law d: world 3's State rows, coupling carry, watchdog and solver-adapter rows are, after each of 5 periods,
    the ones it was parked with (the other worlds move). Unparked, it moves and the clock takes its bound at once."""
    held = parked_run["held"]
    assert len(held) > len(STATE) - 2 and any(k.startswith("mjc.") for k in held)
    for p, row in enumerate(parked_run["periods"]):
        assert row["worlds"][P] == held, f"period {p}: {[k for k in held if row['worlds'][P][k] != held[k]]}"
        assert row["worlds"][0]["state.body_q"] != parked_run["periods"][p - 1]["worlds"][0]["state.body_q"]
    after = parked_run["after"][0]
    assert after["clock"][0] == CONTROL_DT / 32 and after["parked"] is None
    assert after["worlds"][P]["state.body_q"] != held["state.body_q"]
    assert after["census"]["j_world"] == [3, 3, 3, 5]


def _untouched(sched) -> bool:
    """No re-run, escalation, ladder trip or force-jump cooldown was recorded."""
    floor, trips, refines, reruns, ladder, onset, rollbacks = _sticky(sched)
    return (floor, trips, refines, reruns, rollbacks) == (0, 0, 0, 0, 0) and max(ladder + onset, default=-1) < 0


def _write(runner, world: int) -> None:
    """A driver's write into the State the next period starts from: ``world``'s box 5 cm higher, rising at 3 m/s."""
    box = int(np.flatnonzero(runner.model.body_world.numpy() == world)[0])
    q, qd = runner.state.body_q.numpy().copy(), runner.state.body_qd.numpy().copy()
    q[box, 2] += 0.05
    qd[box, :3] = (0.0, 0.0, 3.0)
    runner.state.body_q.assign(q)
    runner.state.body_qd.assign(qd)


@pytest.mark.compile_heavy
@pytest.mark.law
@pytest.mark.timeout(600)
def test_a_refreshed_write_is_held_and_an_unrefreshed_one_is_gone(identical) -> None:
    """A driver writes another State into the parked slot (the world's real state, mirrored back). After
    ``refresh_parked`` the slot stays bitwise at the written rows for 5 periods, with no re-run, no ladder or onset
    change and an OK watch state, and worlds 0-2 stay bitwise on the identical heavy batch. A write without the
    refresh is gone after one exchange."""
    from alloy.core.errors import CouplingError

    sched, runner = _cloth([M_HEAVY] * P + [M_LIGHT])
    sched.park_worlds([P], runner.state)
    parked = _world(runner.model, runner.state, sched, P)
    _write(runner, P)
    sched.refresh_parked([P], runner.state)
    written = _world(runner.model, runner.state, sched, P)
    assert {k for k in written if written[k] != parked[k]} == {"state.body_q", "state.body_qd"}
    got = _periods(sched, runner, 5)
    _same(identical, got, worlds=range(P))
    assert all(row["worlds"][P] == written for row in got)
    assert all(row["rows"]["watch_state"][P] == int(WatchState.OK) for row in got)
    assert runner.rollbacks == runner.onset_reruns == 0 and _untouched(sched)
    _write(runner, P)                                    # no refresh: the next exchange's hold undoes the write
    assert _world(runner.model, runner.state, sched, P) != written
    runner.exchange(got[-1]["clock"][0])
    assert _world(runner.model, runner.state, sched, P) == written
    with pytest.raises(CouplingError, match="not all parked"):
        sched.refresh_parked([0], runner.state)


def _sim_rerun(masses, park) -> dict:
    """Through IntegratedSimulation (health, per-world rollback, the kept worlds' take-back): 3 periods, with a
    force jump in world 1's period 1. Per period: the clock, every world's digest (State and stepper arrays; a
    world's digest also hashes the batch's global rows, so it is compared between runs, not over time) and world 3's
    rows."""
    from alloy.core.registry import isolated_registry
    from alloy.integration import build_simulation
    from alloy.scene.builtin_plugins import register_builtin_solvers
    from alloy.testing import integration as I

    with isolated_registry(empty=True):
        register_builtin_solvers()
        model = cloth_seed_model(masses, [0.3] * W, SPEC.contact_ke, SPEC, device="cpu", dz=-0.002)
        sched = cloth_scheduler(model, SPEC)
        built = I.built_from_model(model, [I.entry_spec(model, n, 1 / 480) for n in ("mjc", "vbd")],
                                   control_dt=CONTROL_DT)
        sim = build_simulation(built, "standard", stepper=sched, graph="off")
        if len(park):
            sched.park_worlds(list(park), sim.state)
        out = {"held": _world(model, sim.state, sched, P), "rows": []}
        _jumping(sched, sim, 1, 1)
        for _ in range(3):
            sim.step_period()
            rep = sim._last_coupling
            out["rows"].append({"clock": (rep.dt_x, dict(rep.substeps)), "digest": I.digests(sim).copy(),
                                "watch": np.array(rep.watch_state), "slot": _world(model, sim.state, sched, P)})
        out.update(rollbacks=sim.rollbacks, onset=sched._onset.reruns, severity=sim.health.poll().severity.copy())
        return out


@pytest.mark.compile_heavy
@pytest.mark.law
@pytest.mark.timeout(600)
def test_another_worlds_rerun_leaves_the_parked_world_held() -> None:
    """World 1 (not parked) takes a force jump in period 1: IntegratedSimulation rolls it back, re-runs the period
    finer and gives the kept worlds their first run back. World 3, parked, is bitwise held through all of it, health
    flags no world, and worlds 0-2 end every period bitwise as in the same re-run of four identical heavy worlds."""
    want, got = _sim_rerun([M_HEAVY] * W, ()), _sim_rerun([M_HEAVY] * P + [M_LIGHT], [P])
    assert want["rollbacks"] == got["rollbacks"] == 1 and want["onset"] == got["onset"] == 1
    assert [x["clock"] for x in want["rows"]] == [y["clock"] for y in got["rows"]]
    for p, (x, y) in enumerate(zip(want["rows"], got["rows"], strict=True)):
        assert np.array_equal(x["digest"][:P], y["digest"][:P]), f"period {p}"
        assert y["slot"] == got["held"], f"period {p}: {[k for k in y['slot'] if y['slot'][k] != got['held'][k]]}"
        assert x["slot"] != want["held"] and y["watch"][P] == int(WatchState.OK)
    assert not got["severity"].any(), got["severity"]
