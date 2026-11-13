# SPDX-License-Identifier: Apache-2.0
"""Gap #10b, the full per-world exchange clock: its laws, written before the fix ([internal note] notes/gap-10b/LAWS.md).

The option is ``alloy.build(scene, clock_worlds=n)``: up to n worlds that need a finer exchange clock than the rest
("residents") step in a second pack of n slots on its own clock; the others step at their own, coarser rung. Today
every world takes the finest world's rung. The option is in the tree: every law here runs with it (the L2 rounding bound is stated in its test).

Everything the tests ask of the implementation is in the adapter block (``_build`` to ``_fine_state``)."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

import alloy

W, SLOTS, SEED = 4, 1, 0
LIGHT, HEAVY = 500.0, 8000.0          # [kg/m3] the box's density range: 13.5 g to 216 g at a 3 cm edge
NAMES = ["box", "cloth"]


# --- adapter: the only lines that know the implementation's names ---------------------------------------------------

def _has_option() -> bool:
    return "clock_worlds" in inspect.signature(alloy.build).parameters


def _build(scene: alloy.Scene, clock_worlds: int = 0, seed: int = SEED) -> Any:
    """The option when the tree has it; today's build otherwise (so a cost law fails on its number, not a TypeError)."""
    opt = {"clock_worlds": clock_worlds} if clock_worlds and _has_option() else {}
    return alloy.build(scene, device="cpu", seed=seed, **opt)


def _clock(sim: Any) -> Any:
    clock = getattr(sim.engine, "world_clock", None)
    assert clock is not None, "no per-world clock on this simulation (gap #10b)"
    return clock


def _resident(sim: Any) -> np.ndarray:
    return np.asarray(_clock(sim).resident, dtype=bool).copy()


def _fine_state(sim: Any, world: int) -> dict[str, np.ndarray]:
    """World ``world``'s state as its fine-pack slot holds it."""
    clock = _clock(sim)
    slot = int(np.asarray(clock.slot_of)[world])
    assert slot >= 0, f"world {world} is not resident"
    return {k: np.array(v, copy=True) for k, v in clock.fine.state(NAMES, world=slot).items() if k != "t"}


def _exchanges(sim: Any) -> tuple[int, int]:
    """(taken, what one shared clock takes) over the run: the option's own books, else today's census."""
    if getattr(sim.engine, "world_clock", None) is not None:
        row = sim.report()["world_clock"]
        return int(row["exchanges"]), int(row["exchanges_shared"])
    total = sim.engine.stepper.clock_census()
    return int(total["world_exchanges"]), int(total["world_exchanges"])


# --- the rig --------------------------------------------------------------------------------------------------------

def _scene(worlds: int = W, lo: float = LIGHT, hi: float = HEAVY) -> alloy.Scene:
    """A 3 cm box resting on a 4 mm sheet on a table, per world (test_fine_rerun's rig); the box's density is drawn
    per world in [lo, hi], so the lightest world's interface bound sets today's clock for all of them."""
    side = 0.03
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.4, 0.4, 0.1), pos=(0, 0, 0.05), static=True, friction=0.5)
    scene.add_cloth("cloth", size=(0.2, 0.2), pos=(0, 0, 0.102), resolution=(9, 9), mass=0.03, thickness=0.004,
                    self_contact=False)
    scene.add_box("box", size=(side,) * 3, pos=(0, 0, 0.104 + 0.001 + side / 2), density=hi)
    scene.randomize("box", "density", lo, hi)
    return scene


def _census(sim: Any) -> dict[str, Any]:
    return sim.engine._last_coupling.notes["census"]


def _state(sim: Any) -> dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in sim.state(NAMES).items() if k != "t"}


def _run(clock_worlds: int = 0, periods: int = 6, scene: alloy.Scene | None = None, lift: int | None = None,
         seed: int = SEED) -> tuple[list[dict[str, Any]], Any]:
    """Per period: every world's state, the census, who is resident. ``lift``: that world's box is lifted 5 cm at
    period 3 and lands (a first contact: a force jump in that world alone)."""
    sim = _build(scene or _scene(), clock_worlds, seed)
    rows = []
    for p in range(periods):
        if lift is not None and p == 3:
            sim.set_pose("box", pos=(0.0, 0.0, 0.17), world=lift)
        sim.step(1)
        clock = getattr(sim.engine, "world_clock", None)
        rows.append({"state": _state(sim), "census": _census(sim), "dt_x": float(sim.engine._last_coupling.dt_x),
                     "resident": None if clock is None else _resident(sim), "t": float(sim.t),
                     "need": None if clock is None else np.asarray(clock.need).copy()})
    return rows, sim


def _rung(sim: Any, row: dict[str, Any]) -> int:
    """The rung the coarse pack ran the period at."""
    return int(round(np.log2(sim.dt / row["dt_x"])))


def _same(a: dict[str, np.ndarray], b: dict[str, np.ndarray], worlds: Any, what: str) -> None:
    for k in a:
        np.testing.assert_array_equal(a[k][worlds], b[k][worlds], err_msg=f"{what}: {k}")


def _differs(a: dict[str, np.ndarray], b: dict[str, np.ndarray], world: int) -> bool:
    return any(not np.array_equal(a[k][world], b[k][world]) for k in a)


def _fine_world(rows: list[dict[str, Any]]) -> int:
    """The one world whose own rung is the finest at the end of the run (the rig has exactly one)."""
    j = np.asarray(rows[-1]["census"]["j_world"])
    assert np.count_nonzero(j == j.max()) == 1 and j.max() > np.sort(j)[-2], j
    return int(np.argmax(j))


# --- pass today -----------------------------------------------------------------------------------------------------

@pytest.mark.law
@pytest.mark.timeout(900)
def test_the_mixed_batch_has_a_rung_spread_and_today_pays_the_finest_for_all() -> None:
    """The rig's premise, and today's cost (L3's "before"): one world's own rung is finer than the others', the
    shared clock runs at it, and every world takes 2^J exchanges per period."""
    rows, sim = _run(periods=3)
    try:
        census = rows[-1]["census"]
        j = np.asarray(census["j_world"])
        fine = _fine_world(rows)
        shared = _rung(sim, rows[-1])
        assert census["drift"] is False and shared == j[fine]
        assert census["world_exchanges"] == W * 2 ** shared
        assert census["per_world_exchanges"] == int(np.sum(2 ** j)) < census["world_exchanges"]
        rest = int(np.delete(j, fine).max())
        assert W * 2 ** rest + SLOTS * 2 ** shared < 0.8 * W * 2 ** shared     # a 1-slot fine pack saves >= 1.25x
        assert sim.check()["ok"]
    finally:
        sim.close()


@pytest.mark.law
@pytest.mark.timeout(900)
def test_keyword_off_is_one_run_and_one_pack() -> None:
    """L6 (a) and L5's keyword-off half: two builds of the mixed batch end bitwise equal, and no second pack exists."""
    a, sa = _run(periods=3)
    b, sb = _run(periods=3)
    try:
        assert getattr(sa.engine, "world_clock", None) is None
        for p, (x, y) in enumerate(zip(a, b, strict=True)):
            assert x["dt_x"] == y["dt_x"] and x["t"] == y["t"], p
            _same(x["state"], y["state"], slice(None), f"period {p}")
    finally:
        sa.close()
        sb.close()


# --- fail today: the gap --------------------------------------------------------------------------------------------

@pytest.mark.law
@pytest.mark.timeout(900)
def test_cost_is_each_packs_own_rung_not_the_finest_for_all() -> None:
    """L3: with one fine slot the run takes W 2^Jc + n 2^Jf exchanges per period once the fine world is resident,
    at least 1.25x fewer than one shared clock's W 2^J. Today's count is W 2^J: this fails on the number."""
    rows, sim = _run(SLOTS, periods=5)
    try:
        taken, shared = _exchanges(sim)
        assert taken * 1.25 <= shared, (taken, shared)
        j = np.asarray(rows[-1]["census"]["j_world"])
        fine = int(np.argmax(j))
        row = sim.report()["world_clock"]
        assert row["j_fine"] == j[fine] and row["j_coarse"] == int(np.delete(j, fine).max())
        assert sim.check()["ok"]
    finally:
        sim.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_a_resident_does_not_feel_another_worlds_first_contact() -> None:
    """L1: a box lands in a non-resident world at period 3. Today that changes the fine world's bits (asserted with
    the option off; measured: through the shared pack, at an unchanged dt_x); with the option the resident world is
    bitwise the run without the landing."""
    base, s0 = _run(periods=6)
    fine = _fine_world(base)
    other = (fine + 1) % W
    hit, s1 = _run(periods=6, lift=other)
    s0.close()
    s1.close()
    assert _differs(base[-1]["state"], hit[-1]["state"], fine), "today the landing did not reach the fine world: a weak rig"
    assert _has_option(), "alloy.build has no clock_worlds keyword"
    calm, a = _run(SLOTS, periods=6)
    hit, b = _run(SLOTS, periods=6, lift=other)
    try:
        for p, (x, y) in enumerate(zip(calm, hit, strict=True)):
            assert x["resident"][fine] == y["resident"][fine], p
            assert not y["resident"][other], f"period {p}: the lifted world took the slot"
            if x["resident"][fine]:
                _same(x["state"], y["state"], fine, f"period {p}, resident world {fine}")
        assert calm[-1]["resident"][fine] and _differs(calm[-1]["state"], hit[-1]["state"], other)
    finally:
        a.close()
        b.close()


def _spread(a: list[dict[str, Any]], b: list[dict[str, Any]], worlds: Any) -> float:
    """The largest absolute difference of ``worlds``' state between two runs, over every period and key."""
    return max(float(np.max(np.abs(x["state"][k][worlds].astype(np.float64) - y["state"][k][worlds].astype(np.float64))))
               for x, y in zip(a, b, strict=True) for k in x["state"])


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_the_coarse_worlds_are_a_run_with_the_residents_slots_parked() -> None:
    """L2 (a), law F1, white box and bitwise: the worlds left in the coarse pack are the same batch run without the
    option in which the residents' slots are parked by hand in the same periods (same layout, same contact counts: the
    cross-world rounding leak of notes/gap-10b/leak.md cannot enter), and so is the coarse clock."""
    on, a = _run(SLOTS, periods=5)
    ref = _build(_scene(), 0)
    try:
        assert on[0]["resident"].any() and all(np.array_equal(r["resident"], on[0]["resident"]) for r in on)
        res = np.flatnonzero(on[0]["resident"])
        rest = [w for w in range(W) if w not in res]
        ref.engine.stepper.park_worlds(res.tolist(), ref.engine.state)
        for p, row in enumerate(on):
            ref.step(1)
            assert float(ref.engine._last_coupling.dt_x) == row["dt_x"], p
            _same(row["state"], _state(ref), rest, f"period {p}, non-resident worlds")
    finally:
        a.close()
        ref.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_the_coarse_clock_is_the_finest_need_among_the_unparked_worlds() -> None:
    """L2 (b): every period, including the one the fine world is parked in and the first calm and landing ones after
    it, the coarse pack's exchange level equals the finest need among the worlds it still steps: a resident's need
    (a landing in it included) never raises it."""
    for lift in (None, 0):
        base, s0 = _run(periods=1)
        fine = _fine_world(base)
        s0.close()
        rows, sim = _run(SLOTS, periods=6, lift=None if lift is None else fine)
        try:
            for p, row in enumerate(rows):
                rest = ~row["resident"]
                assert rest.sum() == W - SLOTS if row["resident"].any() else rest.all(), p
                assert _rung(sim, row) == int(row["need"][rest].max()), (lift, p, row["need"], row["resident"])
            assert rows[0]["resident"][fine]
        finally:
            sim.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_the_other_worlds_stay_within_the_leaks_spread_of_a_run_without_the_landing() -> None:
    """L2 (c), the original black-box comparison: a landing in the resident world (lifted at period 3) against the
    calm run, compared on the non-resident worlds. A separate defect (notes/gap-10b/leak.md, docs/known_gaps) makes
    a world's cloth step round differently when the batch's total soft-contact count changes, with or without the
    option, so this cannot be bitwise today. Bound: 10 x the spread the same comparison has with the option OFF on
    this rig (measured here); if that spread is zero the option itself leaks and this test must fail."""
    base, s0 = _run(periods=6)
    fine = _fine_world(base)
    hit, s1 = _run(periods=6, lift=fine)
    s0.close()
    s1.close()
    rest = [w for w in range(W) if w != fine]
    off = _spread(base, hit, rest)
    calm, a = _run(SLOTS, periods=6)
    hit, b = _run(SLOTS, periods=6, lift=fine)
    try:
        on = _spread(calm, hit, rest)
        print(f"L2 spread of the non-resident worlds, option off {off:.3e}, option on {on:.3e}")
        assert off > 0.0, "no leak with the option off: this rig no longer shows it, the bound would be meaningless"
        assert on <= 10 * off, (on, off)
        assert calm[-1]["resident"][fine] and _differs(calm[-1]["state"], hit[-1]["state"], fine)
    finally:
        a.close()
        b.close()


@pytest.mark.law
@pytest.mark.timeout(900)
def test_reads_and_inputs_reach_a_resident() -> None:
    """L4: each period a resident's read is its fine slot's state, bit for bit, at one ``sim.t`` for all; a pose set
    on a resident world shows in the next read and the world steps from it, whichever pack it is in."""
    sim = _build(_scene(), SLOTS)
    try:
        _clock(sim)
        sim.step(3)
        res = np.flatnonzero(_resident(sim))
        assert res.size == 1
        e = int(res[0])
        for _ in range(2):
            sim.step(1)
            got = _state(sim)
            for k, v in _fine_state(sim, e).items():
                np.testing.assert_array_equal(got[k][e], v, err_msg=k)
            assert np.isclose(sim.t, _clock(sim).fine.t, rtol=0, atol=1e-12)
        pos = np.array([0.02, 0.0, 0.16])
        sim.set_pose("box", pos=tuple(pos), world=e)       # a write brings the world back first (WP4's ruling E)
        assert not _resident(sim)[e]
        np.testing.assert_allclose(sim.state(["box"], world=e)["box.pos"], pos, rtol=0, atol=1e-6)
        before = _state(sim)["box.pos"][e].copy()
        sim.step(2)
        assert _state(sim)["box.pos"][e, 2] < before[2] - 1e-4        # it fell: the world went on stepping from the pose
        if _resident(sim)[e]:                                          # back in a slot: the read is the slot's again
            np.testing.assert_array_equal(_state(sim)["box.pos"][e], _fine_state(sim, e)["box.pos"])
        assert sim.check()["ok"]
    finally:
        sim.close()


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(2400)
def test_moves_repeat_with_the_seed_and_reset_clears_them() -> None:
    """L5: the same seed gives the same moves and the same bits; ``sim.reset()`` returns every world to its start,
    leaves nobody resident, and the run after it repeats the first."""
    assert _has_option(), "alloy.build has no clock_worlds keyword"
    a, sa = _run(SLOTS, periods=5)
    b, sb = _run(SLOTS, periods=5)
    try:
        assert any(r["resident"].any() for r in a)
        for p, (x, y) in enumerate(zip(a, b, strict=True)):
            assert np.array_equal(x["resident"], y["resident"]) and x["dt_x"] == y["dt_x"], p
            _same(x["state"], y["state"], slice(None), f"period {p}")
        sb.close()
        fresh = _build(_scene(), SLOTS)
        start = _state(fresh)
        fresh.close()
        sa.reset()
        assert not _resident(sa).any()                  # (sim.t is not reset today, with or without the option)
        _same(_state(sa), start, slice(None), "after reset")
        for p in range(5):
            sa.step(1)
            assert np.array_equal(_resident(sa), a[p]["resident"]), p
            _same(_state(sa), a[p]["state"], slice(None), f"period {p} after reset")
    finally:
        sa.close()


@pytest.mark.law
@pytest.mark.timeout(900)
def test_identical_worlds_take_nobody_and_change_no_bit() -> None:
    """L6 (b): with every box at one density nobody is resident, the books show no saving and the run is the
    keyword-off run bit for bit."""
    assert _has_option(), "alloy.build has no clock_worlds keyword"
    off, s0 = _run(periods=4, scene=_scene(lo=HEAVY, hi=HEAVY))
    on, s1 = _run(SLOTS, periods=4, scene=_scene(lo=HEAVY, hi=HEAVY))
    try:
        assert not any(r["resident"].any() for r in on)
        taken, shared = _exchanges(s1)
        assert taken == shared
        for p, (x, y) in enumerate(zip(off, on, strict=True)):
            assert x["dt_x"] == y["dt_x"], p
            _same(x["state"], y["state"], slice(None), f"period {p}")
    finally:
        s0.close()
        s1.close()
