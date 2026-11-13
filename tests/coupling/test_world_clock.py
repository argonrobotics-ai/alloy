# SPDX-License-Identifier: Apache-2.0
"""Per-world exchange clock, the residency rule (alloy/coupling/world_clock.py): pure host laws."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from alloy.coupling.world_clock import Moves, WorldClock, WorldClockConfig, split_cost


def _clock(worlds: int = 64, slots: int = 4, **kw) -> WorldClock:
    return WorldClock(worlds, WorldClockConfig(slots=slots, **kw))


def _all(worlds: int) -> np.ndarray:
    return np.ones(worlds, dtype=bool)


def _check_maps(clock: WorldClock) -> None:
    """slot_of and world_in are inverse maps: no slot or world is held twice."""
    for w in np.flatnonzero(clock.resident):
        assert clock.world_in[clock.slot_of[w]] == w
    held = clock.world_in[clock.world_in >= 0]
    assert len(set(held.tolist())) == held.size == int(clock.resident.sum())
    for s in np.flatnonzero(clock.world_in >= 0):
        assert clock.slot_of[clock.world_in[s]] == s


def test_identical_worlds_never_reside():
    """The no-op: equal rungs give no resident, no move, and the shared clock's cost, whatever the rung does."""
    clock = _clock()
    for j in (4, 7, 8, 5, 5, 9):
        rungs = np.full(64, j)
        assert not clock.update(rungs, _all(64))
        assert not clock.resident.any()
        cost = clock.cost(rungs)
        assert cost["split"] == cost["shared"] == 64 * 2**j and cost["gain"] == 1.0
        assert clock.rungs() == (j, j)


def test_one_fine_world_of_64_resides_at_once():
    rungs = np.full(64, 4)
    rungs[17] = 8
    clock = _clock()
    moves = clock.update(rungs, _all(64))
    assert moves == Moves(enter=((17, 0),), leave=())
    assert clock.resident.sum() == 1 and clock.slot_of[17] == 0 and clock.world_in[0] == 17
    assert clock.rungs() == (4, 8)
    cost = clock.cost(rungs)
    assert cost["shared"] == 64 * 2**8 == 16384
    assert cost["split"] == 64 * 2**4 + 4 * 2**8 == 2048       # every fine slot steps, idle ones too
    assert cost["gain"] == pytest.approx(8.0)


def test_more_fine_worlds_than_slots():
    """The finest take the slots; the coarse pack runs at the finest world left out."""
    rungs = np.full(64, 3)
    rungs[[5, 9, 20, 33, 40, 41]] = [9, 9, 8, 8, 6, 6]
    clock = _clock()
    moves = clock.update(rungs, _all(64))
    assert moves.enter == ((5, 0), (9, 1), (20, 2), (33, 3))
    assert clock.rungs() == (6, 9)
    assert clock.cost(rungs)["split"] == 64 * 2**6 + 4 * 2**9
    _check_maps(clock)


def test_smallest_k_of_least_cost():
    """A world whose stay would not lower the coarse rung is not taken (ties on a rung are never split)."""
    rungs = np.full(16, 2)
    rungs[[1, 2, 3]] = 6                   # three worlds at the top rung, two slots: no k lowers Jc
    clock = _clock(16, 2)
    assert not clock.update(rungs, _all(16)) and not clock.resident.any()
    rungs[3] = 2                           # two left: both must go for Jc to fall
    assert clock.update(rungs, _all(16)).enter == ((1, 0), (2, 1))


def test_min_gain_refuses_a_small_saving():
    rungs = np.full(8, 4)
    rungs[0] = 5                           # shared 256; split 8*16 + 4*32 = 256: no gain
    clock = _clock(8, 4)
    assert not clock.update(rungs, _all(8))
    rungs[0] = 6                           # shared 512; split 128 + 256 = 384: 1.33x
    assert not _clock(8, 4, min_gain_num=3, min_gain_den=2).update(rungs, _all(8))
    assert _clock(8, 4, min_gain_num=5, min_gain_den=4).update(rungs, _all(8)).enter == ((0, 0),)


def test_hysteresis_no_chatter_on_an_alternating_rung():
    clock = _clock(leave_after=3)
    fine, calm = np.full(64, 4), np.full(64, 4)
    fine[7] = 8
    moves = [clock.update(fine if i % 2 == 0 else calm, _all(64)) for i in range(20)]
    assert moves[0].enter == ((7, 0),)
    assert not any(moves[1:])
    assert clock.report()["enters"] == 1 and clock.report()["leaves"] == 0


def test_resident_leaves_after_leave_after_updates_and_counts_in_the_fine_rung():
    clock = _clock(leave_after=3, min_gain_num=2, min_gain_den=1)
    rungs = np.full(64, 4)
    rungs[7] = 8
    clock.update(rungs, _all(64))
    calm = np.full(64, 4)
    calm[7] = 5                            # 1.78x: under min_gain, not chosen, but still resident: Jf follows it
    for _ in range(2):
        assert not clock.update(calm, _all(64))
        assert clock.rungs() == (4, 5)
        assert clock.cost(calm)["split"] == 64 * 2**4 + 4 * 2**5
    assert clock.update(calm, _all(64)) == Moves(leave=((0, 7),))
    assert not clock.resident.any() and clock.rungs() == (5, 5)


def test_ineligible_resident_leaves_at_once():
    clock = _clock()
    rungs = np.full(64, 4)
    rungs[[7, 30]] = 8
    clock.update(rungs, _all(64))
    eligible = _all(64)
    eligible[7] = False
    moves = clock.update(rungs, eligible)
    assert moves.leave == ((0, 7),) and not clock.resident[7]
    # world 7 now bounds the coarse pack at 8, so world 30's stay saves nothing: it lingers by hysteresis only
    assert clock.rungs() == (8, 8) and clock.resident[30]
    never = np.zeros(64, dtype=bool)
    assert clock.update(rungs, never).leave == ((1, 30),)


def test_a_finer_entrant_displaces_the_coarsest_unwanted_resident():
    clock = _clock(16, 2, leave_after=10)
    rungs = np.full(16, 2)
    rungs[[3, 4]] = [7, 6]
    assert clock.update(rungs, _all(16)).enter == ((3, 0), (4, 1))
    rungs[[3, 4]] = [3, 2]                 # both calm, kept by hysteresis
    rungs[[10, 11]] = [8, 8]
    moves = clock.update(rungs, _all(16))
    assert moves.leave == ((1, 4), (0, 3))                     # the coarsest goes first
    assert moves.enter == ((10, 1), (11, 0))
    _check_maps(clock)
    assert clock.rungs() == (3, 8)


def test_evict_off_a_newcomer_waits_for_a_free_slot():
    assert WorldClockConfig(slots=2).evict is True and WorldClockConfig(slots=2).leave_after == 10
    clock = _clock(16, 2, leave_after=3, evict=False)
    rungs = np.full(16, 2)
    rungs[[3, 4]] = [7, 6]
    clock.update(rungs, _all(16))
    rungs[[3, 4]] = [3, 2]
    rungs[[10, 11]] = [8, 8]
    for _ in range(2):                     # the newcomers wait outside and bound the coarse pack
        assert not clock.update(rungs, _all(16))
        assert clock.rungs() == (8, 3) and not clock.resident[[10, 11]].any()
    moves = clock.update(rungs, _all(16))
    assert moves.leave == ((0, 3), (1, 4)) and moves.enter == ((10, 0), (11, 1))
    assert clock.rungs() == (3, 8)


def test_lowest_free_slot_first():
    clock = _clock(16, 3, leave_after=1)
    rungs = np.full(16, 2)
    rungs[[1, 2, 3]] = [9, 8, 7]
    clock.update(rungs, _all(16))
    rungs[1] = 2                           # slot 0 frees: 2 and 3 alone still pay
    assert clock.update(rungs, _all(16)).leave == ((0, 1),)
    rungs[12] = 9
    assert clock.update(rungs, _all(16)).enter == ((12, 0),)


def test_state_round_trip_gives_the_same_later_moves():
    rng = np.random.default_rng(3)
    seq = [(rng.integers(2, 9, 24), rng.random(24) > 0.1) for _ in range(40)]
    a = _clock(24, 3, leave_after=2)
    for j, ok in seq[:15]:
        a.update(j, ok)
    b = _clock(24, 3, leave_after=2)
    b.load_state(a.state())
    state = a.state()
    for j, ok in seq[15:]:
        assert a.update(j, ok) == b.update(j, ok)
    for key, val in a.state().items():
        np.testing.assert_array_equal(val, b.state()[key])
    assert state["counts"][0] == 15        # state() is a copy: later updates did not change it
    with pytest.raises(ValueError):
        _clock(25, 3).load_state(state)


def test_determinism_and_reset():
    rng = np.random.default_rng(11)
    seq = [(rng.integers(0, 10, 32), rng.random(32) > 0.2) for _ in range(60)]
    a, b = _clock(32, 4), _clock(32, 4)
    first = [a.update(j, ok) for j, ok in seq]
    assert first == [b.update(j, ok) for j, ok in seq]
    a.reset()
    assert not a.resident.any() and (a.world_in == -1).all() and a.report()["updates"] == 0
    assert first == [a.update(j, ok) for j, ok in seq]


def _brute_force(j: np.ndarray, eligible: np.ndarray, slots: int) -> int:
    """The least cost over every eligible subset of at most ``slots`` worlds (the rule only tries the k finest)."""
    ids = np.flatnonzero(eligible).tolist()
    best = split_cost(j, np.zeros(j.size, dtype=bool), slots)[0]
    for k in range(1, slots + 1):
        for sub in itertools.combinations(ids, k):
            mask = np.zeros(j.size, dtype=bool)
            mask[list(sub)] = True
            best = min(best, split_cost(j, mask, slots)[0])
    return best


def test_chosen_set_is_the_cost_minimum_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(300):
        worlds = int(rng.integers(3, 10))
        slots = int(rng.integers(1, min(4, worlds)))
        j = rng.integers(0, 7, worlds)
        eligible = rng.random(worlds) > 0.2
        clock = _clock(worlds, slots, min_gain_num=1, min_gain_den=1)
        chosen = clock.choose(j, eligible)
        assert not (chosen & ~eligible).any() and chosen.sum() <= slots
        cost = split_cost(j, chosen, slots)[0]
        assert cost == _brute_force(j, eligible, slots)
        # the smallest such set: dropping its coarsest member costs strictly more
        if chosen.any():
            fewer = chosen.copy()
            fewer[max(np.flatnonzero(chosen), key=lambda w: (-j[w], w))] = False
            assert split_cost(j, fewer, slots)[0] > cost


@pytest.mark.parametrize("seed", [0, 1])
def test_moves_never_assign_a_slot_or_world_twice(seed):
    rng = np.random.default_rng(5 + seed)
    clock = _clock(20, 3, leave_after=2, min_gain_num=1, min_gain_den=1, evict=bool(seed))
    for _ in range(400):
        before = clock.resident.copy()
        moves = clock.update(rng.integers(0, 8, 20), rng.random(20) > 0.15)
        _check_maps(clock)
        worlds = [w for w, _ in moves.enter] + [w for _, w in moves.leave]
        assert len(set(worlds)) == len(worlds)                 # no world both enters and leaves in one update
        assert len({s for _, s in moves.enter}) == len(moves.enter)
        assert all(before[w] for _, w in moves.leave) and not any(before[w] for w, _ in moves.enter)
        split = clock.cost(clock.state()["j"])
        assert (split["j_coarse"], split["j_fine"]) == clock.rungs()


def test_bad_arguments():
    for kw in ({"slots": 0}, {"slots": True}, {"slots": 2, "leave_after": 0}, {"slots": 2, "min_gain_num": 3, "min_gain_den": 4}, {"slots": 2, "min_gain_num": 1.5}, {"slots": 2, "fine_cost_den": 0}):
        with pytest.raises(ValueError):
            WorldClockConfig(**kw)
    with pytest.raises(ValueError):
        WorldClock(4, WorldClockConfig(slots=4))
    with pytest.raises(ValueError):
        _clock(8, 2).update(np.zeros(7, dtype=int), _all(8))


# ---- integer gate, hysteresis on the gate, weighted cost, re-run asks ----
def test_integer_gate_equals_the_float_gate_where_the_float_is_exact():
    rng = np.random.default_rng(21)
    for _ in range(300):
        worlds = int(rng.integers(3, 12))
        slots = int(rng.integers(1, min(4, worlds)))
        j = rng.integers(0, 8, worlds)
        eligible = rng.random(worlds) > 0.2
        num, den = [(5, 4), (3, 2), (2, 1), (1, 1), (9, 8)][int(rng.integers(0, 5))]
        clock = _clock(worlds, slots, min_gain_num=num, min_gain_den=den)
        got = clock.choose(j, eligible)
        free = _clock(worlds, slots, min_gain_num=1, min_gain_den=1).choose(j, eligible)
        if free.any():
            shared = split_cost(j, np.zeros(worlds, dtype=bool), slots)[0]
            old = free if shared >= (num / den) * split_cost(j, free, slots)[0] else np.zeros(worlds, dtype=bool)
        else:
            old = free
        np.testing.assert_array_equal(got, old)


def _gain_gate_rungs(good: bool) -> np.ndarray:
    rungs = np.full(8, 4)
    rungs[0] = 6 if good else 5            # good: 512 vs 384 (1.33x); bad: 256 vs 256
    return rungs


def test_gate_hysteresis_residents_stay_through_a_dipping_gate():
    clock = _clock(8, 4, leave_after=3)
    assert clock.update(_gain_gate_rungs(True), _all(8)).enter == ((0, 0),)
    for i in range(31):                    # the gate alternates: no leave, no second enter
        assert not clock.update(_gain_gate_rungs(i % 2 == 1), _all(8))
    assert clock.report()["enters"] == 1 and clock.report()["leaves"] == 0
    assert clock.state()["counts"][3] == 1  # the last update failed the gate once


def test_gate_failing_leave_after_times_releases_and_enter_is_not_delayed():
    clock = _clock(8, 4, leave_after=3)
    clock.update(_gain_gate_rungs(True), _all(8))
    for _ in range(2):
        assert not clock.update(_gain_gate_rungs(False), _all(8))
    assert clock.update(_gain_gate_rungs(False), _all(8)) == Moves(leave=((0, 0),))
    assert clock.state()["counts"][3] == 0
    assert clock.update(_gain_gate_rungs(True), _all(8)).enter == ((0, 0),)     # enter at once again
    clock.reset()
    assert clock.state()["counts"][3] == 0


def test_weighted_cost_takes_fewer_residents_as_the_weight_grows_and_none_when_it_never_pays():
    rungs = np.full(64, 3)
    rungs[[0, 1]] = [8, 6]                 # one resident leaves Jc 6, both Jc 3; Jf is 8 either way
    counts = []
    for num in (1, 4, 16, 1000):
        clock = _clock(64, 4, fine_cost_num=num, min_gain_num=1, min_gain_den=1)
        counts.append(int(clock.choose(rungs, _all(64)).sum()))
    assert counts == [2, 2, 0, 0]           # 512 + 1024 * weight against 16384: pays up to weight 15
    clock = _clock(64, 4, fine_cost_num=3, fine_cost_den=2)
    clock.update(rungs, _all(64))
    cost = clock.cost(rungs)
    assert cost["split"] == 64 * 8 + 4 * 256 and cost["split_weighted"] == (64 * 8 * 2 + 4 * 256 * 3, 2)
    assert cost["shared_weighted"] == (64 * 256 * 2, 2)
    assert _clock().cost(rungs)["split_weighted"] == (64 * 256, 1)


def test_rerun_entrants_enter_despite_a_failing_gate():
    clock = _clock(8, 4)
    rungs = _gain_gate_rungs(False)
    assert not clock.update(rungs, _all(8))
    ask = np.zeros(8, dtype=bool)
    ask[0] = True
    moves = clock.update(rungs, _all(8), rerun=ask)
    assert moves == Moves(enter=((0, 0),)) and moves.refused == ()
    assert clock.rungs() == (4, 5)


def test_rerun_without_a_slot_is_refused_and_with_evict_it_evicts():
    rungs = np.full(16, 2)
    rungs[[3, 4]] = [7, 6]
    ask = np.zeros(16, dtype=bool)
    ask[10] = True
    late = np.full(16, 2)
    late[[3, 4]] = [3, 2]                  # both residents calm, kept by hysteresis
    late[10] = 5
    for evict in (False, True):
        clock = _clock(16, 2, evict=evict)
        clock.update(rungs, _all(16))
        moves = clock.update(late, _all(16), rerun=ask)
        if evict:
            assert moves == Moves(enter=((10, 1),), leave=((1, 4),))        # the coarsest resident goes
            assert moves.refused == ()
        else:
            assert not moves and moves.refused == (10,)
            assert not clock.resident[10]
    clock = _clock(16, 2)
    bad = np.zeros(16, dtype=bool)
    bad[9] = True
    ineligible = _all(16)
    ineligible[9] = False
    assert clock.update(late, ineligible, rerun=bad).refused == (9,)


def test_rerun_entrant_leaves_at_the_next_quiet_update():
    """No minimum stay: once its need is no finer than the coarse pack's and the best set does not hold it, a re-run
    entrant goes back (an idle fine pack then costs nothing). One the best set holds becomes an ordinary resident."""
    rungs = np.full(16, 2)
    rungs[3] = 4
    ask = np.zeros(16, dtype=bool)
    ask[3] = True
    clock = _clock(16, 1, leave_after=50, min_gain_num=100, min_gain_den=1)      # the gate never holds
    assert clock.update(rungs, _all(16), rerun=ask).enter == ((3, 0),)
    quiet = np.full(16, 2)
    assert clock.update(quiet, _all(16)).leave == ((0, 3),)
    assert not clock.resident.any()


def test_enter_now_takes_free_slots_or_nobody():
    clock = _clock(16, 2, leave_after=50, min_gain_num=100, min_gain_den=1)
    assert clock.enter_now([5, 9]).enter == ((5, 0), (9, 1)) and clock.resident[[5, 9]].all()
    assert clock.enter_now([1]) == clock.enter_now([1]) and clock.enter_now([1]).refused == (1,)    # full: nobody moves
    assert clock.resident.sum() == 2
    clock.release(5)
    assert clock.enter_now([9]).refused == (9,)                                  # already resident
    assert clock.enter_now([1, 2]).refused == (1, 2)                             # one free slot for two: none enters
    assert clock.enter_now([1]).enter == ((1, 0),)
    assert clock.update(np.full(16, 2), _all(16)).leave == ((0, 1), (1, 9))      # quiet entrants leave at once


def test_state_round_trip_with_the_gate_counter_and_rerun_asks():
    rng = np.random.default_rng(8)
    seq = [(rng.integers(2, 9, 24), rng.random(24) > 0.1, rng.random(24) > 0.9) for _ in range(40)]
    a = _clock(24, 3, leave_after=2, fine_cost_num=3, fine_cost_den=2)
    for j, ok, ask in seq[:15]:
        a.update(j, ok, ask)
    b = _clock(24, 3, leave_after=2, fine_cost_num=3, fine_cost_den=2)
    b.load_state(a.state())
    for j, ok, ask in seq[15:]:
        assert a.update(j, ok, ask) == b.update(j, ok, ask)
    for key, val in a.state().items():
        np.testing.assert_array_equal(val, b.state()[key])
