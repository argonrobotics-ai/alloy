# SPDX-License-Identifier: Apache-2.0
"""WP4 item (a) for SCR-WP9-ROD F3: the proxy pipelines' sticky contact matching is cleared whenever worlds go back
(alloy/coupling/match_reset.py): sim.reset (all or masked), a rollback (rolled-back worlds; the kept ones at the next
period), a replayed period. The hook logic on stubs, then the real sticky pipeline of a welded grip on a rod (CPU)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.coupling.match_reset import MatchResetMixin, world_bools
from alloy.coupling.proxy_contacts import ProxyContactPipeline


class _Base:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def reset(self, state, world_mask=None, flags=None) -> None:
        self.calls.append("reset")

    def on_rollback(self, world_mask=None) -> None:
        self.calls.append("rollback")

    def replay_period(self) -> None:
        self.calls.append("replay")

    def choose_clock(self, control_dt, last):
        self.calls.append("clock")
        return control_dt, {}


class _Factory:
    def __init__(self) -> None:
        self.got: list = []

    def reset_matching(self, worlds) -> int:
        self.got.append(None if worlds is None else worlds.tolist())
        return 1


class _Coupler(MatchResetMixin, _Base):
    def __init__(self) -> None:
        super().__init__()
        self.f = _Factory()

    def _contact_factories(self):
        return [self.f]


def test_a_rollback_clears_its_worlds_now_and_the_kept_ones_after_the_rerun() -> None:
    c = _Coupler()
    c.on_rollback(np.array([1, 0, 1], dtype=np.int32))
    assert c.f.got == [[True, False, True]] and c.calls == ["rollback"]
    c.choose_clock(0.01, None)                                 # the re-run: kept worlds not restored yet
    assert c.f.got == [[True, False, True]]
    c.choose_clock(0.01, None)                                 # the next period: Simulation restored them
    assert c.f.got[-1] == [False, True, False]
    c.choose_clock(0.01, None)
    assert len(c.f.got) == 2 and c.calls == ["rollback", "clock", "clock", "clock"]


def test_nested_rollbacks_clear_every_world_either_kept_and_a_whole_rollback_keeps_none() -> None:
    c = _Coupler()
    c.on_rollback(np.array([True, False, False]))
    c.choose_clock(0.01, None)
    c.on_rollback(np.array([False, True, False]))            # the re-run rolled back too
    c.choose_clock(0.01, None)
    c.choose_clock(0.01, None)
    assert c.f.got == [[True, False, False], [False, True, False], [True, True, True]]
    c = _Coupler()
    c.on_rollback(None)
    c.choose_clock(0.01, None)
    c.choose_clock(0.01, None)
    assert c.f.got == [None]


def test_reset_and_replay_clear_and_an_empty_mask_clears_nothing() -> None:
    c = _Coupler()
    c.reset(None, world_mask=np.array([0, 1]))
    c.replay_period()
    assert c.f.got == [[False, True], None] and c.calls == ["reset", "replay"]
    assert c.reset_contact_matching(np.zeros(2, bool)) == 0 and len(c.f.got) == 2
    assert world_bools(None) is None and world_bools([2, 0]).tolist() == [True, False]


class _Pipe:
    def __init__(self, mode: str) -> None:
        self.contact_matching, self.masks = mode, []

    def reset_contact_matching(self, mask) -> None:
        self.masks.append(None if mask is None else mask.numpy().tolist())


def test_the_factory_clears_only_pipelines_that_match_with_newtons_mask() -> None:
    f = ProxyContactPipeline()
    view = SimpleNamespace(world_count=3, device="cpu")
    sticky, plain = _Pipe("sticky"), _Pipe("disabled")
    f.built = [(view, sticky, None, None), (view, plain, None, None)]
    assert f.reset_matching(np.array([True, False, True])) == 1
    assert sticky.masks == [[True, False, True, False]] and plain.masks == []     # the global entry stays unset
    assert f.reset_matching(np.array([False, True, False, True])) == 1            # Newton's own mask: cut to 3
    assert sticky.masks[-1] == [False, True, False, False]
    f.reset_matching(None)
    assert sticky.masks[-1] is None
    with pytest.raises(ValueError):
        f.reset_matching(np.array([True]))


def _grip(worlds: int, welded: bool = True):
    """WP9's R4 rig (tests/api/test_rods_grip.py): a clamped cable's end held on a MuJoCo wrist, by pads welded to it
    or by sliding pads that close on the cable at period 2 (a force jump: the period re-runs)."""
    import alloy
    from tests.api.test_rods_grip import L, N, R, Z, _wrist

    sc = alloy.Scene(worlds=worlds, dt=0.01)
    sc.add_rod("cable", length=L, radius=R, pos=(0.0, 0.0, Z), material="cable_usb_composite", segments=N,
               clamp="start")
    sc.add_robot("hand", mjcf=_wrist(welded))
    return alloy.build(sc, device="cpu")


def _run(sim, periods: int, targets=None) -> np.ndarray:
    """(periods, worlds, points, 3): the cable after each period; ``targets(k)`` set before period k."""
    out = []
    for k in range(periods):
        if targets is not None:
            sim.set_targets(targets(k))
        sim.step(1)
        out.append(sim.state(["cable"])["cable.points"].copy())
    return np.stack(out)


def _hold(k: int) -> dict:
    return {"roll": 0.05}


def _matchers(sim) -> list:
    return [p._contact_matcher for f in sim.engine.stepper._contact_factories() for _, p, _, _ in f.built
            if getattr(p, "contact_matching", "disabled") != "disabled"]


@pytest.mark.timeout(900)
def test_the_rod_grips_sticky_matcher_is_cleared_by_reset_and_rollback(monkeypatch) -> None:
    """2 worlds: after 2 periods the matcher holds contacts; a masked reset of world 1 marks world 1 only; a full
    reset empties it; a per-world rollback marks the rolled-back world."""
    import warp as wp

    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")                # the CPU route as shipped (the suite forces Warp)
    sim = _grip(2)
    try:
        (m,) = _matchers(sim)
        sim.set_targets({"roll": 0.05})
        sim.step(2)
        assert int(m._prev_count.numpy()[0]) > 0
        sim.engine.reset(wp.array([0, 1], dtype=wp.int32))       # the engine's masked reset (envs use it)
        assert m._reset_world_mask.numpy().tolist() == [False, True, False]
        assert int(m._prev_count.numpy()[0]) > 0                 # world 0 keeps its anchors
        sim.reset()
        assert int(m._prev_count.numpy()[0]) == 0
        sim.step(1)
        sim.engine.stepper.on_rollback(np.array([1, 0], dtype=np.int32))
        assert m._reset_world_mask.numpy().tolist() == [True, False, False]
    finally:
        sim.close()


@pytest.mark.timeout(900)
@pytest.mark.parametrize("sticky", [True, False])
def test_a_reset_rod_grip_episode_replays_the_fresh_one_bit_for_bit(monkeypatch, sticky) -> None:
    """A fresh episode F, reset, A, reset, B: A == F and B == A, bit for bit, sticky on and off (the coordinator's
    18:13Z test, made exact by item (b): before reset_worlds, A missed F by 2.35e-5 m sticky on and 3.06e-5 m off, the
    rod's VBD history going on across the reset)."""
    from alloy.integration import stepper

    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    if not sticky:
        monkeypatch.setattr(stepper, "_steps_a_rod", lambda model, dst: False)
    sim = _grip(1)
    try:
        assert len(_matchers(sim)) == int(sticky)

        f = _run(sim, 4, _hold)
        sim.reset()
        a = _run(sim, 4, _hold)
        sim.reset()
        b = _run(sim, 4, _hold)
        assert np.array_equal(a, f), float(np.abs(a - f).max())
        assert np.array_equal(a, b), float(np.abs(a - b).max())
        assert sim.check()["ok"]
    finally:
        sim.close()


class _Memory:
    def __init__(self, w: int) -> None:
        self.last = np.arange(w, dtype=np.int64)


class _VBD:
    """Records SolverVBD.reset calls (isinstance passes: the class is patched in for the test)."""

    def __init__(self) -> None:
        self.calls: list = []
        self.model = SimpleNamespace(world_count=3)
        self.device = "cpu"

    def reset(self, state, world_mask=None, flags=None) -> None:
        self.calls.append((state, None if world_mask is None else world_mask.numpy().tolist(), flags))


def test_reset_worlds_cold_starts_vbd_history_and_forgets_the_cooldowns_of_those_worlds(monkeypatch) -> None:
    from alloy.coupling import match_reset

    monkeypatch.setattr(match_reset, "SolverVBD", _VBD)
    c = _Coupler()
    vbd, other = _VBD(), SimpleNamespace(reset=lambda *a, **k: pytest.fail("a non-VBD entry was reset"))
    c._entries = {"rod": SimpleNamespace(solver=vbd, state_0="s0"), "mujoco": SimpleNamespace(solver=other)}
    c._onset, c._ladder = _Memory(3), _Memory(3)
    c.reset_worlds(np.array([0, 1, 0]))
    assert c.f.got == [[False, True, False]] and vbd.calls == [("s0", [False, True, False, False], 0)]
    assert c._onset.last.tolist() == [0, match_reset.NEVER, 2] and c._ladder.last.tolist() == [0, match_reset.NEVER, 2]
    c.reset_worlds(None)
    assert c.f.got[-1] is None and vbd.calls[-1] == ("s0", None, 0)
    assert (c._onset.last == match_reset.NEVER).all() and (c._ladder.last == match_reset.NEVER).all()
    c.reset_worlds(np.zeros(3, bool))                                        # nothing selected: nothing touched
    assert len(vbd.calls) == 2 and len(c.f.got) == 2


def _episode(k: int) -> dict:
    """The probe's episode: pads close at period 0, the wrist rolls to PHI over periods 10-25."""
    from tests.api.test_rods_grip import PHI

    return {"lf": 0.02, "rf": 0.02, "roll": PHI * min(max(k - 9, 0) / 15.0, 1.0)}


@pytest.mark.law
@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_reset_episodes_of_a_grip_whose_first_contact_is_a_force_jump_replay_the_fresh_one(monkeypatch) -> None:
    """WP9's sliding pads close on the cable (a force-jump re-run at their first contact, period 2) and the wrist rolls
    to 0.3 rad: 30 periods fresh, then reset and again, twice: episodes 2 and 3 equal episode 1 bit for bit. Before
    reset_worlds the first episode's re-run left the second's first contact inside its cooldown (no re-run, 0.11 m
    off by period 2), and the rod's VBD history went on across the reset."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    sim = _grip(1, welded=False)
    try:
        f = _run(sim, 30, _episode)
        assert sim.engine.rollbacks > 0                              # the force jump re-ran a period
        for _ in range(2):
            sim.reset()
            a = _run(sim, 30, _episode)
            assert np.array_equal(a, f), float(np.abs(a - f).max())
        assert sim.check()["ok"]
    finally:
        sim.close()


@pytest.mark.law
@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("welded", [True, False])
def test_a_masked_reset_replays_its_world_and_leaves_the_kept_one_untouched(monkeypatch, welded) -> None:
    """2 worlds, 8 periods fresh (R); after a full reset (bitwise, above), 4 periods, a masked reset of world 1 and 4
    more. World 1 replays R's first 4 periods and world 0 goes on as in R, bit for bit. With sliding pads, world 1's
    first contact after the reset (period 6) re-runs that period for both worlds, world 0 taking back its first run
    (per-world rollback): bitwise through period 6. From period 7 world 0 is 1.7 um off R: a kept world's contact
    anchors are cleared after another world's re-run (item (a); keeping the re-run's anchors puts it 0.72 mm off).
    That comes from any per-world rollback, reset or not; a bitwise kept world needs its anchors and VBD history
    taken back with its state (gap #10b P1, [internal note] notes/wp4-perworld-clock/DESIGN.md)."""
    import warp as wp

    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    sim = _grip(2, welded=welded)
    hold = _hold if welded else _episode                         # constant over the 8 periods either way
    try:
        ref = _run(sim, 8, hold)
        sim.reset()
        head = _run(sim, 4, hold)
        sim.engine.reset(wp.array([0, 1], dtype=wp.int32))
        tail = _run(sim, 4, hold)
        assert np.array_equal(head, ref[:4]), float(np.abs(head - ref[:4]).max())
        exact = 4 if welded else 3                               # sliding: through the re-run of period 6
        assert np.array_equal(tail[:exact, 0], ref[4:4 + exact, 0]), float(np.abs(tail[:, 0] - ref[4:, 0]).max())
        assert np.abs(tail[exact:, 0] - ref[4 + exact:, 0]).max(initial=0.0) < 1.0e-5
        assert welded or sim.engine.rollbacks > 0                   # the sliding pads' re-runs
        assert np.array_equal(tail[:, 1], ref[:4, 1]), float(np.abs(tail[:, 1] - ref[:4, 1]).max())
        assert sim.check()["ok"]
    finally:
        sim.close()
