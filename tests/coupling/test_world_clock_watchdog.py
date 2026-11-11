# SPDX-License-Identifier: Apache-2.0
"""#10b P1 (rerun_worlds): the energy watchdog's reference rows move with a world (alloy.coupling.fine_rerun).

A world's watch levels are judged against its own reference: E_scale (the trip thresholds are fractions of it), the
datum z_ref (its energy is measured about it), m_tot and L_char (what E_scale is made of), its column of the 1 s
window ring, and the pack's elapsed time (the warn allowance). They are set from the world's start state and are not
in ``state_arrays``; a fine pack's slot holds the start of the world it was built as. With per-world starts
(``pos=(worlds, 3)``) the two differ.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

ROWS = ("e_scale", "z_ref", "m_tot", "l_char")
E, SLOT = 1, 0            # world 1 re-runs in the fine pack's only slot, built as world 0 (the one that starts higher)


def _scene(worlds: int, drop: float):
    """test_fine_rerun's 0.2 kg box resting on a sheet on a table, per world; world 0's box starts ``drop`` higher,
    so its extent, start energy and E_scale are larger than the other worlds'."""
    import alloy

    mass = 0.2
    side = (mass / 8000.0) ** (1.0 / 3.0)
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.4, 0.4, 0.1), pos=(0, 0, 0.05), static=True, friction=0.5)
    scene.add_cloth("cloth", size=(0.2, 0.2), pos=(0, 0, 0.102), resolution=(9, 9), mass=0.03, thickness=0.004,
                    self_contact=False)
    z = 0.104 + 0.001 + side / 2 + np.where(np.arange(worlds) == 0, drop, 0.0)
    scene.add_box("box", size=(side,) * 3, pos=np.stack([np.zeros(worlds), np.zeros(worlds), z], 1), mass=mass)
    return alloy.build(scene, device="cpu")


def _rows(stepper) -> dict[str, np.ndarray]:
    wd = stepper._watchdog
    out = {n: getattr(wd, n).numpy().copy() for n in ROWS}
    out["elapsed_ns"] = wd.elapsed_ns.numpy().copy()
    out["ring"] = None if wd.ring is None else wd.ring.numpy().copy()
    out["head"] = wd.head
    return out


def _run(fine: bool, drop: float = 0.5, trip: int = 1, periods: int = 3):
    """World ``E`` trips once in period ``trip`` (the first attempt's report says it escalated). Returns each period's
    state, report rows and force-jump peaks, the watchdog rows of both packs as the fine pack starts its re-run, and
    the sim."""
    from alloy.coupling.capture_passes import PER_WORLD_REPORT_FIELDS
    from alloy.coupling.fine_rerun import FineRerun
    from alloy.coupling.types import WatchState

    sim = _scene(3, drop)
    eng, st = sim.engine, sim.engine.stepper
    seen: dict[str, dict] = {}
    if fine:
        eng.fine_rerun = fr = FineRerun(lambda n: _scene(n, drop), 1)
        pack = fr.pack()
        step = pack.step_period

        def watched(*a, **k):
            seen["coarse"], seen["fine"] = _rows(st), _rows(pack.stepper)
            return step(*a, **k)

        pack.step_period = watched
    report, done = st.report, {}

    def tripped():
        rep = report()
        if eng.periods == trip and not done:
            done[trip] = True
            ws = np.asarray(rep.watch_state).copy()
            ws[E] = int(WatchState.ESCALATED)
            rep = dataclasses.replace(rep, watch_state=ws)
        return rep

    st.report = tripped
    out = []
    for _ in range(periods):
        sim.step(1)
        rep = eng._last_coupling
        s = sim.state(["box", "cloth"])
        out.append({"state": {k: np.array(v, copy=True) for k, v in s.items() if k != "t"},
                    "rows": {n: np.array(getattr(rep, n)) for n in PER_WORLD_REPORT_FIELDS},
                    "peak": np.array(rep.notes["exchange_peak"]), "ledger": st.energy_ledger(), "after": _rows(st)})
    return out, seen, sim


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_a_moved_world_keeps_its_own_watchdog_reference() -> None:
    """Law (CPU). Worlds start at different heights, so their watchdog references differ (checked: E_scale and L_char
    of world 0 against world 1). World 1 trips and re-runs in the fine pack's slot 0, built as world 0.
    1. While it is there, the slot's E_scale, z_ref, m_tot and L_char, its window-ring column and the pack's elapsed
       time and ring head are world 1's, bit for bit.
    2. Every period then ends as today's whole re-run, bit for bit: every world's state, the report's per-world rows
       (watch state, interface work, energy), each world's largest single-exchange injection as a fraction of its
       E_scale (the force-jump trigger's input), and the watchdog's ledger, ring and elapsed time afterwards (the
       ledger's last energy and gain: the re-run world's; a kept world's are its first run's here, and today those of
       a re-run whose state it does not keep)."""
    base, _, a = _run(False)
    got, seen, b = _run(True)
    try:
        fr = b.engine.fine_rerun
        assert fr.reruns == 1 and fr.fallbacks == {}, fr.fallbacks
        coarse, fine = seen["coarse"], seen["fine"]
        assert coarse["e_scale"][0] > 1.5 * coarse["e_scale"][E] and coarse["l_char"][0] > coarse["l_char"][E]
        for n in ROWS:
            assert fine[n][SLOT] == coarse[n][E], (n, fine[n][SLOT], coarse[n][E], coarse[n][0])
        assert fine["elapsed_ns"][0] == coarse["elapsed_ns"][0] and fine["head"] == coarse["head"]
        assert fine["ring"] is not None and np.array_equal(fine["ring"][:, SLOT], coarse["ring"][:, E])
        for p, (x, y) in enumerate(zip(base, got, strict=True)):
            for k in x["state"]:
                np.testing.assert_array_equal(x["state"][k], y["state"][k], err_msg=f"period {p} {k}")
            for n in x["rows"]:
                np.testing.assert_array_equal(x["rows"][n], y["rows"][n], err_msg=f"period {p} report {n}")
            np.testing.assert_array_equal(x["peak"], y["peak"], err_msg=f"period {p} exchange_peak")
            for n in x["ledger"]:              # energy, gain: the last exchange's read, the re-run world's here; today
                w = [E] if n in ("energy", "gain") else slice(None)    # a kept world's is its discarded re-run's
                np.testing.assert_array_equal(x["ledger"][n][w], y["ledger"][n][w], err_msg=f"period {p} ledger {n}")
            for n in ("ring", "elapsed_ns", "head"):
                np.testing.assert_array_equal(x["after"][n], y["after"][n], err_msg=f"period {p} watchdog {n}")
    finally:
        a.close()
        b.engine.fine_rerun.close()
        b.close()


def _packs():
    """A 3-world coarse pack and a 1-slot fine pack (no stepping), the coarse watchdog's reference rows, ring (6
    periods), head and elapsed time filled with distinct values per world."""
    from alloy.coupling.fine_rerun import _Rows

    coarse, fine = _scene(3, 0.5), _scene(1, 0.5)
    wc, wf = coarse.engine.stepper._watchdog, fine.engine.stepper._watchdog
    for i, n in enumerate(ROWS):
        getattr(wc, n).assign(np.array([10.0, 20.0, 30.0]) * (i + 1) + 0.25)
    wc.ring_for(1.0 / 6)
    wc.ring.assign(np.arange(wc.ring.shape[0] * 3, dtype=np.float64).reshape(-1, 3) + 0.5)
    wc.head, wc.has_reference = 4, True
    wc.elapsed_ns.assign(np.array([123456789], dtype=np.int64))
    return coarse, fine, wc, wf, _Rows(np.array([E]), np.array([SLOT]), 3, 1)


def test_a_moved_world_carries_its_watchdog_rows_to_the_slot_and_back() -> None:
    """Fast law (CPU, no stepping). The coarse watchdog holds distinct values per world; world 1 moves to slot 0:
    the slot's reference rows, ring column, head and elapsed time are world 1's, and a second move writes the ring in
    place. Coming back: the ring columns, elapsed time and the world's last energy and gain are the slot's (the
    other worlds' kept)."""
    from alloy.coupling.fine_rerun import _watch_back, _watch_reference

    coarse, fine, wc, wf, r = _packs()
    try:
        _watch_reference(fine.engine.stepper, coarse.engine.stepper, r)
        for n in ROWS:
            assert getattr(wf, n).numpy()[SLOT] == getattr(wc, n).numpy()[E], n
        assert np.array_equal(wf.ring.numpy()[:, SLOT], wc.ring.numpy()[:, E])
        assert wf.head == wc.head == 4 and wf.elapsed_ns.numpy()[0] == 123456789 and wf.has_reference
        ring = wf.ring
        wc.ring.assign(wc.ring.numpy() + 100.0)
        _watch_reference(fine.engine.stepper, coarse.engine.stepper, r)
        assert wf.ring is ring and np.array_equal(ring.numpy()[:, SLOT], wc.ring.numpy()[:, E])
        wf.ring.assign(wf.ring.numpy() * -1.0)           # the slot ran: ring, elapsed time, energy and gain moved
        wf.elapsed_ns.assign(np.array([987654321], dtype=np.int64))
        wf.energy.assign(np.array([7.5])), wf.gain.assign(np.array([-2.5]))
        wc.energy.assign(np.array([1.0, 2.0, 3.0])), wc.gain.assign(np.array([4.0, 5.0, 6.0]))
        before = wc.ring.numpy().copy()
        _watch_back(coarse.engine.stepper, fine.engine.stepper, r)
        assert np.array_equal(wc.ring.numpy()[:, E], wf.ring.numpy()[:, SLOT])
        assert np.array_equal(wc.ring.numpy()[:, [0, 2]], before[:, [0, 2]])
        assert wc.elapsed_ns.numpy()[0] == 987654321
        assert list(wc.energy.numpy()) == [1.0, 7.5, 3.0] and list(wc.gain.numpy()) == [4.0, -2.5, 6.0]
    finally:
        coarse.close()
        fine.close()
