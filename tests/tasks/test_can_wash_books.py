# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can books and readout pieces on synthetic two-world reads (alloy/tasks/can_wash_books.py,
can_wash_run.py, can_wash_scene.py ``start``): a drift in one column of world 1 fails world 1 only (DC5's review)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.tasks import can_wash_books as B
from alloy.tasks.can_wash_run import finite, shelf_g, valve_distance
from alloy.tasks.can_wash_scene import CROP, DEBRIS_LOAD, DEBRIS_SHARE, FILM_H, WASH, langmuir
from alloy.tasks.rinse_flush_checks import RHO, Columns
from alloy.tasks.rinse_flush_policy import NOZZLE
from alloy.tasks.rinse_score import ledger_closed

W, K, NODES, EDGES, C = 2, 2, 2, 3, 6
TANK, POUR = 1, 2
SOAP0, SOIL0 = 50e-6, 300e-6                                       # kg at the start


def _read() -> dict:
    """A start read that every book closes on: the film's and the network's books at their start values."""
    z = np.zeros(W)
    out = {"h": np.zeros((W, C)), "ledger": np.full(W, 1e-4), "film_volume": np.full(W, 1e-4),
           "supplied_volume": z.copy(), "pending_volume": z.copy(), "supply_sent": z.copy(),
           "supply_amount": np.zeros((W, NODES)), "supply_edge_ledger": np.zeros((W, EDGES)),
           "species_ledger": np.tile([SOAP0, 0.0], (W, 1)), "species_mass": np.tile([SOAP0 - 1e-6, 0.0], (W, 1)),
           "adsorbed_mass": np.tile([1e-6, 0.0], (W, 1)), "species_out": np.zeros((W, 4, K)),
           "soil_mass": np.full(W, SOIL0), "soil_removed_mass": z.copy(), "soil_start_mass": np.full(W, SOIL0)}
    from alloy.plugins.surface_flow.supply import LEDGER  # noqa: PLC0415
    from alloy.plugins.surface_flow.supply_species import SP_LEDGER  # noqa: PLC0415

    out |= {f"supply_{c}": z.copy() for c in LEDGER} | {f"supply_species_{c}": np.zeros((W, K)) for c in SP_LEDGER}
    out["supply_species_in_flight"] = np.zeros((W, K))
    for c in ("supplied_volume", "pending_volume"):
        out[f"supply_{c}"] = z.copy()
    return out


def _ref(out: dict, net: np.ndarray) -> dict:
    return {**B.books(out, net, TANK, POUR), "start_kg": np.tile([SOAP0, SOIL0], (W, 1))}


def _closed(out: dict, net: np.ndarray, ref: dict) -> list[bool]:
    water, rel, kg = B.ledgers(out, net, ref, TANK, POUR)
    return [ledger_closed(float(water[w]), float(rel[w]), float(kg[w])) for w in range(W)]


def _moves() -> list[tuple[str, tuple, float]]:
    """(key, index in world 1, amount): one book's term each."""
    return [("ledger", (1,), 1e-12), ("species_ledger", (1, 0), 1e-8), ("species_ledger", (1, 1), 1e-8),
            ("species_mass", (1, 0), 1e-8), ("adsorbed_mass", (1, 0), 1e-8), ("species_out", (1, 0, 1), 1e-8),
            ("supply_species_tanks", (1, 0), 1e-8), ("supply_species_lump_pool", (1, 1), 1e-8),
            *((f"supply_species_{c}", (1, 0), 1e-8) for c in B.NET_OUT),
            ("supply_amount", (1, TANK), 1e-9), ("supply_pooled", (1,), 1e-9), ("supply_to_tank", (1,), 1e-9),
            ("supply_withdrawn", (1,), 1e-9), ("supply_edge_ledger", (1, POUR), 1e-9),
            ("soil_mass", (1,), 1e-8), ("soil_removed_mass", (1,), 1e-8)]


def test_every_book_closes_at_the_start():
    out, net = _read(), np.zeros((W, NODES, K))
    assert _closed(out, net, _ref(out, net)) == [True, True]


@pytest.mark.parametrize("key, idx, amount", _moves())
def test_a_drift_in_one_term_fails_that_world_only(key, idx, amount):
    out, net = _read(), np.zeros((W, NODES, K))
    ref = _ref(out, net)
    out[key][idx] += amount
    assert _closed(out, net, ref) == [True, False], key


def test_a_network_tank_drift_and_the_supply_cross_ledger_fail_too():
    out, net = _read(), np.zeros((W, NODES, K))
    ref = _ref(out, net)
    net[1, TANK, 0] += 1e-8                                        # soap appears in world 1's tank
    assert _closed(out, net, ref) == [True, False]
    net[1, TANK, 0] = 0.0
    out["supply_sent"][:] = 1e-3                                  # both hoses ran; world 1 sent water nowhere booked
    out["supplied_volume"][:] = 1e-3
    out["supply_sent"][1] += 1e-9
    assert _closed(out, net, ref) == [True, False]


def test_moves_between_books_keep_them_closed():
    """Soap pooled from the film into the tank, soil eroded under the pool into it, a pour sent onto the film."""
    out, net = _read(), np.zeros((W, NODES, K))
    ref = _ref(out, net)
    for w in range(W):
        out["species_mass"][w, 0] -= 2e-6                          # film -> tank (pooled)
        out["species_out"][w, 0, 0] += 2e-6
        out["supply_species_tanks"][w, 0] += 2e-6
        net[w, TANK, 0] += 2e-6
        out["soil_mass"][w] -= 5e-6                                 # adhered -> tank (flooded soil)
        out["soil_removed_mass"][w] += 5e-6
        net[w, TANK, 1] += 5e-6
        out["supply_species_tanks"][w, 1] += 5e-6
        out["species_out"][w, 0, 1] += 5e-6
        out["species_ledger"][w, 1] += 0.0
        out["supply_amount"][w, TANK] += 3e-3                     # 3 L pooled
        out["supply_pooled"][w] += 3e-3
    assert _closed(out, net, ref) == [True, True]


def test_soap_that_leaves_the_scene_is_booked_not_lost():
    """Soap off the foot's open edge (the film's species_out) and a pour's jet from the tank that hits nothing
    (``lost``) leave the scene; every book stays closed (DC5's mutation 15)."""
    out, net = _read(), np.zeros((W, NODES, K))
    net[:, TANK, 0] = 5e-6
    ref = _ref(out, net)
    out["species_mass"][:, 0] -= 3e-6                              # off the open edge
    out["species_out"][:, 2, 0] += 3e-6
    out["species_ledger"][:, 0] += 0.0
    net[:, TANK, 0] -= 2e-6                                        # poured, missed everything
    out["supply_species_lost"][:, 0] += 2e-6
    assert _closed(out, net, ref) == [True, True]
    out["supply_species_lost"][1, 0] -= 1e-8                      # world 1 loses track of 10 µg
    assert _closed(out, net, ref) == [True, False]


def test_drops_that_come_back_are_counted_once():
    """wp9-dropsp: film species that leave with drops are in species_out; a drop that lands puts them back in the film
    (drops_species_landed) and one into the pool puts them in its tank (drops_species_tanks, booked lump_pool): the
    system book subtracts both (WP9's note, ``dropped_back``)."""
    out, net = _read(), np.zeros((W, NODES, K))
    out["drops_species_landed"], out["drops_species_tanks"] = np.zeros((W, K)), np.zeros((W, K))
    ref = _ref(out, net)
    out["species_out"][:, 1, 0] += 3e-6                            # shed as drops; 2 µg land back on the film
    out["drops_species_landed"][:, 0] += 2e-6
    out["species_mass"][:, 0] -= 1e-6                              # the film: -3 + 2
    net[:, TANK, 0] += 1e-6                                        # and 1 µg falls into the pool
    out["supply_species_lump_pool"][:, 0] += 1e-6
    out["drops_species_tanks"][:, 0] += 1e-6
    assert _closed(out, net, ref) == [True, True]
    out["drops_species_landed"][1, 0] += 1e-8                     # world 1 books 10 µg landed that never came
    assert _closed(out, net, ref) == [True, False]


def test_a_pour_in_flight_and_soil_waiting_in_the_film_keep_the_books_closed():
    """Mid-pour, the tank is down by q·dt·c that the supply books next exchange (DC5's ``species_in_flight``); the
    soil law's removal waits in ``src_species`` for the film's next step (``with_pending``). Both books close with the
    terms and fail without them (DC5's review: a read mid-pour or mid-spray scored None)."""
    area = np.full(C, 0.01)
    out, net = _read(), np.zeros((W, NODES, K))
    out["src_species"] = np.zeros((W, C, K))
    ref = _ref(B.with_pending(out, area), net)
    net[:, TANK, 0] -= 2e-6                                        # into the pour's water, in flight
    out["supply_species_in_flight"][:, 0] += 2e-6
    out["soil_mass"] -= 3e-6                                       # removed, waiting for the film's step
    out["soil_removed_mass"] += 3e-6
    out["src_species"][:, 4, 1] += 3e-6 / area[4]
    assert _closed(B.with_pending(out, area), net, ref) == [True, True]
    out["supply_species_in_flight"][1, 0] = 0.0                     # world 1 forgets what flies
    assert _closed(out, net, ref) == [True, False]
    out["supply_species_in_flight"][1, 0] = 2e-6
    out["src_species"][1, 4, 1] = 0.0                               # world 1 forgets what waits
    assert _closed(B.with_pending(out, area), net, ref) == [True, False]
    with pytest.raises(KeyError, match="supply_species_in_flight"):
        B.net_book({k: v for k, v in out.items() if k != "supply_species_in_flight"}, net)


def test_the_settle_checks_quiet_from_their_2_s_then_every_second():
    from alloy.tasks.can_wash_run import settle_steps  # noqa: PLC0415
    from alloy.tasks.rinse_flush_checks import SETTLE_MIN, SETTLE_STEP  # noqa: PLC0415

    steps = settle_steps(30.0)
    assert steps[0] == SETTLE_MIN == 2.0 and set(steps[1:]) == {SETTLE_STEP} and sum(steps) == pytest.approx(30.0)
    assert settle_steps(2.5) == [2.0, 0.5] and settle_steps(0.04) == [0.04] and settle_steps(0.0) == [0.0]


def test_a_world_is_quiet_only_when_its_free_water_is_too():
    """Hose off, can still, the film's books steady: quiet unless the can's free water still changes by more than
    0.2 mL or 2 % per second (GA's task.py:843-844): world 0 drains 0.3 of 20 mL (quiet), world 1 0.3 of 5 mL (not)."""
    from alloy.tasks.can_wash_run import quiet_now  # noqa: PLC0415

    books = np.zeros((W, 2))
    prev, now = (books, np.array([20.0, 5.0])), (books, np.array([19.7, 4.7]))
    assert quiet_now(prev, now, np.zeros(W), np.zeros(W), 1.0).tolist() == [True, False]
    assert quiet_now(prev, now, np.zeros(W), np.zeros(W), 2.0).tolist() == [True, True]      # per second
    assert quiet_now(prev, (books + [[0.0, 2.0], [0.0, 0.0]], now[1]), np.zeros(W), np.zeros(W),
                     1.0).tolist() == [False, False]           # world 0's film still sheds 2 g/s


def test_species_close_on_their_own():
    """GA ORs per species. World 0: soap drifts 1e-11 kg of a 0.1 kg start (closed by the relative test) and soil
    1e-13 kg of a 1e-7 kg start (closed by the absolute one), so the world is closed, where the largest relative and
    the largest absolute drift taken across species would fail it. World 1: soap 1e-8 kg of 50 mg fails both."""
    drift = np.array([[1e-11, 1e-13], [1e-8, 0.0]])
    rel, kg = B.closed(drift, np.array([[0.1, 1e-7], [SOAP0, SOIL0]]))
    assert [ledger_closed(0.0, float(rel[w]), float(kg[w])) for w in range(W)] == [True, False]


def test_finite_counts_every_measure_and_the_overflow_bits():
    from tests.tasks.test_can_wash_score import _readout  # noqa: PLC0415

    r = _readout()
    two = type(r)(**{k: (np.repeat(v, 2) if isinstance(v, np.ndarray) else v) for k, v in vars(r).items()})
    assert finite(two, np.array([False, True])).tolist() == [True, False]
    two.free_ml[0] = np.nan
    assert finite(two, np.zeros(2, bool)).tolist() == [False, True]


def test_shelf_reads_each_world():
    out = {"h": np.zeros((W, 10)), "supplied": np.zeros((W, 10)), "supply_caught_by": np.zeros((W, 5))}
    out["supplied"][1, 8] = 2e-6                                  # 2 mL on world 1's shelf chart
    out["supply_caught_by"][0, 3] = 1e-6                          # 1 mL caught by world 0's post
    assert shelf_g(out, 6, Columns(outlet=1, shelf=(3,))) == pytest.approx([1e-6 * RHO * 1e3, 2e-6 * RHO * 1e3])


def test_the_valve_is_measured_at_its_body_not_its_nozzle():
    """Docked head down 0.06 m under the holster's centre: 0.06 m (parked); its nozzle would be 0.164 m."""
    h = np.asarray(CROP.bay.holster)
    q = np.array([[*h, 0.0, 0.0, 0.0, 1.0], [*(h - [0.0, 0.0, 0.06]), 0.0, 0.0, 0.0, 1.0]])
    d = valve_distance(CROP, q)
    assert d == pytest.approx([0.0, 0.06]) and 0.06 + NOZZLE > 0.15


class _Recorder:
    """A flow that records what ``CanWash.start`` writes (charts as built: floor, can, shelf)."""

    def __init__(self, task, worlds: int) -> None:
        from alloy.plugins.surface_flow.charts import revolution_geometry  # noqa: PLC0415

        prof = revolution_geometry(task.can.chart("can")).prof
        shapes = [task.bay.cells, (len(prof) - 1, task.can.n_theta), *[task.bay.shelf_cells()] * len(task.bay.levels)]
        starts = np.concatenate([[0], np.cumsum([a * b for a, b in shapes])])
        self.tables = SimpleNamespace(shapes=shapes, cells_of=lambda k: slice(starts[k], starts[k + 1]))
        self.area = np.concatenate([np.zeros(starts[1]), revolution_geometry(task.can.chart("can")).area])
        self.calls, self.worlds, self.C = {}, worlds, int(starts[-1])
        self.process = SimpleNamespace(set_soil=lambda v, chart: self.calls.__setitem__("soil", (np.array(v), chart)))

    def __getattr__(self, name):
        if name.startswith("set_"):
            return lambda *a, **k: self.calls.__setitem__(name if name != "set_species" else "soap", (a, k))
        raise AttributeError(name)

    def read(self) -> dict:
        return {"h": np.zeros((self.worlds, self.C))}


def test_start_writes_their_film_walls_and_debris():
    task, worlds = CROP, 16
    fl = _Recorder(task, worlds)
    made = task.start(fl, worlds, seed=3)
    s = made["strength"]
    assert s[0] == WASH and ((s[1:] >= 0.67 * WASH) & (s[1:] <= 1.67 * WASH)).all()
    (h,), kw = fl.calls["set_depth"]
    assert kw["chart"] == task.can_chart and np.allclose(h, FILM_H)
    (m, _), _ = fl.calls["soap"]
    assert np.allclose(m, FILM_H * s[:, None, None])
    (ads, _), _ = fl.calls["set_adsorbed"]
    assert np.allclose(ads, langmuir(s)[:, None, None])
    debris, chart = fl.calls["soil"]
    assert chart == task.can_chart
    flat, inner = debris.reshape(worlds, -1), task.regions()["inner"]
    assert (flat[:, ~inner] == 0.0).all()
    on = flat[:, inner] > 0.0
    assert on.mean() == pytest.approx(DEBRIS_SHARE, abs=0.04)
    assert flat[:, inner][on].min() >= 0.2 * DEBRIS_LOAD and flat.max() <= DEBRIS_LOAD
    area = fl.area[fl.tables.cells_of(task.can_chart)]
    expect = DEBRIS_SHARE * DEBRIS_LOAD * 0.6 * area[inner].sum()   # their 0.5e-3 x 2 x U(0.2, 1) on half
    assert (flat * area).sum(axis=1).mean() == pytest.approx(expect, rel=0.05)
