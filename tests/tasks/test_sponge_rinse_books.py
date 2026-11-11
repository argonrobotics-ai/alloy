# SPDX-License-Identifier: Apache-2.0
"""rinse-sponge books and readout pieces on synthetic two-world reads (alloy/tasks/sponge_rinse_run.py): a drift in
one term of world 1 fails world 1 only; hand-offs between the holders stay closed (DC5's review)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import sponge_rinse_run as R
from alloy.tasks.rinse_flush_checks import quiet_books
from alloy.tasks.rinse_score import ledger_closed
from alloy.tasks.sponge_rinse_score import SpongeReadout

W, K, NODES, EDGES = 2, 3, 3, 7
TANKS, ALL = (0, 1, 2), tuple(range(EDGES))
SOAP0 = 44.5e-6


def _read() -> tuple[dict, dict, np.ndarray]:
    from alloy.plugins.surface_flow.supply import LEDGER  # noqa: PLC0415
    from alloy.plugins.surface_flow.supply_species import SP_LEDGER  # noqa: PLC0415

    z = np.zeros(W)
    out = {"h": np.zeros((W, 4)), "ledger": z.copy(), "film_volume": z.copy(), "supplied_volume": np.full(W, 4e-5),
           "pending_volume": z.copy(), "to_lumps_volume": z.copy(), "drops_lump_wet": z.copy(),
           "drops_landed": z.copy(), "supply_amount": np.full((W, NODES), 0.0707),
           "supply_edge_ledger": np.zeros((W, EDGES)), "species_ledger": np.zeros((W, K)),
           "species_mass": np.zeros((W, K)), "species_out": np.zeros((W, 4, K)), "drained_total_volume": z.copy(),
           "spilled_volume": z.copy(), "drops_to_tanks": z.copy(), "drops_species_landed": np.zeros((W, K)),
           "drops_species_tanks": np.zeros((W, K))}
    out |= {f"supply_{c}": z.copy() for c in LEDGER} | {f"supply_species_{c}": np.zeros((W, K)) for c in SP_LEDGER}
    out["supply_species_in_flight"] = np.zeros((W, K))
    out["supply_lump_pool"] = np.full(W, -2e-5)
    lumps = {"lump_ledger": np.full((W, 1), 59.3e-6), "lump_species_ledger": np.tile([[[SOAP0, 0.0, 0.0]]], (W, 1, 1)),
             "lump_species": np.tile([[[SOAP0, 0.0, 0.0]]], (W, 1, 1)), "lump_squeezed": np.full((W, 1), 4e-5),
             "lump_pooled": np.full((W, 1), 2e-5), "lump_pool_pending": np.zeros((W, 1)),
             "lump_uptake": np.zeros((W, 1)), "lump_water": np.full((W, 1), 39.3e-6),
             "lump_drip_tanks": np.zeros((W, 1)),
             "lump_species_drip_pending": np.zeros((W, 1, K)), "lump_species_dropped": np.zeros((W, 1, K))}
    return out, lumps, np.zeros((W, NODES, K))


def _ref(out: dict, lumps: dict, net: np.ndarray) -> dict:
    return {**R.books(out, lumps, net, TANKS, ALL), "start_kg": np.tile([SOAP0, 0.0, 0.0], (W, 1))}


def _closed(out: dict, lumps: dict, net: np.ndarray, ref: dict) -> list[bool]:
    water, rel, kg = R.ledgers(out, lumps, net, ref, TANKS, ALL)
    return [ledger_closed(float(water[w]), float(rel[w]), float(kg[w])) for w in range(W)]


def _moves() -> list[tuple[str, str, tuple, float]]:
    """(book, key, index in world 1, amount): one term of one book each."""
    from alloy.tasks.can_wash_books import GONE, NET_OUT  # noqa: PLC0415

    return [("out", "ledger", (1,), 1e-12), ("lumps", "lump_ledger", (1, 0), 1e-12),
            ("out", "supply_amount", (1, 2), 1e-9), ("out", "supply_edge_ledger", (1, 6), 1e-9),   # of 212 L
            ("out", "supply_lump_pool", (1,), 1e-12), ("lumps", "lump_pooled", (1, 0), 1e-12),
            ("lumps", "lump_squeezed", (1, 0), 1e-12), ("out", "supplied_volume", (1,), 1e-12),
            ("out", "to_lumps_volume", (1,), 1e-12), ("lumps", "lump_uptake", (1, 0), 1e-12),
            ("out", "species_ledger", (1, 0), 1e-8), ("lumps", "lump_species_ledger", (1, 0, 0), 1e-8),
            ("lumps", "lump_species", (1, 0, 1), 1e-8), ("out", "species_mass", (1, 2), 1e-8),
            ("out", "drops_to_tanks", (1,), 1e-12), ("lumps", "lump_drip_tanks", (1, 0), 1e-12),
            ("out", "drops_species_landed", (1, 0), 1e-8), ("out", "drops_species_tanks", (1, 0), 1e-8),
            ("lumps", "lump_species_drip_pending", (1, 0, 0), 1e-8), ("lumps", "lump_species_dropped", (1, 0, 0), 1e-8),
            *(("out", f"supply_species_{c}", (1, 0), 1e-8) for c in (*NET_OUT, *GONE))]


def test_every_book_closes_at_the_start():
    out, lumps, net = _read()
    assert _closed(out, lumps, net, _ref(out, lumps, net)) == [True, True]


@pytest.mark.parametrize("book, key, idx, amount", _moves())
def test_a_drift_in_one_term_fails_that_world_only(book, key, idx, amount):
    out, lumps, net = _read()
    ref = _ref(out, lumps, net)
    ({"out": out, "lumps": lumps}[book])[key][idx] += amount
    assert _closed(out, lumps, net, ref) == [True, False], key


def test_the_supply_cross_ledger_counts_for_worlds_whose_supply_ran():
    out, lumps, net = _read()
    ref = _ref(out, lumps, net)
    out["supply_sent"][:] = out["supplied_volume"]                # a stream sent what the film got (the squeeze's
    assert _closed(out, lumps, net, ref) == [True, True]         # share aside: none here)
    out["supply_sent"][1] += 1e-9
    assert _closed(out, lumps, net, ref) == [True, False]


def test_hand_offs_between_the_holders_keep_every_book_closed():
    """A swap: 5 mL from the rinse bowl into the pores, then a 9 mL squeeze onto the ledge and down its drain; 2 mL
    between bowls; 3 µg of soap from the pores into the rinse bowl, and 1 µg from the wash bowl into nothing (a
    nozzle's jet that hits nothing: ``lost``, one of ``GONE``)."""
    out, lumps, net = _read()
    ref = _ref(out, lumps, net)
    for w in range(W):
        lumps["lump_pooled"][w, 0] += 5e-6                        # bowl -> pores
        out["supply_lump_pool"][w] -= 5e-6
        out["supply_amount"][w, 1] -= 5e-6                        # the rinse bowl gives it (lump_pool, a tank column)
        lumps["lump_squeezed"][w, 0] += 9e-6                      # pores -> ledge -> drain, away
        out["supplied_volume"][w] += 9e-6
        out["supply_amount"][w, 0] -= 2e-6                        # 2 mL from the wash bowl to the sanitizer
        out["supply_amount"][w, 2] += 2e-6
        lumps["lump_species"][w, 0, 0] -= 3e-6                   # soap: pores -> rinse bowl, booked by the supply
        net[w, 1, 0] += 3e-6
        out["supply_species_lump_pool"][w, 0] += 3e-6
        net[w, 0, 0] -= 1e-6                                     # wash bowl -> nothing
        out["supply_species_lost"][w, 0] += 1e-6
    assert _closed(out, lumps, net, ref) == [True, True]
    out["supply_amount"][1, 0] -= 1e-9                           # world 1's wash bowl loses 1 µL
    assert _closed(out, lumps, net, ref) == [True, False]


def test_drops_and_drips_keep_every_book_closed():
    """The pores drip 2 µg of soap: it waits, then leaves as drops, 1 µg landing on the ledge and 0.5 µg in the rinse
    bowl (0.5 µg still in the air); a 3 µL drop and a 1 µL drip join the rinse bowl's water and a 2 µL drop lands on
    the ledge; a film drop carries 1 µg of the ledge's soap off and back onto it (wp9-dropsp: drops carry the film's
    species, in species_out)."""
    out, lumps, net = _read()
    ref = _ref(out, lumps, net)
    for w in range(W):
        lumps["lump_species"][w, 0, 0] -= 2e-6                   # pores -> drip, waiting
        lumps["lump_species_drip_pending"][w, 0, 0] += 2e-6
        lumps["lump_species_drip_pending"][w, 0, 0] -= 2e-6      # -> drops
        lumps["lump_species_dropped"][w, 0, 0] += 2e-6
        out["drops_species_landed"][w, 0] += 1e-6               # onto the ledge
        out["species_mass"][w, 0] += 1e-6
        out["drops_species_tanks"][w, 0] += 0.5e-6              # into the rinse bowl (pool_add_sp: lump_pool)
        net[w, 1, 0] += 0.5e-6
        out["supply_species_lump_pool"][w, 0] += 0.5e-6
        for v, key in ((3e-9, "drops"), (1e-9, "drip")):          # water into the rinse bowl, booked lump_pool
            out["supply_lump_pool"][w] += v
            out["supply_amount"][w, 1] += v
            if key == "drops":
                out["drops_to_tanks"][w] += v
            else:
                lumps["lump_drip_tanks"][w, 0] += v
        out["drops_landed"][w] += 2e-9                           # the pores' water onto the ledge as a drop
        out["supplied_volume"][w] += 2e-9
        out["species_out"][w, 2, 0] += 1e-6                     # a film drop off the ledge and back
        out["drops_species_landed"][w, 0] += 1e-6
    assert _closed(out, lumps, net, ref) == [True, True]
    out["drops_species_landed"][1, 0] += 1e-8                     # world 1 books 10 µg landed that never came
    assert _closed(out, lumps, net, ref) == [True, False]


def test_the_faucet_in_flight_and_a_squeeze_waiting_in_the_film_keep_the_books_closed():
    """The faucet's water carries 1 µg of the wash bowl's soap in flight (DC5's ``species_in_flight``); a squeeze's
    2 µg wait in the ledge's ``src_species`` for the film's next step (``with_pending``). Every book closes with the
    terms and fails without them (DC5's review: a read mid-squeeze would score None)."""
    from alloy.tasks.can_wash_books import with_pending  # noqa: PLC0415

    area = np.full(4, 0.01)
    out, lumps, net = _read()
    out["src_species"] = np.zeros((W, 4, K))
    ref = _ref(with_pending(out, area), lumps, net)
    net[:, 0, 0] -= 1e-6                                          # wash bowl -> the faucet's water, in flight
    out["supply_species_in_flight"][:, 0] += 1e-6
    lumps["lump_species"][:, 0, 0] -= 2e-6                       # pores -> the ledge's cells, waiting
    out["src_species"][:, 2, 0] += 2e-6 / area[2]
    assert _closed(with_pending(out, area), lumps, net, ref) == [True, True]
    out["supply_species_in_flight"][1, 0] = 0.0
    assert _closed(out, lumps, net, ref) == [True, False]
    out["supply_species_in_flight"][1, 0] = 1e-6
    out["src_species"][1, 2, 0] = 0.0
    assert _closed(with_pending(out, area), lumps, net, ref) == [True, False]


def test_intact_is_ga_s_drop_line():
    assert R.intact_of(np.array([[3.3, 0.66, 0.94], [3.3, 0.66, 0.249]])).tolist() == [True, False]


def test_the_settle_watches_the_pores_water():
    r = SpongeReadout(pore_soap_kg=np.zeros(W), pore_m3=np.array([41.4e-6, 50.0e-6]), pos=np.zeros((W, 3)),
                      faucet_gpm=np.zeros(W), held=np.zeros(W, bool), intact=np.ones(W, bool), water_rel=np.zeros(W),
                      species_rel=np.zeros(W), species_kg=np.zeros(W), finite=np.ones(W, bool))
    out = {"film_volume": np.full(W, 1e-6), "drained_total_volume": np.zeros(W), "spilled_volume": np.zeros(W)}
    books, pores = R.quiet_state(out, r)
    assert np.allclose(books, quiet_books(out)) and pores == pytest.approx([41.4, 50.0])
