# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can: the books behind ``no_simulation_fault``, closed per world against the start's (pure functions
of ``SurfaceFlow.read()`` and the network's species).

GA closes the whole system (``task.py:66-71``: relative, relative_surf, relative_solids). Here, each against the start:
- water: the film's ledger (to the larger of what came in and the start film), DC5's supply cross ledger
  (``paint_supply.supply_drift``, for worlds whose hose ran), and the cavity tank's water: what it holds against
  pooled - withdrawn + to_tank + weir + lump_pool + what its pour edge carried (``tank_book``);
- species (detergent, soil), in kg:
  - the film's species ledger;
  - the network's (``net_book``): its tanks + what it booked out of them less in + what its nozzles' water carries
    in flight (``in_flight``: booked at the next exchange);
  - the whole system's (``system_book``): adhered soil + film + adsorbed + tanks + what left the film other than into
    tanks + what left the network into nothing + in flight + what the film's writers added and the film has not taken
    in yet (``with_pending``) - what drops took back (``dropped_back``). This also closes the hand-offs between the
    books (the soil law's removal into the film or a pool's tank, the film's drains into tanks, a pour mid-flight);
  - the soil stage's (adhered + removed = start).
  A species is closed when its largest drift is under 1e-9 of its start or under 1e-12 kg (``rinse_score``), each
  species on its own, as GA's OR.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alloy.tasks.paint_supply import supply_drift
from alloy.tasks.rinse_score import LEDGER_REL, SPECIES_ABS_KG

NET_IN = ("tanks", "lump_pool")         # supply species columns into the network's tanks
NET_OUT = ("withdrawn", "sent", "lost", "overspray", "mist", "caught", "lumped")   # out of them (to_tank moves
                                                                                 # species between tanks)
GONE = ("lost", "overspray", "mist", "caught", "lumped")    # out of the network into nothing (no chart, no pool)
TANK_IN = ("pooled", "to_tank", "weir", "lump_pool")        # supply water columns into tanks
SOIL = 1                                                    # the soil's column in the film's species (soap, soil)


def net_book(out: dict, net_species: np.ndarray) -> np.ndarray:
    """``[W, K]`` the network's species book (DC5's supply species ledger): what its tanks hold, plus what the
    nozzles and withdrawal films took out of them (booked, and in flight: ``in_flight``), less what drains and pools
    put in. Constant while it closes."""
    moved = sum(out[f"supply_species_{c}"] for c in NET_OUT) - sum(out[f"supply_species_{c}"] for c in NET_IN)
    return net_species.sum(axis=1) + moved + in_flight(out)


def in_flight(out: dict) -> np.ndarray:
    """``[W, K]`` kg in the nozzles' water in flight: out of the tanks, booked as sent at the next exchange (DC5's
    ``species_in_flight``, 2101b399f; a pour read mid-dump left the books open by 7.1e-8 kg of soap without it). A read
    without it raises: a silent 0 would open the books at every read mid-pour (DC5's review)."""
    if "supply_species_in_flight" not in out:
        raise KeyError("supply_species_in_flight: the flow's read has no species in flight (DC5's supply read, "
                       "2101b399f); the species books need it")
    return np.asarray(out["supply_species_in_flight"])


def tank_book(out: dict, tank: int | Sequence[int], pour: int | Sequence[int]) -> np.ndarray:
    """``[W]`` m³ the tanks hold less what came in and went out: the supply's pooled - withdrawn + to_tank + weir +
    lump_pool (summed over the network's tanks, so ``tank`` lists them all) and the ledgers of the edges into them
    (``pour``: every edge whose node is one of the tanks)."""
    came = sum(out[f"supply_{c}"] for c in TANK_IN) - out["supply_withdrawn"]
    held = out["supply_amount"][:, np.atleast_1d(tank)].sum(axis=1)
    return held - came - out["supply_edge_ledger"][:, np.atleast_1d(pour)].sum(axis=1)


def dropped_back(out: dict) -> np.ndarray | float:
    """``[W, K]`` kg of species that drops took back: landed on a chart (the film's again, as ``species_supplied``)
    or into a pool's tank (the network's, through ``pool_add_sp``); the drop stage's ``drops_species_landed`` and
    ``drops_species_tanks``, 0 without one. Film species that leave with drops are in ``species_out`` too, so the
    system book subtracts these (WP9's note: WP15 found the double count reviewing wp9-dropsp)."""
    return np.asarray(out.get("drops_species_landed", 0.0)) + np.asarray(out.get("drops_species_tanks", 0.0))


def with_pending(out: dict, area: np.ndarray) -> dict:
    """``out`` with ``film_species_pending`` [W, K] kg: what the film's writers have added this step and the film
    has not taken in yet (``src_species``: the soil law's removal, a jet's or a squeeze's species, a landed drop's).
    It is out of its holder's book already, so the system book counts it (DC5's review of the can's stage 2: 1.1e-6 kg
    of soil mid-spray, 12.7 mg in one period of a turn)."""
    src = out.get("src_species")
    out["film_species_pending"] = 0.0 if src is None else (src * area[None, :, None]).sum(axis=1)
    return out


def system_book(out: dict, net_species: np.ndarray) -> np.ndarray:
    """``[W, K]`` kg of each species anywhere or gone: adhered soil, film, walls, tanks, what left the film other
    than into tanks, and what left the network into nothing (module doc)."""
    total = out["species_mass"] + net_species.sum(axis=1) + out["species_out"].sum(axis=1)
    total = total - out["supply_species_tanks"] + sum(out[f"supply_species_{c}"] for c in GONE) - dropped_back(out)
    total = total + out.get("film_species_pending", 0.0) + in_flight(out)
    if "adsorbed_mass" in out:
        total = total + out["adsorbed_mass"]
    total = total.copy()
    total[:, SOIL] += out["soil_mass"]
    return total


def books(out: dict, net_species: np.ndarray, tank: int | Sequence[int], pour: int | Sequence[int]) -> dict:
    """Every book's value now; later readouts close against the start's (``set_depth`` and ``set_species`` leave the
    film's books at the start film, not at 0)."""
    return {"water": np.asarray(out["ledger"], float).copy(), "film_m3": np.asarray(out["film_volume"], float).copy(),
            "tank": tank_book(out, tank, pour), "species": np.asarray(out["species_ledger"], float).copy(),
            "net": net_book(out, net_species), "system": system_book(out, net_species)}


def closed(drift_kg: np.ndarray, start_kg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``[W]`` (relative, absolute) species drift for ``rinse_score.ledger_closed``, from ``[W, K]`` drifts against
    ``[W, K]`` starts: a species closed by either test counts 0 in the other, so the gate ORs per species."""
    rel = drift_kg / np.maximum(start_kg, 1e-30)
    return (np.where(drift_kg < SPECIES_ABS_KG, 0.0, rel).max(axis=1),
            np.where(rel < LEDGER_REL, 0.0, drift_kg).max(axis=1))


def ledgers(out: dict, net_species: np.ndarray, ref: dict, tank: int | Sequence[int],
            pour: int | Sequence[int]) -> tuple[np.ndarray, ...]:
    """``[W]`` water drift (relative), species drift (relative, kg) since ``ref`` (``books`` at the start, with
    ``start_kg`` [W, K]: the detergent on and in the can and the soil's start load) (module doc)."""
    W = out["h"].shape[0]
    put_in = np.maximum(out["supplied_volume"] + out["pending_volume"], ref["film_m3"])
    water = np.maximum(np.abs(out["ledger"] - ref["water"]),
                       np.abs(tank_book(out, tank, pour) - ref["tank"])) / np.maximum(put_in, 1e-30)
    ran = np.asarray(out["supply_sent"]) > 0.0
    water = np.maximum(water, [supply_drift(out, w) if ran[w] else 0.0 for w in range(W)])
    drift = np.maximum.reduce([np.abs(out["species_ledger"] - ref["species"]),
                               np.abs(net_book(out, net_species) - ref["net"]),
                               np.abs(system_book(out, net_species) - ref["system"])])
    drift[:, SOIL] = np.maximum(drift[:, SOIL],
                                np.abs(out["soil_mass"] + out["soil_removed_mass"] - out["soil_start_mass"]))
    rel, kg = closed(drift, ref["start_kg"])
    return water, rel, kg


__all__ = ["GONE", "NET_IN", "NET_OUT", "TANK_IN", "books", "closed", "dropped_back", "in_flight", "ledgers",
           "net_book", "system_book", "tank_book", "with_pending"]
