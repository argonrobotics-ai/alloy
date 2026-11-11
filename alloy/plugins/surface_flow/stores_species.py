# SPDX-License-Identifier: Apache-2.0
"""Species in the tools' stores and trays (v2): each bin and each tray carries its own species [kg], so a brush
dipped in a blue well lays blue, one that then dips in red lays the mix, and film a store takes back mixes in.

- A tray starts at ``Tray(species=)`` x its volume; a store's bins at ``Store(species=)`` x their start load.
- Pick-up moves the tray's concentration with the volume taken (``picked``).
- The nip, a metering brush and a blade's fill lay at the bin's concentration as the exchange reaches them and take
  film back at the cell's (``taken``); the film books them as before (``species_supplied`` / ``species_out``).
- Drips, releases and a lifted gob leave at the bin's concentration (``given``), through DC5's channel with a supply.
- A full blade's overflow (stores_lift.py) leaves at the cut's concentration: ``given``, then ``tool_overflow_sp``.

Books, per world and species: ``Σ bins + given - taken - picked`` is the stores' start species, and
``Σ trays + picked`` the trays'. v1 kept each store at its own fixed concentration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.cell_kernels import MAX_SPECIES, Species, vec4d


@wp.struct
class StoreSpecies:
    load: wp.array3d[wp.float64]     # [W, bins, K] kg in each bin's load
    tray: wp.array3d[wp.float64]     # [W, trays, K] kg in each tray
    taken: wp.array3d[wp.float64]    # [W, n, K] kg each store took from the film
    picked: wp.array3d[wp.float64]   # [W, n, K] kg it picked up from trays
    given: wp.array3d[wp.float64]    # [W, n, K] kg it laid, let go, or sent on as overflow


def species_pairs(label: str, species: Any) -> tuple:
    """``{name: kg/m³}`` as sorted, checked pairs (hashable)."""
    pairs = tuple(sorted(dict(species or {}).items()))
    for name, v in pairs:
        if not (np.isfinite(v) and v >= 0.0):
            raise PluginError(f"{label}: species {name!r} must be finite and >= 0 kg/m³, got {v}")
    return pairs


def tray_conc(trays: Sequence[Any], names: Sequence[str], stores: np.ndarray | None = None) -> np.ndarray:
    """``[T, MAX_SPECIES]`` kg/m³ of each tray's liquid, in the film's species order. A tray with ``species=None``
    holds the stores' paint: ``stores`` (``[n, MAX_SPECIES]``, the store tools' rows), which must then agree."""
    conc = np.zeros((max(len(trays), 1), MAX_SPECIES))
    paints = np.unique(np.asarray(stores if stores is not None else np.zeros((0, MAX_SPECIES))), axis=0)
    for t, tray in enumerate(trays):
        if tray.species is None:
            if len(paints) > 1:
                raise PluginError(f"tray {t}: species=None means the stores' paint, but the stores carry "
                                  f"{len(paints)} different ones; give the tray its own species")
            if len(paints):
                conc[t] = paints[0]
            continue
        for name, v in tray.species:
            if name not in names:
                raise PluginError(f"tray {t}: no species {name!r}; the film carries {tuple(names)}")
            conc[t, list(names).index(name)] = v
    return conc


@wp.func
def bin_conc(ssp: StoreSpecies, count: int, w: int, b: int, vol: wp.float64) -> vec4d:
    """The bin's concentration [kg/m³] at its load ``vol`` [m³] (0 when empty)."""
    c = vec4d()
    if vol > wp.float64(0.0):
        for s in range(count):
            c[s] = ssp.load[w, b, s] / vol
    return c


@wp.func
def book_nip(ssp: StoreSpecies, w: int, k: int, b: int, s: int, kg: wp.float64):
    """``kg`` of species s into the film from bin b (< 0: taken from it into the bin)."""
    ssp.load[w, b, s] = ssp.load[w, b, s] - kg
    if kg > wp.float64(0.0):
        ssp.given[w, k, s] = ssp.given[w, k, s] + kg
    else:
        ssp.taken[w, k, s] = ssp.taken[w, k, s] - kg


@wp.func
def tray_take(ssp: StoreSpecies, count: int, w: int, k: int, t: int, b: int, take: wp.float64, vol: wp.float64):
    """A bin picks ``take`` [m³] up from tray t holding ``vol``: the tray's species go with it."""
    if take <= wp.float64(0.0) or vol <= wp.float64(0.0):
        return
    f = wp.min(take / vol, wp.float64(1.0))
    for s in range(count):
        m = ssp.tray[w, t, s] * f
        ssp.tray[w, t, s] = ssp.tray[w, t, s] - m
        ssp.load[w, b, s] = ssp.load[w, b, s] + m
        ssp.picked[w, k, s] = ssp.picked[w, k, s] + m


@wp.func
def let_go(ssp: StoreSpecies, count: int, w: int, k: int, b: int, v: wp.float64, vol: wp.float64) -> vec4d:
    """Bin b, holding ``vol`` [m³], lets ``v`` of it go (a drip, a release, a gob): its species leave with it."""
    out = vec4d()
    if v <= wp.float64(0.0) or vol <= wp.float64(0.0):
        return out
    f = wp.min(v / vol, wp.float64(1.0))
    for s in range(count):
        m = ssp.load[w, b, s] * f
        ssp.load[w, b, s] = ssp.load[w, b, s] - m
        ssp.given[w, k, s] = ssp.given[w, k, s] + m
        out[s] = m
    return out


def species_read(flow: Any, out: dict) -> None:
    """The stores' and trays' species and their books, ``[W, K]`` kg (host_api.read)."""
    K = flow.sp.count
    if not K or getattr(flow, "store_sp", None) is None:
        return
    load, given = out["tool_load_species"][..., :K], out["tool_species_given"][..., :K]
    taken, picked = out["tool_species_taken"][..., :K], out["tool_species_picked"][..., :K]
    out["store_species_ledger"] = load.sum(axis=1) + given.sum(axis=1) - taken.sum(axis=1) - picked.sum(axis=1)
    out["tray_species_ledger"] = out["tray_species"][..., :K].sum(axis=1) + picked.sum(axis=1)


__all__ = ["StoreSpecies", "bin_conc", "book_nip", "let_go", "species_pairs", "species_read", "tray_conc",
           "tray_take"]
