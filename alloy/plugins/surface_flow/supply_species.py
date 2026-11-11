# SPDX-License-Identifier: Apache-2.0
"""Species through the supply's transfers (WP15's species, interface: ``src_species`` / ``drained_species``).

The film carries up to four named species (``SurfaceFlow.Config.species``). The lumped network carries positional
ones (each node's ``species`` tuple), which the network dict names in ``species``. A film species reaches a tank only
if the network carries it under that name. ``NetSpecies`` bundles both sides and the supply's species ledger:

- drains and flooded pools move the film's ``drained_species`` into their tanks' species, or book it ``away``;
- withdrawal films take the pool tank's concentration into ``src_species``, out of the tank;
- nozzles send their species with their water, booked where it went: onto charts (``sent``), into pools (``to_tank``;
  the network's species go into the tank's species), ``lost``, ``overspray`` and ``mist``, as the volume ledger.
  A nozzle's concentration is its fixed ``Nozzle.species`` value, else its node's; a fixed value also goes into a
  pool's tank for the network species of that name.

Ledger ``[W, 8, K]`` kg, per film species, in the columns below.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.supply_deposit import node_conc
from alloy.plugins.surface_flow.supply_kernels import (CHART, LUMP, POOL, SHAPE, SPRAY, NozzleSpec, PoolSpec,
                                                       weight_sum)
from alloy.plugins.surface_flow.supply_spec import STAGE

SP_TANKS = wp.constant(0)      # into tanks (drains and flooded pools)
SP_AWAY = wp.constant(1)       # drained or pooled where no tank holds that species (or a drain to no tank)
SP_WITHDRAWN = wp.constant(2)  # out of pool tanks into withdrawal films
SP_SENT = wp.constant(3)       # in what the nozzles deposited on charts
SP_TO_TANK = wp.constant(4)    # in what they sent into pools
SP_LOST = wp.constant(5)       # in a jet that hit no chart or pool
SP_OVERSPRAY = wp.constant(6)  # in a spray's rays that hit no chart or pool
SP_MIST = wp.constant(7)       # in a spray's (1 - efficiency) share
SP_CAUGHT = wp.constant(8)     # in what shapes caught
SP_LUMPED = wp.constant(9)     # in what jets sent onto lumped bodies
SP_LUMP_POOL = wp.constant(10)  # into pools' tanks from lumps (WP9's pool_add_sp; out: negative)
SP_REFUSED = wp.constant(11)  # handed over with no volume (supply_stage), taken beyond a tank (supply_lump)
SP_STAGED = wp.constant(12)   # handed to the supply by film stages with their drips (supply_stage)
N_SP_LEDGER = 13
SP_LEDGER = ("tanks", "away", "withdrawn", "sent", "to_tank", "lost", "overspray", "mist", "caught", "lumped",
             "lump_pool", "refused", "staged")


@wp.struct
class NetSpecies:
    n_film: wp.int32                         # the film's species count K (0: none)
    map: wp.vec4i                            # film species k -> the network's species index, or -1
    medium: wp.array[wp.int32]
    bulk: wp.array[wp.float64]
    p_ref: wp.array[wp.float64]
    poly: wp.array[wp.float64]
    pressure: wp.array2d[wp.float64]
    volume: wp.array2d[wp.float64]
    species: wp.array3d[wp.float64]          # [W, n, Kn] kg in each node
    out_conc: wp.array3d[wp.float64]         # [W, n, Kn] kg/m³ each node's outflows carried in the last network step
    inflow: wp.array3d[wp.float64]           # [W, n, Kn] kg/s into nodes over the next network step
    drained: wp.array3d[wp.float64]          # the film's drained_species [W, C, K] kg/m² (read and zeroed here)
    src: wp.array3d[wp.float64]              # the film's src_species [W, C, K] kg/m² (added to here)
    led: wp.array3d[wp.float64]              # [W, 4, K] kg
    drip: wp.array3d[wp.float64]             # [W, D, Kn] kg/m³ of the drips in flight (node -2 - d; supply_holes)
    sdrip: wp.array3d[wp.float64]            # [W, E, K] kg/m³ (film species) of stage drips (node -1000 - e)


@wp.func
def tank_conc(ns: NetSpecies, w: int, node: int, k: int) -> wp.float64:
    """Film species k's concentration in network node ``node`` [kg/m³] (0 if the network does not carry it)."""
    kn = ns.map[k]
    if kn < 0:
        return wp.float64(0.0)
    return node_conc(w, node, kn, ns.medium, ns.bulk, ns.p_ref, ns.poly, ns.pressure, ns.volume, ns.species)


@wp.func
def into_tank(ns: NetSpecies, w: int, tank: int, k: int, m: wp.float64, dt: wp.float64):
    """``m`` kg of film species k into tank node ``tank`` over the next network step, or away."""
    kn = ns.map[k]
    if tank >= 0 and kn >= 0:
        ns.inflow[w, tank, kn] = ns.inflow[w, tank, kn] + m / dt
        ns.led[w, SP_TANKS, k] = ns.led[w, SP_TANKS, k] + m
    else:
        ns.led[w, SP_AWAY, k] = ns.led[w, SP_AWAY, k] + m


@wp.func
def nozzle_conc(ns: NetSpecies, w: int, fixed: wp.vec4d, node: int, k: int) -> wp.float64:
    """Film species k in what a nozzle sends [kg/m³]: its fixed value, or (-1) what its edge carried out of its node
    in the last network step (``out_conc``, as the edge's species ledger booked it: the last of a tank that step
    emptied too, where the node's mass over its amount reads 0), or a drip's (node -2 - d: what its vessel held when
    it left)."""
    if fixed[k] >= wp.float64(0.0):
        return fixed[k]
    if node <= -1000:
        return ns.sdrip[w, -1000 - node, k]
    kn = ns.map[k]
    if kn < 0:
        return wp.float64(0.0)
    if node <= -2:
        return ns.drip[w, -2 - node, kn]
    return ns.out_conc[w, node, kn]                      # what the edge carried out of its node (also as it emptied)


@wp.func
def net_conc(ns: NetSpecies, w: int, fixed: wp.vec4d, node: int, kn: int) -> wp.float64:
    """Network species kn in what a nozzle sends [kg/m³]: the fixed value of the film species of its name, else what
    its edge carried (``nozzle_conc``)."""
    for k in range(ns.n_film):
        if ns.map[k] == kn and fixed[k] >= wp.float64(0.0):
            return fixed[k]
    if node <= -1000:                                     # a stage drip: the film species of that name
        c = wp.float64(0.0)
        for k in range(ns.n_film):
            if ns.map[k] == kn:
                c = ns.sdrip[w, -1000 - node, k]
        return c
    if node <= -2:
        return ns.drip[w, -2 - node, kn]
    return ns.out_conc[w, node, kn]


@wp.func
def book_sent(ns: NetSpecies, w: int, col: int, v: wp.float64, fixed: wp.vec4d, node: int):
    """``v`` m³ of a nozzle's water into ledger column ``col``, at its concentration of each film species (into a
    pool, a species the network does not carry is booked ``away``: no tank holds it)."""
    for k in range(ns.n_film):
        ck = col
        if col == SP_TO_TANK and ns.map[k] < 0:
            ck = int(SP_AWAY)
        ns.led[w, ck, k] = ns.led[w, ck, k] + v * nozzle_conc(ns, w, fixed, node, k)


@wp.kernel
def species_flight(dt: wp.float64, nozzles: wp.array[NozzleSpec], discharge: wp.array2d[wp.float64],
                   nconc: wp.array[wp.vec4d], nnode: wp.array[wp.int32], ns: NetSpecies, out: wp.array2d[wp.float64]):
    """Per world, each film species in the nozzles' water in flight (the last step's Q·dt at each nozzle's
    concentration, nozzles in order): what ``species_sent`` books at the next exchange, all of it (JOB3: a network's
    species book reads the tank's loss a step before the supply books it)."""
    w = wp.tid()
    for k in range(out.shape[1]):
        out[w, k] = wp.float64(0.0)
    for n in range(nozzles.shape[0]):
        q = -discharge[w, nozzles[n].edge]
        if q <= wp.float64(0.0):
            continue
        for k in range(ns.n_film):
            out[w, k] = out[w, k] + q * dt * nozzle_conc(ns, w, nconc[n], nnode[n], k)


@wp.kernel
def species_sent(dt: wp.float64, nozzles: wp.array[NozzleSpec], ray_wt: wp.array[wp.float32],
                 pools: wp.array[PoolSpec], discharge: wp.array2d[wp.float64], hit_kind: wp.array2d[wp.int32],
                 hit_idx: wp.array2d[wp.int32], nconc: wp.array[wp.vec4d], nnode: wp.array[wp.int32], n_net: int,
                 ns: NetSpecies):
    """Per world, the species in what the nozzles sent this exchange, by where it went (rays in order, as
    ``supply_deposit.deposit`` books the volume), and the network's species into the pools' tanks."""
    w = wp.tid()
    for n in range(nozzles.shape[0]):
        noz = nozzles[n]
        q_all = -discharge[w, noz.edge]
        if q_all <= wp.float64(0.0):
            continue
        eff = wp.float64(noz.efficiency)
        book_sent(ns, w, SP_MIST, q_all * (wp.float64(1.0) - eff) * dt, nconc[n], nnode[n])
        norm = weight_sum(noz, ray_wt)
        for r in range(noz.ray_start, noz.ray_start + noz.ray_count):
            q = q_all * eff * wp.float64(ray_wt[r]) / norm
            kind = hit_kind[w, r]
            col = int(SP_SENT)
            if kind == POOL:
                col = int(SP_TO_TANK)
                tank = pools[hit_idx[w, r]].tank
                for kn in range(n_net):
                    ns.inflow[w, tank, kn] = ns.inflow[w, tank, kn] + q * net_conc(ns, w, nconc[n], nnode[n], kn)
            elif kind == LUMP:
                col = int(SP_LUMPED)
            elif kind == SHAPE:
                col = int(SP_CAUGHT)
            elif kind != CHART:
                col = int(SP_LOST)
                if noz.pattern == SPRAY:
                    col = int(SP_OVERSPRAY)
            book_sent(ns, w, col, q * dt, nconc[n], nnode[n])


@wp.kernel
def drain_species(dt: wp.float64, dstart: wp.array[wp.int32], dcell: wp.array[wp.int32], dto: wp.array[wp.int32],
                  area: wp.array[wp.float64], ns: NetSpecies):
    """Each drain's species this exchange (Σ drained_species·area, fixed order) into its tank, or away; zeroed."""
    w = wp.tid()
    for d in range(dto.shape[0]):
        for k in range(ns.n_film):
            m = wp.float64(0.0)
            for i in range(dstart[d], dstart[d + 1]):
                c = dcell[i]
                m = m + ns.drained[w, c, k] * area[c]
                ns.drained[w, c, k] = wp.float64(0.0)
            into_tank(ns, w, dto[d], k, m, dt)


# ---- host side -----------------------------------------------------------------------------------------------------

def net_species(ctx: Any, film: Any, net: Any, names: tuple[str, ...], fluid: Any) -> NetSpecies:
    """The film's and the network's species bundled, film species mapped to the network's by name (``names``: the
    network dict's ``species``, one name per position of the nodes' species tuples)."""
    sp = getattr(film, "sp", None)
    film_names = tuple(getattr(film.config, "species", ()) or ()) if sp is not None else ()
    if len(names) != len(set(names)) or (names and len(names) != net.species):
        raise PluginError(f"network species names {names} must be distinct, one per position of the nodes' species "
                          f"tuples ({net.species})")
    if film_names and net.species and not names:
        raise PluginError(f"the film carries species {film_names} and the network {net.species}: name the network's "
                          f"(network['species'] = (...)) so the supply can map them")
    ns = NetSpecies()
    ns.n_film = len(film_names)
    m = [names.index(n) if n in names else -1 for n in film_names]
    ns.map = wp.vec4i(*(m + [-1] * (4 - len(m))))
    dummy = ctx.zeros((1, 1, 1), wp.float64)
    ns.drained, ns.src = (sp.drained, sp.src) if ns.n_film else (dummy, dummy)
    ns.drip = dummy                                      # the drips' (supply_holes) when there are any
    ns.sdrip = dummy                                     # the stage drips' (supply_stage)
    ns.led = film.add_state("supply_species_ledger", ctx.zeros((ctx.world_count, N_SP_LEDGER, max(ns.n_film, 1)),
                                                                 wp.float64))
    return refresh(ns, fluid)


def refresh(ns: NetSpecies, fluid: Any) -> NetSpecies:
    """Point ``ns`` at the network's current arrays (before each launch that uses it)."""
    ns.medium, ns.bulk, ns.p_ref, ns.poly = fluid.medium, fluid.bulk, fluid.p_ref, fluid.poly
    ns.pressure, ns.volume, ns.species, ns.inflow = fluid.pressure, fluid.volume, fluid.species, fluid.inflow_species
    ns.out_conc = fluid.out_conc
    return ns


def nozzle_tables(nozzles: tuple[Any, ...], net: Any, film_names: tuple[str, ...], ns: NetSpecies,
                  ) -> tuple[list[tuple[float, ...]], list[int]]:
    """Per nozzle: each film species' fixed concentration, or -1 for its node's (0 if the network has no such
    species), and its source node (the d-th drip nozzle: -2 - d, its drips' own concentrations; the e-th stage drip:
    -1000 - e, every film species from the drip itself, -1)."""
    conc, node, drips, stages = [], [], 0, 0
    for noz in nozzles:
        fixed = dict(noz.species or {})
        unknown = sorted(set(fixed) - set(film_names))
        if unknown or any(v < 0 for v in fixed.values()):
            raise PluginError(f"nozzle {noz.name!r}: species {unknown} not on the film {film_names}, or < 0")
        c = [float(fixed[n]) if n in fixed else (-1.0 if ns.map[k] >= 0 or noz.drip_of == STAGE else 0.0)
             for k, n in enumerate(film_names)]
        conc.append(tuple(c + [0.0] * (4 - len(c))))
        if noz.drip_of == STAGE:
            node.append(-1000 - stages)
            stages += 1
        elif noz.drip_of:
            node.append(-2 - drips)
            drips += 1
        else:
            node.append(net.node_index(net.edges[net.edge_index(noz.edge)].a))
    return conc or [(0.0,) * 4], node or [0]


def species_reads(ns: NetSpecies) -> dict[str, Any]:
    """The species ledger's columns [W, K] kg, as ``species_<column>``."""
    led = ns.led.numpy()[:, :, :ns.n_film]
    return {f"species_{name}": led[:, k].copy() for k, name in enumerate(SP_LEDGER)}


__all__ = ["N_SP_LEDGER", "NetSpecies", "SP_AWAY", "SP_CAUGHT", "SP_LEDGER", "SP_LOST", "SP_LUMPED", "SP_LUMP_POOL",
           "SP_MIST", "SP_OVERSPRAY", "SP_REFUSED", "SP_SENT", "SP_STAGED", "SP_TANKS", "SP_TO_TANK", "SP_WITHDRAWN",
           "book_sent", "drain_species", "into_tank", "net_conc", "net_species", "nozzle_conc", "nozzle_tables",
           "refresh", "species_flight", "species_reads", "species_sent", "tank_conc"]
