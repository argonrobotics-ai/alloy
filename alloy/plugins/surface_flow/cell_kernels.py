# SPDX-License-Identifier: Apache-2.0
"""The film step's cell side (design note §3.3): the outflow limiter, the cell update with its float64 ledger, the
limited discharge for the next substep's advection, and the device tick of a strided film. The supply's pending
inflow (``src_acc``) is applied over the film step and zeroed on its last substep. Each element has one writer;
there are no atomics.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.surface_flow.kernels import OPEN, WALL, Cells, Faces, Fluid, ToolField, inflow
from alloy.plugins.surface_flow.links import cell_dir


@wp.struct
class Book:
    """Per-cell float64 ledger [m³] (summed per world on read) and the drain interface arrays."""

    supplied: wp.array2d[wp.float64]
    spilled: wp.array2d[wp.float64]
    evaporated: wp.array2d[wp.float64]
    absorbed: wp.array2d[wp.float64]         # [m³] into a porous substrate (the "absorption" input, design §4.8)
    drained_total: wp.array2d[wp.float64]
    clipped: wp.array2d[wp.float64]
    dripped: wp.array2d[wp.float64]          # [m³] film the drop stage took from hanging cells (design §4.7)
    shed: wp.array2d[wp.float64]             # [m³] open-edge outflux of cells with no drain, until the drops read it
    shed_mom: wp.array2d[wp.vec3]            # [m³·m/s] that outflux times its face velocity (chart frame)
    flooded: wp.array2d[wp.int32]            # the supply's pool a cell is under (-1: none): its film joins the pool
    pooled: wp.array2d[wp.float64]           # [m] per area this exchange step: the supply reads and zeroes it
    pooled_total: wp.array2d[wp.float64]     # [m³] ledger
    travel: wp.array2d[wp.float32]           # [m] how far the film at a cell has moved (a run's length; paint sag)
    drained: wp.array2d[wp.float64]          # [m] per area this exchange step: the supply reads and zeroes it
    drain_of: wp.array[wp.int32]
    drain_cap: wp.array2d[wp.float64]        # [m/s]


MAX_SPECIES = 4
vec4d = wp.types.vector(length=4, dtype=wp.float64)


@wp.struct
class Species:
    """Dissolved or suspended species carried by the film (surfactant, soil, sanitizer): mass per area, moved with
    the film's limited fluxes from the donor cell. Evaporation leaves them behind; drains, pools, open edges and drips
    take them with the film (an open edge with no drain into ``shed`` for the drop stage, as the water's ``shed``).
    Ledgers in kg, per cell."""

    count: wp.int32                         # K <= MAX_SPECIES (0: none)
    mass: wp.array3d[wp.float64]            # [W, C, K] kg/m²
    conc: wp.array3d[wp.float64]            # [W, C, K] kg/m³ at the substep's start (the donors' concentrations)
    src: wp.array3d[wp.float64]             # [W, C, K] kg/m² pending (src_species: writers ADD, the film step applies)
    supplied: wp.array3d[wp.float64]        # [W, C, K] kg
    out: wp.array3d[wp.float64]             # [W, C, K] kg that left with film (edges, drains, pools, drips)
    shed: wp.array3d[wp.float64]            # [W, C, K] kg over open edges with no drain, until the drops read it
    drained: wp.array3d[wp.float64]         # [W, C, K] kg/m² into drains and pools this exchange (the supply's)
    clipped: wp.array3d[wp.float64]         # [W, C, K] kg (<= 0) a clip made: a donor took a cell below 0
    src_conc: wp.array[wp.float64]          # [K] kg/m³ the host's ``source`` carries (set_source(species=))


@wp.kernel
def conc_kernel(cells: Cells, tick: wp.array[wp.int32], stride: int, on: wp.array2d[wp.int32], h_dry: wp.float64,
                h: wp.array2d[wp.float64], sp: Species):
    """Each cell's species concentration before the substep moves anything (the cell kernel reads its donors')."""
    w, c = wp.tid()
    if tick[w] % stride != 0 or on[w, cells.tile[c]] == 0:
        return
    for k in range(sp.count):
        sp.conc[w, c, k] = wp.float64(0.0)
        if h[w, c] > h_dry:
            sp.conc[w, c, k] = sp.mass[w, c, k] / h[w, c]


@wp.func
def outflow(faces: Faces, q: wp.array2d[wp.float32], w: int, f: int, c: int) -> wp.float64:
    """Volume rate per unit dt [m³/s] that face f takes out of cell c (0 if it brings film in)."""
    qf = wp.float64(q[w, f])
    if faces.a[f] == c:
        return wp.max(qf, wp.float64(0.0)) * wp.float64(faces.length[f])
    return wp.max(-qf, wp.float64(0.0)) * wp.float64(faces.length[f])


@wp.kernel
def limit_kernel(cells: Cells, faces: Faces, tick: wp.array[wp.int32], stride: int, on: wp.array2d[wp.int32],
                 h: wp.array2d[wp.float64],
                 q: wp.array2d[wp.float32],
                 acc: wp.array2d[wp.float64], inv_dt: wp.float64, src_fixed: wp.array2d[wp.float64], dt: wp.float64,
                 ratio: wp.array2d[wp.float64]):
    w, c = wp.tid()
    if tick[w] % stride != 0 or on[w, cells.tile[c]] == 0:
        return
    fs = cells.face[c]
    out = wp.float64(0.0)
    for s in range(4):
        f = fs[s]
        if faces.kind[f] != WALL:
            out += outflow(faces, q, w, f, c) * dt
    avail = (h[w, c] + inflow(acc, inv_dt, src_fixed, w, c) * dt) * cells.area[c]
    avail = wp.max(avail, wp.float64(0.0))             # a depth rounded below 0 limits to 0, never to -inf
    r = wp.float64(1.0)
    if out > avail:
        r = avail / out
    ratio[w, c] = r


@wp.kernel
def cell_kernel(cells: Cells, faces: Faces, fl: Fluid, wet: wp.array[wp.vec2], tick: wp.array[wp.int32], stride: int,
                on: wp.array2d[wp.int32], q: wp.array2d[wp.float32], u: wp.array2d[wp.float32],
                ratio: wp.array2d[wp.float64], inv_dt: wp.float64, src_fixed: wp.array2d[wp.float64], last: int,
                dt: wp.float64, book: Book, h: wp.array2d[wp.float64], cwet: wp.array2d[wp.int32],
                acc: wp.array2d[wp.float64], sp: Species, tf: ToolField, src_mom: wp.array2d[wp.vec3]):
    w, c = wp.tid()
    if tick[w] % stride != 0 or on[w, cells.tile[c]] == 0:
        return
    area = cells.area[c]
    fs = cells.face[c]
    dv = wp.float64(0.0)
    edge = wp.float64(0.0)
    mom = wp.vec3(0.0, 0.0, 0.0)
    vel = wp.vec3(0.0, 0.0, 0.0)                        # the cell's film velocity: the mean of its faces'
    dm = vec4d()                                        # species moved in (kg), and out through open edges
    dm_edge = vec4d()
    for s in range(4):
        f = fs[s]
        kind = faces.kind[f]
        if kind == WALL:
            continue
        vel += 0.5 * u[w, f] * cell_dir(faces, f, c)
        qf = wp.float64(q[w, f])
        donor = faces.a[f]
        if qf < 0.0:
            donor = faces.b[f]
        if donor < 0:
            donor = c
        flux = qf * wp.float64(faces.length[f]) * dt * ratio[w, donor]      # along the face's direction, a -> b
        if faces.a[f] == c:
            dv -= flux
        else:
            dv += flux
        for k in range(sp.count):                       # species ride the flux at the donor's concentration
            sf = flux * sp.conc[w, donor, k]
            if faces.a[f] == c:
                dm[k] = dm[k] - sf
            else:
                dm[k] = dm[k] + sf
            if kind == OPEN:
                dm_edge[k] = dm_edge[k] + wp.abs(sf)
        if kind == OPEN:
            edge += wp.abs(flux)
            mom += wp.float32(wp.abs(flux)) * u[w, f] * faces.dir[f]
    rate = inflow(acc, inv_dt, src_fixed, w, c)
    book.supplied[w, c] = book.supplied[w, c] + rate * area * dt
    hn = h[w, c] + (dv + rate * area * dt) / area
    if edge > 0.0:
        if book.drain_of[c] >= 0:
            book.drained[w, c] = book.drained[w, c] + edge / area
            book.drained_total[w, c] = book.drained_total[w, c] + edge
        else:
            book.spilled[w, c] = book.spilled[w, c] + edge
            book.shed[w, c] = book.shed[w, c] + edge              # the drop stage reads and zeroes it (§4.7)
            book.shed_mom[w, c] = book.shed_mom[w, c] + mom
    floor = wp.float64(0.0)
    evap = fl.evaporation
    absorb = wp.float64(0.0)
    if fl.per_sink == 1:                                # the process stage's rates (design §4.8): solvent only
        evap = wp.float64(fl.evap_cell[w, c])
        absorb = wp.float64(fl.absorb_cell[w, c])
    if evap > 0.0 and hn > floor:
        e = wp.min(evap * dt, hn - floor)
        h1 = hn - e
        e = hn - h1                                     # book what left, exactly (WP11: a uniform film rounded alike)
        hn = h1
        book.evaporated[w, c] = book.evaporated[w, c] + e * area
    if absorb > 0.0 and hn > floor:
        a = wp.min(absorb * dt, hn - floor)
        h1 = hn - a
        a = hn - h1
        hn = h1
        book.absorbed[w, c] = book.absorbed[w, c] + a * area
    before_take = hn                                    # what drains and pools take a share of (species with it)
    if book.drain_of[c] >= 0 and hn > floor:
        take = wp.min(book.drain_cap[w, c] * dt, hn - floor)
        h1 = hn - take
        take = hn - h1
        hn = h1
        book.drained[w, c] = book.drained[w, c] + take
        book.drained_total[w, c] = book.drained_total[w, c] + take * area
    pool = book.flooded[w, c]
    if pool >= 0 and hn > floor:                        # under a pool: the film joins it (interface note, pools)
        book.pooled[w, c] = book.pooled[w, c] + hn
        book.pooled_total[w, c] = book.pooled_total[w, c] + hn * area
        hn = floor
    if sp.count > 0:
        carry_species(sp, book, w, c, area, dt, inv_dt, last, dm, dm_edge, before_take, hn, src_fixed[w, c])
    if hn < 0.0:                                        # float64 rounding of a fully drained cell
        book.clipped[w, c] = book.clipped[w, c] + hn * area
        hn = wp.float64(0.0)
    h[w, c] = hn
    if hn > fl.h_wet:
        book.travel[w, c] = book.travel[w, c] + wp.length(vel) * wp.float32(dt)
    if last != 0:
        acc[w, c] = wp.float64(0.0)                     # the film step applied the pending inflow
        src_mom[w, c] = wp.vec3(0.0, 0.0, 0.0)          # and the pending momentum
    # the cell can change at its next step: the dry-tile skip keeps its tile (tiles.py)
    takes = hn > wp.float64(0.0) and (evap > wp.float64(0.0) or absorb > wp.float64(0.0) or book.drain_of[c] >= 0
                                      or pool >= 0)
    if tf.on == 1 and hn > wp.float64(0.0):
        takes = takes or tf.cover[w, c] > 0.0             # a tool pushes trace film too (WP2 1b)
    cwet[w, c] = 0
    if hn > fl.h_dry or rate > wp.float64(0.0) or takes:
        cwet[w, c] = 1
    if last == 0:                                       # a species source still pending: its later substeps apply it
        for k in range(sp.count):
            if sp.src[w, c, k] != wp.float64(0.0):
                cwet[w, c] = 1


@wp.func
def carry_species(sp: Species, book: Book, w: int, c: int, area: wp.float64, dt: wp.float64, inv_dt: wp.float64,
                  last: int, dm: vec4d, dm_edge: vec4d, moved: wp.float64, left: wp.float64, fixed: wp.float64):
    """The cell's species after the flux (``dm``), the pending sources, and what drains and pools took with the film:
    the share ``1 - left/moved`` of the film the cell had after the flux (evaporation takes none). A negative mass
    from a donor's concentration is clipped to 0 before the take and booked in ``clipped`` (as the water's), so
    ``out`` stays what left, equal to what its destinations book (``shed``, ``drained``), and the ledger stays exact."""
    to_drain = book.drain_of[c] >= 0
    for k in range(sp.count):
        add = (sp.src[w, c, k] * inv_dt + fixed * sp.src_conc[k]) * dt * area     # kg, with the inflow
        sp.supplied[w, c, k] = sp.supplied[w, c, k] + add
        m = sp.mass[w, c, k] * area + dm[k] + add
        if m < wp.float64(0.0):                         # before the take: drains and pools never get a negative mass
            sp.clipped[w, c, k] = sp.clipped[w, c, k] + m          # the species the clip makes (<= 0)
            m = wp.float64(0.0)
        taken = wp.float64(0.0)
        if moved > wp.float64(0.0) and left < moved:
            share = wp.float64(1.0) - wp.max(left, wp.float64(0.0)) / moved
            if book.flooded[w, c] >= 0 or to_drain:
                taken = m * share                       # <= m, so the cell keeps m - taken >= 0
        m = m - taken
        sp.out[w, c, k] = sp.out[w, c, k] + dm_edge[k] + taken
        into_tanks = taken
        if to_drain:
            into_tanks = into_tanks + dm_edge[k]        # a drain on an open edge takes what leaves over it
        else:
            sp.shed[w, c, k] = sp.shed[w, c, k] + dm_edge[k]     # it leaves with the shed film (drops_kernels)
        if to_drain or book.flooded[w, c] >= 0:
            sp.drained[w, c, k] = sp.drained[w, c, k] + into_tanks / area
        sp.mass[w, c, k] = m / area
        if last != 0:
            sp.src[w, c, k] = wp.float64(0.0)


@wp.kernel
def limited_kernel(faces: Faces, tick: wp.array[wp.int32], stride: int, on: wp.array2d[wp.int32],
                   q_new: wp.array2d[wp.float32],
                   ratio: wp.array2d[wp.float64], u_new: wp.array2d[wp.float32], g_new: wp.array2d[wp.float32],
                   q: wp.array2d[wp.float32], u: wp.array2d[wp.float32], gface: wp.array2d[wp.float32]):
    """The face state for the next substep: velocity, the transverse estimate, and the discharge that actually moved
    (``q·ratio[donor]``), so the momentum advection follows the mass flux in draining cells (WP2 review M2). Only on
    the film's steps."""
    w, f = wp.tid()
    if tick[w] % stride != 0 or on[w, faces.tile[f]] == 0:
        return
    qf = q_new[w, f]
    donor = faces.a[f]
    if qf < 0.0:
        donor = faces.b[f]
    if donor < 0:
        donor = faces.a[f]
        if donor < 0:
            donor = faces.b[f]
    if donor < 0:
        q[w, f] = qf
    else:
        q[w, f] = qf * wp.float32(ratio[w, donor])
    u[w, f] = u_new[w, f]                         # the face state for the next substep
    gface[w, f] = g_new[w, f]


@wp.kernel
def tick_kernel(tick: wp.array[wp.int32]):
    """One more exchange step; the film steps when ``tick % stride == 0`` (a device counter: graphs capture it)."""
    w = wp.tid()
    tick[w] = tick[w] + 1


@wp.kernel
def rebank_kernel(tick: wp.array[wp.int32], stride: int, dt: wp.float64, old: wp.array2d[wp.float64],
                  now: wp.array2d[wp.float64], acc: wp.array2d[wp.float64]):
    """The host changed ``source`` between a strided film's steps. The step reads the source it then sees over its
    whole window (``inflow``), so the exchanges already past in the window (``tick % stride``) bank the difference
    old - now in ``src_acc``: every exchange gets the rate it was set with, exactly."""
    w, c = wp.tid()
    e = tick[w] % stride
    d = old[w, c] - now[w, c]
    if e == 0 or d == wp.float64(0.0):
        return
    acc[w, c] = acc[w, c] + d * dt * wp.float64(e)


@wp.kernel
def rebank_species_kernel(tick: wp.array[wp.int32], stride: int, dt: wp.float64, old: wp.array2d[wp.float64],
                          now: wp.array2d[wp.float64], old_conc: wp.array[wp.float64], sp: Species):
    """``rebank_kernel`` for the species the host source carries: the past exchanges of the window bank
    old·c_old - now·c_now in ``src_species``."""
    w, c = wp.tid()
    e = tick[w] % stride
    if e == 0:
        return
    for k in range(sp.count):
        d = old[w, c] * old_conc[k] - now[w, c] * sp.src_conc[k]
        if d != wp.float64(0.0):
            sp.src[w, c, k] = sp.src[w, c, k] + d * dt * wp.float64(e)


__all__ = ["Book", "MAX_SPECIES", "Species", "carry_species", "cell_kernel", "conc_kernel", "limit_kernel",
           "limited_kernel", "outflow", "rebank_kernel", "rebank_species_kernel", "tick_kernel"]
