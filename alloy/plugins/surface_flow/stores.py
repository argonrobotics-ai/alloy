# SPDX-License-Identifier: Apache-2.0
"""Tool stores (design note §4.1): the paint a roller's nap, a brush's bristles or a pad carries.

A store is binned around a roller's cover (``bins``, 32 by default) and is one bin for other tools. The bins turn
with the cover's body, so a roller loaded on one side lays unevenly. Each exchange, ``stores_kernel`` runs one thread
per world over its tools in a fixed order:
- **The nip.** Once per passage, when the tool's centre line crosses a cell it touches, the cell's film and the
  contact bin's free load ``l`` split. ``l`` is the load above the retained share, spread over the bin's nap area.
  The surface keeps ``(1 - split)·(h + l)``: a wetter surface gives paint to the tool, a drier one takes it.
  - The roller's contact bin is the one facing the chart; a brush or pad has one bin.
  - Splitting once per crossing matters. A rate form completes only part of the split per passage, and splitting
    every exchange in contact drifts to another fixed point.
  - Store tools do not plough the film as ceilings; the nip replaces that.
- **Pick-up:** a bin below a tray's level, inside its footprint, fills toward capacity at ``pickup``. It takes the
  volume from the tray.
- **Drip:** a bin in neither holds ``held·capacity`` and drips the rest at ``drip`` (1/s). "No drip at retention" is
  this rule.

Every volume moves exactly, in float64:
- the film books what tools take in ``to_tools``;
- the tools book their loads, ``dripped`` and ``picked``;
- the trays book their volume.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import com_world, point_velocity
from alloy.plugins.surface_flow.cell_kernels import MAX_SPECIES, Species, vec4d
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.stores_fill import BEAR, ON_BASE, fill_lows, pass_low, sole_cover
from alloy.plugins.surface_flow.stores_lift import LIFT_N, gravity_dir, overflow_species, stick_share, stick_volume
from alloy.plugins.surface_flow.stores_species import StoreSpecies, bin_conc, let_go, species_pairs, tray_take
from alloy.plugins.surface_flow.stores_meter import U_MIN, meter, metered, nip_heading, nip_species
from alloy.plugins.surface_flow.tool_geometry import ROLLER, ToolSet, cell_point, pose_of, tool_under, window

ROLLER_BINS = 32


@dataclass(frozen=True)
class Store:
    """A tool's store. ``capacity`` [m³] when saturated. Of it, ``retained`` is never released, and ``held`` stays
    when the tool is lifted out of the tray: 0.25 and 0.75 for a roller cover (design note §7). ``split`` is the share
    of the nip film the tool keeps (about 0.5 for a forward roll). ``pickup`` and ``drip`` are rates [1/s], and
    ``initial`` is the loaded share at the start. ``species``: what its load carries [kg/m³] by name (paint's solids);
    film it takes leaves with the film's species (booked out), film it lays brings these (booked supplied).
    ``express``: the share of the free load the nap expresses into the nip (GA: about twice the nap's strain); the
    surface keeps ``(1 - split)·(h + express·l)`` and the bin keeps the rest (SOFT-3's SCR-SOFT-3-3).
    ``scrape``: a blade that cuts all that stands above its underside into the store, over its whole footprint every
    exchange (GA's plough, coupling.py:521-525; ``split`` must be 1): a cell keeps ``min(h, max(gap - base, 0))``.
    While it touches, it fills the lows below its bearing band from its free load, up to its lowest level, or the
    base level where it bears on bare base (``fill_lows``, GA's knife, :556-596)."""

    capacity: float
    retained: float = 0.25
    held: float = 0.75
    split: float = 0.5
    pickup: float = 5.0
    drip: float = 1.0
    bins: int = 0                  # 0: 32 around a roller's cover, 1 for other tools
    initial: float = 0.0
    species: Any = None            # {name: kg/m³} of the load (kept as sorted pairs); None: none
    express: float = 1.0           # share of the free load in the nip, (0, 1]
    scrape: bool = False           # cut all above the underside (split 1); fill the lows from the carry
    meter: float = 0.0             # a brush: GA's metering constant C (stores_meter.py); 0: the nip split
    meter_max: float = 150e-6      # the metered film's cap h_max [m] (GA BRUSH_PAINT, params.py:177-178)
    fill: str = "knife"            # a scraping store's fill: "knife" (bearing band) or "profile" (a sealant nose)
    meter_viscosity: float = 0.0   # the paint's viscosity at the brush's shear [Pa·s]; 0: the film's + tau_y·50um/U
    stick_area: float = 0.0        # a blade's paste-holding face [m²] for its lifted stick limit; 0: the footprint

    def __post_init__(self) -> None:
        if not (math.isfinite(self.capacity) and self.capacity > 0.0):
            raise PluginError(f"store: capacity must be finite and > 0, got {self.capacity}")
        for name in ("retained", "held", "split", "initial"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise PluginError(f"store: {name} must be in [0, 1], got {v}")
        if not (math.isfinite(self.express) and 0.0 < self.express <= 1.0):
            raise PluginError(f"store: express must be in (0, 1], got {self.express}")
        if not (math.isfinite(self.meter) and self.meter >= 0.0 and math.isfinite(self.meter_max)
                and self.meter_max > 0.0 and math.isfinite(self.meter_viscosity) and self.meter_viscosity >= 0.0):
            raise PluginError(f"store: meter must be >= 0, meter_max > 0 and meter_viscosity >= 0, got "
                              f"{self.meter}, {self.meter_max}, {self.meter_viscosity}")
        if self.meter > 0.0 and self.scrape:
            raise PluginError("store: a metering brush cannot also scrape")
        if not (math.isfinite(self.stick_area) and self.stick_area >= 0.0):
            raise PluginError(f"store: stick_area must be finite and >= 0 m², got {self.stick_area}")
        if self.fill not in ("knife", "profile"):
            raise PluginError(f"store: fill must be 'knife' or 'profile' (stores_fill.py), got {self.fill!r}")
        if self.scrape and self.split != 1.0:
            raise PluginError(f"store: a scraping blade cuts all that stands above it every exchange (GA's plough): "
                              f"split must be 1, got {self.split}")
        if self.retained > self.held:
            raise PluginError(f"store: retained ({self.retained}) must not exceed held ({self.held})")
        for name in ("pickup", "drip"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0.0):
                raise PluginError(f"store: {name} must be finite and >= 0, got {v}")
        if self.bins < 0:
            raise PluginError(f"store: bins must be >= 0, got {self.bins}")
        pairs = tuple(sorted(dict(self.species or {}).items()))
        for name, v in pairs:
            if not (math.isfinite(v) and v >= 0.0):
                raise PluginError(f"store: species {name!r} must be finite and >= 0 kg/m³, got {v}")
        object.__setattr__(self, "species", pairs)                 # hashable: ((name, kg/m³), ...)


@dataclass(frozen=True)
class Tray:
    """A pool of the film's liquid that stores pick up from: a paint tray's well. Its floor's centre and x, y half
    extents are in the world frame, and ``volume`` [m³] fills it at the start, with ``species`` ({name: kg/m³}, a
    paint well's pigment; stores_species.py). None: the stores' own paint (``Store(species=)``, as before p36)."""

    center: tuple[float, float, float]
    half: tuple[float, float]
    volume: float
    species: Any = None

    def __post_init__(self) -> None:
        if len(self.half) != 2 or min(self.half) <= 0.0 or not (math.isfinite(self.volume) and self.volume >= 0.0):
            raise PluginError(f"tray: half extents must be > 0 and volume >= 0, got {self.half}, {self.volume}")
        if self.species is not None:
            object.__setattr__(self, "species", species_pairs("tray", self.species))


def store_spec(spec: Store | dict | None) -> Store | None:
    if spec is None or isinstance(spec, Store):
        return spec
    return Store(**spec)


def tray_spec(spec: Tray | dict) -> Tray:
    return spec if isinstance(spec, Tray) else Tray(**{k: tuple(v) if isinstance(v, list) else v
                                                         for k, v in spec.items()})


@wp.struct
class StoreSet:
    has: wp.array[wp.int32]          # [n] 1: the tool has a store
    bins: wp.array[wp.int32]         # [n]
    start: wp.array[wp.int32]        # [n] its first bin in the flat [W, bins] arrays
    capacity: wp.array[wp.float64]   # [n] per bin [m³]
    retained: wp.array[wp.float64]   # [n] share
    held: wp.array[wp.float64]
    split: wp.array[wp.float64]
    express: wp.array[wp.float64]    # [n] share of the free load in the nip
    scrape: wp.array[wp.int32]       # [n] 1: cut only the film above the underside, never lay
    meter: wp.array[wp.float64]      # [n] a brush's metering constant C (0: the nip split)
    fill_profile: wp.array[wp.int32] # [n] 1: a scraping store fills up to its own profile (a sealant nose)
    meter_max: wp.array[wp.float64]  # [n] its metered film's cap [m]
    meter_eta: wp.array[wp.float64]  # [n] its paint's viscosity at the brush's shear [Pa·s] (0: from the film)
    stick: wp.array[wp.float64]      # [n] a scraping store's level stick limit [m³] (stores_lift.py)
    down: wp.vec3                    # gravity's direction, world
    mu: wp.float64                   # the film's (plastic) viscosity [Pa·s], for the metering law
    tau: wp.float64                  # its yield stress [Pa]
    pickup: wp.array[wp.float64]     # [1/s]
    drip: wp.array[wp.float64]
    area: wp.array[wp.float64]       # [n] a bin's nap area [m²]


@wp.struct
class TraySet:
    center: wp.array[wp.vec3]
    half: wp.array[wp.vec2]
    area: wp.array[wp.float64]
    count: wp.int32


@dataclass
class StoreTables:
    stores: StoreSet
    trays: TraySet
    bin_count: int
    start: np.ndarray                # [n] first bin
    bins: np.ndarray                 # [n]


def bins_of(spec: Any) -> int:
    if spec.store is None:
        return 0
    return spec.store.bins or (ROLLER_BINS if spec.shape == "roller" else 1)


def store_species(tools: Sequence[Any], names: Sequence[str]) -> np.ndarray:
    """``[n, MAX_SPECIES]`` kg/m³ of each tool's load, in the film's species order."""
    conc = np.zeros((max(len(tools), 1), MAX_SPECIES))
    for k, t in enumerate(tools):
        for name, v in dict(getattr(t.store, "species", None) or {}).items():
            if name not in names:
                raise PluginError(f"store of tool {k}: no species {name!r}; the film carries {tuple(names)}")
            conc[k, list(names).index(name)] = v
    return conc


def build_stores(ctx: Any, tools: Sequence[Any], trays: Sequence[Tray], species: Sequence[str] = (),
                 material: Any = None) -> StoreTables:
    """Store and tray tables for ``tools`` (``applicators.Applicator`` specs); ``material`` (the film's
    FilmMaterial) gives a metering brush its viscosity and yield stress."""
    bins = np.asarray([bins_of(t) for t in tools], np.int32)
    start = np.concatenate([[0], np.cumsum(bins)[:-1]]).astype(np.int32)
    s = StoreSet()

    def per_tool(fn: Any, dtype: Any = wp.float64) -> wp.array:
        vals = [fn(t) if t.store is not None else 0.0 for t in tools]
        return ctx.array(np.asarray(vals, np.int32 if dtype == wp.int32 else np.float64), dtype)

    s.has = per_tool(lambda t: 1, wp.int32)
    s.bins, s.start = ctx.array(bins, wp.int32), ctx.array(start, wp.int32)
    s.capacity = per_tool(lambda t: t.store.capacity / bins_of(t))
    s.retained, s.held, s.split, s.express = (per_tool(lambda t, k=k: getattr(t.store, k))
                                              for k in ("retained", "held", "split", "express"))
    s.pickup, s.drip = (per_tool(lambda t, k=k: getattr(t.store, k)) for k in ("pickup", "drip"))
    s.scrape = per_tool(lambda t: int(t.store.scrape), wp.int32)
    s.meter, s.meter_max, s.meter_eta = (per_tool(lambda t, k=k: getattr(t.store, k))
                                         for k in ("meter", "meter_max", "meter_viscosity"))
    s.fill_profile = per_tool(lambda t: int(t.store.fill == "profile"), wp.int32)
    s.mu = float(getattr(material, "viscosity", 0.0) or 0.0)
    s.tau = float(getattr(material, "yield_stress", 0.0) or 0.0)
    s.area = per_tool(lambda t: (2.0 * math.pi * t.size[0] / bins_of(t)) * 2.0 * t.size[1] if t.shape == "roller"
                      else t.footprint)
    s.stick = ctx.array(np.asarray([stick_volume(t, material, ctx.gravity) for t in tools], np.float64), wp.float64)
    s.down = gravity_dir(ctx.gravity)
    tr = TraySet()
    tr.center = ctx.array(np.asarray([t.center for t in trays] or [(0.0, 0.0, 0.0)], np.float32), wp.vec3)
    tr.half = ctx.array(np.asarray([t.half for t in trays] or [(0.0, 0.0)], np.float32), wp.vec2)
    tr.area = ctx.array(np.asarray([4.0 * t.half[0] * t.half[1] for t in trays] or [1.0], np.float64), wp.float64)
    tr.count = len(trays)
    return StoreTables(stores=s, trays=tr, bin_count=int(bins.sum()), start=start, bins=bins)


@wp.func
def tray_at(trays: TraySet, tray_vol: wp.array2d[wp.float64], w: int, p: wp.vec3) -> int:
    """The tray whose liquid the world point ``p`` is in (below its level, inside its footprint), else -1."""
    for t in range(trays.count):
        o = trays.center[t]
        hh = trays.half[t]
        level = o[2] + wp.float32(tray_vol[w, t] / trays.area[t])
        if wp.abs(p[0] - o[0]) <= hh[0] and wp.abs(p[1] - o[1]) <= hh[1] and p[2] < level:
            return t
    return -1


@wp.func
def bin_direction(ts: ToolSet, k: int, b: int, nb: int) -> wp.vec3:
    """Bin b's outward direction on a roller's cover, tool body frame (perpendicular to the axis)."""
    a = wp.normalize(ts.axis[k])
    e1 = wp.normalize(wp.cross(a, wp.vec3(0.0, 0.0, 1.0)))
    if wp.length(wp.cross(a, wp.vec3(0.0, 0.0, 1.0))) < 1.0e-6:
        e1 = wp.normalize(wp.cross(a, wp.vec3(1.0, 0.0, 0.0)))
    e2 = wp.cross(a, e1)
    th = 2.0 * wp.pi * (float(b) + 0.5) / float(nb)
    return wp.cos(th) * e1 + wp.sin(th) * e2


@wp.func
def contact_bin(ts: ToolSet, k: int, nb: int, T: wp.transform, X: wp.transform) -> int:
    """The roller bin facing the chart (its direction closest to the chart's inward normal)."""
    down = wp.transform_vector(wp.transform_inverse(T), wp.transform_vector(X, wp.vec3(0.0, 0.0, -1.0)))
    best = int(0)
    score = float(-2.0)
    for b in range(nb):
        s = wp.dot(bin_direction(ts, k, b, nb), down)
        if s > score:
            score = s
            best = b
    return best


@wp.kernel
def stores_kernel(ts: ToolSet, ss: StoreSet, trays: TraySet, n_tools: int, charts: wp.array[Chart],
                  tool_body: wp.array2d[wp.int32], cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform],
                  body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                  body_com: wp.array[wp.vec3], base: wp.array2d[wp.float32], area: wp.array[wp.float64],
                  dt: wp.float64, load: wp.array2d[wp.float64], tray_vol: wp.array2d[wp.float64],
                  h: wp.array2d[wp.float64], to_tools: wp.array2d[wp.float64], cwet: wp.array2d[wp.int32],
                  dripped: wp.array2d[wp.float64], picked: wp.array2d[wp.float64], nip: wp.array2d[wp.vec3],
                  nip_bin: wp.array2d[wp.vec2], sp: Species, thixo: int, lam: wp.array2d[wp.float32],
                  release: wp.array2d[wp.float64], drip_on: int, released: wp.array2d[wp.float64],
                  drip_v: wp.array2d[wp.float64], drip_p: wp.array2d[wp.vec3], drip_u: wp.array2d[wp.vec3],
                  drip_sp: wp.array3d[wp.float64], normal: wp.array2d[wp.float32], hm: wp.array2d[wp.float64],
                  paste_mask: wp.array2d[wp.float32], overflow: wp.array2d[wp.float64],
                  overflow_sp: wp.array3d[wp.float64], over_on: int, dropped: wp.array2d[wp.float64],
                  ssp: StoreSpecies, support: wp.array2d[wp.float32], tick: wp.array[wp.int32],
                  nose_low: wp.array2d[wp.float32], nose_seen: wp.array2d[wp.int32]):
    """Per world, each store tool: the nip split, tray pick-up, drips and the owner's ``release`` (share of the free
    load let go this exchange). With ``drip_on`` (a supply), drips and releases leave at the lowest dripping bin's point
    into the stage-drips channel (``drip_*``, DC5's): the supply casts them along gravity and zeroes the channel.
    A scraping store's overflow (``over_on``) and lifted gob: stores_lift.py."""
    w = wp.tid()
    for k in range(n_tools):
        if ss.has[k] == 0:
            continue
        kc = ts.chart[k]
        ch = charts[kc]
        tb, cb = tool_body[w, k], cbody[w, kc]
        ps = pose_of(ts, k, body_q, tb, cb, clocal[w, kc])
        T = body_q[tb]
        nb, b0, cap = ss.bins[k], ss.start[k], ss.capacity[k]
        bc = int(0)
        if ts.shape[k] == ROLLER:
            bc = contact_bin(ts, k, nb, T, ps.X)
        cell = wp.min(ch.step[0], ch.step[1])
        keep = ss.retained[k] * cap
        free = wp.max(load[w, b0 + bc] - keep, wp.float64(0.0))
        nap = free / ss.area[k]                                  # the bin's free load as a film thickness [m]
        if ts.shape[k] == ROLLER:
            # a bin's arc rolls over as much surface as its own area, so its nap thickness is the one it brought into
            # the nip (a brush's bristles redistribute: it uses its current load)
            held_in = nip_bin[w, k]
            if int(held_in[0]) != bc + 1:
                held_in = wp.vec2(float(bc + 1), float(nap))
                nip_bin[w, k] = held_in
            nap = wp.float64(held_in[1])
        prev = nip[w, k]                                         # (x, y, 1): the centre line's last position
        now = wp.vec2(ps.c[0], ps.c[1])
        heading = nip_heading(ts, k, ps, now - wp.vec2(prev[0], prev[1]))      # a roller: across its axis
        ahead = wp.vec2(heading[0], heading[1])
        dist = heading[2]
        crossing = prev[2] > 0.5 and dist > 1.0e-9
        nip[w, k] = wp.vec3(now[0], now[1], 1.0)
        touching = int(0)
        win = window(ch, ps)
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                c = ch.cell_start + i * ch.ny + j
                p = cell_point(ch, i, j)
                under = tool_under(ts, k, ps, p, cell)
                cov = under.cover
                if ss.scrape[k] == 1:
                    cov = sole_cover(cov)                        # a blade's sole: whole cells (GA's ins)
                if cov <= 0.0:
                    continue
                if ss.fill_profile[k] == 1:                      # a nose: its lowest underside this pass (fill)
                    pass_low(nose_low, nose_seen, tick[w], w, c, under.gap)
                reach = ss.express[k] * nap                      # the nap reaches its expressed share
                if ss.meter[k] > wp.float64(0.0):
                    reach = wp.float64(0.0)                      # a metering brush: where its tips touch (DC6 review)
                if ss.scrape[k] == 1:                            # a blade bears within GA's 0.05 mm (coupling.py:478):
                    reach = wp.float64(ON_BASE)                  # at gap 0 float32 decided it (SC9's knives)
                rides = ss.scrape[k] == 1 and under.gap <= support[w, c] + wp.float32(ON_BASE)   # GA's support
                if wp.float64(under.gap) > wp.float64(base[w, c]) + h[w, c] + reach and not rides:
                    continue                                     # no contact here
                touching = 1
                if thixo != 0:
                    lam[w, c] = 0.0                              # the nip breaks the film's structure (P:796)
                if ss.meter[k] > wp.float64(0.0):
                    continue                                     # a metering brush lays below (stores_meter)
                along = wp.dot(wp.vec2(p[0] - prev[0], p[1] - prev[1]), ahead)
                if ss.scrape[k] == 0 and (not crossing or along <= 0.0 or along > dist):
                    continue                                     # the centre line did not cross it this exchange
                target = (wp.float64(1.0) - ss.split[k]) * (h[w, c] + ss.express[k] * nap)
                if ss.scrape[k] == 1:                            # a blade: only what stands above its underside
                    above = h[w, c] - wp.max(wp.float64(under.gap) - wp.float64(base[w, c]), wp.float64(0.0))
                    target = h[w, c] - ss.split[k] * wp.max(above, wp.float64(0.0))
                dv = (target - h[w, c]) * area[c] * wp.float64(cov)
                room = load[w, b0 + bc] - cap                    # (<= 0) the most the bin can take
                spill = ss.scrape[k] == 1 and over_on == 1       # a full blade's cut goes on to the end bands
                if dv > wp.float64(0.0):
                    dv = wp.min(dv, ss.express[k] * free)        # lay no more than the expressed free load
                else:                                            # take no more than the film, or the bin holds
                    dv = wp.max(dv, -h[w, c] * area[c])
                    if not spill:
                        dv = wp.max(dv, room)
                hn = h[w, c] + dv / area[c]
                if hn < wp.float64(0.0):                         # the nip emptied the cell: h - (h·a)/a can round
                    dv = -h[w, c] * area[c]                      # below 0 (SOFT-2's trowel: -5e-20, then NaN)
                    hn = wp.float64(0.0)
                over = wp.float64(0.0)
                if spill and dv < room:
                    over = room - dv                             # the cut beyond capacity (stores_lift.py)
                    overflow_species(sp, w, c, dv, over, h[w, c], area[c], overflow_sp, k, ssp, b0 + bc)
                    overflow[w, k] = overflow[w, k] + over
                free = free - (dv + over)
                nip_species(sp, bin_conc(ssp, sp.count, w, b0 + bc, load[w, b0 + bc]), w, c, dv, h[w, c], area[c],
                            ssp, k, b0 + bc)
                h[w, c] = hn
                load[w, b0 + bc] = load[w, b0 + bc] - (dv + over)
                to_tools[w, c] = to_tools[w, c] - dv
                if h[w, c] > wp.float64(0.0):
                    cwet[w, c] = 1
        if ss.meter[k] > wp.float64(0.0) and touching == 1 and crossing and dist / float(dt) > U_MIN:
            v = dist / float(dt)                                 # a brush meters its film while it slides (GA)
            hm[w, k] = metered(ts, k, ch, ps, cell, base, area, h, w, ahead, v, normal[w, k], ss.meter[k],
                               ss.meter_max[k], ss.mu, ss.tau, ss.meter_eta[k])
            laid = meter(ts, k, ch, ps, cell, base, area, h, cwet, to_tools, sp,
                         bin_conc(ssp, sp.count, w, b0 + bc, load[w, b0 + bc]), w, hm[w, k], free,
                         cap - load[w, b0 + bc], ssp, b0 + bc)
            free = free - laid
            load[w, b0 + bc] = load[w, b0 + bc] - laid
        if ss.scrape[k] == 1 and touching == 1:                 # a blade fills the lows below it from its carry
            laid = fill_lows(ch, ts, k, ps, cell, base, area, h, cwet, to_tools, sp,
                             bin_conc(ssp, sp.count, w, b0 + bc, load[w, b0 + bc]), w, free, ss.fill_profile[k],
                             paste_mask, ssp, b0 + bc, support, nose_low)
            free = free - laid
            load[w, b0 + bc] = load[w, b0 + bc] - laid
        out_v = wp.float64(0.0)                                  # released + dripped: the drip channel's volume
        out_sp = vec4d()                                         # and its species (stores_species.py)
        low = wp.vec3(0.0, 0.0, 1.0e30)                          # the lowest bin that let go, world
        share = wp.clamp(release[w, k], wp.float64(0.0), wp.float64(1.0))
        lifted = ss.scrape[k] == 1 and touching == 0 and normal[w, k] < LIFT_N   # a blade in a tray's heap rests
        if lifted and tray_at(trays, tray_vol, w, wp.transform_point(T, ts.center[k])) >= 0:   # on it (WP6)
            lifted = False
        stick = wp.float64(0.0)
        pend = wp.float64(0.0)                                   # the pending overflow falls with a lifted gob
        if lifted:
            stick = ss.stick[k] * stick_share(wp.transform_vector(T, ts.normal[k]), ss.down) / wp.float64(nb)
            pend = overflow[w, k]
            dropped[w, k] = dropped[w, k] + pend
            overflow[w, k] = wp.float64(0.0)
        for b in range(nb):
            p = wp.transform_point(T, ts.center[k])
            if ts.shape[k] == ROLLER:
                p = wp.transform_point(T, ts.center[k] + ts.size[k][0] * bin_direction(ts, k, b, nb))
            gone = share * wp.max(load[w, b0 + b] - keep, wp.float64(0.0))       # the owner lets go of the free load
            out_sp = out_sp + let_go(ssp, sp.count, w, k, b0 + b, gone, load[w, b0 + b])
            load[w, b0 + b] = load[w, b0 + b] - gone
            released[w, k] = released[w, k] + gone
            if lifted:                                           # a lifted blade keeps what sticks (stores_lift.py)
                gob = wp.max(load[w, b0 + b] - stick, wp.float64(0.0))
                out_sp = out_sp + let_go(ssp, sp.count, w, k, b0 + b, gob, load[w, b0 + b])
                load[w, b0 + b] = load[w, b0 + b] - gob
                dropped[w, k] = dropped[w, k] + gob
                gone = gone + gob
                if b == 0:
                    gone = gone + pend
            dipped = int(0)
            for t in range(trays.count):
                o = trays.center[t]
                hh = trays.half[t]
                level = o[2] + wp.float32(tray_vol[w, t] / trays.area[t])
                if wp.abs(p[0] - o[0]) <= hh[0] and wp.abs(p[1] - o[1]) <= hh[1] and p[2] < level:
                    dipped = 1
                    take = wp.min((wp.float64(1.0) - wp.exp(-ss.pickup[k] * dt)) * (cap - load[w, b0 + b]),
                                  tray_vol[w, t])                   # never past capacity (WP2 4c)
                    take = wp.max(take, wp.float64(0.0))
                    tray_take(ssp, sp.count, w, k, t, b0 + b, take, tray_vol[w, t])     # its pigment with it
                    load[w, b0 + b] = load[w, b0 + b] + take
                    tray_vol[w, t] = tray_vol[w, t] - take
                    picked[w, k] = picked[w, k] + take
            d = wp.float64(0.0)
            excess = load[w, b0 + b] - ss.held[k] * cap
            if dipped == 0 and not (touching == 1 and b == bc) and excess > wp.float64(0.0):
                d = excess * (wp.float64(1.0) - wp.exp(-ss.drip[k] * dt))
                out_sp = out_sp + let_go(ssp, sp.count, w, k, b0 + b, d, load[w, b0 + b])
                load[w, b0 + b] = load[w, b0 + b] - d
                dripped[w, k] = dripped[w, k] + d
            if gone + d > wp.float64(0.0):
                out_v = out_v + gone + d
                if p[2] < low[2]:
                    low = p
        if drip_on == 1 and out_v > wp.float64(0.0):
            drip_v[w, k] = drip_v[w, k] + out_v
            drip_p[w, k] = low
            drip_u[w, k] = point_velocity(body_qd, com_world(body_q, body_com, tb), tb, low)
            for s in range(sp.count):                            # the bins' species; the overflow's own
                drip_sp[w, k, s] = drip_sp[w, k, s] + out_sp[s]
                if lifted:
                    drip_sp[w, k, s] = drip_sp[w, k, s] + overflow_sp[w, k, s]
        if lifted:
            for s in range(sp.count):
                overflow_sp[w, k, s] = wp.float64(0.0)


def check_bins(label: str, tools: Sequence[Any], tool_speed: float, dt: float) -> list[str]:
    """A roller reads its contact bin at the current pose only: rolling more than a bin's arc per exchange skips
    bins, which then never reach the nip and keep their paint (WP2 review 4b). Warns per roller; returns the names."""
    fast = []
    for k, t in enumerate(tools):
        if getattr(t, "store", None) is None or t.shape != "roller":
            continue
        arc = 2.0 * math.pi * float(t.size[0]) / bins_of(t)
        if tool_speed * dt > arc:
            fast.append(f"tool {k}")
            logging.getLogger("alloy.plugins.surface_flow").warning(
                "%s: roller tool %d rolls up to %.3g mm per exchange (tool_speed %.3g m/s), more than a bin's arc "
                "%.3g mm: some bins can skip the nip and keep their load; use fewer bins or a smaller dt",
                label, k, 1e3 * tool_speed * dt, tool_speed, 1e3 * arc)
    return fast


__all__ = ["BEAR", "ROLLER_BINS", "Store", "StoreSet", "StoreTables", "Tray", "TraySet", "bins_of", "build_stores",
           "check_bins", "nip_species", "store_species", "store_spec", "stores_kernel", "tray_spec"]
