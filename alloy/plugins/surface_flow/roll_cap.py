# SPDX-License-Identifier: Apache-2.0
"""The roll's cap with spill at the blade ends (WP13 for WP15, SCR-WP13-ROLLCAP-1; docs/notes/wp13-roll-cap.md), and a
full scraping blade's overflow laid in the same end bands with no top (GA's carrying knife; WP15's stores_lift.py).

Gym-Anything's concrete finishing (coupling.py:597-636, ``_spill`` :774-823) caps a sliding sole's roll at its prism,
``cap = ½·(rise + lift)²·L_face``, and lays what exceeds it beside the blade: "the tags a mason brushes off".
- ``rise``: the span of the sole's underside over the cells of its footprint the paste reaches (a level sole: 0).
  GA takes the whole footprint (coupling.py:615-616); on a chart wider than the paste that gave SOFT-2's jointer rod,
  resting on a wall's arrises, its radius instead of the groove's depth.
- ``lift``: ``Applicator.roll_lift``, how far a bevelled or turned edge lifts the roll (GA's ROLL_LIFT: float 3 mm,
  trowel 1 mm, knife 1 mm). ``None`` keeps the uncapped roll, bit for bit.
- ``L_face``: the sole's long side; its short side when it slides along its length (the end leads, ``_end_leads``).
- The spill lies in a band SPILL_BAND (20 mm) beyond the long axis' two ends, over the sole's width along the slide,
  deepest at the end; when the end leads, beside its leading corners (not on an edger's lip side). Cells the sole
  covers by more than 25 % take none. No bead rises above ``max(s_low, s0) + rise + lift`` (``s0``: the lowest
  strip's paste ahead). A band cell takes its share of the whole band: the share a chart wall cuts off stays in the
  roll and is offered again the next substep (GA counts it as gone). What does not fit under the top stays in the roll
  (GA's SPILL_WIDEN = 0). Species go at the roll's mixed concentration (GA's packet). An edger's rise is its flat's.

Here the roll is film on the cells ahead of the tool (``tool_kernels``: the zone with ``push`` set and no cover, the
slope rule's 45° wedge), so it is measured on the grid with no state: along each strip of zone cells (the push
direction's main axis, as ``roll_excess``), the wedge runs from the footprint while the surface falls more than
TOE_FALL·dx per cell, and ``s0``, the surface where it stops, is the paste the roll rides on. The roll is
``Σ (s - s0)⁺·A`` over the wedges. Paste beyond the toe never counts (a sunk sole's layer ahead, lows and ridges
further on, a sloped slab). The spill is taken from the wedges in proportion to ``(s - s0)⁺`` and laid in a fixed
order; species move with it at the roll's concentration; the volume and species ledgers are exact (the last band cell
takes the remainder). The geometry runs here, once per exchange (one thread per world); the substep runs per strip and
per band row in roll_cap_kernels.py, with sums in a fixed order: float64, deterministic, no atomics, no host syncs.
Plane charts only.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from alloy.plugins.surface_flow import roll_cap_kernels as RK
from alloy.plugins.surface_flow.cell_kernels import Species
from alloy.plugins.surface_flow.charts import PLANE, Chart
from alloy.plugins.surface_flow.kernels import Cells, ToolField
from alloy.plugins.surface_flow.roll_cap_kernels import SPILL_BAND, RollGeom, RollWork, perp, vec4d
from alloy.plugins.surface_flow.tool_kernels import FRONT
from alloy.plugins.surface_flow.tool_geometry import (DISC, EDGER, RECT, V_PUSH, Pose, ToolSet, Under, along, cell_at,
                                                      cell_point, centre_under, edge_of, in_cell, lip_in, lip_side,
                                                      pose_of, relative, tool_under, under_at, widen, window)

TOUCH = wp.constant(1.0e-6)          # m: a covered cell whose surface is within this of the sole bears on paste
@wp.func
def _sole_axes(shape: int, size: wp.vec2, a: wp.vec3, ahead: wp.vec2) -> wp.vec4:
    """(long axis x, y, half length, half width) on the chart plane: a rect's (or an edger's) longer half extent, a
    roller's axis, a disc broadside to the slide."""
    a2 = wp.normalize(wp.vec2(a[0], a[1]) + wp.vec2(1.0e-12, 0.0))
    if shape == DISC:
        p = perp(ahead)
        return wp.vec4(p[0], p[1], size[0], size[0])
    if shape == RECT or shape == EDGER:         # an edger is a rect with a profile on one side
        if size[0] >= size[1]:
            return wp.vec4(a2[0], a2[1], size[0], size[1])
        p = perp(a2)
        return wp.vec4(p[0], p[1], size[1], size[0])
    return wp.vec4(a2[0], a2[1], size[1], size[0])          # a roller: (radius, half width along its axis)


@wp.func
def _on_flat(ts: ToolSet, k: int, ps: Pose, u: Under) -> bool:
    """An edger's rise is its bearing flat's: its quarter round and lip (in from the lip face by less than the radius)
    take no part (GA report §3.7)."""
    if ts.shape[k] != EDGER:
        return True
    return lip_in(u, ts.size[k], ts.edge[k], ps) >= ts.edge[k][0]


@wp.func
def _shut(ts: ToolSet, k: int, ps: Pose, lng: wp.vec2, end: int) -> int:
    """When an edger's end leads, its lip (down the form's face) closes the roll's end on its side: the band beside
    that corner would lie on the form's top, and GA lays nothing there (report §3.7)."""
    if ts.shape[k] != EDGER or end == 0:
        return 0
    s = lip_side(ps) * ts.edge[k][3]
    y = wp.dot(wp.vec2(s[0], s[1]), perp(lng))
    if wp.abs(y) < 0.5:                                      # the lip is not on a long side
        return 0
    return wp.where(y > 0.0, 1, -1)


@wp.func
def _footrow(ts: ToolSet, k: int, ch: Chart, cells: Cells, ps: Pose, base: wp.array2d[wp.float32],
             h: wp.array2d[wp.float64], w: int, i: int, win: wp.vec4i, cell: float) -> wp.vec3:
    """Row i of tool k's footprint: (the lowest point of its underside over the covered cells, and the lowest and
    highest over those the paste reaches)."""
    lo_all = float(1.0e30)
    lo = float(1.0e30)
    hi = float(-1.0e30)
    for j in range(win[2], win[3] + 1):
        c = cell_at(ch, i, j)
        u = under_at(ts, k, ch, cells, ps, i, j, c, cell)
        if u.cover <= 0.0:
            continue
        lo_all = wp.min(lo_all, u.gap)
        if _on_flat(ts, k, ps, u) and h[w, c] + wp.float64(base[w, c]) >= wp.float64(u.gap - TOUCH):
            lo = wp.min(lo, u.gap)
            hi = wp.max(hi, u.gap)
    return wp.vec3(lo_all, lo, hi)


@wp.func
def _takes(ts: ToolSet, carry: wp.array[wp.int32], k: int) -> bool:
    """A ceiling tool with a lift (its roll's cap) or a scraping store whose cut beyond capacity overflows
    (stores_lift.py; GA's carrying knife)."""
    return (ts.roll_lift[k] >= 0.0 and ts.ceiling[k] == 1) or carry[k] == 1


@wp.kernel
def footprint_kernel(ts: ToolSet, charts: wp.array[Chart], cells: Cells, tool_body: wp.array2d[wp.int32],
                     cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                     base: wp.array2d[wp.float32], h: wp.array2d[wp.float64], carry: wp.array[wp.int32],
                     foot: wp.array3d[wp.vec3]):
    """Once per exchange, before ``roll_geom_kernel``: each capped tool's footprint, one thread per world, tool and
    row of its window."""
    w, k, i = wp.tid()
    kc = ts.chart[k]
    ch = charts[kc]
    if not _takes(ts, carry, k) or ch.kind != PLANE:
        return
    ps = pose_of(ts, k, body_q, tool_body[w, k], cbody[w, kc], clocal[w, kc])
    win = window(ch, ps)
    if i >= win[0] and i <= win[1]:
        foot[w, k, i] = _footrow(ts, k, ch, cells, ps, base, h, w, i, win, edge_of(ch))


@wp.func
def _footprint(foot: wp.array3d[wp.vec3], w: int, k: int, win: wp.vec4i) -> wp.vec2:
    """(s_low, rise): the lowest point and span of tool k's underside over the covered cells the paste reaches (the
    whole footprint's lowest point, rise 0, when it reaches none). Min and max over the rows: exact in any order."""
    lo_all = float(1.0e30)
    lo = float(1.0e30)
    hi = float(-1.0e30)
    for i in range(win[0], win[1] + 1):
        f = foot[w, k, i]
        lo_all = wp.min(lo_all, f[0])
        lo = wp.min(lo, f[1])
        hi = wp.max(hi, f[2])
    if hi < lo:
        return wp.vec2(lo_all, 0.0)
    return wp.vec2(lo, hi - lo)


@wp.func
def _set_geom(g: RollGeom, w: int, k: int, ts: ToolSet, ch: Chart, ps: Pose, foot: wp.array3d[wp.vec3],
              win: wp.vec4i, ahead: wp.vec2):
    fp = _footprint(foot, w, k, win)
    ax = _sole_axes(ts.shape[k], ts.size[k], ps.a, ahead)
    lng = wp.vec2(ax[0], ax[1])
    lead = wp.dot(ahead, lng)
    end = int(0)
    face = 2.0 * ax[2]
    if wp.abs(lead) > wp.abs(wp.dot(ahead, perp(lng))):
        end = wp.where(lead >= 0.0, 1, -1)
        face = 2.0 * ax[3]
    rl = wp.float64(fp[1]) + wp.float64(ts.roll_lift[k])
    g.cap[w, k] = wp.float64(0.5) * rl * rl * wp.float64(face)
    g.rise_lift[w, k] = rl
    g.s_low[w, k] = wp.float64(fp[0])
    g.c[w, k] = wp.vec2(ps.c[0], ps.c[1])
    g.ldir[w, k] = lng
    g.half[w, k] = wp.vec2(ax[2], ax[3])
    g.end[w, k] = end
    g.shut[w, k] = _shut(ts, k, ps, lng, end)
    if wp.abs(ahead[0]) >= wp.abs(ahead[1]):
        g.axis[w, k] = wp.vec2i(0, wp.where(ahead[0] > 0.0, 1, -1))
    else:
        g.axis[w, k] = wp.vec2i(1, wp.where(ahead[1] > 0.0, 1, -1))
    n = wp.max(FRONT, int(SPILL_BAND / wp.min(ch.step[0], ch.step[1])) + 2)
    g.win[w, k] = widen(ch, win, n)
    g.on[w, k] = 1


@wp.kernel
def roll_geom_kernel(ts: ToolSet, n_tools: int, charts: wp.array[Chart], prof: wp.array[wp.vec3],
                     tool_body: wp.array2d[wp.int32], cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform],
                     body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                     foot: wp.array3d[wp.vec3], carry: wp.array[wp.int32], g: RollGeom):
    """Once per exchange, after ``tools_kernel``: each capped tool's sole, cap and window (one thread per world)."""
    w = wp.tid()
    for k in range(n_tools):
        g.on[w, k] = 0
        kc = ts.chart[k]
        ch = charts[kc]
        if not _takes(ts, carry, k) or ch.kind != PLANE:
            continue
        tb, cb = tool_body[w, k], cbody[w, kc]
        ps = pose_of(ts, k, body_q, tb, cb, clocal[w, kc])
        win = window(ch, ps)
        if win[1] < win[0]:
            continue
        cell = edge_of(ch)
        centre = centre_under(ts, k, ch, prof, ps, cell)
        slide = along(relative(body_q, body_qd, body_com, tb, cb, ps, centre), centre)
        if wp.length(slide) <= V_PUSH:                          # as tools_kernel: no roll forms
            continue
        ahead = wp.normalize(in_cell(slide, centre) + wp.vec2(1.0e-12, 0.0))
        _set_geom(g, w, k, ts, ch, ps, foot, win, ahead)


@wp.struct
class Poses:
    """The tools' and charts' bodies at the exchange's poses (a scraping blade's own cover of the band cells)."""

    body_q: wp.array[wp.transform]
    tool_body: wp.array2d[wp.int32]
    cbody: wp.array2d[wp.int32]
    clocal: wp.array2d[wp.transform]


@wp.func
def _sole(ts: ToolSet, k: int, pz: Poses, w: int) -> Pose:
    kc = ts.chart[k]
    return pose_of(ts, k, pz.body_q, pz.tool_body[w, k], pz.cbody[w, kc], pz.clocal[w, kc])


@wp.func
def _ov_fit(cells: Cells, ch: Chart, tf: ToolField, ts: ToolSet, ps: Pose, w: int, k: int, g: RollGeom, i: int,
            j: int, share: wp.float64) -> wp.float64:
    """What band cell (i, j) takes of a carry tool's overflow: its share of the whole band, with no top (GA). A cell
    any ceiling or the blade itself covers by more than 25 % takes none (GA's _cover(sole) <= 0.25): its share stays
    pending."""
    c = cell_at(ch, i, j)
    p = cell_point(ch, i, j)
    if tf.cover[w, c] > RK.BAND_COVER or tool_under(ts, k, ps, p, edge_of(ch)).cover > RK.BAND_COVER:
        return wp.float64(0.0)
    wt = RK.band_weight(g, w, k, p)
    if wt <= 0.0:
        return wp.float64(0.0)
    return share * wp.float64(wt) * wp.float64(cells.area[c])


@wp.func
def _ov_species(overflow_sp: wp.array3d[wp.float64], sp: Species, w: int, k: int, frac: wp.float64) -> vec4d:
    dm = vec4d()
    for q in range(sp.count):
        dm[q] = overflow_sp[w, k, q] * frac
    return dm


@wp.func
def _ov_lay(cells: Cells, w: int, c: int, v: wp.float64, dm: vec4d, h: wp.array2d[wp.float64], sp: Species,
            cwet: wp.array2d[wp.int32], to_tools: wp.array2d[wp.float64]):
    """Lays v [m³] and dm [kg] on cell c: the film takes them back from the tools (to_tools), the species as supplied
    (the stores kernel booked them out with the cut)."""
    RK.lay_cell(cells, w, c, v, dm, h, sp, cwet)
    for q in range(sp.count):
        sp.supplied[w, c, q] = sp.supplied[w, c, q] + dm[q]
    to_tools[w, c] = to_tools[w, c] - v


@wp.kernel
def overflow_band_kernel(cells: Cells, charts: wp.array[Chart], ts: ToolSet, k: int, g: RollGeom,
                         overflow: wp.array2d[wp.float64], work: RollWork):
    """Per world: the band's whole weight on the chart's lattice, cells off the chart included, so each cell's share
    is exact and the off-chart share (GA: gone, coupling.py:779-781) is known. total = (pending, share per unit
    weight, off-chart volume, -)."""
    w = wp.tid()
    work.go[w] = 0
    if g.on[w, k] == 0 or overflow[w, k] <= wp.float64(0.0):
        return
    ch = charts[ts.chart[k]]
    a = wp.float64(cells.area[cell_at(ch, 0, 0)])                # a plane chart's cells are alike
    c, r = g.c[w, k], wp.length(g.half[w, k]) + RK.SPILL_BAND + wp.max(ch.step[0], ch.step[1])
    whole = wp.float64(0.0)
    off = wp.float64(0.0)
    i0, i1 = int(wp.floor((c[0] - r + ch.half[0]) / ch.step[0])), int(wp.floor((c[0] + r + ch.half[0]) / ch.step[0]))
    j0, j1 = int(wp.floor((c[1] - r + ch.half[1]) / ch.step[1])), int(wp.floor((c[1] + r + ch.half[1]) / ch.step[1]))
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            wa = wp.float64(RK.band_weight(g, w, k, cell_point(ch, i, j))) * a
            whole += wa
            if i < 0 or i >= ch.nx or j < 0 or j >= ch.ny:
                off += wa
    if whole <= wp.float64(0.0):
        return
    v0 = overflow[w, k]
    work.total[w] = vec4d(v0, v0 / whole, v0 * (off / whole), wp.float64(0.0))
    work.go[w] = 1


@wp.kernel
def overflow_fit_kernel(cells: Cells, charts: wp.array[Chart], ts: ToolSet, k: int, g: RollGeom, tf: ToolField,
                        pz: Poses, work: RollWork):
    """Each band row's share of tool k's pending overflow, summed along the row in order (one thread a row)."""
    w, i = wp.tid()
    win = g.win[w, k]
    if work.go[w] == 0 or i < win[0] or i > win[1]:
        return
    ch, ps = charts[ts.chart[k]], _sole(ts, k, pz, w)
    vol = wp.float64(0.0)
    last = int(-1)
    for j in range(win[2], win[3] + 1):
        f = _ov_fit(cells, ch, tf, ts, ps, w, k, g, i, j, work.total[w][1])
        if f > wp.float64(0.0):
            vol += f
            last = j
    work.fit[w, i] = vol
    work.last[w, i] = last


@wp.kernel
def overflow_sum_kernel(k: int, g: RollGeom, work: RollWork):
    """The rows in order: what lands (at most the pending less the off-chart share: total[3]), its last cell."""
    w = wp.tid()
    if work.go[w] == 0:
        return
    win = g.win[w, k]
    vol = wp.float64(0.0)
    end = wp.vec2i(-1, -1)
    for i in range(win[0], win[1] + 1):
        vol += work.fit[w, i]
        if work.last[w, i] >= 0:
            end = wp.vec2i(i, work.last[w, i])
    t = work.total[w]
    work.total[w] = vec4d(t[0], t[1], t[2], wp.max(wp.min(vol, t[0] - t[2]), wp.float64(0.0)))
    work.took[w] = vol
    work.end_cell[w] = end


@wp.kernel
def overflow_lay_kernel(cells: Cells, charts: wp.array[Chart], ts: ToolSet, k: int, g: RollGeom, tf: ToolField,
                        pz: Poses, overflow_sp: wp.array3d[wp.float64], work: RollWork, h: wp.array2d[wp.float64],
                        sp: Species, cwet: wp.array2d[wp.int32], to_tools: wp.array2d[wp.float64]):
    """Band row i lays its cells' shares (scaled to what lands) with the overflow's species; the band's last cell is
    left to ``overflow_last_kernel``."""
    w, i = wp.tid()
    win = g.win[w, k]
    if work.go[w] == 0 or work.took[w] <= wp.float64(0.0) or i < win[0] or i > win[1]:
        return
    ch, ps = charts[ts.chart[k]], _sole(ts, k, pz, w)
    total, end = work.total[w], work.end_cell[w]
    scale = total[3] / work.took[w]
    vol = wp.float64(0.0)
    acc = vec4d()
    for j in range(win[2], win[3] + 1):
        c = cell_at(ch, i, j)
        v = _ov_fit(cells, ch, tf, ts, ps, w, k, g, i, j, total[1]) * scale
        if v <= wp.float64(0.0) or (i == end[0] and j == end[1]):
            continue
        dm = _ov_species(overflow_sp, sp, w, k, v / total[0])
        _ov_lay(cells, w, c, v, dm, h, sp, cwet, to_tools)
        vol += v
        acc += dm
    work.laid[w, i] = vol
    work.laid_sp[w, i] = acc


@wp.kernel
def overflow_last_kernel(cells: Cells, charts: wp.array[Chart], ts: ToolSet, k: int, g: RollGeom, work: RollWork,
                         overflow: wp.array2d[wp.float64], overflow_sp: wp.array3d[wp.float64],
                         h: wp.array2d[wp.float64], sp: Species, cwet: wp.array2d[wp.int32],
                         to_tools: wp.array2d[wp.float64], spill: wp.array2d[wp.float64],
                         off: wp.array2d[wp.float64], off_sp: wp.array3d[wp.float64]):
    """The band's last cell takes the remainder; the off-chart share is booked as gone (``tool_overflow_off``); the
    pending overflow and its species fall by exactly both (exact ledgers). Covered cells' shares stay pending."""
    w = wp.tid()
    if work.go[w] == 0:
        return
    total, land = work.total[w], work.total[w][3]
    if work.took[w] > wp.float64(0.0):
        win = g.win[w, k]
        vol = wp.float64(0.0)
        acc = vec4d()
        for i in range(win[0], win[1] + 1):
            vol += work.laid[w, i]
            acc += work.laid_sp[w, i]
        end = work.end_cell[w]
        every = _ov_species(overflow_sp, sp, w, k, land / total[0])
        _ov_lay(cells, w, cell_at(charts[ts.chart[k]], end[0], end[1]), land - vol, every - acc, h, sp, cwet,
                to_tools)
    gone = _ov_species(overflow_sp, sp, w, k, total[2] / total[0])
    laid = _ov_species(overflow_sp, sp, w, k, land / total[0])
    for q in range(sp.count):
        overflow_sp[w, k, q] = overflow_sp[w, k, q] - laid[q] - gone[q]
        off_sp[w, k, q] = off_sp[w, k, q] + gone[q]
    overflow[w, k] = overflow[w, k] - land - total[2]
    off[w, k] = off[w, k] + total[2]
    spill[w, k] = spill[w, k] + land


class RollCap:
    """Host side: the per (world, tool) geometry, its scratch and the launches (exchange: after ``tools_kernel``;
    substep: after the roll's slope passes, per capped tool in order; overflow: after the stores kernel, per scraping
    store). ``build`` returns None when no tool has a ``roll_lift`` and none scrapes."""

    def __init__(self, ctx: Any, tools: Any, worlds: int, add_state: Any, lines: int, species: int) -> None:
        self.tools, n = tools, (worlds, tools.count)
        self.capped = [k for k, s in enumerate(tools.specs) if s.roll_lift is not None]
        self.carry = [k for k, s in enumerate(tools.specs) if s.store is not None and s.store.scrape]
        self.carry_flags = wp.array([int(k in self.carry) for k in range(tools.count)], dtype=wp.int32,
                                    device=ctx.device)
        self.spill = add_state("tool_roll_spill", ctx.zeros(n, wp.float64))   # [W, tools] m³ laid at the ends
        self.off = add_state("tool_overflow_off", ctx.zeros(n, wp.float64))   # [W, tools] m³ overflow off the chart
        self.off_sp = add_state("tool_overflow_off_sp", ctx.zeros((*n, species), wp.float64))   # [W, tools, K] kg
        g = self.geom = RollGeom()
        g.on, g.end, g.shut = ctx.zeros(n, wp.int32), ctx.zeros(n, wp.int32), ctx.zeros(n, wp.int32)
        g.axis, g.win = ctx.zeros(n, wp.vec2i), ctx.zeros(n, wp.vec4i)
        g.c, g.ldir, g.half = ctx.zeros(n, wp.vec2), ctx.zeros(n, wp.vec2), ctx.zeros(n, wp.vec2)
        g.cap, g.rise_lift, g.s_low = ctx.zeros(n, wp.float64), ctx.zeros(n, wp.float64), ctx.zeros(n, wp.float64)
        self.lines, wl = lines, (worlds, lines)            # strips and band rows: at most a chart's longest side
        self.foot = ctx.zeros((worlds, tools.count, lines), wp.vec3)     # footprint rows (footprint_kernel)
        wk = self.work = RollWork()
        wk.wedge, wk.last = ctx.zeros(wl, wp.vec3i), ctx.zeros(wl, wp.int32)
        wk.s0, wk.roll, wk.gave, wk.fit, wk.laid = (ctx.zeros(wl, wp.float64) for _ in range(5))
        wk.gave_sp, wk.laid_sp = ctx.zeros(wl, vec4d), ctx.zeros(wl, vec4d)
        wk.go, wk.end_cell = ctx.zeros(worlds, wp.int32), ctx.zeros(worlds, wp.vec2i)
        wk.total, wk.took_sp = ctx.zeros(worlds, vec4d), ctx.zeros(worlds, vec4d)
        wk.took = ctx.zeros(worlds, wp.float64)

    @classmethod
    def build(cls, ctx: Any, tools: Any, worlds: int, add_state: Any, shapes: list, species: int = 1) -> RollCap | None:
        """``shapes``: the charts' (nx, ny); ``species``: the film's species slots (>= 1)."""
        if tools is None or all(s.roll_lift is None and (s.store is None or not s.store.scrape) for s in tools.specs):
            return None
        return cls(ctx, tools, worlds, add_state, max(max(nx, ny) for nx, ny in shapes), species)

    def exchange(self, flow: Any, state: Any) -> None:
        """Each capped tool's sole, cap and window at the exchange's poses (after ``tools_kernel``)."""
        t, W, dev = flow.tables, self.geom.on.shape[0], flow.ctx.device
        wp.launch(footprint_kernel, dim=(W, self.tools.count, self.lines),
                  inputs=[self.tools.set, t.charts, flow.cells, self.tools.body, t.cbody, t.clocal, state.body_q,
                          flow.base, flow.h, self.carry_flags], outputs=[self.foot], device=dev)
        wp.launch(roll_geom_kernel, dim=W,
                  inputs=[self.tools.set, self.tools.count, t.charts, t.prof, self.tools.body, t.cbody, t.clocal,
                          state.body_q, state.body_qd, flow.ctx.model.body_com, self.foot, self.carry_flags],
                  outputs=[self.geom], device=dev)

    def overflow(self, flow: Any, state: Any) -> None:
        """A full scraping store's cut beyond capacity (``flow.tool_overflow``, stores_lift.py), once per exchange after
        the stores kernel: laid in the end bands with no top; the off-chart share is booked as gone
        (``tool_overflow_off``); covered cells' shares stay pending."""
        W, dev, g, wk = self.geom.on.shape[0], flow.ctx.device, self.geom, self.work
        lines, head = (W, self.lines), [flow.cells, flow.tables.charts, self.tools.set]
        pz = Poses()
        pz.body_q, pz.tool_body = state.body_q, self.tools.body
        pz.cbody, pz.clocal = flow.tables.cbody, flow.tables.clocal
        out = [flow.h, flow.sp, flow.cwet, flow.to_tools]
        for k in self.carry:
            wp.launch(overflow_band_kernel, dim=W, inputs=[*head, k, g, flow.tool_overflow], outputs=[wk], device=dev)
            wp.launch(overflow_fit_kernel, dim=lines, inputs=[*head, k, g, flow.tf, pz], outputs=[wk], device=dev)
            wp.launch(overflow_sum_kernel, dim=W, inputs=[k, g], outputs=[wk], device=dev)
            wp.launch(overflow_lay_kernel, dim=lines, inputs=[*head, k, g, flow.tf, pz, flow.tool_overflow_sp, wk],
                      outputs=out, device=dev)
            wp.launch(overflow_last_kernel, dim=W, inputs=[*head, k, g, wk],
                      outputs=[flow.tool_overflow, flow.tool_overflow_sp, *out, self.spill, self.off, self.off_sp],
                      device=dev)

    def substep(self, flow: Any, tick: Any, stride: int) -> None:
        """The cap and spill, after the roll's slope passes (roll_cap_kernels.py: the stages and their order)."""
        W, dev, g, wk = self.geom.on.shape[0], flow.ctx.device, self.geom, self.work
        lines, head = (W, self.lines), [flow.cells, flow.tables.charts, self.tools.set]
        for k in self.capped:
            due = [k, tick, stride, g]
            wp.launch(RK.wedge_kernel, dim=lines, inputs=[*head, *due, flow.tf, flow.base, flow.h], outputs=[wk],
                      device=dev)
            wp.launch(RK.roll_kernel, dim=W, inputs=due, outputs=[wk], device=dev)
            wp.launch(RK.fit_kernel, dim=lines, inputs=[*head, *due, flow.tf, flow.base, flow.h], outputs=[wk],
                      device=dev)
            wp.launch(RK.fit_sum_kernel, dim=W, inputs=due, outputs=[wk], device=dev)
            wp.launch(RK.take_kernel, dim=lines, inputs=[*head, *due, flow.base, wk], outputs=[flow.h, flow.sp],
                      device=dev)
            wp.launch(RK.take_sum_kernel, dim=W, inputs=due, outputs=[wk], device=dev)
            wp.launch(RK.lay_kernel, dim=lines, inputs=[*head, *due, flow.tf, flow.base, wk],
                      outputs=[flow.h, flow.sp, flow.cwet], device=dev)
            wp.launch(RK.last_kernel, dim=W, inputs=[*head, *due, flow.tf, flow.base, wk],
                      outputs=[flow.h, flow.sp, flow.cwet, self.spill], device=dev)


__all__ = ["SPILL_BAND", "RollCap", "RollGeom", "footprint_kernel", "roll_geom_kernel"]
