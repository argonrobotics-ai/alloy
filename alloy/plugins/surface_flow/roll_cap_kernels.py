# SPDX-License-Identifier: Apache-2.0
"""The roll cap's substep, in parallel (WP13; the law, the grid method and the geometry are in roll_cap.py).

Each film substep, after the roll's slope passes, per capped tool in order (``RollCap.substep``):
1. ``wedge_kernel`` (one thread per world and strip): the strip's wedge and its roll above the paste it rides on.
2. ``roll_kernel`` (per world): the strips summed in order. At or under the cap nothing more runs (bit for bit);
   past it, the roll's top and each band cell's share.
3. ``fit_kernel`` (per world and band row): what each band cell takes, up to the top; ``fit_sum_kernel`` (per world):
   the band's total and its last cell.
4. ``take_kernel`` (per world and strip): each wedge gives (fit / roll) of its film above s0, with its species;
   ``take_sum_kernel`` (per world): the total given.
5. ``lay_kernel`` (per world and band row): every band cell but the last lays its part, with the roll's species;
   ``last_kernel`` (per world): the last band cell takes the remainder (exact ledgers) and the spill is booked.

Every sum runs over strips or rows in a fixed order, in float64, with no atomics or host syncs: deterministic and
graph-capturable. A strip's cells and a row's cells belong to that thread alone.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.surface_flow.cell_kernels import Species
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.kernels import Cells, ToolField
from alloy.plugins.surface_flow.tool_geometry import ToolSet, cell_at, cell_point

SPILL_BAND = wp.constant(0.020)      # m: the band beyond the blade's ends the overflow lies in (GA SPILL_BAND)
BAND_COVER = wp.constant(0.25)       # a cell the sole covers by more than this takes no spill (GA ``_cover <= 0.25``)
TOE_FALL = wp.constant(0.25)         # the wedge ends where the surface falls less than this x dx per cell
vec4d = wp.types.vector(length=4, dtype=wp.float64)    # per species (cell_kernels.MAX_SPECIES = 4)


@wp.struct
class RollGeom:
    """Per (world, tool), once per exchange (``roll_cap.roll_geom_kernel``)."""

    on: wp.array2d[wp.int32]          # 1: a ceiling tool with a lift, sliding (its roll forms)
    axis: wp.array2d[wp.vec2i]        # the strips: (0: along i, 1: along j; step ±1 toward the push)
    win: wp.array2d[wp.vec4i]         # footprint + zone + band (i0, i1, j0, j1)
    c: wp.array2d[wp.vec2]            # the sole's centre on the chart
    ldir: wp.array2d[wp.vec2]         # its long axis (unit)
    half: wp.array2d[wp.vec2]         # (half length along ``ldir``, half width across)
    end: wp.array2d[wp.int32]         # 0: broadside; ±1: the end leads (sign along ``ldir``)
    shut: wp.array2d[wp.int32]        # ±1: the side (across ``ldir``) an edger's lip closes when its end leads; 0
    cap: wp.array2d[wp.float64]       # ½·(rise + lift)²·L_face [m³]
    rise_lift: wp.array2d[wp.float64]  # rise + lift [m]
    s_low: wp.array2d[wp.float64]     # the sole's lowest point over the paste, above the chart surface [m]


@wp.struct
class RollWork:
    """The substep's scratch for the tool being capped: per (world, strip or band row) and per world."""

    wedge: wp.array2d[wp.vec3i]       # strip: (start, first, last) along it; first < 0: no wedge
    s0: wp.array2d[wp.float64]        # strip: the paste its wedge rides on [m]
    roll: wp.array2d[wp.float64]      # strip: its roll above s0 [m³]
    gave: wp.array2d[wp.float64]      # strip: what its wedge gave [m³]
    gave_sp: wp.array2d[vec4d]        # strip: the species it gave [kg]
    fit: wp.array2d[wp.float64]       # band row: what its cells take [m³]
    last: wp.array2d[wp.int32]        # band row: its last cell that takes any (j), or -1
    laid: wp.array2d[wp.float64]      # band row: what it laid [m³]
    laid_sp: wp.array2d[vec4d]        # band row: the species it laid [kg]
    go: wp.array[wp.int32]            # world: 1 while this substep spills
    total: wp.array[vec4d]            # world: (roll, top, share per unit weight, band fit)
    took: wp.array[wp.float64]        # world: what the wedges gave [m³]
    took_sp: wp.array[vec4d]          # world: the species they gave [kg]
    end_cell: wp.array[wp.vec2i]      # world: the band's last cell (i, j): it takes the remainder


@wp.struct
class Wedge:
    """One strip's roll: cells ``start`` .. ``last`` (along the strip) above ``s0``, the paste it rides on. ``start``
    is the tool's partly covered leading row when there is one (the roll's apex leans on the face there; the plough
    sends what stands above the ceiling there on into the roll), else ``first``, the first zone cell."""

    start: int
    first: int
    last: int
    s0: wp.float64


@wp.func
def perp(v: wp.vec2) -> wp.vec2:
    return wp.vec2(-v[1], v[0])


@wp.func
def _cell(ch: Chart, a: int, l: int, m: int) -> int:
    if a == 0:
        return cell_at(ch, m, l)
    return cell_at(ch, l, m)


@wp.func
def _surface(h: wp.array2d[wp.float64], base: wp.array2d[wp.float32], w: int, c: int) -> wp.float64:
    return h[w, c] + wp.float64(base[w, c])


@wp.func
def _zone(tf: ToolField, w: int, c: int) -> bool:
    d = tf.push[w, c]
    return tf.cover[w, c] <= 0.0 and (d[0] != 0.0 or d[1] != 0.0)


@wp.func
def _first(ch: Chart, tf: ToolField, w: int, a: int, step: int, l: int, lo: int, hi: int) -> int:
    """Strip ``l``'s first zone cell past the footprint (-1: none)."""
    m = wp.where(step > 0, lo, hi)
    seen = int(0)
    for _it in range(hi - lo + 1):
        c = _cell(ch, a, l, m)
        if tf.cover[w, c] > 0.0:
            seen = 1
        elif seen == 1 and _zone(tf, w, c):
            return m
        m += step
    return -1


@wp.func
def _wedge(ch: Chart, tf: ToolField, h: wp.array2d[wp.float64], base: wp.array2d[wp.float32], w: int, a: int,
           step: int, l: int, lo: int, hi: int, fall: wp.float64) -> Wedge:
    """Strip ``l``: the first zone cell past the footprint, then on while the surface falls more than ``fall``."""
    wd = Wedge()
    wd.first = _first(ch, tf, w, a, step, l, lo, hi)
    wd.start = wd.first
    wd.last = -1
    wd.s0 = wp.float64(0.0)
    if wd.first < 0:
        return wd
    lead = wd.first - step
    if lead >= lo and lead <= hi:
        cv = tf.cover[w, _cell(ch, a, l, lead)]
        if cv > 0.0 and cv < 1.0:
            wd.start = lead
    cur = wd.first
    s = _surface(h, base, w, _cell(ch, a, l, cur))
    s0 = s
    for _it in range(hi - lo + 1):
        nxt = cur + step
        if nxt < lo or nxt > hi:
            break
        cn = _cell(ch, a, l, nxt)
        sn = _surface(h, base, w, cn)
        s0 = sn
        if not _zone(tf, w, cn) or sn >= s - fall:
            break
        cur = nxt
        s = sn
    wd.last = cur
    wd.s0 = s0
    return wd


@wp.func
def _above(h: wp.array2d[wp.float64], base: wp.array2d[wp.float32], w: int, c: int, s0: wp.float64) -> wp.float64:
    """A wedge cell's roll depth: its surface above s0."""
    return wp.max(_surface(h, base, w, c) - s0, wp.float64(0.0))


@wp.func
def _strips(g: RollGeom, w: int, k: int) -> wp.vec4i:
    """(lo, hi, l0, l1): the window along the strips and across them."""
    win = g.win[w, k]
    if g.axis[w, k][0] == 0:
        return win
    return wp.vec4i(win[2], win[3], win[0], win[1])


@wp.func
def band_weight(g: RollGeom, w: int, k: int, p: wp.vec3) -> float:
    """The band's weight at chart point p: 1 at the blade's end (or side, when the end leads), 0 at SPILL_BAND."""
    rel = wp.vec2(p[0], p[1]) - g.c[w, k]
    lng, hf, e = g.ldir[w, k], g.half[w, k], g.end[w, k]
    x = wp.dot(rel, lng)
    y = wp.dot(rel, perp(lng))
    if e == 0:
        d = wp.abs(x) - hf[0]
        if d <= 0.0 or d > SPILL_BAND or wp.abs(y) > hf[1]:
            return 0.0
        return 1.0 - d / SPILL_BAND
    d = wp.abs(y) - hf[1]
    if y * float(g.shut[w, k]) > 0.0:                       # beside an edger's lip: on the form's top
        return 0.0
    if d <= 0.0 or d > SPILL_BAND or wp.abs(float(e) * x - hf[0]) > SPILL_BAND:
        return 0.0
    return 1.0 - d / SPILL_BAND


@wp.func
def band_total(g: RollGeom, w: int, k: int) -> wp.float64:
    """The whole band's ∫weight dA (cells off the chart included): 2·½·band·(the sole's width) per end, or beside
    the leading end's two sides over ±band (one side beside an edger's lip is shut)."""
    if g.end[w, k] == 0:
        return wp.float64(2.0 * SPILL_BAND * g.half[w, k][1])
    return wp.float64(wp.where(g.shut[w, k] == 0, 2.0, 1.0) * SPILL_BAND * SPILL_BAND)


@wp.func
def _cell_fit(cells: Cells, ch: Chart, tf: ToolField, h: wp.array2d[wp.float64], base: wp.array2d[wp.float32],
              w: int, k: int, g: RollGeom, i: int, j: int, c: int, total: vec4d) -> wp.float64:
    """What band cell c takes of the spill: its share of the whole band (``total[2]`` per unit weight), up to the
    roll's top (``total[1]``)."""
    if tf.cover[w, c] > BAND_COVER or _zone(tf, w, c):
        return wp.float64(0.0)
    wt = band_weight(g, w, k, cell_point(ch, i, j))
    if wt <= 0.0:
        return wp.float64(0.0)
    area = wp.float64(cells.area[c])
    room = wp.max(total[1] - _surface(h, base, w, c), wp.float64(0.0)) * area
    return wp.min(total[2] * wp.float64(wt) * area, room)


@wp.func
def _due(tick: wp.array[wp.int32], stride: int, g: RollGeom, w: int, k: int) -> bool:
    return tick[w] % stride == 0 and g.on[w, k] == 1


@wp.kernel
def wedge_kernel(cells: Cells, charts: wp.array[Chart], ts: ToolSet, k: int, tick: wp.array[wp.int32], stride: int,
                 g: RollGeom, tf: ToolField, base: wp.array2d[wp.float32], h: wp.array2d[wp.float64], work: RollWork):
    """Each strip's wedge and its roll above the paste it rides on (one thread per world and strip)."""
    w, l = wp.tid()
    r = _strips(g, w, k)
    if not _due(tick, stride, g, w, k) or l < r[2] or l > r[3]:
        return
    ch = charts[ts.chart[k]]
    a, step = g.axis[w, k][0], g.axis[w, k][1]
    wd = _wedge(ch, tf, h, base, w, a, step, l, r[0], r[1], wp.float64(TOE_FALL * ch.step[a]))
    vol = wp.float64(0.0)
    m = wd.start
    for _it in range(wp.where(wd.first < 0, 0, wp.abs(wd.last - wd.start) + 1)):
        c = _cell(ch, a, l, m)
        vol += _above(h, base, w, c, wd.s0) * wp.float64(cells.area[c])
        m += step
    work.wedge[w, l] = wp.vec3i(wd.start, wd.first, wd.last)
    work.s0[w, l] = wd.s0
    work.roll[w, l] = vol


@wp.kernel
def roll_kernel(k: int, tick: wp.array[wp.int32], stride: int, g: RollGeom, work: RollWork):
    """The roll over every strip and the lowest strip's s0 (the paste the roll rides on, for its top: a toe left on a
    ridge or the corner's pile does not lift it). Past the cap: the top and the share per unit of band weight."""
    w = wp.tid()
    if not _due(tick, stride, g, w, k):
        return
    r = _strips(g, w, k)
    vol = wp.float64(0.0)
    s0 = wp.float64(1.0e30)
    for l in range(r[2], r[3] + 1):
        if work.wedge[w, l][1] >= 0:
            vol += work.roll[w, l]
            s0 = wp.min(s0, work.s0[w, l])
    work.go[w] = 0
    if vol <= g.cap[w, k]:
        return
    top = wp.max(g.s_low[w, k], s0) + g.rise_lift[w, k]
    work.total[w] = vec4d(vol, top, (vol - g.cap[w, k]) / band_total(g, w, k), wp.float64(0.0))
    work.go[w] = 1


@wp.func
def _spills(tick: wp.array[wp.int32], stride: int, g: RollGeom, work: RollWork, w: int, k: int) -> bool:
    return _due(tick, stride, g, w, k) and work.go[w] == 1


@wp.kernel
def fit_kernel(cells: Cells, charts: wp.array[Chart], ts: ToolSet, k: int, tick: wp.array[wp.int32], stride: int,
               g: RollGeom, tf: ToolField, base: wp.array2d[wp.float32], h: wp.array2d[wp.float64], work: RollWork):
    """What each band cell of row i takes, summed along the row in order (one thread per world and row)."""
    w, i = wp.tid()
    win = g.win[w, k]
    if not _spills(tick, stride, g, work, w, k) or i < win[0] or i > win[1]:
        return
    ch = charts[ts.chart[k]]
    vol = wp.float64(0.0)
    last = int(-1)
    for j in range(win[2], win[3] + 1):
        f = _cell_fit(cells, ch, tf, h, base, w, k, g, i, j, cell_at(ch, i, j), work.total[w])
        if f > wp.float64(0.0):
            vol += f
            last = j
    work.fit[w, i] = vol
    work.last[w, i] = last


@wp.kernel
def fit_sum_kernel(k: int, tick: wp.array[wp.int32], stride: int, g: RollGeom, work: RollWork):
    """The band's fit over its rows in order, and its last cell."""
    w = wp.tid()
    if not _spills(tick, stride, g, work, w, k):
        return
    win = g.win[w, k]
    vol = wp.float64(0.0)
    end = wp.vec2i(-1, -1)
    for i in range(win[0], win[1] + 1):
        vol += work.fit[w, i]
        if work.last[w, i] >= 0:
            end = wp.vec2i(i, work.last[w, i])
    t = work.total[w]
    work.total[w] = vec4d(t[0], t[1], t[2], vol)
    work.end_cell[w] = end
    if vol <= wp.float64(0.0):
        work.go[w] = 0


@wp.kernel
def take_kernel(cells: Cells, charts: wp.array[Chart], ts: ToolSet, k: int, tick: wp.array[wp.int32], stride: int,
                g: RollGeom, base: wp.array2d[wp.float32], work: RollWork, h: wp.array2d[wp.float64], sp: Species):
    """Strip l's wedge gives (fit / roll) of each cell's film above s0 (at most its film), with its species."""
    w, l = wp.tid()
    r = _strips(g, w, k)
    if not _spills(tick, stride, g, work, w, k) or l < r[2] or l > r[3]:
        return
    ch = charts[ts.chart[k]]
    a, step, wd = g.axis[w, k][0], g.axis[w, k][1], work.wedge[w, l]
    frac = work.total[w][3] / work.total[w][0]
    s0 = work.s0[w, l]
    vol = wp.float64(0.0)
    acc = vec4d()
    m = wd[0]
    for _it in range(wp.where(wd[1] < 0, 0, wp.abs(wd[2] - wd[0]) + 1)):
        c = _cell(ch, a, l, m)
        m += step
        h0 = h[w, c]
        dh = wp.min(frac * _above(h, base, w, c, s0), h0)
        if dh <= wp.float64(0.0):
            continue
        area = wp.float64(cells.area[c])
        for q in range(sp.count):
            dm = sp.mass[w, c, q] * (dh / h0)
            sp.mass[w, c, q] = sp.mass[w, c, q] - dm
            acc[q] = acc[q] + dm * area
        h[w, c] = h0 - dh
        vol += dh * area
    work.gave[w, l] = vol
    work.gave_sp[w, l] = acc


@wp.kernel
def take_sum_kernel(k: int, tick: wp.array[wp.int32], stride: int, g: RollGeom, work: RollWork):
    """What the wedges gave, over the strips in order."""
    w = wp.tid()
    if not _spills(tick, stride, g, work, w, k):
        return
    r = _strips(g, w, k)
    vol = wp.float64(0.0)
    acc = vec4d()
    for l in range(r[2], r[3] + 1):
        vol += work.gave[w, l]
        acc += work.gave_sp[w, l]
    work.took[w] = vol
    work.took_sp[w] = acc
    if vol <= wp.float64(0.0):
        work.go[w] = 0


@wp.func
def lay_cell(cells: Cells, w: int, c: int, v: wp.float64, dm: vec4d, h: wp.array2d[wp.float64], sp: Species,
              cwet: wp.array2d[wp.int32]):
    area = wp.float64(cells.area[c])
    for q in range(sp.count):
        sp.mass[w, c, q] = sp.mass[w, c, q] + dm[q] / area
    h[w, c] = h[w, c] + v / area
    cwet[w, c] = 1


@wp.kernel
def lay_kernel(cells: Cells, charts: wp.array[Chart], ts: ToolSet, k: int, tick: wp.array[wp.int32], stride: int,
               g: RollGeom, tf: ToolField, base: wp.array2d[wp.float32], work: RollWork, h: wp.array2d[wp.float64],
               sp: Species, cwet: wp.array2d[wp.int32]):
    """Band row i lays what was taken in proportion to what each cell fits, with the roll's species; the band's last
    cell is left to ``last_kernel``."""
    w, i = wp.tid()
    win = g.win[w, k]
    if not _spills(tick, stride, g, work, w, k) or i < win[0] or i > win[1]:
        return
    ch = charts[ts.chart[k]]
    total, end = work.total[w], work.end_cell[w]
    scale = work.took[w] / total[3]
    vol = wp.float64(0.0)
    acc = vec4d()
    for j in range(win[2], win[3] + 1):
        c = cell_at(ch, i, j)
        f = _cell_fit(cells, ch, tf, h, base, w, k, g, i, j, c, total)
        if f <= wp.float64(0.0) or (i == end[0] and j == end[1]):
            continue
        dm = work.took_sp[w] * (f / total[3])
        lay_cell(cells, w, c, f * scale, dm, h, sp, cwet)
        vol += f * scale
        acc += dm
    work.laid[w, i] = vol
    work.laid_sp[w, i] = acc


@wp.kernel
def last_kernel(cells: Cells, charts: wp.array[Chart], ts: ToolSet, k: int, tick: wp.array[wp.int32], stride: int,
                g: RollGeom, tf: ToolField, base: wp.array2d[wp.float32], work: RollWork, h: wp.array2d[wp.float64],
                sp: Species, cwet: wp.array2d[wp.int32], spill: wp.array2d[wp.float64]):
    """The band's last cell takes the remainder (the volume and species ledgers are exact); the spill is booked."""
    w = wp.tid()
    if not _spills(tick, stride, g, work, w, k):
        return
    win = g.win[w, k]
    vol = wp.float64(0.0)
    acc = vec4d()
    for i in range(win[0], win[1] + 1):
        vol += work.laid[w, i]
        acc += work.laid_sp[w, i]
    spill[w, k] = spill[w, k] + work.took[w]
    ch = charts[ts.chart[k]]
    end = work.end_cell[w]
    c = cell_at(ch, end[0], end[1])
    if _cell_fit(cells, ch, tf, h, base, w, k, g, end[0], end[1], c, work.total[w]) <= wp.float64(0.0):
        return
    lay_cell(cells, w, c, work.took[w] - vol, work.took_sp[w] - acc, h, sp, cwet)


__all__ = ["BAND_COVER", "SPILL_BAND", "TOE_FALL", "RollGeom", "RollWork", "band_total", "band_weight", "fit_kernel",
           "fit_sum_kernel", "last_kernel", "lay_cell", "lay_kernel", "perp", "roll_kernel", "take_kernel",
           "take_sum_kernel", "wedge_kernel"]
