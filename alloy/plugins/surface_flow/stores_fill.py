# SPDX-License-Identifier: Apache-2.0
"""A scraping store's fill (``Store(scrape=True)``): what it carries goes into the lows under it, as GA's tools do.

GA fills what lies below the bearing part of the sole from the roll, every step it touches (concrete_finishing/
coupling.py:556-596), in proportion ``min(1, carry/want)`` when the carry is short:

- a knife (``Store(fill="knife")``, the default; a margin trowel): the cells whose sole is within ``BEAR`` of its
  lowest point fill up to that lowest level, no higher than the highest level it rests on (a rising blade lays
  nothing), and only up to the support where it bears on it (GA's ``on_base`` and ``base_level``, :454-458,
  578-581): the input ``support`` (GA's ``support_height``: a board, a mesh's fibre tops; the base by default)
  under its sole's deepest point, when that lies within ``ON_BASE`` of it, so a blade gliding on bare faces lays no
  film on them;
- a sealant nose (``fill="profile"``; GA's tooling spatula): every cell under it fills up to the nose's own profile
  there, up to both walls of the joint (:561-567, 580-584), but no higher than the lowest its underside has been
  over the cell in this pass (``pass_low``: every exchange since the cell came under it). A nose moving away from the
  joint presses nothing into it. GA fills under the whole nose up to its current profile whenever any cell touches
  it, so a lifted nose whose end grazed the bead past its stroke laid its whole carry under its raised arc (SOFT-2's
  tool-sealant: 0.29 ml in a 20 ms lift). Sliding at its height, it fills as GA's does; its bob of tens of µm sets
  the level to its lowest.

Both fill only where the input ``paste_mask`` is set (GA's ``reg.paste_mask``: the joint's cells, not the faces
beside it), 1 everywhere by default.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.surface_flow.cell_kernels import Species, vec4d
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.stores_meter import nip_species
from alloy.plugins.surface_flow.stores_species import StoreSpecies
from alloy.plugins.surface_flow.tool_geometry import FREE_GAP, Pose, ToolSet, cell_point, tool_under, window

BEAR = wp.constant(4.0e-4)    # m: GA's bearing band, 0.3 mm (coupling.py:53) plus its 0.1 mm slack (:562)
ON_BASE = wp.constant(5.0e-5)  # m: a sole this close over the base bears on it (GA's on_base, coupling.py:478)


@wp.func
def sole_cover(cover: float) -> float:
    """A scraping sole's share of a cell: all of it when the cell's centre lies under the sole, else none (GA's
    ``ins``). A soft edge cut part of a leading cell, which then stood above the underside and kept a rising blade
    touching while it refilled the cut cells under itself from its carry (WP6's trowel)."""
    if cover >= 0.5:
        return 1.0
    return 0.0


@wp.func
def pass_low(low: wp.array2d[wp.float32], seen: wp.array2d[wp.int32], tick: int, w: int, c: int,
             gap: float) -> float:
    """A nose's lowest underside over cell ``c`` [m over the chart] in its current pass, this exchange's ``gap``
    included: the pass runs while the nose covers the cell every exchange (``seen`` the last exchange it did). One
    nose per cell at a time."""
    prior = float(FREE_GAP)
    if seen[w, c] == tick - 1:
        prior = low[w, c]
    g = wp.min(prior, gap)
    low[w, c] = g
    seen[w, c] = tick
    return g


@wp.func
def fill_room(profile: int, gap: float, cover: float, low: float, top: wp.float64, base: wp.float32,
              h: wp.float64, mask: wp.float32) -> wp.float64:
    """A cell's room to fill [m] (0: not filled): up to the nose's profile (``profile``), or below a knife's bearing
    band up to ``top``."""
    if cover <= 0.0 or mask < wp.float32(0.5):
        return wp.float64(0.0)
    if profile == 1:
        return wp.max(wp.float64(gap) - wp.float64(base) - h, wp.float64(0.0))
    if gap - low >= BEAR:
        return wp.float64(0.0)
    return wp.max(top - wp.float64(base) - h, wp.float64(0.0))


@wp.func
def fill_lows(ch: Chart, ts: ToolSet, k: int, ps: Pose, cell: float, base: wp.array2d[wp.float32],
              area: wp.array[wp.float64], h: wp.array2d[wp.float64], cwet: wp.array2d[wp.int32],
              to_tools: wp.array2d[wp.float64], sp: Species, conc: vec4d, w: int, free: wp.float64, profile: int,
              mask: wp.array2d[wp.float32], ssp: StoreSpecies, b: int, support: wp.array2d[wp.float32],
              low_pass: wp.array2d[wp.float32]) -> wp.float64:
    """Lay the carry into the lows under the tool (module docstring), in proportion when short. Returns the volume
    laid [m³]."""
    win = window(ch, ps)
    low = float(FREE_GAP)
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            under = tool_under(ts, k, ps, cell_point(ch, i, j), cell)
            if sole_cover(under.cover) > 0.0:
                low = wp.min(low, under.gap)
    top = wp.float64(low)
    rest = wp.float64(-1.0e30)                                  # the highest level it rests on (module docstring)
    pen = float(-1.0e30)                                        # its deepest point into the support, and that level
    level = float(0.0)
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            c = ch.cell_start + i * ch.ny + j
            under = tool_under(ts, k, ps, cell_point(ch, i, j), cell)
            if sole_cover(under.cover) <= 0.0:
                continue
            surface = wp.float64(base[w, c]) + h[w, c]
            if surface >= wp.float64(under.gap) - wp.float64(ON_BASE):     # it bears here
                rest = wp.max(rest, wp.min(surface, wp.float64(under.gap)))
            sup = support[w, c]
            if sup < -1.0e29:
                sup = base[w, c]                                 # no support given: the base
            if sup - under.gap > pen:
                pen = sup - under.gap
                level = sup
    if rest > wp.float64(-1.0e29):                              # a rising blade fills nothing above where it rests
        top = wp.min(top, rest)
    if pen >= -wp.float32(ON_BASE):                             # bearing on the support: up to its level there
        top = wp.min(top, wp.float64(level))
    want = wp.float64(0.0)
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            c = ch.cell_start + i * ch.ny + j
            under = tool_under(ts, k, ps, cell_point(ch, i, j), cell)
            cov = sole_cover(under.cover)
            gap = under.gap
            if profile == 1:
                gap = low_pass[w, c]                              # no higher than its lowest this pass (stores.py)
            room = fill_room(profile, gap, cov, low, top, base[w, c], h[w, c], mask[w, c])
            want = want + room * area[c] * wp.float64(cov)
    if want <= wp.float64(0.0) or free <= wp.float64(0.0):
        return wp.float64(0.0)
    frac = wp.min(wp.float64(1.0), free / want)
    laid = wp.float64(0.0)
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            c = ch.cell_start + i * ch.ny + j
            under = tool_under(ts, k, ps, cell_point(ch, i, j), cell)
            cov = sole_cover(under.cover)
            gap = under.gap
            if profile == 1:
                gap = low_pass[w, c]                              # no higher than its lowest this pass (stores.py)
            room = fill_room(profile, gap, cov, low, top, base[w, c], h[w, c], mask[w, c])
            dv = room * area[c] * wp.float64(cov) * frac
            if dv > wp.float64(0.0):
                nip_species(sp, conc, w, c, dv, h[w, c], area[c], ssp, k, b)
                h[w, c] = h[w, c] + dv / area[c]
                to_tools[w, c] = to_tools[w, c] - dv
                cwet[w, c] = 1
                laid = laid + dv
    return laid


__all__ = ["BEAR", "ON_BASE", "fill_lows", "fill_room", "pass_low", "sole_cover"]
