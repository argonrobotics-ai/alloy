# SPDX-License-Identifier: Apache-2.0
"""A metering brush (``Store(meter=C)``): GA's brush lays and picks up toward a metered film, not by a nip split.

GA's law (painting/brush.py:381-391): a sliding brush meters the film under its touching tips to

    h_m = min(L_c·sqrt(C·eta·U / w), h_max),

with ``L_c`` the contact length along the slide, ``U`` the slide speed, ``w`` the line load (the brush's normal force
over its contact width), ``eta`` the paint's viscosity at the shear ``U / 50 um`` and ``C`` the brush's metering
constant (params.py:149-158: sash 0.075, stencil 0.012, round no. 8 0.60). Cells below ``h_m`` are laid from the free
load, cells above it give back into the room left, each scaled down together when the load or the room is short
(GA's give/want, brush.py:430-452).

Here the contact patch is the cells the underside touches (``gap <= base + h``). Its width is its extent across the
slide (to the cell), ``L_c`` its area over that width, and ``w`` the pad's normal force over the width
(the spring load ``k·δ·A`` from ``tool_force_kernel``, explicit or implicit pad alike): set the pad stiffness to
GA's tuft stiffness density to match its line load. GA's ``eta``
is the paint's Cross viscosity at ``U / 50 um`` with no structure (brush.py:386, paint.py:98-101), close to its
high-shear ``eta_inf``. Pass it as ``Store(meter_viscosity=)``: a paint card's viscosity is its low-shear (Bingham,
sag) value, tens of times higher. Not modelled: the stipple a lifting brush leaves (brush.py:464-470) and the bristle
mechanics behind ``L_c``.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.surface_flow.cell_kernels import Species, vec4d
from alloy.plugins.surface_flow.stores_species import StoreSpecies, book_nip
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.tool_geometry import ROLLER, Pose, ToolSet, cell_point, tool_under, window

SHEAR_GAP = wp.constant(5.0e-5)     # m: GA's metering shear rate is U / 50 um (brush.py:387)
U_MIN = wp.constant(1.0e-3)         # m/s: a brush sliding slower than this meters nothing (GA: no tip slid)


@wp.func
def nip_species(sp: Species, conc: vec4d, w: int, c: int, dv: wp.float64, h: wp.float64, area: wp.float64,
                ssp: StoreSpecies, k: int, b: int):
    """Species with a nip's net transfer ``dv`` [m³] (> 0: laid) between bin b of store k and a cell of film ``h``
    [m]: taken film leaves with the cell's concentration (``species_out``) into the bin, laid film brings the bin's
    ``conc`` (``species_supplied``) out of it (stores_species.py). A net transfer, not an exchange."""
    for s in range(sp.count):
        if dv < wp.float64(0.0) and h > wp.float64(0.0):
            gone = sp.mass[w, c, s] * wp.min(-dv / (h * area), wp.float64(1.0))
            sp.mass[w, c, s] = sp.mass[w, c, s] - gone
            sp.out[w, c, s] = sp.out[w, c, s] + gone * area
            book_nip(ssp, w, k, b, s, -gone * area)
        elif dv > wp.float64(0.0):
            add = dv * conc[s]
            sp.mass[w, c, s] = sp.mass[w, c, s] + add / area
            sp.supplied[w, c, s] = sp.supplied[w, c, s] + add
            book_nip(ssp, w, k, b, s, add)


@wp.func
def nip_heading(ts: ToolSet, k: int, ps: Pose, step: wp.vec2) -> wp.vec3:
    """The nip's crossing direction and length this exchange, ``(ahead, dist)``: along the step, or for a roller
    across its axis (the rolling direction; SCR-SOFT-3-4). A robot-held cover exchanging every MuJoCo step moved
    0.39 mm with a heading wandering 0.1-0.6 deg: bands across each step's own direction skipped or re-split cells
    114 mm out along the axis (hide 0.975 on SOFT-3's full wall)."""
    dist = wp.length(step)
    ahead = wp.vec2(0.0, 0.0)
    if dist > 1.0e-9:
        ahead = step / dist
    if ts.shape[k] == ROLLER:
        roll = wp.vec2(-ps.a[1], ps.a[0])
        rl = wp.length(roll)
        if rl > 1.0e-6:
            roll = roll / rl
            s = wp.dot(step, roll)
            dist = wp.abs(s)
            ahead = roll
            if s < 0.0:
                ahead = -roll
    return wp.vec3(ahead[0], ahead[1], dist)


@wp.func
def touches(ts: ToolSet, k: int, ch: Chart, ps: Pose, cell: float, base: wp.array2d[wp.float32],
            h: wp.array2d[wp.float64], w: int, i: int, j: int) -> float:
    """The cover of cell (i, j) if the underside touches its film (gap <= base + h), else 0."""
    c = ch.cell_start + i * ch.ny + j
    under = tool_under(ts, k, ps, cell_point(ch, i, j), cell)
    if under.cover <= 0.0 or wp.float64(under.gap) > wp.float64(base[w, c]) + h[w, c]:
        return 0.0
    return under.cover


@wp.func
def metered(ts: ToolSet, k: int, ch: Chart, ps: Pose, cell: float, base: wp.array2d[wp.float32],
            area: wp.array[wp.float64], h: wp.array2d[wp.float64], w: int, ahead: wp.vec2, speed: float,
            force: float, c_m: wp.float64, h_max: wp.float64, mu: wp.float64, tau: wp.float64,
            eta_m: wp.float64) -> wp.float64:
    """h_m over the contact patch (0: nothing touches). The patch's width is its extent across the slide, to the
    cell; its length along it is its area over that width (steady as cells slide in and out of a flat pad)."""
    win = window(ch, ps)
    lat = wp.vec2(-ahead[1], ahead[0])
    c0, c1 = float(1.0e30), float(-1.0e30)
    a = float(0.0)
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            cov = touches(ts, k, ch, ps, cell, base, h, w, i, j)
            if cov > 0.0:
                p = cell_point(ch, i, j)
                across = wp.dot(wp.vec2(p[0] - ps.c[0], p[1] - ps.c[1]), lat)
                c0, c1 = wp.min(c0, across), wp.max(c1, across)
                a += cov * float(area[ch.cell_start + i * ch.ny + j])
    if a <= 0.0:
        return wp.float64(0.0)
    wc = (c1 - c0) + wp.abs(lat[0]) * ch.step[0] + wp.abs(lat[1]) * ch.step[1]
    lc = a / wc
    line = wp.max(force / wc, 1.0e-3)                                                 # N/m (GA's floor)
    eta = mu + tau * wp.float64(SHEAR_GAP / wp.max(speed, U_MIN))                     # Bingham at U / 50 um
    if eta_m > wp.float64(0.0):
        eta = eta_m                                       # the paint's high-shear viscosity (GA: Cross at ~3000/s)
    return wp.min(wp.float64(lc) * wp.sqrt(c_m * eta * wp.float64(speed) / wp.float64(line)), h_max)


@wp.func
def meter(ts: ToolSet, k: int, ch: Chart, ps: Pose, cell: float, base: wp.array2d[wp.float32],
          area: wp.array[wp.float64], h: wp.array2d[wp.float64], cwet: wp.array2d[wp.int32],
          to_tools: wp.array2d[wp.float64], sp: Species, conc: vec4d, w: int, hm: wp.float64, free: wp.float64,
          room: wp.float64, ssp: StoreSpecies, b: int) -> wp.float64:
    """Drive the contact cells toward ``hm``: lay from ``free``, pick up into ``room``, each scaled down together
    when short. Returns the net volume laid [m³] (< 0: picked up)."""
    win = window(ch, ps)
    lay, pick = wp.float64(0.0), wp.float64(0.0)
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            cov = touches(ts, k, ch, ps, cell, base, h, w, i, j)
            if cov > 0.0:
                c = ch.cell_start + i * ch.ny + j
                d = (hm - h[w, c]) * area[c] * wp.float64(cov)
                if d > wp.float64(0.0):
                    lay = lay + d
                else:
                    pick = pick - d
    s_lay, s_pick = wp.float64(1.0), wp.float64(1.0)
    if lay > free:
        s_lay = wp.max(free, wp.float64(0.0)) / lay
    if pick > room:
        s_pick = wp.max(room, wp.float64(0.0)) / pick
    net = wp.float64(0.0)
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            cov = touches(ts, k, ch, ps, cell, base, h, w, i, j)
            if cov > 0.0:
                c = ch.cell_start + i * ch.ny + j
                d = (hm - h[w, c]) * area[c] * wp.float64(cov)
                if d > wp.float64(0.0):
                    d = d * s_lay
                else:
                    d = d * s_pick
                hn = h[w, c] + d / area[c]
                if hn < wp.float64(0.0):                         # emptied: no rounding below 0 (p21)
                    d = -h[w, c] * area[c]
                    hn = wp.float64(0.0)
                nip_species(sp, conc, w, c, d, h[w, c], area[c], ssp, k, b)
                h[w, c] = hn
                to_tools[w, c] = to_tools[w, c] - d
                if hn > wp.float64(0.0):
                    cwet[w, c] = 1
                net = net + d
    return net


__all__ = ["SHEAR_GAP", "U_MIN", "meter", "metered", "nip_heading", "nip_species", "touches"]
