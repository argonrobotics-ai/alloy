# SPDX-License-Identifier: Apache-2.0
"""Applicators on the film, parallel over (world, tool, window cell) (``tool_kernels``' two kernels split into passes).

``tool_kernels.tools_kernel`` and ``tool_force_kernel`` ran one thread per world over every tool's window cells. On a
whole pad at 2 mm a tool's window is tens of thousands of cells, so they took 0.4-0.9 s of a period's 0.4-1.0 s on an
L4 (SC9's profile, WP4's trowel and WP6's darby; one world as slow as sixteen). Here:

- **Ceilings** (``tools_kernel``): the old windows are cleared over (world, tool, cell); each tool's window, roll
  flag and sliding velocity come from one thread per (world, tool); then each tool in turn writes its cells over
  (world, cell) and marks the roll's zone ahead. The tools keep their fixed order, one launch each, so where tools
  overlap the lowest underside wins as before, and every cell's values are the serial kernel's, bit for bit.
- **Pads and drag** (``tool_force_kernel``; ``tool_passes_force``): blocks of ``BLOCK`` window cells sum their
  cells' springs, loads, drag and wrenches; each tool adds its blocks in order. Every sum runs in a fixed order, so
  runs stay deterministic, with no float atomics. The order differs from the serial kernel's, so the sums differ from
  it at float32 rounding.

A tool's window is at most ``window_cells`` cells (its reach and the chart's step; a revolution chart's whole grid),
fixed at setup, so the launches keep their sizes under graph capture.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.surface_flow.charts import PLANE, Chart
from alloy.plugins.surface_flow.kernels import Cells, ToolField
from alloy.plugins.surface_flow.pad_implicit import PadSums
from alloy.plugins.surface_flow.tool_geometry import (FREE_GAP, LIFT, V_PUSH, ToolSet, across_of, along, cell_at,
                                                      centre_under, edge_of, in_cell, pose_of, push_at, relative,
                                                      under_at, widen, win_of)
from alloy.plugins.surface_flow.tool_kernels import FRONT

BLOCK = 64                                   # window cells per thread in the force sums


@wp.func
def win_cell(win: wp.vec4i, n: int) -> wp.vec3i:
    """The ``n``-th cell (row i, column j, 1) of ``win`` in the serial loops' order (i outer, j inner); (0, 0, 0)
    past its end."""
    ni = win[1] - win[0] + 1
    nj = win[3] - win[2] + 1
    if ni <= 0 or nj <= 0 or n >= ni * nj:
        return wp.vec3i(0, 0, 0)
    return wp.vec3i(win[0] + n // nj, win[2] + n % nj, 1)


@wp.func
def win_size(win: wp.vec4i) -> int:
    ni = win[1] - win[0] + 1
    nj = win[3] - win[2] + 1
    if ni <= 0 or nj <= 0:
        return 0
    return ni * nj


# --- ceilings ------------------------------------------------------------------------------------------------------

@wp.kernel
def clear_kernel(ts: ToolSet, charts: wp.array[Chart], wins: wp.array2d[wp.vec4i], tf: ToolField,
                 ceil: wp.array2d[wp.float32], cover: wp.array2d[wp.float32], vel: wp.array2d[wp.vec3],
                 push: wp.array2d[wp.vec2], press: wp.array2d[wp.float32]):
    """The cells each tool wrote the step before, back to free (every writer writes the same values)."""
    w, k, n = wp.tid()
    ij = win_cell(wins[w, k], n)
    if ij[2] == 0:
        return
    c = cell_at(charts[ts.chart[k]], ij[0], ij[1])
    ceil[w, c] = FREE_GAP
    cover[w, c] = 0.0
    vel[w, c] = wp.vec3(0.0, 0.0, 0.0)
    tf.wceil[w, c] = FREE_GAP
    tf.wcover[w, c] = 0.0
    tf.wvel[w, c] = wp.vec3(0.0, 0.0, 0.0)
    push[w, c] = wp.vec2(0.0, 0.0)
    press[w, c] = 0.0
    tf.pore[w, c] = 0


@wp.kernel
def windows_kernel(ts: ToolSet, charts: wp.array[Chart], prof: wp.array[wp.vec3], tool_body: wp.array2d[wp.int32],
                   cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                   body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], maxw: wp.array[wp.int32],
                   wins: wp.array2d[wp.vec4i], rolls: wp.array2d[wp.int32], slide: wp.array2d[wp.vec3],
                   across: wp.array2d[wp.float32], touched: wp.array2d[wp.int32], over: wp.array2d[wp.int32]):
    """Each tool's window at the new poses (widened by the roll's zone when a ceiling tool slides), its sliding
    velocity at its centre and its half width across that, as ``tools_kernel``; ``over`` marks a window larger than
    the launch (never, by ``window_cells``)."""
    w, k = wp.tid()
    kc = ts.chart[k]
    ch = charts[kc]
    tb, cb = tool_body[w, k], cbody[w, kc]
    ps = pose_of(ts, k, body_q, tb, cb, clocal[w, kc])
    win = win_of(ch, prof, ps)
    centre = centre_under(ts, k, ch, prof, ps, edge_of(ch))
    s = along(relative(body_q, body_qd, body_com, tb, cb, ps, centre), centre)
    roll = ts.ceiling[k] == 1 and win[1] >= win[0] and wp.length(s) > V_PUSH
    lat = wp.cross(centre.n, wp.normalize(s + wp.vec3(1.0e-12, 0.0, 0.0)))
    across[w, k] = across_of(ts.shape[k], ts.size[k], ps, wp.normalize(lat + wp.vec3(0.0, 1.0e-12, 0.0)))
    if roll:
        win = widen(ch, win, FRONT)
        rolls[w, k] = 1
    else:
        rolls[w, k] = 0
    wins[w, k] = win
    slide[w, k] = s
    touched[w, k] = 0
    if win_size(win) > maxw[k]:
        over[w, k] = 1


@wp.kernel
def write_kernel(k: int, ts: ToolSet, charts: wp.array[Chart], cells: Cells, tool_body: wp.array2d[wp.int32],
                 cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                 body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3], wins: wp.array2d[wp.vec4i],
                 base: wp.array2d[wp.float32], h: wp.array2d[wp.float64], tf: ToolField,
                 ceil: wp.array2d[wp.float32], cover: wp.array2d[wp.float32], vel: wp.array2d[wp.vec3],
                 push: wp.array2d[wp.vec2], press: wp.array2d[wp.float32], cwet: wp.array2d[wp.int32],
                 pad_ratio: wp.array2d[wp.float32], touched: wp.array2d[wp.int32]):
    """Tool ``k``'s underside over its window's cells (``tools_kernel``'s loop, one cell per thread)."""
    w, n = wp.tid()
    ij = win_cell(wins[w, k], n)
    if ij[2] == 0:
        return
    kc = ts.chart[k]
    ch = charts[kc]
    tb, cb = tool_body[w, k], cbody[w, kc]
    ps = pose_of(ts, k, body_q, tb, cb, clocal[w, kc])
    i, j = ij[0], ij[1]
    c = cell_at(ch, i, j)
    under = under_at(ts, k, ch, cells, ps, i, j, c, edge_of(ch))
    if under.cover <= 0.0 or under.gap - base[w, c] - wp.float32(h[w, c]) > LIFT:
        return                                                     # not under it, or lifted off this cell's film
    wp.atomic_add(touched, w, k, 1)
    if under.gap < tf.wceil[w, c]:                                 # the work fields: every tool, a store's too
        tf.wceil[w, c] = under.gap
        tf.wcover[w, c] = under.cover
        tf.wvel[w, c] = along(relative(body_q, body_qd, body_com, tb, cb, ps, under), under)
    pk = ts.stiffness[k] * pad_ratio[w, k]
    press[w, c] = wp.max(press[w, c], pk * wp.max(base[w, c] - under.gap, 0.0))      # the pad
    if ts.ceiling[k] == 1 and under.gap < ceil[w, c]:
        ceil[w, c] = under.gap
        cover[w, c] = under.cover
        tf.pore[w, c] = 0
        if ts.porous[k] == 1 and tf.room[w, k] > wp.float64(0.0):
            tf.pore[w, c] = k + 1                                  # a sponge with room: it absorbs, not ploughs
        v = along(relative(body_q, body_qd, body_com, tb, cb, ps, under), under)
        vel[w, c] = v
        push[w, c] = push_at(v, under, ps)
        if h[w, c] > wp.float64(0.0):
            cwet[w, c] = 1


@wp.kernel
def roll_kernel(k: int, ts: ToolSet, charts: wp.array[Chart], cells: Cells, tool_body: wp.array2d[wp.int32],
                cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                wins: wp.array2d[wp.vec4i], rolls: wp.array2d[wp.int32], slide: wp.array2d[wp.vec3],
                across: wp.array2d[wp.float32], touched: wp.array2d[wp.int32], cover: wp.array2d[wp.float32],
                push: wp.array2d[wp.vec2]):
    """Tool ``k``'s roll zone ahead, once its cells are written: uncovered cells get its push direction."""
    w, n = wp.tid()
    if rolls[w, k] == 0 or touched[w, k] == 0:
        return
    ij = win_cell(wins[w, k], n)
    if ij[2] == 0:
        return
    kc = ts.chart[k]
    ch = charts[kc]
    ps = pose_of(ts, k, body_q, tool_body[w, k], cbody[w, kc], clocal[w, kc])
    i, j = ij[0], ij[1]
    c = cell_at(ch, i, j)
    under = under_at(ts, k, ch, cells, ps, i, j, c, edge_of(ch))
    if under.cover > 0.0 or cover[w, c] > 0.0:
        return
    ahead = wp.normalize(in_cell(slide[w, k], under) + wp.vec2(1.0e-12, 0.0))
    rel = in_cell(under.foot - ps.c, under)
    if wp.dot(rel, ahead) > 0.0 and wp.abs(rel[0] * ahead[1] - rel[1] * ahead[0]) <= across[w, k]:
        push[w, c] = ahead


# --- host -----------------------------------------------------------------------------------------------------------

def window_cells(tools: Any, tables: Any) -> list[int]:
    """Per tool, the most cells its window can hold: a plane chart's reach plus a step each way (``window``), two
    rows of margin, and the roll's zone (``FRONT``) for a ceiling tool, clamped to the chart; a revolution chart's
    whole grid (``rev_window`` and ``widen`` keep to one turn)."""
    charts = tables.charts.numpy()
    out = []
    for s in tools.specs:
        ch = charts[s.chart]
        nx, ny = int(ch["nx"]), int(ch["ny"])
        if int(ch["kind"]) != PLANE:
            out.append(nx * ny)
            continue
        step = (float(ch["step"][0]), float(ch["step"][1]))
        reach = s.size[0] if s.shape == "disc" else math.hypot(*s.size)
        r = reach + max(step)
        grow = 2 * FRONT if s.store is None else 0
        ni = min(nx, int(math.ceil(2.0 * r / step[0])) + 3 + grow)
        nj = min(ny, int(math.ceil(2.0 * r / step[1])) + 3 + grow)
        out.append(ni * nj)
    return out


class ToolPasses:
    """The passes' bounds and scratch for one flow's tools (``tool_steps.setup_tools``)."""

    def __init__(self, ctx: Any, tools: Any, tables: Any, W: int) -> None:
        n = tools.count
        self.maxw = window_cells(tools, tables)
        self.maxw_arr = ctx.array(np.asarray(self.maxw, np.int32), wp.int32)
        self.n_blocks = max(1, -(-max(self.maxw) // BLOCK))
        B = (W, n, self.n_blocks)
        self.rolls, self.touched = ctx.zeros((W, n), wp.int32), ctx.zeros((W, n), wp.int32)
        self.slide, self.across = ctx.zeros((W, n), wp.vec3), ctx.zeros((W, n), wp.float32)
        self.over = ctx.zeros((W, n), wp.int32)            # a window over its launch (a bug in window_cells)
        self.fwin = ctx.zeros((W, n), wp.vec4i)
        self.p_load, self.p_drag, self.p_pushed = (ctx.zeros(B, wp.float32) for _ in range(3))
        self.p_sums = ctx.zeros(B, PadSums)
        self.p_tool, self.p_face = ctx.zeros(B, wp.spatial_vector), ctx.zeros(B, wp.spatial_vector)
        self.active = ctx.zeros((W, n), wp.int32)
        self.scale, self.eps, self.ratio = (ctx.zeros((W, n), wp.float32) for _ in range(3))
        self.pad_tool, self.pad_face = ctx.zeros((W, n), wp.spatial_vector), ctx.zeros((W, n), wp.spatial_vector)

    def check(self) -> None:
        """Raise if a window ever outgrew its launch (host read: call between periods, e.g. in tests)."""
        if int(self.over.numpy().max(initial=0)) > 0:
            raise RuntimeError(f"surface_flow: a tool's window outgrew its launch ({self.maxw} cells)")


def ceilings(flow: Any, state_out: Any) -> None:
    """``tools_kernel``'s work in passes (module docstring)."""
    t, tp, tools, dev, W = flow.tables, flow.tool_passes, flow.tools, flow.ctx.device, flow.ctx.world_count
    tf, n = flow.tf, tools.count
    wp.launch(clear_kernel, dim=(W, n, max(tp.maxw)), inputs=[tools.set, t.charts, tools.wins, tf],
              outputs=[tf.ceil, tf.cover, tf.vel, tf.push, tf.press], device=dev)
    wp.launch(windows_kernel, dim=(W, n),
              inputs=[tools.set, t.charts, t.prof, tools.body, t.cbody, t.clocal, state_out.body_q,
                      state_out.body_qd, flow.ctx.model.body_com, tp.maxw_arr],
              outputs=[tools.wins, tp.rolls, tp.slide, tp.across, tp.touched, tp.over], device=dev)
    for k in range(n):
        wp.launch(write_kernel, dim=(W, tp.maxw[k]),
                  inputs=[k, tools.set, t.charts, flow.cells, tools.body, t.cbody, t.clocal, state_out.body_q,
                          state_out.body_qd, flow.ctx.model.body_com, tools.wins, flow.base, flow.h, tf],
                  outputs=[tf.ceil, tf.cover, tf.vel, tf.push, tf.press, flow.cwet, flow.tool_pad_ratio, tp.touched],
                  device=dev)
        wp.launch(roll_kernel, dim=(W, tp.maxw[k]),
                  inputs=[k, tools.set, t.charts, flow.cells, tools.body, t.cbody, t.clocal, state_out.body_q,
                          tools.wins, tp.rolls, tp.slide, tp.across, tp.touched, tf.cover],
                  outputs=[tf.push], device=dev)


__all__ = ["BLOCK", "ToolPasses", "ceilings", "win_cell", "win_size", "window_cells"]
