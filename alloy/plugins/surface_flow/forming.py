# SPDX-License-Identifier: Apache-2.0
"""Plastic forming of a paste's bulk under tools (design note §3.7; phase 2 note, "Plastic forming").

A deep paste (a slab under a float, mortar under a jointer) is a plastic substrate. Its surface is the chart's
``base``: float32 for the film, with the float64 ``bulk`` state underneath it. Where a tool presses the surface above
the bearing capacity ``p_bc = (2 + π)·τ_y`` (Prandtl's punch), the surface sinks at ``(p - p_bc)/C_Y``.

Per exchange, ``forming_kernel`` runs one thread per world over its tools in a fixed order:
1. the cells under the tool sink, never below its underside;
2. the displaced volume goes to the ring of uncovered cells next to the footprint: ahead of a sliding tool (the roll
   a float pushes), or all around a pressing one (the heave beside a punch). Equal shares, with the last share the
   remainder, so the volume is exact.

Below ``p_bc`` nothing moves (law F-18). The pad pressure ``p = k·δ`` is the tool's, as in ``tool_force_kernel``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.charts import Chart
from alloy.plugins.surface_flow.tool_geometry import (V_PUSH, Pose, ToolSet, cell_point, pose_of, relative,
                                                      tangential, tool_under, window)


BEARING = wp.constant(wp.float64(2.0 + math.pi))      # Prandtl's N_c


@dataclass(frozen=True)
class Bulk:
    """A paste's plastic bulk: yield stress [Pa], and ``resistance`` C_Y [Pa·s/m], the extra pressure per sinking
    speed. Fit C_Y to a punch test. A fresh concrete slab's τ_y is about 1.6 kPa (NISTIR 6094 at 100 mm slump)."""

    yield_stress: float
    resistance: float

    def __post_init__(self) -> None:
        for name in ("yield_stress", "resistance"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v > 0.0):
                raise PluginError(f"bulk: {name} must be finite and > 0, got {v}")

    @property
    def bearing(self) -> float:
        """Prandtl's bearing capacity of a flat punch [Pa]."""
        return float(BEARING) * self.yield_stress


def bulk_spec(spec: Bulk | dict | None) -> Bulk | None:
    if spec is None or isinstance(spec, Bulk):
        return spec
    if isinstance(spec, dict):
        return Bulk(**spec)
    raise PluginError(f"bulk: expected a Bulk or a dict, got {spec!r}")


@wp.func
def in_ring(ts: ToolSet, k: int, ch: Chart, ps: Pose, i: int, j: int, cell: float) -> int:
    """1 when cell (i, j) is outside tool k's footprint but next to a cell inside it."""
    if tool_under(ts, k, ps, cell_point(ch, i, j), cell).cover > 0.0:
        return 0
    if i > 0 and tool_under(ts, k, ps, cell_point(ch, i - 1, j), cell).cover > 0.0:
        return 1
    if i < ch.nx - 1 and tool_under(ts, k, ps, cell_point(ch, i + 1, j), cell).cover > 0.0:
        return 1
    if j > 0 and tool_under(ts, k, ps, cell_point(ch, i, j - 1), cell).cover > 0.0:
        return 1
    if j < ch.ny - 1 and tool_under(ts, k, ps, cell_point(ch, i, j + 1), cell).cover > 0.0:
        return 1
    return 0


@wp.kernel
def forming_kernel(ts: ToolSet, n_tools: int, charts: wp.array[Chart], tool_body: wp.array2d[wp.int32],
                   cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                   body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                   bulk_yield: wp.array2d[wp.float32],
                   resistance: wp.float64, dt: wp.float64, area: wp.array[wp.float64], bulk: wp.array2d[wp.float64],
                   pad_ratio: wp.array2d[wp.float32], sink: wp.array2d[wp.float32], base: wp.array2d[wp.float32]):
    w = wp.tid()
    for k in range(n_tools):
        kc = ts.chart[k]
        ch = charts[kc]
        tb, cb = tool_body[w, k], cbody[w, kc]
        ps = pose_of(ts, k, body_q, tb, cb, clocal[w, kc])
        win = window(ch, ps)
        cell = wp.min(ch.step[0], ch.step[1])
        centre = tool_under(ts, k, ps, wp.vec3(ps.c[0], ps.c[1], 0.0), cell)
        v = tangential(relative(body_q, body_qd, body_com, tb, cb, ps, centre))
        ahead = wp.length(v) > V_PUSH
        ring = int(0)                                             # the cells that take the displaced paste
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                p = cell_point(ch, i, j)
                if in_ring(ts, k, ch, ps, i, j, cell) == 1:
                    if not ahead or wp.dot(wp.vec3(p[0] - ps.c[0], p[1] - ps.c[1], 0.0), v) > 0.0:
                        ring += 1
        settle = sink[w, k]                                       # the surface's sinking speed (last exchange)
        sink[w, k] = 0.0
        if ring == 0:
            continue
        moved = wp.float64(0.0)
        pressed = wp.float64(0.0)
        stiffness = wp.float64(ts.stiffness[k]) * wp.float64(pad_ratio[w, k])   # an implicit pad's
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                c = ch.cell_start + i * ch.ny + j
                under = tool_under(ts, k, ps, cell_point(ch, i, j), cell)
                if under.cover <= 0.0:
                    continue
                pen = bulk[w, c] - wp.float64(under.gap)
                if pen <= wp.float64(0.0):
                    continue
                vr = relative(body_q, body_qd, body_com, tb, cb, ps, under)
                approach = wp.float64(ts.damping[k] * (-vr[2] - settle))        # the pad's damper, as on the tool
                p_bc = BEARING * wp.float64(bulk_yield[w, c])            # Prandtl's punch, per cell (setting paste)
                over = stiffness * (pen + approach) - p_bc
                pressed = pressed + wp.float64(under.cover) * area[c]
                if over <= wp.float64(0.0):
                    continue
                d = wp.min(over / resistance * dt, pen) * wp.float64(under.cover)
                bulk[w, c] = bulk[w, c] - d
                moved = moved + d * area[c]
        if pressed > wp.float64(0.0):
            sink[w, k] = wp.float32(moved / (pressed * dt))
        left = moved
        n = int(0)
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                p = cell_point(ch, i, j)
                if in_ring(ts, k, ch, ps, i, j, cell) == 0:
                    continue
                if ahead and wp.dot(wp.vec3(p[0] - ps.c[0], p[1] - ps.c[1], 0.0), v) <= 0.0:
                    continue
                c = ch.cell_start + i * ch.ny + j
                n += 1
                share = moved / wp.float64(ring)
                if n == ring:
                    share = left                                  # the remainder: the volume is exact
                left = left - share
                bulk[w, c] = bulk[w, c] + share / area[c]
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                c = ch.cell_start + i * ch.ny + j
                base[w, c] = wp.float32(bulk[w, c])


__all__ = ["Bulk", "bulk_spec", "forming_kernel"]
