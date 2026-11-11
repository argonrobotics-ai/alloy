# SPDX-License-Identifier: Apache-2.0
"""Tool contact and ink transfer kernels of the deposition plug-in.

One thread per world: each world owns its board map, its tools and their ledgers, and walks its tools in a fixed
order, so every element has one writer and there are no atomics. Both kernels evaluate every tool's contact from the
pre-solve state with the same functions, so the transfer sees exactly the pressure the force kernel applied.

Contact (removal v1's compliant pad, ``alloy.plugins.removal.kernels``): each board cell under a tool carries
``p = k·(δ + τ·dδ/dt)``, never negative. A domed tip (a marker nib, a paraboloid: a sphere's small-patch limit) lowers
δ by its sag ``r²/2R`` at in-plane distance ``r`` from the apex, so its contact patch is a disc of radius
``a = sqrt(2·R·δ0)`` that grows with the force.

Transfer:
- a marker lays ``flow·p·A·dt`` of ink into its channel in every contact cell, limited by its reservoir. With a
  ``saturation`` (kg/m²) the board takes less as it fills: the rate is scaled by ``1 - held/capacity``. Summed over
  the patch, the marker lays ``flow·F_N`` per second, so a stroke at speed v leaves ``flow·F_N / v`` per metre;
- a wiper lifts ``1 - exp(-cover·ds/wipe_length)`` of every channel from each contact cell it slides over by ``ds``
  (removal v1's wipe law) into its sink.

The ledger is exact: the map is float32, and every transfer moves exactly ``float64(new) - float64(old)`` of a cell
to or from a float64 reservoir or sink. So per world, ``Σ map + Σ reservoirs + Σ sinks`` is constant to float64
rounding.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.deposition.tools import Tip
from alloy.plugins.funcs import add_wrench, wrench_at
from alloy.plugins.removal.kernels import Frames, Grid, Hit, cell_hit, frames, pad_window, pressure, slip_velocity
from alloy.plugins.removal.surface import moved_point

MARKER = wp.constant(0)


@wp.func
def tip_hit(fr: Frames, tip: Tip, grid: Grid, i: int, j: int) -> Hit:
    """Removal's pad contact for cell (i, j) on a flat board, less the dome's sag."""
    hit = cell_hit(fr, tip.pad, grid, i, j, wp.float32(0.0))
    if tip.curvature > 0.0:
        r = hit.pad - fr.center
        hit.depth = hit.depth - 0.5 * tip.curvature * wp.dot(r, r)
    return hit


@wp.kernel
def tip_force_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                     body_com: wp.array[wp.vec3], tools: wp.array2d[wp.int32], face_body: wp.array[wp.int32],
                     face_local: wp.array[wp.transform], tips: wp.array[Tip], grid: Grid, nx: int, ny: int,
                     dt: wp.float32, mobility: wp.array2d[wp.float32], body_f: wp.array[wp.spatial_vector]):
    """ADD every tool's contact pressure and friction to its body, and the reactions to a floating board."""
    w = wp.tid()
    fb = face_body[w]
    area = grid.cell[0] * grid.cell[1]
    on_face = wp.spatial_vector()
    for t in range(tips.shape[0]):
        tip = tips[t]
        tb = tools[w, t]
        fr = frames(body_q, body_com, tb, fb, face_local[w], tip.pad)
        win = pad_window(fr, tip.pad, grid, nx, ny)
        load = float(0.0)
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                hit = tip_hit(fr, tip, grid, i, j)
                load += pressure(hit, slip_velocity(body_qd, fr, tb, fb, hit), fr, tip.pad) * hit.cover * area
        if load <= 0.0:
            continue
        # explicit viscous friction stays stable while (mu F / eps)·dt·mobility <= 1/2 (removal v1, stability.py)
        eps = wp.max(tip.pad.slip_eps, 2.0 * tip.pad.friction * load * dt * mobility[w, t])
        on_tool = wp.spatial_vector()
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                hit = tip_hit(fr, tip, grid, i, j)
                v = slip_velocity(body_qd, fr, tb, fb, hit)
                f_n = pressure(hit, v, fr, tip.pad) * hit.cover * area
                if f_n <= 0.0:
                    continue
                vt = v - wp.dot(v, fr.n_s) * fr.n_s
                f = f_n * fr.n_s - (tip.pad.friction * f_n / wp.max(wp.length(vt), eps)) * vt
                on_tool = on_tool + wrench_at(f, hit.pad, fr.tool_com)
                on_face = on_face - wrench_at(f, hit.pad, fr.face_com)
        add_wrench(body_f, tb, on_tool)
    if fb >= 0:
        add_wrench(body_f, fb, on_face)


@wp.func
def lay(ink: wp.array4d[wp.float32], w: int, i: int, j: int, c: int, want: wp.float64,
        left: wp.float64) -> wp.float64:
    """Add up to ``min(want, left)`` to one cell; return exactly what the float32 cell gained."""
    old = ink[w, i, j, c]
    new = wp.float32(wp.float64(old) + wp.min(want, left))
    moved = wp.float64(new) - wp.float64(old)
    if moved > left:                                     # float32 rounded up past the reservoir: lay nothing
        return wp.float64(0.0)
    ink[w, i, j, c] = new
    return moved


@wp.func
def lift(ink: wp.array4d[wp.float32], w: int, i: int, j: int, keep: wp.float32) -> wp.float64:
    """Scale every channel of one cell by ``keep``; return exactly what left the cell."""
    gone = wp.float64(0.0)
    for c in range(ink.shape[3]):
        old = ink[w, i, j, c]
        new = old * keep
        gone = gone + (wp.float64(old) - wp.float64(new))
        ink[w, i, j, c] = new
    return gone


@wp.func
def held(ink: wp.array4d[wp.float32], w: int, i: int, j: int) -> wp.float32:
    total = wp.float32(0.0)
    for c in range(ink.shape[3]):
        total = total + ink[w, i, j, c]
    return total


@wp.kernel
def transfer_kernel(q_in: wp.array[wp.transform], qd_in: wp.array[wp.spatial_vector], q_out: wp.array[wp.transform],
                    body_com: wp.array[wp.vec3], tools: wp.array2d[wp.int32], face_body: wp.array[wp.int32],
                    face_local: wp.array[wp.transform], tips: wp.array[Tip], grid: Grid, dt: wp.float32,
                    saturation: wp.float32, ink: wp.array4d[wp.float32], reservoir: wp.array2d[wp.float64],
                    laid: wp.array2d[wp.float64], sink: wp.array2d[wp.float64], load: wp.array2d[wp.float32]):
    """Markers lay ink and wipers lift it, tool by tool in a fixed order, with an exact float64 ledger."""
    w = wp.tid()
    fb = face_body[w]
    area = grid.cell[0] * grid.cell[1]
    capacity = saturation * area
    nx, ny = ink.shape[1], ink.shape[2]
    for t in range(tips.shape[0]):
        tip = tips[t]
        tb = tools[w, t]
        fr = frames(q_in, body_com, tb, fb, face_local[w], tip.pad)
        win = pad_window(fr, tip.pad, grid, nx, ny)
        total = float(0.0)
        left = reservoir[w, t]
        gained = wp.float64(0.0)
        lifted = wp.float64(0.0)
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                hit = tip_hit(fr, tip, grid, i, j)
                p = pressure(hit, slip_velocity(qd_in, fr, tb, fb, hit), fr, tip.pad)
                if p <= 0.0:
                    continue
                a = hit.cover * area
                total += p * a
                if tip.kind == MARKER:
                    want = tip.flow * p * a * dt
                    if capacity > 0.0:
                        want = want * wp.clamp(1.0 - held(ink, w, i, j) / capacity, 0.0, 1.0)
                    moved = lay(ink, w, i, j, tip.colour, wp.float64(want), left)
                    left = left - moved
                    gained = gained + moved
                else:
                    slip = (moved_point(q_in, q_out, tb, hit.pad) - hit.pad) - (
                        moved_point(q_in, q_out, fb, hit.spot) - hit.spot)
                    ds = wp.length(slip - wp.dot(slip, fr.n_s) * fr.n_s)
                    lifted = lifted + lift(ink, w, i, j, wp.exp(-hit.cover * ds / tip.wipe_length))
        reservoir[w, t] = left
        laid[w, t] = laid[w, t] + gained
        sink[w, t] = sink[w, t] + lifted
        load[w, t] = total


__all__ = ["tip_force_kernel", "tip_hit", "transfer_kernel"]
