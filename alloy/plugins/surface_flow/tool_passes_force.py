# SPDX-License-Identifier: Apache-2.0
"""Applicators' pads and drag, parallel over (world, tool, block of window cells) (``tool_passes``' module docstring:
``tool_force_kernel`` split into passes).

A block of ``BLOCK`` window cells per thread sums its cells' springs, loads and drag; one thread per (world, tool) adds
the blocks in order and solves the implicit pad; the blocks then sum their cells' wrenches; one thread per world adds
each tool's blocks in order to its body and the chart's. Every sum runs in a fixed order, so runs stay deterministic,
with no float atomics; the order differs from the serial kernel's, so the sums differ from it at float32 rounding.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from alloy.plugins.funcs import add_wrench, com_world, wrench_at
from alloy.plugins.surface_flow.charts import PLANE, Chart
from alloy.plugins.surface_flow.kernels import Cells, Fluid
from alloy.plugins.surface_flow.pad_implicit import PadSums, pad_add, pad_wrench, pad_zero
from alloy.plugins.surface_flow.tool_geometry import (ToolSet, along, cell_at, edge_of, off, pose_of, relative,
                                                      under_at, win_of)
from alloy.plugins.surface_flow.tool_kernels import film_shear
from alloy.plugins.surface_flow.tool_passes import BLOCK, win_cell, win_size




@wp.kernel
def force_windows_kernel(ts: ToolSet, charts: wp.array[Chart], prof: wp.array[wp.vec3],
                         tool_body: wp.array2d[wp.int32], cbody: wp.array2d[wp.int32],
                         clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform], maxw: wp.array[wp.int32],
                         fwin: wp.array2d[wp.vec4i], over: wp.array2d[wp.int32]):
    """Each tool's window at the pose ``apply`` reads (unwidened, as ``tool_force_kernel``)."""
    w, k = wp.tid()
    kc = ts.chart[k]
    ps = pose_of(ts, k, body_q, tool_body[w, k], cbody[w, kc], clocal[w, kc])
    win = win_of(charts[kc], prof, ps)
    fwin[w, k] = win
    if win_size(win) > maxw[k]:
        over[w, k] = 1


@wp.kernel
def force_sums_kernel(ts: ToolSet, charts: wp.array[Chart], cells: Cells, tool_body: wp.array2d[wp.int32],
                      cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                      body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                      sink: wp.array2d[wp.float32], h: wp.array2d[wp.float64], base: wp.array2d[wp.float32],
                      area: wp.array[wp.float64], fl: Fluid, h_wet: float, dt: float, fwin: wp.array2d[wp.vec4i],
                      p_load: wp.array3d[wp.float32], p_drag: wp.array3d[wp.float32],
                      p_pushed: wp.array3d[wp.float32], p_sums: wp.array3d[PadSums]):
    """One block of a tool's window: its cells' pad loads, implicit springs and film drag (``tool_force_kernel``'s
    first loop), summed in the cells' order."""
    w, k, b = wp.tid()
    win = fwin[w, k]
    n0 = b * BLOCK
    n1 = wp.min(n0 + BLOCK, win_size(win))
    load = float(0.0)
    drag = float(0.0)
    pushed = float(0.0)
    sums = pad_zero()
    if n0 < n1:
        kc = ts.chart[k]
        ch = charts[kc]
        tb, cb = tool_body[w, k], cbody[w, kc]
        ps = pose_of(ts, k, body_q, tb, cb, clocal[w, kc])
        cell = edge_of(ch)
        implicit = ts.implicit[k] == 1
        tool_com = com_world(body_q, body_com, tb)
        for n in range(n0, n1):
            ij = win_cell(win, n)
            c = cell_at(ch, ij[0], ij[1])
            under = under_at(ts, k, ch, cells, ps, ij[0], ij[1], c, cell)
            if under.cover <= 0.0:
                continue
            a = under.cover * wp.float32(area[c])
            film = under.gap - base[w, c]
            nc = wp.transform_vector(ps.X, under.n)                       # this cell's normal, world
            if film <= 0.0:
                load += ts.stiffness[k] * (-film) * a
                if implicit:                                      # the springs and the damper's sink offset
                    vz = off(relative(body_q, body_qd, body_com, tb, cb, ps, under), under)
                    ka = ts.stiffness[k] * a
                    r = wp.transform_point(ps.X, under.point) - tool_com
                    sums = pad_add(sums, ka, nc, r, ka * (-film - (ts.damping[k] + dt) * sink[w, k]))
                    pushed += wp.max(ka * (-film + ts.damping[k] * (-vz - sink[w, k])), 0.0)
            elif wp.float32(h[w, c]) > h_wet and wp.float32(h[w, c]) >= film - 1.0e-6:
                v = along(relative(body_q, body_qd, body_com, tb, cb, ps, under), under)
                drag += film_shear(fl, w, c, film, v) * a
    p_load[w, k, b] = load
    p_drag[w, k, b] = drag
    p_pushed[w, k, b] = pushed
    p_sums[w, k, b] = sums


@wp.func
def sums_add(a: PadSums, b: PadSums) -> PadSums:
    out = PadSums()
    out.ktt = a.ktt + b.ktt
    out.ktr = a.ktr + b.ktr
    out.krr = a.krr + b.krr
    out.force = a.force + b.force
    out.torque = a.torque + b.torque
    return out


@wp.kernel
def force_solve_kernel(ts: ToolSet, charts: wp.array[Chart], tool_body: wp.array2d[wp.int32],
                       cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                       body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                       mobility: wp.array2d[wp.float32], dt: float, body_mass: wp.array[wp.float32],
                       body_inertia: wp.array[wp.mat33], gravity: wp.vec3, n_blocks: int,
                       p_load: wp.array3d[wp.float32], p_drag: wp.array3d[wp.float32],
                       p_pushed: wp.array3d[wp.float32], p_sums: wp.array3d[PadSums],
                       drag_scale: wp.array2d[wp.float32], pad_ratio: wp.array2d[wp.float32],
                       normal: wp.array2d[wp.float32], active: wp.array2d[wp.int32], scale_a: wp.array2d[wp.float32],
                       eps_a: wp.array2d[wp.float32], ratio_a: wp.array2d[wp.float32],
                       pad_tool: wp.array2d[wp.spatial_vector], pad_face: wp.array2d[wp.spatial_vector]):
    """Each tool's sums over its blocks, in order; the drag cap, the friction's regularization and the implicit
    pad's solve (``tool_force_kernel`` between its loops)."""
    w, k = wp.tid()
    load = float(0.0)
    drag = float(0.0)
    pushed = float(0.0)
    sums = pad_zero()
    for b in range(n_blocks):
        load += p_load[w, k, b]
        drag += p_drag[w, k, b]
        pushed += p_pushed[w, k, b]
        sums = sums_add(sums, p_sums[w, k, b])
    drag_scale[w, k] = 1.0
    pad_ratio[w, k] = 1.0
    normal[w, k] = load                                           # the pad's spring load k·δ·A [N] (a brush's press)
    active[w, k] = 0
    pad_tool[w, k] = wp.spatial_vector()
    pad_face[w, k] = wp.spatial_vector()
    if load <= 0.0 and drag <= 0.0:
        return
    active[w, k] = 1
    kc = ts.chart[k]
    ch = charts[kc]
    tb, cb = tool_body[w, k], cbody[w, kc]
    ps = pose_of(ts, k, body_q, tb, cb, clocal[w, kc])
    scale = wp.min(1.0, 0.5 / wp.max(drag * mobility[w, k] * dt, 1.0e-12))
    drag_scale[w, k] = scale
    scale_a[w, k] = scale
    eps_a[w, k] = wp.max(ts.slip_eps, 2.0 * ts.friction[k] * load * dt * mobility[w, k])
    ratio = float(1.0)
    if ts.implicit[k] == 1 and pushed > 0.0:                      # backward Euler on the tool against its pad
        nw = wp.transform_vector(ps.X, wp.vec3(0.0, 0.0, 1.0))    # off the surface: a plane's normal
        if ch.kind != PLANE:
            nw = -wp.transform_vector(ps.X, ps.n)                 # a curved surface: against the pad's
        tool_com = com_world(body_q, body_com, tb)
        face_com = wp.vec3(0.0, 0.0, 0.0)
        if cb >= 0:
            face_com = com_world(body_q, body_com, cb)
        rot = wp.quat_to_matrix(wp.transform_get_rotation(body_q[tb]))
        qd = body_qd[tb]
        pad = pad_wrench(sums, body_mass[tb], rot * body_inertia[tb] * wp.transpose(rot),
                         wp.vec3(qd[0], qd[1], qd[2]), wp.vec3(qd[3], qd[4], qd[5]), gravity, ts.damping[k], dt)
        fn = wp.dot(wp.vec3(pad[0], pad[1], pad[2]), nw)
        if fn > 0.0:                                              # no adhesion: a pad only pushes
            ratio = fn / pushed
            pad_tool[w, k] = pad
            pad_face[w, k] = (-wrench_at(wp.vec3(pad[0], pad[1], pad[2]), tool_com, face_com)
                              - wp.spatial_vector(0.0, 0.0, 0.0, pad[3], pad[4], pad[5]))
        else:
            ratio = 0.0
        pad_ratio[w, k] = ratio
    ratio_a[w, k] = ratio


@wp.kernel
def force_wrench_kernel(ts: ToolSet, charts: wp.array[Chart], cells: Cells, tool_body: wp.array2d[wp.int32],
                        cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform],
                        body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                        body_com: wp.array[wp.vec3], sink: wp.array2d[wp.float32], h: wp.array2d[wp.float64],
                        base: wp.array2d[wp.float32], area: wp.array[wp.float64], fl: Fluid, h_wet: float,
                        fwin: wp.array2d[wp.vec4i], active: wp.array2d[wp.int32], scale_a: wp.array2d[wp.float32],
                        eps_a: wp.array2d[wp.float32], ratio_a: wp.array2d[wp.float32],
                        p_tool: wp.array3d[wp.spatial_vector], p_face: wp.array3d[wp.spatial_vector]):
    """One block of a tool's window: its cells' pad, friction and drag wrenches on the tool and the chart's body
    (``tool_force_kernel``'s second loop), summed in the cells' order."""
    w, k, b = wp.tid()
    on_tool = wp.spatial_vector()
    on_face = wp.spatial_vector()
    win = fwin[w, k]
    n0 = b * BLOCK
    n1 = wp.min(n0 + BLOCK, win_size(win))
    if active[w, k] == 1 and n0 < n1:
        kc = ts.chart[k]
        ch = charts[kc]
        tb, cb = tool_body[w, k], cbody[w, kc]
        ps = pose_of(ts, k, body_q, tb, cb, clocal[w, kc])
        cell = edge_of(ch)
        implicit = ts.implicit[k] == 1
        scale, eps, ratio = scale_a[w, k], eps_a[w, k], ratio_a[w, k]
        tool_com = com_world(body_q, body_com, tb)
        face_com = wp.vec3(0.0, 0.0, 0.0)
        if cb >= 0:
            face_com = com_world(body_q, body_com, cb)
        for n in range(n0, n1):
            ij = win_cell(win, n)
            c = cell_at(ch, ij[0], ij[1])
            under = under_at(ts, k, ch, cells, ps, ij[0], ij[1], c, cell)
            if under.cover <= 0.0:
                continue
            a = under.cover * wp.float32(area[c])
            film = under.gap - base[w, c]
            vr = relative(body_q, body_qd, body_com, tb, cb, ps, under)
            vt = along(vr, under)
            vw = wp.transform_vector(ps.X, vt)
            f = wp.vec3(0.0, 0.0, 0.0)
            if film <= 0.0:                                       # on the substrate: the pad, Coulomb friction
                p = wp.max(ts.stiffness[k] * (-film + ts.damping[k] * (-off(vr, under) - sink[w, k])), 0.0)
                f = p * a * wp.transform_vector(ps.X, under.n) - (ts.friction[k] * p * a / wp.max(wp.length(vw),
                                                                                                  eps)) * vw
                if implicit:                                      # the normal part is the solve's; friction scales
                    f = -(ratio * ts.friction[k] * p * a / wp.max(wp.length(vw), eps)) * vw
            elif wp.float32(h[w, c]) > h_wet and wp.float32(h[w, c]) >= film - 1.0e-6:
                f = -(scale * film_shear(fl, w, c, film, vt) * a) * vw
            else:
                continue
            at = wp.transform_point(ps.X, under.point)
            on_tool = on_tool + wrench_at(f, at, tool_com)
            on_face = on_face - wrench_at(f, at, face_com)
    p_tool[w, k, b] = on_tool
    p_face[w, k, b] = on_face


@wp.kernel
def force_apply_kernel(ts: ToolSet, n_tools: int, n_blocks: int, tool_body: wp.array2d[wp.int32],
                       cbody: wp.array2d[wp.int32], active: wp.array2d[wp.int32],
                       pad_tool: wp.array2d[wp.spatial_vector], pad_face: wp.array2d[wp.spatial_vector],
                       p_tool: wp.array3d[wp.spatial_vector], p_face: wp.array3d[wp.spatial_vector],
                       body_f: wp.array[wp.spatial_vector]):
    """Per world, each tool in order: its pad's and its cells' wrenches to the tool, their reaction to the chart's
    body (one writer per world: no atomics)."""
    w = wp.tid()
    for k in range(n_tools):
        if active[w, k] == 0:
            continue
        on_tool = pad_tool[w, k]
        on_face = pad_face[w, k]
        for b in range(n_blocks):
            on_tool = on_tool + p_tool[w, k, b]
            on_face = on_face + p_face[w, k, b]
        add_wrench(body_f, tool_body[w, k], on_tool)
        cb = cbody[w, ts.chart[k]]
        if cb >= 0:
            add_wrench(body_f, cb, on_face)


def pads(flow: Any, state: Any, forces: Any, dt: float, h_wet: float) -> None:
    """``tool_force_kernel``'s work in passes (module docstring)."""
    t, tp, tools, dev, W = flow.tables, flow.tool_passes, flow.tools, flow.ctx.device, flow.ctx.world_count
    n, m = tools.count, flow.ctx.model
    wp.launch(force_windows_kernel, dim=(W, n),
              inputs=[tools.set, t.charts, t.prof, tools.body, t.cbody, t.clocal, state.body_q, tp.maxw_arr],
              outputs=[tp.fwin, tp.over], device=dev)
    wp.launch(force_sums_kernel, dim=(W, n, tp.n_blocks),
              inputs=[tools.set, t.charts, flow.cells, tools.body, t.cbody, t.clocal, state.body_q, state.body_qd,
                      m.body_com, flow.tool_sink, flow.h, flow.base, t.area, flow.fluid, h_wet, float(dt), tp.fwin],
              outputs=[tp.p_load, tp.p_drag, tp.p_pushed, tp.p_sums], device=dev)
    wp.launch(force_solve_kernel, dim=(W, n),
              inputs=[tools.set, t.charts, tools.body, t.cbody, t.clocal, state.body_q, state.body_qd, m.body_com,
                      tools.mobility, float(dt), m.body_mass, m.body_inertia, flow.gravity, tp.n_blocks, tp.p_load,
                      tp.p_drag, tp.p_pushed, tp.p_sums],
              outputs=[flow.tool_drag_scale, flow.tool_pad_ratio, flow.tool_normal, tp.active, tp.scale, tp.eps,
                       tp.ratio, tp.pad_tool, tp.pad_face], device=dev)
    wp.launch(force_wrench_kernel, dim=(W, n, tp.n_blocks),
              inputs=[tools.set, t.charts, flow.cells, tools.body, t.cbody, t.clocal, state.body_q, state.body_qd,
                      m.body_com, flow.tool_sink, flow.h, flow.base, t.area, flow.fluid, h_wet, tp.fwin, tp.active,
                      tp.scale, tp.eps, tp.ratio],
              outputs=[tp.p_tool, tp.p_face], device=dev)
    wp.launch(force_apply_kernel, dim=W,
              inputs=[tools.set, n, tp.n_blocks, tools.body, t.cbody, tp.active, tp.pad_tool, tp.pad_face,
                      tp.p_tool, tp.p_face],
              outputs=[forces.body_f], device=dev)

__all__ = ["pads"]
