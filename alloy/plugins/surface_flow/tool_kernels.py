# SPDX-License-Identifier: Apache-2.0
"""Applicators on the film, device side (design note §4.1; phase 2 note, "Applicators").

Runs use ``tool_passes`` / ``tool_passes_force``, which split ``tools_kernel`` and ``tool_force_kernel`` into passes
over (world, tool, window cell) for the GPU; the two kernels here stay as their reference (``tool_steps.PARALLEL``).

A tool's underside (a flat disc or rectangle, or a roller's cylinder) is a ceiling over the cells under it:
- ``tools_kernel`` (one thread per world, once per exchange step, at the post-solve poses) writes each covered cell's
  ``ceil`` (the underside's height above the chart surface along its normal), ``cover``, ``vel`` (the tool's sliding
  velocity over the surface, chart frame) and ``push`` (where film above the ceiling goes: ahead of a sliding tool,
  out from under a pressing one). It first clears the cells it wrote the step before. Where tools overlap, the
  lowest underside wins; the tools are walked in a fixed order, so runs are deterministic.
- each film substep, ``excess_kernel`` and ``displace_kernel`` move the film above the ceiling one cell along
  ``push`` (a gather: one writer per cell, volume exact), and the face kernel shears the film under a sliding tool
  toward the Couette mean ``U/2`` (``kernels.py``);
- ``tool_force_kernel`` (one thread per world, in ``apply``) presses the tool off the substrate like removal's
  compliant pad (``p = k·(δ + τ·dδ/dt)``) with Coulomb friction, and drags it by the shear ``μU/h + τ_y`` of film
  it glides on; the reaction goes to the chart's body. The explicit viscous drag is capped so that
  ``Σc·mobility·dt <= 1/2`` (removal's rule).
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import add_wrench, com_world, wrench_at
from alloy.plugins.surface_flow.cell_kernels import Species
from alloy.plugins.surface_flow.charts import PLANE, Chart
from alloy.plugins.surface_flow.pad_implicit import pad_add, pad_wrench, pad_zero
from alloy.plugins.surface_flow.kernels import (INTERIOR, Cells, Faces, Fluid, ToolField, face_rheology, face_slot,
                                                neighbour)
from alloy.plugins.surface_flow.tool_geometry import (DISC, FREE_GAP, LIFT, RECT, ROLLER, V_PUSH, Pose, ToolSet, Under,
                                                      across_of, along, cell_at, centre_under, edge_of, in_cell, off,
                                                      pose_of,
                                                      push_at, relative, under_at, underside, widen, win_of, window)

vec4d = wp.types.vector(length=4, dtype=wp.float64)
FRONT = wp.constant(16)              # cells ahead of a sliding ceiling tool where its roll forms (the slope rule)
ROLL_SLOPE = wp.constant(1.0)        # the roll's front face: 45°, cross-section H²/2 (WP6, grid-independent)
ROLL_PASSES = 8                      # slope-rule passes per substep: the wedge relaxes ~one cell per pass


@wp.kernel
def tools_kernel(ts: ToolSet, n_tools: int, charts: wp.array[Chart], prof: wp.array[wp.vec3], cells: Cells,
                 tool_body: wp.array2d[wp.int32], cbody: wp.array2d[wp.int32], clocal: wp.array2d[wp.transform],
                 body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                 wins: wp.array2d[wp.vec4i], base: wp.array2d[wp.float32], h: wp.array2d[wp.float64], tf: ToolField,
                 ceil: wp.array2d[wp.float32], cover: wp.array2d[wp.float32], vel: wp.array2d[wp.vec3],
                 push: wp.array2d[wp.vec2], press: wp.array2d[wp.float32], cwet: wp.array2d[wp.int32],
                 pad_ratio: wp.array2d[wp.float32]):
    """The tools' undersides over the cells at the new poses. A cell a ceiling tool newly covers joins the skip's wet
    set when it holds any film: the tool pushes trace film below the dry threshold too (WP2 review 1b). Uncovered cells
    up to FRONT cells ahead of a sliding ceiling tool get its push direction (cover 0): its roll forms there by the
    slope rule (``excess_kernel``). ``press`` is the pad's pressure, an implicit pad's included (``pad_ratio``)."""
    w = wp.tid()
    for k in range(n_tools):                                     # the cells written the step before
        ch = charts[ts.chart[k]]
        win = wins[w, k]
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                c = cell_at(ch, i, j)
                ceil[w, c] = FREE_GAP
                cover[w, c] = 0.0
                vel[w, c] = wp.vec3(0.0, 0.0, 0.0)
                tf.wceil[w, c] = FREE_GAP
                tf.wcover[w, c] = 0.0
                tf.wvel[w, c] = wp.vec3(0.0, 0.0, 0.0)
                push[w, c] = wp.vec2(0.0, 0.0)
                press[w, c] = 0.0
                tf.pore[w, c] = 0
    for k in range(n_tools):
        kc = ts.chart[k]
        ch = charts[kc]
        tb, cb = tool_body[w, k], cbody[w, kc]
        ps = pose_of(ts, k, body_q, tb, cb, clocal[w, kc])
        win = win_of(ch, prof, ps)
        cell = edge_of(ch)
        centre = centre_under(ts, k, ch, prof, ps, cell)
        slide = along(relative(body_q, body_qd, body_com, tb, cb, ps, centre), centre)
        rolls = ts.ceiling[k] == 1 and win[1] >= win[0] and wp.length(slide) > V_PUSH
        lat = wp.cross(centre.n, wp.normalize(slide + wp.vec3(1.0e-12, 0.0, 0.0)))
        across = across_of(ts.shape[k], ts.size[k], ps, wp.normalize(lat + wp.vec3(0.0, 1.0e-12, 0.0)))
        if rolls:                                                 # the roll's zone ahead joins the window
            win = widen(ch, win, FRONT)
        wins[w, k] = win
        touched = int(0)
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                c = cell_at(ch, i, j)
                under = under_at(ts, k, ch, cells, ps, i, j, c, cell)
                if under.cover <= 0.0 or under.gap - base[w, c] - wp.float32(h[w, c]) > LIFT:
                    continue                                       # not under it, or lifted off this cell's film
                touched += 1
                if under.gap < tf.wceil[w, c]:                     # the work fields: every tool, a store's too
                    tf.wceil[w, c] = under.gap
                    tf.wcover[w, c] = under.cover
                    tf.wvel[w, c] = along(relative(body_q, body_qd, body_com, tb, cb, ps, under), under)
                pk = ts.stiffness[k] * pad_ratio[w, k]
                press[w, c] = wp.max(press[w, c], pk * wp.max(base[w, c] - under.gap, 0.0))  # the pad
                if ts.ceiling[k] == 1 and under.gap < ceil[w, c]:
                    ceil[w, c] = under.gap
                    cover[w, c] = under.cover
                    tf.pore[w, c] = 0
                    if ts.porous[k] == 1 and tf.room[w, k] > wp.float64(0.0):
                        tf.pore[w, c] = k + 1                      # a sponge with room: it absorbs, not ploughs
                    v = along(relative(body_q, body_qd, body_com, tb, cb, ps, under), under)
                    vel[w, c] = v
                    push[w, c] = push_at(v, under, ps)
                    if h[w, c] > wp.float64(0.0):
                        cwet[w, c] = 1
        if not rolls or touched == 0:
            continue
        for i in range(win[0], win[1] + 1):                         # the roll's zone ahead of a sliding tool
            for j in range(win[2], win[3] + 1):
                c = cell_at(ch, i, j)
                under = under_at(ts, k, ch, cells, ps, i, j, c, cell)
                if under.cover > 0.0 or cover[w, c] > 0.0:
                    continue
                ahead = wp.normalize(in_cell(slide, under) + wp.vec2(1.0e-12, 0.0))
                rel = in_cell(under.foot - ps.c, under)
                if wp.dot(rel, ahead) > 0.0 and wp.abs(rel[0] * ahead[1] - rel[1] * ahead[0]) <= across:
                    push[w, c] = ahead


@wp.func
def film_shear(fl: Fluid, w: int, c: int, film: float, v: wp.vec3) -> float:
    """Shear per unit sliding speed [Pa·s/m] of a tool gliding on a film ``film`` thick: ``μ/h + τ_y/|U|``, with the
    cell's rheology (``kernels.face_rheology``)."""
    rheo = face_rheology(fl, w, c, c)
    return fl.rho * rheo[1] / film + rheo[0] / wp.max(wp.length(v), 1.0e-3)


@wp.kernel
def tool_force_kernel(ts: ToolSet, n_tools: int, charts: wp.array[Chart], prof: wp.array[wp.vec3], cells: Cells,
                      tool_body: wp.array2d[wp.int32], cbody: wp.array2d[wp.int32],
                      clocal: wp.array2d[wp.transform], body_q: wp.array[wp.transform],
                      body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                      mobility: wp.array2d[wp.float32], sink: wp.array2d[wp.float32], h: wp.array2d[wp.float64],
                      base: wp.array2d[wp.float32], area: wp.array[wp.float64], fl: Fluid, h_wet: float,
                      dt: float, body_mass: wp.array[wp.float32], body_inertia: wp.array[wp.mat33], gravity: wp.vec3,
                      body_f: wp.array[wp.spatial_vector], drag_scale: wp.array2d[wp.float32],
                      pad_ratio: wp.array2d[wp.float32],
                      normal: wp.array2d[wp.float32]):
    """ADD each tool's pad pressure on the substrate with Coulomb friction, and the shear of film it glides on, to
    the tool; the reaction to the chart's body. ``drag_scale``: this exchange's cap on the explicit viscous drag (1:
    none; idempotent, as apply may run more than once per exchange)."""
    w = wp.tid()
    for k in range(n_tools):
        kc = ts.chart[k]
        ch = charts[kc]
        tb, cb = tool_body[w, k], cbody[w, kc]
        ps = pose_of(ts, k, body_q, tb, cb, clocal[w, kc])
        win = win_of(ch, prof, ps)
        cell = edge_of(ch)
        drag = float(0.0)                                         # Σ viscous coefficients [N·s/m]
        load = float(0.0)
        drag_scale[w, k] = 1.0
        normal[w, k] = 0.0                                        # the pad's spring load k·δ·A [N] (a brush's press)
        pad_ratio[w, k] = 1.0
        implicit = ts.implicit[k] == 1
        sums = pad_zero()
        pushed = float(0.0)                                       # the explicit pad's normal force [N]
        nw = wp.transform_vector(ps.X, wp.vec3(0.0, 0.0, 1.0))           # off the surface: a plane's normal
        if ch.kind != PLANE:
            nw = -wp.transform_vector(ps.X, ps.n)                         # a curved surface: against the pad's
        tool_com = com_world(body_q, body_com, tb)
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                c = cell_at(ch, i, j)
                under = under_at(ts, k, ch, cells, ps, i, j, c, cell)
                if under.cover <= 0.0:
                    continue
                a = under.cover * wp.float32(area[c])
                film = under.gap - base[w, c]
                nc = wp.transform_vector(ps.X, under.n)                   # this cell's normal, world
                if film <= 0.0:
                    load += ts.stiffness[k] * (-film) * a
                    if implicit:                                  # the springs and the damper's sink offset
                        vz = off(relative(body_q, body_qd, body_com, tb, cb, ps, under), under)
                        ka = ts.stiffness[k] * a
                        r = wp.transform_point(ps.X, under.point) - tool_com
                        # the paste's surface sinks too: the damper's offset and the step's own sinking
                        sums = pad_add(sums, ka, nc, r, ka * (-film - (ts.damping[k] + dt) * sink[w, k]))
                        pushed += wp.max(ka * (-film + ts.damping[k] * (-vz - sink[w, k])), 0.0)
                elif wp.float32(h[w, c]) > h_wet and wp.float32(h[w, c]) >= film - 1.0e-6:
                    v = along(relative(body_q, body_qd, body_com, tb, cb, ps, under), under)
                    drag += film_shear(fl, w, c, film, v) * a
        normal[w, k] = load
        if load <= 0.0 and drag <= 0.0:
            continue
        scale = wp.min(1.0, 0.5 / wp.max(drag * mobility[w, k] * dt, 1.0e-12))
        drag_scale[w, k] = scale
        eps = wp.max(ts.slip_eps, 2.0 * ts.friction[k] * load * dt * mobility[w, k])
        on_tool = wp.spatial_vector()
        on_face = wp.spatial_vector()
        face_com = wp.vec3(0.0, 0.0, 0.0)
        if cb >= 0:
            face_com = com_world(body_q, body_com, cb)
        ratio = float(1.0)
        if implicit and pushed > 0.0:                             # backward Euler on the tool against its pad
            rot = wp.quat_to_matrix(wp.transform_get_rotation(body_q[tb]))
            qd = body_qd[tb]
            pad = pad_wrench(sums, body_mass[tb], rot * body_inertia[tb] * wp.transpose(rot),
                             wp.vec3(qd[0], qd[1], qd[2]), wp.vec3(qd[3], qd[4], qd[5]), gravity, ts.damping[k],
                             dt)
            fn = wp.dot(wp.vec3(pad[0], pad[1], pad[2]), nw)
            if fn > 0.0:                                          # no adhesion: a pad only pushes
                ratio = fn / pushed
                on_tool = on_tool + pad
                on_face = on_face - wrench_at(wp.vec3(pad[0], pad[1], pad[2]), tool_com, face_com)
                on_face = on_face - wp.spatial_vector(0.0, 0.0, 0.0, pad[3], pad[4], pad[5])
            else:
                ratio = 0.0
            pad_ratio[w, k] = ratio
        for i in range(win[0], win[1] + 1):
            for j in range(win[2], win[3] + 1):
                c = cell_at(ch, i, j)
                under = under_at(ts, k, ch, cells, ps, i, j, c, cell)
                if under.cover <= 0.0:
                    continue
                a = under.cover * wp.float32(area[c])
                film = under.gap - base[w, c]
                vr = relative(body_q, body_qd, body_com, tb, cb, ps, under)
                vt = along(vr, under)
                vw = wp.transform_vector(ps.X, vt)
                f = wp.vec3(0.0, 0.0, 0.0)
                if film <= 0.0:                                   # on the substrate: the pad, Coulomb friction
                    # the damper acts on the approach to the surface, which a yielding paste moves down (forming.py)
                    p = wp.max(ts.stiffness[k] * (-film + ts.damping[k] * (-off(vr, under) - sink[w, k])), 0.0)
                    f = p * a * wp.transform_vector(ps.X, under.n) - (ts.friction[k] * p * a / wp.max(wp.length(vw),
                                                                                                      eps)) * vw
                    if implicit:                                  # the normal part is the solve's; friction scales
                        f = -(ratio * ts.friction[k] * p * a / wp.max(wp.length(vw), eps)) * vw
                elif wp.float32(h[w, c]) > h_wet and wp.float32(h[w, c]) >= film - 1.0e-6:
                    f = -(scale * film_shear(fl, w, c, film, vt) * a) * vw
                else:
                    continue
                at = wp.transform_point(ps.X, under.point)
                on_tool = on_tool + wrench_at(f, at, tool_com)
                on_face = on_face - wrench_at(f, at, face_com)
        add_wrench(body_f, tb, on_tool)
        if cb >= 0:
            add_wrench(body_f, cb, on_face)


@wp.func
def push_weights(cells: Cells, faces: Faces, d: wp.vec2, c: int) -> vec4d:
    """Shares of film pushed along ``d`` out of cell c through its faces (i-, i+, j-, j+): split by |d| between the
    x and y neighbours ahead, over those that exist (interior faces). float64, and they sum to exactly 1 (the second
    is the first's complement), so the move is volume exact; all 0 when no neighbour exists."""
    fs = cells.face[c]
    wt = vec4d(wp.float64(0.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
    sx = 0
    if d[0] > 0.0:
        sx = 1
    sy = 2
    if d[1] > 0.0:
        sy = 3
    ax = wp.float64(0.0)
    ay = wp.float64(0.0)
    if faces.kind[fs[sx]] == INTERIOR:
        ax = wp.float64(wp.abs(d[0]))
    if faces.kind[fs[sy]] == INTERIOR:
        ay = wp.float64(wp.abs(d[1]))
    if ax + ay <= wp.float64(0.0):
        return wt
    share = ax / (ax + ay)
    wt[sx] = share
    wt[sy] = wp.float64(1.0) - share
    return wt


@wp.func
def roll_excess(cells: Cells, faces: Faces, h: wp.array2d[wp.float64], base: wp.array2d[wp.float32], w: int, c: int,
                d: wp.vec2) -> wp.float64:
    """In front of a sliding tool: half the film surface's rise over ROLL_SLOPE toward the cell ahead (along d's main
    axis), so a roll of cross-section A stands as a 45° wedge of height sqrt(2A) whatever the cell size (WP6)."""
    fs = cells.face[c]
    s = 2
    if wp.abs(d[0]) >= wp.abs(d[1]):
        s = 0
        if d[0] > 0.0:
            s = 1
    elif d[1] > 0.0:
        s = 3
    f = fs[s]
    if faces.kind[f] != INTERIOR:
        return wp.float64(0.0)
    n = neighbour(faces, f, c)
    rise = h[w, c] + wp.float64(base[w, c]) - h[w, n] - wp.float64(base[w, n])
    over = rise - wp.float64(ROLL_SLOPE * faces.dist[f])
    if over <= wp.float64(0.0):
        return wp.float64(0.0)
    return wp.min(wp.float64(0.5) * over, h[w, c])


@wp.kernel
def excess_kernel(cells: Cells, faces: Faces, tick: wp.array[wp.int32], stride: int, on: wp.array2d[wp.int32],
                  tf: ToolField, base: wp.array2d[wp.float32], h: wp.array2d[wp.float64], roll_only: int, sp: Species,
                  excess: wp.array2d[wp.float64]):
    """The film above a tool's underside [m of this cell's depth] that leaves this substep (one cell on), and its
    species at the cell's concentration (``tf.sp_ex``)."""
    w, c = wp.tid()
    if tick[w] % stride != 0 or on[w, cells.tile[c]] == 0:
        return
    e = wp.float64(0.0)
    cv = tf.cover[w, c]
    d = tf.push[w, c]
    pore = tf.pore[w, c] > 0
    if cv > 0.0 and roll_only == 0:
        over = h[w, c] + wp.float64(base[w, c]) - wp.float64(tf.ceil[w, c])
        wt = push_weights(cells, faces, d, c)
        if over > wp.float64(0.0) and (pore or wt[0] + wt[1] + wt[2] + wt[3] > wp.float64(0.0)):
            e = wp.min(wp.float64(cv) * over, h[w, c])
    elif cv <= 0.0 and (d[0] != 0.0 or d[1] != 0.0):          # the roll's zone: steeper than its slope moves on
        e = roll_excess(cells, faces, h, base, w, c, d)
    excess[w, c] = e
    if sp.count > 0:
        share = wp.float64(0.0)
        if e > wp.float64(0.0):
            share = e / h[w, c]
        for k in range(sp.count):
            tf.sp_ex[w, c, k] = sp.mass[w, c, k] * share
            if pore:
                tf.absorb_sp[w, c, k] = tf.absorb_sp[w, c, k] + tf.sp_ex[w, c, k]
    if pore:
        tf.absorb[w, c] = tf.absorb[w, c] + e                      # into the sponge, not on to a neighbour


@wp.kernel
def displace_kernel(cells: Cells, faces: Faces, tick: wp.array[wp.int32], stride: int, on: wp.array2d[wp.int32],
                    tf: ToolField, h_dry: wp.float64, excess: wp.array2d[wp.float64], sp: Species,
                    h: wp.array2d[wp.float64], cwet: wp.array2d[wp.int32]):
    """Each cell gives its excess and gathers what its neighbours push into it (one writer per cell, volume exact);
    the species go with it (WP9's review: a pressed pad left them behind, soap 10 -> 4e4 kg/m³)."""
    w, c = wp.tid()
    if tick[w] % stride != 0 or on[w, cells.tile[c]] == 0:
        return
    fs = cells.face[c]
    hn = h[w, c] - excess[w, c]
    for s in range(4):
        f = fs[s]
        if faces.kind[f] != INTERIOR:
            continue
        n = neighbour(faces, f, c)
        e = excess[w, n]
        if e <= wp.float64(0.0) or tf.pore[w, n] > 0:
            continue
        wt = push_weights(cells, faces, tf.push[w, n], n)
        hn += e * cells.area[n] * wt[face_slot(cells, n, f)] / cells.area[c]
    h[w, c] = hn
    if hn > h_dry:
        cwet[w, c] = 1
    for k in range(sp.count):                                    # the same move for each species
        m = sp.mass[w, c, k] - tf.sp_ex[w, c, k]
        for s in range(4):
            f = fs[s]
            if faces.kind[f] != INTERIOR:
                continue
            n = neighbour(faces, f, c)
            if excess[w, n] <= wp.float64(0.0) or tf.pore[w, n] > 0:
                continue
            wt = push_weights(cells, faces, tf.push[w, n], n)
            m += tf.sp_ex[w, n, k] * cells.area[n] * wt[face_slot(cells, n, f)] / cells.area[c]
        sp.mass[w, c, k] = m


@wp.kernel
def count_capped_kernel(drag_scale: wp.array2d[wp.float32], drag_capped: wp.array2d[wp.int32]):
    """Once per exchange (update): count the exchanges whose drag cap bound (the tool felt less viscous drag than
    physical; WP2 review)."""
    w, k = wp.tid()
    if drag_scale[w, k] < 1.0:
        drag_capped[w, k] = drag_capped[w, k] + 1


__all__ = ["DISC", "FREE_GAP", "Pose", "RECT", "ROLLER", "ToolSet", "Under", "count_capped_kernel",
           "displace_kernel", "excess_kernel", "pose_of", "push_weights", "tool_force_kernel", "tools_kernel",
           "underside", "window"]                                  # geometry re-exported from tool_geometry.py
