# SPDX-License-Identifier: Apache-2.0
"""Pad contact and wear kernels of the removal plug-in.

One thread per world: each world owns its grid and its tool, so every element has one writer and no atomics are
needed. Both kernels walk the same cells with the same functions, so the wear step sees exactly the pressure the
force step applied (the pressure depends only on the pre-solve state and the plug-in's grid).

Pad model (Winkler foundation): each grid cell under the pad face carries pressure
``p = stiffness·(δ + damping·dδ/dt)`` (never negative), where ``δ`` is how far the pad face lies below the cell's
surface point along the surface normal. Friction opposes the slip of the pad over the cell, ``μ·p`` per area,
viscous below ``slip_eps`` (raised per step so the explicit friction stays stable on both bodies). Boundary cells count with the fraction of the cell under the pad, so the load does not
jump as the pad moves over the grid.

Wear (Archard, per area): a cell under pressure ``p`` that the pad slides over by ``ds`` loses depth
``dh = (K / H)·p·ds``; with ``ds = v·dt`` this is Preston's ``dh/dt = k_p·p·v`` with ``k_p = K / H``. Integrated
over the contact, the removed volume is ``V = (K / H)·F_N·s`` (Archard). The removed mass ``ρ·A·dh`` goes to the
tool's sink (fraction ``capture``) or stays on the cell as loose material; a wiping pad lifts loose material from a
cell by the fraction ``1 - exp(-ds / wipe_length)``.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import add_wrench, com_world, point_velocity, wrench_at
from alloy.plugins.removal.surface import cell_window, moved_point, patch_frame

MIN_FACING = wp.constant(0.2)   # pad and surface normals closer to perpendicular than this: edge-on, no contact


@wp.struct
class Pad:
    center: wp.vec3          # pad face centre, tool body frame [m]
    normal: wp.vec3          # unit pad face normal (out of the pad), tool body frame
    radius: wp.float32       # [m]
    stiffness: wp.float32    # [Pa/m]
    damping: wp.float32      # Kelvin-Voigt time [s]
    friction: wp.float32     # Coulomb coefficient
    slip_eps: wp.float32     # [m/s]


@wp.struct
class Grid:
    half: wp.vec2            # half extent of the face [m]
    cell: wp.vec2            # cell size [m]
    feedback: wp.int32       # 1: the surface height (profile - depth) moves the contact


@wp.struct
class Wear:
    rate: wp.float32         # Preston coefficient K / H [1/Pa]
    density: wp.float32      # [kg/m^3]
    capture: wp.float32      # fraction of removed mass sent to the tool's sink
    wipe_length: wp.float32  # [m]; 0: no pick-up of loose material


@wp.struct
class Frames:
    face: wp.transform       # patch frame, world
    center: wp.vec3          # pad centre, world
    n_pad: wp.vec3           # pad normal, world
    n_s: wp.vec3             # surface normal, world (out of the material)
    facing: wp.float32       # -n_pad . n_s: 1 when the pad lies flat on the face
    tool_com: wp.vec3
    face_com: wp.vec3


@wp.struct
class Hit:
    spot: wp.vec3            # surface point of the cell, world
    pad: wp.vec3             # pad face point above it (along n_s), world
    depth: wp.float32        # indentation δ [m]: > 0 when the pad face is below the surface
    cover: wp.float32        # fraction of the cell under the pad, [0, 1]


@wp.func
def frames(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], tool: wp.int32, face_body: wp.int32,
           face_local: wp.transform, pad: Pad) -> Frames:
    fr = Frames()
    fr.face = patch_frame(body_q, face_body, face_local)
    fr.center = wp.transform_point(body_q[tool], pad.center)
    fr.n_pad = wp.normalize(wp.transform_vector(body_q[tool], pad.normal))
    fr.n_s = wp.transform_vector(fr.face, wp.vec3(0.0, 0.0, 1.0))
    fr.facing = -wp.dot(fr.n_pad, fr.n_s)
    fr.tool_com = com_world(body_q, body_com, tool)
    fr.face_com = wp.vec3(0.0, 0.0, 0.0)
    if face_body >= 0:
        fr.face_com = com_world(body_q, body_com, face_body)
    return fr


@wp.func
def pad_window(fr: Frames, pad: Pad, grid: Grid, nx: int, ny: int) -> wp.vec4i:
    """Cells the pad may cover; empty when the pad is edge-on or its centre is more than a radius above the face."""
    c = wp.transform_point(wp.transform_inverse(fr.face), fr.center)
    if fr.facing < MIN_FACING or c[2] > pad.radius:
        return wp.vec4i(0, -1, 0, -1)
    reach = pad.radius + wp.max(grid.cell[0], grid.cell[1])
    return cell_window(c[0] + grid.half[0], c[1] + grid.half[1], reach, grid.cell, nx, ny)


@wp.func
def cell_height(profile: wp.array3d[wp.float64], depth: wp.array3d[wp.float64], w: int, i: int, j: int,
                grid: Grid) -> wp.float32:
    if grid.feedback == 0:
        return wp.float32(0.0)
    return wp.float32(profile[w, i, j] - depth[w, i, j])


@wp.func
def cell_hit(fr: Frames, pad: Pad, grid: Grid, i: int, j: int, height: wp.float32) -> Hit:
    hit = Hit()
    x = (float(i) + 0.5) * grid.cell[0] - grid.half[0]
    y = (float(j) + 0.5) * grid.cell[1] - grid.half[1]
    hit.spot = wp.transform_point(fr.face, wp.vec3(x, y, height))
    t = wp.dot(fr.n_pad, fr.center - hit.spot) / wp.dot(fr.n_pad, fr.n_s)
    hit.pad = hit.spot + t * fr.n_s
    hit.depth = -t
    size = 0.5 * (grid.cell[0] + grid.cell[1])
    hit.cover = wp.clamp((pad.radius - wp.length(hit.pad - fr.center)) / size + 0.5, 0.0, 1.0)
    return hit


@wp.func
def slip_velocity(body_qd: wp.array[wp.spatial_vector], fr: Frames, tool: wp.int32, face_body: wp.int32,
                  hit: Hit) -> wp.vec3:
    """Velocity of the pad point relative to the surface point under it."""
    v = point_velocity(body_qd, fr.tool_com, tool, hit.pad)
    if face_body >= 0:
        v = v - point_velocity(body_qd, fr.face_com, face_body, hit.spot)
    return v


@wp.func
def pressure(hit: Hit, v_rel: wp.vec3, fr: Frames, pad: Pad) -> wp.float32:
    if hit.depth <= 0.0 or hit.cover <= 0.0:
        return wp.float32(0.0)
    closing = -wp.dot(v_rel, fr.n_s)
    return wp.max(pad.stiffness * (hit.depth + pad.damping * closing), 0.0)


@wp.kernel
def pad_force_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                     body_com: wp.array[wp.vec3], mobility: wp.array[wp.float32], tool: wp.array[wp.int32],
                     face_body: wp.array[wp.int32], face_local: wp.array[wp.transform], pad: Pad, grid: Grid,
                     dt: wp.float32, profile: wp.array3d[wp.float64], depth: wp.array3d[wp.float64],
                     body_f: wp.array[wp.spatial_vector]):
    """ADD the pad's pressure and friction to the tool (and the reaction to the workpiece body, if it has one)."""
    w = wp.tid()
    tb, fb = tool[w], face_body[w]
    fr = frames(body_q, body_com, tb, fb, face_local[w], pad)
    win = pad_window(fr, pad, grid, depth.shape[1], depth.shape[2])
    area = grid.cell[0] * grid.cell[1]
    load = float(0.0)
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            hit = cell_hit(fr, pad, grid, i, j, cell_height(profile, depth, w, i, j, grid))
            load += pressure(hit, slip_velocity(body_qd, fr, tb, fb, hit), fr, pad) * hit.cover * area
    if load <= 0.0:
        return
    # explicit viscous friction is stable while (mu F / eps)·dt·mobility <= 1/2, mobility = summed point mobility of
    # tool and floating workpiece at the pad (translation and rotation, stability.py): raise eps for heavy loads
    eps = wp.max(pad.slip_eps, 2.0 * pad.friction * load * dt * mobility[w])
    on_tool = wp.spatial_vector()
    on_face = wp.spatial_vector()
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            hit = cell_hit(fr, pad, grid, i, j, cell_height(profile, depth, w, i, j, grid))
            v = slip_velocity(body_qd, fr, tb, fb, hit)
            f_n = pressure(hit, v, fr, pad) * hit.cover * area
            if f_n <= 0.0:
                continue
            vt = v - wp.dot(v, fr.n_s) * fr.n_s
            f = f_n * fr.n_s - (pad.friction * f_n / wp.max(wp.length(vt), eps)) * vt
            on_tool = on_tool + wrench_at(f, hit.pad, fr.tool_com)
            on_face = on_face - wrench_at(f, hit.pad, fr.face_com)
    add_wrench(body_f, tb, on_tool)
    if fb >= 0:
        add_wrench(body_f, fb, on_face)


@wp.kernel
def wear_kernel(q_in: wp.array[wp.transform], qd_in: wp.array[wp.spatial_vector], q_out: wp.array[wp.transform],
                body_com: wp.array[wp.vec3], tool: wp.array[wp.int32], face_body: wp.array[wp.int32],
                face_local: wp.array[wp.transform], pad: Pad, grid: Grid, wear: Wear,
                profile: wp.array3d[wp.float64], depth: wp.array3d[wp.float64], loose: wp.array3d[wp.float64],
                removed: wp.array[wp.float64], sink: wp.array[wp.float64], load: wp.array[wp.float32],
                contact: wp.array[wp.int32]):
    """Remove material under the pad over the step and move it to the cell (loose) or the tool's sink, exactly."""
    w = wp.tid()
    tb, fb = tool[w], face_body[w]
    fr = frames(q_in, body_com, tb, fb, face_local[w], pad)
    win = pad_window(fr, pad, grid, depth.shape[1], depth.shape[2])
    area = grid.cell[0] * grid.cell[1]
    total = float(0.0)
    cells = int(0)
    cut = wp.float64(0.0)
    picked = wp.float64(0.0)
    for i in range(win[0], win[1] + 1):
        for j in range(win[2], win[3] + 1):
            hit = cell_hit(fr, pad, grid, i, j, cell_height(profile, depth, w, i, j, grid))
            p = pressure(hit, slip_velocity(qd_in, fr, tb, fb, hit), fr, pad)
            if p <= 0.0:
                continue
            total += p * hit.cover * area
            cells += 1
            slip = (moved_point(q_in, q_out, tb, hit.pad) - hit.pad) - (moved_point(q_in, q_out, fb, hit.spot) - hit.spot)
            ds = wp.length(slip - wp.dot(slip, fr.n_s) * fr.n_s)
            lifted = wp.float64(0.0)
            if wear.wipe_length > 0.0:
                lifted = loose[w, i, j] * wp.float64(1.0 - wp.exp(-hit.cover * ds / wear.wipe_length))
            dh = wp.float64(hit.cover * wear.rate * p * ds)
            dm = wp.float64(wear.density * area) * dh
            kept = dm * wp.float64(1.0 - wear.capture)
            depth[w, i, j] = depth[w, i, j] + dh
            loose[w, i, j] = loose[w, i, j] - lifted + kept
            cut = cut + dm
            picked = picked + lifted + (dm - kept)
    removed[w] = removed[w] + cut
    sink[w] = sink[w] + picked
    load[w] = total
    contact[w] = cells


__all__ = ["Grid", "Pad", "Wear", "pad_force_kernel", "wear_kernel"]
