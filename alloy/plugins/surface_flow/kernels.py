# SPDX-License-Identifier: Apache-2.0
"""The film step (design note §3.2-3.3): staggered finite volumes, depth-averaged momentum with implicit wall friction.

Per substep, each element with one writer and no atomics:
1. ``potential_kernel`` (world, cell): ``P = g_n·(h + b) + p_c/ρ``, ``p_c = -σ·∇²(h + b) + σ·K`` (K: substrate
   curvature, + convex), with a contact-angle condition where the film meets dry cells;
2. ``face_kernel`` (world, face): ``u' = (u/dt + G - u·∇u + τ/(ρh_f)) / (1/dt + 3ν/(h_f²·φ) + S/h_f)`` (S: film
   arriving at rest), ``q = h_d·u'`` (h_d upstream of u'), ``G = g_t - ∂P/∂n``, φ the thin-layer Bingham factor
   (Balmforth & Craster 1999), Stelling & Duinmeijer's momentum-conservative advection, and a speed cap;
3. ``limit_kernel`` and ``cell_kernel`` (``cell_kernels.py``): the outflow limiter, then each cell's fluxes, sources,
   evaporation and drains, every volume booked in float64. ``frames_kernel`` resolves the frame's gravity per exchange.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.surface_flow.geometry import chart_xform
from alloy.plugins.surface_flow.kernels_structs import Cells, Faces, ToolField

INTERIOR = wp.constant(0)
WALL = wp.constant(1)
OPEN = wp.constant(2)


@wp.struct
class Fluid:
    rho: wp.float32
    nu: wp.float32
    tau_y: wp.float32
    sigma: wp.float32
    beta: wp.float32                  # 1: momentum advection on; 0: off (creeping films)
    u_cap: wp.float32
    h_dry: wp.float64                 # below this a face carries no flux
    h_wet: wp.float64                 # below this a face carries no momentum flux (wet/dry fronts)
    evaporation: wp.float64           # m/s
    per_cell: wp.int32                # 1: the yield_stress and viscosity inputs give the rheology per cell
    yield_cell: wp.array2d[wp.float32]   # [W, C] Pa (the "yield_stress" input)
    visc_cell: wp.array2d[wp.float32]    # [W, C] Pa·s (the "viscosity" input; never below the material's)
    per_sink: wp.int32                # 1: the evaporation and absorption inputs give the film's sinks per cell
    evap_cell: wp.array2d[wp.float32]    # [W, C] m/s (the "evaporation" input; starts at the material's)
    absorb_cell: wp.array2d[wp.float32]  # [W, C] m/s (the "absorption" input: into a porous substrate)
    thixo: wp.int32                   # 1: the yield scales with the structure lam (rheology.py, Moore)
    lam: wp.array2d[wp.float32]       # [W, C] structure in [0, 1]
    krieger: wp.int32                 # 1: the viscosity scales with the solids' Krieger-Dougherty factor
    vfac: wp.array2d[wp.float32]      # [W, C] that factor (>= 1)


@wp.func
def face_rheology(fl: Fluid, w: int, a: int, b: int) -> wp.vec2:
    """(τ_y [Pa], ν [m²/s]) on the face between cells a and b (-1: none): the material's, or the per-cell inputs'
    mean, with ν never below the material's (the step bound assumes it); times the mean structure (yield) and the
    solids' Krieger factor (viscosity) where those are on (rheology.py)."""
    if fl.per_cell == 0 and fl.thixo == 0 and fl.krieger == 0:
        return wp.vec2(fl.tau_y, fl.nu)
    ca = a
    cb = b
    if ca < 0:
        ca = cb
    if cb < 0:
        cb = ca
    ty = fl.tau_y
    nu = fl.nu
    if fl.per_cell != 0:
        ty = 0.5 * (fl.yield_cell[w, ca] + fl.yield_cell[w, cb])
        nu = wp.max(0.5 * (fl.visc_cell[w, ca] + fl.visc_cell[w, cb]) / fl.rho, fl.nu)
    if fl.thixo != 0:
        ty = ty * 0.5 * (fl.lam[w, ca] + fl.lam[w, cb])
    if fl.krieger != 0:
        nu = nu * 0.5 * (fl.vfac[w, ca] + fl.vfac[w, cb])
    return wp.vec2(ty, nu)


TOUCH = wp.constant(1.0e-6)           # m: a film this close below a tool's underside touches it


@wp.func
def touch(tf: ToolField, h: wp.array2d[wp.float64], base: wp.array2d[wp.float32], w: int, c: int) -> float:
    """The share of cell c whose film reaches a tool's underside."""
    if wp.float32(h[w, c]) + base[w, c] < tf.ceil[w, c] - TOUCH:
        return 0.0
    return tf.cover[w, c]


@wp.func
def face_touch(tf: ToolField, h: wp.array2d[wp.float64], base: wp.array2d[wp.float32], w: int, a: int,
               b: int) -> float:
    """1 when face (a, b) is under a tool and the film fills the gap on one side, else 0. The cells' covers are the
    footprint smoothed over a cell (``tool_kernels.soft_edge``), so the face's midpoint is inside it exactly when
    they sum to more than 1: a sharp edge at the faces, which the film needs to leave half a blade's gap behind."""
    if tf.cover[w, a] + tf.cover[w, b] <= 1.0:
        return 0.0
    if wp.max(touch(tf, h, base, w, a), touch(tf, h, base, w, b)) <= 0.0:
        return 0.0
    return 1.0


@wp.struct
class Frame:
    g: wp.array2d[wp.vec3]            # [W, K] gravity minus the frame's acceleration, chart frame
    omega: wp.array2d[wp.vec3]        # [W, K] the frame's angular velocity, chart frame
    alpha: wp.array2d[wp.vec3]        # [W, K] the frame's angular acceleration, chart frame (the Euler force)
    com: wp.array2d[wp.vec3]          # [W, K] the body's COM, chart frame


@wp.kernel
def frames_kernel(body_q: wp.array[wp.transform], qd_in: wp.array[wp.spatial_vector],
                  qd_out: wp.array[wp.spatial_vector], cbody: wp.array2d[wp.int32],
                  clocal: wp.array2d[wp.transform], gravity: wp.vec3, dt: wp.float32, tick: wp.array[wp.int32],
                  stride: int, g_sum: wp.array2d[wp.vec3], alpha_sum: wp.array2d[wp.vec3],
                  g_out: wp.array2d[wp.vec3], omega_out: wp.array2d[wp.vec3], alpha_out: wp.array2d[wp.vec3]):
    """Each chart's gravity minus its frame's acceleration, angular velocity and acceleration, in its own frame. A
    strided film gets the mean over its window, so one exchange's spike is not applied k times (WP2 review)."""
    w, k = wp.tid()
    b = cbody[w, k]
    rot = wp.transform_get_rotation(chart_xform(body_q, b, clocal[w, k]))
    g = gravity
    omega = wp.vec3(0.0, 0.0, 0.0)
    alpha = wp.vec3(0.0, 0.0, 0.0)
    if b >= 0:
        q0 = qd_in[b]
        q1 = qd_out[b]
        g = g - (wp.vec3(q1[0], q1[1], q1[2]) - wp.vec3(q0[0], q0[1], q0[2])) / dt
        omega = wp.vec3(q1[3], q1[4], q1[5])
        alpha = (omega - wp.vec3(q0[3], q0[4], q0[5])) / dt
    gs = g_sum[w, k] + wp.quat_rotate_inv(rot, g)
    als = alpha_sum[w, k] + wp.quat_rotate_inv(rot, alpha)
    if tick[w] % stride == 0:
        n = float(stride)
        g_out[w, k] = gs / n
        alpha_out[w, k] = als / n
        omega_out[w, k] = wp.quat_rotate_inv(rot, omega)
        gs = wp.vec3(0.0, 0.0, 0.0)
        als = wp.vec3(0.0, 0.0, 0.0)
    g_sum[w, k] = gs
    alpha_sum[w, k] = als


@wp.func
def inflow(acc: wp.array2d[wp.float64], inv_dt: wp.float64, fixed: wp.array2d[wp.float64], w: int,
           c: int) -> wp.float64:
    """The film's source over its step [m/s]: the supply's pending volume per area (``src_acc``) over the film step,
    plus the steady ``source`` input."""
    return wp.max(acc[w, c] * inv_dt + fixed[w, c], wp.float64(0.0))


@wp.func
def body_force(fr: Frame, w: int, k: int, p: wp.vec3) -> wp.vec3:
    """Gravity per unit mass felt at chart-frame point ``p``: gravity, minus the frame's acceleration, the centrifugal
    and the Euler terms (Coriolis is left out)."""
    om = fr.omega[w, k]
    r = p - fr.com[w, k]
    return fr.g[w, k] - wp.cross(om, wp.cross(om, r)) - wp.cross(fr.alpha[w, k], r)


@wp.func
def neighbour(faces: Faces, f: int, c: int) -> int:
    if faces.b[f] == c:
        return faces.a[f]
    return faces.b[f]


@wp.func
def inside_cell(faces: Faces, f: int) -> int:
    """The chart cell of an open or wall face."""
    if faces.a[f] < 0:
        return faces.b[f]
    return faces.a[f]


@wp.func
def dry(cells: Cells, h: wp.array2d[wp.float64], w: int, c: int, h_cl: float) -> bool:
    """Dry substrate for a contact line: no film above h_cl, and not a drain's hole (SC9: a drained cell is empty)."""
    return h[w, c] <= wp.float64(h_cl) and cells.drain[c] < 0


@wp.func
def front_weights(cells: Cells, faces: Faces, h: wp.array2d[wp.float64], w: int, c: int, h_cl: float) -> wp.vec2:
    """Per-axis weights of wet cell c's contact line: an axis is present when a face looks onto a dry cell, each
    present axis ``a/sqrt(a_i² + a_j²)`` (a front 1, a corner 1/√2, a strip's sides 1; WP2 review F2, F2b)."""
    fs = cells.face[c]
    present = wp.vec2(0.0, 0.0)
    for s in range(4):
        f = fs[s]
        if faces.kind[f] == INTERIOR:
            if dry(cells, h, w, neighbour(faces, f, c), h_cl):
                present[s // 2] = 1.0
    m = wp.length(present)
    if m > 0.0:
        present = present / m
    return present


@wp.func
def face_slot(cells: Cells, c: int, f: int) -> int:
    """Which of cell c's faces (i-, i+, j-, j+) face f is."""
    fs = cells.face[c]
    slot = int(0)
    for s in range(4):
        if fs[s] == f:
            slot = s
    return slot


@wp.kernel
def potential_kernel(cells: Cells, faces: Faces, fr: Frame, fl: Fluid, wet: wp.array[wp.vec2],
                     tick: wp.array[wp.int32], stride: int, on: wp.array2d[wp.int32], h: wp.array2d[wp.float64],
                     base: wp.array2d[wp.float32], pot: wp.array2d[wp.float32]):
    """``P = g_n·(h + b) + p_c/ρ``. With a contact angle (``wet[k] = (h_cl, tan θ)``, tan θ < 0: none), a wet cell
    (h > h_cl) next to a dry one meets it at slope tan θ in its curvature (a height-function contact-angle condition,
    which holds on coarse cells); dry cells carry no capillary pressure."""
    w, c = wp.tid()
    if tick[w] % stride != 0 or on[w, cells.tile[c]] == 0:
        return
    k = cells.chart[c]
    g = body_force(fr, w, k, cells.pos[c])
    gn = wp.max(-wp.dot(g, cells.nrm[c]), 0.0)          # hydrostatic only where gravity presses the film on
    eta = wp.float32(h[w, c]) + base[w, c]
    contact = wet[k]
    angle = contact[1] >= 0.0
    is_wet = h[w, c] > wp.float64(contact[0])
    pc = float(0.0)
    if fl.sigma > 0.0 and (is_wet or not angle):
        lap = float(0.0)
        fs = cells.face[c]
        weight = wp.vec2(0.0, 0.0)
        if angle:
            weight = front_weights(cells, faces, h, w, c, contact[0])
        for s in range(4):
            f = fs[s]
            if faces.kind[f] != INTERIOR:
                continue
            o = neighbour(faces, f, c)
            if angle and dry(cells, h, w, o, contact[0]):
                # the surface reaches the dry side at θ along the line's normal
                lap -= contact[1] * weight[s // 2] * faces.length[f]
            else:
                lap += (wp.float32(h[w, o]) + base[w, o] - eta) * faces.length[f] / faces.dist[f]
        pc = -fl.sigma * lap / wp.float32(cells.area[c]) + fl.sigma * cells.curv[c]
    pot[w, c] = gn * eta + pc / fl.rho


@wp.func
def upwind_depth(h: wp.array2d[wp.float64], faces: Faces, w: int, g: int, sign: float) -> wp.float64:
    a = faces.a[g]
    b = faces.b[g]
    if (sign > 0.0 and a >= 0) or b < 0:
        return h[w, a]
    return h[w, b]


@wp.func
def face_velocity(u: wp.array2d[wp.float32], h: wp.array2d[wp.float64], faces: Faces, fl: Fluid, w: int,
                  g: int) -> float:
    """Velocity at face g; 0 on walls and where its donor cell is thinner than ``h_wet``."""
    if g < 0:
        return 0.0
    if faces.kind[g] == WALL:
        return 0.0
    ug = u[w, g]
    if upwind_depth(h, faces, w, g, ug) <= fl.h_wet:
        return 0.0
    return ug


@wp.func
def disch(q: wp.array2d[wp.float32], faces: Faces, w: int, g: int) -> float:
    if g < 0:
        return 0.0
    if faces.kind[g] == WALL:
        return 0.0
    return q[w, g]


@wp.func
def advection(uf: wp.array2d[wp.float32], q: wp.array2d[wp.float32], h: wp.array2d[wp.float64], faces: Faces,
              fl: Fluid, w: int, f: int) -> float:
    """The advective acceleration at face f in the momentum-conservative velocity form of Stelling & Duinmeijer
    (2003): ``[(q̄u*)_b - (q̄u*)_a - u_f·(q̄_b - q̄_a)] / (h̄_f·Δx)`` along the face and the same across it, with
    ``q̄`` the mean discharge at the momentum cell's sides and ``u*`` the upwind face velocity. Momentum enters a
    face that is still at rest (a wet/dry front then runs at Ritter's speed); first order, no curvilinear terms."""
    a = faces.a[f]
    b = faces.b[f]
    hbar = 0.5 * wp.float32(h[w, a] + h[w, b])
    if hbar <= wp.float32(fl.h_wet):
        return 0.0
    u = uf[w, f]
    qf = q[w, f]
    line = faces.line[f]
    qa = 0.5 * (disch(q, faces, w, line[0]) + qf)
    qb = 0.5 * (qf + disch(q, faces, w, line[1]))
    ua = u
    if qa > 0.0:
        ua = face_velocity(uf, h, faces, fl, w, line[0])
    ub = face_velocity(uf, h, faces, fl, w, line[1])
    if qb > 0.0:
        ub = u
    along = (qb * ub - qa * ua - u * (qb - qa)) / faces.dist[f]
    perp = faces.perp[f]
    par = faces.par[f]
    qt_hi = 0.5 * (disch(q, faces, w, perp[1]) + disch(q, faces, w, perp[3]))
    qt_lo = 0.5 * (disch(q, faces, w, perp[0]) + disch(q, faces, w, perp[2]))
    u_lo = u
    if qt_lo > 0.0:
        u_lo = face_velocity(uf, h, faces, fl, w, par[0])
    u_hi = face_velocity(uf, h, faces, fl, w, par[1])
    if qt_hi > 0.0:
        u_hi = u
    across = (qt_hi * u_hi - qt_lo * u_lo - u * (qt_hi - qt_lo)) / faces.length[f]
    return fl.beta * (along + across) / hbar


@wp.func
def transverse(gface: wp.array2d[wp.float32], faces: Faces, w: int, f: int) -> float:
    """The mean of the driving acceleration on the faces across face f (last substep): its transverse component."""
    perp = faces.perp[f]
    total = float(0.0)
    n = float(0.0)
    for s in range(4):
        if perp[s] >= 0:
            total += gface[w, perp[s]]
            n += 1.0
    if n == 0.0:
        return 0.0
    return total / n


@wp.kernel
def face_kernel(cells: Cells, faces: Faces, fr: Frame, fl: Fluid, wet: wp.array[wp.vec2], tf: ToolField,
                tick: wp.array[wp.int32], stride: int, on: wp.array2d[wp.int32], h: wp.array2d[wp.float64],
                base: wp.array2d[wp.float32],
                pot: wp.array2d[wp.float32],
                u: wp.array2d[wp.float32], mom: wp.array2d[wp.vec3],
                q: wp.array2d[wp.float32], g_old: wp.array2d[wp.float32], acc: wp.array2d[wp.float64],
                inv_dt: wp.float64, src_fixed: wp.array2d[wp.float64], dt: wp.float32, u_new: wp.array2d[wp.float32],
                q_new: wp.array2d[wp.float32], gface: wp.array2d[wp.float32]):
    w, f = wp.tid()
    if tick[w] % stride != 0 or on[w, faces.tile[f]] == 0:
        return
    kind = faces.kind[f]
    if kind == WALL:
        u_new[w, f] = 0.0
        q_new[w, f] = 0.0
        gface[w, f] = 0.0
        return
    a = faces.a[f]
    b = faces.b[f]
    d = faces.dir[f]
    G = wp.dot(body_force(fr, w, faces.chart[f], faces.pos[f]), d)
    tau = float(0.0)
    added = wp.float64(0.0)                            # m/s of film arriving at rest: a momentum sink
    hd = wp.float64(0.0)
    ha = wp.float64(0.0)
    hb = wp.float64(0.0)
    if kind == OPEN:
        inside = a
        if a < 0:
            inside = b
        hd = h[w, inside]
        ha = hd
        hb = hd
        tau = wp.dot(mom[w, inside] * wp.float32(inv_dt), d) / fl.rho
        added = inflow(acc, inv_dt, src_fixed, w, inside)
    else:
        ha = h[w, a]
        hb = h[w, b]
        contact = wet[faces.chart[f]]
        wa = not dry(cells, h, w, a, contact[0])
        if contact[1] >= 0.0 and wa == dry(cells, h, w, b, contact[0]):
            # a contact line: it advances only where the edge is steeper than θ (no hysteresis in phase 1)
            s = (wp.float32(ha - hb) + base[w, a] - base[w, b]) / faces.dist[f]
            wet_cell = a
            if not wa:
                s = -s
                wet_cell = b
            axis = face_slot(cells, wet_cell, f) // 2
            edge = contact[1] * front_weights(cells, faces, h, w, wet_cell, contact[0])[axis]
            drive = wp.max(s - edge, 0.0) * fl.sigma / (fl.rho * faces.dist[f] * faces.dist[f])
            if wa:
                G = G + drive
            else:
                G = G - drive
        else:
            G = G - (pot[w, b] - pot[w, a]) / faces.dist[f]
        hd = hb
        if G > 0.0:
            hd = ha
        tau = 0.5 * (wp.dot(mom[w, a] * wp.float32(inv_dt), faces.dira[f])
                     + wp.dot(mom[w, b] * wp.float32(inv_dt), faces.dirb[f])) / fl.rho   # pending momentum [Pa]
        added = wp.float64(0.5) * (inflow(acc, inv_dt, src_fixed, w, a) + inflow(acc, inv_dt, src_fixed, w, b))
    gface[w, f] = G
    if hd <= fl.h_dry and fl.beta > 0.0 and kind == INTERIOR:
        hd = wp.max(ha, hb)              # a front face keeps its momentum when G points back (uphill run-up, WP2 F1)
    if hd <= fl.h_dry:
        u_new[w, f] = 0.0
        q_new[w, f] = 0.0
        return
    hf = wp.float32(hd)
    phi = float(1.0)
    rheo = face_rheology(fl, w, a, b)
    tau_y = rheo[0]
    nu = rheo[1]
    if tau_y > 0.0:
        gt = transverse(g_old, faces, w, f)
        mag = wp.sqrt(G * G + gt * gt)
        y = hf - tau_y / (fl.rho * wp.max(mag, 1.0e-20))
        if y <= 0.0:
            u_new[w, f] = 0.0                          # the plug stands: friction is infinite
            q_new[w, f] = 0.0
            return
        phi = y * y * (3.0 * hf - y) / (2.0 * hf * hf * hf)
    adv = float(0.0)
    if fl.beta > 0.0 and kind == INTERIOR:
        adv = advection(u, q, h, faces, fl, w, f)
    # under a tool the film is sheared between two walls: friction 12ν/h² (not 3ν/h²) toward the Couette mean U/2
    chi = float(0.0)
    slide = float(0.0)
    gap = float(0.0)
    if tf.on != 0:
        if kind == OPEN:
            ci = inside_cell(faces, f)
            chi = float(0.0)
            if tf.cover[w, ci] > 0.5 and touch(tf, h, base, w, ci) > 0.0:
                chi = 1.0
            slide = wp.dot(tf.vel[w, ci], d)
            gap = tf.ceil[w, ci] - base[w, ci]
        else:
            chi = face_touch(tf, h, base, w, a, b)
            slide = 0.5 * (wp.dot(tf.vel[w, a], faces.dira[f]) + wp.dot(tf.vel[w, b], faces.dirb[f]))
            gap = wp.min(tf.ceil[w, a] - base[w, a], tf.ceil[w, b] - base[w, b])
    visc = nu / (hf * hf * phi)
    friction = (3.0 + 9.0 * chi) * visc + wp.float32(added) / hf
    un = (u[w, f] / dt + G - adv + tau / hf + 6.0 * chi * visc * slide) / (1.0 / dt + friction)
    un = wp.clamp(un, -fl.u_cap, fl.u_cap)
    if kind == OPEN:                                   # film only leaves through an open edge
        if b < 0:
            un = wp.max(un, 0.0)
        else:
            un = wp.min(un, 0.0)
    donor = hb                                         # the mass flux takes its depth from upstream of u
    if un > 0.0:
        donor = ha
    if chi > 0.0:                                      # under a tool the film fills the gap up to its edge
        donor = wp.max(donor, wp.float64(chi * wp.max(gap, 0.0)))
    u_new[w, f] = un
    q_new[w, f] = 0.0
    if donor > fl.h_dry:
        q_new[w, f] = un * wp.float32(donor)


__all__ = ["Cells", "Faces", "Fluid", "Frame", "INTERIOR", "OPEN", "TOUCH", "ToolField", "WALL", "body_force",
           "face_kernel", "face_rheology", "face_slot", "face_touch", "frames_kernel", "front_weights", "inflow", "inside_cell",
           "neighbour", "potential_kernel", "touch"]
