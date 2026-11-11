# SPDX-License-Identifier: Apache-2.0
"""Heat v3 kernels: enthalpy finite volumes on a rectilinear grid, the arc's beam and its shadows (``grid`` module).

Cells are world-major ``[W, N]`` with N = nx·ny·nz and z fastest: c = (ix·ny + iy)·nz + iz. ``cell`` is 0 for air and
part + 1 for metal. Faces f = 2·axis + side (side 0: the minus face, 1: the plus face).

Conservation: the conductance of a face is A / (r_minus + r_plus [+ 1/h_c]), each side computing the same sum in the
same order (the minus cell's r = d/2k first), and the flux A·(T_j − T_i) / r into i is the exact negative of the flux
into j. Each cell adds its increment to a float32 enthalpy with a float32 remainder (two-sum), so a slowly cooling
cell does not round away its small steps; the ledger books the arc and every loss per cell in float64.
"""

from __future__ import annotations

import warp as wp

SIGMA_SB = 5.670374419e-8  # W/(m²·K⁴), Stefan-Boltzmann
BIG = 1.0e30
GAUSS_X = wp.constant(0.7745966692414834)  # √(3/5): 3-point Gauss-Legendre on [-1, 1]
GAUSS_W0 = wp.constant(4.0 / 9.0)  # weights normalised to sum 1: 5/18, 8/18, 5/18
GAUSS_W1 = wp.constant(5.0 / 18.0)


@wp.struct
class Grid:
    nx: int
    ny: int
    nz: int
    ex: wp.array(dtype=float)  # cell edges (nx + 1), grid frame [m]
    ey: wp.array(dtype=float)
    ez: wp.array(dtype=float)
    sink: int  # bit f set: the grid's boundary face f conducts to the ambient through h_sink (a table)
    h_air: float  # W/(m²·K), convection on faces open to air
    eps: float  # emissivity of faces open to air
    h_sink: float  # W/(m²·K)
    h_contact: float  # W/(m²·K) between metal cells of different parts until both have reached the solidus
    weld: int  # the cell code of deposited weld metal (no contact resistance: it wets what it lands on); -1: none
    t_solidus: float  # K


@wp.struct
class Table:
    t: wp.array(dtype=float)  # K at the nodes
    h: wp.array(dtype=float)  # J/kg, strictly increasing
    k: wp.array(dtype=float)  # W/(m·K)
    count: int
    h_liquidus: float  # φ = H / H_l
    h_boil: float  # the cap
    c_low: float  # J/(kg·K), extrapolation below the first node


@wp.struct
class Arc:
    p: wp.vec3  # a point on the beam's axis, grid frame [m]
    d: wp.vec3  # beam direction (torch to work), unit
    power: float  # W into the work
    sigma: float  # m, the Gaussian's standard deviation across the beam
    v: wp.vec3  # m/s, the arc point's velocity over the exchange (grid frame; torch body only, else 0)


@wp.func
def lookup(tb: Table, h: float) -> wp.vec3:
    """(T, k, φ) of the specific enthalpy ``h``: linear in H between table nodes (no division by T_l − T_s)."""
    n = tb.count
    temp = float(0.0)
    cond = float(0.0)
    if h <= tb.h[0]:
        temp = tb.t[0] + (h - tb.h[0]) / tb.c_low
        cond = tb.k[0]
    elif h >= tb.h[n - 1]:
        temp = tb.t[n - 1]
        cond = tb.k[n - 1]
    else:
        lo = int(0)
        hi = n - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if tb.h[mid] <= h:
                lo = mid
            else:
                hi = mid
        f = (h - tb.h[lo]) / (tb.h[hi] - tb.h[lo])
        temp = tb.t[lo] + f * (tb.t[hi] - tb.t[lo])
        cond = tb.k[lo] + f * (tb.k[hi] - tb.k[lo])
    phi = float(0.0)
    if tb.h_liquidus > 0.0:
        phi = wp.clamp(h / tb.h_liquidus, 0.0, 1.0)
    return wp.vec3(temp, cond, phi)


@wp.func
def width(g: Grid, axis: int, i: int) -> float:
    if axis == 0:
        return g.ex[i + 1] - g.ex[i]
    if axis == 1:
        return g.ey[i + 1] - g.ey[i]
    return g.ez[i + 1] - g.ez[i]


@wp.func
def extent(g: Grid, axis: int) -> int:
    if axis == 0:
        return g.nx
    if axis == 1:
        return g.ny
    return g.nz


@wp.func
def stride(g: Grid, axis: int) -> int:
    if axis == 0:
        return g.ny * g.nz
    if axis == 1:
        return g.nz
    return 1


@wp.func
def unit(axis: int) -> wp.vec3:
    if axis == 0:
        return wp.vec3(1.0, 0.0, 0.0)
    if axis == 1:
        return wp.vec3(0.0, 1.0, 0.0)
    return wp.vec3(0.0, 0.0, 1.0)


@wp.func
def face_area(dx: float, dy: float, dz: float, axis: int) -> float:
    """The face's area as the product of the other two widths, in a fixed order (both sides get the same bits)."""
    if axis == 0:
        return dy * dz
    if axis == 1:
        return dx * dz
    return dx * dy


@wp.func
def surface_loss(g: Grid, ti: float, ta: float) -> float:
    """W/m² from a face open to air: convection plus grey radiation, at the cell's temperature."""
    return g.h_air * (ti - ta) + g.eps * SIGMA_SB * (ti * ti * ti * ti - ta * ta * ta * ta)


@wp.kernel
def derive(tb: Table, cell: wp.array2d(dtype=wp.int32), mass: wp.array(dtype=float), h: wp.array2d(dtype=float),
           lo: wp.array2d(dtype=float), t: wp.array2d(dtype=float), k: wp.array2d(dtype=float),
           peak: wp.array2d(dtype=float), evap: wp.array2d(dtype=wp.float64)):
    """T and k of every metal cell from its enthalpy; H above the boiling cap is booked as evaporated energy."""
    w, c = wp.tid()
    if cell[w, c] == 0:
        return
    hv = h[w, c] + lo[w, c]
    if hv > tb.h_boil:
        excess = wp.float64(h[w, c]) + wp.float64(lo[w, c]) - wp.float64(tb.h_boil)
        evap[w, c] = evap[w, c] + wp.float64(mass[c]) * excess
        h[w, c] = tb.h_boil
        lo[w, c] = 0.0
        hv = tb.h_boil
    r = lookup(tb, hv)
    t[w, c] = r[0]
    k[w, c] = r[1]
    peak[w, c] = wp.max(peak[w, c], r[0])


@wp.func
def conductance(g: Grid, cell: wp.array2d(dtype=wp.int32), k: wp.array2d(dtype=float),
                peak: wp.array2d(dtype=float), w: int, c: int, c2: int, side: int, di: float, dj: float,
                area: float) -> float:
    """Series conductance [W/K] of the face between metal cells c and c2 (c2 on ``side`` of c): the minus cell's
    half-width resistance first, then the plus cell's, then the contact resistance between unfused parent parts
    (weld metal wets what it lands on: no contact resistance)."""
    ri = 0.5 * di / k[w, c]
    rj = 0.5 * dj / k[w, c2]
    r = ri + rj
    if side == 0:
        r = rj + ri
    if cell[w, c2] != cell[w, c] and cell[w, c] != g.weld and cell[w, c2] != g.weld:
        if peak[w, c] < g.t_solidus or peak[w, c2] < g.t_solidus:
            if g.h_contact <= 0.0:
                return 0.0
            r = r + 1.0 / g.h_contact
    return area / r


@wp.kernel
def flux(g: Grid, cell: wp.array2d(dtype=wp.int32), mass: wp.array(dtype=float), t: wp.array2d(dtype=float),
         k: wp.array2d(dtype=float), peak: wp.array2d(dtype=float), arc_w: wp.array2d(dtype=float),
         ambient: wp.array(dtype=float), dt: float, h: wp.array2d(dtype=float), lo: wp.array2d(dtype=float),
         arc_in: wp.array2d(dtype=wp.float64), lost: wp.array2d(dtype=wp.float64)):
    """One explicit step of one metal cell: conduction through its six faces, losses where they are open, the arc."""
    w, c = wp.tid()
    me = cell[w, c]
    if me == 0:
        return
    iz = c % g.nz
    iy = (c // g.nz) % g.ny
    ix = c // (g.nz * g.ny)
    dx = width(g, 0, ix)
    dy = width(g, 1, iy)
    dz = width(g, 2, iz)
    ti = t[w, c]
    ta = ambient[w]
    q = float(0.0)
    loss = float(0.0)
    for f in range(6):
        axis = f // 2
        side = f % 2
        idx = ix
        if axis == 1:
            idx = iy
        elif axis == 2:
            idx = iz
        nidx = idx + 2 * side - 1
        di = width(g, axis, idx)
        area = face_area(dx, dy, dz, axis)
        if nidx >= 0 and nidx < extent(g, axis):
            c2 = c + (2 * side - 1) * stride(g, axis)
            if cell[w, c2] != 0:
                gface = conductance(g, cell, k, peak, w, c, c2, side, di, width(g, axis, nidx), area)
                q = q + gface * (t[w, c2] - ti)
            else:
                loss = loss + area * surface_loss(g, ti, ta)
        elif (g.sink >> f) & 1 == 1:
            loss = loss + area * (ti - ta) / (0.5 * di / k[w, c] + 1.0 / g.h_sink)
        else:
            loss = loss + area * surface_loss(g, ti, ta)
    dh = dt * (q - loss + arc_w[w, c]) / mass[c]
    hv = h[w, c]
    s = hv + dh                       # two-sum: s + err == hv + dh exactly
    bp = s - hv
    err = (hv - (s - bp)) + (dh - bp)
    rem = lo[w, c] + err
    h2 = s + rem
    lo[w, c] = rem - (h2 - s)
    h[w, c] = h2
    arc_in[w, c] = arc_in[w, c] + wp.float64(arc_w[w, c]) * wp.float64(dt)
    lost[w, c] = lost[w, c] + wp.float64(loss) * wp.float64(dt)


@wp.func
def lit(g: Grid, cell: wp.array2d(dtype=wp.int32), w: int, o: wp.vec3, d: wp.vec3, jx: int, jy: int,
        jz: int) -> bool:
    """True when the ray from ``o`` along ``d``, starting in air cell (jx, jy, jz), leaves the grid without
    meeting metal (a voxel walk, Amanatides and Woo 1987, on the rectilinear edges)."""
    x = jx
    y = jy
    z = jz
    for _step in range(g.nx + g.ny + g.nz + 3):
        if x < 0 or x >= g.nx or y < 0 or y >= g.ny or z < 0 or z >= g.nz:
            return True
        if cell[w, (x * g.ny + y) * g.nz + z] != 0:
            return False
        tx = float(BIG)
        ty = float(BIG)
        tz = float(BIG)
        if d[0] > 0.0:
            tx = (g.ex[x + 1] - o[0]) / d[0]
        elif d[0] < 0.0:
            tx = (g.ex[x] - o[0]) / d[0]
        if d[1] > 0.0:
            ty = (g.ey[y + 1] - o[1]) / d[1]
        elif d[1] < 0.0:
            ty = (g.ey[y] - o[1]) / d[1]
        if d[2] > 0.0:
            tz = (g.ez[z + 1] - o[2]) / d[2]
        elif d[2] < 0.0:
            tz = (g.ez[z] - o[2]) / d[2]
        if tx <= ty and tx <= tz:
            if d[0] > 0.0:
                x = x + 1
            else:
                x = x - 1
        elif ty <= tz:
            if d[1] > 0.0:
                y = y + 1
            else:
                y = y - 1
        else:
            if d[2] > 0.0:
                z = z + 1
            else:
                z = z - 1
    return True


@wp.func
def beam(a: Arc, pt: wp.vec3) -> float:
    """The beam's power per unit area normal to it at ``pt`` [1/m², times power]: a Gaussian of the distance from
    its axis."""
    v = pt - a.p
    perp = v - wp.dot(v, a.d) * a.d
    s2 = a.sigma * a.sigma
    return wp.exp(-wp.dot(perp, perp) / (2.0 * s2)) / (2.0 * wp.pi * s2)


@wp.func
def face_integral(a: Arc, fc: wp.vec3, t1: wp.vec3, t2: wp.vec3, h1: float, h2: float) -> float:
    """∫ beam over the face rectangle centred at ``fc`` (half-widths h1 along t1, h2 along t2) / its area:
    3 x 3 Gauss-Legendre."""
    acc = float(0.0)
    for i in range(3):
        u = float(i - 1) * GAUSS_X
        wu = GAUSS_W0
        if i != 1:
            wu = GAUSS_W1
        for j in range(3):
            v = float(j - 1) * GAUSS_X
            wv = GAUSS_W0
            if j != 1:
                wv = GAUSS_W1
            acc = acc + wu * wv * beam(a, fc + (u * h1) * t1 + (v * h2) * t2)
    return acc


@wp.kernel
def arc_power(g: Grid, cell: wp.array2d(dtype=wp.int32), arcs: wp.array(dtype=Arc), arc_w: wp.array2d(dtype=float)):
    """Arc power [W] into each metal cell for this exchange: the Gaussian beam on every face that is open to air,
    faces the torch and sees it along the beam without metal in between, times the face's cosine to the beam."""
    w, c = wp.tid()
    arc_w[w, c] = 0.0
    a = arcs[w]
    if cell[w, c] == 0 or a.power <= 0.0:
        return
    iz = c % g.nz
    iy = (c // g.nz) % g.ny
    ix = c // (g.nz * g.ny)
    dx = width(g, 0, ix)
    dy = width(g, 1, iy)
    dz = width(g, 2, iz)
    centre = wp.vec3(g.ex[ix] + 0.5 * dx, g.ey[iy] + 0.5 * dy, g.ez[iz] + 0.5 * dz)
    sizes = wp.vec3(dx, dy, dz)
    reach = 5.0 * a.sigma + 0.5 * wp.length(sizes)
    total = float(0.0)
    for f in range(6):
        axis = f // 2
        side = f % 2
        nrm = unit(axis) * float(2 * side - 1)
        cosv = -wp.dot(nrm, a.d)
        if cosv > 0.0:
            idx = ix
            if axis == 1:
                idx = iy
            elif axis == 2:
                idx = iz
            nidx = idx + 2 * side - 1
            inside = nidx >= 0 and nidx < extent(g, axis)
            open_face = True
            if inside:
                open_face = cell[w, c + (2 * side - 1) * stride(g, axis)] == 0
            fc = centre + (0.5 * sizes[axis]) * nrm
            v = fc - a.p
            perp = v - wp.dot(v, a.d) * a.d
            if open_face and wp.length(perp) <= reach:
                seen = True
                if inside:
                    jx = ix
                    jy = iy
                    jz = iz
                    if axis == 0:
                        jx = nidx
                    elif axis == 1:
                        jy = nidx
                    else:
                        jz = nidx
                    seen = lit(g, cell, w, fc, -a.d, jx, jy, jz)
                if seen:
                    a1 = (axis + 1) % 3
                    a2 = (axis + 2) % 3
                    area = face_area(dx, dy, dz, axis)
                    mean = face_integral(a, fc, unit(a1), unit(a2), 0.5 * sizes[a1], 0.5 * sizes[a2])
                    total = total + a.power * cosv * area * mean
    arc_w[w, c] = total


@wp.kernel
def arc_frame(q_in: wp.array(dtype=wp.transform), q_out: wp.array(dtype=wp.transform),
              torch: wp.array(dtype=wp.int32), carrier: wp.array(dtype=wp.int32), tip: wp.vec3, axis: wp.vec3,
              origin: wp.vec3, pose: wp.array2d(dtype=float), power: wp.array(dtype=float),
              sigma: wp.array(dtype=float), feed: wp.array(dtype=float), drop_power: float, dt: float,
              arcs: wp.array(dtype=Arc)):
    """Each world's beam in the grid frame for this exchange: from the torch body, midway between the step's start
    and end poses (second order in the travel per exchange), or from the ``arc_pose`` input (world frame). The beam
    carries the arc's power less what the droplets carry (``drop_power`` W per m/s of wire feed; grid_deposit)."""
    w = wp.tid()
    p = wp.vec3(pose[w, 0], pose[w, 1], pose[w, 2])
    d = wp.vec3(pose[w, 3], pose[w, 4], pose[w, 5])
    v = wp.vec3(0.0, 0.0, 0.0)
    b = torch[w]
    if b >= 0:
        p = 0.5 * (wp.transform_point(q_in[b], tip) + wp.transform_point(q_out[b], tip))
        d = wp.transform_vector(q_in[b], axis) + wp.transform_vector(q_out[b], axis)
        v = (wp.transform_point(q_out[b], tip) - wp.transform_point(q_in[b], tip)) / dt
    cb = carrier[w]
    if cb >= 0:
        inv = wp.transform_inverse(q_out[cb])
        p = wp.transform_point(inv, p)
        d = wp.transform_vector(inv, d)
        v = wp.transform_vector(inv, v)
    out = Arc()
    out.v = v
    out.p = p - origin
    out.d = wp.normalize(d)
    out.power = wp.max(power[w] - drop_power * feed[w], 0.0)
    out.sigma = sigma[w]
    arcs[w] = out
