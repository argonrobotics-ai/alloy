# SPDX-License-Identifier: Apache-2.0
"""Heat v2 device kernels (docs/notes/heat-v2-design.md §2, §4): T(H), particle conduction, heaters, the update.

Slots: thermal particle ``s`` of world ``w`` is slot ``s`` (world-major, ``P`` per world). ``gidx[s]`` is its particle
and ``slot_of[i]`` the reverse (-1 for a particle without temperature). Enthalpy, temperature and every heat sum are
float64 (WP13 review: in float32, late equalisation steps fall below H's spacing and diffusion stalls).

Neighbour rows come from the SPH package's sorted, world-grouped ``NeighborLists``, so every sum runs in particle-id
order: a world gives the same bits alone or in a batch, and no float atomics are needed. The energy ledger is per
slot (float64, each slot its own accumulator); only a power heater's contact area is summed across particles, in int64
fixed point (``AREA_FP``), and each particle's share uses the same rounded area, so the shares sum to 1. A particle
that is not ACTIVE (an SPH spare, a retired particle) takes no heat at all: SPH parks spares at one point, so spares
inside a plate would otherwise share its power (WP13 review). Conservation needs complete neighbour rows: a row that
overflows keeps j in i's row while dropping i from j's, and that pair's flux flows one way. A body heater (MODE_BODY)
pays each particle's exchange with a WP13 ``Thermal`` body in whole µJ and adds the opposite into the body's
``external_heat`` (int64 atomics), so the two ledgers match exactly; the sub-µJ remainder carries per particle and
heater.

Areas (they were πr² of the collision radius before step 4, a solver parameter, not the body's surface):
- A heater's contact area per touching particle is its footprint (m/ρ)^(2/3): on a lattice exactly its share of a
  flat face (s²), for SPH particles 4r².
- A particle within its radius plus ``gap`` of a heater box face rests on it with the share of its footprint that
  lies over the face (``face_query``) and that no lower particle covers (``resting_share``): its footprint disc
  (area s²) minus the overlaps, in the heater's plane, with the discs (each its own size) of neighbours at least
  ``LOWER`` (¼) spacing nearer the box. MPM holds particles anywhere within a
  voxel of a collider, so a melt's rim sat 0.7-4.6 spacings up and never touched a fixed 0.7-spacing band (WP13
  re-review R1); with the share, a wide gap no longer counts the particles above the bottom ones. Total touching area
  over the base: lattices and jittered or lifted layers 0.96-1.03, a close-packed bilayer 0.96 (its bottom alone:
  0.92), a random three-layer slab 0.97; ``melt_chocolate``'s puddle 0.86-1.03 over 60 minutes (a yes/no 35° cone
  read the close-packed bilayer 1.75 and the thin puddle up to 1.28).
- Convection acts on each particle's exposed area (``particle_area`` doc; kernel in ``particle_area_kernels``): the
  footprint times its open faces, the smaller of two estimates (from the filled fraction of the neighbourhood moment,
  and from a 62-direction cone view of its neighbours), each calibrated so a flat face reads one footprint per column.
  A face on a heater is not open to the air: a touching particle loses one footprint.
- Heaters and convection act through h_eff = 1/(1/h + d/(2k)): h in series with the half cell between the particle's
  centre (where its T lives) and the surface (cell-centred Robin; WP13 review of step 4: without it the first layer
  took heat up to 1 + h·d/(2k) times too fast, 1.39x in the laws' water bar).
Free-surface correction (an in-house, conservative member of the second-moment family of Schwaiger 2008, IJNME
75:647, and Fatehi and Manzari 2011, Comput. Math. Appl. 61:482, which apply M_i⁻¹ per particle and do not conserve):
- A particle near a free surface misses neighbours and under-conducts, even along the surface (a 7 × 7 particle bar
  diffused 10.5 % slow).
- Each particle's second-moment tensor M_i = −Σ_j V_j F(r) r ⊗ r is the identity in a full neighbourhood. The
  operator's response to a field varying along a unit direction e is e·M_i·e, so every pair's flux is scaled by
  2 / (e·M_i·e + e·M_j·e), with e the pair direction.
- That keeps each pair antisymmetric (energy exact) and corrects conduction direction by direction.
- An isotropic scalar, trace(M)/3, over-corrected the bar by 8 %.
- Each directional moment is floored at ``GAMMA_MIN``, so an isolated pair is boosted at most 2x.
- It corrects only the second moment along each pair. The first moment (Σ V F r ≠ 0 at a surface) and M's
  off-diagonal terms are left alone. H-1 and WP13's lattice probes validate it (bar, slab, sheet, jittered bar).
- What is left (WP13 review): the second layer under each free face conducts about 3.7 % too much along the face.
  The pair factor is symmetric, so boosting a face particle also boosts its partner one layer in, whose own
  neighbourhood was full. The error therefore grows as a body thins: +1.1 % on an 11-particle cross-section, +1.8 %
  on 7, and over 2 % below about 6. Edges and corners add only about 0.03 %; with the side faces removed, −0.03 %.
- Stability: the factor exceeds 1 only where neighbours are missing, so on lattices the operator's largest
  eigenvalue stays within 1.1x the bulk value; check_dt's 0.1 ρ c hs²/k keeps a 3x margin.
"""

from __future__ import annotations

import warp as wp

from alloy._vendor.newton import ParticleFlags

F64 = wp.float64
GAMMA_MIN = wp.constant(0.5)
R_GAS = wp.constant(8.314462618)  # J/(mol·K)
NO_BOIL = 1.0e300  # h_b of a material that does not boil
AREA_FP = wp.constant(1.0e12)  # heater contact area fixed point: 1 µm²
MODE_TEMPERATURE = wp.constant(0)
MODE_POWER = wp.constant(1)
MODE_BODY = wp.constant(2)
AREA_CAP = wp.constant(6.0)  # footprints: a lone particle's exposure (particle_area.CAP)
UJ = wp.constant(1.0e6)  # µJ per J: a body heater's exchange is whole µJ (WP13 Thermal.external_heat units)
ACTIVE = wp.constant(int(ParticleFlags.ACTIVE))
LOWER = wp.constant(0.25)  # spacings: a neighbour this much nearer a heater can hold a particle off it


@wp.struct
class MatTable:
    density: wp.array[wp.float64]
    c_s: wp.array[wp.float64]
    c_l: wp.array[wp.float64]
    k_s: wp.array[wp.float64]
    k_l: wp.array[wp.float64]
    t_s: wp.array[wp.float64]
    t_l: wp.array[wp.float64]
    h_l: wp.array[wp.float64]
    h_b: wp.array[wp.float64]  # enthalpy at the boiling point (NO_BOIL: never)
    l_v: wp.array[wp.float64]  # J/kg latent heat of vaporisation
    kin_a: wp.array[wp.float64]  # 1/s doneness pre-exponential (0: no doneness)
    kin_ea: wp.array[wp.float64]  # J/mol doneness activation energy


@wp.struct
class Heaters:
    shape: wp.array[wp.int32]  # [W * K] global shape id (-1: none)
    mode: wp.array[wp.int32]  # [K] MODE_TEMPERATURE, MODE_POWER or MODE_BODY
    value: wp.array2d[wp.float64]  # [W, K] K (temperature) or W (power): the plug-in input "heater_values"
    area: wp.array[wp.int64]  # [W * K] contact area this step, AREA_FP units
    conductance: wp.float64  # W/(m²·K), particle to a temperature heater
    efficiency: wp.float64  # power heaters
    gap: float  # m beyond the particle radius
    count: int  # K
    body_slot: wp.array[wp.int32]  # [W * K] a body heater's Thermal slot w·B + b (-1: none)
    body_temp: wp.array[wp.float64]  # Thermal.temperature, flat [W * B]
    body_heat: wp.array[wp.int64]  # Thermal.external_heat, flat [W * B] µJ (this kernel adds −Q)
    residual: wp.array[wp.float64]  # [W * P * K] µJ a body exchange has not yet paid, per (slot, heater) (state)


@wp.struct
class Slots:
    gidx: wp.array[wp.int32]  # slot -> particle
    slot_of: wp.array[wp.int32]  # particle -> slot (-1)
    mat: wp.array[wp.int32]  # slot -> MatTable row
    mass: wp.array[wp.float64]  # slot mass [kg]
    power_density: wp.array[wp.float64]  # slot volumetric source [W/kg]
    face: wp.array[wp.float64]  # slot footprint (m/ρ)^(2/3) [m²]: its share of a flat face (heater contact area)
    per_world: int


@wp.struct
class Cloud:
    q: wp.array[wp.vec3]
    mass: wp.array[float]
    radius: wp.array[float]
    flags: wp.array[wp.int32]
    nbr: wp.array2d[wp.int32]
    count: wp.array[wp.int32]
    hs: float  # smoothing length (support / 2)


@wp.struct
class Shapes:
    body: wp.array[wp.int32]
    xf: wp.array[wp.transform]
    scale: wp.array[wp.vec3]
    body_q: wp.array[wp.transform]


@wp.struct
class Field:
    temp: wp.array[wp.float64]
    liquid: wp.array[wp.float64]
    moment: wp.array[wp.mat33d]
    boiled: wp.array[wp.float64]
    exposure: wp.array[wp.float64]  # [m²] exposed area for convection (particle_area doc)


@wp.struct
class Env:
    ambient: wp.array[wp.float64]  # [W] K (input "ambient")
    convection: wp.float64
    dt: wp.float64  # thermal step (dt × time_scale)


# --- functions -----------------------------------------------------------------------------------------------------

@wp.func
def temperature(h: wp.float64, m: int, mt: MatTable) -> wp.vec2d:
    """(T, φ) of specific enthalpy ``h`` for material ``m`` (phase_materials module doc)."""
    ts = mt.t_s[m]
    hl = mt.h_l[m]
    zero = F64(0.0)
    if h < zero:
        return wp.vec2d(ts + h / mt.c_s[m], zero)
    if h <= hl:
        phi = F64(1.0)
        if hl > zero:
            phi = h / hl
        return wp.vec2d(ts + phi * (mt.t_l[m] - ts), phi)
    return wp.vec2d(mt.t_l[m] + (h - hl) / mt.c_l[m], F64(1.0))


@wp.func
def conductivity(phi: wp.float64, m: int, mt: MatTable) -> wp.float64:
    return mt.k_s[m] + phi * (mt.k_l[m] - mt.k_s[m])


@wp.func
def cubic_dwdr(r: float, hs: float) -> float:
    """dW/dr of the 3-D cubic spline with smoothing length hs (support 2 hs)."""
    q = r / hs
    sigma = 1.0 / (wp.pi * hs * hs * hs * hs)
    if q < 1.0:
        return sigma * (-3.0 * q + 2.25 * q * q)
    if q < 2.0:
        return sigma * (-0.75 * (2.0 - q) * (2.0 - q))
    return 0.0


vec5 = wp.types.vector(length=5, dtype=float)


@wp.func
def face_query(x: wp.vec3, xf: wp.transform, half: wp.vec3, side: float) -> vec5:
    """Point x against the box face it lies most outside of (its dominant axis): (signed distance along that axis,
    the face's outward normal, the share of a side × side footprint centred on x that lies over the face). A particle
    past a plate's edge has no footprint over it (WP13 re-review r3: it counted 0.74 before)."""
    p = wp.transform_point(wp.transform_inverse(xf), x)
    d = wp.vec3(wp.abs(p[0]) - half[0], wp.abs(p[1]) - half[1], wp.abs(p[2]) - half[2])
    axis = int(0)
    if d[1] > d[axis]:
        axis = 1
    if d[2] > d[axis]:
        axis = 2
    n = wp.vec3(0.0, 0.0, 0.0)
    n[axis] = wp.sign(p[axis])
    over = float(1.0)
    for k in range(3):
        if k != axis:
            over *= wp.clamp((half[k] - wp.abs(p[k])) / side + 0.5, 0.0, 1.0)
    n = wp.transform_vector(xf, n)
    return vec5(d[axis], n[0], n[1], n[2], over)


@wp.func
def lens_share(d: float, r1: float, r2: float) -> float:
    """The share of a disc of radius r1 that a disc of radius r2 at centre distance d overlaps."""
    if d >= r1 + r2:
        return 0.0
    if d <= wp.abs(r1 - r2):
        return wp.min(r2 * r2 / (r1 * r1), 1.0)
    a = r1 * r1 * wp.acos(wp.clamp((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1), -1.0, 1.0))
    b = r2 * r2 * wp.acos(wp.clamp((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2), -1.0, 1.0))
    c = 0.5 * wp.sqrt(wp.max((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2), 0.0))
    return (a + b - c) / (wp.pi * r1 * r1)


@wp.func
def resting_share(s: int, i: int, n: wp.vec3, sl: Slots, cl: Cloud) -> float:
    """The share of slot s's footprint that rests on the heater: 1 minus the overlaps, in the heater's plane, of its
    footprint disc (area s²) with those of the thermal neighbours at least ``LOWER`` spacings nearer the heater along
    its normal n, each with its own footprint (≥ 0). A particle over a lower one rests on it, not on the heater."""
    sp = float(wp.sqrt(sl.face[s]))
    ri = sp / wp.sqrt(wp.pi)
    xi = cl.q[i]
    covered = float(0.0)
    for k in range(cl.count[i]):
        j = cl.nbr[i, k]
        sj = sl.slot_of[j]
        if sj >= 0 and active(cl, j):
            e = cl.q[j] - xi
            h = wp.dot(e, n)
            if h < -LOWER * sp:
                rj = float(wp.sqrt(sl.face[sj])) / wp.sqrt(wp.pi)
                covered += lens_share(wp.sqrt(wp.max(wp.dot(e, e) - h * h, 0.0)), ri, rj)
    return wp.max(1.0 - covered, 0.0)


@wp.func
def touch_share(s: int, i: int, shape: int, sl: Slots, cl: Cloud, sh: Shapes, gap: float) -> wp.float64:
    """The share of slot s's footprint (particle i) on box shape ``shape``: 0 beyond its radius plus ``gap`` from the
    face, else the part of its footprint over the face times ``resting_share``."""
    xf = sh.xf[shape]
    b = sh.body[shape]
    if b >= 0:
        xf = sh.body_q[b] * xf
    fq = face_query(cl.q[i], xf, sh.scale[shape], float(wp.sqrt(sl.face[s])))
    if fq[0] > cl.radius[i] + gap or fq[4] <= 0.0:
        return F64(0.0)
    return F64(fq[4] * resting_share(s, i, wp.vec3(fq[1], fq[2], fq[3]), sl, cl))


@wp.func
def contact_area(face: wp.float64) -> wp.int64:
    """A particle's heater contact area (its footprint) in AREA_FP units: both heater passes use this rounded value."""
    return wp.int64(face * F64(AREA_FP))


@wp.func
def active(cl: Cloud, i: int) -> bool:
    return (cl.flags[i] & ACTIVE) != 0


@wp.func
def boiled_off(boiled: wp.float64, mass: wp.float64, m: int, mt: MatTable) -> bool:
    """A particle whose boiled energy reached m·L_v has evaporated: it leaves heat exchange."""
    lv = mt.l_v[m]
    return lv > F64(0.0) and boiled >= mass * lv


@wp.func
def takes_heat(s: int, i: int, sl: Slots, boiled: wp.array[wp.float64], cl: Cloud, mt: MatTable) -> bool:
    """One predicate for every pass (WP13 review of step 2): an ACTIVE particle that has not evaporated."""
    return active(cl, i) and not boiled_off(boiled[s], sl.mass[s], sl.mat[s], mt)


@wp.func
def directional(m: wp.mat33d, e: wp.vec3d) -> wp.float64:
    return wp.max(F64(GAMMA_MIN), wp.dot(e, m * e))


@wp.func
def conduction(s: int, sl: Slots, fd: Field, cl: Cloud, mt: MatTable) -> wp.float64:
    """dH/dt [J/(kg·s)] of slot s from Cleary-Monaghan conduction over its neighbour row (free-surface corrected)."""
    i = sl.gidx[s]
    m = sl.mat[s]
    ti = fd.temp[s]
    mi = fd.moment[s]
    ki = conductivity(fd.liquid[s], m, mt)
    rhoi = mt.density[m]
    xi = cl.q[i]
    eta2 = F64(0.01 * cl.hs * cl.hs)
    acc = F64(0.0)
    for n in range(cl.count[i]):
        j = cl.nbr[i, n]
        sj = sl.slot_of[j]
        rij = xi - cl.q[j]
        d2 = F64(wp.dot(rij, rij))
        if sj >= 0 and d2 > F64(0.0):  # coincident particles exchange nothing (no direction)
            mj = sl.mat[sj]
            kj = conductivity(fd.liquid[sj], mj, mt)
            d = wp.sqrt(d2)
            e = wp.vec3d(F64(rij[0]), F64(rij[1]), F64(rij[2])) / d
            corr = directional(mi, e) + directional(fd.moment[sj], e)
            coef = F64(8.0) * F64(cl.mass[j]) * ki * kj / (rhoi * mt.density[mj] * (ki + kj) * corr)
            acc += coef * (ti - fd.temp[sj]) * d * F64(cubic_dwdr(float(d), cl.hs)) / (d2 + eta2)
    return acc


@wp.func
def surface_conductance(h: wp.float64, s: int, sl: Slots, fd: Field, mt: MatTable) -> wp.float64:
    """h_eff = 1 / (1/h + d/(2k)): the coefficient h in series with the half cell between the particle's centre and
    the surface (d = (m/ρ)^(1/3), k at its φ): the cell-centred Robin condition (WP13 review of step 4)."""
    if h <= F64(0.0):
        return F64(0.0)
    k = conductivity(fd.liquid[s], sl.mat[s], mt)
    return F64(1.0) / (F64(1.0) / h + wp.sqrt(sl.face[s]) / (F64(2.0) * k))


@wp.func
def body_exchange(power: wp.float64, idx: int, slot: int, ht: Heaters, dt: wp.float64) -> wp.float64:
    """A body heater's exchange this step in whole µJ; the Thermal body gets exactly the opposite. The sub-µJ remainder
    carries to the next step per (slot, heater) ``idx`` (no dead band, no bias between two heaters)."""
    x = power * dt * F64(UJ) + ht.residual[idx]
    uj = wp.int64(wp.floor(x + F64(0.5)))
    ht.residual[idx] = x - F64(uj)
    wp.atomic_add(ht.body_heat, slot, -uj)
    return F64(uj) / (F64(UJ) * dt)


@wp.func
def on_heater(s: int, w: int, sl: Slots, cl: Cloud, sh: Shapes, ht: Heaters) -> wp.float64:
    """The share of slot s's footprint on its world's heater shapes (at most 1)."""
    i = sl.gidx[s]
    share = F64(0.0)
    for k in range(ht.count):
        shape = ht.shape[w * ht.count + k]
        if shape >= 0:
            share += touch_share(s, i, shape, sl, cl, sh, ht.gap)
    return wp.min(share, F64(1.0))


@wp.func
def heater_power(s: int, w: int, sl: Slots, fd: Field, cl: Cloud, sh: Shapes, ht: Heaters, mt: MatTable,
                 dt: wp.float64) -> wp.float64:
    """Power [W] slot s takes from the heaters it touches."""
    i = sl.gidx[s]
    h_face = surface_conductance(ht.conductance, s, sl, fd, mt) * sl.face[s]  # W/K through a whole footprint
    power = F64(0.0)
    for k in range(ht.count):
        hk = w * ht.count + k
        shape = ht.shape[hk]
        if shape >= 0:
            share = touch_share(s, i, shape, sl, cl, sh, ht.gap)
            if share > F64(0.0):
                area = sl.face[s] * share
                g = h_face * share
                mode = ht.mode[k]
                if mode == MODE_TEMPERATURE:
                    power += g * (ht.value[w, k] - fd.temp[s])
                elif mode == MODE_POWER:
                    total = ht.area[hk]
                    if total > wp.int64(0):
                        power += ht.efficiency * ht.value[w, k] * F64(contact_area(area)) / F64(total)
                elif ht.body_slot[hk] >= 0:
                    b = ht.body_slot[hk]
                    power += body_exchange(g * (ht.body_temp[b] - fd.temp[s]), s * ht.count + k, b, ht, dt)
    return power


# --- kernels -------------------------------------------------------------------------------------------------------

@wp.kernel(enable_backward=False)
def derive(enthalpy: wp.array[wp.float64], mat: wp.array[wp.int32], mt: MatTable, temp: wp.array[wp.float64],
           liquid: wp.array[wp.float64]):
    s = wp.tid()
    tp = temperature(enthalpy[s], mat[s], mt)
    temp[s] = tp[0]
    liquid[s] = tp[1]


@wp.kernel(enable_backward=False)
def keep_max(now: wp.array[wp.int32], worst: wp.array[wp.int32]):
    """worst[w] = max over exchanges of now[w] (a per-exchange count that cannot wrap)."""
    w = wp.tid()
    worst[w] = wp.max(worst[w], now[w])


@wp.kernel(enable_backward=False)
def set_phase(sl: Slots, mt: MatTable, boiled: wp.array[wp.float64], phase: wp.array[wp.int8]):
    """Neighbour-search inclusion from the (restorable) state: an evaporated particle drops out of the rows."""
    s = wp.tid()
    keep = wp.int8(1)
    if boiled_off(boiled[s], sl.mass[s], sl.mat[s], mt):
        keep = wp.int8(-1)
    phase[sl.gidx[s]] = keep


@wp.kernel(enable_backward=False)
def heater_area(sl: Slots, cl: Cloud, sh: Shapes, ht: Heaters, mt: MatTable, boiled: wp.array[wp.float64]):
    """Pass 1: each heater's contact area per world (Σ footprints of the particles that take heat and touch it; int64,
    order-free). Power heaters share their power by it; the others report it."""
    s = wp.tid()
    i = sl.gidx[s]
    if not takes_heat(s, i, sl, boiled, cl, mt):
        return
    w = s / sl.per_world
    for k in range(ht.count):
        shape = ht.shape[w * ht.count + k]
        if shape >= 0:
            share = touch_share(s, i, shape, sl, cl, sh, ht.gap)
            if share > F64(0.0):
                wp.atomic_add(ht.area, w * ht.count + k, contact_area(sl.face[s] * share))


@wp.kernel(enable_backward=False)
def heater_idle(ht: Heaters, dt: wp.float64, idle: wp.array[wp.float64]):
    """A power heater that touches no particle taking heat puts its η·P·dt nowhere: book it as idle [J]."""
    hk = wp.tid()
    k = hk % ht.count
    if ht.mode[k] == MODE_POWER and ht.shape[hk] >= 0 and ht.area[hk] == wp.int64(0):
        idle[hk] = idle[hk] + ht.efficiency * ht.value[hk / ht.count, k] * dt


@wp.kernel(enable_backward=False)
def rate(sl: Slots, fd: Field, cl: Cloud, mt: MatTable, sh: Shapes, ht: Heaters, env: Env,
         out: wp.array[wp.float64], heated: wp.array[wp.float64], cooled: wp.array[wp.float64]):
    """Pass 2: dH/dt [J/(kg·s)] of slot s (conduction, heaters, volumetric source, surface convection); ``heated`` /
    ``cooled`` [J] accumulate its heat gain and ambient loss. Inactive and evaporated particles take nothing."""
    s = wp.tid()
    i = sl.gidx[s]
    m = sl.mat[s]
    if not takes_heat(s, i, sl, fd.boiled, cl, mt):
        out[s] = F64(0.0)
        return
    w = s / sl.per_world
    power = sl.power_density[s] * sl.mass[s] + heater_power(s, w, sl, fd, cl, sh, ht, mt, env.dt)
    loss = F64(0.0)
    if env.convection > F64(0.0):
        area = fd.exposure[s]
        # the share of its footprint on a heater is not open to the air
        area = wp.max(F64(0.0), area - sl.face[s] * on_heater(s, w, sl, cl, sh, ht))
        loss = surface_conductance(env.convection, s, sl, fd, mt) * area * (fd.temp[s] - env.ambient[w])
    out[s] = conduction(s, sl, fd, cl, mt) + (power - loss) / sl.mass[s]
    heated[s] = heated[s] + power * env.dt
    cooled[s] = cooled[s] + loss * env.dt


@wp.kernel(enable_backward=False)
def update(rate_in: wp.array[wp.float64], dt: wp.float64, sl: Slots, mt: MatTable, cl: Cloud,
           enthalpy: wp.array[wp.float64], temp: wp.array[wp.float64], liquid: wp.array[wp.float64],
           boiled: wp.array[wp.float64], done: wp.array[wp.float64]):
    """Pass 3: explicit Euler on H; boiling holds H at H_b and books the excess as boiled energy [J] (vapour fraction
    ψ = boiled / (m L_v)); then T and φ; doneness by the exact exponential step at this T. A slot that takes no heat
    (a parked spare, an evaporated particle) is left alone: it neither boils nor cooks."""
    s = wp.tid()
    if not takes_heat(s, sl.gidx[s], sl, boiled, cl, mt):
        return
    m = sl.mat[s]
    h = enthalpy[s] + dt * rate_in[s]
    hb = mt.h_b[m]
    if h > hb:
        boiled[s] = boiled[s] + (h - hb) * sl.mass[s]
        h = hb
    enthalpy[s] = h
    tp = temperature(h, m, mt)
    temp[s] = tp[0]
    liquid[s] = tp[1]
    a = mt.kin_a[m]
    if a > F64(0.0):
        k = a * wp.exp(-mt.kin_ea[m] / (F64(R_GAS) * tp[0]))
        done[s] = F64(1.0) - (F64(1.0) - done[s]) * wp.exp(-k * dt)
