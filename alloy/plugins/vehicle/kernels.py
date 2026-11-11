# SPDX-License-Identifier: Apache-2.0
"""Tyre contact, drivetrain and wheel-spin kernels of the vehicle plug-in (one thread per world).

Contact. A wheel of radius R at point ``center`` of its body (the chassis, or e.g. an oscillating steer axle) spins
about that body's y axis turned by its steering angle. Its lowest point sits ``d`` below a flat ground at height
``ground`` (``d`` > 0 in contact) and carries ``F_z = k·d + c·dd/dt`` (never negative).

The tyre is Pacejka's transient (relaxation-length) tyre (Pacejka, *Tyre and Vehicle Dynamics*, ch. 7): the tread's
deflections ``u`` (along the wheel) and ``q`` (across) are per-wheel states, driven by the contact's slip velocity::

    du/dt = (ω·R - v_x) - |v_x|·u/σ_x        κ' = u/σ_x       F_x = D·sin(C_x·atan(B_x·κ'))
    dq/dt = v_y - |v_x|·q/σ_y                tan α' = q/σ_y   F_y = -D·sin(C_y·atan(B_y·α'))    D = μ·F_z

At speed the steady state is the magic formula at the slips κ = (ω·R - v_x)/|v_x| and tan α = v_y/|v_x|. At
standstill the tread is a spring of stiffness ``D·B·C/σ`` that saturates through the magic formula, so a braked or
loaded vehicle holds instead of creeping, with no slip floor and no switch. Two low-speed rules (WP13 review):
- The deflections are projected back onto ``[min(-peak, s_ss), max(peak, s_ss)]`` with ``s_ss`` the signed
  steady-state slip: past the peak only on the side the contact is sliding now. A wheel that spun or slid at
  standstill, or a skid that stops and rolls back, keeps no more than the peak-slip deflection. A deflection pulled back
  from past the peak to the peak lands on the rising branch at the same force (``C·atan(B·s') = π - C·atan(B·s)``):
  the force stays continuous as a skid stops, and the tread then releases from the sliding force, not the peak.
- A damper ``c_low`` acts on the rate of the deflection projected onto the peak-slip boundary: 0 while the tread
  slides past the peak, so sliding or spinning at standstill adds no viscous force. It fades out smoothly by
  ``v_low``; the force is capped at the friction ellipse ``|(F_x, F_y)| <= D``.

Wheel spin (``update``). ``I·dω/dt = T_drive - R·F_x - T_fric`` is solved together with ``u`` implicitly against the
body's end-of-step velocity. The brake and rolling resistance ``T_fric = brake + c_rr·F_z·R`` are Coulomb torques
(Karnopp): the wheel stays locked while the torque that would turn it is under ``T_fric``, and never reverses. The motor
gives ``T_m = u·T_max·clamp(1 - sign(u)·ω_m/ω_free, 0, 1)`` at ``ω_m = gear·mean(ω_driven)``; an open differential
splits ``gear·T_m`` equally over the driven wheels.
"""

from __future__ import annotations

import warp as wp

from alloy.plugins.funcs import add_wrench, com_world, point_velocity, wrench_at
from alloy.plugins.vehicle.surfaces import Hit, Pair, Surfaces, surface_velocity, wheel_hits
from alloy.plugins.vehicle.wheels import Drive, Tyre, Wheel

HALF_PI = wp.constant(1.5707963267948966)
V_REF = wp.constant(1.0e-4)      # m/s: smallest speed the steady-state slip bound divides by (a locked wheel keeps
                                 # its slip of 1 until it has truly stopped)
SLIP_CAP = wp.constant(1.0e3)    # largest stored slip (a wheel spinning at standstill)


@wp.struct
class Contact:
    point: wp.vec3               # the deepest contact's point, world
    fwd: wp.vec3                 # its tread heading in the surface plane, world
    lat: wp.vec3                 # surface normal x heading
    axis: wp.vec3                # the wheel's axle, world
    fz: wp.float32               # load over the wheel's contacts [N]
    vx: wp.float32               # load-weighted slip velocity along the tread, relative to the surfaces [m/s]
    vy: wp.float32               # and across
    fz_a: wp.float32             # the deepest contact's share of the load
    fz_b: wp.float32             # the second contact's (a valley)
    pair: Pair


@wp.func
def hit_load(h: Hit, axis: wp.vec3, wheel: Wheel, body: wp.int32, body_q: wp.array[wp.transform],
             body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3]) -> wp.vec3:
    """(F_z, v_x, v_y) of one contact: load from the tyre's radial spring and damper, slip velocity of the wheel's
    contact point relative to the surface body, in that contact's tread frame. A trailing arm (``wheel.slope``) moves
    the contact -slope along the body's x per metre of compression, so it slips by that much more."""
    if h.depth <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    v = point_velocity(body_qd, com_world(body_q, body_com, body), body, h.point)
    v = v - surface_velocity(body_q, body_qd, body_com, h.body, h.point)
    vn = wp.dot(v, h.normal)                               # -(compression rate)
    fwd = wp.normalize(wp.cross(axis, h.normal))
    lat = wp.cross(h.normal, fwd)
    fz = h.weight * wp.max(wheel.stiffness * h.depth - wheel.damping * vn, 0.0)
    v = v + wheel.slope * vn * wp.transform_vector(body_q[body], wp.vec3(1.0, 0.0, 0.0))
    return wp.vec3(fz, wp.dot(v, fwd), wp.dot(v, lat))


@wp.func
def steer_angle(wheel: Wheel, steer: wp.float32) -> wp.float32:
    """The wheel's steering angle for the input ``steer``: ``factor·steer``, or with Ackermann (``ack`` = L > 0, the
    distance to the unsteered axle) the angle that points the wheel square to the turning centre on that axle's line,
    ``factor·atan2(L·tan δ, L - y·tan δ)`` (δ = steer, the centreline's angle; y the wheel's lateral position). A
    steer axle's inner wheel then turns further than its outer one, as on a truck's steering linkage."""
    if wheel.ack <= 0.0:
        return wheel.steer * steer
    t = wp.tan(wp.clamp(steer, -1.4, 1.4))
    return wheel.steer * wp.atan2(wheel.ack * t, wheel.ack - wheel.center[1] * t)


@wp.func
def contact(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
            body: wp.int32, wheel: Wheel, steer: wp.float32, surf: Surfaces, w: wp.int32) -> Contact:
    """The wheel's contacts with the drivable surfaces (``fz`` = 0 off them): its load, and the load-weighted slip
    velocity its tread sees (WP13: the deflection state is per wheel, split over its contacts by load)."""
    ct = Contact()
    xf = body_q[body]
    ang = steer_angle(wheel, steer)
    axis = wp.transform_vector(xf, wp.vec3(-wp.sin(ang), wp.cos(ang), 0.0))
    ct.axis = axis
    centre = wp.transform_point(xf, wheel.center)
    pair = wheel_hits(w, centre, axis, wheel.radius, surf, body_q)
    ct.pair = pair
    if pair.a.depth <= 0.0:
        return ct
    la = hit_load(pair.a, axis, wheel, body, body_q, body_qd, body_com)
    lb = hit_load(pair.b, axis, wheel, body, body_q, body_qd, body_com)
    ct.point = pair.a.point
    ct.fwd = wp.normalize(wp.cross(axis, pair.a.normal))
    ct.lat = wp.cross(pair.a.normal, ct.fwd)
    ct.fz_a = la[0]
    ct.fz_b = lb[0]
    ct.fz = la[0] + lb[0]
    if lb[0] <= 0.0:
        ct.vx = la[1]
        ct.vy = la[2]
    elif ct.fz > 0.0:
        ct.vx = (la[0] * la[1] + lb[0] * lb[1]) / ct.fz
        ct.vy = (la[0] * la[2] + lb[0] * lb[2]) / ct.fz
    return ct


@wp.func
def apply_hit(h: Hit, load: wp.float32, f: wp.vec2, ct: Contact, body: wp.int32, body_q: wp.array[wp.transform],
              body_com: wp.array[wp.vec3], body_f: wp.array[wp.spatial_vector]):
    """ADD one contact's share of the tyre force to the wheel's body, and its reaction to the surface's body."""
    if load <= 0.0:
        return
    fwd = wp.normalize(wp.cross(ct.axis, h.normal))
    lat = wp.cross(h.normal, fwd)
    share = load / ct.fz
    force = share * (f[0] * fwd + f[1] * lat) + load * h.normal
    add_wrench(body_f, body, wrench_at(force, h.point, com_world(body_q, body_com, body)))
    if h.body >= 0:
        add_wrench(body_f, h.body, wrench_at(-force, h.point, com_world(body_q, body_com, h.body)))


@wp.func
def steered(mob: wp.array3d[wp.float32], w: wp.int32, i: wp.int32, ang: wp.float32) -> wp.vec2:
    """The wheel's point mobility along and across its steered tread (``fwdᵀ·W·fwd``, ``latᵀ·W·lat``) from the
    in-plane 2x2 ``W`` stored in its body's frame."""
    c = wp.cos(ang)
    s = wp.sin(ang)
    wxx = mob[w, i, 0]
    wxy = mob[w, i, 1]
    wyy = mob[w, i, 2]
    return wp.vec2(wxx * c * c + 2.0 * wxy * c * s + wyy * s * s, wxx * s * s - 2.0 * wxy * c * s + wyy * c * c)


@wp.func
def magic(d: wp.float32, b: wp.float32, c: wp.float32, slip: wp.float32) -> wp.float32:
    return d * wp.sin(c * wp.atan(b * slip))


@wp.func
def magic_slope(d: wp.float32, b: wp.float32, c: wp.float32, slip: wp.float32) -> wp.float32:
    """dF/d(slip) of the magic formula, floored at 0 past the peak."""
    bs = b * slip
    return wp.max(d * c * b * wp.cos(c * wp.atan(bs)) / (1.0 + bs * bs), 0.0)


@wp.func
def peak_slip(b: wp.float32, c: wp.float32) -> wp.float32:
    """The slip where the magic formula peaks: C·atan(B·s) = π/2 (any slip for C <= 1: returns a large bound)."""
    if c <= 1.0:
        return wp.float32(1.0e3)
    return wp.tan(HALF_PI / c) / b


@wp.func
def fade(tyre: Tyre, vx: wp.float32) -> wp.float32:
    """1 at standstill, 0 from ``v_low`` up, smooth (C1) between: the low-speed damper's weight."""
    s = wp.min(wp.abs(vx) / tyre.v_low, 1.0)
    return 0.5 * (1.0 + wp.cos(3.14159265 * s))


@wp.func
def damper(tyre: Tyre, big: wp.float32, b: wp.float32, c: wp.float32, sigma: wp.float32,
           mob: wp.float32) -> wp.float32:
    """``c_low = 2·ζ·sqrt(k/w)``: ζ of critical for the tread spring ``k = D·B·C/σ`` on the body's mobility ``w``."""
    return 2.0 * tyre.zeta * wp.sqrt(big * b * c / sigma / wp.max(mob, 1.0e-9))


@wp.func
def ellipse(fx: wp.float32, fy: wp.float32, big: wp.float32) -> wp.vec2:
    norm = wp.sqrt(fx * fx + fy * fy)
    if norm > big and norm > 0.0:
        return wp.vec2(fx, fy) * (big / norm)
    return wp.vec2(fx, fy)


@wp.func
def tyre_force(ct: Contact, tyre: Tyre, defl: wp.vec2, rate: wp.vec2, mob: wp.vec2) -> wp.vec2:
    """(F_x, F_y) from the tread deflections and their last projected rates, capped at the friction ellipse."""
    big = tyre.mu * ct.fz
    fx = magic(big, tyre.bx, tyre.cx, defl[0] / tyre.sigma_x)
    fy = -magic(big, tyre.by, tyre.cy, wp.atan(defl[1] / tyre.sigma_y))
    w = fade(tyre, ct.vx)
    fx += w * damper(tyre, big, tyre.bx, tyre.cx, tyre.sigma_x, mob[0]) * rate[0]
    fy -= w * damper(tyre, big, tyre.by, tyre.cy, tyre.sigma_y, mob[1]) * rate[1]
    return ellipse(fx, fy, big)


@wp.kernel
def tyre_force_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                      body_com: wp.array[wp.vec3], wheel_body: wp.array2d[wp.int32], wheels: wp.array[Wheel],
                      tyre: Tyre, surf: Surfaces, steer: wp.array[wp.float32], defl: wp.array3d[wp.float32],
                      rate: wp.array3d[wp.float32], mobility: wp.array3d[wp.float32],
                      body_f: wp.array[wp.spatial_vector]):
    """ADD every tyre's force (load, drive/brake, cornering) to its wheel's body at its contacts, and the reaction to
    the surfaces' bodies (one thread per world owns every body in it: no atomics)."""
    w = wp.tid()
    for i in range(wheels.shape[0]):
        b = wheel_body[w, i]
        ct = contact(body_q, body_qd, body_com, b, wheels[i], steer[w], surf, w)
        if ct.fz <= 0.0:
            continue
        f = tyre_force(ct, tyre, wp.vec2(defl[w, i, 0], defl[w, i, 1]), wp.vec2(rate[w, i, 0], rate[w, i, 1]),
                       steered(mobility, w, i, steer_angle(wheels[i], steer[w])))
        apply_hit(ct.pair.a, ct.fz_a, f, ct, b, body_q, body_com, body_f)
        apply_hit(ct.pair.b, ct.fz_b, f, ct, b, body_q, body_com, body_f)


@wp.func
def motor_torque(drive: Drive, throttle: wp.float32, speed: wp.float32) -> wp.float32:
    u = wp.clamp(throttle, -1.0, 1.0)
    if u == 0.0:
        return wp.float32(0.0)
    fade_m = wp.clamp(1.0 - wp.sign(u) * speed / drive.free_speed, 0.0, 1.0)   # reverse torque (plugging) caps at peak
    return u * drive.torque * fade_m


@wp.func
def rising(b: wp.float32, c: wp.float32, slip: wp.float32) -> wp.float32:
    """The slip on the magic formula's rising branch with the same force as ``slip`` past the peak."""
    x = wp.max(3.14159265 - c * wp.atan(b * wp.abs(slip)), 0.0)
    return wp.sign(slip) * wp.tan(x / c) / b


@wp.func
def project(x: wp.float32, sigma: wp.float32, steady: wp.float32, peak: wp.float32, b: wp.float32,
            c: wp.float32) -> wp.float32:
    """Deflection ``x`` with its slip ``x/σ`` kept in ``[min(-peak, steady), max(peak, steady)]``; pulled back from past
    the peak to the peak, it lands on the rising branch at the same force instead."""
    s = x / sigma
    hi = wp.max(peak, steady)
    lo = wp.min(-peak, steady)
    if s >= lo and s <= hi:
        return x
    target = wp.where(s > hi, hi, lo)
    if wp.abs(target) <= peak * 1.000001 and c > 1.0:
        return rising(b, c, s) * sigma
    return target * sigma


@wp.func
def karnopp(inertia: wp.float32, rhs: wp.float32, fric: wp.float32, dt: wp.float32) -> wp.float32:
    """Implicit Coulomb friction on the spin: ``inertia·ω' = rhs - dt·fric·sign``, stuck at 0 while |rhs| <= dt·fric."""
    if wp.abs(rhs) <= dt * fric:
        return wp.float32(0.0)
    return (rhs - wp.sign(rhs) * dt * fric) / inertia


@wp.func
def spin_and_tread(ct: Contact, tyre: Tyre, wheel: Wheel, om: wp.float32, u: wp.float32, q: wp.float32,
                   drive_t: wp.float32, brake: wp.float32, mob_x: wp.float32, dt: wp.float32) -> wp.vec2:
    """(ω', u') solved together, implicitly: the tread's ODE in u and the spin's in ω, coupled through
    F_x ≈ s·F(u) + (s·dF/du + c_low/dt)·(u' - u), with ``s`` the friction-ellipse scale the body sees at (u, q)
    (combined slip), and the brake and rolling resistance as Karnopp friction."""
    big = tyre.mu * ct.fz
    r = wheel.radius
    decay = 1.0 + dt * wp.abs(ct.vx) / tyre.sigma_x
    a = (u - dt * ct.vx) / decay                         # u' = a + b·ω'
    b = dt * r / decay
    slip = u / tyre.sigma_x
    fx = magic(big, tyre.bx, tyre.cx, slip)
    cap = ellipse(fx, -magic(big, tyre.by, tyre.cy, wp.atan(q / tyre.sigma_y)), big)
    scale = wp.where(wp.abs(fx) > 0.0, cap[0] / fx, 1.0)
    k = scale * magic_slope(big, tyre.bx, tyre.cx, slip) / tyre.sigma_x
    if wp.abs(slip) < peak_slip(tyre.bx, tyre.cx):       # the damper acts only below the peak (not while sliding)
        k += fade(tyre, ct.vx) * damper(tyre, big, tyre.bx, tyre.cx, tyre.sigma_x, mob_x) / dt
    inertia = wheel.inertia + dt * r * k * b
    rhs = wheel.inertia * om + dt * (drive_t - r * scale * fx - r * k * (a - u))
    fric = wp.clamp(brake, 0.0, 1.0) * wheel.brake + tyre.rolling * ct.fz * r
    new = karnopp(inertia, rhs, fric, dt)
    return wp.vec2(new, a + b * new)


@wp.func
def lateral(q: wp.float32, tyre: Tyre, steady: wp.float32) -> wp.float32:
    """The lateral deflection projected like ``project``, in tan α' = q/σ_y (the formula takes α')."""
    t = q / tyre.sigma_y
    peak = wp.tan(wp.min(peak_slip(tyre.by, tyre.cy), 1.5))
    hi = wp.max(peak, steady)
    lo = wp.min(-peak, steady)
    if t >= lo and t <= hi:
        return q
    target = wp.where(t > hi, hi, lo)
    if wp.abs(target) <= peak * 1.000001 and tyre.cy > 1.0:
        return wp.tan(rising(tyre.by, tyre.cy, wp.atan(t))) * tyre.sigma_y
    return target * tyre.sigma_y


@wp.kernel
def wheel_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                 wheel_body: wp.array2d[wp.int32], wheels: wp.array[Wheel], tyre: Tyre, surf: Surfaces,
                 drive: Drive, throttle: wp.array[wp.float32], brake: wp.array[wp.float32], steer: wp.array[wp.float32],
                 mobility: wp.array3d[wp.float32], dt: wp.float32, omega: wp.array2d[wp.float32],
                 defl: wp.array3d[wp.float32], rate: wp.array3d[wp.float32], force: wp.array3d[wp.float32],
                 slip: wp.array3d[wp.float32], motor: wp.array2d[wp.float32]):
    """Step every wheel's spin and tread deflections against the end-of-step body velocity; project the deflections
    onto max(peak slip, |steady-state slip|); record the tyre readings."""
    w = wp.tid()
    mean = float(0.0)
    for i in range(wheels.shape[0]):
        if wheels[i].driven != 0:
            mean += omega[w, i]
    speed = drive.gear * mean / wp.float32(drive.driven)
    torque = motor_torque(drive, throttle[w], speed)
    share = drive.gear * torque / wp.float32(drive.driven)
    peak_x = peak_slip(tyre.bx, tyre.cx)
    peak_y = wp.tan(wp.min(peak_slip(tyre.by, tyre.cy), 1.5))
    for i in range(wheels.shape[0]):
        wheel = wheels[i]
        ct = contact(body_q, body_qd, body_com, wheel_body[w, i], wheel, steer[w], surf, w)
        drive_t = wp.where(wheel.driven != 0, share, 0.0)
        u = defl[w, i, 0]
        q = defl[w, i, 1]
        if ct.fz <= 0.0:                                 # off the ground: the tread relaxes, the wheel spins free
            omega[w, i] = karnopp(wheel.inertia, wheel.inertia * omega[w, i] + dt * drive_t,
                                  wp.clamp(brake[w], 0.0, 1.0) * wheel.brake, dt)
            for k in range(2):
                defl[w, i, k] = 0.0
                rate[w, i, k] = 0.0
                slip[w, i, k] = 0.0
            for k in range(3):
                force[w, i, k] = 0.0
            continue
        mob = steered(mobility, w, i, steer_angle(wheel, steer[w]))
        sol = spin_and_tread(ct, tyre, wheel, omega[w, i], u, q, drive_t, brake[w], mob[0], dt)
        vref = wp.max(wp.abs(ct.vx), V_REF)
        steady_x = wp.clamp((sol[0] * wheel.radius - ct.vx) / vref, -SLIP_CAP, SLIP_CAP)
        u_new = project(sol[1], tyre.sigma_x, steady_x, peak_x, tyre.bx, tyre.cx)
        q_new = (q + dt * ct.vy) / (1.0 + dt * wp.abs(ct.vx) / tyre.sigma_y)
        q_new = lateral(q_new, tyre, wp.clamp(ct.vy / vref, -SLIP_CAP, SLIP_CAP))
        omega[w, i] = sol[0]
        defl[w, i, 0] = u_new
        defl[w, i, 1] = q_new
        ex = peak_x * tyre.sigma_x                       # the damper sees the rate inside the peak-slip boundary
        ey = peak_y * tyre.sigma_y
        rate[w, i, 0] = (wp.clamp(u_new, -ex, ex) - wp.clamp(u, -ex, ex)) / dt
        rate[w, i, 1] = (wp.clamp(q_new, -ey, ey) - wp.clamp(q, -ey, ey)) / dt
        f = tyre_force(ct, tyre, wp.vec2(u_new, q_new), wp.vec2(rate[w, i, 0], rate[w, i, 1]), mob)
        force[w, i, 0] = f[0]
        force[w, i, 1] = f[1]
        force[w, i, 2] = ct.fz
        slip[w, i, 0] = u_new / tyre.sigma_x
        slip[w, i, 1] = wp.atan(q_new / tyre.sigma_y)
    motor[w, 0] = speed
    motor[w, 1] = torque


__all__ = ["Contact", "contact", "magic", "motor_torque", "peak_slip", "tyre_force", "tyre_force_kernel",
           "wheel_kernel"]
