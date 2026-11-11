# SPDX-License-Identifier: Apache-2.0
"""Law S2 of J2 metal sheets (SCR-WP7-15 P3): a strip dented by a rigid cylinder and released.

A strip of a sourced metal card (E, ν, σ_y0, H as the card says) rests on two fixed frictionless supports (span L). A
kinematic cylinder comes down on mid-span to a depth, holds, lifts away; the strip springs back and settles.

**What is compared.** Statics fixes the moment along a simply supported strip from the measured support reaction R:
M(a) = R a at lever a from a support, whatever the strip's stiffness. Each hinge column's peak curvature is then the
plane-strain section's κ(M) (closed form, monotonic) and its kept curvature κ - M/D after the elastic unloading; the
strip settles to its kept rest angles, so the residual set is their lattice integral Σ κ_r a w (w = l_h = dx, half at
mid). The law checks those hinge by hinge at the end of the hold: the section's moment at each hinge's curvature
against R a (statics), its kept curvature against an elastic unloading from there (κ - M(κ)/D), and the settled set
against the lattice sum. It reports the beam integral's κ(R a) per hinge (``beam_profile``) and the set against the
continuous beam integral at the measured peak (``beam_dent``) as measured values, because two things make the set
comparison a poor discriminator of the yield model:

- 3-point bending is ill-conditioned: the set moves 15-25x as fast as the load and 4-6x as fast as the peak near the
  section's plateau (the plastic zone ends where M = M_y, and κ(M) is nearly flat towards M_p). The mid hinge carries
  the plateau's moment, so its curvature reads within a few % of M only.
- the film's hinge network is more compliant under a concentrated load than the plate: its diagonal hinges and the
  rows' zig-zag (the membrane is scaled 1e-4 for VBD) take a share of the bend that the y hinges do not see; elastic
  3-point bending of 5-20 mm strips deflected 1.20-1.33x the plane-strain beam (a narrow plate: 1.10). That extra is
  elastic (those hinges stay below yield), so it inflates the peak but not the kept set; docs/known_gaps.

**Quasi-static by mass scaling.** At metal stiffness VBD's Gauss-Seidel does not reach a stiff sheet's statics in a
step's iterations (a 1 mm steel strip under gravity sagged 400x its static sag at 20 iterations). The set is a static
quantity, so the law scales the density (``mass_scale``) with gravity off and drives the cylinder slowly against the
strip's lowest mode (period ~0.4 s at 10⁴). Hinge damping stays light while the strip yields (VBD's damping moment is
kd |e| dθ/dt: at kd = 0.05 ke it held the flowing mid hinges 11 % below the statics) and is raised for the free
settle, when nothing yields any more.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.solvers.plastic import closed_forms as CF

from .emit import add_film_mesh
from .rigs import FilmRig, grid_mesh
from .sheet_laws import hinge_angles, metal_spec, section


def curvature(m: float, t: float, Ep: float, syp: float, hp: float) -> float:
    """κ >= 0 with M(κ) = m per unit width (the section's closed form inverted; M rises with κ)."""
    from scipy.optimize import brentq  # noqa: PLC0415

    if m <= 0.0:
        return 0.0
    hi = 12.0 * m / (Ep * t**3)
    while CF.rect_moment(hi, t, Ep, syp, hp) < m:
        hi *= 2.0
        if hi > 1e9:
            raise ValueError(f"moment {m} beyond the section's capacity")
    return brentq(lambda k: CF.rect_moment(k, t, Ep, syp, hp) - m, 0.0, hi, xtol=1e-12 * hi, rtol=1e-12)


def beam_profile(material: str, thickness: float, width: float, reaction: float,
                 levers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(κ, κ_r) at ``levers`` [m] from a support for support reaction ``reaction`` [N]: M = R a / width per unit
    width through the plane-strain section, κ_r = κ - M/D (module doc)."""
    Ep, syp, hp, D = section(metal_spec(material, thickness))
    m = reaction * np.asarray(levers, float) / width
    k = np.array([curvature(mi, thickness, Ep, syp, hp) for mi in m])
    return k, k - m / D


def _deflections(P: float, span: float, width: float, t: float, sec, dx: float | None = None,
                 n: int = 400) -> tuple[float, float]:
    """(peak, set) at mid-span of a simply supported strip under a mid load P [N] (module doc): the integral over
    ``n`` cells of each half, or with ``dx`` the film's hinge lattice (hinges every dx from mid, each bent by κ(M) of
    its own lever over its l_h = dx; the mid hinge counted once)."""
    Ep, syp, hp, D = sec
    if dx is None:
        a, w = (np.arange(n) + 0.5) * (0.5 * span / n), np.full(n, 0.5 * span / n)
    else:
        a, w = _lattice(span, dx)
    m = 0.5 * P * a / width
    k = np.array([curvature(mi, t, Ep, syp, hp) for mi in m])
    return float(np.sum(k * a * w)), float(np.sum((k - m / D) * a * w))


def _lattice(span: float, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Hinge levers from mid outwards and their weights in the half-span sums (the mid hinge's half is dx/2)."""
    a = 0.5 * span - np.arange(int(round(0.5 * span / dx))) * dx
    w = np.full(len(a), dx)
    w[0] = 0.5 * dx
    return a, w


def beam_dent(material: str, thickness: float, span: float, width: float, peak: float,
              dx: float | None = None) -> tuple[float, float]:
    """(P, set): the mid load that gives ``peak`` and the set it leaves (module doc); ``dx``: on the hinge lattice."""
    from scipy.optimize import brentq  # noqa: PLC0415

    sec = section(metal_spec(material, thickness))
    f = lambda p: _deflections(p, span, width, thickness, sec, dx)                 # noqa: E731
    hi = 1.0
    while f(hi)[0] < peak:
        hi *= 2.0
    P = brentq(lambda p: f(p)[0] - peak, 0.0, hi, rtol=1e-9)
    return P, f(P)[1]


@dataclass
class DentRow:
    depth: float          # m, the cylinder's travel past first touch
    peak: float           # m, the strip's peak mid-span deflection (relative to its ends over the supports)
    reaction: float       # N, one support's reaction at the end of the hold (from its contacts)
    set: float            # m, the residual mid-span set after release and settling
    set_rest: float       # m, the lattice integral of the kept rest curvatures
    set_beam: float       # m, the continuous beam integral's set for the measured peak
    lever: np.ndarray     # m, hinge column levers from a support, mid first (both halves averaged)
    k_hold: np.ndarray    # 1/m, their curvature at the end of the hold
    m_hold: np.ndarray    # N m / m, the section's moment at that curvature (closed form)
    k_kept: np.ndarray    # 1/m, their kept rest curvature after settling
    k_beam: np.ndarray    # 1/m, the beam's κ(R a) at the measured reaction
    kr_beam: np.ndarray   # 1/m, and its kept κ - M/D
    max_strain: float     # the largest layer ε̄p
    steps: int
    wall_s: float


@dataclass
class StripRig:
    rig: FilmRig
    v: np.ndarray         # rest vertices (strip centred on the origin, mid-plane at z = 0)
    top: float            # m, the cylinder axis height at first touch
    mid: np.ndarray       # vertex ids of the mid-span column
    ends: np.ndarray      # vertex ids over the supports
    supports: tuple       # shape ids of the two supports
    columns: dict         # lever [m] -> ids of the y hinges at that lever (both halves)
    rest0: np.ndarray     # the authored rest angles
    contact_ke: float


def strip_rig(material: str = "steel_carbon", thickness: float = 1.0e-3, span: float = 0.1, width: float = 0.01,
              dx: float = 5.0e-3, r_press: float = 5.0e-3, dt: float = 1.0e-3, iterations: int = 20,
              mass_scale: float = 1.0e4, bend_damping: float = 1.0e-3, contact_ke: float = 1.0e5,
              press_mu: float = 0.5, extras: dict | None = None, device: str = "cpu") -> StripRig:
    """The S2 strip on two frictionless box supports with the kinematic cylinder (module doc). The strip's
    mid-plane at z0 = 5 mm (float32 keeps a step's motion there); the support tops and the cylinder one contact
    radius off it. ``span / dx`` even (a vertex column under the cylinder), ``width / dx`` whole. The cylinder has
    friction ``press_mu`` (VBD mixes μ as a geometric mean, so the supports' 0 stays 0), which holds the mid-span
    column in x: frictionless all round, the strip drifts along x and falls off a support. ``extras``: FilmExtras
    overrides (e.g. ``{"membrane_scale": 1e-2}``)."""
    spec = metal_spec(material, thickness)
    spec = replace(spec, density=spec.density * mass_scale).with_extras(bend_damping=bend_damping, **(extras or {}))
    n_span, ny = int(round(span / dx)), int(round(width / dx))
    if n_span % 2 or abs(n_span * dx - span) > 1e-9 or ny < 1 or abs(ny * dx - width) > 1e-9:
        raise ValueError(f"span {span} must be an even and width {width} a whole number of cells dx {dx}")
    nx = n_span + 4
    v, tris = grid_mesh(nx, ny, dx)
    v = v - np.array([0.5 * nx * dx, 0.5 * width, 0.0])
    gap, z0 = spec.extras.contact_radius, 0.005
    rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)        # capsule axis along y
    top = z0 + gap + r_press
    hard = newton.ModelBuilder.ShapeConfig(mu=0.0, ke=contact_ke)

    def build(b: newton.ModelBuilder) -> dict:
        p0, _ = add_film_mesh(b, v, tris, spec, pos=(0.0, 0.0, z0))
        sup = [b.add_shape_box(-1, xform=wp.transform(wp.vec3(sx * span, 0.0, z0 - gap - 0.002), wp.quat_identity()),
                               hx=0.25 * dx, hy=width, hz=0.002, cfg=hard) for sx in (-0.5, 0.5)]
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, top), wp.quat_identity()), mass=0.0,
                          is_kinematic=True)
        b.add_shape_capsule(body, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), rot), radius=r_press,
                            half_height=width, cfg=replace(hard, density=0.0, mu=press_mu))
        return {"p0": [p0], "body": [body], "supports": sup}

    rig = FilmRig(build=build, device=device, dt=dt, iterations=iterations, gravity=0.0,
                  vbd={"alloy_initial_guess": "adaptive"})
    m = rig.model
    m.soft_contact_ke, m.soft_contact_kd, m.soft_contact_mu = contact_ke, 0.0, press_mu
    rig.pipe = newton.CollisionPipeline(m, soft_contact_gap=0.002)
    rig.contacts = rig.pipe.contacts()
    tol = 1e-3 * dx
    return StripRig(rig=rig, v=v, top=top, mid=np.flatnonzero(np.abs(v[:, 0]) < tol),
                    ends=np.flatnonzero(np.abs(np.abs(v[:, 0]) - 0.5 * span) < tol),
                    supports=tuple(int(i) for i in rig.names[0]["supports"]),
                    columns=_columns(m.edge_indices.numpy(), v, span, tol), rest0=m.edge_rest_angle.numpy().copy(),
                    contact_ke=contact_ke)


def s2_strip_dent(material: str = "steel_carbon", thickness: float = 1.0e-3, span: float = 0.1,
                  width: float = 0.01, dx: float = 5.0e-3, depth: float = 6.0e-3, r_press: float = 5.0e-3,
                  speed: float = 3.0e-3, hold: float = 0.5, settle: float = 2.0, dt: float = 1.0e-3,
                  iterations: int = 20, mass_scale: float = 1.0e4, bend_damping: float = 1.0e-3,
                  settle_damping: float = 0.05, device: str = "cpu") -> DentRow:
    """S2 (module doc): the cylinder travels ``depth`` down at ``speed``, holds ``hold`` s, lifts the same way,
    leaves, and the strip settles ``settle`` s with the hinge damping raised to ``settle_damping``."""
    t0 = time.perf_counter()
    s = strip_rig(material, thickness, span, width, dx, r_press, dt, iterations, mass_scale, bend_damping,
                  device=device)
    rig, peak, n = s.rig, 0.0, 0
    path = _press_path(depth, speed, dt, hold)
    held = len(path) - int(round(depth / (speed * dt)))                        # the last step of the hold
    for z in path:
        _place(rig, s.top - z)
        rig.step(1)
        n += 1
        peak = max(peak, deflection(s))
        if n == held:
            k_hold, reaction = column_curvatures(s)[0], support_reaction(s)
    _place(rig, s.top + 0.05)                                                   # well away
    set_damping(rig, settle_damping)
    for _ in range(int(round(settle / dt))):
        rig.step(1)
        n += 1
    lever = np.array(sorted(s.columns, reverse=True))
    k_kept = column_curvatures(s)[1]
    k_beam, kr_beam = beam_profile(material, thickness, width, reaction, lever)
    _, w = _lattice(span, dx)
    Ep, syp, hp, _ = section(metal_spec(material, thickness))
    return DentRow(depth=depth, peak=peak, reaction=reaction, set=deflection(s),
                   set_rest=float(np.sum(k_kept * lever * w)),
                   set_beam=beam_dent(material, thickness, span, width, peak)[1], lever=lever, k_hold=k_hold,
                   m_hold=np.array([CF.rect_moment(k, thickness, Ep, syp, hp) for k in k_hold]),
                   k_kept=k_kept, k_beam=k_beam, kr_beam=kr_beam,
                   max_strain=float(rig.solver._j2.ebar_max.numpy().max()), steps=n,
                   wall_s=time.perf_counter() - t0)


def _columns(ei: np.ndarray, v: np.ndarray, span: float, tol: float) -> dict:
    """lever -> the interior y-edge hinges (across the strip) at that lever from a support, between the supports."""
    out: dict = {}
    for e, (o0, o1, a, c) in enumerate(ei):
        if o0 < 0 or o1 < 0 or abs(v[a, 0] - v[c, 0]) > tol or abs(v[a, 0]) > 0.5 * span - tol:
            continue
        lever = round(0.5 * span - abs(float(v[a, 0])), 9)
        out.setdefault(lever, []).append(e)
    return {k: np.asarray(e, np.int64) for k, e in out.items()}


def column_curvatures(s: StripRig) -> tuple[np.ndarray, np.ndarray]:
    """(κ, κ_kept) per hinge column, levers descending (mid first), sag positive: -θ/l_h and -(θ̄ - θ̄0)/l_h."""
    m = s.rig.model
    levers = sorted(s.columns, reverse=True)
    ids = np.concatenate([s.columns[a] for a in levers])
    lh = m.film.hinge_length.numpy()[ids]
    th = hinge_angles(s.rig.q().astype(np.float64), m.edge_indices.numpy()[ids]) / lh
    kept = (m.edge_rest_angle.numpy()[ids] - s.rest0[ids]) / lh
    cut = np.cumsum([len(s.columns[a]) for a in levers])[:-1]
    return (np.array([-g.mean() for g in np.split(th, cut)]), np.array([-g.mean() for g in np.split(kept, cut)]))


def support_reaction(s: StripRig) -> float:
    """One support's mean vertical reaction [N] from its contact penetrations (ke (r - d) n_z, as VBD's penalty), plus
    the share the film's skin guard carried for the particles on it (``SolverFilm.contact_guard_reaction``)."""
    c, m = s.rig.contacts, s.rig.model
    k = int(c.soft_contact_count.numpy()[0])
    sh, pr = c.soft_contact_shape.numpy()[:k], c.soft_contact_particle.numpy()[:k]
    on = np.isin(sh, s.supports)
    if not on.any():
        return 0.0
    q, r = s.rig.q()[pr[on]], m.particle_radius.numpy()[pr[on]]
    n, p = c.soft_contact_normal.numpy()[:k][on], c.soft_contact_body_pos.numpy()[:k][on]   # static: world frame
    pen = np.maximum(r - np.sum(n * (q - p), axis=1), 0.0)
    guard = getattr(s.rig.solver, "contact_guard_reaction", None)
    held = float(guard.numpy()[np.unique(pr[on]), 2].sum()) if guard is not None else 0.0
    return float((np.sum(s.contact_ke * pen * n[:, 2]) + held) / len(s.supports))


def deflection(s: StripRig) -> float:
    """Mid-span deflection below the ends over the supports [m]."""
    q = s.rig.q()
    return float(q[s.ends, 2].mean() - q[s.mid, 2].mean())


def set_damping(rig: FilmRig, beta: float) -> None:
    """Hinge damping kd = beta ke (VBD's damping moment kd |e| dθ/dt)."""
    bp = rig.model.edge_bending_properties.numpy()
    bp[:, 1] = beta * bp[:, 0]
    rig.model.edge_bending_properties.assign(bp)


def _press_path(depth: float, speed: float, dt: float, hold: float) -> np.ndarray:
    """The cylinder's travel per step: down to ``depth`` at ``speed``, hold, back up."""
    down = np.arange(1, int(round(depth / (speed * dt))) + 1) * speed * dt
    return np.concatenate([down, np.full(int(round(hold / dt)), depth), down[::-1]])


def _place(rig: FilmRig, z: float) -> None:
    """The kinematic cylinder's pose (both states, so VBD's contacts see it this step)."""
    b = int(rig.names[0]["body"][0])
    for st in (rig.s0, rig.s1):
        q = st.body_q.numpy()
        q[b] = (0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0)
        st.body_q.assign(q)
