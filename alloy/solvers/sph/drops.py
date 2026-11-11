# SPDX-License-Identifier: Apache-2.0
"""Surface-tension laws of SolverDFSPH (SCR-DC5-5, SCR-WP9-7): Laplace pressure and Rayleigh oscillation of a free
drop, resting-drop jitter, and the sessile-drop contact angle. Zero gravity; the liquid is a ``Liquid`` (water-like
defaults: r = 0.1 mm, sigma = 0.072 N/m; model "csf", csf.py, or "akinci2013", tension.py)."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from .build import add_fluid_points, wall_box
from .config import DFSPHConfig, register_custom_attributes


@dataclass(frozen=True)
class Liquid:
    mode: str = "csf"  # Config.surface_tension
    sigma: float = 0.072  # [N/m]
    rho0: float = 1000.0  # [kg/m^3]
    mu: float = 1.0e-3  # [Pa s]
    viscosity: str = "xsph"  # Config.viscosity
    r: float = 1.0e-4  # particle radius [m]
    tol: float | None = None  # density and divergence tolerance (None: the config default)


WATER = Liquid()
VISCOUS = Liquid(mu=0.05, viscosity="weiler2018")  # glycerol-water-like: the l = 2 mode decays within ~2 periods
AKINCI_WATER = Liquid(mode="akinci2013")


def drop_points(radius_cells: float, spacing: float, hemisphere: bool = False,
                cap_angle: float = 90.0) -> np.ndarray:
    """Lattice points within ``radius_cells`` spacings of the origin; with ``hemisphere`` a spherical cap of the
    hemisphere's volume and contact angle ``cap_angle`` [deg] on a plate at z = 0 (layers from z = r)."""
    if not hemisphere:
        n = int(math.ceil(radius_cells)) + 1
        ax = np.arange(-n, n + 1) * spacing
        pts = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
        return pts[np.linalg.norm(pts, axis=1) <= radius_cells * spacing + 1.0e-12]
    rs, zc = _cap(radius_cells * spacing, cap_angle)
    n = int(math.ceil(rs / spacing)) + 1
    ax = np.arange(-n, n + 1) * spacing
    az = (np.arange(0, 2 * n + 1) + 0.5) * spacing
    pts = np.stack(np.meshgrid(ax, ax, az, indexing="ij"), -1).reshape(-1, 3)
    return pts[np.linalg.norm(pts - np.array([0.0, 0.0, zc]), axis=1) <= rs + 1.0e-12]


def _cap(radius: float, angle: float) -> tuple[float, float]:
    """(sphere radius, centre height) of the cap with contact angle ``angle`` [deg] and a hemisphere's volume."""
    c = math.cos(math.radians(angle))
    rs = radius * (2.0 / (2.0 - 3.0 * c + c**3)) ** (1.0 / 3.0)
    return rs, -rs * c


def _solver(points: np.ndarray, liquid: Liquid, adhesion: float = 0.0, plate_half: float | None = None,
            contact_angle: float | None = None, device=None, ambient_pressure: float | None = None):
    """Model and solver: the points as liquid, and with ``plate_half`` a static plate (top face z = 0, half extent
    plate_half + 8 spacings) whose ``sph:shape_contact_angle`` is ``contact_angle`` [deg] (None: free);
    ``ambient_pressure`` set: the plate is a ``sph:shape_seal`` shape (ambient.py)."""
    from .solver import SolverDFSPH

    cfg = DFSPHConfig(particle_radius=liquid.r, surface_tension=liquid.mode, viscosity=liquid.viscosity,
                      ambient_pressure=ambient_pressure)
    if liquid.tol is not None:
        cfg = dataclasses.replace(cfg, max_density_error=liquid.tol, max_divergence_error=liquid.tol)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    register_custom_attributes(b)
    b.begin_world()
    if plate_half is not None:
        half = plate_half + 8.0 * cfg.spacing
        angle = None if contact_angle is None else math.radians(contact_angle)
        wall_box(b, (0.0, 0.0, -cfg.h), (half, half, cfg.h), contact_angle=angle, seal=ambient_pressure is not None)
    add_fluid_points(b, cfg, points, liquid.rho0, liquid.mu, surface_tension=liquid.sigma, adhesion=adhesion)
    b.end_world()
    m = b.finalize(device=device)
    return m, SolverDFSPH(m, cfg)


def _settle(s, a, c, steps: int, dt: float, damping: float = 0.9):
    """Damped steps (velocities x ``damping`` after each) so the lattice relaxes into the drop's rest shape."""
    for _ in range(steps):
        s.step(a, c, None, None, dt)
        a, c = c, a
        a.particle_qd.assign(damping * a.particle_qd.numpy())
    return a, c


def _radius(m, rho0: float) -> float:
    return (3.0 * m.particle_count * float(m.particle_mass.numpy()[0]) / rho0 / (4.0 * math.pi)) ** (1.0 / 3.0)


def l2_moment(q: np.ndarray) -> float:
    """Axisymmetric l = 2 shape moment <z^2> - (<x^2> + <y^2>) / 2 about the centroid [m^2]."""
    x = q - q.mean(axis=0)
    return float(np.mean(x[:, 2] ** 2) - 0.5 * np.mean(x[:, 0] ** 2 + x[:, 1] ** 2))


def fit_damped(t: np.ndarray, y: np.ndarray, omega_guess: float) -> tuple[float, float]:
    """(omega_d, lambda) of y = A exp(-lambda t) cos(omega_d t + phi) + c by least squares."""
    from scipy.optimize import curve_fit

    def f(t, A, lam, w, ph, c):
        return A * np.exp(-lam * t) * np.cos(w * t + ph) + c

    best = None
    for wg in (0.7 * omega_guess, omega_guess, 1.3 * omega_guess):
        try:
            p, _ = curve_fit(f, t, y, p0=[np.ptp(y) / 2, 50.0, wg, 0.0, float(np.mean(y))], maxfev=20000)
        except RuntimeError:
            continue
        res = float(np.sum((f(t, *p) - y) ** 2))
        if best is None or res < best[1]:
            best = (p, res)
    if best is None:
        raise RuntimeError("fit_damped: no fit converged")
    return abs(float(best[0][2])), float(best[0][1])


def rayleigh_drop(radius_cells: float = 14.0, liquid: Liquid = WATER, eps: float = 0.05, v0: float | None = None,
                  settle: int = 800, periods: float = 3.0, device=None) -> dict:
    """Free drop: settle, then the axisymmetric l = 2 velocity field v0 (-x, -y, 2z) / R, v0 = eps omega_2 R (a
    deformation of about eps at every size) unless given. The l = 2 moment over the first 3 periods is fitted as a
    damped oscillation (omega_d, lambda); ratio_0 = sqrt(omega_d^2 + lambda^2) / omega_2 compares the undamped
    frequency with Rayleigh's omega_2 = sqrt(8 sigma / (rho0 R^3)), R from the liquid volume. ``amp`` is the l = 2
    amplitude per period over ``periods`` (in <r^2> units) and ``offset`` the mean moment over the last 2 (a
    permanent set shows as an offset, wp9-results.md)."""
    m, s = _solver(drop_points(radius_cells, 2.0 * liquid.r), liquid, device=device)
    R = _radius(m, liquid.rho0)
    w_th = math.sqrt(8.0 * liquid.sigma / (liquid.rho0 * R**3))
    dt = s.config.dt_max  # the capillary step limit
    a, c = m.state(), m.state()
    a, c = _settle(s, a, c, settle, dt)
    q0 = a.particle_q.numpy()
    x = q0 - q0.mean(axis=0)
    r2 = float((x**2).sum(axis=1).mean())
    v0 = eps * w_th * R if v0 is None else v0
    a.particle_qd.assign((v0 * np.stack([-x[:, 0], -x[:, 1], 2.0 * x[:, 2]], 1) / R).astype(np.float32))
    t, y = [], []
    for k in range(int(periods * 2.0 * math.pi / w_th / dt) + 1):
        s.step(a, c, None, None, dt)
        a, c = c, a
        if k % 2 == 0:
            t.append((k + 1) * dt)
            y.append(l2_moment(a.particle_q.numpy().astype(np.float64)))
    t, y, period = np.array(t), np.array(y), 2.0 * math.pi / w_th
    early = t < 3.0 * period
    w_d, lam = fit_damped(t[early], y[early], w_th)
    amp = [float(np.ptp(y[(t >= k * period) & (t < (k + 1) * period)]) / 2.0 / r2) for k in range(int(periods))]
    return {"N": m.particle_count, "R": R, "omega_rayleigh": w_th, "omega_d": w_d, "lambda": lam,
            "ratio_d": w_d / w_th, "ratio_0": math.hypot(w_d, lam) / w_th, "dt": dt, "amp": amp,
            "offset": float(y[t > (periods - 2.0) * period].mean() / r2), "q_settled": l2_moment(q0.astype(np.float64)),
            "q_end": float(y[-1]), "q_peak": float(np.abs(y).max())}


def laplace_drop(radius_cells: float = 10.0, liquid: Liquid = WATER, settle: int = 600, measure: int = 150,
                 device=None) -> dict:
    """Interior pressure of a settled free drop (particles deeper than 2h, mean over ``measure`` damped steps)
    against Laplace's 2 sigma / R; p = rho0 (k_density + k_divergence) of the DFSPH solves."""
    m, s = _solver(drop_points(radius_cells, 2.0 * liquid.r), liquid, device=device)
    n, R, dt = m.particle_count, _radius(m, liquid.rho0), s.config.dt_max
    a, c = m.state(), m.state()
    a, c = _settle(s, a, c, settle, dt)
    ps = []
    for _ in range(measure):
        a, c = _settle(s, a, c, 1, dt)
        x = a.particle_q.numpy().astype(np.float64)
        inner = np.linalg.norm(x - x.mean(axis=0), axis=1) < R - 2.0 * s.config.h
        k = s._kd.numpy()[:n] / dt**2 + s._kv.numpy()[:n] / dt
        ps.append(float(liquid.rho0 * k[inner].mean()))
    lap = 2.0 * liquid.sigma / R
    return {"N": n, "R": R, "R_h": R / s.config.h, "p_in": float(np.mean(ps)), "p_std": float(np.std(ps)),
            "laplace": lap, "ratio": float(np.mean(ps)) / lap}


def resting_jitter(radius_cells: float = 8.0, liquid: Liquid = WATER, settle: int = 400, steps: int = 300,
                   device=None) -> dict:
    """Mean and p99 particle speed of a settled free drop over ``steps`` undamped steps, and the capillary speed
    sqrt(sigma / (rho R)) they compare to."""
    m, s = _solver(drop_points(radius_cells, 2.0 * liquid.r), liquid, device=device)
    dt = s.config.dt_max
    a, c = m.state(), m.state()
    a, c = _settle(s, a, c, settle, dt)
    for _ in range(steps):
        s.step(a, c, None, None, dt)
        a, c = c, a
    v = np.linalg.norm(a.particle_qd.numpy(), axis=1)
    return {"mean": float(v.mean()), "p99": float(np.percentile(v, 99)),
            "capillary": math.sqrt(liquid.sigma / (liquid.rho0 * radius_cells * 2.0 * liquid.r))}


def cap_fit_angle(q: np.ndarray, surf: np.ndarray, r: float, z_min: float) -> float:
    """Contact angle [deg] of the sphere fitted (linear least squares) through the surface particles above
    ``z_min``, its radius grown by r to the liquid's edge, cut by the plate z = 0 (through the liquid)."""
    x = q[(surf != 0) & (q[:, 2] > z_min)]
    if len(x) < 10:
        return float("nan")
    sol, *_ = np.linalg.lstsq(np.c_[2.0 * x, np.ones(len(x))], (x * x).sum(axis=1), rcond=None)
    rs = math.sqrt(sol[3] + sol[:3] @ sol[:3]) + r
    return math.degrees(math.acos(float(np.clip(-sol[2] / rs, -1.0, 1.0))))


def height_angle(q: np.ndarray, volume: float, r: float) -> float:
    """Contact angle [deg] of the spherical cap with the liquid's volume V and apex height H (top particle centre +
    r): R_cap = (3V / (pi H^2) + H) / 3, cos theta = 1 - H / R_cap."""
    H = float(np.percentile(q[:, 2], 99.9)) + r
    r_cap = (3.0 * volume / (math.pi * H * H) + H) / 3.0
    return math.degrees(math.acos(float(np.clip(1.0 - H / r_cap, -1.0, 1.0))))


def sessile_drop(radius_cells: float = 10.0, liquid: Liquid = WATER, contact_angle: float | None = 90.0,
                 adhesion: float = 0.0, start_angle: float = 90.0, settle: int = 4000, damping: float = 0.98,
                 chunk: int = 1000, device=None, ambient_pressure: float | None = None) -> dict:
    """A spherical cap (contact angle ``start_angle`` [deg], the volume of a ``radius_cells`` hemisphere) on a flat
    static plate (z = 0) with ``sph:shape_contact_angle`` = ``contact_angle`` [deg] (csf), lightly damped to rest.
    Two measures every ``chunk`` steps: ``theta`` (height_angle) and ``theta_fit`` (cap_fit_angle over the surface
    particles above 2h; csf only). Target: csf ``contact_angle``; akinci2013 Young's acos(adhesion - 1).
    ``ambient_pressure`` set: the plate is sealed (SCR-SC10-1), as a tip that holds a pendant drop."""
    sp = 2.0 * liquid.r
    target = contact_angle if liquid.mode == "csf" else math.degrees(math.acos(float(np.clip(adhesion - 1.0, -1, 1))))
    rs, _ = _cap(radius_cells * sp, target)
    base = rs * math.sin(math.radians(target)) if target < 90.0 else rs
    pts = drop_points(radius_cells, sp, hemisphere=True, cap_angle=start_angle)
    m, s = _solver(pts, liquid, adhesion=adhesion, plate_half=max(float(np.abs(pts[:, :2]).max()), base),
                   contact_angle=contact_angle, device=device, ambient_pressure=ambient_pressure)
    n = m.particle_count
    V = n * float(m.particle_mass.numpy()[0]) / liquid.rho0
    a, c = m.state(), m.state()
    series = []
    for _ in range(max(settle // chunk, 1)):
        a, c = _settle(s, a, c, chunk, s.config.dt_max, damping)
        q = a.particle_q.numpy().astype(np.float64)
        surf = s._tension.surf.numpy()[:n] if hasattr(s._tension, "surf") else None
        fit = cap_fit_angle(q, surf, liquid.r, 2.0 * s.config.h) if surf is not None else float("nan")
        series.append((height_angle(q, V, liquid.r), fit))
    v = np.linalg.norm(a.particle_qd.numpy(), axis=1)
    return {"theta": series[-1][0], "theta_fit": series[-1][1], "target": target, "series": series, "N": n,
            "on_plate": bool(q[:, 2].min() < sp), "lost": int(np.sum(q[:, 2] < -liquid.r)), "v_max": float(v.max())}


# ---------------------------------------------------------------------------------------- pendant drop (SCR-SC10-2)
C18_WATER = Liquid(r=0.075e-3)  # C18's particle size: a 20 uL drop has R/h ~ 5.6
C18_VISCOUS = Liquid(r=0.075e-3, mu=0.05, viscosity="weiler2018")  # csf's validated viscous range at C18's size
# C18's liquid (alloy-task): dyed 80 % glycerol-water, ~60 mPa s; rho and sigma of 80 wt% glycerol at 20 C (assumed)
C18_GLYCEROL = Liquid(r=0.075e-3, rho0=1208.0, sigma=0.065, mu=0.06, viscosity="weiler2018")


def tate_volume(tip_radius: float, sigma: float = 0.072, rho: float = 1000.0, g: float = 9.81) -> float:
    """Tate's drop volume with the Harkins-Brown correction [m^3]: V = r sigma / (rho g F(r / V^(1/3))), F the
    Lando-Oakley 1967 fit of Harkins and Brown's table, 0.14782 + 0.27896 x - 0.166 x^2 (0.3 <= x <= 1.2; used a
    little below it for C18's tip, x ~ 0.25), solved by fixed point."""
    v = 2.0 * math.pi * tip_radius * sigma * 0.6 / (rho * g)
    for _ in range(50):
        x = tip_radius / v ** (1.0 / 3.0)
        v = tip_radius * sigma / (rho * g * (0.14782 + 0.27896 * x - 0.166 * x * x))
    return v


def pendant_drop(tip_radius: float, volume: float, liquid: Liquid = C18_VISCOUS, contact_angle: float = 100.0,
                 ambient_pressure: float | None = None, duration: float = 0.5,
                 settle: float = 0.05, damping: float = 0.98, dt: float = 0.02 / 512, device=None) -> dict:
    """A drop of ``volume`` [m^3] hanging under the flat end face (z = 0) of a vertical round rod of ``tip_radius``
    (analytic cylinder, ``contact_angle`` [deg]; with ``ambient_pressure`` a ``sph:shape_seal`` shape, so the face
    may pull on the drop, ambient.py), gravity on: a spherical cap whose contact line sits on the face's edge, relaxed
    for ``settle`` (velocities x ``damping`` per step, gravity on: the lattice cap's first steps kick at m/s) and then
    run undamped for ``duration``. ``held``: at the end at least 90 % of the liquid is within 3 mm below the cap
    (a detached drop has fallen ~0.4 m/s x 0.2 s further). Measured (SCR-SC10-2, wp9-results.md): C18_VISCOUS holds
    0.85 x tate_volume and loses 1.0 and 1.15 x at tip radii 0.75 and 1.25 mm, sealed or not, and C18_GLYCEROL
    0.85 x of its own; C18_WATER falls at 0.85 x within 0.2-0.3 s (csf's low-viscosity energy injection, csf.py)."""
    from .solver import SolverDFSPH

    sp = 2.0 * liquid.r
    cfg = DFSPHConfig(particle_radius=liquid.r, surface_tension=liquid.mode, viscosity=liquid.viscosity,
                      ambient_pressure=ambient_pressure, xsph_c=0.3,
                      world_bounds=((-0.03, -0.03, -0.1), (0.03, 0.03, 0.03)))
    # cap below z = 0 with base radius a = tip_radius and volume V: V = pi h (3 a^2 + h^2) / 6, solve for h
    a = tip_radius
    roots = np.roots([1.0, 0.0, 3.0 * a * a, -6.0 * volume / math.pi])  # h^3 + 3 a^2 h = 6 V / pi: one real root
    h = float(max(r_.real for r_ in roots if abs(r_.imag) < 1.0e-12 * a))
    rs = (a * a + h * h) / (2.0 * h)
    zc = -(h - rs)  # sphere centre (z < 0 when the cap is more than a hemisphere)
    n = int(math.ceil(max(rs, a) / sp)) + 2
    ax = (np.arange(-n, n + 1) + 0.5) * sp
    g = np.stack(np.meshgrid(ax, ax, -(np.arange(2 * n + 2) + 0.5) * sp, indexing="ij"), -1).reshape(-1, 3)
    keep = (np.linalg.norm(g - np.array([0.0, 0.0, zc]), axis=1) <= rs) & (g[:, 2] > -h)
    pts = g[keep]
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    register_custom_attributes(b)
    b.begin_world()
    L = 8.0e-3
    attrs = {"sph:shape_boundary": 1, "sph:shape_contact_angle": math.radians(contact_angle),
             "sph:shape_seal": int(ambient_pressure is not None)}
    b.add_shape_cylinder(-1, xform=wp.transform((0.0, 0.0, 0.5 * L), wp.quat_identity()), radius=tip_radius,
                         half_height=0.5 * L, label="tip", custom_attributes=attrs)
    add_fluid_points(b, cfg, pts, liquid.rho0, liquid.mu, surface_tension=liquid.sigma)
    b.end_world()
    m = b.finalize(device=device)
    s = SolverDFSPH(m, cfg)
    st, st2 = _settle(s, m.state(), m.state(), int(round(settle / dt)), dt, damping)
    for _ in range(int(round(duration / dt))):
        s.step(st, st2, None, None, dt)
        st, st2 = st2, st
    q = st.particle_q.numpy()
    act = (m.particle_flags.numpy() & 1) != 0
    near = act & (q[:, 2] > -3.0e-3 - h)
    V = len(pts) * float(m.particle_mass.numpy()[0]) / liquid.rho0
    return {"held": bool(near.mean() >= 0.9), "near_fraction": float(near.mean()), "N": len(pts), "volume": V,
            "cap_height": h, "lowest": float(q[act, 2].min()) if act.any() else float("nan"),
            "v_max": float(np.abs(st.particle_qd.numpy()).max())}
