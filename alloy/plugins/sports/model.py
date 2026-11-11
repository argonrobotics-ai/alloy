# SPDX-License-Identifier: Apache-2.0
"""Football physics references (host NumPy, float64): the ball's contact, the turf, the air.

The plug-ins (``PressureBall``, ``BallAero``) evaluate the same formulas in Warp; the laws check them against these
functions, and these functions against the sourced numbers below. Exact quotes and URLs are in ``README.md``.

**Ball.** IFAB Law 2: circumference 68-70 cm, mass 410-450 g, pressure 0.6-1.1 atm at sea level. FIFA's test
pressure is 0.8 bar. Mid values: R = 0.69 m / 2π, m = 0.43 kg. Inertia: a thin shell, I = ⅔·m·R² (ESTIMATE).

**Contact (pressure shell).** For an overlap δ of the undeformed sphere with a surface, the inflation pressure carries
the load over the flattened patch, and the shell adds its own stiffness:

    F_el(δ) = p_g(δ)·π(2Rδ - δ²) + k_s·δ
    p_g(δ)  = (p_0 + p_atm)·(V_0 / (V_0 - ΔV))^γ - p_atm,     ΔV = πδ²(3R - δ)/3,  V_0 = 4πR³/3,  γ = 1.4

The gas compresses adiabatically (an impact lasts ~10 ms). The shell's loss is a dashpot c_b on the ball's own
deformation, so the force is max(0, F_el + c_b·δ̇). With a fixed c_b, a softer (less inflated) ball is more damped
relative to its stiffness, so the restitution rises with pressure. ``SHELL_C`` is fitted so that a 2.00 m drop onto a
rigid floor at 0.8 bar rebounds 1.45 m in air (FIFA's turf-test ball spec, 1.45 ± 0.03 m; Quality Pro 1.35-1.55 m).
``SHELL_K`` is an ESTIMATE (the bare shell, 10 % of the gas term at 0.8 bar).

**Turf** is a spring-dashpot k_t·u^1.5 + c_t·u̇ in series with the ball. The interface between them is massless, so
the turf compression u follows from F_ball(δ - u, δ̇ - u̇) = F_turf(u, u̇). ``TURF_C`` is fitted to a 0.80 m rebound
from 2.00 m, the middle of FIFA's 0.60-1.00 m band (natural pitch rating). k_t is an ESTIMATE. Rolling resistance
is a torque C_rr·F_n·a against the ball's rolling, where a is the contact arm. It decelerates a rolling ball at
C_rr·g / (1 + I/(m·R²)). ``TURF_ROLL`` puts FIFA's ramp test (ball released 1.000 m up) at 7 m, inside 4.0-10.0 m.

**Air.** Drag ½ρAC_d|v|v with a drag crisis: C_d = 0.45 below Re 1e5 (ESTIMATE, a rough sphere) and 0.17 above
Re 2e5 (Goff & Carré 2012: "CD ≈ 0.17" at 15.6-19.3 m/s, supercritical above v ≈ 14 m/s, Re ≈ 2e5), joined by a
smoothstep. Lift (Magnus) ½ρAC_L|v|² along ω × v with C_L = 0.36·tanh(S/0.2), S = Rω⊥/|v|. That gives 0.355 at
S = 0.5 (Goff & Carré: "CL ≈ 0.35" at Sp ≈ 0.5, "levelling off" above 0.3). The shape below 0.3 is an ESTIMATE.
Spin decays by the torque -½ρA·R²·C_M·|v|·ω, with C_M = 0.02 (ESTIMATE, no source found).
"""

from __future__ import annotations

import math

import numpy as np

G = 9.81
R_BALL = 0.69 / (2.0 * math.pi)        # m: IFAB mid circumference 69 cm
M_BALL = 0.43                          # kg: IFAB 410-450 g
I_FACTOR = 2.0 / 3.0                   # I = I_FACTOR·m·R² (thin shell, ESTIMATE)
P_TEST = 0.8e5                         # Pa gauge: FIFA test pressure "Outdoor 0.8bar"
P_IFAB = (0.6 * 101325.0, 1.1 * 101325.0)   # Pa gauge: Law 2 range, 0.6-1.1 atm
P_ATM = 101325.0
GAMMA = 1.4
SHELL_K = 5.5e3                        # N/m, bare shell (ESTIMATE)
SHELL_C = 12.44                        # N·s/m, fitted: 2.00 m drop at 0.8 bar, rigid floor, in air -> 1.45 m
TURF_K = 2.5e5                         # N/m^1.5 (ESTIMATE)
TURF_C = 66.6                          # N·s/m, fitted: 2.00 m drop at 0.8 bar on turf, in air -> 0.80 m
                                       # (the compliant branch; ~780 also gives 0.80 m)
TURF_MU = 0.6                          # ball-turf friction (ESTIMATE)
TURF_ROLL = 0.141                      # C_rr, fitted: FIFA ramp ball roll 7.0 m (band 4.0-10.0 m)
RHO_AIR, MU_AIR = 1.204, 1.81e-5       # kg/m³, Pa·s (20 °C, sea level)
CD_SUB, CD_SUPER = 0.45, 0.17
RE_LO, RE_HI = 1.0e5, 2.0e5           # crisis start (ESTIMATE) and end (Goff & Carré: Re ≈ 2e5 at 14 m/s)
CL_MAX, CL_S0 = 0.36, 0.2             # CL_MAX fitted (one point: C_L ≈ 0.35 at S 0.5); CL_S0 ESTIMATE
CM_SPIN = 0.02                         # ESTIMATE

FIFA_DROP = 2.00                       # m, from the bottom of the ball
FIFA_REBOUND_TEST_BALL = (1.42, 1.48)  # m, rigid floor (turf manual: 1.45 ± 0.03)
FIFA_REBOUND_PRO = (1.35, 1.55)        # m, metal plate, 20 °C (Quality Pro)
FIFA_TURF_REBOUND = (0.60, 1.00)       # m, 2.00 m drop on the pitch
FIFA_BALL_ROLL = (4.0, 10.0)           # m, ramp release 1.000 m
RAMP_HEIGHT, RAMP_RAIL_GAP = 1.000, 0.100


def inertia(m: float = M_BALL, r: float = R_BALL) -> float:
    return I_FACTOR * m * r * r


def area(r: float = R_BALL) -> float:
    return math.pi * r * r


# --- contact ----------------------------------------------------------------------------------------------------

def shell_force(d: np.ndarray | float, p: float = P_TEST, r: float = R_BALL, ks: float = SHELL_K) -> np.ndarray:
    """Elastic force [N] of the ball at overlap ``d`` [m] (clamped to [0, R]) and gauge pressure ``p`` [Pa]."""
    d = np.clip(np.asarray(d, float), 0.0, r)
    v0 = 4.0 / 3.0 * math.pi * r ** 3
    dv = math.pi * d * d * (3.0 * r - d) / 3.0
    pg = (p + P_ATM) * (v0 / (v0 - dv)) ** GAMMA - P_ATM
    return pg * math.pi * (2.0 * r * d - d * d) + ks * d


def shell_stiffness(d: float, p: float = P_TEST, r: float = R_BALL, ks: float = SHELL_K) -> float:
    """dF/dδ [N/m] (central difference)."""
    h = 1e-6
    return float((shell_force(d + h, p, r, ks) - shell_force(max(d - h, 0.0), p, r, ks)) / (d + h - max(d - h, 0.0)))


def turf_force(u: float, kt: float = TURF_K) -> float:
    return kt * max(u, 0.0) ** 1.5


def series_step(d: float, dd: float, u: float, dt: float, p: float, c_b: float, k_t: float, c_t: float
                ) -> tuple[float, float]:
    """One implicit step of the turf compression ``u`` under total overlap ``d`` and rate ``dd``; returns (u_new, F).
    The same update runs in the kernel (``kernels.turf_contact``)."""
    def g(x: float) -> float:
        return (c_b * dd + float(shell_force(d - x, p)) - turf_force(x, k_t)) / (c_b + c_t)

    jac = -(shell_stiffness(max(d - u, 0.0), p) + 1.5 * k_t * math.sqrt(max(u, 0.0))) / (c_b + c_t)
    un = min(max(u + dt * g(u) / (1.0 - dt * jac), 0.0), max(d, 0.0))
    f = turf_force(un, k_t) + c_t * (un - u) / dt
    if f <= 0.0:                                     # the ball has left: the turf relaxes through its dashpot
        un = u / (1.0 + dt * 1.5 * k_t * math.sqrt(max(u, 0.0)) / c_t)
        f = 0.0
    return un, f


def impact(v_in: float, p: float = P_TEST, surface: str = "rigid", m: float = M_BALL, dt: float = 2e-6,
           c_b: float = SHELL_C, k_t: float = TURF_K, c_t: float = TURF_C) -> dict:
    """A normal impact at ``v_in`` [m/s] onto a ``"rigid"`` floor or ``"turf"``, with gravity, float64, step ``dt``.
    Returns the rebound speed, restitution, contact time [s], peak force [N] and peak overlap [m]."""
    d, dd, u, t = 0.0, float(v_in), 0.0, 0.0
    peak_f = peak_d = 0.0
    loaded = False
    while t < 0.2:
        if surface == "turf":
            u, f = series_step(d, dd, u, dt, p, c_b, k_t, c_t)
        else:
            f = max(0.0, float(shell_force(d, p)) + c_b * dd) if d > 0.0 else 0.0
        loaded = loaded or f > 0.0
        if loaded and f == 0.0 and dd < 0.0:
            break
        dd += dt * (G - f / m)
        d += dt * dd
        t += dt
        peak_f, peak_d = max(peak_f, f), max(peak_d, d)
    return {"v_out": -dd, "e": -dd / v_in, "contact_s": t, "peak_n": peak_f, "peak_overlap_m": peak_d}


# --- air --------------------------------------------------------------------------------------------------------

def reynolds(v: float, r: float = R_BALL) -> float:
    return RHO_AIR * abs(v) * 2.0 * r / MU_AIR


def cd(re: np.ndarray | float, sub: float = CD_SUB, sup: float = CD_SUPER) -> np.ndarray:
    """Drag coefficient with the crisis between RE_LO and RE_HI (smoothstep)."""
    s = np.clip((np.asarray(re, float) - RE_LO) / (RE_HI - RE_LO), 0.0, 1.0)
    return sub + (sup - sub) * s * s * (3.0 - 2.0 * s)


def cl(s: np.ndarray | float, cl_max: float = CL_MAX, s0: float = CL_S0) -> np.ndarray:
    return cl_max * np.tanh(np.asarray(s, float) / s0)


def aero(v: np.ndarray, w: np.ndarray, r: float = R_BALL, cd_sub: float = CD_SUB, cd_sup: float = CD_SUPER,
         cl_max: float = CL_MAX, cm: float = CM_SPIN) -> tuple[np.ndarray, np.ndarray]:
    """(force [N], torque [N·m]) on a ball moving at ``v`` with spin ``w`` in still air."""
    v, w = np.asarray(v, float), np.asarray(w, float)
    sp = float(np.linalg.norm(v))
    if sp < 1e-9:
        return np.zeros(3), np.zeros(3)
    q = 0.5 * RHO_AIR * area(r)
    f = -q * float(cd(reynolds(sp, r), cd_sub, cd_sup)) * sp * v
    wxv = np.cross(w, v)
    n = float(np.linalg.norm(wxv))
    if n > 1e-12:
        f = f + q * float(cl(r * n / (sp * sp), cl_max)) * sp * sp * wxv / n
    return f, -q * r * r * cm * sp * w


def flight(x0: np.ndarray, v0: np.ndarray, w0: np.ndarray, seconds: float, dt: float = 1e-4, m: float = M_BALL,
           gravity: float = G, **kw: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RK4 flight in float64 (gravity, drag, Magnus, spin decay); returns (t, x [K,3], w [K,3]) every step."""
    ii = inertia(m)

    def rhs(y: np.ndarray) -> np.ndarray:
        f, tq = aero(y[3:6], y[6:9], **kw)
        return np.concatenate([y[3:6], f / m + np.array([0.0, 0.0, -gravity]), tq / ii])

    y = np.concatenate([x0, v0, w0]).astype(float)
    n = int(round(seconds / dt))
    out = np.empty((n + 1, 9))
    out[0] = y
    for k in range(n):
        a = rhs(y)
        b = rhs(y + 0.5 * dt * a)
        c = rhs(y + 0.5 * dt * b)
        e = rhs(y + dt * c)
        y = y + dt / 6.0 * (a + 2 * b + 2 * c + e)
        out[k + 1] = y
    return np.arange(n + 1) * dt, out[:, 0:3], out[:, 6:9]


def fall_speed(h: float, m: float = M_BALL, c_d: float = CD_SUB) -> float:
    """Speed after falling ``h`` from rest under quadratic drag (closed form, constant C_d)."""
    k = 0.5 * RHO_AIR * area() * c_d / m
    return math.sqrt(G / k * (1.0 - math.exp(-2.0 * k * h)))


def rise_height(v: float, m: float = M_BALL, c_d: float = CD_SUB) -> float:
    """Height reached rising from speed ``v`` under quadratic drag (closed form, constant C_d)."""
    k = 0.5 * RHO_AIR * area() * c_d / m
    return math.log(1.0 + k * v * v / G) / (2.0 * k)


def drop_rebound(h: float = FIFA_DROP, p: float = P_TEST, surface: str = "rigid", air: bool = True, **kw: float
                 ) -> dict:
    """FIFA drop test: rebound height [m] (bottom of the ball) after a drop from ``h``, in air or vacuum. Below
    Re 1e5 (~6.8 m/s) the drag coefficient is constant, so the closed forms hold."""
    v_in = fall_speed(h) if air else math.sqrt(2.0 * G * h)
    hit = impact(v_in, p, surface, **kw)
    up = rise_height(hit["v_out"]) if air else hit["v_out"] ** 2 / (2.0 * G)
    return {**hit, "v_in": v_in, "rebound_m": up}


# --- rolling ----------------------------------------------------------------------------------------------------

def roll_decel(c_rr: float = TURF_ROLL) -> float:
    """Deceleration [m/s²] of a ball rolling without slip against a resistance torque C_rr·m·g·R."""
    return c_rr * G / (1.0 + I_FACTOR)


def ramp_exit_speed(h: float = RAMP_HEIGHT, gap: float = RAMP_RAIL_GAP, r: float = R_BALL) -> float:
    """FIFA ball-roll ramp: the centre drops from ``h`` + the rail height to R; on the rails the ball rolls on the
    radius sqrt(R² - (gap/2)²). Rolling onto the turf keeps the angular momentum about the new contact point."""
    a = math.sqrt(r * r - 0.25 * gap * gap)
    drop = h + a - r
    v1 = math.sqrt(2.0 * G * drop / (1.0 + I_FACTOR * (r / a) ** 2))
    return v1 * (1.0 + I_FACTOR * r / a) / (1.0 + I_FACTOR)


def roll_distance(v0: float, c_rr: float = TURF_ROLL) -> float:
    return v0 * v0 / (2.0 * roll_decel(c_rr))


__all__ = [n for n in dir() if n.isupper() or n in (
    "aero", "area", "cd", "cl", "drop_rebound", "fall_speed", "flight", "impact", "inertia", "ramp_exit_speed",
    "reynolds", "rise_height", "roll_decel", "roll_distance", "series_step", "shell_force", "shell_stiffness",
    "turf_force")]
