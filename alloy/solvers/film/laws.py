# SPDX-License-Identifier: Apache-2.0
"""Film laws FILM-1..5 (design note §9): each builds its rig, runs, and returns measured numbers beside targets.

Targets and sources:
- FILM-1 cantilever: ASTM D1388 / heavy elastica, chord angle 41.5 deg at overhang L = 2c, c = (D/w)^(1/3).
- FILM-2 hanging stretch: p99 principal strain ≤ cap (σ_y/E, ExxonMobil HD 5201 D882 data).
- FILM-3 fold-release: return angle = the dead-fold test's (US10562684: 76 um HDPE coex 61/84.8 deg).
- FILM-4 falling cone: basket coffee filter 0.86 g, 87/151 mm, 47 mm tall, V_T 1.06-1.12 m/s (arXiv 2206.14902).
- FILM-5 energy: no per-step gain of film KE + PE net of air work above 1e-2·E_scale (D20); air work ≤ 0.
Each returns a plain dict so tests assert on it and bench scripts print it.
"""

from __future__ import annotations

import math

import numpy as np

from .config import FilmConfig
from .emit import add_film_mesh
from .materials import FilmSpec
from .rigs import FilmRig, frustum_mesh, grid_mesh, pin


def elastica_chord_angle(lam: float) -> float:
    """Chord angle (deg) from the clamp to the tip of a heavy cantilever with L/c = lam, clamped horizontal:
    θ'' = −lam³·(1 − s)·cos θ, θ(0) = 0, θ'(1) = 0 (s = arc length / L). 41.49 deg at lam = 2 (D1388's 41.5)."""
    from scipy.integrate import solve_bvp

    s = np.linspace(0.0, 1.0, 200)
    y = np.zeros((2, s.size))
    y[0] = 0.5 * s
    sol = solve_bvp(lambda t, u: np.vstack([u[1], -lam**3 * (1.0 - t) * np.cos(u[0])]),
                    lambda a, b: np.array([a[0], b[1]]), s, y, tol=1e-8, max_nodes=100000)
    th = sol.sol(s)[0]
    return math.degrees(math.atan2(np.trapezoid(np.sin(th), s), np.trapezoid(np.cos(th), s)))


def _energy_scale(rig: FilmRig, length: float) -> float:
    m = float(rig.model.particle_mass.numpy().sum()) / rig.worlds
    return m * 9.81 * length          # no 1 mJ floor (D20 has one): stricter for gram-scale films


def _ledger(rig: FilmRig, length: float) -> dict:
    led = rig.solver.ledger.numbers()
    es = _energy_scale(rig, length)
    return {"air_work_J": float(led["film_air_work"].max()), "step_gain_max_rel": float(led["film_step_gain_max"].max() / es),
            "projection_gain_max_rel": float(led["film_projection_gain_max"].max() / es), "E_scale_J": es}


def cantilever(spec: FilmSpec, *, dx: float = 2.5e-3, lam: float = 2.0, width: float = 0.025, clamp: float = 0.01,
               dt: float = 1.0e-4, seconds: float = 3.0, iterations: int = 10, device: str = "cpu",
               average: float = 1.0) -> dict:
    """FILM-1: strip clamped over ``clamp`` (pinned), overhang L = lam·c; mean chord angle over the last
    ``average`` s (the strip swings, air damps it)."""
    c = spec.bending_length
    L = lam * c
    nx, ny = int(round((L + clamp) / dx)), max(1, int(round(width / dx)))
    v, t = grid_mesh(nx, ny, dx)
    v[:, 0] -= clamp
    tip = np.flatnonzero(np.abs(v[:, 0] - v[:, 0].max()) < 1e-9)

    def build(b):
        p0, _ = add_film_mesh(b, v, t, spec, pos=(0.0, 0.0, 0.5))
        pin(b, p0 + np.flatnonzero(v[:, 0] <= 1e-9))
        return {"tip": p0 + tip}

    rig = FilmRig(build, device=device, dt=dt, iterations=iterations)

    def angle(r):
        q = r.q()[r.names[0]["tip"]]
        return math.degrees(math.atan2(-(q[:, 2].mean() - 0.5), q[:, 0].mean()))

    rig.run(seconds - average)
    series = rig.run(average, every=0.05, probe=angle)
    target = elastica_chord_angle(lam)
    a = float(np.mean(series))
    return {"angle_deg": a, "target_deg": target, "swing_deg": float(np.ptp(series)), "L_m": L, "c_m": c,
            "strain_p99_over_cap": float(np.percentile(rig.solver.strain_ratio(rig.s0), 99)),
            **_ledger(rig, L)}


def hang(spec: FilmSpec, *, width: float = 0.10, height: float = 0.15, dx: float = 5e-3, load_kg: float = 0.0,
         dt: float = 1.0e-3, seconds: float = 1.0, iterations: int = 10, device: str = "cpu",
         film: FilmConfig | None = None) -> dict:
    """FILM-2: sheet hung by its top edge (pinned row); with ``load_kg`` that mass spread over the bottom row (a
    handle strip carrying a bag's contents: inertia and weight of the load, as an item's contact load has; a bare
    force load on 0.4 mg vertices is not a physical case). Returns the principal strain over the cap (p99, max)
    and the sheet's length change."""
    nx, ny = int(round(width / dx)), int(round(height / dx))
    v, t = grid_mesh(nx, ny, dx, plane="xz")
    top = np.flatnonzero(np.abs(v[:, 2]) < 1e-9)
    bottom = np.flatnonzero(np.abs(v[:, 2] + ny * dx) < 1e-9)

    def build(b):
        p0, _ = add_film_mesh(b, v, t, spec, pos=(0.0, 0.0, 1.0))
        for p in p0 + bottom:
            b.particle_mass[int(p)] += load_kg / len(bottom)
        pin(b, p0 + top)
        return {"bottom": p0 + bottom}

    rig = FilmRig(build, device=device, dt=dt, iterations=iterations, film=film or FilmConfig())
    rig.run(seconds)
    r = rig.solver.strain_ratio(rig.s0)
    zb = rig.q()[rig.names[0]["bottom"], 2].mean() - 1.0
    stress = load_kg * 9.81 / (width * spec.thickness)
    return {"strain_p99_over_cap": float(np.percentile(r, 99)), "strain_max_over_cap": float(r.max()),
            "length_change": float(-zb / (ny * dx) - 1.0), "cap": spec.strain_cap,
            "sigma_over_E": stress / spec.youngs, "sigma_over_yield": stress / spec.yield_stress,
            **_ledger(rig, height)}


def fold_release(spec: FilmSpec, *, leg: float = 0.02, width: float = 0.01, dx: float = 2e-3, gap_deg: float = 2.0,
                 dt: float = 2.5e-4, seconds: float = 2.0, iterations: int = 10, device: str = "cpu") -> dict:
    """FILM-3: a strip folded flat about x = 0 (leg B rotated π − gap about the fold line, no gravity), leg A pinned,
    released; the fold opens by the hinge's elastic range. Returns the final opening (return) angle A = 180 − θ."""
    n_leg, ny = int(round(leg / dx)), max(1, int(round(width / dx)))
    v, t = grid_mesh(2 * n_leg, ny, dx)
    v[:, 0] -= leg                                          # fold line at x = 0
    rest = v.copy()
    ang = math.pi - math.radians(gap_deg)
    folded = v.copy()
    B = v[:, 0] > 1e-9
    folded[B, 0] = v[B, 0] * math.cos(ang)
    folded[B, 2] = v[B, 0] * math.sin(ang)
    tipB = np.flatnonzero(np.abs(v[:, 0] - leg) < 1e-9)

    def build(b):
        p0, _ = add_film_mesh(b, rest, t, spec, pos=(0.0, 0.0, 0.5))
        for n in range(len(v)):
            b.particle_q[p0 + n] = (float(folded[n, 0]), float(folded[n, 1]), 0.5 + float(folded[n, 2]))
        pin(b, p0 + np.flatnonzero(v[:, 0] < -1e-9))
        return {"tipB": p0 + tipB}

    rig = FilmRig(build, device=device, dt=dt, iterations=iterations, gravity=0.0)

    def opening(r):
        q = r.q()[r.names[0]["tipB"]].mean(0) - np.array([0.0, 0.0, 0.5])
        return math.degrees(math.atan2(q[2], q[0]))        # 0 = flat open, 180 - gap = folded

    start = opening(rig)
    series = rig.run(seconds, every=0.05, probe=opening)
    theta = float(np.mean(series[-int(len(series) * 0.25):]))
    rest = rig.model.edge_rest_angle.numpy()
    return {"fold_deg_start": start, "fold_deg_final": theta, "return_angle_deg": 180.0 - theta,
            "target_return_deg": math.degrees(spec.extras.yield_angle), "series_deg": [round(x, 1) for x in series],
            "rest_deg_max": float(np.degrees(np.abs(rest).max())), **_ledger(rig, leg)}


def falling_cone(spec: FilmSpec, *, mass: float = 0.86e-3, r_bottom: float = 0.0435, r_top: float = 0.0755,
                 height: float = 0.047, drop_s: float = 1.2, dt: float = 1.0e-3, iterations: int = 10,
                 device: str = "cpu", film: FilmConfig | None = None) -> dict:
    """FILM-4: the coffee filter frustum released at rest, open end up; terminal speed = mean descent speed over
    the last 0.3 s. Areal density is set so the mesh has the filter's mass."""
    v, t = frustum_mesh(r_bottom, r_top, height)
    area = 0.5 * np.linalg.norm(np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]), axis=1).sum()
    sp = FilmSpec(youngs=spec.youngs, poisson=spec.poisson, density=mass / (area * spec.thickness),
                  thickness=spec.thickness, yield_stress=spec.yield_stress, rayleigh_beta=spec.rayleigh_beta,
                  extras=spec.extras)

    def build(b):
        p0, _ = add_film_mesh(b, v, t, sp, pos=(0.0, 0.0, 3.0))
        return {"all": p0 + np.arange(len(v))}

    rig = FilmRig(build, device=device, dt=dt, iterations=iterations, film=film or FilmConfig())
    m = rig.model.particle_mass.numpy()

    def vz(r):
        ids = r.names[0]["all"]
        return float((r.qd()[ids, 2] * m[ids]).sum() / m[ids].sum())

    series = rig.run(drop_s, every=0.05, probe=vz)
    tail = [-x for x in series[-6:]]
    q = rig.q()[rig.names[0]["all"]]
    c = q.mean(0)
    rim = np.flatnonzero(np.abs(v[:, 2] - height) < 1e-9)
    shape = {"rim_radius_m": float(np.linalg.norm((q[rim] - c)[:, :2], axis=1).mean()),
             "height_m": float(q[rim, 2].mean() - q[0, 2])}
    return {"terminal_mps": float(np.mean(tail)), "tail_spread_mps": float(np.ptp(tail)), "target_mps": (1.06, 1.12),
            "area_m2": float(area), **shape, **_ledger(rig, 1.0)}
