# SPDX-License-Identifier: Apache-2.0
"""Plasticity twins for XB4-10's MuJoCo 4.0 re-score ([internal note] xb4/mujoco4/RERUN.md, "What the plasticity builder
must expose"; SCR-WP7-15).

`column_case(height_m, column, device)` stands a free column under its own weight on Alloy's soft-body solver
(`SolverVBDAlloy`, the soft_vbd route's class) with the J2 hook of `alloy.solvers.plastic.tets`.
- The column is 40 x 40 mm, ρ 1000, E 1e5 Pa, ν 0, base fixed, as MuJoCo's `plastic_free_column`.
- Yield: MuJoCo's J2 yield strain ε_y 0.005 read as its kernel reads it, σ_y = 3 E ε_y / (2 (1 + ν)), 750 Pa at ν 0.
  The analytic onset is ρ g H_c = σ_y (uniaxial stress at the base), H_c = 76.45 mm.
- Gravity ramps from 0 over `RAMP_S` (1 s), then holds for the rest of the 3 s, so the start carries no dynamic
  overshoot. MuJoCo settles for 3 s with damping 0.05; the ramp is the one setting that differs.
- Solver: dt 5 ms, 20 iterations, VBD's adaptive start (`alloy.solvers.vbd.accel`). The start makes a static
  equilibrium a fixed point: the elastic column settles to 0.999 of its static compression ρ g H²/(2E). With Newton's
  inertial start it settles 1.6-2x past it at 20-100 iterations, and 1.04x with Chebyshev 0.9 (CPU probe, 11-08),
  which would read the onset ~4 % early.
- Mesh: 2 x 2 hexes across (with ν 0 the stress is uniform across), each split in 5 tets as Newton's add_soft_grid.
  Layers are 1 mm for the bottom 8 mm, where the column first yields, then grow by 1.3x to at most 8 mm. A base
  element's stress is its centroid's, 0.5 mm up, so the twin's own onset is H_c + 0.5 mm at the most.
- Returns `{plastic_change, top_drop_m, ok}`:
  - `plastic_change` is the largest equivalent plastic strain after 3 s, 0 when the column stayed elastic;
  - `top_drop_m` is the top face's mean drop;
  - `ok` means finite and stable.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

COLUMN = {"width_m": 0.04, "rho_kg_m3": 1000.0, "E_pa": 1.0e5, "nu": 0.0, "yield_strain_j2": 0.005,  # rerun.COLUMN
          "gravity_m_s2": 9.81, "seconds": 3.0}
RAMP_S = 1.0               # s, gravity ramp
DT = 5.0e-3                # s
ITERATIONS = 20            # VBD iterations per step
BETA = 5.0e-3              # s, Rayleigh β (k_damp = β μ): ζ ≈ 0.5 on the column's first axial mode (~33 Hz)
CELLS = 2                  # hexes across each side (ν 0: the stress is uniform across)
BASE_LAYER, BASE_SPAN, GROWTH, MAX_LAYER = 1.0e-3, 8.0e-3, 1.3, 8.0e-3   # m, m, -, m


def yield_stress(column: dict[str, Any]) -> float:
    """σ_y [Pa] from the column dict: `sigma_y_pa` if given, else MuJoCo's 3 E ε_y / (2 (1 + ν))."""
    if column.get("sigma_y_pa") is not None:
        return float(column["sigma_y_pa"])
    E, nu = float(column["E_pa"]), float(column.get("nu", 0.0))
    return 3.0 * E * float(column.get("yield_strain_j2", 0.005)) / (2.0 * (1.0 + nu))


def layers(height: float) -> np.ndarray:
    """z levels [m]: fine at the base, growing upward, ending at `height`."""
    z = [0.0]
    dz = BASE_LAYER
    while z[-1] < height - 1e-12:
        if z[-1] >= BASE_SPAN - 1e-12:
            dz = min(dz * GROWTH, MAX_LAYER)
        z.append(min(z[-1] + dz, height))
    if len(z) > 2 and z[-1] - z[-2] < 0.3 * BASE_LAYER:     # no sliver on top
        z.pop(-2)
    return np.asarray(z)


def column_mesh(width: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """(vertices [V, 3], tets [T, 4]) of the graded column, Newton add_soft_grid's 5-tet split per hex."""
    zs, n = layers(height), CELLS
    xs = np.linspace(-0.5 * width, 0.5 * width, n + 1)
    verts = np.array([(x, y, z) for z in zs for y in xs for x in xs])

    def g(x: int, y: int, z: int) -> int:
        return (n + 1) * (n + 1) * z + (n + 1) * y + x

    tets = []
    for z in range(len(zs) - 1):
        for y in range(n):
            for x in range(n):
                v0, v1, v2, v3 = g(x, y, z), g(x + 1, y, z), g(x + 1, y, z + 1), g(x, y, z + 1)
                v4, v5, v6, v7 = g(x, y + 1, z), g(x + 1, y + 1, z), g(x + 1, y + 1, z + 1), g(x, y + 1, z + 1)
                if (x & 1) ^ (y & 1) ^ (z & 1):
                    tets += [(v0, v1, v4, v3), (v2, v3, v6, v1), (v5, v4, v1, v6), (v7, v6, v3, v4), (v4, v1, v6, v3)]
                else:
                    tets += [(v1, v2, v5, v0), (v3, v0, v7, v2), (v4, v7, v0, v5), (v6, v5, v2, v7), (v5, v2, v7, v0)]
    return verts, np.asarray(tets, dtype=np.int32)


def _signed_volumes(verts: np.ndarray, tets: np.ndarray) -> np.ndarray:
    a, b, c, d = (verts[tets[:, k]] for k in range(4))
    return np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0


def build_column(height_m: float, column: dict[str, Any], device: str) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    """(model, solver, bottom vertex ids, top vertex ids) of the column twin."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.solvers.plastic.tets import mark_tets  # noqa: PLC0415
    from alloy.solvers.vbd.base import SolverVBDAlloy  # noqa: PLC0415

    E, nu, rho = float(column["E_pa"]), float(column.get("nu", 0.0)), float(column["rho_kg_m3"])
    mu, lam = E / (2.0 * (1.0 + nu)), E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    verts, tets = column_mesh(float(column.get("width_m", 0.04)), height_m)
    flip = _signed_volumes(verts, tets) < 0.0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)            # gravity is set every step (the ramp)
    b.add_soft_mesh(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                    vertices=[wp.vec3(*v) for v in verts], indices=tets.flatten().tolist(), density=rho, k_mu=mu,
                    k_lambda=lam, k_damp=BETA * mu, add_surface_mesh_edges=False, label="column")
    bottom = np.nonzero(verts[:, 2] == 0.0)[0]
    top = np.nonzero(np.isclose(verts[:, 2], height_m))[0]
    for i in bottom:
        b.particle_mass[int(i)] = 0.0
    mark_tets(b, 0, len(tets), yield_stress(column))
    b.color()
    model = b.finalize(device=device)
    solver = SolverVBDAlloy(model, iterations=ITERATIONS, alloy_initial_guess="adaptive")
    return model, solver, bottom, top


def column_case(height_m: float, column: dict[str, Any] | None = None, device: str = "cpu") -> dict[str, Any]:
    """RERUN.md's hook (module doc)."""
    column = {**COLUMN, **(column or {})}
    model, solver, _, top = build_column(float(height_m), column, device)
    s0, s1, control = model.state(), model.state(), model.control()
    g = float(column.get("gravity_m_s2", 9.81))
    z0 = float(s0.particle_q.numpy()[top, 2].mean())
    steps = round(float(column.get("seconds", 3.0)) / DT)
    for k in range(steps):
        model.set_gravity((0.0, 0.0, -g * min(1.0, (k + 1) * DT / RAMP_S)))
        solver.step(s0, s1, control, None, DT)
        s0, s1 = s1, s0
    q = s0.particle_q.numpy()
    ebar = solver.plastic.ebar.numpy() if getattr(solver, "plastic", None) is not None else np.zeros(1)
    finite = bool(np.isfinite(q).all() and np.isfinite(ebar).all())
    drop = z0 - float(q[top, 2].mean())
    return {"plastic_change": float(ebar.max()) if finite else math.nan, "top_drop_m": drop,
            "ok": finite and abs(drop) < 0.5 * height_m}


# --- bend and stay (P1 law V3) ---------------------------------------------------------------------------------------

BAR = {"length_m": 0.05, "side_m": 0.01, "rho_kg_m3": 1000.0, "E_pa": 1.0e5, "sigma_y_pa": 750.0, "hardening_pa": 0.0}
BEND_RAMP_S, BEND_HOLD_S, BEND_SETTLE_S = 3.0, 1.0, 2.0
BEND_ITERATIONS, BEND_CHEBYSHEV, BEND_BETA = 60, 0.9, 0.025    # (s) β: about critical on the bar's first bending mode


def bend_case(K: float, bar: dict[str, Any] | None = None, device: str = "cpu", nz: int = 4) -> dict[str, Any]:
    """Bend and stay: a square tet bar (ν 0) clamped at x = 0 takes a follower couple M = K M_y at x = L: a uniform
    moment, so no plastic zone can localise. M ramps up over 3 s, holds 1 s, ramps down, then the bar settles 2 s.

    Only the middle 60 % yields, a gauge length with elastic grips: the clamp and the loaded face carry point loads
    that would yield locally. The curvature comes from a circle through three centreline points at 0.3 / 0.5 / 0.7 L.

    Returned against the section closed form (`closed_forms.rect_moment`, uniaxial):
    - `loaded` and `residual` curvature, with their closed-form values `loaded_cf` = κ(M) and
      `residual_cf` = κ(M) - M/(EI);
    - `ebar_max` in the gauge;
    - `ebar_grips`, which is 0.

    The tets bend about 10 % stiff at 4 elements through the depth (linear tets): docs/notes/plasticity-design.md."""
    from scipy.optimize import brentq  # noqa: PLC0415

    from alloy._vendor import newton  # noqa: PLC0415
    from alloy.solvers.plastic import closed_forms as C  # noqa: PLC0415
    from alloy.solvers.plastic.tets import mark_tets  # noqa: PLC0415
    from alloy.solvers.vbd.base import SolverVBDAlloy  # noqa: PLC0415

    p = {**BAR, **(bar or {})}
    keys = ("length_m", "side_m", "E_pa", "sigma_y_pa", "hardening_pa", "rho_kg_m3")
    L, a, E, sy, H, rho = (float(p[k]) for k in keys)
    nl = round(nz * L / a)
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    b.add_soft_grid(pos=wp.vec3(0.0, -a / 2, -a / 2), rot=wp.quat_identity(), vel=wp.vec3(), dim_x=nl, dim_y=1,
                    dim_z=nz, cell_x=L / nl, cell_y=a, cell_z=a / nz, density=rho, k_mu=E / 2, k_lambda=0.0,
                    k_damp=BEND_BETA * E / 2, fix_left=True)
    cx = np.asarray(b.particle_q)[np.asarray(b.tet_indices)].mean(axis=1)[:, 0]
    gauge = (cx >= 0.2 * L) & (cx <= 0.8 * L)
    for t in np.nonzero(gauge)[0]:
        mark_tets(b, int(t), int(t) + 1, sy, H)
    b.color()
    m = b.finalize(device=device)
    m.set_gravity((0.0, 0.0, 0.0))
    solver = SolverVBDAlloy(m, iterations=BEND_ITERATIONS, alloy_initial_guess="adaptive",
                            alloy_chebyshev_rho=BEND_CHEBYSHEV)
    q0 = m.particle_q.numpy()
    face = np.isclose(q0[:, 0], L)
    top, bot = face & np.isclose(q0[:, 2], a / 2), face & np.isclose(q0[:, 2], -a / 2)
    line = np.isclose(q0[:, 1], -a / 2) & np.isclose(q0[:, 2], 0.0)
    idx = [int(np.nonzero(line & np.isclose(q0[:, 0], f * L))[0][0]) for f in (0.3, 0.5, 0.7)]
    My = sy * a ** 3 / 6
    M = K * My
    sz2 = float((q0[face, 2] ** 2).sum())
    s0, s1, control = m.state(), m.state(), m.control()
    out = []
    phases = ((BEND_RAMP_S, 0.0, 1.0), (BEND_HOLD_S, 1.0, 1.0), (BEND_RAMP_S, 1.0, 0.0), (BEND_SETTLE_S, 0.0, 0.0))
    for dur, f0, f1 in phases:
        n = round(dur / DT)
        for k in range(n):
            moment = M * (f0 + (f1 - f0) * (k + 1) / n)
            s0.particle_f.assign(_couple(s0.particle_q.numpy(), face, top, bot, sz2, moment))
            solver.step(s0, s1, control, None, DT)
            s0, s1 = s1, s0
        out.append(_curvature(s0.particle_q.numpy()[idx]))
    ky = 2.0 * sy / (E * a)
    k_cf = brentq(lambda k: C.rect_moment(k, a, E, sy, H) * a - M, 1e-12, 1e3 * ky)
    eb = solver.plastic.ebar.numpy()
    return {"loaded": out[1], "residual": out[3], "loaded_cf": k_cf, "residual_cf": k_cf - M / (E * a ** 4 / 12),
            "ebar_max": float(eb[gauge].max()), "ebar_grips": float(eb[~gauge].max()), "tets": int(m.tet_count)}


def _couple(q: np.ndarray, face: np.ndarray, top: np.ndarray, bot: np.ndarray, sz2: float,
            moment: float) -> np.ndarray:
    """Particle forces of a follower couple on the end face: linear across its current height, along its axis."""
    ez = q[top].mean(axis=0) - q[bot].mean(axis=0)
    ez /= np.linalg.norm(ez)
    d = np.cross(np.array([0.0, 1.0, 0.0]), ez)
    d /= np.linalg.norm(d)
    z = (q[face] - q[face].mean(axis=0)) @ ez
    f = np.zeros_like(q)
    f[face] = -(moment / sz2) * z[:, None] * d[None, :]
    return f.astype(np.float32)


def _curvature(p: np.ndarray) -> float:
    """Signed curvature of the circle through three points (x-z plane)."""
    p1, p2, p3 = p[:, [0, 2]]
    a, b, c = np.linalg.norm(p2 - p1), np.linalg.norm(p3 - p2), np.linalg.norm(p3 - p1)
    cross = (p2 - p1)[0] * (p3 - p1)[1] - (p2 - p1)[1] * (p3 - p1)[0]
    return float(2.0 * cross / (a * b * c))
