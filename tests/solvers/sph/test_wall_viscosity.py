# SPDX-License-Identifier: Apache-2.0
"""SCR-SC1-1 acceptance: wall (boundary) viscosity of the Weiler 2018 solve (``wall_viscosity.py``).

- moving-wall no-slip: honey resting on a floor that slides at U is dragged to U at the wall (within 10 %); with
  free-slip walls (``boundary_viscosity=None``) it stays put;
- flowing Nusselt films (SCR-SC1-3): honey films 1, 2 and 4 mm thick on a vertical static wall, seeded with every
  particle jittered (an uneven near-wall layer, as after flow), drain in steady state at the Nusselt mean speed
  rho g e^2 / (3 mu) (within 15 %);
- rod coat (SCR-SC1-3): a honey coat on a vertical rod, Cartesian seeding, drains at the closed-form mean speed of a
  film on the outside of a tube (within 15 %); with 10x the wall viscosity it stays stable (no particle in the rod);
- ledger: the sliding floor's reported work (``alloy_shape_work``) pays for the liquid's kinetic energy.
The bitwise-unchanged requirement with ``boundary_viscosity=None`` is the rest of the SPH suite (goldens) plus
``scripts``-free check ``test_none_is_bitwise_the_free_slip_path``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.solvers.sph.build import add_fluid_points, wall_box
from alloy.solvers.sph.config import BOUNDARY_SDF, register_custom_attributes
from alloy.testing.sph import default_config

HONEY_RHO, HONEY_MU, G = 1420.0, 6.0, 9.81
R = 0.5e-3  # particle radius [m]


def _solver(m, cfg):
    from alloy.solvers.sph.solver import SolverDFSPH

    return SolverDFSPH(m, cfg)


def sliding_floor(U: float, mu_b: float | None, steps: int = 100, dt: float = 1.0e-3, n: int = 6,
                  layers: int = 4) -> dict:
    """An n x n x layers honey block on a kinematic floor sliding at U along +x from rest. Returns the mean x
    velocity of the bottom layer and of the block at the end, the liquid's kinetic energy and the floor's work."""
    s = 2.0 * R
    cfg = default_config(R, viscosity="weiler2018", boundary_viscosity=mu_b, integrate_bodies="none")
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    register_custom_attributes(b)
    b.begin_world()
    floor = b.add_body(xform=wp.transform_identity(), label="floor")
    wall_box(b, (0.0, 0.0, -cfg.h), (0.02, 0.02, cfg.h), body=floor)  # top face at z = 0
    g = (np.arange(n) + 0.5) * s - 0.5 * n * s
    pts = np.array([(x, y, R + s * k) for k in range(layers) for y in g for x in g])
    add_fluid_points(b, cfg, pts, HONEY_RHO, HONEY_MU)
    b.end_world()
    m = b.finalize(device="cpu")
    sv = _solver(m, cfg)
    mass = m.particle_mass.numpy().astype(np.float64)
    work = wp.zeros(m.shape_count, dtype=wp.float64, device=m.device)
    a, c = m.state(), m.state()
    x, w_floor = 0.0, 0.0
    for _ in range(steps):
        a.body_q.assign([wp.transform((x, 0.0, 0.0), wp.quat_identity())])
        a.body_qd.assign(np.array([[U, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        x += U * dt
        sv.step(a, c, None, None, dt)
        a, c = c, a
        sv.alloy_shape_work(work)
        w_floor += float(work.numpy().sum())
    q, v = a.particle_q.numpy(), a.particle_qd.numpy().astype(np.float64)
    bottom = q[:, 2] < 1.5 * s
    return {"v_bottom": float(v[bottom, 0].mean()), "v_mean": float(v[:, 0].mean()),
            "ke": float(0.5 * np.sum(mass * np.sum(v * v, axis=1))), "w_floor": w_floor, "drop": float(q[:, 2].min())}


def _run(m, cfg, T: float, dt: float):
    sv = _solver(m, cfg)
    a, c = m.state(), m.state()
    for _ in range(int(round(T / dt))):
        sv.step(a, c, None, None, dt)
        a, c = c, a
    return a.particle_q.numpy().astype(np.float64), a.particle_qd.numpy().astype(np.float64)


def flowing_film(e: float, mu_b: float | None = HONEY_MU, T: float = 0.05, layers: int = 4, seed: int = 0) -> dict:
    """A honey film e thick (``layers`` x 2r) on a static vertical wall (x <= 0), 12 particles wide and 60 tall,
    every particle jittered by up to +-0.3 spacing, draining for T. Mean downward speed and film thickness (from the
    particle count) in a window clear of the film's edges, and the closest approach to the wall."""
    r = e / (2.0 * layers)
    s = 2.0 * r
    cfg = default_config(r, viscosity="weiler2018", boundary_viscosity=mu_b)
    ny, nz = 12, 60
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    register_custom_attributes(b)
    b.begin_world()
    wall_box(b, (-cfg.h, 0.0, 0.0), (cfg.h, ny * s, nz * s))
    rng = np.random.default_rng(seed)
    g = np.array([(r + s * i, (j + 0.5 - 0.5 * ny) * s, (k + 0.5 - 0.5 * nz) * s)
                  for i in range(layers) for j in range(ny) for k in range(nz)])
    pts = g + rng.uniform(-0.3 * s, 0.3 * s, g.shape)
    pts[:, 0] = np.maximum(pts[:, 0], 0.5 * r)
    add_fluid_points(b, cfg, pts, HONEY_RHO, HONEY_MU)
    b.end_world()
    q, v = _run(b.finalize(device="cpu"), cfg, T, 2.0 * r)  # dt = 1 ms at r = 0.5 mm
    win = (np.abs(q[:, 1]) < 0.25 * ny * s) & (np.abs(q[:, 2]) < nz * s / 6.0)
    e_eff = win.sum() * s**3 / ((0.5 * ny * s) * (nz * s / 3.0))
    return {"u": float(-v[win, 2].mean()), "e": e_eff, "x_min": float(q[:, 0].min()), "v_max": float(np.abs(v).max())}


def _tube_film_mean(R: float, bo: float) -> float:
    """Mean speed of a film between a tube of radius R and its free surface at bo (Bird, Stewart and Lightfoot,
    Transport Phenomena, 2nd ed., problem 2B.6, film on the outside of a circular tube):
    v_z(r) = rho g R^2 / (4 mu) (1 - (r/R)^2 + 2 (bo/R)^2 ln(r/R))."""
    rr = np.linspace(R, bo, 2001)
    vz = HONEY_RHO * G * R * R / (4.0 * HONEY_MU) * (1.0 - (rr / R) ** 2 + 2.0 * (bo / R) ** 2 * np.log(rr / R))
    return float(np.trapezoid(vz * rr, rr) / np.trapezoid(rr, rr))


def rod_coat(mu_b: float | None = HONEY_MU, R: float = 3.0e-3, e: float = 1.5e-3, r: float = 0.25e-3,
             T: float = 0.05) -> dict:
    """A honey coat e thick on a static vertical rod (analytic cylinder, radius R, axis z), seeded on the Cartesian
    lattice (an uneven layer against the curved wall), draining for T. Mean downward speed and coat outer radius
    (from the particle count) in a window clear of the coat's ends; closest approach to the rod axis."""
    s = 2.0 * r
    cfg = default_config(r, viscosity="weiler2018", boundary_viscosity=mu_b)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    register_custom_attributes(b)
    b.begin_world()
    b.add_shape_cylinder(-1, radius=R, half_height=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
                         custom_attributes={"sph:shape_boundary": BOUNDARY_SDF})
    ax = (np.arange(-int((R + e) / s) - 1, int((R + e) / s) + 2) + 0.5) * s
    gx, gy, gz = np.meshgrid(ax, ax, (np.arange(40) + 0.5 - 20) * s, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[(rad >= R + 0.5 * r) & (rad <= R + e - 0.5 * r)]
    add_fluid_points(b, cfg, pts, HONEY_RHO, HONEY_MU)
    b.end_world()
    q, v = _run(b.finalize(device="cpu"), cfg, T, 2.0 * r)
    win = np.abs(q[:, 2]) < 5.0 * s
    bo = math.sqrt(R * R + win.sum() * s**3 / (math.pi * 10.0 * s))
    return {"u": float(-v[win, 2].mean()), "bo": bo, "rad_min": float(np.hypot(q[:, 0], q[:, 1]).min()),
            "v_max": float(np.abs(v).max())}


def test_sliding_wall_drags_honey_to_wall_speed():
    U = 0.02
    o = sliding_floor(U, HONEY_MU)
    assert abs(o["v_bottom"] - U) <= 0.10 * U, o
    assert abs(o["v_mean"] - U) <= 0.10 * U, o


def test_free_slip_floor_does_not_drag():
    U = 0.02
    o = sliding_floor(U, None)
    assert abs(o["v_bottom"]) < 0.05 * U, o


def test_sliding_floor_work_pays_for_the_liquid_energy():
    """Impulsive start: the floor's work m U^2 is half kinetic energy, half dissipated (the ledger must not see a gain)."""
    U = 0.02
    o = sliding_floor(U, HONEY_MU)
    assert o["w_floor"] > 0.0, o
    assert 0.3 * o["w_floor"] < o["ke"] <= 1.01 * o["w_floor"], o


@pytest.mark.law
@pytest.mark.parametrize("e", [pytest.param(1.0e-3, marks=pytest.mark.slow), 2.0e-3, 4.0e-3])
def test_flowing_film_drains_at_the_nusselt_speed(e):
    """Steady drainage is reached within e^2 / nu (< 4 ms here); 0.05 s keeps the window clear of both edges."""
    o = flowing_film(e)
    nusselt = HONEY_RHO * G * o["e"] ** 2 / (3.0 * HONEY_MU)
    assert abs(o["u"] - nusselt) <= 0.15 * nusselt, (o, nusselt)
    assert o["x_min"] > 0.0, o


@pytest.mark.law
def test_rod_coat_drains_at_the_tube_film_speed():
    R = 3.0e-3
    o = rod_coat(R=R)
    ref = _tube_film_mean(R, o["bo"])
    assert abs(o["u"] - ref) <= 0.15 * ref, (o, ref)
    assert o["rad_min"] > R, o


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_ten_times_wall_viscosity_stays_stable():
    """WP9's case (main went unstable after ~0.5 s): 10x mu_b on a flowing film for 0.6 s stays out of the wall, with
    bounded speeds, never faster than no-slip. It drains slower than Nusselt (the over-coupled first layer sticks, so
    the effective no-slip plane sits ~1.5 r off the wall): a stability case, not an accuracy one."""
    o = flowing_film(2.0e-3, mu_b=10.0 * HONEY_MU, T=0.6)
    nusselt = HONEY_RHO * G * o["e"] ** 2 / (3.0 * HONEY_MU)
    assert o["x_min"] > 0.0, o
    assert o["v_max"] < 3.0 * nusselt, (o, nusselt)
    assert 0.0 < o["u"] <= 1.15 * nusselt, (o, nusselt)


@pytest.mark.parametrize("mu_b", [None, 0.0])
def test_none_is_bitwise_the_free_slip_path(mu_b):
    """``None`` and 0 take the has_wall = 0 path: bitwise equal to each other over a sliding-floor run."""
    a = sliding_floor(0.02, mu_b, steps=10)
    b = sliding_floor(0.02, None, steps=10)
    assert a == b
