# SPDX-License-Identifier: Apache-2.0
"""LIQ-8 (SPEC §5.2): the leak monitor flags a particle whose step path crosses a wall and nothing in normal
motion; a particle in a slit narrower than 2r settles (opposing-wall compliance), a particle fast enough to cross a
wall in one step is stopped at it (sweep, exact for box walls at any speed), a collapsed cluster is not thrown out
(diagonal floor), a lone particle in a closing wedge is not kicked (contact limit) and a particle driven into a box
tank's corner stays inside (projection passes; the monitor counts particles left inside a wall);
SolverDFSPH.reset brings retired particles back; the GPU campaign runs B0 + B1 on the pour scene."""

from __future__ import annotations

import os

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

import alloy.solvers.sph.solver as solver_module
from alloy.solvers.sph.build import add_box_tank, add_fluid_points
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.solver import SolverDFSPH
from alloy.testing.sph import (closing_wedge_max_speed, collapsed_cluster, default_config, liq8_campaign, one_particle,
                               run, slit, water_column, write_result)

DT = 1.0e-3
POUR_DT = 0.02 / 64  # the pour scene's substep
R = 2.0e-3


def test_leak_monitor_flags_a_tunnel_and_not_normal_motion():
    sc = water_column(4, 4, 4)
    s = sc.make_solver()
    mon = s.enable_leak_monitor()
    a, b = sc.model.state(), sc.model.state()
    s.prime(a, DT)
    a, b = run(s, a, b, 20, DT)
    assert int(mon.counts()["leaks"][0]) == 0 and mon.counts()["max_depth_r"][0] < 0.5
    assert int(mon.counts()["in_solid"][0]) == 0
    q = a.particle_q.numpy()
    i = int(np.argmin(q[:, 0] + 1e-3 * q[:, 2]))  # a particle next to the x_lo wall (the wall is h = 4r thick)
    tunnel = sc.model.state()
    moved = q.copy()
    moved[i, 0] = -s.config.h - s.config.particle_radius  # through the wall, out behind it
    tunnel.particle_q.assign(moved)
    empty = wp.zeros(1, dtype=wp.transform)
    mon.observe(a, tunnel, empty, empty)
    c = mon.counts()
    assert int(c["leaks"][0]) == 1 and c["max_depth_r"][0] >= 1.5  # crossed at T/2 = 2r (exact for box walls)
    mon.observe(a, tunnel, empty, empty)
    assert int(mon.counts()["leaks"][0]) == 1  # each particle counts once per episode


def test_leak_monitor_counts_a_particle_left_inside_a_wall():
    """A step that ends 0.1 r inside the x_lo wall is too shallow for a leak (path depth < 1 r) but counts as in
    solid: the signature of a particle sliding out along a tiled wall's seams 1 um deep (leaks.py)."""
    sc = water_column(4, 4, 4)
    s = sc.make_solver()
    mon = s.enable_leak_monitor()
    a, b = sc.model.state(), sc.model.state()
    s.prime(a, DT)
    a, b = run(s, a, b, 5, DT)
    q = a.particle_q.numpy()
    i = int(np.argmin(q[:, 0] + 1e-3 * q[:, 2]))
    inside = sc.model.state()
    moved = q.copy()
    moved[i, 0] = -0.1 * s.config.particle_radius
    inside.particle_q.assign(moved)
    empty = wp.zeros(1, dtype=wp.transform)
    mon.observe(a, inside, empty, empty)
    c = mon.counts()
    assert int(c["leaks"][0]) == 0 and int(c["in_solid"][0]) == 1, c
    assert c["in_solid_depth_r"][0] == pytest.approx(0.1, rel=1e-3)
    mon.observe(a, inside, empty, empty)
    assert int(mon.counts()["in_solid"][0]) == 1  # once per episode


@pytest.mark.parametrize("sx,sy", [(1, -1), (-1, -1), (1, 1), (-1, 1)])
def test_particle_driven_into_a_tank_corner_stays_inside(sx, sy):
    """One particle 1 um from the floor and two walls of a box tank, driven into that corner (1 m/s into each wall,
    2 m/s down), ends every step inside the tank and never inside a wall. Each push carries it into the extent of the
    next tiled box; with two projection passes the (+x, -y) corner left it 1 um inside the y wall, and it slid out
    along the seams, 1 mm deep after 10 steps, at a path depth of 1e-3 r (no leak counted; integrate.py)."""
    cfg = DFSPHConfig(particle_radius=1.0e-3)
    w, eps = 6 * cfg.spacing, 1.0e-6
    b = newton.ModelBuilder()
    register_custom_attributes(b)
    b.begin_world()
    add_box_tank(b, (0.0, 0.0, 0.0), (w, w, 10 * cfg.spacing), wall=cfg.h)
    x0 = (w - eps if sx > 0 else eps, w - eps if sy > 0 else eps, 3 * eps)
    add_fluid_points(b, cfg, np.array([x0]), 1000.0, 1.0e-3, (0.0, 0.0, 0.0))
    b.end_world()
    m = b.finalize(device="cpu")
    s = SolverDFSPH(m, cfg)
    mon = s.enable_leak_monitor()
    a, c = m.state(), m.state()
    a.particle_qd.assign(np.array([(float(sx), float(sy), -2.0)], np.float32))
    for _ in range(10):
        s.step(a, c, None, None, 1.0e-4)
        a, c = c, a
        q = a.particle_q.numpy()[0]
        assert 0.0 <= q[0] <= w and 0.0 <= q[1] <= w and q[2] >= 0.0, q
    assert int(mon.counts()["in_solid"][0]) == 0


def _play(scene, steps: int):
    """Step a scene at the pour substep; returns the solver, its leak monitor, max |v| and particle 0's x per step."""
    s = scene.make_solver()
    mon = s.enable_leak_monitor()
    a, b = scene.model.state(), scene.model.state()
    speed, x = [], []
    for _ in range(steps):
        s.step(a, b, None, None, POUR_DT)
        a, b = b, a
        speed.append(float(np.linalg.norm(a.particle_qd.numpy(), axis=1).max()))
        x.append(a.particle_q.numpy()[0].copy())
    return s, mon, np.array(speed), np.array(x)


@pytest.mark.parametrize("gap", [2.0e-3, 2.6e-3])
def test_particle_in_a_slit_narrower_than_2r_settles(gap):
    """The LIQ-8 blow-up: two walls closer than 2r cancel each other's Psi gradient, the density solve overshoots
    across the slit and the particle rattles (without the opposing-wall compliance: 3 m/s forever in the 2.6 mm slit;
    51 m/s and a wall crossing on the pour scene). With it the particle settles in the slit."""
    s, mon, v, x = _play(slit(gap, 0.2 * R, 0.5), 600)
    assert v.max() < 1.2 and v[-100:].mean() < 0.15, (v.max(), v[-100:].mean())
    assert 0.0 < x[:, 0].min() and x[:, 0].max() < gap  # never inside either wall
    assert int(mon.counts()["leaks"][0]) == 0 and int(s.stats.diagnostics()["projected"][0]) == 0


def _shot(speed: float):  # towards an h-thick wall (x 9-17 mm, beyond h: not in the particle's slots)
    return one_particle([((9.0e-3 + 2.0 * R, 0.0, 0.0), (2.0 * R, 0.02, 0.02))], (0.0, 0.0, 0.0), (speed, 0.0, 0.0))


@pytest.mark.parametrize("speed", [60.0, 3700.0])
def test_fast_particle_is_stopped_at_the_wall_it_would_cross(monkeypatch, speed):
    """60 m/s: one step = 18.75 mm. 3700 m/s: one step = 1.16 m = 64 x 18 mm, so 64 evenly spaced path samples land
    at 0 and 18 mm and skip the wall; the box test is exact (a collapsed cluster reached 2,666 m/s before the
    diagonal floor)."""
    s, mon, _v, x = _play(_shot(speed), 1)
    assert 8.9e-3 < x[0, 0] < 9.0e-3 and int(s.stats.diagnostics()["swept"][0]) == 1
    assert int(mon.counts()["leaks"][0]) == 0
    monkeypatch.setattr(solver_module, "SWEEP", 1.0e9)  # sweep off: it crosses, the monitor's all-shape test sees it
    s, mon, _v, x = _play(_shot(speed), 1)
    assert x[0, 0] > 17.0e-3 and int(s.stats.diagnostics()["swept"][0]) == 0
    assert int(mon.counts()["leaks"][0]) == 1 and mon.counts()["max_depth_r"][0] > 1.9  # T/2 = 2r, exact


@pytest.mark.parametrize("theta,speed", [(5.0, 0.5), (10.0, 0.2)])
def test_lone_particle_in_a_closing_wedge_is_not_kicked(theta, speed):
    """The two escapes of the fixed-floor campaign: the cup's bottom corner pressed into the receiver wall traps a lone
    particle (r_i = self + walls = 2-4) and the density solve kicked it out at 3-7 m/s, over the tray rim. Without the
    contact limit this scene gives 5.3 m/s (5 deg, 0.5 m/s) and 3.7 m/s (10 deg, 0.2 m/s), measured; with it the
    particle leaves at about the wall speed plus Config.contact_separation_speed (0.5 m/s) and gravity."""
    assert closing_wedge_max_speed(theta, speed) < 2.0


@pytest.mark.parametrize("spread", [1.0e-6, 1.0e-4])
def test_collapsed_cluster_is_not_thrown_out(spread):
    """Four particles within ``spread`` of one point have rho > rho0 but almost no density gradient; without the
    diagonal floor the density solve threw them apart at 740 m/s (1 um) and 27 m/s (0.1 mm) within two steps
    (measured on this scene, pressure.py), enough to leave the pour scene's tray in one substep. With it they separate
    at < 10 m/s."""
    s, mon, v, _x = _play(collapsed_cluster(4, spread), 20)
    assert v.max() < 10.0, v.max()
    assert int(mon.counts()["leaks"][0]) == 0


def test_reset_restores_retired_particles_and_ledger():
    cfg = default_config(world_bounds=((-0.004, -0.004, -0.004), (0.012, 0.012, 0.006)))  # top layers escape
    sc = water_column(4, 4, 6, worlds=2, cfg=cfg)
    s = sc.make_solver()
    a, b = sc.model.state(), sc.model.state()
    s.step(a, b, None, None, DT)
    assert np.all(s.report().escaped > 0)
    s.reset(b, world_mask=wp.array([1, 0, 0], dtype=wp.bool))
    flags = sc.model.particle_flags.numpy()
    world = sc.model.particle_world.numpy()
    assert np.all(flags[world == 0] & 1) and not np.all(flags[world == 1] & 1)
    assert s.report().escaped[0] == 0 and s.report().escaped[1] > 0
    led = s.mass_ledger()
    assert led["out"][0] == 0 and led["initial"][0] == led["active"][0]


@pytest.mark.gpu
@pytest.mark.nightly
@pytest.mark.slow                   # over 1 h on an L4: out of the GPU suite's 1 h shards
@pytest.mark.timeout(6 * 3600)
def test_liq8_bully_pour_campaign_gpu():
    """LIQ-8: 0 particles through walls or escaping over >= 10k bully episodes on the pour scene (B0 + B1)."""
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    worlds = int(os.environ.get("ALLOY_LIQ8_WORLDS", "256"))
    gens = int(os.environ.get("ALLOY_LIQ8_GENERATIONS", "40"))
    ckpt = os.path.join(os.environ.get("ALLOY_WP9_RESULTS", "."), "wp9_liq8_checkpoint.pkl")
    out = liq8_campaign(worlds, gens, device="cuda:0", checkpoint=ckpt,
                        progress=lambda rows: write_result("wp9_liq8_progress", {"rows": rows}))
    write_result("wp9_liq8_gpu", out)
    assert out["totals"]["episodes"] >= worlds * (gens + 1)
    assert out["totals"]["leaks"] == 0 and out["totals"]["escaped"] == 0


def test_exits_classified_by_face():
    """Particles leaving the escape box are counted by face in the step they leave (then retired)."""
    cfg = default_config(world_bounds=((-0.004, -0.004, -0.004), (0.012, 0.012, 0.006)))  # top layers leave
    sc = water_column(4, 4, 6, cfg=cfg)
    s = sc.make_solver()
    mon = s.enable_leak_monitor()
    a, b = sc.model.state(), sc.model.state()
    s.step(a, b, None, None, DT)
    c = mon.counts()
    assert c["exit_top"][0] == c["escaped"][0] > 0
    assert c["exit_bottom"][0] == 0 and c["exit_side"][0] == 0 and c["leaks"][0] == 0
