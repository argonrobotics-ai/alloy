# SPDX-License-Identifier: Apache-2.0
"""Resting-jitter rule (SCR-SC-C2, solve_driver.py): the per-world density tolerance follows
min(max_density_error, v dt / H) from the hydrostatic head, v = max(rest_jitter_speed, ratio x |mean velocity|), is
floored, and is off with None or without gravity. Quick CPU runs on primed columns; the law itself is
A13.resting_jitter (test_laws.py)."""

from __future__ import annotations

import dataclasses

from alloy._vendor import newton
import numpy as np
import pytest

from alloy.solvers.sph.build import add_box_tank, add_fluid_block
from alloy.solvers.sph.config import DFSPHConfig, register_custom_attributes
from alloy.solvers.sph.solver import SolverDFSPH

DT = 1.25e-3


def _column(nz: int, cfg: DFSPHConfig, gravity: float = -9.81, worlds: tuple[int, ...] = ()) -> SolverDFSPH:
    s_ = cfg.spacing
    b = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    register_custom_attributes(b)
    for n in worlds or (nz,):
        b.begin_world()
        add_box_tank(b, (0.0, 0.0, 0.0), (6 * s_, 6 * s_, (n + 6) * s_), wall=cfg.h)
        add_fluid_block(b, cfg, (0.0, 0.0, 0.0), (6, 6, n))
        b.end_world()
    m = b.finalize(device="cpu")
    s = SolverDFSPH(m, cfg)
    a, c = m.state(), m.state()
    s.prime(a, DT)
    for _ in range(20):  # past the few-step transient after prime
        s.step(a, c, None, None, DT)
        a, c = c, a
    return s


def test_tolerance_follows_the_depth_per_world():
    cfg = DFSPHConfig(particle_radius=2.0e-3, rest_jitter_flow_ratio=0.0)
    s = _column(0, cfg, worlds=(8, 25))
    tol = s.report().density_tol
    expected = cfg.rest_jitter_speed * DT / (np.array([8, 25]) * cfg.spacing)
    assert np.allclose(tol, expected, rtol=0.15), (tol, expected)  # head ~ 1.08 depth (dynamic part)


def test_flow_allowance_uses_the_mean_velocity_before_the_step():
    """v = max(rest_jitter_speed, ratio U), U = |mean velocity| of the world's liquid in the state the step starts
    from (not gravity's g dt): bulk flow keeps a looser tolerance; a sideways kick of the whole column."""
    cfg = DFSPHConfig(particle_radius=2.0e-3)
    s = _column(0, cfg, worlds=(8, 25))
    a, c = s.model.state(), s.model.state()
    s.prime(a, DT)
    v = a.particle_qd.numpy()
    v[:, 0] += 0.02  # 20 mm/s sideways
    a.particle_qd.assign(v)
    v = a.particle_qd.numpy()[: s.model.particle_count].astype(np.float64)
    world = s.model.particle_world.numpy()[: s.model.particle_count]
    u = np.array([np.linalg.norm(v[world == w].mean(0)) for w in (0, 1)])
    s.step(a, c, None, None, DT)
    head = s._head.numpy().astype(np.float64)
    depth = 1.5 * head[:, 1] / (head[:, 0] * DT * DT * 9.81)
    expected = np.minimum(cfg.max_density_error,
                          np.maximum(cfg.rest_jitter_speed, cfg.rest_jitter_flow_ratio * u) * DT / depth)
    assert np.all(u > 0.015) and np.all(expected < cfg.max_density_error), (u, expected)
    assert np.allclose(s.report().density_tol, expected, rtol=1e-3), (s.report().density_tol, expected)


def test_off_without_gravity_with_none_and_floored():
    cfg = DFSPHConfig(particle_radius=2.0e-3)
    assert _column(8, cfg, gravity=0.0).report().density_tol[0] == pytest.approx(cfg.max_density_error)
    off = dataclasses.replace(cfg, rest_jitter_speed=None)
    assert _column(8, off).report().density_tol[0] == pytest.approx(cfg.max_density_error)
    floor = dataclasses.replace(cfg, min_density_error=2.0e-4)
    assert _column(8, floor).report().density_tol[0] == pytest.approx(2.0e-4)


def test_config_validation():
    with pytest.raises(ValueError, match="rest_jitter_speed"):
        DFSPHConfig(rest_jitter_speed=0.0).validate()
    with pytest.raises(ValueError, match="rest_jitter_flow_ratio"):
        DFSPHConfig(rest_jitter_flow_ratio=1.5).validate()
    with pytest.raises(ValueError, match="min_density_error"):
        DFSPHConfig(min_density_error=2.0e-3).validate()
