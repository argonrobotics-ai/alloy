# SPDX-License-Identifier: Apache-2.0
"""SolverDFSPH plug-in contract (SPEC §3.8 items 1-6) on CPU."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp
from alloy._vendor.newton.solvers import SolverBase

import alloy.solvers.sph  # noqa: F401  (registers the capability)
from alloy.core.newton_compat import CouplingInterface
from alloy.core.protocols import AlloySolverPlugin, DetCapability, StabilityEstimator
from alloy.core.registry import HealthSample, estimator_for, health_adapter_for, solver, state_arrays_for
from alloy.solvers.sph.config import DFSPHConfig, snap_down, stable_dt
from alloy.solvers.sph.solver import SolverDFSPH
from alloy.solvers.sph.types import SPHReport
from alloy.testing.sph import default_config, water_column


@pytest.fixture(scope="module")
def two_world_scene():
    return water_column(4, 4, 6, worlds=2, fills=[6, 4])


def test_capability_and_protocols(two_world_scene):
    s = two_world_scene.make_solver()
    assert isinstance(s, SolverBase) and isinstance(s, CouplingInterface)
    assert isinstance(s, AlloySolverPlugin) and isinstance(s, StabilityEstimator)
    assert estimator_for(s) is s
    cap = solver("sph")
    assert cap.routes == frozenset({"liquid_sph"}) and cap.determinism == DetCapability.NATIVE
    assert cap.coupling and "particle" in cap.owns and cap.emitter is not None
    built = cap.factory(two_world_scene.model, {"particle_radius": 1.0e-3, "xsph_c": 0.02})
    assert isinstance(built, SolverDFSPH) and built.config.xsph_c == 0.02
    with pytest.raises(TypeError):
        cap.factory(two_world_scene.model, {"bogus": 1})


def test_state_arrays_worlds_capacity_and_report(two_world_scene):
    s = two_world_scene.make_solver()
    st0, st1 = two_world_scene.model.state(), two_world_scene.model.state()
    s.step(st0, st1, None, None, 1.0e-3)
    arrays = s.alloy_state_arrays(st1)
    assert {"sph.kappa_v", "sph.kappa_d", "sph.phase", "particle_flags"} <= set(arrays)
    assert set(s.alloy_state_worlds()) <= set(arrays)
    assert set(state_arrays_for(s)) == set(arrays)
    assert set(s.alloy_capacity_report()) == {"neighbor_overflow", "emitter_overflow"}
    r = s.report()
    assert isinstance(r, SPHReport) and r.density_err.shape == (2,) and r.unconverged.dtype == bool
    assert np.all(r.mass_active > 0.0) and np.all(r.escaped == 0)
    sample = HealthSample(iters=wp.zeros(2, dtype=wp.int32), residual=wp.zeros(2, dtype=wp.float32))
    health_adapter_for(s)(s, sample)
    assert np.all(sample.iters.numpy() == r.iters_density + r.iters_divergence)


def test_step_is_graph_capturable_and_matches_eager():
    """Fixed capacity, no host sync or allocation inside step: a CPU capture replays bitwise like eager stepping."""
    sc = water_column(4, 4, 6)
    s = sc.make_solver()
    a, b = sc.model.state(), sc.model.state()
    s.step(a, b, None, None, 1.0e-3)
    kd, kv = wp.clone(s._kd), wp.clone(s._kv)
    g0, g1 = sc.model.state(), sc.model.state()
    e0, e1 = sc.model.state(), sc.model.state()
    for st in (g0, e0):
        wp.copy(st.particle_q, b.particle_q)
        wp.copy(st.particle_qd, b.particle_qd)
    with wp.ScopedCapture(device="cpu") as cap:
        s.step(g0, g1, None, None, 1.0e-3)
    wp.copy(s._kd, kd)
    wp.copy(s._kv, kv)
    wp.capture_launch(cap.graph)
    graph_q = g1.particle_q.numpy().copy()
    wp.copy(s._kd, kd)
    wp.copy(s._kv, kv)
    s.step(e0, e1, None, None, 1.0e-3)
    assert graph_q.tobytes() == e1.particle_q.numpy().tobytes()


def test_stable_dt_energy_and_choose_dt(two_world_scene):
    sc = two_world_scene
    s = sc.make_solver()
    st = sc.model.state()
    v = np.zeros((sc.model.particle_count, 3), np.float32)
    w = sc.model.particle_world.numpy()
    v[w == 1, 0] = 2.0
    st.particle_qd.assign(v)
    out = wp.zeros(2, dtype=wp.float32)
    s.alloy_stable_dt(st, None, out)
    cfg = s.config
    assert out.numpy()[0] == pytest.approx(cfg.dt_max)
    fast = dataclasses.replace(cfg, cfl=cfg.speed_margin * cfg.cfl)
    assert out.numpy()[1] == pytest.approx(stable_dt(fast, 2.0), rel=1e-6), \
        "speed_margin 0.25 of the CFL step while the liquid moves fast: 0.25 x cfl x 2r / v (COUPLE-9)"
    z_ref = wp.array([0.0, 0.01], dtype=wp.float64)
    e = wp.zeros(2, dtype=wp.float64)
    s.alloy_energy(st, z_ref, e)
    m = sc.model.particle_mass.numpy().astype(np.float64)
    q = st.particle_q.numpy().astype(np.float64)
    ref = [np.sum(m[w == k] * (0.5 * (v[w == k].astype(np.float64) ** 2).sum(1) + 9.81 * (q[w == k, 2] - zr)))
           for k, zr in ((0, 0.0), (1, 0.01))]
    assert np.allclose(e.numpy(), ref, rtol=1e-6)
    assert snap_down(7.0e-4, 0.02) == pytest.approx(0.02 / 32)
    assert snap_down(1.0, 0.02) == 0.02 and snap_down(1.0e-9, 0.02) == pytest.approx(0.02 / 1024)


def test_reset_clears_masked_worlds_only(two_world_scene):
    sc = two_world_scene
    s = sc.make_solver()
    a, b = sc.model.state(), sc.model.state()
    s.prime(a, 1.0e-3)
    kd = s._kd.numpy().copy()
    assert np.all(kd[: sc.model.particle_count] >= 0.0) and kd.max() > 0.0
    s.reset(a, world_mask=wp.array([0, 1, 0], dtype=wp.bool))
    w = sc.model.particle_world.numpy()
    after = s._kd.numpy()[: sc.model.particle_count]
    assert np.array_equal(after[w == 0], kd[: sc.model.particle_count][w == 0])
    assert np.all(after[w == 1] == 0.0)


def test_config_validation_and_viscosity_choice():
    with pytest.raises(ValueError):
        DFSPHConfig(particle_radius=-1.0).validate()
    with pytest.raises(ValueError):
        DFSPHConfig(jacobi_omega=0.9).validate()
    cfg = default_config()
    assert dataclasses.replace(cfg, viscosity="none").viscosity == "none"
    sc = water_column(3, 3, 3, cfg=cfg)
    assert sc.make_solver().viscosity == "xsph"  # water 1e-3 Pa·s < 1e-2 Pa·s


def test_tension_attributes_need_the_config():
    """sph:surface_tension / sph:adhesion act only with surface_tension="akinci2013" (tension.py); with "none" they
    are refused, never silently dropped (SCR-SC1-2 ruling)."""
    DFSPHConfig(surface_tension="akinci2013").validate()
    for name in ("surface_tension", "adhesion"):
        sc = water_column(3, 3, 3)
        a = getattr(sc.model.sph, name).numpy()
        a[0] = 0.07
        getattr(sc.model.sph, name).assign(a)
        with pytest.raises(ValueError, match=f"sph:{name}"):
            sc.make_solver()
