# SPDX-License-Identifier: Apache-2.0
"""Film plug-in contract (CPU): registration through load_plugins, MRO lookups (capability, estimator, health, state),
the D20 air-work export (WP5 ruling 2026-11-03) and the ledger signs."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.core import registry as R
from alloy.solvers.film import FilmConfig, SolverFilm
from alloy.solvers.film.emit import add_film_mesh
from alloy.solvers.film.materials import HDPE_15UM
from alloy.solvers.film.plugin import CAPABILITY, ROUTE, _factory
from alloy.solvers.film.rigs import FilmRig, grid_mesh, pin

pytestmark = pytest.mark.compile_heavy


def _rig(worlds: int = 2, pinned: bool = True, **film) -> FilmRig:
    v, t = grid_mesh(6, 6, 5e-3, plane="xz")

    def build(b):
        p0, _ = add_film_mesh(b, v, t, HDPE_15UM, pos=(0.0, 0.0, 0.5))
        if pinned:
            pin(b, p0 + np.flatnonzero(np.abs(v[:, 2]) < 1e-9))
        return {}

    return FilmRig(build, worlds=worlds, film=FilmConfig(**film))


def test_load_plugins_registers_film() -> None:
    assert "alloy.solvers.film" in R.load_plugins()
    assert R.solver("film") == CAPABILITY and ROUTE in R.registered_routes()
    assert CAPABILITY.solver_cls is SolverFilm and CAPABILITY.determinism == R.DetCapability.WARP_LOWERING


def test_mro_lookups_and_factory() -> None:
    from alloy.coupling import register_builtins

    register_builtins()
    rig = _rig()
    s = rig.solver
    assert R.capability_for(s).name == "film"
    assert R.estimator_for(s) is not None                     # SolverVBD's ParticleEstimator by MRO
    arrays = R.state_arrays_for(s)
    assert "film_edge_rest_angle" in arrays and any(k.startswith("vbd") for k in arrays)
    maps = R.state_world_maps_for(s)
    assert maps["film_edge_rest_angle"].shape[0] == rig.model.edge_count
    built = _factory(rig.model, {"iterations": 3, "strain_iterations": 5})
    assert built.film.strain_iterations == 5 and built.iterations == 3


def test_health_adapter_reports_strain_per_world() -> None:
    rig = _rig()
    rig.run(0.02)
    W = rig.model.world_count
    sample = R.HealthSample(iters=wp.zeros(W, dtype=wp.int32, device="cpu"),
                            residual=wp.zeros(W, dtype=wp.float32, device="cpu"))
    R.health_adapter_for(rig.solver)(rig.solver, sample)
    r = sample.residual.numpy()
    assert np.all(np.isfinite(r)) and np.all(r > 0.0) and np.all(sample.iters.numpy() == 16)
    assert abs(r[0] - r[1]) < 1e-6                            # identical worlds


class _FakeHealth:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def add_external_work(self, w_ext: wp.array) -> None:
        assert w_ext.dtype == wp.float64
        self.calls.append(w_ext.numpy().copy())


def test_air_work_is_dissipation_and_exports_to_w_ext() -> None:
    rig = _rig(pinned=False)                                   # a falling sheet: air opposes the motion
    rig.run(0.05)
    led = rig.solver.ledger.numbers()
    assert np.all(led["film_air_work"] < 0.0)                 # still air: the air only takes energy out
    h = _FakeHealth()
    rig.solver.export_air_work(h)
    assert np.allclose(h.calls[0], led["film_air_work"])      # nothing exported before: all of it
    assert np.all(rig.solver.ledger.air_since.numpy() == 0.0)
    rig.run(0.01)
    rig.solver.export_air_work(h)
    total = rig.solver.ledger.numbers()["film_air_work"]
    assert np.all(h.calls[1] < 0.0) and np.allclose(h.calls[0] + h.calls[1], total)


def test_reset_restores_authored_rest_angles_per_world() -> None:
    rig = _rig(worlds=2)
    rest0 = rig.model.edge_rest_angle.numpy().copy()
    rig.model.edge_rest_angle.assign(rest0 + 0.1)
    rig.solver.reset(rig.s0, wp.array([True, False, False], dtype=wp.bool, device="cpu"))
    ew = rig.solver.edge_world().numpy()
    after = rig.model.edge_rest_angle.numpy()
    assert np.array_equal(after[ew == 0], rest0[ew == 0]) and np.allclose(after[ew == 1], rest0[ew == 1] + 0.1)
    rig.solver.reset(rig.s0)
    assert np.array_equal(rig.model.edge_rest_angle.numpy(), rest0)


def test_library_film_materials_keep_their_fold_angle() -> None:
    """SCR-SOFT-1-1: the library film materials get the clip's dead-fold yield angles, not an elastic hinge."""
    import math

    from alloy.solvers.film.materials import FILM_DEFAULTS

    assert FILM_DEFAULTS["film_hdpe"].yield_angle == pytest.approx(math.radians(73.0))
    for name in ("film_ldpe", "film_lldpe"):
        assert FILM_DEFAULTS[name].yield_angle == pytest.approx(math.radians(152.0))


def test_body_particle_list_holds_a_worlds_particles() -> None:
    """A sheet wrapped round a body puts every particle in the body's soft-contact list: 1.25x a world's particles as
    a power of two, at least Newton's 256, at most BODY_PARTICLE_BYTES over all bodies."""
    from types import SimpleNamespace as M

    from alloy.solvers.film.plugin import BODY_PARTICLE_BYTES, body_particle_buffer

    assert body_particle_buffer(M(particle_count=64 * 1536, world_count=64, body_count=128)) == 2048
    assert body_particle_buffer(M(particle_count=100, world_count=1, body_count=3)) == 256
    big = body_particle_buffer(M(particle_count=4096 * 4000, world_count=4096, body_count=4096 * 20))
    assert big * 4 * 4096 * 20 <= BODY_PARTICLE_BYTES and big >= 256
