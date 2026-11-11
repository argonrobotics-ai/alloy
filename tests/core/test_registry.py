# SPDX-License-Identifier: Apache-2.0
"""Solver/estimator/health/state registry round trip (SPEC §3.1)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.core import registry as R
from alloy.core.errors import RegistryError
from alloy.core.protocols import DetCapability
from alloy.testing.fakes import FakeSolver
from alloy.testing.scenes import box_on_plane


def _cap(name: str = "fake", **kw) -> R.SolverCapability:
    args = dict(name=name, factory=lambda view, kwargs: FakeSolver(view), routes={"rigid"}, owns={"body"},
                coupling=True, uniform_worlds=False, per_contact_friction=False, determinism=DetCapability.NATIVE)
    args.update(kw)
    return R.SolverCapability(**args)


class Sub(FakeSolver):
    pass


def test_solver_round_trip() -> None:
    with R.isolated_registry(empty=True):
        cap = _cap()
        R.register_solver(cap)
        R.register_solver(cap)                       # idempotent
        assert R.solver("fake") is cap and R.registered_solvers() == ("fake",)
        R.register_solver(_cap("other", routes={"rigid", "chain"}, friction_mix="harmonic"))
        assert [c.name for c in R.solvers_for_route("rigid")] == ["fake", "other"]
        assert R.registered_routes() == frozenset({"rigid", "chain"})
        assert R.solver("other").friction_mix_code == 3
        with pytest.raises(RegistryError):
            R.register_solver(_cap(per_contact_friction=True))
        R.register_solver(_cap(per_contact_friction=True), replace=True)
        with pytest.raises(LookupError):
            R.solver("missing")
    assert "fake" not in R.registered_solvers()      # isolated registry restored


def test_capability_validation() -> None:
    with pytest.raises(RegistryError):
        _cap(name="")
    with pytest.raises(RegistryError):
        _cap(friction_mix="median")
    assert isinstance(_cap().routes, frozenset) and _cap(determinism=2).determinism is DetCapability.NATIVE


def test_adapters_by_mro() -> None:
    model = box_on_plane(2).model
    solver, sub = FakeSolver(model), Sub(model)
    with R.isolated_registry(empty=True):
        assert R.health_adapter_for(solver) is None and R.state_arrays_for(solver) == {}

        def health(s, sample: R.HealthSample) -> None:
            sample.iters.fill_(3)

        R.register_health_adapter(FakeSolver, health)
        R.register_state_adapter(FakeSolver, lambda s: {"inv_mass": s.model.body_inv_mass})
        assert R.health_adapter_for(sub) is health
        sample = R.HealthSample(wp.zeros(2, dtype=wp.int32), wp.full(2, np.nan, dtype=wp.float32))
        R.health_adapter_for(solver)(solver, sample)
        assert list(sample.iters.numpy()) == [3, 3]
        assert set(R.state_arrays_for(sub)) == {"inv_mass"}
        with pytest.raises(RegistryError):
            R.register_state_adapter(solver, lambda s: {})  # an instance, not a class


def test_estimator_plugin_returns_self_else_registered() -> None:
    model = box_on_plane(1).model
    solver = FakeSolver(model)
    with R.isolated_registry(empty=True):
        assert R.estimator_for(solver) is solver      # FakeSolver implements StabilityEstimator

        class NoEst:
            pass

        marker = object()
        R.register_estimator(NoEst, lambda s: marker)
        assert R.estimator_for(NoEst()) is marker
        assert R.estimator_for(object()) is None
