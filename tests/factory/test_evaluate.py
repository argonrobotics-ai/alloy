# SPDX-License-Identifier: Apache-2.0
"""evaluate_plugin and the §3.8 contract checks: the template passes, broken plug-ins fail the right check."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from alloy.bully.direct import DirectSim
from alloy.bully.types import BullyTarget, SearchConfig
from alloy.core.registry import isolated_registry
from alloy.factory.contract import ContractReport, dynamic_checks, static_checks
from alloy.factory.evaluate import evaluate_plugin, load_manifest, verdict_from_json, verdict_to_json
from alloy.factory.template import scenes
from alloy.factory.template.solver import SolverTemplate

MANIFEST = Path("alloy/factory/template/manifest.json")
SMALL = SearchConfig(worlds=8, generations=1, horizon_s=0.5, budget_gpu_h=0.01)


@pytest.fixture
def registry():
    with isolated_registry():
        import alloy.factory.template.plugin as plugin  # noqa: F401  (registers on first import)
        from alloy.core.registry import register_solver

        register_solver(plugin.CAPABILITY, replace=True)
        yield


def test_template_passes_the_contract(registry):
    v = evaluate_plugin(load_manifest(MANIFEST), "cpu", cfg=SMALL, replay_runs=2)
    assert v.contract_ok, v.reasons
    assert not any(r.startswith("contract") for r in v.reasons)
    assert v.findings == ()  # B0 + B1 found nothing on the template scene
    # only producers that have not merged yet (WP6 law runner, WP5 lint_atomics) may block it, and they are reported
    assert all("not_run" in r for r in v.reasons), v.reasons
    assert v.ship == (not v.reasons)
    if importlib.util.find_spec("alloy.bench.law") is not None:
        assert [r.verdict for r in v.law_results] == ["pass"]
    back = verdict_from_json(verdict_to_json(v))
    assert back.reasons == v.reasons and back.plugin == "template"


class _Allocates(SolverTemplate):
    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        wp.zeros(4, dtype=float, device=self.model.device)  # contract violation: allocation inside step
        super().step(state_in, state_out, control, contacts, dt, observables=observables)


class _ReadsBack(SolverTemplate):
    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        super().step(state_in, state_out, control, contacts, dt, observables=observables)
        self.model.particle_q.numpy()  # contract violation: host readback inside step


class _ResetsAll(SolverTemplate):
    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, None, flags)  # ignores the mask: resets every world


class _HiddenCounter(SolverTemplate):
    """Lists no state, ignores the reset mask, and its step depends on a counter reset never clears."""

    def alloy_state_arrays(self, state):
        return {}

    def alloy_state_worlds(self):
        return {}

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        self._n = getattr(self, "_n", 0) + 1
        super().step(state_in, state_out, control, contacts, dt * (1.0 + 1e-3 * self._n), observables=observables)


class _NoisyAttrs(SolverTemplate):
    @staticmethod
    def register_custom_attributes(builder):
        from alloy._vendor import newton

        builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
            name="leak", dtype=wp.float32, default=0.0, frequency=newton.Model.AttributeFrequency.PARTICLE,
            namespace="other"))


def _target(cls) -> BullyTarget:
    base = scenes.bully_target()

    def make_sim(worlds: int, device: str) -> DirectSim:
        model = scenes.drop_model(worlds, device)
        return DirectSim(model, cls(model), control_dt=0.02, substeps=16, collide=False)

    return BullyTarget(base.name, None, make_sim, base.actions, action_kind="particle_f", action_index=(0, 1, 2))


@pytest.mark.parametrize("cls,check", [(_Allocates, "no_alloc_in_step"), (_ReadsBack, "no_readback_in_step"),
                                       (_ResetsAll, "per_world_reset")])
def test_dynamic_violations_are_caught(registry, cls, check):
    rep = ContractReport()
    dynamic_checks(cls, _target(cls), "cpu", rep)
    assert rep.checks[check] is not None, rep.checks
    others = {k: v for k, v in rep.checks.items() if k != check and v is not None}
    assert others == {}, others


def test_hidden_state_cannot_pass_with_empty_state_arrays(registry):
    rep = ContractReport()
    dynamic_checks(_HiddenCounter, _target(_HiddenCounter), "cpu", rep)
    for check in ("per_world_reset", "repeat_r1", "state_arrays_sufficient"):
        assert rep.checks[check] is not None, (check, rep.checks)


def test_static_violations_are_caught(registry):
    rep = ContractReport()
    static_checks(_NoisyAttrs, rep)
    assert "other:leak" in (rep.checks["attribute_namespace"] or "")

    class _NotCoupled:  # neither SolverBase nor CouplingInterface, no members
        plugin_name = "x"

    rep2 = ContractReport()
    static_checks(_NotCoupled, rep2)
    assert rep2.checks["bases"] and rep2.checks["members"] and rep2.checks["capability"]


def test_template_physics_rests_on_the_ground():
    sim = scenes.make_sim(1, "cpu")
    for _ in range(50):  # 1 s
        sim.step_period()
    z = sim.state.particle_q.numpy()[:, 2]
    # resting overlap m g / ke = 0.05 * 9.81 / 1e4 = 49 um below the radius
    np.testing.assert_allclose(z, scenes.RADIUS - 0.05 * 9.81 / 1e4, atol=5e-6)
