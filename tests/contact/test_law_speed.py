# SPDX-License-Identifier: Apache-2.0
"""A5.speed (alloy.contact.law_speed): registration, spec, world layout, the scalar maths on a synthetic log, and the
stepped law on CPU with explicit impratio settings (1: MuJoCo's soft friction collapses at 0.4 mm/s; 100: Coulomb)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.bench import default_route as DR
from alloy.bench import law as L
from alloy.bench import solvers as SV
from alloy.bench.runner import RunContext, build_model, plan_worlds, run_law
from alloy.bench.twin import DeviceTwinLog
from alloy.contact import law_speed as S


def _model(twin: S.SlidingSpeedTwin):
    ctx = RunContext(S.LAW_SPEED, "cpu", 0, "mujoco", True)
    worlds = plan_worlds(S.LAW_SPEED, twin.conditions(True), ctx, None)
    return build_model(S.LAW_SPEED, twin, worlds, ctx)


class _Fixed(S.SlidingSpeedTwin):
    """The law with an explicit impratio (route defaults may change; this pins the physics under test)."""

    impratio = 1.0

    def settings_for(self, solver: str) -> SV.SolverSettings:
        routed = DR.routed_mujoco_settings(self.model)
        return SV.SolverSettings("mujoco", {**routed.kwargs, "impratio": self.impratio})


def test_registered_with_the_wp3_laws() -> None:
    assert L.discover(("alloy.contact.laws",)) == {"alloy.contact.laws": None}
    assert "A5.speed" in {s.id for s, _ in L.registered_laws()}
    spec = S.LAW_SPEED
    assert spec.solvers == ("mujoco",) and spec.producer == "WP3" and spec.suite == "analytic"
    assert spec.metrics["force_rel_err"].abs == pytest.approx(0.02)
    assert spec.dt == pytest.approx(1.25e-3) and spec.duration_s == pytest.approx(1.0)


def test_world_layout_is_an_actuated_slider_on_the_implicitfast_route() -> None:
    twin = S.SlidingSpeedTwin()
    model = _model(twin)
    assert len(twin.worlds) == len(S.LOADS) * len(S.SPEEDS) == model.world_count
    assert model.joint_dof_count == 2 * model.world_count  # slide x (driven) + slide z per world
    assert np.allclose(model.body_mass.numpy(), S.peg_mass()) and S.peg_mass() == pytest.approx(0.064)
    assert DR.routed_integrator(model) == "implicitfast"   # the slide is actuated, so not the rk4 free-body rule


def test_scalars_take_the_worst_condition() -> None:
    twin = S.SlidingSpeedTwin()
    twin.worlds = list(twin.conditions(True))
    twin.normal = np.array([S.peg_mass() * 9.81 + c.params["load"] for c in twin.worlds])
    log = DeviceTwinLog(len(twin.worlds), 10, S.DT, "cpu")
    ratio = np.array([1.0, 0.99, 0.5, 1.01])
    force = (S.MU * twin.normal * ratio)[:, None] * np.ones((1, log.n_samples))
    log.channel("force").assign(force.astype(np.float32))
    log.channel("z").assign(np.full((len(twin.worlds), log.n_samples), -1e-4, np.float32))
    out = twin.scalars(log)
    assert out["force_rel_err"] == pytest.approx([0.5], rel=1e-5)
    assert twin.monitors["penetration_um"]["load1N_v0.4mm_s"] == pytest.approx(100.0, rel=1e-4)


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.parametrize("impratio, passes", [(1.0, False), (100.0, True)])
def test_sliding_force_at_impratio(impratio: float, passes: bool) -> None:
    twin = _Fixed()
    twin.impratio = impratio
    res = run_law(S.LAW_SPEED, twin, None, device="cpu", quick=True)
    (verdict,) = res.verdicts
    ratio = res.monitors["force_over_mu_n"]
    assert (verdict.verdict == "pass") is passes, ratio
    if passes:
        assert all(abs(r - 1.0) < 2e-3 for r in ratio.values()), ratio
    else:
        assert ratio["load20N_v0.4mm_s"] < 0.1 and ratio["load20N_v4mm_s"] < 0.5, ratio  # soft friction, viscous
