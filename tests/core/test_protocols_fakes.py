# SPDX-License-Identifier: Apache-2.0
"""Every *Like protocol has a fake that satisfies it; the fakes do the work tests rely on (SPEC §6.1, D17)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.cards.types import MaterialCard, ObjectCard, Placement, Primitive, Randomization, SceneCard, Surface
from alloy.core import protocols as P
from alloy.core.quantity import Q
from alloy.precision.types import HealthFlags, Severity
from alloy.route.types import Route, SceneContext
from alloy.testing import fakes as F
from alloy.testing.scenes import box_on_plane


@pytest.fixture
def scene():
    return box_on_plane(4)


def test_every_like_protocol_has_a_satisfying_fake(scene) -> None:
    model = scene.model
    solver = F.FakeSolver(model)
    stepper = F.FakeStepper(solver)
    fakes = {
        P.ContactLayerLike: F.FakeContactLayer(model), P.GapMonitorLike: F.FakeGapMonitor(),
        P.HealthLike: F.FakeHealth(model), P.StepperLike: stepper, P.TopologyLike: F.FakeTopology(4),
        P.CardLibraryLike: F.FakeCardLibrary(), P.RouterLike: F.FakeRouter(), P.SceneAssemblerLike: F.FakeAssembler(),
        P.SimulationLike: F.FakeSimulation(model, stepper), P.PeriodGraphLike: F.FakePeriodGraph(lambda: None),
        P.StabilityEstimator: solver, P.AlloySolverPlugin: solver,
    }
    likes = {obj for name, obj in vars(P).items() if name.endswith("Like")}
    assert likes <= set(fakes)
    for proto, fake in fakes.items():
        assert isinstance(fake, proto), proto.__name__
    assert not isinstance(object(), P.StepperLike)


def test_fake_simulation_latches_seeded_blowup(scene) -> None:
    solver = F.FakeSolver(scene.model, F.FakeSolver.Config(blowup_speed=5.0))
    health = F.FakeHealth(scene.model)
    sim = F.FakeSimulation(scene.model, F.FakeStepper(solver), health=health,
                           contact_layer=F.FakeContactLayer(scene.model), gap_monitor=F.FakeGapMonitor(0, 4),
                           topology=F.FakeTopology(4))
    qd = sim.state.body_qd.numpy()
    qd[2, 0] = 6.0
    sim.state.body_qd.assign(qd)
    sim.step_period()
    rep = sim.report()
    assert rep["steps"] == 16 and rep["t"] == 0.02
    assert rep["health"].invalid_worlds == (2,)
    assert rep["health"].flags[2] & int(HealthFlags.SPEED_FAIL) and rep["health"].flags[2] & int(HealthFlags.LATCHED)
    assert np.all(np.isfinite(sim.state.body_q.numpy()))
    others = sim.state.body_q.numpy()[[0, 1, 3]]
    assert np.allclose(others[:, 2], 0.1)                  # resting worlds untouched (floor at 0.1 m)


def test_fake_health_inject_and_nonfinite(scene) -> None:
    health = F.FakeHealth(scene.model)
    health.inject(1, HealthFlags.PEN_WARN, Severity.DEGRADED, step=7)
    state = scene.model.state()
    q = state.body_q.numpy()
    q[3, 2] = np.nan
    state.body_q.assign(q)
    health.update(9, state, state, None)
    rep = health.poll()
    assert list(rep.severity) == [0, 1, 0, 2] and rep.first_bad_step[1] == 7 and rep.first_bad_step[3] == 9
    assert rep.flags[3] & int(HealthFlags.NONFINITE_Q)


def test_fake_solver_is_deterministic_and_batch_invariant() -> None:
    def run(n: int) -> np.ndarray:
        sc = box_on_plane(n, gap=0.5)
        sim = F.FakeSimulation(sc.model, F.FakeStepper(F.FakeSolver(sc.model, F.FakeSolver.Config(damping=0.01))))
        sim.step_period()
        return sim.state.body_q.numpy()

    one, many = run(1), run(16)
    assert all(row.tobytes() == one[0].tobytes() for row in many)


def test_fake_solver_observables(scene) -> None:
    solver = F.FakeSolver(scene.model)
    obs = solver.observables({"residual", "iterations"})
    state = scene.model.state()
    solver.step(state, scene.model.state(), None, None, 1e-3, observables=obs)
    assert list(obs.iterations.numpy()) == [3] * 4 and np.all(obs.residual.numpy() == 0.0)


def test_fake_library_router_assembler() -> None:
    mu = Q(0.4, "default")
    steel = MaterialCard("steel", Q(7800.0, "catalog", note="handbook"), Surface(mu, mu, Q(0.5, "default")))
    peg = ObjectCard("peg", Primitive("box", (0.01, 0.02, 0.05)), "steel")
    lib = F.FakeCardLibrary({"steel": steel}, {"peg": peg})
    res = lib.resolve(peg, Placement("peg", (0, 0, 0, 0, 0, 0, 1), (Randomization("loads.grip_force", 5.0, 30.0),)))
    assert res.extents == (0.1, 0.04, 0.02) and res.worst_case == {"loads.grip_force": 30.0}
    assert "density" not in res.default_leaves and "loads.grip_force" in res.default_leaves
    assert lib.put(peg, reason="test").name == "peg.yaml"
    ctx = SceneContext({}, 1, frozenset({"rigid"}))
    assert F.FakeRouter().route(res, ctx).route is Route.RIGID
    built = F.FakeAssembler().assemble(SceneCard((), ()), "cpu", world_count=2, world_id_offset=8)
    assert built.world_count == 2 and built.global_world_id(1) == 9 and built.pair_table.material_index("default") == 0
    assert isinstance(built.world_origin, wp.array)


def test_synthetic_rig_is_deterministic() -> None:
    a = F.synthetic_rig("R3.incline_friction", {"mu_static": 0.5}, 0.01, seed=3)
    b = F.synthetic_rig("R3.incline_friction", {"mu_static": 0.5}, 0.01, seed=3)
    c = F.synthetic_rig("R3.incline_friction", {"mu_static": 0.5}, 0.01, seed=4)
    assert a.sha256 == b.sha256 != c.sha256
    assert len(a.split("calib")) == 3 and len(a.split("valid")) == 2
    assert np.isclose(np.mean([t.scalars["slip_angle"] for t in a.trials]), np.arctan(0.5), atol=0.02)
