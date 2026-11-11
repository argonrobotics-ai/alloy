# SPDX-License-Identifier: Apache-2.0
"""Interface-review round 1 (SCRs filed against wp0@330f165): each accepted change has a test here."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import warp as wp

from alloy.cards.types import Elastic, MaterialCard, ObjectCard, Placement, Primitive, Randomization, SceneCard, Surface
from alloy.core import errors as E
from alloy.core import protocols as P
from alloy.core import registry as R
from alloy.core.errors import SnapshotError
from alloy.core.protocols import DetCapability
from alloy.core.quantity import Q
from alloy.core.snapshot import WorldSnapshot, mask_from_worlds
from alloy.testing import fakes as F
from alloy.testing.scenes import box_on_plane


def _steel(youngs: float = 2.0e11) -> MaterialCard:
    mu = Q(0.4, "default")
    return MaterialCard("steel", Q(7800.0, "catalog", note="handbook"), Surface(mu, mu, Q(0.5, "default")),
                        elastic=Elastic(Q(youngs, "catalog", note="handbook"), Q(0.3, "default")))


def test_card_fake_paths_hash_worst_case() -> None:
    """SCR-WP1-1/2/3/5."""
    peg = ObjectCard("peg", Primitive("box", (0.01, 0.02, 0.05)), "steel")
    lib = F.FakeCardLibrary({"steel": _steel()}, {"peg": peg})
    rand = (Randomization("material.elastic.youngs", 1e11, 3e11), Randomization("loads.grip_force", 5.0, 30.0))
    res = lib.resolve(peg, Placement("peg", (0, 0, 0, 0, 0, 0, 1), rand))
    assert res.worst_case == {"material.elastic.youngs": 1e11, "loads.grip_force": 30.0}
    assert {"material.surface.contact_stiffness", "material.surface.restitution",
            "material.elastic.poisson"} <= set(res.default_leaves)
    assert res.material.surface.contact_stiffness == Q(2.0e6, "default")
    assert all("[" not in leaf for leaf in res.default_leaves)
    other = F.FakeCardLibrary({"steel": _steel(1.9e11)}, {"peg": peg}).resolve(peg, Placement("peg", (0,) * 7, rand))
    assert other.card_hash != res.card_hash                     # material edits change the hash
    assert lib.resolve(peg, Placement("peg", (1,) * 7, rand)).card_hash == res.card_hash   # pose does not
    assert F.default_leaves((Q(1.0, "default"), Q(2.0, "default")), "x") == ("x.0", "x.1")


def test_stepper_rollback_and_observers() -> None:
    """SCR-WP4-1, SCR-WP7-2, SCR-WP5-6."""
    sc = box_on_plane(2, gap=0.3)
    stepper = F.FakeStepper(F.FakeSolver(sc.model, F.FakeSolver.Config(floor_z=None)))
    seen: list[float] = []
    gap_healths: list[object] = []

    class Gap(F.FakeGapMonitor):
        def update(self, state_prev, state, health=None):
            gap_healths.append(health)

    health = F.FakeHealth(sc.model)
    sim = F.FakeSimulation(sc.model, stepper, health=health, gap_monitor=Gap())
    sim.add_step_observer(lambda s_in, s_out, control, contacts, dt: seen.append(dt))
    stepper.request_rollbacks = 1
    sim.step_period()
    assert sim.rollbacks == 1 and stepper.rollbacks == 1
    assert sim.steps == 16 and sim.clock.ticks == 1024           # the re-run replaced the rolled-back period
    assert len(seen) == 32                                      # observers saw both runs
    z = sim.state.body_q.numpy()[:, 2]
    assert np.allclose(z, 0.4 - 0.5 * 9.81 * 0.02**2, atol=1e-3)   # one period of free fall, not two
    assert all(h is health.health for h in gap_healths)
    assert isinstance(sim, P.SimulationLike) and sim.state_arrays() == {} and sim.state_world_maps() == {}


def test_topology_fake_arrays() -> None:
    """SCR-WP10-2."""
    topo = F.FakeTopology(3, body_count=5)
    assert isinstance(topo, P.TopologyLike)
    assert topo.step_events.numpy().tolist() == [0, 0, 0] and topo.body_active.numpy().tolist() == [1] * 5


def test_topology_adjust_contacts_is_optional() -> None:
    """SCR-SC5-1: ``ContactAdjuster`` sits beside ``TopologyLike``; ``adjust_contacts`` is the no-op default."""

    class _NoBlades:                     # a TopologyLike without the capability (WP10's TopologyController today)
        step_events = body_active = None

        def pre_step(self, state, contacts, dt): ...

        def apply(self, state): ...

        def ledger(self): ...

    plain, topo = _NoBlades(), F.FakeTopology(2)
    assert isinstance(plain, P.TopologyLike) and not isinstance(plain, P.ContactAdjuster)
    assert isinstance(topo, P.TopologyLike) and isinstance(topo, P.ContactAdjuster)
    for t in (plain, None, topo):
        P.adjust_contacts(t, None, None)
    assert topo.adjusts == 1
    assert list(inspect.signature(P.ContactAdjuster.adjust_contacts).parameters) == ["self", "state", "contacts"]


def test_assembler_world_ids() -> None:
    """SCR-WP8-2."""
    built = F.FakeAssembler().assemble(SceneCard((), ()), "cpu", world_count=3, world_ids=[7, 2, 11])
    assert [built.global_world_id(i) for i in range(3)] == [7, 2, 11]
    with pytest.raises(ValueError):
        F.FakeAssembler().assemble(SceneCard((), ()), "cpu", world_count=2, world_ids=[1])
    assert "world_ids" in inspect.signature(P.SceneAssemblerLike.assemble).parameters


def test_capability_for_and_world_maps() -> None:
    """SCR-WP5-1, SCR-WP4-3, SCR-WP9-4."""
    model = box_on_plane(2).model
    solver = F.FakeSolver(model)
    with R.isolated_registry(empty=True):
        assert R.capability_for(solver) is None
        cap = R.SolverCapability("fake", lambda v, kw: F.FakeSolver(v), {"rigid"}, {"body"}, True, False, False,
                                 DetCapability.NATIVE, solver_cls=F.FakeSolver)
        R.register_solver(cap)
        assert R.capability_for(solver) is cap
        R.register_state_adapter(F.FakeSolver, lambda s: {"m": s.model.body_inv_mass},
                                 world_map=lambda s: {"m": s.model.body_world})
        assert set(R.state_world_maps_for(solver)) == {"m"}
        R.register_state_adapter(F.FakeSolver, lambda s: {})
        assert R.state_world_maps_for(solver) == {}
    maps = solver.alloy_state_worlds()
    state = model.state()
    snap = WorldSnapshot(model, state, solver.alloy_state_arrays(state), extra_world=maps)
    assert snap.whole_arrays == ()


def test_load_plugins_registers_sph() -> None:
    """SCR-WP9-5: plug-ins are discovered under alloy.solvers, not hard-coded by WP2/WP7."""
    names = R.plugin_packages()
    assert "alloy.solvers.sph" in names and all(".contrib" not in n for n in names)
    loaded = R.load_plugins()
    assert "alloy.solvers.sph" in loaded and R.load_plugins() == loaded      # idempotent
    assert "liquid_sph" in R.registered_routes()


def test_snapshot_host_round_trip() -> None:
    """SCR-WP5-4: save in one 'simulation', load into a fresh one, restore."""
    a, b = box_on_plane(2, gap=0.1).model, box_on_plane(2, gap=0.1).model
    sa, sb = a.state(), b.state()
    q = sa.body_q.numpy()
    q[:, 0] = [1.5, -2.5]
    sa.body_q.assign(q)
    snap_a = WorldSnapshot(a, sa)
    snap_a.save(sa)
    words = snap_a.to_host()
    snap_b = WorldSnapshot(b, sb)
    snap_b.from_host(words)
    snap_b.restore(sb, mask_from_worlds([1], 2, b.device))
    assert sb.body_q.numpy()[1].tobytes() == q[1].tobytes() and sb.body_q.numpy()[0, 0] == 0.0
    bad = dict(words)
    bad.pop(next(iter(bad)))
    with pytest.raises(SnapshotError):
        snap_b.from_host(bad)
    wrong = {k: v[:-1] for k, v in words.items()}
    with pytest.raises(SnapshotError):
        snap_b.from_host(wrong)


def test_mask_on_another_device_is_rejected() -> None:
    """Found on the L4: a CPU mask with a CUDA snapshot silently restored the wrong worlds."""
    if not wp.is_cuda_available():
        pytest.skip("needs a CPU and a CUDA device")
    model = box_on_plane(2, device="cuda:0").model
    snap = WorldSnapshot(model, model.state())
    with pytest.raises(SnapshotError, match="device"):
        snap.restore(model.state(), mask_from_worlds([1], 2, "cpu"))


def test_new_errors() -> None:
    """SCR-WP7-3, SCR-WP8-1, SCR-WP13-2."""
    for name in ("TraceError", "BullyError", "FactoryError", "GraphCaptureError", "LaunchError"):
        assert issubclass(getattr(E, name), E.AlloyError)
    err = E.PluginContractError("contract", ["no estimator"])
    assert isinstance(err, E.FactoryError) and err.failures == ["no estimator"]
    assert E.SandboxViolation("x", ["a.py"]).paths == ["a.py"]
    gap = E.HeldOutGapError("refused", {"mu": (0.3, 0.2)})               # SCR-WP11-1
    assert isinstance(gap, E.FitError) and issubclass(E.SplitError, E.FitError) and gap.gaps == {"mu": (0.3, 0.2)}


def test_new_fields_and_signatures() -> None:
    """SCR-WP1-7, WP2-1, WP2-4, WP5-2, WP6-1/2, WP13-1."""
    from alloy.bench.types import SimTwin, TwinLog
    from alloy.cards.types import Features
    from alloy.parallel.types import Trace
    from alloy.precision.types import FramePolicy
    from alloy.route.types import SceneContext
    from alloy.scene.types import PairTable

    assert Features().bend_stiffness is None and Features().twist_stiffness is None
    ctx = SceneContext({}, 0, frozenset())
    assert ctx.partners == {} and ctx.needs_l1 is False and ctx.other_dynamic_entries == 0
    assert FramePolicy().far_invalid is False
    t = Trace("h", 0.02, {"q": np.zeros((3, 2))}, "src", "sha")
    assert t.columns == {} and t.initial == {} and t.periods == 3
    assert "state" in inspect.signature(PairTable.apply).parameters
    assert "solver" in inspect.signature(SimTwin.make_solver).parameters and hasattr(SimTwin, "conditions")
    assert {"channel", "accumulator", "column", "times"} <= set(dir(TwinLog))
    built = F.FakeAssembler().assemble(SceneCard((), ()), "cpu", world_count=1)
    assert built.pair_table.model is None and built.pair_table.override_k is None and built.world_ids is None


def test_world_map_protocol_members() -> None:
    """SCR-WP4-3/WP9-4/WP5-3 share one convention: every extra-array provider also gives item->world maps."""
    assert hasattr(P.StepperLike, "state_world_maps") and hasattr(P.AlloySolverPlugin, "alloy_state_worlds")
    assert hasattr(P.SimulationLike, "state_world_maps") and "model" in P.SimulationLike.__annotations__
    assert isinstance(F.FakeSolver(box_on_plane(1).model), P.AlloySolverPlugin)
    assert wp.types.type_size_in_bytes(wp.int8) == 1


def test_shape_work_source_is_optional() -> None:
    """SCR-WP9-6: ``alloy_shape_work`` is an optional capability; plug-ins without it still meet the §3.8 contract."""
    solver = F.FakeSolver(box_on_plane(1).model)
    assert isinstance(solver, P.AlloySolverPlugin) and not isinstance(solver, P.ShapeWorkSource)

    class _Walls:
        def alloy_shape_work(self, out: wp.array) -> None:
            out.zero_()

    assert isinstance(_Walls(), P.ShapeWorkSource)
    assert list(inspect.signature(P.ShapeWorkSource.alloy_shape_work).parameters) == ["self", "out"]


def test_sph_report_density_tol_is_optional() -> None:
    """SCR-WP9-8: ``SPHReport.density_tol`` ([W], the resting-jitter rule's effective tolerance) defaults to None."""
    import dataclasses

    from alloy.solvers.sph.types import SPHReport

    fields = {f.name: f for f in dataclasses.fields(SPHReport)}
    assert fields["density_tol"].default is None and list(fields)[-1] == "density_tol"
