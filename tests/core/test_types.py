# SPDX-License-Identifier: Apache-2.0
"""Every WP0 types.py imports, every dataclass's hints resolve at runtime, and the shared numbers agree (D2, D3)."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import typing

import numpy as np
import pytest

TYPE_MODULES = [f"alloy.{p}.types" for p in ("cards", "route", "scene", "contact", "coupling", "precision",
                                              "parallel", "topology", "bench", "bully", "factory", "fit",
                                              "solvers.sph")]

NAMED_IN_SPEC = {  # §7 WP0: every type the spec names is defined with its fields
    "alloy.bully.types": ["EpisodeResult", "ActionSpace", "BullyLimits", "BullyTarget", "Finding", "ReplayResult",
                          "SearchConfig", "BullySeverity"],
    "alloy.parallel.types": ["Trace", "Pack", "PackStats", "PackSpec", "PackPlan", "WorldProfile", "LadderRung",
                             "StructurePlan", "IslandPlan", "StructureClass", "WorldRequest", "PackPolicy"],
    "alloy.precision.types": ["RepeatResult", "PrecisionObservables", "StepHealth", "HealthReport", "CheckPolicy",
                              "PrecisionConfig", "HealthFlags", "Severity", "Determinism", "GpuTier"],
    "alloy.solvers.sph.types": ["SPHReport"],
    "alloy.route.types": ["RodCard", "Route", "RouteDecision", "RouterConfig", "SceneContext"],
    "alloy.cards.types": ["CardIssue", "Attachment", "EmitterSpec", "BladeSpec", "PrecisionFeature", "ObjectCard",
                          "MaterialCard", "PairCard", "SceneCard", "ResolvedObject", "LoadEnvelope"],
    "alloy.contact.types": ["AlloyContactSidecar", "PrecisionPairSpec", "GapReport", "ContactClass", "ContactFlags"],
    "alloy.core.registry": ["HealthSample", "SolverCapability"],
    "alloy.scene.types": ["BuiltScene", "PairTable", "RouteReport", "EntrySpec", "CouplerChoice", "ObjectHandle"],
    "alloy.coupling.types": ["CouplingReport", "SchedulerPolicy", "AlloyEntry", "PairPolicy", "SleepPolicy"],
    "alloy.topology.types": ["BondSet", "Blade", "TopologyLedger", "TopologyBudget", "BondKind"],
    "alloy.bench.types": ["LawSpec", "LawResult", "ScoreRow", "RigDataset", "SimTwin", "Tolerance", "MetricVerdict"],
    "alloy.factory.types": ["AgentRunner", "PluginManifest", "FactoryVerdict"],
    "alloy.fit.types": ["FitParam", "FitProblem", "FitConfig", "FitResult"],
}


@pytest.mark.parametrize("module", TYPE_MODULES)
def test_types_import_and_hints_resolve(module: str) -> None:
    mod = importlib.import_module(module)
    for name, obj in vars(mod).items():
        if inspect.isclass(obj) and obj.__module__ == module and dataclasses.is_dataclass(obj):
            typing.get_type_hints(obj)   # WP1's YAML loader walks these


@pytest.mark.parametrize(("module", "names"), sorted(NAMED_IN_SPEC.items()))
def test_spec_named_types_exist(module: str, names: list[str]) -> None:
    mod = importlib.import_module(module)
    missing = [n for n in names if not hasattr(mod, n)]
    assert not missing


def test_check_policy_and_bully_limits_agree() -> None:
    from alloy.bully.types import BullyLimits
    from alloy.precision.types import CheckPolicy

    c, b = CheckPolicy(), BullyLimits()
    assert (b.pen_limit, b.e_gain_s2_rel, b.e_gain_s1_rel) == (c.pen_limit_m, c.energy_jump_rel, c.energy_s1_rel)
    assert c.pen_s1_m == 1e-2 and c.speed_fail_mps == 20.0 and b.v_max == 10.0


@pytest.mark.parametrize(("clearance", "max_pen", "s2", "s1"), [
    (2e-4, 5e-5, 1e-4, 1e-4),      # 0.2 mm pegs: plan target p99 <= 0.05 mm, max <= 0.1 mm (D2)
    (32e-6, 8e-6, 16e-6, 16e-6),   # M10 6H/6g: radial 16 µm -> 8 µm / 16 µm (D2)
    (None, 5e-5, 1e-4, 5e-4),
    (2e-3, 5e-5, 1e-4, 5e-4),
])
def test_precision_pair_gates(clearance: float | None, max_pen: float, s2: float, s1: float) -> None:
    from alloy.contact.types import PrecisionPairSpec

    spec = PrecisionPairSpec(0, 1, clearance=clearance)
    assert spec.resolved_max_penetration() == pytest.approx(max_pen)
    assert spec.s2_limit() == pytest.approx(s2) and spec.s1_limit() == pytest.approx(s1)
    assert PrecisionPairSpec(0, 1, max_penetration=1e-5).resolved_max_penetration() == 1e-5


def test_health_flags_bits_and_struct() -> None:
    import warp as wp

    from alloy.precision.types import HealthFlags, StepHealth

    bits = sorted(int(f) for f in HealthFlags)
    assert bits == [1 << i for i in range(len(bits))] and bits[-1] < 1 << 32  # one bit each, no gaps, fits uint32
    arr = wp.zeros(2, dtype=StepHealth)
    assert set(arr.numpy().dtype.names) >= {"flags", "severity", "first_bad_step", "energy_j", "argmax_body"}


def test_precision_observables_is_a_newton_container() -> None:
    from alloy._vendor.newton.solvers import SolverBase

    from alloy.precision.types import PrecisionObservables

    assert issubclass(PrecisionObservables, SolverBase.Observables)
    assert set(PrecisionObservables._observable_fields()) >= {"residual", "iterations"}


def test_misc_helpers() -> None:
    from alloy.bench.types import LawResult, MetricVerdict, Quantity, RigDataset, Tolerance
    from alloy.cards.types import PairCard
    from alloy.core.quantity import Q
    from alloy.coupling.types import CouplingReport
    from alloy.precision.types import HealthReport, PrecisionConfig
    from alloy.testing.fakes import synthetic_rig

    assert Tolerance(1e-3, 0.1).bound(-2.0) == pytest.approx(0.201)
    q = Quantity("x", "m", 1.0, 0.0, 1, 0.0)
    mv = [MetricVerdict("x", "m", None, q, 0.0, 1.0, None, "mean", v) for v in ("pass", "inconclusive")]
    assert LawResult("A1", "xpbd", "cpu", tuple(mv), {}, {}, 0.0, False).verdict == "inconclusive"
    assert LawResult("A1", "xpbd", "cpu", (), {}, {}, 0.0, False).verdict == "not_run"
    mu = Q(0.3, "default")
    assert PairCard("steel", "pp", mu, mu).key == ("pp", "steel")
    assert PrecisionConfig().digest() == PrecisionConfig().digest() != PrecisionConfig(fuse_fp=False).digest()
    assert HealthReport.ok(3).invalid_worlds == () and CouplingReport.quiet(1e-3, {"a": 1}, 2).rollbacks == 0
    rig: RigDataset = synthetic_rig("R4.drop_bounce", {"restitution": 0.5, "drop_height": 0.2})
    assert all(np.isclose(t.scalars["rebound_height"], 0.05) for t in rig.trials)
    with pytest.raises(ValueError):
        rig.split("test")  # type: ignore[arg-type]
