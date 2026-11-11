# SPDX-License-Identifier: Apache-2.0
"""effective_determinism and apply_precision (SPEC D9, §3.7; PRC-13 rules, CPU part): reported mode = what runs,
every downgrade has a reason and sets DET_DOWNGRADED; the live-solver probe reads instance options."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import warp as wp

from alloy.core import registry
from alloy.core.protocols import DetCapability
from alloy.precision.apply import apply_precision
from alloy.precision.determinism import effective_determinism, native_share, probe_mode, solver_det
from alloy.precision.types import CheckPolicy, Determinism, HealthFlags, PrecisionConfig
from alloy.testing.precision import HealthLoop
from alloy.testing.scenes import box_on_plane

GPU = SimpleNamespace(is_cuda=True)
CPU = SimpleNamespace(is_cuda=False)
R2R, NG = wp.DeterministicMode.RUN_TO_RUN, wp.DeterministicMode.NOT_GUARANTEED


def _solver(name: str, **attrs: Any):
    return type(name, (), attrs)()


def test_cpu_is_always_run_to_run() -> None:
    for name in ("SolverMuJoCo", "SolverXPBD", "SolverKamino"):
        assert solver_det(_solver(name), 4096, CPU).mode == Determinism.RUN_TO_RUN


@pytest.mark.parametrize("solver, worlds, mode", [
    (_solver("SolverSteel", deterministic=R2R), 4096, Determinism.RUN_TO_RUN),
    (_solver("SolverSteel", deterministic=NG), 16, Determinism.NONE),
    (_solver("SolverXPBD", _module_options={"k": {"deterministic": R2R}}), 1024, Determinism.RUN_TO_RUN),
    (_solver("SolverXPBD", _module_options={"k": {"deterministic": R2R}, "t": {"deterministic": NG}}), 8,
     Determinism.NONE),
    (_solver("SolverMuJoCo", _deterministic=R2R), 64, Determinism.RUN_TO_RUN),
    (_solver("SolverMuJoCo", _deterministic=R2R), 65, Determinism.NONE),
    (_solver("SolverMuJoCo", _deterministic=NG), 4, Determinism.NONE),
    (_solver("SolverKamino"), 4, Determinism.NONE),
])
def test_gpu_rules(solver, worlds: int, mode: Determinism) -> None:
    det = solver_det(solver, worlds, GPU)
    assert det.mode == mode
    assert (det.reason is not None) == (mode == Determinism.NONE) or det.reason.endswith("fallback table used")


def test_registry_capability_wins_over_fallback() -> None:
    plugin = _solver("MyNativePlugin")
    type(plugin).determinism = DetCapability.NATIVE
    assert solver_det(plugin, 4096, GPU).mode == Determinism.RUN_TO_RUN
    assert native_share([plugin]) == 1.0 and native_share([plugin, _solver("SolverMuJoCo")]) == 0.0


def test_effective_minimum_and_reasons(device: str) -> None:
    cfg = PrecisionConfig()
    mode, reasons = effective_determinism(cfg, [], 4)
    assert mode == Determinism.RUN_TO_RUN and "no solvers given" in reasons
    mode, reasons = effective_determinism(PrecisionConfig(determinism=Determinism.NONE), [], 4)
    assert mode == Determinism.NONE
    mode, reasons = effective_determinism(PrecisionConfig(determinism=Determinism.GPU_TO_GPU), [], 4)
    assert mode == Determinism.RUN_TO_RUN and any("GPU_TO_GPU" in r for r in reasons)


def test_live_xpbd_probe(device: str) -> None:
    scene = box_on_plane(2, device=device)
    on = scene.make_solver("xpbd", deterministic=R2R)
    off = scene.make_solver("xpbd", deterministic=NG)
    assert probe_mode(on) == Determinism.RUN_TO_RUN and probe_mode(off) == Determinism.NONE


def test_downgrade_sets_flag(device: str) -> None:
    det_kw = {"deterministic": R2R}  # XPBD with Warp's lowering on: RUN_TO_RUN on GPU, and on CPU anyway
    gpu_like = PrecisionConfig(determinism=Determinism.GPU_TO_GPU)  # never effective in v1 -> downgraded
    loop = HealthLoop(box_on_plane(2, device=device), checks=CheckPolicy(), config=gpu_like, solver_kwargs=det_kw)
    loop.step(1)
    rep = loop.monitor.poll()
    assert rep.det_mode == Determinism.RUN_TO_RUN and any("GPU_TO_GPU" in r for r in rep.det_reasons)
    assert (rep.flags & int(HealthFlags.DET_DOWNGRADED)).all()
    clean = HealthLoop(box_on_plane(2, device=device), checks=CheckPolicy(), config=PrecisionConfig(),
                       solver_kwargs=det_kw)
    clean.step(1)
    assert not (clean.monitor.poll().flags & int(HealthFlags.DET_DOWNGRADED)).any()
    if wp.get_device(device).is_cuda:  # default XPBD (no lowering) on a GPU: NONE, flagged, with the reason
        stock = HealthLoop(box_on_plane(2, device=device), checks=CheckPolicy(), config=PrecisionConfig())
        stock.step(1)
        rep = stock.monitor.poll()
        assert rep.det_mode == Determinism.NONE and (rep.flags & int(HealthFlags.DET_DOWNGRADED)).all()
        assert any("SolverXPBD" in r for r in rep.det_reasons)


@dataclass
class _Entry:
    name: str
    capability: str
    solver_kwargs: dict = field(default_factory=dict)


def _cap(name: str, det: DetCapability) -> registry.SolverCapability:
    return registry.SolverCapability(name=name, factory=lambda view, kw: None, routes=frozenset({"RIGID"}),
                                     owns=frozenset({"body"}), coupling=True, uniform_worlds=True,
                                     per_contact_friction=False, determinism=det)


def test_apply_precision(device: str) -> None:
    scene = box_on_plane(4, device=device)
    entries = [_Entry("rigid", "mujoco"), _Entry("cloth", "vbd"), _Entry("steel", "steel"),
               _Entry("pinned", "mujoco", {"deterministic": R2R}), _Entry("x", "unknown")]
    built = SimpleNamespace(model=scene.model, entries=entries, world_count=4)
    with registry.isolated_registry(empty=True):
        registry.register_solver(_cap("mujoco", DetCapability.WARP_LOWERING))
        registry.register_solver(_cap("vbd", DetCapability.WARP_LOWERING))
        registry.register_solver(_cap("steel", DetCapability.NATIVE))
        out = apply_precision(built, PrecisionConfig())
        fast = apply_precision(built, PrecisionConfig(determinism=Determinism.NONE))
    assert out["solvers"]["rigid"] == {"deterministic": NG}
    assert out["solvers"]["cloth"] == {"deterministic": R2R}
    assert out["solvers"]["steel"] == {"deterministic": R2R}
    assert out["solvers"]["pinned"] == {} and out["solvers"]["x"] == {}
    assert out["use_graph"] is False  # VBD and the pinned MuJoCo entry run Warp's lowering
    assert out["bulk_pipeline"]["deterministic"] and out["precision_pipeline"]["deterministic"]
    assert any("unknown" in r for r in out["reasons"])
    assert fast["bulk_pipeline"]["deterministic"] is False and fast["solvers"]["cloth"] == {}


def test_pipeline_probe(device: str) -> None:
    from alloy._vendor import newton

    scene = box_on_plane(2, device=device)
    sorted_pipe = newton.CollisionPipeline(scene.model, deterministic=True)
    plain = newton.CollisionPipeline(scene.model, deterministic=False)
    assert solver_det(sorted_pipe, 1024, GPU).mode == Determinism.RUN_TO_RUN
    det = solver_det(plain, 1024, GPU)
    assert det.mode == Determinism.NONE and "CollisionPipeline" in det.reason


def test_native_plugin_without_switch() -> None:
    from alloy.testing.core import FakeSolver

    scene = box_on_plane(2, device="cpu")
    assert solver_det(FakeSolver(scene.model), 4096, GPU).mode == Determinism.RUN_TO_RUN


def test_capability_for_is_the_source_of_truth() -> None:
    """D9: a registered SolverCapability (solver_cls set) beats the fallback table."""
    from alloy.precision.determinism import capability_of

    cls = type("SolverXPBD", (), {})  # the fallback table says WARP_LOWERING for this name
    with registry.isolated_registry(empty=True):
        cap = _cap("xpbd-native", DetCapability.NATIVE)
        registry.register_solver(registry.SolverCapability(**{**cap.__dict__, "solver_cls": cls}))
        got, note = capability_of(cls())
    assert got == DetCapability.NATIVE and note is None
    assert capability_of(cls())[0] == DetCapability.WARP_LOWERING  # registry restored: fallback again
