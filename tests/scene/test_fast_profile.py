# SPDX-License-Identifier: Apache-2.0
"""Opt-in fast route profile (SCR-WP3-8): the default route is unchanged. Under the fast profile, a precision scene
runs impratio 2 with the scoped noslip on its precision pairs. T09's chain-attachment CG entry runs MuJoCo Warp's
Newton solver with chain armature 1e-5 (FP.CHAIN_ARMATURE): WP3's measured setting (same kwargs, same Model arrays)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from alloy.core.errors import AssemblyError
from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.scene import fast_profile as FP
from alloy.scene.assembler import SceneAssembler, digest_model
from alloy.scene.noslip_route import NOSLIP_KEY, install_entry_noslip
from alloy.testing import scene as S

ROD_PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}


def _library():
    try:
        from alloy.cards.library import CardLibrary
    except ImportError:
        pytest.skip("WP1 CardLibrary not merged")
    return CardLibrary()


def _build(lib, card, profile: str, worlds: int = 1):
    with isolated_registry(empty=True):
        asm = SceneAssembler(lib, Router(rod_self_test=lambda rod, h: dict(ROD_PASS)), route_profile=profile)
        return asm.assemble(card, "cpu", world_count=worlds)


def _mujoco_kwargs(built) -> dict:
    (mj,) = [e for e in built.entries if e.capability == "mujoco"]
    return dict(mj.solver_kwargs)


def _pairs(built) -> tuple[tuple[int, int], ...]:
    return tuple((p.shape_a, p.shape_b) for p in built.precision_pairs)


def test_t09_default_route_unchanged_and_fast_is_wp3_setting():
    """Fast T09 = default T09 + WP3's measured knobs (WP12's `with_entry_kwargs` + `chain_armature` on the default):
    the same entry kwargs and bit-identical Model arrays; the noslip request adds the precision pair scope."""
    from alloy.integration.throughput import chain_armature, with_entry_kwargs

    lib = _library()
    card = lib.scene("task_t09")
    default, fast = _build(lib, card, "default", 2), _build(lib, card, "fast", 2)
    kd, kf = _mujoco_kwargs(default), _mujoco_kwargs(fast)
    assert (kd["solver"], kd["impratio"], kd[NOSLIP_KEY]) == ("cg", 300.0, (50, 1e-10))     # the route
    assert FP.FAST_EFFECT not in default.report.scene_effects and FP.FAST_CAVEAT not in default.report.warnings

    wp3 = _mujoco_kwargs(with_entry_kwargs(default, {"solver": "newton", "impratio": 2.0}))
    assert kf.pop(NOSLIP_KEY) == (50, 1e-10, _pairs(fast)) and _pairs(fast) == _pairs(default)
    wp3.pop(NOSLIP_KEY)
    assert kf == wp3
    assert chain_armature(default, FP.CHAIN_ARMATURE) > 0
    assert digest_model(fast.model) == digest_model(default.model)
    assert {FP.FAST_EFFECT, FP.NEWTON_EFFECT} <= set(fast.report.scene_effects)
    assert FP.FAST_CAVEAT in fast.report.warnings and "0.99-1.06 mm" in FP.FAST_CAVEAT


def test_fast_profile_without_chain_keeps_the_solver():
    """A precision scene without a chain attachment: impratio 2 and the pair noslip; solver and model as routed."""
    lib = _library()
    card = lib.scene("demo_peg_d12")
    default, fast = _build(lib, card, "default"), _build(lib, card, "fast")
    kd, kf = _mujoco_kwargs(default), _mujoco_kwargs(fast)
    assert kd["impratio"] == 300.0 and kf.pop("impratio") == FP.FAST_IMPRATIO
    assert kf.pop(NOSLIP_KEY) == (*FP.FAST_NOSLIP, _pairs(fast)) and _pairs(fast)
    kd.pop("impratio")
    kd.pop(NOSLIP_KEY, None)
    assert kf == kd
    assert digest_model(fast.model) == digest_model(default.model)
    assert FP.NEWTON_EFFECT not in fast.report.scene_effects


def test_fast_profile_leaves_scenes_without_precision_pairs():
    card, lib = S.box_on_table(worlds=1)
    default, fast = _build(lib, card, "default"), _build(lib, card, "fast")
    assert _mujoco_kwargs(fast) == _mujoco_kwargs(default)
    assert FP.FAST_EFFECT not in fast.report.scene_effects and FP.FAST_CAVEAT not in fast.report.warnings


def test_choose_fast_skips_discrete():
    t = SimpleNamespace(decision=SimpleNamespace(entry="mujoco", scene_effects=()))
    assert not FP.choose_fast([t], ((1, 0),), "discrete").on
    assert FP.choose_fast([t], ((1, 0),), "implicitfast").on


def test_unknown_route_profile_raises():
    with pytest.raises(AssemblyError, match="route profile"):
        SceneAssembler(S.library([]), route_profile="turbo")


def test_entry_noslip_scopes_the_precision_pairs(monkeypatch):
    from alloy.contact import noslip as ns

    seen = {}
    monkeypatch.setattr(ns, "pair_geoms", lambda solver, pairs: tuple((p.shape_b, p.shape_a) for p in pairs))
    monkeypatch.setattr(ns, "install_noslip", lambda solver, spec: seen.setdefault("spec", spec))
    opt = SimpleNamespace(integrator=int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST),
                          cone=int(mujoco.mjtCone.mjCONE_ELLIPTIC), enableflags=0)
    solver = SimpleNamespace(mj_model=SimpleNamespace(opt=opt), mjw_model=object())
    assert install_entry_noslip(solver, (50, 1e-10, ((1, 0),))) == "scoped"
    assert dataclasses.astuple(seen.pop("spec"))[:3] == (50, 1e-10, "robot")
    assert install_entry_noslip(solver, (50, 1e-10)) == "scoped"
    assert seen["spec"].pairs == ()


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_fast_peg_installs_pair_noslip_and_steps():
    """End to end on CPU: the factory builds impratio 2 and routes Alloy's scoped noslip with the peg/hole geom pair."""
    from alloy.contact import noslip as ns
    from alloy.sim import Simulation

    lib = _library()
    with isolated_registry(empty=True):
        from alloy.scene.builtin_plugins import register_builtin_solvers

        register_builtin_solvers()
        built = SceneAssembler(lib, route_profile="fast").assemble(lib.scene("demo_peg_d12"), "cpu", world_count=1)
        sim = Simulation(built)
        (solver,) = [s for s in sim.stepper.solvers() if hasattr(s, "mjw_model")]
        assert solver.mj_model.opt.impratio == pytest.approx(FP.FAST_IMPRATIO)
        _, route = ns._ROUTES[id(solver.mjw_model)]     # (weakref to the model, ScopedNoslip)
        want = {tuple(sorted(g)) for g in ns.pair_geoms(solver, built.precision_pairs)}
        assert {tuple(map(int, g)) for g in route.scope.pairs} == want and want
        for _ in range(5):
            sim.step_period()
        assert np.isfinite(sim.state.body_q.numpy()).all()
