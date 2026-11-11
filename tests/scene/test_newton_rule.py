# SPDX-License-Identifier: Apache-2.0
"""SCR-DC1-2: a chain attachment keeps MuJoCo Warp's Newton solver when the float32 Hessian stays positive definite
(eps32·|S·JᵀDJ·S| < λ_min(S·M·S), S = diag(M)^-1/2), or with evidence; else CG as before."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from alloy.scene import newton_evidence as EV
from alloy.scene import newton_rule as NR

sys.path.insert(0, str(Path(__file__).parent))
import test_weld_impedance as T  # noqa: E402


@pytest.fixture
def fresh_rule(tmp_path, monkeypatch):
    """The rule's cache in tmp_path and an empty memo, so the probe runs."""
    monkeypatch.setattr(NR, "_cache_path", lambda: tmp_path / "newtonrule.json")
    monkeypatch.setattr(NR, "_MEMO", {})
    return tmp_path


def test_margin_is_unit_free():
    M = np.array([[2.0e-3, 1.0e-5], [1.0e-5, 1.0e-6]])        # a 2 g body and a 1e-6 kg·m² hinge, coupled
    K = np.array([[50.0, 0.0], [0.0, 0.0]])                    # a stiff row on the translational dof only
    g = NR.margin(M, K)
    S = np.diag([10.0, 1.0e3])                                 # change of units: the ratio does not move
    assert NR.margin(S @ M @ S, S @ K @ S).ratio == pytest.approx(g.ratio, rel=1e-9)
    assert g.ratio == pytest.approx(NR.EPS32 * (50.0 / 2.0e-3) / np.linalg.eigvalsh(
        np.diag(np.diag(M) ** -0.5) @ M @ np.diag(np.diag(M) ** -0.5))[0], rel=1e-9)
    assert NR.margin(M, np.zeros_like(M)).ratio == 0.0
    assert NR.margin(M, 1e9 * K).ratio > NR.RULE_BOUND


def test_t09_keeps_cg(fresh_rule):
    from alloy.cards.library import CardLibrary
    from alloy.scene.assembler import SceneAssembler

    lib = CardLibrary(assets=None)
    built = SceneAssembler(lib).assemble(lib.scene("task_t09"), "cpu", world_count=1)
    line = next(x for x in built.report.entry_options if x.startswith("mujoco solver: "))
    assert "CG kept" in line and "impratio 300" in line
    assert "mujoco:solver=cg" in built.report.scene_effects
    assert next(e for e in built.entries if e.name == "mujoco").solver_kwargs["solver"] == "cg"
    probe = next(iter(NR._MEMO.values()))
    assert probe["finite"] and probe["ratio"] > 10 * NR.RULE_BOUND            # 347 measured (1 s settle)
    assert (fresh_rule / "newtonrule.json").exists()


def test_chain_without_contact_keeps_newton(fresh_rule, tmp_path):
    from alloy.cards.types import Attachment, Placement
    from alloy.sim import Simulation

    objs = (Placement(T.BLOCK, (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0)),
            Placement("usb_cable_1m", (0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0)))     # clear of the block
    built = _assemble_stubbed(T._library(tmp_path), objs, (Attachment(T.BLOCK, "usb_cable_1m:0", "weld"),))
    line = next(x for x in built.report.entry_options if x.startswith("mujoco solver: "))
    assert "Newton for the chain attachment" in line                 # a hanging cable: no constraint row, margin 0
    assert "mujoco:solver=cg" not in built.report.scene_effects
    assert next(e for e in built.entries if e.name == "mujoco").solver_kwargs.get("solver") != "cg"
    sim = Simulation(built)                                          # MuJoCo Warp, float32 Newton
    for _ in range(3):
        sim.step_period()
    assert np.isfinite(sim.state.body_q.numpy()).all()


def test_evidence_opts_a_failing_scene_into_newton(monkeypatch):
    from alloy.cards.library import CardLibrary
    from alloy.scene.assembler import SceneAssembler

    lib = CardLibrary(assets=None)
    scene = lib.scene("task_t09")
    cand = SceneAssembler(lib, newton_candidate=True).assemble(scene, "cpu", world_count=1)
    key = EV._key(cand)                                              # the candidate prints the key the rule uses
    assert "mujoco:solver=cg" not in cand.report.scene_effects
    short = EV.Record("task_t09", key, 512, 50, 0, "cuda:0", "test", "test")
    monkeypatch.setattr(EV, "RECORDS", (short,))
    assert not short.valid() and EV.find(key) is None
    assert "CG kept" in _solver_line(lib, scene)
    good = EV.Record("task_t09", key, 1024, 50, 0, "cuda:0", "test", "test")
    monkeypatch.setattr(EV, "RECORDS", (good,))
    line = _solver_line(lib, scene)
    assert "by evidence (task_t09: 1024 worlds x 50 periods on cuda:0, 0 INVALID" in line
    monkeypatch.setattr(EV, "RECORDS", (EV.Record("task_t09", key, 1024, 50, 1, "cuda:0", "test", "test"),))
    assert "CG kept" in _solver_line(lib, scene)                     # one INVALID world: no opt-out


def _assemble_stubbed(lib, objects, attachments):
    """As test_weld_impedance._assemble, with the rod self-test stubbed to a pass (its numbers are not under test)."""
    from alloy.cards.types import SceneCard
    from alloy.route.router import Router
    from alloy.scene.assembler import SceneAssembler
    from alloy.testing.integration import ROD_SELF_TEST_PASS

    router = Router(rod_self_test=lambda rod, h: dict(ROD_SELF_TEST_PASS))
    card = SceneCard(robots=(), objects=objects, attachments=attachments, control_dt=0.02)
    return SceneAssembler(lib, router).assemble(card, "cpu", world_count=1)


def _solver_line(lib, scene) -> str:
    from alloy.scene.assembler import SceneAssembler

    built = SceneAssembler(lib).assemble(scene, "cpu", world_count=1)
    return next(x for x in built.report.entry_options if x.startswith("mujoco solver: "))
