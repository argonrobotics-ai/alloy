# SPDX-License-Identifier: Apache-2.0
"""WP12 wiring: profiles, the Scheduler bridge and which real parts ``wire`` builds per scene (CPU)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.core.registry import isolated_registry
from alloy.coupling.types import CouplingReport
from alloy.integration import PROFILES, IntegratedSimulation, build_simulation, profile, wire
from alloy.integration.simulation import merge_reports
from alloy.integration.wiring import _health
from alloy.integration.stepper import entry_solvers
from alloy.testing import integration as I
from alloy.testing import scene as S


def test_profiles():
    assert set(PROFILES) == {"standard", "precision", "fast", "debug"}
    p = profile("standard", native_contacts="newton", graph="on")
    assert p.native_contacts == "newton" and p.wants_graph(False) == (True, True)
    assert profile("standard").wants_graph(False) == (False, False)
    assert profile("standard").wants_graph(True) == (True, False)
    assert profile("debug").wants_graph(True) == (False, False)
    cfg = profile("precision").precision_config("cpu")
    assert cfg.checks.on_invalid == "raise" and cfg.native_contacts == "mujoco"
    assert profile("standard", native_contacts="newton").precision_config("cpu").native_contacts == "newton"
    assert not any(profile(n).precision_config("cpu").mjw_route_cg for n in PROFILES)     # SCR-DC5-4 (ii): opt-in
    assert profile("standard", mjw_route_cg=True).precision_config("cpu").mjw_route_cg
    with pytest.raises(ValueError):
        profile("nope")


def test_merge_reports_takes_kept_worlds_from_first():
    first = CouplingReport.quiet(1e-3, {"a": 1}, 3)
    rerun = CouplingReport.quiet(5e-4, {"a": 2}, 3)
    first = merge_reports(first, first, np.ones(3, np.int32))
    rerun_ws = np.array([2, 0, 2], np.int32)
    rerun = rerun.__class__(**{**rerun.__dict__, "watch_state": rerun_ws})
    out = merge_reports(first, rerun, np.array([1, 0, 1], np.int32))
    assert out.watch_state.tolist() == [0, 0, 0] and out.dt_x == 5e-4


def _scene(name):
    return {n: (c, lib) for n, c, lib in S.mixed_scenes()}[name]


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_wire_single_entry_and_newton_buffer():
    card, lib = _scene("box_on_table")
    with isolated_registry(empty=True):
        built = I.assemble(card, lib, "cpu", 2)
        parts = wire(built, "standard")
        assert type(parts.stepper).__name__ == "SingleEntryStepper"
        assert parts.contact_layer is None and parts.health is not None and parts.native == "mujoco"
        assert list(parts.solvers) == ["mujoco"]
        newton = wire(built, profile("standard", native_contacts="newton"))
        assert type(newton.contact_layer).__name__ == "AlloyContactLayer" and newton.gap_monitor is None
        sim = build_simulation(built, "standard", native_contacts="newton")
        assert isinstance(sim, IntegratedSimulation) and sim.solvers
        sim.step_period()
        assert np.isfinite(sim.state.body_q.numpy()).all()
        assert sim.report()["graph"]["mode"] == "eager"


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_wire_coupled_scene_uses_scheduler():
    card, lib = _scene("randomized")
    with isolated_registry(empty=True):
        built = I.assemble(card, lib, "cpu", 2)
        parts = wire(built, "standard")
        assert type(parts.stepper).__name__ in ("Scheduler", "SchedulerADMM")
        assert set(entry_solvers(parts.stepper)) == {e.name for e in built.entries if e.name}
        stock = wire(built, profile("standard", stock_coupler=True))
        assert type(stock.stepper).__name__ == "StockProxyStepper"


def test_health_gets_the_stepper_capacity_flags(monkeypatch):
    """SCR-DC5-19: a stepper's ``capacity_flags()`` (WP4's proxy contact pipeline) join the health monitor's capacity
    flags (CONTACT_OVERFLOW, fail closed); a stepper without them adds none. End to end, DC5's drip ledger test
    (emitted SPH particles under the Scheduler) is the check that the flag does not trip on emission."""
    import alloy.precision.health as H

    seen = []
    monkeypatch.setattr(H, "HealthMonitor", lambda *a, **k: seen.append(k["capacity_flags"]) or "health")
    flag = wp.zeros(1, dtype=wp.int32, device="cpu")
    parts = SimpleNamespace(contact_layer=None, binding=None, stepper=SimpleNamespace(capacity_flags=lambda: [flag]),
                            built=SimpleNamespace(model=None, precision_pairs=()),
                            precision=SimpleNamespace(checks=None), solvers={})
    assert _health(parts) == "health" and seen[-1] == [flag]
    parts.stepper = SimpleNamespace()
    assert _health(parts) == "health" and seen[-1] == []
