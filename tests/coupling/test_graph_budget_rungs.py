# SPDX-License-Identifier: Apache-2.0
"""Graph-budget rungs (speed-now S2, opt-in ``ClockPolicy.graph_budget_rungs``): an exchange over the graph node budget
caps the entry with the most substeps one rung lower, and the clock takes a finer dt_x at the same entry step.

The numbers are the falling towel of robot_cloth_liquid with the contact gate (64 worlds, L4, period 14): the gate
released the pairs (dt_x 1.25 ms), MuJoCo's stable step 1 ms and VBD's 0.27 ms gave 4 and 16 substeps, and that
exchange was 74,344 graph nodes, over the 65,536 budget."""

import math
from types import SimpleNamespace

import pytest

from alloy.coupling.clocks import ClockController
from alloy.coupling.graph_budget_rungs import GraphBudgetRungsMixin
from alloy.coupling.types import ClockPolicy
from alloy.integration import simulation as sim_mod

CONTROL = 0.01
STABLE = {"mujoco": 1.0e-3, "vbd": 0.27e-3}           # [s] the entries' stable steps (period 14)
NODES = 74_344


def _clock() -> ClockController:
    c = ClockController(ClockPolicy(), {"mujoco": 1, "vbd": 1}, {"mujoco": True, "vbd": True})
    c.safety_for = {"mujoco": 0.5, "vbd": 0.5}
    return c


def _capped() -> tuple[ClockController, object]:
    c = _clock()
    first = c.decide(CONTROL, math.inf, STABLE)       # released pairs: no interface bound
    assert c.cap_rung(first.substeps) == "vbd"
    return c, first


def test_cap_takes_a_finer_exchange_at_the_same_entry_steps():
    c, first = _capped()
    assert first.dt_x == pytest.approx(1.25e-3) and first.substeps == {"mujoco": 4, "vbd": 16}
    assert c.rung_cap == {"vbd": 8}
    d = c.decide(CONTROL, math.inf, STABLE)
    assert d.dt_x == pytest.approx(0.625e-3) and d.substeps == {"mujoco": 2, "vbd": 8}
    for name in d.substeps:                           # every entry keeps its own step
        assert d.dt_x / d.substeps[name] == pytest.approx(first.dt_x / first.substeps[name])
    assert not d.bound_unmet


def test_capped_rung_stays_within_its_cap_as_the_towel_slows_and_speeds():
    c, _ = _capped()
    c.decide(CONTROL, math.inf, STABLE)
    slow, fast = {"mujoco": 1.0e-3, "vbd": 0.6e-3}, {"mujoco": 1.0e-3, "vbd": 0.1e-3}
    seen = [c.decide(CONTROL, math.inf, slow) for _ in range(30)]
    assert seen[-1].dt_x == pytest.approx(1.25e-3)    # the level relaxes back (one step per quiet stretch)
    seen += [c.decide(CONTROL, math.inf, fast) for _ in range(5)]
    assert seen[-1].dt_x == pytest.approx(0.3125e-3)  # 8 x 0.5 x 0.1 ms = 0.4 ms: the grid's 0.3125 ms at once
    assert all(d.substeps["vbd"] <= 8 for d in seen)


def test_a_rung_held_above_its_cap_by_hysteresis_drops_at_once():
    c = _clock()
    assert c.decide(CONTROL, math.inf, STABLE).substeps["vbd"] == 16
    calmer = {"mujoco": 1.0e-3, "vbd": 0.6e-3}           # wants 8 at 1.25 ms; the hysteresis holds 16 for a while
    assert c.decide(CONTROL, math.inf, calmer).substeps["vbd"] == 16
    assert c.cap_rung({"mujoco": 4, "vbd": 16}) == "vbd"
    d = c.decide(CONTROL, math.inf, calmer)              # 1.25 ms already meets the cap bound: no jump
    assert d.dt_x == pytest.approx(1.25e-3) and d.substeps["vbd"] == 8


def test_an_interface_bound_finer_than_the_cap_still_rules():
    c, _ = _capped()
    d = c.decide(CONTROL, 1.0e-4, STABLE)
    assert d.dt_x == pytest.approx(CONTROL / 128) and d.substeps["vbd"] == 1


@pytest.mark.parametrize(("rungs", "adaptive", "stable"), [
    ({"a": 1, "b": 1}, {"a": True, "b": True}, {"a": 1.0, "b": 1.0}),            # already at the first rung
    ({"a": 1, "b": 4}, {"a": True, "b": False}, {"a": 1.0, "b": 1.0e-4}),        # the fixed entry is not capped
    ({"a": 1, "b": 4}, {"a": True, "b": True}, {"a": 1.0, "b": math.nan}),       # no estimate: a cap binds nothing
])
def test_cap_rung_needs_an_adaptive_estimated_entry_above_the_first_rung(rungs, adaptive, stable):
    c = ClockController(ClockPolicy(), rungs, adaptive)
    c.decide(CONTROL, CONTROL, stable)
    assert c.cap_rung(rungs) is None and c.rung_cap == {}


@pytest.mark.parametrize("on", [False, True])
def test_graph_over_budget_follows_the_flag(on):
    c = _clock()
    d = c.decide(CONTROL, math.inf, STABLE)
    stepper = GraphBudgetRungsMixin()
    stepper._alloy_policy = SimpleNamespace(clock=ClockPolicy(graph_budget_rungs=on))
    stepper._clock = c
    assert stepper.graph_over_budget(d.substeps, NODES) == ("vbd" if on else None)
    assert c.rung_cap == ({"vbd": 8} if on else {})
    assert c.decide(CONTROL, math.inf, STABLE).substeps["vbd"] == (8 if on else 16)


def test_simulation_reports_an_over_budget_key_to_the_stepper(monkeypatch):
    monkeypatch.setattr(sim_mod, "graph_nodes", lambda graph: NODES)
    calls = []
    fake = SimpleNamespace(stepper=SimpleNamespace(graph_over_budget=lambda sub, n: calls.append((sub, n))),
                           substeps={"mujoco": 4, "vbd": 16}, exchange_nodes={}, eager_keys={},
                           graph_node_budget=1 << 16)
    graph = SimpleNamespace(_graph=object(), invalidate=lambda: None)
    assert sim_mod.IntegratedSimulation._fits(fake, "key", graph) is False
    assert fake.eager_keys == {"key": NODES} and calls == [({"mujoco": 4, "vbd": 16}, NODES)]


@pytest.mark.parametrize(("chunk", "reported"), [(1, True), (2, False)])
def test_a_check_pattern_over_budget_reports_at_one_exchange_only(monkeypatch, chunk, reported):
    """graph_checks._sized counts a chunk's graph with health's checks: at one exchange over the budget the key goes
    eager for good and the stepper hears of it; at 2+ later periods only take smaller chunks (WP12's review)."""
    from alloy.integration import graph_checks as gc

    monkeypatch.setattr(gc, "graph_nodes", lambda graph: NODES * chunk)
    calls = []
    fake = SimpleNamespace(stepper=SimpleNamespace(graph_over_budget=lambda sub, n: calls.append((sub, n))),
                           substeps={"mujoco": 4, "vbd": 16}, exchange_nodes={}, eager_keys={}, steps=0,
                           graph_node_budget=1 << 16)
    graph = SimpleNamespace(_graph=None, invalidate=lambda: None)
    graph._capture = lambda: setattr(graph, "_graph", object())
    assert gc._sized(fake, "key", chunk, graph) is False
    assert fake.exchange_nodes == {"key": NODES}
    assert fake.eager_keys == ({"key": NODES} if reported else {})
    assert calls == ([({"mujoco": 4, "vbd": 16}, NODES)] if reported else [])
