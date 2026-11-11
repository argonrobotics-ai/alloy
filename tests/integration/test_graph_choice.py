# SPDX-License-Identifier: Apache-2.0
"""Lever #3 ([internal note] notes/mixed-speed-1107.md): each graph key is timed once eagerly and once from replayed
graphs, and a key whose graphs are slower than ``margin`` x its eager period runs eagerly from then on.

CPU: ``GraphChoice``'s bookkeeping with stand-in graphs and a stand-in clock. GPU (``-m gpu``): IntegratedSimulation
on a box on a table, 4 worlds. With a margin of 3 every key keeps its graphs (graphs are far faster there); with a
margin of 0 every timed key goes eager, its graphs are not launched again, and the run stays finite. graph="on" builds
no timing choice.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.integration import graph_choice as gc
from alloy.integration.graph_choice import GraphChoice


def _graph(launches: int, captured: bool = True) -> SimpleNamespace:
    return SimpleNamespace(launches=launches, _graph=object() if captured else None)


def _clock(monkeypatch, seconds: list[float]) -> None:
    """``time.perf_counter`` in graph_choice returns these values in turn."""
    it = iter(seconds)
    monkeypatch.setattr(gc.time, "perf_counter", lambda: next(it))


def test_what_each_period_times() -> None:
    ch, key = GraphChoice(), ("k",)
    assert ch.what(key, None) == "eager"                  # the key's first, eager period
    ch.eager_s[key] = 1.0
    assert ch.what(key, None) is None
    assert ch.what(key, _graph(0)) is None                # a graph's first launch (it may capture) is not timed
    assert ch.what(key, _graph(3, captured=False)) is None
    assert ch.what(key, _graph(3)) == "graph"
    ch.graph_s[key] = 0.5
    assert ch.what(key, _graph(5)) is None                # timed once
    assert ch.what(("other",), _graph(5)) is None         # no eager time yet: nothing to compare with


@pytest.mark.parametrize(("graph_s", "eager"), [(0.4, False), (1.4, False), (1.6, True)])
def test_a_key_goes_eager_only_when_its_graphs_are_slower_by_the_margin(monkeypatch, graph_s, eager) -> None:
    ch, key = GraphChoice(margin=1.5), ("k",)
    _clock(monkeypatch, [10.0, 11.0, 20.0, 20.0 + graph_s])
    with ch.timed(key, ch.what(key, None), "cpu"):
        pass
    with ch.timed(key, ch.what(key, _graph(1)), "cpu"):
        pass
    assert ch.eager_s[key] == pytest.approx(1.0) and ch.graph_s[key] == pytest.approx(graph_s)
    assert ch.eager(key) is eager
    assert ch.stats() == {"timed_keys": 1, "slow_graph_keys": [round(graph_s, 2)] if eager else []}


def test_times_are_per_exchange(monkeypatch) -> None:
    """SCR-WP10-3: the eager time is one exchange (a key's second warm-up exchange), the graph time a whole period;
    both are stored per exchange, so a 16-exchange graph period of 1.6 s is 0.1 s against an eager 0.2 s."""
    ch, key = GraphChoice(margin=1.5), ("k",)
    _clock(monkeypatch, [0.0, 0.2, 1.0, 2.6])
    with ch.timed(key, "eager", "cpu", 1):
        pass
    with ch.timed(key, ch.what(key, _graph(1)), "cpu", 16):
        pass
    assert ch.eager_s[key] == pytest.approx(0.2) and ch.graph_s[key] == pytest.approx(0.1) and not ch.eager(key)


def test_untimed_periods_are_left_alone(monkeypatch) -> None:
    ch = GraphChoice()
    monkeypatch.setattr(gc.time, "perf_counter", lambda: pytest.fail("an untimed period read the clock"))
    with ch.timed(("k",), None, "cpu"):
        pass
    assert not ch.eager_s and not ch.graph_s


@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("margin", [3.0, 0.0])        # 3: no flake on a loaded GPU (graphs win by far here)
def test_graph_choice_in_a_simulation(margin) -> None:
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    from alloy.core.registry import isolated_registry
    from alloy.integration import build_simulation
    from alloy.scene.builtin_plugins import register_builtin_solvers
    from alloy.testing import integration as I
    from alloy.testing import scene as S

    with isolated_registry(empty=True):
        register_builtin_solvers()
        card, lib = S.box_on_table(4)
        assert build_simulation(I.assemble(card, lib, "cuda:0", 4), "standard", graph="on").graph_choice is None
        sim = build_simulation(I.assemble(card, lib, "cuda:0", 4), "standard", graph="auto")
        sim.graph_choice.margin = margin
        for _ in range(12):     # 16 exchanges per period, full checks every 10: 5 check patterns, so the first
            sim.step_period()   # timing is the 7th period (never a capture or a first launch; test_graph_checks.py)
        stats = sim.graph_stats()
        launched = stats["launches"]
        for _ in range(3):
            sim.step_period()
        assert stats["timed_keys"] >= 1
        if margin == 0.0:
            assert len(stats["slow_graph_keys"]) == stats["timed_keys"]
            assert sim.graph_stats()["launches"] == launched           # slow keys are not launched again
        else:
            assert stats["slow_graph_keys"] == [] and sim.graph_stats()["launches"] > launched
        assert np.isfinite(sim.state.body_q.numpy()).all()


def _box_on_cloth(graph: str, periods: int = 6) -> tuple[list[int], list[np.ndarray]]:
    """WP4's box on cloth, 2 worlds, CPU (graph "on": Warp CPU graphs). A sticky escalation before period 3 makes a
    fresh key mid-run. Per period: the Python exchange calls (a capture runs its body once) and the digests."""
    from alloy.core.registry import isolated_registry
    from alloy.integration import build_simulation
    from alloy.scene.builtin_plugins import register_builtin_solvers
    from alloy.testing import integration as I
    from alloy.testing.coupling import ClothSceneSpec, cloth_scheduler, coupling_box_on_cloth

    spec = ClothSceneSpec(box_mass=0.2, cells=8)
    with isolated_registry(empty=True):
        register_builtin_solvers()
        model = coupling_box_on_cloth(2, spec, drop_offsets=[0.0, 0.002], device="cpu")
        sched = cloth_scheduler(model, spec, vbd_adaptive=False)
        built = I.built_from_model(model, [I.entry_spec(model, n, 1 / 480) for n in ("mjc", "vbd")], control_dt=0.01)
        sim = build_simulation(built, "standard", stepper=sched, graph=graph)
        count, exchange = [0], sim._exchange

        def counted(dt: float) -> None:
            count[0] += 1
            exchange(dt)

        sim._exchange = counted
        calls, digests = [], []
        for k in range(periods):
            if k == 3:
                sched._clock.j_floor = sched._clock.j + 1
            count[0] = 0
            sim.step_period()
            calls.append(count[0])
            digests.append(I.digests(sim).copy())
    return calls, digests


@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_a_new_keys_first_period_warms_up_two_exchanges_then_replays() -> None:
    """SCR-WP10-3 law (CPU graphs): a key's first period runs two exchanges eagerly and replays one-exchange graphs
    (one per check pattern) for the rest, a key first seen mid-run (period 3) too; every period's digests equal the
    all-eager run's (today's first period of a key was all eager)."""
    on, d_on = _box_on_cloth("on")
    off, d_off = _box_on_cloth("off")
    print(f"python exchange calls per period: graphs {on}, eager {off}")
    for k, (a, b) in enumerate(zip(d_on, d_off)):
        assert np.array_equal(a, b), f"period {k}"
    assert off[3] == 2 * off[0]                                   # the escalated key: twice the exchanges
    assert on[0] == 4 and on[3] == 4, on      # 2 eager + 1 captured body per check pattern (cheap, full every 10)
