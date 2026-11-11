# SPDX-License-Identifier: Apache-2.0
"""WP12 acceptance: an eager run equals the graph run bitwise (SPEC §7 WP12, D9 CPU population: all scenes).

Every WP2 mixed scene (rigid, chain, cloth, soft, MPM, coupled under WP4's Scheduler/SchedulerADMM) is built by
``alloy.integration.build_simulation`` twice on CPU, eager and with captured period graphs (Warp records CPU graphs,
APIC), stepped two control periods, and compared by WP5 per-world digests over State plus every snapshotted solver,
stepper and topology array. Warp's CPU capture cannot record ``wp.copy`` between non-contiguous arrays; a scene that
hits it is reported not_run on CPU (the GPU test covers CUDA graphs).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.core.errors import GraphCaptureError
from alloy.core.registry import isolated_registry
from alloy.testing import integration as I
from alloy.testing import scene as S

pytestmark = [pytest.mark.compile_heavy, pytest.mark.integration, pytest.mark.timeout(900)]

PERIODS = 3
APIC_LIMIT = "APIC capture does not yet support"
CMD = "PYTHONPATH=. python -m pytest tests/integration/test_graph_eager.py -q"


CONTROL_DT = 0.005   # 4x fewer steps per period than the 0.02 s default (CPU budget)
CPU_ONLY_LIMITS = {   # scene -> why its period cannot be captured on CPU (the GPU test covers CUDA graphs)
    "box_cloth_cable": "SchedulerADMM copies between non-contiguous arrays, which Warp 1.18's CPU capture (APIC) "
                       "cannot record (measured: GraphCaptureError at capture)",
    "sand_heap": "Newton MPM on CPU iterates on the host reading residuals each batch "
                 "(newton@ac7f7830:newton/_src/solvers/implicit_mpm/solve_rheology.py:1817-1819); CUDA uses capture_while",
}
SLOW = {"box_cloth_cable", "randomized", "sand_heap", "sponge_on_table", "stack_pair_card"}   # > 20 s each on CPU


def _scenes():
    return {name: (dataclasses.replace(card, control_dt=CONTROL_DT), lib) for name, card, lib in S.mixed_scenes()}


def _params():
    return [pytest.param(n, marks=pytest.mark.slow) if n in SLOW else n for n in sorted(_scenes())]


def _run(card, lib, graph: str):
    from alloy.integration import build_simulation

    built = I.assemble(card, lib, "cpu", 2)
    sim = build_simulation(built, "standard", graph=graph)
    for _ in range(PERIODS):
        sim.step_period()
    return sim, I.digests(sim)


@pytest.mark.parametrize("name", _params())
def test_cpu_eager_equals_graph(name):
    card, lib = _scenes()[name]
    if name in CPU_ONLY_LIMITS:
        I.record("WP12-EAGER-GRAPH-CPU", None, "not_run", CMD, scene=name, reason=CPU_ONLY_LIMITS[name])
        pytest.skip(CPU_ONLY_LIMITS[name])
    with isolated_registry(empty=True):
        eager, d_eager = _run(card, lib, "off")
        try:
            graph, d_graph = _run(card, lib, "on")
        except GraphCaptureError as exc:
            if APIC_LIMIT not in str(exc):
                raise
            I.record("WP12-EAGER-GRAPH-CPU", None, "not_run", CMD, scene=name, reason=str(exc)[:300])
            pytest.skip(f"Warp CPU capture limit (CUDA covered by the GPU test): {str(exc)[:160]}")
    stats = graph.graph_stats()
    same = bool(np.array_equal(d_eager, d_graph))
    I.record("WP12-EAGER-GRAPH-CPU", float(same), "pass" if same else "fail", CMD, scene=name,
             stepper=type(eager.stepper).__name__, steps=eager.steps, graph=stats)
    assert stats["mode"] == "graph" and stats["captures"] >= 1 and stats["host_syncs"] == 0
    assert eager.steps == graph.steps and eager.clock.ticks == graph.clock.ticks
    assert same, f"{name}: eager and graph digests differ in worlds {np.nonzero((d_eager != d_graph).any(1))[0]}"


def test_cpu_chunked_graph_equals_eager():
    """A period captured as several chunks (``max_graph_steps`` below the steps per period) equals eager bitwise."""
    from alloy.integration import build_simulation

    card, lib = _scenes()["cloth_on_table"]
    with isolated_registry(empty=True):
        eager, d_eager = _run(card, lib, "off")
        sim = build_simulation(I.assemble(card, lib, "cpu", 2), "standard", graph="on")
        sim.max_graph_steps = 2
        for _ in range(PERIODS):
            sim.step_period()
        d_graph = I.digests(sim)
    n = sim.clock.steps_per_period
    assert n > 2, "the scene must have more steps per period than the chunk"
    first = n - 2                     # SCR-WP10-3: period 1 runs 2 eager exchanges, then n - 2 one-exchange graphs
    assert sim.graph_stats()["launches"] == first + (PERIODS - 1) * (n // 2)
    assert sim.steps == eager.steps and np.array_equal(d_eager, d_graph)
