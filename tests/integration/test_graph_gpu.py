# SPDX-License-Identifier: Apache-2.0
"""WP12 GPU acceptance (L4, SPEC §7 WP12): graph mode with every real part injected.

- peg-hole (WP3 meshed peg in a 0.1 mm-clearance bore, precision pair) with the contact layer, gap monitor, health
  and the stepper runs in CUDA-graph mode with PAR-14 = 0 (host syncs inside a captured period), on both
  ``native_contacts`` values; the eager run is compared too (MuJoCo is non-NATIVE on GPU: informational, D9);
- eager == graph bitwise on the D9 GPU population (all-NATIVE: two FakeSolver entries under WP4's Scheduler, health);
- a MuJoCo + VBD scene under WP4's Scheduler captures with PAR-14 = 0 (residual loop unrolled, non-NATIVE);
- WP5's cloth drop under SchedulerADMM replays within the graph node budget (it segfaulted at 1.19 M nodes);
- a key whose one exchange is over the budget is never launched and runs eagerly (cloth drop, MPM sand).
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import warp as wp

from alloy.core.registry import isolated_registry
from alloy.testing import integration as I
from alloy.testing import scene as S

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.timeout(1200)]

DEV = os.environ.get("ALLOY_DEVICE", "cuda:0")
CONTROL_DT = 0.02
PERIODS = 5
CMD = "PYTHONPATH=. python -m pytest tests/integration/test_graph_gpu.py -q -m gpu"


@pytest.fixture(autouse=True)
def _cuda():
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    with isolated_registry(empty=True):
        from alloy.scene.builtin_plugins import register_builtin_solvers

        register_builtin_solvers()
        yield


def _peg_built(worlds: int, mode: str):
    contact = pytest.importorskip("alloy.testing.contact")
    if not hasattr(contact, "peg_hole_scene"):
        pytest.skip("WP3 peg_hole_scene is not on main yet (branch wp3)")
    from alloy.contact.precision import size_l2

    scene = contact.peg_hole_scene(worlds, device=DEV, mode=mode)
    dt = size_l2(scene.spec, m_eff=scene.peg_mass, design_force=50.0, contact_speed=0.05,
                 control_dt=CONTROL_DT).dt
    entry = I.entry_spec(scene.model, "mujoco", dt, integrator="euler", cone="elliptic",
                         impratio=scene.spec.mj_impratio, nconmax=256, njmax=1024, iterations=100, ls_iterations=50)
    return I.built_from_model(scene.model, [entry], control_dt=CONTROL_DT, precision_pairs=(scene.spec,))


def _run(built, periods: int = PERIODS, **overrides):
    from alloy.integration import build_simulation

    sim = build_simulation(built, "standard", **overrides)
    for _ in range(periods):
        sim.step_period()
    wp.synchronize_device(DEV)
    return sim


@pytest.mark.parametrize("native", ["mujoco", "newton"])
def test_peg_hole_graph_par14(native):
    graph = _run(_peg_built(16, "sdf"), native_contacts=native, graph="auto")
    eager = _run(_peg_built(16, "sdf"), native_contacts=native, graph="off")
    stats, rep = graph.graph_stats(), graph.report()
    q_g, q_e = graph.state.body_q.numpy(), eager.state.body_q.numpy()
    diff = float(np.max(np.abs(q_g - q_e)))
    gap = rep["gap"]
    I.record("PAR-14", float(graph.host_syncs), "pass" if graph.host_syncs == 0 else "fail", CMD,
             scene="peg_hole_sdf", native=native, graph=stats, steps=graph.steps,
             eager_graph_max_abs_body_q=diff, invalid=list(rep["health"].invalid_worlds),
             gap_p99_m=None if gap is None else float(np.max(gap.pair_p99_m)))
    assert type(graph.contact_layer).__name__ == "AlloyContactLayer" and graph.gap_monitor is not None
    n, chunk = graph.clock.steps_per_period, max(stats["chunks"])   # one key: counted with one-exchange graphs first
    first = n - 2                     # SCR-WP10-3: period 1 runs 2 eager exchanges, then n - 2 one-exchange graphs
    assert stats["mode"] == "graph" and stats["captures"] >= 1
    assert stats["launches"] == first + (PERIODS - 1) * (n // chunk)
    assert graph.host_syncs == 0
    assert np.isfinite(q_g).all() and not rep["health"].invalid_worlds


def _native_built(worlds: int):
    from alloy.testing.coupling import fake_drop_model

    model = fake_drop_model([0.12 + 0.01 * (w % 7) for w in range(worlds)], device=DEV)
    entries = [I.entry_spec(model, name, 1.0 / 480.0) for name in ("src", "dst")]
    return I.built_from_model(model, entries, control_dt=CONTROL_DT)


def test_native_population_eager_equals_graph():
    from alloy.testing.coupling import fake_scheduler

    out = {}
    for graph in ("off", "auto"):
        built = _native_built(64)
        sim = _run(built, stepper=fake_scheduler(built.model), graph=graph)
        out[graph] = (sim, I.digests(sim))
    (eager, d_e), (graph, d_g) = out["off"], out["auto"]
    same = bool(np.array_equal(d_e, d_g))
    I.record("WP12-EAGER-GRAPH-GPU", float(same), "pass" if same else "fail", CMD, scene="fake_drop_scheduler",
             worlds=64, graph=graph.graph_stats(), population="D9 all-NATIVE")
    assert graph.graph_stats()["mode"] == "graph" and graph.host_syncs == 0
    assert same, f"worlds differ: {np.nonzero((d_e != d_g).any(1))[0]}"


def test_mpm_graph_par14():
    """MPM sand under graph capture (wiring sets a persistent sparse grid, max_active_cell_count)."""
    card, lib = {n: (c, lb) for n, c, lb in S.mixed_scenes()}["sand_heap"]
    graph = _run(I.assemble(card, lib, DEV, 4), periods=3, graph="auto")
    q = graph.state.particle_q.numpy()
    I.record("PAR-14", float(graph.host_syncs), "pass" if graph.host_syncs == 0 else "fail", CMD,
             scene="sand_heap_mpm", graph=graph.graph_stats(), notes=graph.parts.notes)
    assert graph.graph_stats()["mode"] == "graph" and graph.host_syncs == 0
    assert np.isfinite(q).all()


def _coupled_card(name: str):
    if name == "box_on_cloth":
        return S.scene([S.table(), S.box(), S.cloth()], [S.place("table"), S.place("box", z=0.051),
                                                         S.place("towel", z=0.12)], worlds=16)
    return {n: (c, lb) for n, c, lb in S.mixed_scenes()}[name]


@pytest.mark.parametrize("name", ["box_on_cloth", "box_cloth_cable"])
def test_coupled_scheduler_graph_par14(name):
    """MuJoCo + VBD under WP4's Scheduler: box on cloth, and box + cloth + cable, whose strong pair puts dt_x on the
    finest grid step (1,024 exchanges per period, captured as 64-exchange chunks)."""
    card, lib = _coupled_card(name)
    graph = _run(I.assemble(card, lib, DEV, 16), periods=3, graph="auto")
    eager = _run(I.assemble(card, lib, DEV, 16), periods=3, graph="off")
    q_g, q_e = graph.state.particle_q.numpy(), eager.state.particle_q.numpy()
    I.record("PAR-14", float(graph.host_syncs), "pass" if graph.host_syncs == 0 else "fail", CMD,
             scene=f"{name}_coupled", stepper=type(graph.stepper).__name__, graph=graph.graph_stats(),
             eager_graph_max_abs_particle_q=float(np.max(np.abs(q_g - q_e))))
    assert type(graph.stepper).__name__ in ("Scheduler", "SchedulerADMM")
    assert graph.graph_stats()["mode"] == "graph" and graph.host_syncs == 0
    assert np.isfinite(q_g).all()


def test_admm_graph_par14():
    """WP4's SchedulerADMM (MuJoCo ball + link, VBD payload + cloth; Jacobi ADMM, residual stop) in graph mode."""
    from alloy.testing.coupling import admm_model, admm_scheduler

    out = {}
    for graph in ("auto", "off"):
        model = admm_model([0.01 * (w % 4) for w in range(16)], device=DEV)
        entries = [I.entry_spec(model, name, 1.0 / 480.0) for name in ("mjc", "vbd")]
        built = I.built_from_model(model, entries, control_dt=CONTROL_DT)
        out[graph] = _run(built, periods=3, stepper=admm_scheduler(model, gamma=0.0), graph=graph)
    graph, eager = out["auto"], out["off"]
    diff = float(np.max(np.abs(graph.state.body_q.numpy() - eager.state.body_q.numpy())))
    I.record("PAR-14", float(graph.host_syncs), "pass" if graph.host_syncs == 0 else "fail", CMD,
             scene="admm_ball_cloth", stepper=type(graph.stepper).__name__, graph=graph.graph_stats(),
             eager_graph_max_abs_body_q=diff)
    assert type(graph.stepper).__name__ == "SchedulerADMM"
    assert graph.graph_stats()["mode"] == "graph" and graph.host_syncs == 0
    assert np.isfinite(graph.state.body_q.numpy()).all()


@pytest.mark.timeout(2400)
def test_admm_cloth_drop_graph_within_node_budget():
    """WP5's MuJoCo + VBD case (SCR-DC5-4): ``demo_cloth_drop`` under SchedulerADMM (20 ADMM iterations unrolled
    under capture, 16 VBD substeps) captured 16 exchanges as one 1.19 M-node graph, and cudaGraphLaunch segfaulted
    on the first replay (period 3, after the eager warm-up of each substep key). The node budget sizes the chunk."""
    from alloy.cards.library import CardLibrary

    lib = CardLibrary()
    graph = _run(I.assemble(lib.scene("demo_cloth_drop"), lib, DEV, 8), periods=6, graph="auto")
    stats, rep = graph.graph_stats(), graph.report()
    within = bool(stats["sizes"]) and all(c == 1 or c * v <= graph.graph_node_budget for c, v in stats["sizes"])
    I.record("WP12-GRAPH-NODES", float(max(stats["exchange_nodes"])),
             "pass" if within and graph.host_syncs == 0 else "fail", CMD, scene="demo_cloth_drop_admm",
             stepper=type(graph.stepper).__name__, graph=stats, budget=graph.graph_node_budget,
             invalid=list(rep["health"].invalid_worlds))
    assert type(graph.stepper).__name__ == "SchedulerADMM"
    assert stats["mode"] == "graph" and stats["launches"] > 0 and graph.host_syncs == 0 and within
    assert np.isfinite(graph.state.particle_q.numpy()).all()


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("name", ["demo_cloth_drop", "sand_heap"])
def test_over_budget_key_runs_eager(name):
    """A key whose one exchange is over the node budget (forced here: a budget of 1 node) is captured and counted,
    never launched, and runs eagerly (WP5's ordered-VBD exchange segfaulted in its first launch). The dropped graph
    must not break the eager periods after it (a never-launched probe capture broke MPM's next graph launch)."""
    from alloy.cards.library import CardLibrary
    from alloy.integration import build_simulation

    if name == "sand_heap":
        card, lib = {n: (c, lb) for n, c, lb in S.mixed_scenes()}[name]
        built = I.assemble(card, lib, DEV, 4)
    else:
        lib = CardLibrary()
        built = I.assemble(lib.scene(name), lib, DEV, 8)
    sim = build_simulation(built, "standard", graph="auto")
    sim.graph_node_budget = 1
    for _ in range(4):
        sim.step_period()
    wp.synchronize_device(DEV)
    stats, q = sim.graph_stats(), sim.state.particle_q.numpy()
    ok = stats["launches"] == 0 and len(stats["eager_keys"]) == stats["captures"] > 0 and np.isfinite(q).all()
    I.record("WP12-GRAPH-OVER-BUDGET", float(len(stats["eager_keys"])), "pass" if ok else "fail", CMD, scene=name,
             stepper=type(sim.stepper).__name__, graph=stats, budget=sim.graph_node_budget)
    assert ok and sim.host_syncs == 0
