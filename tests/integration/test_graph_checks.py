# SPDX-License-Identifier: Apache-2.0
"""Release red (coordinator ruling 2026-11-09, scr/WP7.md): health's checks run at their own steps inside replayed
period graphs (alloy.integration.graph_checks), on CPU graphs. A spy kernel launched with a check records the
device step it ran at, so a replayed graph's checks are seen, not only the Python call at capture."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.integration import profiles as P

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


@wp.kernel
def _seen(step: wp.array(dtype=wp.int64), n: wp.array(dtype=wp.int32), at: wp.array(dtype=wp.int64)):
    i = wp.atomic_add(n, 0, 1)
    if i < at.shape[0]:
        at[i] = step[0]


def spy(health, method: str):
    """Wrap ``health.<method>`` so each run (eager or replayed) records ``health._step``; returns a reader."""
    n = wp.zeros(1, dtype=wp.int32, device=health.device)
    at = wp.full(256, -1, dtype=wp.int64, device=health.device)
    inner = getattr(health, method)

    def wrapped(*a, **k):
        inner(*a, **k)
        wp.launch(_seen, dim=1, inputs=[health._step, n, at], device=health.device)
    setattr(health, method, wrapped)
    return lambda: sorted(int(s) for s in at.numpy()[: int(n.numpy()[0])])


@pytest.fixture(params=["on", "off"])
def graph(request, monkeypatch):
    for k, v in list(P.PROFILES.items()):
        monkeypatch.setitem(P.PROFILES, k, dataclasses.replace(v, graph=request.param))
    return request.param


def cubes(worlds: int = 2) -> alloy.Scene:
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground(friction=1.0)
    s.add_box("c", size=(1.0,) * 3, pos=np.array([[3.0 * w, 0.0, 0.5001] for w in range(worlds)]), density=10.0,
              friction=1.0)
    return s


def test_fast_profile_checks_every_tenth_step(graph) -> None:
    """profile="fast" (cheap checks every 10 steps): the checks run at every tenth step, with graphs as without (before
    the fix, a graph captured at step 2 never checked); at one step per period (main's rigid fast mode) from two
    graphs."""
    sim = alloy.build(cubes(), device="cpu", profile="fast")
    assert sim.engine.use_graph == (graph == "on")
    seen = spy(sim.engine.health, "_check_state")
    sim.step(35)
    assert seen() == list(range(0, sim.physics_steps, 10)) and len(seen()) >= 4
    if graph == "on" and sim.engine.clock.steps_per_period == 1:
        assert sim.engine.graph_stats()["graphs"] == 2, sim.engine.graph_stats()


def test_standard_full_checks_every_tenth_step_at_eight_steps_per_period(graph) -> None:
    """The default profile on a robot scene (8 steps of 1.25 ms per period; full checks every 10 steps): the joint
    and weld checks run at steps 0, 10, ..., 40 over 6 periods, with graphs as without."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    sim = alloy.build(s, device="cpu")
    assert sim.engine.clock.steps_per_period == 8
    seen = spy(sim.engine.health, "_check_joints")
    sim.step(6)
    assert seen() == [0, 10, 20, 30, 40]


def test_a_kicked_world_latches_invalid(graph) -> None:
    """The repro: world 1 kicked to over 20 m/s after step 12 latches INVALID at the check at step 20 and stays frozen
    there (finite, unmoved); world 0 goes on. Before the fix, under graphs, it flew to z = 8 km with sim.check() ok."""
    sim = alloy.build(cubes(), device="cpu", profile="fast")
    sim.step(12)
    sim.apply_wrench("c", force=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1e6]]))
    sim.step(10)
    problems = sim.check()["problems"]
    assert problems and all(p["world"] == 1 for p in problems), problems
    frozen = sim.state(world=1)["c.pos"].copy()
    sim.step(20)
    assert np.isfinite(frozen).all() and np.array_equal(sim.state(world=1)["c.pos"], frozen)
    assert abs(float(sim.state(world=0)["c.pos"][2]) - 0.5) < 2e-3


def test_graphs_match_eager_bitwise(monkeypatch) -> None:
    """CPU graphs replay the eager periods bit for bit, checks included: 30 periods of the fast profile."""
    out = {}
    for mode in ("off", "on"):
        for k, v in list(P.PROFILES.items()):
            monkeypatch.setitem(P.PROFILES, k, dataclasses.replace(v, graph=mode))
        sim = alloy.build(cubes(), device="cpu", profile="fast")
        sim.step(30)
        out[mode] = (sim.state()["c.pos"].copy(), sim.engine.health.record.numpy().copy())
    assert np.array_equal(out["off"][0], out["on"][0])
    assert out["off"][1].tobytes() == out["on"][1].tobytes()


class _Policy:
    level, every, full_every, skip = "full", 4, 6, ()


class _Health:
    cfg = _Policy()

    def is_check_step(self, step: int) -> bool:
        return step % self.cfg.every == 0


def test_full_checks_nest_inside_cheap_ones() -> None:
    """WP5: every 4 with full every 6: step 6 checks nothing (not a cheap step), step 12 runs the full checks."""
    from alloy.integration.graph_checks import pattern

    assert pattern(_Health(), 0, 13) == (2, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 2)


def _kinds_seen(sim, steps: int) -> list[int]:
    cheap, full = spy(sim.engine.health, "_check_state"), spy(sim.engine.health, "_check_joints")
    sim.step(steps // sim.engine.clock.steps_per_period)
    c, f = set(cheap()), set(full())
    return [2 if s in f else 1 if s in c else 0 for s in range(sim.physics_steps)]


@pytest.mark.parametrize("every", [(1, 10), (10, 10), (4, 6), (3, 5)])
def test_pattern_follows_update_eagerly(every, graph) -> None:
    """WP5's drift guard: what HealthMonitor.update really runs, step by step over 48 steps (eager and replayed),
    equals graph_checks.pattern; a host branch on ``step`` added to update() without _kind fails here."""
    from alloy.integration.graph_checks import pattern

    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    sim = alloy.build(s, device="cpu")
    h = sim.engine.health
    h.cfg = dataclasses.replace(h.cfg, level="full", every=every[0], full_every=every[1])
    seen = _kinds_seen(sim, 48)
    assert seen == list(pattern(h, 0, len(seen))), (seen, pattern(h, 0, len(seen)))


def test_a_pattern_over_the_node_budget_runs_eagerly(monkeypatch) -> None:
    """WP0's must-fix: a fresh pattern graph is counted before its first launch; one over the node budget runs that
    chunk eagerly (results still equal the all-eager run's) and the key's per-exchange count is raised (a one-exchange
    key then runs eagerly). CPU graphs report no node count, so a stand-in counts a graph with checks 1,100 nodes and
    one without 100 (budget 500)."""
    from alloy.integration import graph_checks as GC
    from alloy.integration import simulation as S

    tagged: list = []
    real = GC.chunk_graph

    def tag(sim, key, chunk, dt_x):
        g = real(sim, key, chunk, dt_x)
        tagged.append((g, any(GC.pattern(sim.health, sim.steps, chunk))))
        return g
    monkeypatch.setattr(GC, "chunk_graph", tag)
    monkeypatch.setattr(S, "chunk_graph", tag)
    monkeypatch.setattr(GC, "graph_nodes", lambda cg: next(1100 if c else 100 for g, c in tagged if g._graph is cg))
    calls: list = []
    monkeypatch.setattr(GC, "_eager", lambda sim, n, dt: (calls.append((sim.steps, n)),
                                                          [sim._exchange(dt) for _ in range(n)]))
    out = {}
    for mode in ("off", "on"):
        for k, v in list(P.PROFILES.items()):
            monkeypatch.setitem(P.PROFILES, k, dataclasses.replace(v, graph=mode))
        sim = alloy.build(cubes(), device="cpu", profile="fast")
        sim.engine.graph_node_budget = 500
        sim.step(25)
        out[mode] = sim
    assert np.array_equal(out["on"].state()["c.pos"], out["off"].state()["c.pos"])
    assert calls and all(any(s % 10 == 0 for s in range(a, a + n)) for a, n in calls), calls   # check chunks only
    eng = out["on"].engine                       # one exchange over the budget sends its key eager (as _fits does)
    assert max(eng.exchange_nodes.values()) == 1100 and list(eng.eager_keys.values()) == [1100]


def test_out_of_memory_after_the_first_chunk_runs_the_rest_eagerly(monkeypatch) -> None:
    """WP0's should-fix: a launch that runs out of device memory at chunk i > 0 runs the period's remaining exchanges
    eagerly (none twice, none lost) instead of raising."""
    from alloy.integration import graph_checks as GC

    for k, v in list(P.PROFILES.items()):
        monkeypatch.setitem(P.PROFILES, k, dataclasses.replace(v, graph="on"))
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    sim = alloy.build(s, device="cpu")
    sim.step(3)
    eng, real = sim.engine, GC.chunk_graph

    def failing(sim_, key, chunk, dt_x):
        g = real(sim_, key, chunk, dt_x)

        class _Fail:
            _graph = object()

            def launch(self):
                raise RuntimeError("Graph creation error: Warp CUDA error 2: out of memory")

            def invalidate(self):
                g.invalidate()
        return _Fail()
    monkeypatch.setattr(GC, "chunk_graph", failing)
    eng.max_graph_steps = 2                          # chunks of 2, so i > 0 happens inside a period
    steps = sim.physics_steps
    sim.step(1)
    assert sim.physics_steps == steps + 8 and sim.check()["ok"]


def test_lever_three_times_a_key_once_its_check_patterns_repeat() -> None:
    """WP5's rc3 red (GPU only, so no gate saw it): box_on_table steps 16 exchanges per period with checks every step
    and full ones every 10, so each period start mod 10 captures its own graph (5 patterns). Lever #3 times a key only
    on a period whose graphs all launched before (WP10's rule: never a capture or a first launch), so the first timing
    is the 7th period, and no graph is captured after the cycle. CPU graphs with a GraphChoice attached ("auto" times
    on CUDA only)."""
    from alloy.core.registry import isolated_registry
    from alloy.integration import build_simulation
    from alloy.integration.graph_choice import GraphChoice
    from alloy.scene.builtin_plugins import register_builtin_solvers
    from alloy.testing import integration as I
    from alloy.testing import scene as S

    with isolated_registry(empty=True):
        register_builtin_solvers()
        card, lib = S.box_on_table(4)
        sim = build_simulation(I.assemble(card, lib, "cpu", 4), "standard", graph="on")
        assert sim.clock.steps_per_period == 16 and (sim.health.cfg.every, sim.health.cfg.full_every) == (1, 10)
        sim.graph_choice = GraphChoice(margin=3.0)
        rows = []
        for _ in range(9):
            sim.step_period()
            rows.append((sim.graph_stats()["timed_keys"], sim.graph_stats()["graphs"]))
    assert [t for t, _ in rows] == [0] * 6 + [1] * 3, rows
    assert len({g for _, g in rows[5:]}) == 1, rows
