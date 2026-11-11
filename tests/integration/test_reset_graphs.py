# SPDX-License-Identifier: Apache-2.0
"""``IntegratedSimulation.reset`` keeps the captured period graphs when that is safe
(``alloy.integration.reset_graphs``). On the CPU, with period graphs forced on:
- the law: a run with masked resets, a latched world reset and a full reset is the same bit for bit (states, health
  records) with the graphs kept as with them dropped, and the kept run captures nothing new at a reset; also in a
  scene with a randomized value, whose reset draws what the model holds already and writes nothing (TRAIN-FIX 3a);
- the decision: a redraw that changes a randomized value, a direct ``notify_model_changed``,
  ``reset_keeps_graphs = False`` and entries outside the law drop the graphs as before.
The CUDA rows are tests/integration/test_reset_graphs_gpu.py.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

import alloy
from alloy._vendor import newton
from alloy.integration import profiles as P
from alloy.integration import reset_graphs as RG

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""
W = 4


@pytest.fixture(autouse=True)
def _graphs_on(monkeypatch):
    for k, v in list(P.PROFILES.items()):
        monkeypatch.setitem(P.PROFILES, k, dataclasses.replace(v, graph="on"))


def cubes(randomized: bool = False) -> alloy.Scene:
    s = alloy.Scene(worlds=W, dt=0.01)
    s.add_ground(friction=1.0)
    s.add_box("c", size=(0.2,) * 3, pos=np.array([[1.0 * w, 0.0, 0.3 + 0.05 * w] for w in range(W)]), density=500.0,
              friction=0.8)
    if randomized:
        s.randomize("c", "friction", 0.6, 1.0)
    return s


def randomized_cubes() -> alloy.Scene:
    return cubes(randomized=True)


def arm() -> alloy.Scene:
    s = alloy.Scene(worlds=W, dt=0.01)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    return s


def _mask(sim: alloy.Sim, *worlds: int) -> wp.array:
    m = np.zeros(W, dtype=np.int32)
    m[list(worlds)] = 1
    return wp.array(m, dtype=wp.int32, device=sim.engine.model.device)


def _snap(sim: alloy.Sim) -> tuple[bytes, ...]:
    state = sim.state()
    return tuple(np.ascontiguousarray(state[k]).tobytes() for k in sorted(state)) + (
        sim.engine.health.record.numpy().tobytes(),)


def _episode(scene: alloy.Scene, keep: bool, kick: bool) -> tuple[list, dict]:
    """Steps with a masked reset, a world kicked INVALID and reset, and a full reset: (snapshots, graph stats)."""
    sim = alloy.build(scene, device="cpu", seed=0)
    eng = sim.engine
    assert eng.use_graph
    eng.reset_keeps_graphs = keep
    if "arm/shoulder" in sim.names()["joints"]:
        sim.set_targets({"shoulder": np.linspace(0.2, 0.8, W)})
    out = []
    sim.step(6)
    eng.reset(_mask(sim, 1))
    sim.step(5)
    out.append(_snap(sim))
    if kick:
        force = np.zeros((W, 3))
        force[2, 2] = 1e6
        sim.apply_wrench("c", force=force)
        sim.step(12)
        assert 2 in {p["world"] for p in sim.check()["problems"]}          # latched INVALID (SPEED_FAIL), frozen
        sim.apply_wrench("c", force=np.zeros((W, 3)))
        out.append(_snap(sim))
        eng.reset(_mask(sim, 2))
        sim.step(6)
        assert 0.36 < float(sim.state(world=2)["c.pos"][2]) < 0.399        # unlatched: it falls from its start again
        out.append(_snap(sim))
    eng.reset(_mask(sim, 0, 3))
    sim.step(3)
    eng.reset(None)
    sim.step(4)
    out.append(_snap(sim))
    return out, eng.graph_stats()


@pytest.mark.parametrize(("make", "kick"), [(cubes, True), (arm, False), (randomized_cubes, True)])
def test_kept_graphs_replay_the_dropped_run_bit_for_bit(make, kick) -> None:
    """Free cubes on the ground (one latches INVALID and is reset), a servo arm and the cubes with a randomized
    friction: every snapshot is equal, the kept run never drops at a reset and captures fewer graphs."""
    kept, stats_k = _episode(make(), True, kick)
    dropped, stats_d = _episode(make(), False, kick)
    assert kept == dropped
    resets = 4 if kick else 3
    assert (stats_k["reset_kept"], stats_k["reset_dropped"]) == (resets, 0)
    assert (stats_d["reset_kept"], stats_d["reset_dropped"]) == (0, resets)
    assert stats_k["captures"] < stats_d["captures"], (stats_k, stats_d)


def _fast_episode(keep: bool) -> tuple[list, dict]:
    """profile="fast" (cheap checks every 10 steps, period snapshots saved after check steps only): masked resets
    right before a check step and right after it, and of a world latched INVALID."""
    sim = alloy.build(cubes(), device="cpu", seed=0, profile="fast")
    eng, out = sim.engine, []
    eng.reset_keeps_graphs = keep
    per = eng.clock.steps_per_period
    sim.step(10 // per)                                  # the next step is a check step
    eng.reset(_mask(sim, 1))
    sim.step(1)                                          # right after a check
    eng.reset(_mask(sim, 3))
    sim.step(1)
    out.append(_snap(sim))
    force = np.zeros((W, 3))
    force[2, 2] = 1e6
    sim.apply_wrench("c", force=force)
    sim.step(12)
    assert 2 in {p["world"] for p in sim.check()["problems"]}
    sim.apply_wrench("c", force=np.zeros((W, 3)))
    out.append(_snap(sim))
    eng.reset(_mask(sim, 2))
    sim.step(9)
    eng.reset(_mask(sim, 0))
    sim.step(5)
    out.append(_snap(sim))
    return out, eng.graph_stats()


def test_the_fast_profile_keeps_its_graphs_bit_for_bit() -> None:
    """Checks every 10 steps: the check pattern is in the graph key and the period snapshots are saved after check
    steps only (WP7's note)."""
    kept, stats_k = _fast_episode(True)
    dropped, stats_d = _fast_episode(False)
    assert kept == dropped
    assert (stats_k["reset_kept"], stats_k["reset_dropped"], stats_d["reset_dropped"]) == (4, 0, 4)


def test_a_late_health_setter_drops_the_graphs_at_the_next_reset() -> None:
    """Health's setters that rebind an array (after the first step): the next reset drops, as it always did."""
    sim = alloy.build(cubes(), device="cpu", seed=0)
    sim.step(2)
    sim.engine.health.skip_soft_contacts([])
    sim.engine.reset(_mask(sim, 1))
    assert sim.engine.reset_graphs == {"kept": 0, "dropped": 1}


def test_a_masked_reset_captures_nothing_new() -> None:
    """After every check pattern of the schedule is captured (a few periods), resets add no capture."""
    sim = alloy.build(cubes(), device="cpu", seed=0)
    sim.step(12)
    before = sim.engine.graph_stats()["captures"]
    for w in (0, 1, 2):
        sim.engine.reset(_mask(sim, w))
        sim.step(1)
    stats = sim.engine.graph_stats()
    assert stats["captures"] == before and stats["reset_kept"] == 3, stats


def test_a_randomized_reset_drops_the_graphs_only_when_its_redraw_changes_a_value() -> None:
    """A reset draws the reset worlds' values at the scene's draw seed: the values the model holds, so nothing is
    written and the graphs are kept (TRAIN-FIX 3a). A world redrawn at another seed in between gets its values back:
    that redraw notifies the solvers, which may rebind arrays, and drops the graphs as before."""
    from alloy.scene.randomize import apply_randomization

    sim = alloy.build(cubes(randomized=True), device="cpu", seed=0)
    sim.step(2)
    sim.engine.reset(_mask(sim, 1))
    sim.step(1)
    sim.engine.reset(_mask(sim, 2))
    assert sim.engine.reset_graphs == {"kept": 2, "dropped": 0}
    sim.step(1)
    apply_randomization(sim.engine.built, 7, wp.array([3], dtype=wp.int32))
    sim.engine.reset(_mask(sim, 3))
    assert sim.engine.reset_graphs == {"kept": 2, "dropped": 1}


def test_a_direct_notify_drops_the_graphs_at_the_next_reset_only() -> None:
    """Code that edits the model and notifies a solver itself (alloy.episode.layout): the next reset drops, the one
    after keeps."""
    sim = alloy.build(cubes(), device="cpu", seed=0)
    sim.step(2)
    sim.engine.solvers[0].notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    sim.engine.reset(_mask(sim, 1))
    assert sim.engine.reset_graphs == {"kept": 0, "dropped": 1}
    sim.step(1)
    sim.engine.reset(_mask(sim, 1))
    assert sim.engine.reset_graphs == {"kept": 1, "dropped": 1}


def test_a_simulation_without_graphs_gets_no_wrappers(monkeypatch) -> None:
    for k, v in list(P.PROFILES.items()):
        monkeypatch.setitem(P.PROFILES, k, dataclasses.replace(v, graph="off"))
    sim = alloy.build(cubes(), device="cpu", seed=0)
    assert not sim.engine.use_graph and "notify_model_changed" not in vars(sim.engine.solvers[0])
    sim.step(1)
    sim.engine.reset(_mask(sim, 1))
    assert sim.engine.reset_graphs == {"kept": 0, "dropped": 0}


def _fake(**over) -> SimpleNamespace:
    solver = type("SolverMuJoCo", (), {})()
    base = {"reset_keeps_graphs": True, "_notify_watched": True, "_model_notified": False, "solvers": (solver,),
            "stepper": object(), "plugins": None, "topology": None}
    return SimpleNamespace(**{**base, **over})


def test_only_entries_the_law_covers_keep_their_graphs() -> None:
    assert RG.keep(_fake()) == (True, "")
    vbd, synced = type("SolverVBDMomentum", (), {})(), type("SyncedMuJoCo", (), {})()
    assert RG.keep(_fake(solvers=(synced,)))[0] and RG.keep(_fake(solvers=(vbd,)))[0]
    ok, why = RG.keep(_fake(solvers=(synced, vbd)))               # a coupled scene: refused until its GPU row is in
    assert not ok and "more than one entry solver" in why
    for over, word in (({"reset_keeps_graphs": False}, "off"), ({"_model_notified": True}, "model changed"),
                       ({"_notify_watched": False}, "not watched"),
                       ({"solvers": (type("SolverDFSPH", (), {})(),)}, "SolverDFSPH"),
                       ({"solvers": (type("SolverImplicitMPM", (), {})(),)}, "SolverImplicitMPM"),
                       ({"solvers": (type("SolverFilm", (), {})(),)}, "SolverFilm"),
                       ({"solvers": (type("SolverVBDRod", (), {})(),)}, "SolverVBDRod"),
                       ({"stepper": type("SchedulerADMM", (), {})()}, "ADMM"),
                       ({"plugins": object()}, "plug-in"), ({"topology": object()}, "topology")):
        ok, why = RG.keep(_fake(**over))
        assert not ok and word in why, (over, why)
