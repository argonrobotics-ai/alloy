# SPDX-License-Identifier: Apache-2.0
"""Joint springs that yield (SCR-WP7-15 P2): the closed-form laws J1 and J2, off is bitwise, and restores (reset, a
mid-yield snapshot) put the set and the spring reference back, on the MuJoCo C route and MuJoCo Warp (CPU)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import alloy
from alloy.plugins.joint_yield import laws

ROUTES = ("mjc", "warp")


@pytest.fixture(params=ROUTES)
def route(request, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", request.param)
    return request.param


def _rig_sim(rig: laws.Rig, *, worlds: int = 2, yield_on: bool = True):
    scene = alloy.Scene(worlds=worlds, dt=laws.STEP_S)
    scene.add_robot("rig", mjcf=rig.mjcf(), pos=(0, 0, 0.5))
    if yield_on:
        key = "torque" if rig.kind == "hinge" else "force"
        scene.set_joint_yield("rig/j", **{key: rig.limit}, hardening=rig.hardening)
    return alloy.build(scene, device="cpu")


def _drive(sim, speed: float, seconds: float) -> np.ndarray:
    sim.set_targets({"rig/j": np.full(sim.worlds, speed)}, kind="velocity")
    return sim.run(seconds, record=["rig/j"])["rig/j.q"]


def _route_name(sim) -> str:
    return type(sim.engine.solvers[0]).__name__


def test_j1_torsion_spring_cycled_past_yield(route):
    """Forward overload, reverse overload, an elastic poke: each release rests at the closed-form set (permanent set,
    reverse yield at -(τ_y0 + k_h ε̄p), no new set inside the yield band), and ε̄p reads back as the closed form; the
    same at a control period of 4 MuJoCo steps."""
    r = laws.run(laws.J1_RIG)
    assert r.meta["route"] == ("MjcCoreSolver" if route == "mjc" else "SolverMuJoCo")
    assert r.ok
    assert (np.abs(r.closed[0]) > 0.05).all() and (r.closed[1] < 0).all()      # yielded forward, then back past 0
    np.testing.assert_array_equal(r.closed[2], r.closed[1])                      # the poke stays elastic
    np.testing.assert_allclose(r.rests, r.closed, rtol=0, atol=1e-6)
    np.testing.assert_allclose(r.plastic_strain, r.closed_strain, rtol=1e-5, atol=0)
    # 4 MuJoCo steps per control period: the joint still yields after every physics step, so the rests equal the
    # closed form from the one-step run's extremes (WP7's review: the spring reference never lags the set)
    coarse = laws.run(laws.J1_RIG, dt=4 * laws.STEP_S)
    assert coarse.ok
    np.testing.assert_allclose(coarse.rests, r.closed, rtol=0, atol=1e-6)
    np.testing.assert_allclose(coarse.plastic_strain, r.closed_strain, rtol=1e-5, atol=0)
    assert float(np.abs(coarse.closed - r.closed).max()) > 1e-6      # its own frames miss the peaks


def test_j2_clip_overloaded_and_released(route):
    """A sprung slide pushed past its yield force and let go rests at q_max - F_y / k (perfectly plastic)."""
    rig = laws.J2_RIG
    r = laws.run(rig)
    assert r.ok
    np.testing.assert_allclose(r.closed[0], r.extremes[0] - rig.limit / rig.k, rtol=0, atol=1e-12)
    assert rig.springref != 0.0 and (r.closed[0] - rig.springref > 5e-4).all()     # preloaded, then set
    np.testing.assert_allclose(r.rests[0], r.extremes[0] - rig.limit / rig.k, rtol=0, atol=1e-7)


def test_off_is_bitwise(route):
    """No yield set: no plug-in. A yield never reached: the trajectory is bit for bit the stock one (the spring
    reference written each step is the build value, and the C route writes nothing)."""
    stock = _rig_sim(laws.J1_RIG, yield_on=False)
    assert getattr(getattr(stock.engine, "plugins", None), "plugins", {}).get("joint_yield") is None
    high = _rig_sim(dataclasses.replace(laws.J1_RIG, limit=1e3))
    assert "joint_yield" in high.engine.plugins.plugins
    a, b = _drive(stock, 0.5, 0.3), _drive(high, 0.5, 0.3)
    np.testing.assert_array_equal(a, b)
    assert laws.J1_RIG.k * float(np.abs(a).max()) > 2 * laws.J1_RIG.limit      # J1's own yield would have run
    assert float(high.state(["rig/j"])["rig/j.plastic_strain"].max()) == 0.0


def test_reset_puts_the_spring_back(route):
    """After a yield, ``reset`` restores the set to 0 and the next step's spring reference with it: the same drive
    gives the same trajectory bit for bit."""
    sim = _rig_sim(laws.J2_RIG)
    first = _drive(sim, 0.1, 0.15)
    assert float(sim.state(["rig/j"])["rig/j.plastic_strain"].min()) > 1e-4
    sim.reset()
    assert float(sim.state(["rig/j"])["rig/j.plastic_strain"].max()) == 0.0
    np.testing.assert_array_equal(_drive(sim, 0.1, 0.15), first)


def test_mid_yield_restore(route):
    """A snapshot taken while yielding, restored after more yield: the set reads back as saved, the spring reference
    follows it (the re-run tracks the first run), and two re-runs from the snapshot agree bit for bit."""
    sim = _rig_sim(laws.J2_RIG)
    _drive(sim, 0.1, 0.1)
    plugin = sim.engine.plugins["joint_yield"]
    saved = plugin.q_p.numpy().copy()
    assert (saved > 1e-4).all()
    snap = sim.engine._snapshot()
    snap.save(sim.engine.state)
    first = sim.run(0.05, record=["rig/j"])["rig/j.q"]
    assert (plugin.q_p.numpy() > saved + 1e-4).all()
    reruns = []
    for _ in range(2):
        snap.restore(sim.engine.state)
        np.testing.assert_array_equal(plugin.q_p.numpy(), saved)
        reruns.append(sim.run(0.05, record=["rig/j"])["rig/j.q"])
    np.testing.assert_array_equal(reruns[0], reruns[1])
    np.testing.assert_allclose(reruns[0], first, rtol=0, atol=1e-7)          # a stale reference would be 1e-4 off


def test_model_refresh_keeps_the_set(route):
    """``notify_model_changed`` re-syncs MuJoCo's spring references from the build attribute (MuJoCo Warp) and
    rebuilds the C route's per-world models: the set is written again before the very next step. Refreshed mid-yield
    and again once released and settled, the run matches a twin that was never refreshed, set and coordinate."""
    from alloy._vendor import newton

    def refresh(sim) -> None:
        for s in sim.engine.solvers:
            s.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    rig = laws.J2_RIG
    twin, sim = _rig_sim(rig), _rig_sim(rig)
    runs = {}
    for name, x in (("twin", twin), ("sim", sim)):
        a = _drive(x, 0.1, 0.15)
        if x is sim:
            refresh(sim)
        b = _drive(x, 0.0, laws.SETTLE_S)
        if x is sim:
            refresh(sim)
        runs[name] = np.concatenate([a, b, _drive(x, 0.0, 0.2)])
    end = sim.engine.plugins["joint_yield"].q_p.numpy()
    assert (end > 5e-4).all()
    np.testing.assert_allclose(end, twin.engine.plugins["joint_yield"].q_p.numpy(), rtol=0, atol=1e-9)
    np.testing.assert_allclose(runs["sim"], runs["twin"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(runs["sim"][-1], rig.springref + end[:, 0], rtol=0, atol=1e-7)
