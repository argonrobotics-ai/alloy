# SPDX-License-Identifier: Apache-2.0
"""rinse_drain (Gym-Anything drain-wash-compartment): the pot sink's drains, levers, verdict and a shallow run."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from alloy.plugins.harness.scene import build_simulation
from alloy.tasks.rinse_drain import DrainWash, Readout, read, run, verdict
from alloy.tasks.rinse_levers import lever_angles
from alloy.tasks.rinse_sink import BOWLS, DRAIN_CD, DRAIN_D, FILL, LEVER_RANGE, PotSink, drain_edge

G = 9.81


def _torricelli_s(area: float, h0: float, h1: float, f: float = 1.0) -> float:
    """Time for an open tank of ``area`` to fall from head ``h0`` to ``h1`` through the basket drain at opening f."""
    a = DRAIN_CD * f * math.pi * DRAIN_D ** 2 / 4.0
    return 2.0 * area / (a * math.sqrt(2.0 * G)) * (math.sqrt(h0) - math.sqrt(h1))


def test_wash_bowl_drains_by_torricelli():
    """The network alone: the full wash bowl through its open drain follows dV/dt = -Cd·a·sqrt(2gV/A) (law F-14)."""
    from alloy.plugins.fluid import Network  # noqa: PLC0415

    sink = PotSink()
    net = Network(sink.nodes(), sink.edges())
    fs = net.compile("cpu", 1, gravity=G)
    op = fs.opening.numpy()
    op[:] = 0.0
    op[0, net.edge_index(drain_edge("wash"))] = 1.0
    fs.opening.assign(op)
    dt, t, marks = 0.01, 0.0, [0.20, 0.10, 0.03, 0.005]
    got = {}
    while marks:
        fs.step(dt)
        t += dt
        h = fs.amount(fs.pressure.numpy(), fs.volume.numpy())[0, net.node_index("wash")] / sink.area
        while marks and h <= marks[0]:
            got[marks.pop(0)] = t
    for h1, t1 in got.items():
        assert t1 == pytest.approx(_torricelli_s(sink.area, FILL, h1), rel=0.02, abs=dt), h1


@pytest.fixture(scope="module")
def lever_sim():
    task = DrainWash()
    plumb, hand = task.plugins()
    sim = build_simulation(task.scene(1), [plumb, hand], "cpu")
    return task, plumb, hand, sim


def test_lever_sets_the_drain_and_holds(lever_sim):
    """The hand twists the rinse lever half open and lets go: the drain opens by the lever angle, the discharge is
    Cd·a·f·sqrt(2gh), and the stiffened joint friction holds the lever (creep < 0.1° in 2 s)."""
    task, plumb, hand, sim = lever_sim
    target, grip = np.zeros((1, 3), np.float32), np.zeros((1, 3), np.float32)
    target[0, 1], grip[0, 1] = 0.25 * math.pi, 1.0
    hand.set_input("target", target)
    hand.set_input("grip", grip)
    for _ in range(30):
        sim.step_period()
    hand.set_input("grip", np.zeros_like(grip))
    sim.step_period()
    q0 = lever_angles(sim.model, sim.state.body_q.numpy(), hand.levers.numpy())[0, 1]
    for _ in range(40):
        sim.step_period()
    q1 = lever_angles(sim.model, sim.state.body_q.numpy(), hand.levers.numpy())[0, 1]
    assert q0 == pytest.approx(0.25 * math.pi, abs=math.radians(1.0))
    assert abs(q1 - q0) < math.radians(0.1)
    f, net = plumb.fluid, plumb.fluid.net
    e = net.edge_index(drain_edge("rinse"))
    assert f.opening.numpy()[0, e] == pytest.approx(q1 / LEVER_RANGE[1], abs=1e-5)
    h = f.amount(f.pressure.numpy(), f.volume.numpy())[0, net.node_index("rinse")] / PotSink().area
    q_law = DRAIN_CD * (q1 / LEVER_RANGE[1]) * math.pi * DRAIN_D ** 2 / 4.0 * math.sqrt(2.0 * G * h)
    assert -f.discharge.numpy()[0, e] == pytest.approx(q_law, rel=0.01)
    assert f.opening.numpy()[0, net.edge_index(drain_edge("wash"))] == 0.0


def _readout(**kw: object) -> Readout:
    base = dict(t=0.0, volume_l=np.array([[70.0, 70.0, 70.0]]), drained_l=np.zeros((1, 3)), floor_ml=np.zeros(1),
                waste_rad=np.zeros((1, 3)), faucet_gpm=np.zeros(1), flow_gpm=np.zeros(1), held=np.zeros(1, bool),
                water_rel=np.zeros(1), species_kg=np.zeros(1), species_rel=np.zeros(1), finite=np.ones(1, bool))
    base.update(kw)
    return Readout(**base)


def test_verdict_rules():
    """The gates and the score of ``task_set/evaluator.py:97-108`` and ``task_set/scoring.py:55-66``."""
    start = _readout()
    done = _readout(volume_l=np.array([[0.3, 70.0, 70.0]]), drained_l=np.array([[69.7, 0.0, 0.0]]))
    v = verdict(start, done, done, 0)
    assert v.success and v.score == 1.0
    assert verdict(start, start, start, 0).score == 0.0                            # untouched reset scores 0
    cases = {"drained": dict(volume_l=np.array([[0.6, 70.0, 70.0]])),
             "waste_closed": dict(waste_rad=np.array([[0.0, math.radians(6.0), 0.0]])),
             "others_kept": dict(drained_l=np.array([[69.7, 1.2, 0.0]])),
             "floor_dry": dict(floor_ml=np.array([60.0])),
             "water_off_and_hands_free": dict(held=np.ones(1, bool)),
             "no_simulation_fault": dict(water_rel=np.array([1e-6]))}
    for gate, change in cases.items():
        bad = replace(done, **change)
        v = verdict(start, bad, bad, 0)
        assert v.failed() == [gate], gate
        if gate != "no_simulation_fault":
            assert v.score is not None and 0.0 <= v.score < 1.0
    leaky = replace(done, water_rel=np.array([1e-6]))
    assert verdict(start, leaky, leaky, 0).score is None                           # open ledger: unverifiable
    half = _readout(volume_l=np.array([[35.25, 70.0, 70.0]]))
    v = verdict(start, half, half, 0)
    assert v.terms["drained"] == pytest.approx(0.5) and v.score == pytest.approx(0.9 * 0.5)   # closed, released
    # progress is graded on the live state, the gates on the settled one (evaluator.py:110-118): a waste left
    # open drains during the settle, which passes "drained" but earns no extra drained credit
    open_settled = replace(done, waste_rad=np.array([[math.radians(30.0), 0.0, 0.0]]))
    v = verdict(start, half, open_settled, 0)
    assert v.failed() == ["waste_closed"] and v.terms["drained"] == pytest.approx(0.5)
    assert v.score == pytest.approx(0.9 * 0.25)


@pytest.mark.compile_heavy
def test_shallow_wash_drains_in_every_world():
    """The whole task on a shallow wash (6 mm, 1.7 L) in 2 worlds with the lever 30 % and fully open: both pass, the
    water and surfactant ledgers close, the hands-off drain time follows Torricelli at each world's measured lever
    opening to 3 %, and the levers nobody touched rest alike in both worlds."""
    sink = PotSink(fill=(0.006, FILL, FILL))
    res = run(DrainWash(sink=sink, openings=(0.3, 1.0)), worlds=2)
    assert res.success == [True, True], res.failed
    assert res.water_rel_max < 1e-9
    assert max(abs(a) for w in res.waste_deg for a in w) <= 5.0
    for f, t, (v0, v1) in zip(res.open_measured, res.wait_s, res.wait_l, strict=True):
        want = _torricelli_s(sink.area, v0 * 1e-3 / sink.area, v1 * 1e-3 / sink.area, f)
        assert t == pytest.approx(want, rel=0.03), (f, t, want)
    assert res.open_measured[0] == pytest.approx(0.3, abs=0.02)
    assert res.waste_deg[0][1:] == res.waste_deg[1][1:]


def test_bowls_and_levers_match_the_source():
    """Pot sink constants against Gym-Anything (``params.py:170-175``, ``scene.py:399-414``)."""
    sink = PotSink()
    assert sink.area * FILL * 1e3 == pytest.approx(70.2, abs=0.1)     # 0.4532 x 0.6096 x 0.254 m
    assert BOWLS == ("wash", "rinse", "sanitize")
    assert LEVER_RANGE == (0.0, 0.5 * math.pi)
    assert sink.nodes()[0].species == (0.75,)


def test_not_judged_gates_are_none_and_a_judged_failure_still_fails():
    """floor_dry (overflow ledger only), the hands part and the empty intact gate are not judged (AB-gym-3)."""
    start = _readout()
    done = _readout(volume_l=np.array([[0.3, 70.0, 70.0]]), drained_l=np.array([[69.7, 0.0, 0.0]]))
    v = verdict(start, done, done, 0)
    for gate in ("floor_dry", "water_off_and_hands_free", "intact_and_not_dropped"):
        assert v.criteria[gate][0] is None, gate
    assert v.success and v.score == 1.0 and v.judged == (4, 7)
    leaking = replace(done, faucet_gpm=np.array([1.0]))
    v = verdict(start, leaking, leaking, 0)
    assert v.failed() == ["water_off_and_hands_free"] and not v.success
    assert verdict(start, replace(done, floor_ml=np.array([60.0])), replace(done, floor_ml=np.array([60.0])),
                   0).failed() == ["floor_dry"]
