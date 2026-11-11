# SPDX-License-Identifier: Apache-2.0
"""A held ``Sim.apply_wrench`` is outside work in health's energy check (EASE-TEST red, 2026-11-09; CPU).

Before: a held 2 N push on a 0.51 kg ball read as ENERGY_JUMP after about 0.2 s (0.12 J of work), the world went
INVALID + LATCHED and froze, and the fix text blamed overlaps. Health books force plug-ins' work
(``test_health_force_work.py``); the wrench's work is now booked the same way (``health_work.WrenchWork``).

Laws: per world, the booked work equals the ball's gain in kinetic plus potential energy (a push, a torque, and a
world with no wrench that books nothing), every world stays OK; an energy gain the wrench does not explain still
trips ENERGY_JUMP; ``sim.reset()`` drops the held wrench and its booking.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.precision.health_work import WRENCH_ATTR

M, R, G = 0.51, 0.05, 9.81
FORCE = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])          # world 0: the tester's push
TORQUE = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.01, 0.0]])        # world 2: a twist that rolls it


def _ball(worlds: int = 3):
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_sphere("ball", R, pos=(0, 0, R), mass=M)
    sim = alloy.build(scene, device="cpu")
    sim.step(1)                                    # a step before the wrench: the wrench changes a running sim
    return sim


def _energy(sim) -> np.ndarray:
    """Per world: kinetic (translation + spin of a solid ball) plus potential energy [J]."""
    q, qd = sim.engine.state.body_q.numpy(), sim.engine.state.body_qd.numpy()
    spin = 0.4 * M * R * R
    return 0.5 * M * (qd[:, :3] ** 2).sum(1) + 0.5 * spin * (qd[:, 3:] ** 2).sum(1) + M * G * q[:, 2]


def _w_cum(sim) -> np.ndarray:
    return sim.engine.health._ref.w_cum.numpy().astype(np.float64)


@pytest.mark.parametrize("route", ["mjc", "warp"])      # the users' CPU default (MuJoCo C) and the suite's
def test_a_held_wrench_is_booked_work_not_an_energy_jump(route, monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    sim = _ball()
    e0, w0 = _energy(sim), _w_cum(sim)
    sim.apply_wrench("ball", force=FORCE, torque=TORQUE)
    sim.run(0.6)                                   # unbooked, world 0 froze at x = 0.12 m (INVALID at t ~ 0.2 s)
    h = sim.health()
    assert [w["severity"] for w in h["worlds"]] == ["OK"] * 3 and sim.check()["ok"]
    assert sim.state(["ball"])["ball.pos"][0, 0] > 0.45                    # 2.8 m/s² rolling for 0.6 s: 0.5 m
    gain, booked = _energy(sim) - e0, _w_cum(sim) - w0
    assert booked[0] == pytest.approx(gain[0], rel=1e-2) and booked[0] > 0.9
    assert booked[2] == pytest.approx(gain[2], rel=1e-2) and booked[2] > 5e-3
    assert booked[1] == 0.0                                                # no wrench, no work


@pytest.mark.parametrize("route", ["mjc", "warp"])
def test_an_unexplained_gain_still_trips_energy_jump_beside_a_held_wrench(route, monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    sim = _ball(worlds=2)
    sim.apply_wrench("ball", force=(2.0, 0.0, 0.0))
    sim.run(0.3)
    jqd = sim.engine.state.joint_qd.numpy()
    jqd[2] += 2.0                                  # world 0: a 2 m/s upward kick nothing explains
    sim.engine.state.joint_qd.assign(jqd)
    sim.run(0.05)
    w0, w1 = sim.health()["worlds"]
    assert w0["severity"] == "INVALID" and "ENERGY_JUMP" in w0["flags"]
    assert w1["severity"] == "OK"                  # the same push without the kick stays booked


def test_a_reset_drops_the_held_wrench_and_its_work() -> None:
    sim = _ball(worlds=1)
    sim.apply_wrench("ball", force=(2.0, 0.0, 0.0))
    sim.run(0.1)
    ww = getattr(sim.engine, WRENCH_ATTR)
    assert ww.body_f.numpy()[:, 0].max() == 2.0
    sim.reset()
    assert not ww.body_f.numpy().any()
    w0 = _w_cum(sim)
    sim.run(0.2)
    assert abs(sim.state(["ball"])["ball.pos"][0, 0]) < 1e-3                # nothing pushes it now
    assert _w_cum(sim)[0] == w0[0] and sim.health()["worlds"][0]["severity"] == "OK"
