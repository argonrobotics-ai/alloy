# SPDX-License-Identifier: Apache-2.0
"""Kinematic parts make no contacts with static shapes or with each other (alloy.scene.fixed_pairs), CPU.

An outside tester's kinematic ``add_open_cylinder("cup")`` (25 kinematic boxes) resting on the ground made 296
MuJoCo contacts with it and among its own parts. Their corners overlapped (START_PEN), and beside a robot they
overflowed the contact buffer: CONTACT_OVERFLOW, every world INVALID, and MuJoCo Warp's warning ~120 times per step.
Neither side of such a contact can move the other, so the assembler filters those shape pairs.
- The cup resting on the ground, still or driven, beside a free box, on both CPU routes (MuJoCo C, MuJoCo Warp):
  the contacts are the box's alone, as many as without the cup (was 296 with it), health OK with no START_PEN,
  check ok.
- A moving body still meets it: a ball dropped into the kinematic cup lands on its floor.
- MuJoCo Warp prints no overflow warning for the overflows health flags (CONTACT_OVERFLOW, ROW_OVERFLOW,
  UNCONVERGED): those bits of warn_overflow are off, and the overflow word health reads is still written.
- A scene with no kinematic body gets no filter pairs, so it builds and steps as before.
"""

from __future__ import annotations

import pytest

import alloy
from alloy.scene import fixed_pairs as FIX

CUP = {"inner_radius": 0.03, "depth": 0.06, "wall": 0.004}


def _contacts(sim: alloy.Sim) -> int:
    mj = next(s for s in sim.engine.solvers if hasattr(s, "mj_model"))
    if hasattr(mj, "datas"):
        return int(mj.datas[0].ncon)
    return int(mj.mjw_data.nacon.numpy().sum())


@pytest.mark.parametrize("route", ["mjc", "warp"])
@pytest.mark.parametrize("driven", [False, True])
def test_a_kinematic_cup_on_the_ground_makes_no_contacts_with_it(route, driven, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    counts = []
    for cup in (True, False):
        sc = alloy.Scene(dt=0.01)
        sc.add_ground()
        sc.add_box("block", size=(0.03, 0.03, 0.03), pos=(0.2, 0.0, 0.015), mass=0.1)   # something to step
        if cup:
            sc.add_open_cylinder("cup", **CUP, pos=(0, 0, 0), kinematic=True)
        sim = alloy.build(sc, device="cpu")
        if cup and driven:
            sim.drive("cup", velocity=(0.05, 0.0, 0.0), duration=0.2)
        sim.step(20)
        counts.append(_contacts(sim))
        if cup:
            with_cup = sim
    assert counts[0] == counts[1], counts
    sim = with_cup
    health = sim.health()
    worlds = health["worlds"] if isinstance(health, dict) and "worlds" in health else health
    assert all(w["severity"] == "OK" and not w["flags"] for w in worlds), worlds
    assert sim.check()["ok"], sim.check()["problems"]


def test_a_ball_dropped_into_the_kinematic_cup_lands_on_its_floor():
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_open_cylinder("cup", **CUP, pos=(0, 0, 0), kinematic=True)
    sc.add_sphere("ball", 0.01, pos=(0.0, 0.0, 0.05), mass=0.02)
    sim = alloy.build(sc, device="cpu")
    sim.step(50)
    z = float(sim.state(["ball"], world=0)["ball.pos"][2])
    assert z == pytest.approx(CUP["wall"] + 0.01, abs=1.5e-3, rel=0), z         # on the floor's top, not through it
    assert sim.check()["ok"], sim.check()["problems"]


def test_mujoco_warp_prints_no_overflow_that_health_flags(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    from alloy.precision.health_adapters import _mjw_bits

    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_box("block", size=(0.03, 0.03, 0.03), pos=(0, 0, 0.015), mass=0.1)
    sim = alloy.build(sc, device="cpu")
    mj = next(s for s in sim.engine.solvers if hasattr(s, "mj_model"))
    flagged = 0
    for bits in _mjw_bits():
        flagged |= bits
    assert int(mj.mjw_model.opt.warn_overflow) & flagged == 0


def test_a_scene_without_kinematic_bodies_gets_no_filter_pairs():
    from alloy._vendor import newton

    b = newton.ModelBuilder()
    b.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.01)
    body = b.add_body()
    b.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02)
    assert FIX.filter_fixed_pairs(b) == 0 and not b.shape_collision_filter_pairs
    kin = b.add_body(is_kinematic=True)
    b.add_shape_box(kin, hx=0.02, hy=0.02, hz=0.02)
    b.add_shape_box(kin, hx=0.01, hy=0.01, hz=0.01)
    assert FIX.filter_fixed_pairs(b) == 2                     # the kinematic body's two shapes vs the static one
