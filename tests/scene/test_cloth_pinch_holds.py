# SPDX-License-Identifier: Apache-2.0
"""Grasp law (user ruling 2026-11-10 04:35Z; WP2): a pinched cloth holds a load below its friction limit.

Rig (public API): a 4 cm cloth patch (25 vertices, 2 mm thick) hangs between two static pads that squeeze every
vertex by 1.5 um and carries its own weight along them. The pinch's friction capacity C [N] is measured in the rig,
not taken from a formula (the route's contact law is a soft barrier at um squeezes): a patch heavier than the pads
hold slides at g - C / M. The load fraction is r = M g / C.

What VBD's friction does without a stick: smoothed Coulomb on the slip u within one step, f = mu N (2 - u / eps_u)
u / eps_u below eps_u = friction_epsilon x dt. A load r < 1 is carried at the u where f equals it, in every step: a
steady creep

    v = friction_epsilon x (1 - sqrt(1 - r)),

whatever the step and the squeeze. On the route's friction_epsilon 1e-2 m/s that is 2.93 mm/s at r = 0.5: 29 mm in
10 s (measured 2.930 mm/s; 0.513 / 1.634 / 5.530 mm/s at r = 0.1 / 0.3 / 0.8, the formula to 0.05 %). A smaller
friction_epsilon is not a hold: 1e-4 leaves 0.02-0.06 mm/s whatever r, 1e-5 is worse (0.2 mm/s).

The law, with stick friction (`alloy.solvers.vbd.stick`, opt-in ALLOY_FRICTION_STICK=1, WP11): below the limit the
patch settles by a fixed offset (0.65 / 2.06 / 3.76 / 8.4 / 38 um at r = 0.1 / 0.3 / 0.5 / 0.8 / 0.95: each
contact's pre-slip plus the patch's own stretch) and then does not move. Bound: under 1 um of drift from 1 s to 10 s,
at r = 0.1, 0.5 and 0.8 (measured 0.01 um on the stick's first version, 6094febd6; without the stick 4.6, 26 and 50
mm). Above the limit it slides as before: the capacity measured by sliding is the same with and without the stick
(0.4492 against 0.4496 N: kinetic friction is untouched).

Fast tests run 1.5 s; the 10 s law is slow; the GPU law holds 64 worlds for 2 s.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy

G = 9.81
THICK, SIDE, CELLS, SQUEEZE = 0.002, 0.04, 4, 1.5e-6
STICK_ENV = "ALLOY_FRICTION_STICK"
DRIFT_BOUND = 1.0e-6          # m from 1 s to 10 s with the stick (module docstring)
SLIDE_MASS = 0.078            # kg: heavier than the pinch holds (C is about 0.45 N)

pytestmark = pytest.mark.timeout(900)     # 1.5 s of this rig is about a minute on an idle CPU


def _scene(mass: float, worlds: int = 1) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01, worlds=worlds)
    half, face = 0.005, THICK / 2 - SQUEEZE
    for name, sign in (("pad_a", 1.0), ("pad_b", -1.0)):          # 1 m tall: a sliding patch stays between the pads
        scene.add_box(name, size=(SIDE + 0.02, 2 * half, 1.0), pos=(0.0, sign * (face + half), 0.5), static=True,
                      friction=0.5)
    scene.add_cloth("patch", size=(SIDE, SIDE), pos=(0.0, 0.0, 0.9), rpy=(math.pi / 2, 0.0, 0.0), resolution=CELLS,
                    thickness=THICK, mass=mass, self_contact=False)
    return scene


def _drops(mass: float, seconds: float, device: str = "cpu", worlds: int = 1) -> tuple[np.ndarray, np.ndarray, float]:
    """(t [frames], mean drop of the patch per world [frames, worlds] in m, friction_epsilon) every 0.1 s."""
    sim = alloy.build(_scene(mass, worlds), device=device)
    z0 = sim.state(["patch"])["patch.points"][..., 2].mean(axis=1)
    t, drop = [], []
    for k in range(1, int(round(seconds / sim.dt)) + 1):
        sim.step(1)
        if k % 10 == 0:
            t.append(k * sim.dt)
            drop.append(z0 - sim.state(["patch"])["patch.points"][..., 2].mean(axis=1))
    assert sim.check()["ok"]
    eps = float(sim.engine.stepper.solver.friction_epsilon)
    sim.close()
    return np.array(t), np.array(drop), eps


def _capacity() -> float:
    """The pinch's friction capacity [N]: a patch too heavy to hold slides at g - C / M (fitted over 0.3 s)."""
    sim = alloy.build(_scene(SLIDE_MASS), device="cpu")
    z0 = sim.state(["patch"])["patch.points"][0][:, 2].mean()
    t, d = [], []
    for k in range(1, 31):
        sim.step(1)
        t.append(k * sim.dt)
        d.append(float(z0 - sim.state(["patch"])["patch.points"][0][:, 2].mean()))
    sim.close()
    accel = 2.0 * float(np.polyfit(t, d, 2)[0])
    assert 0.2 * G < accel < 0.8 * G, f"the slide must be a clear one: {accel:.2f} m/s2"
    return SLIDE_MASS * (G - accel)


def _rate(t: np.ndarray, drop: np.ndarray, start: float) -> np.ndarray:
    """Steady drop speed per world [m/s] from ``start`` on."""
    late = t >= start
    return np.polyfit(t[late], drop[late], 1)[0]


@pytest.fixture(scope="module")
def plain_capacity() -> float:
    mp = pytest.MonkeyPatch()
    mp.delenv(STICK_ENV, raising=False)
    try:
        return _capacity()
    finally:
        mp.undo()


def test_without_the_stick_a_held_patch_creeps_as_the_smoothed_law_says(plain_capacity, monkeypatch):
    """Today's friction, pinned: at half the limit the patch leaves at friction_epsilon (1 - sqrt(1 - r))."""
    monkeypatch.delenv(STICK_ENV, raising=False)
    r = 0.5
    t, drop, eps = _drops(r * plain_capacity / G, 1.5)
    want = eps * (1.0 - math.sqrt(1.0 - r))
    assert _rate(t, drop, 0.5)[0] == pytest.approx(want, rel=0.02), "the creep formula no longer describes main"
    assert want * 10.0 > 0.02                                    # 2 cm in 10 s: the ridge leaves the fingers


def test_with_the_stick_a_held_patch_stops(plain_capacity, monkeypatch):
    """The grasp law, short form: at half the limit the patch settles in under 0.5 s and then does not move."""
    monkeypatch.setenv(STICK_ENV, "1")
    t, drop, _ = _drops(0.5 * plain_capacity / G, 1.5)
    late = t >= 0.5
    moved = float(np.ptp(drop[late, 0]))
    assert moved < 0.5 * DRIFT_BOUND, f"moved {1e6 * moved:.3f} um between 0.5 and 1.5 s"   # float32 at z 0.9 m: 0.06 um
    assert 0.0 < drop[-1, 0] < 5.0e-5, f"pre-slip {1e6 * drop[-1, 0]:.2f} um"


def test_the_stick_leaves_the_sliding_limit_where_it_was(plain_capacity, monkeypatch):
    monkeypatch.setenv(STICK_ENV, "1")
    assert _capacity() == pytest.approx(plain_capacity, rel=0.02)


@pytest.mark.slow
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("r", [0.1, 0.5, 0.8])
def test_a_pinched_patch_holds_its_load_for_10_s(plain_capacity, monkeypatch, r):
    """The grasp law: under DRIFT_BOUND of motion from 1 s to 10 s at a tenth, half and four fifths of the limit."""
    monkeypatch.setenv(STICK_ENV, "1")
    t, drop, _ = _drops(r * plain_capacity / G, 10.0)
    late = t >= 1.0
    moved = float(np.ptp(drop[late, 0]))
    assert moved < DRIFT_BOUND, f"r {r}: moved {1e6 * moved:.3f} um from 1 s to 10 s"


@pytest.mark.gpu
@pytest.mark.timeout(3600)
def test_64_pinched_patches_hold_on_a_gpu(plain_capacity, monkeypatch):
    """The short law in 64 worlds on a GPU, against the same run without the stick."""
    mass = 0.5 * plain_capacity / G
    monkeypatch.delenv(STICK_ENV, raising=False)
    t, plain, eps = _drops(mass, 2.0, "cuda:0", 64)
    monkeypatch.setenv(STICK_ENV, "1")
    _, held, _ = _drops(mass, 2.0, "cuda:0", 64)
    late = t >= 1.0
    moved = np.ptp(held[late], axis=0)
    print(f"GPU, 64 worlds: without the stick {1e3 * _rate(t, plain, 1.0).min():.3f}-"
          f"{1e3 * _rate(t, plain, 1.0).max():.3f} mm/s (formula {1e3 * eps * (1 - math.sqrt(0.5)):.3f}); with it "
          f"the worst world moved {1e6 * moved.max():.3f} um from 1 s to 2 s")
    assert _rate(t, plain, 1.0).min() > 0.5 * eps * (1.0 - math.sqrt(0.5))        # every world creeps without it
    assert moved.max() < DRIFT_BOUND
