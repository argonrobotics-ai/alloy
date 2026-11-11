# SPDX-License-Identifier: Apache-2.0
"""A MuJoCo body hitting a free particle object mid-air gives it the momentum it loses (gaps-now #10a slice A,
``alloy.coupling.record_momentum``; known gap coupled-record-damping).

Mid-air there is no support, so the body and the particles together change momentum exactly as in a second world where
the body falls free high above them (same route: the coupled cloth itself falls 2 % faster than g on main, 2026-11-08).
VBD's own feedback is its records' force at the last iterate, which is not the impulse an unconverged solve gave the
particles. These two scenes on the suite's Warp CPU route, worst excess against what the particles took
(2026-11-08): with that feedback, a 2 kg ball into an 11 x 11 cloth 3.42 mN·s against 1.60 (214 %) and C3's 0.515 kg
platen onto its 16 g pad 9.59 against 1.76 (543 %); with ``record_momentum``, 0.13 against 1.63 (7.8 %) and 0.015
against 2.17 (0.7 %). The law: within 10 % of what the particles took and 1 % of the body's m V0."""

from __future__ import annotations

import numpy as np
import pytest

import alloy

G = 9.81
LIMIT = 0.01                  # of the body's m V0
SHARE = 0.10                  # of the momentum the particles took


def _launch(sim, body: str, v0: float) -> np.ndarray:
    """Give the free body a downward speed v0 in every world (its free joint and its body velocity)."""
    eng = sim.engine
    bodies = np.asarray(sim.index.per_world(sim.index.body(body), "body")).ravel()
    qd, jqd = eng.state.body_qd.numpy(), eng.state.joint_qd.numpy()
    child, start = eng.model.joint_child.numpy(), eng.model.joint_qd_start.numpy()
    for b in bodies:
        qd[b][2] = -v0
        jqd[int(start[int(np.flatnonzero(child == b)[0])]) + 2] = -v0
    eng.state.body_qd.assign(qd)
    eng.state.joint_qd.assign(jqd)
    return bodies


def _excess(sim, bodies: np.ndarray, mass: float, seconds: float) -> tuple[float, float]:
    """World 0 hits, world 1 (the body high above) falls free on the same route. Returns the worst |z momentum change
    of world 0 - that of world 1| over the run and world 0's particles' gain over world 1's (N·s)."""
    eng = sim.engine
    mp = eng.model.particle_mass.numpy().astype(np.float64)
    pw = eng.model.particle_world.numpy()

    def p(w: int) -> tuple[float, float]:
        v = eng.state.particle_qd.numpy()[:, 2]
        return mass * float(eng.state.body_qd.numpy()[bodies[w]][2]), float((mp[pw == w] * v[pw == w]).sum())

    start, worst = (p(0), p(1)), 0.0
    t0 = sim.t
    while sim.t - t0 < seconds - 1e-9:
        sim.step(1)
        (b0, c0), (b1, c1) = p(0), p(1)
        worst = max(worst, abs((b0 + c0 - sum(start[0])) - (b1 + c1 - sum(start[1]))))
    return worst, (c0 - start[0][1]) - (c1 - start[1][1])


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_ball_into_a_cloth_mid_air():
    """A 2 kg ball at 0.31 m/s (a 5 mm drop) into an 11 x 11 cloth, 0.3 m up (WP4's sparse impact)."""
    scene = alloy.Scene(worlds=2, dt=0.001)
    scene.add_cloth("sheet", size=(0.3, 0.3), pos=(0.0, 0.0, 0.3), resolution=11, mass=0.05, self_contact=False)
    z = 0.3 + 0.002 + 0.03 + 0.0002
    scene.add_sphere("ball", 0.03, pos=np.array([[0.0, 0.0, z], [0.0, 0.0, z + 0.2]]), mass=2.0,
                     material="steel_carbon")
    sim = alloy.build(scene, device="cpu")
    v0 = (2 * G * 0.005) ** 0.5
    worst, gained = _excess(sim, _launch(sim, "ball", v0), 2.0, 0.02)
    assert gained < -1.0e-3                      # the cloth was hit (main and fix: ~3 mN·s)
    assert worst < SHARE * -gained, f"excess {worst * 1e3:.3f} mN·s against the cloth's {-gained * 1e3:.3f}"
    assert worst < LIMIT * 2.0 * v0


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_platen_onto_a_soft_pad_mid_air():
    """C3's 0.515 kg platen at 0.1 m/s onto its 40 x 40 x 10 mm silicone pad, 0.3 m up (no support)."""
    r, h, bh = 0.000571, 0.01, 0.02
    scene = alloy.Scene(worlds=2, dt=0.0025)
    scene.add_box("pad", size=(0.04, 0.04, h), pos=(0.0, 0.0, 0.3), material="tissue_fat_silicone")
    z = 0.3 + h / 2 + 2 * r + r + bh / 2
    scene.add_box("platen", size=(0.06, 0.06, bh), pos=np.array([[0.0, 0.0, z], [0.0, 0.0, z + 0.2]]), mass=0.5153,
                  material="wood_beech")
    sim = alloy.build(scene, device="cpu")
    worst, gained = _excess(sim, _launch(sim, "platen", 0.1), 0.5153, 0.02)
    assert gained < -0.5e-3                      # the pad was hit (main and fix: ~1.9 mN·s)
    assert worst < SHARE * -gained, f"excess {worst * 1e3:.3f} mN·s against the pad's {-gained * 1e3:.3f}"
    assert worst < LIMIT * 0.5153 * 0.1
