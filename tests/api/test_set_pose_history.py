# SPDX-License-Identifier: Apache-2.0
"""``sim.set_pose`` starts a teleported body at rest in a VBD scene (Gemini audit S1-013).

VBD (cloth, soft bodies) and its rod solver keep each body's pose from the last step (``body_q_prev``) and read a
body found elsewhere as having moved there in one step. ``set_pose`` wrote the states only, so a 0.25 m teleport read
as 200 m/s and froze the world (SPEED_FAIL, LATCHED), and an 18.75 mm one gave a resting platen 15 m/s and dragged
the towel under it 1.8 mm with health ok. ``_drive.hold`` now puts those history rows at the new pose."""

from __future__ import annotations

import numpy as np
import pytest

import alloy.api as alloy


def _pusher_scene(kind: str):
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    if kind == "cloth":
        sc.add_cloth("item", size=(0.10, 0.10), pos=(0.0, 0.0, 0.02), resolution=10, mass=0.05)
    elif kind == "soft_vbd":
        sc.add_box("item", size=(0.05, 0.05, 0.04), pos=(0.0, 0.0, 0.025), material="pu_foam_sponge", rigid=False,
                   mass=0.05)
    else:
        sc.add_rod("item", length=0.16, radius=0.008, pos=(0.0, 0.0, 0.02), segments=6, route="rod_vbd")
    sc.add_box("pusher", size=(0.05, 0.12, 0.04), pos=(-0.15, 0.0, 0.03), kinematic=True)
    return alloy.build(sc, worlds=2, device="cpu")


@pytest.mark.compile_heavy
@pytest.mark.parametrize("kind", ["cloth", "soft_vbd", "rod_vbd"])
def test_a_long_teleport_does_not_read_as_speed(kind):
    sim = _pusher_scene(kind)
    try:
        sim.step(1)
        sim.set_pose("pusher", pos=(0.0, 0.25, 0.03), world=1)        # 0.29 m, clear of the item (a rod runs +x)
        sim.step(1)
        h = sim.health()
        assert h["ok"] and h["worlds"][1]["flags"] == [], h["worlds"][1]
        assert np.array_equal(sim.state(["pusher"], world=1)["pusher.vel"], np.zeros(3, np.float32))
    finally:
        sim.close()


@pytest.mark.compile_heavy
def test_a_short_teleport_onto_cloth_gives_no_kick():
    sc = alloy.Scene(dt=0.00125)
    sc.add_ground()
    sc.add_cloth("towel", size=(0.12, 0.12), pos=(0.0, 0.0, 0.002), resolution=12, mass=0.05)
    sc.add_box("platen", size=(0.08, 0.08, 0.025), pos=(0.0, 0.0, 0.040), kinematic=True)
    sim = alloy.build(sc, worlds=2, device="cpu")
    try:
        dx = 15.0 * float(sim.engine.stepper.dt_x)                 # 18.75 mm: under SPEED_FAIL's 20 m/s per step
        sim.set_pose("platen", pos=(0.0, 0.0, 0.0128), world=0)
        sim.set_pose("platen", pos=(-dx, 0.0, 0.0128), world=1)
        sim.step(8)
        for w in (0, 1):
            sim.set_pose("platen", pos=(0.0, 0.0, 0.0128), world=w)
        x0 = sim.state(["towel"])["towel.points"][:, :, 0].mean(axis=1)
        sim.step(1)
        assert np.array_equal(sim.state(["platen"])["platen.vel"], np.zeros((2, 3), np.float32)), "15 m/s before"
        sim.step(7)
        drift = sim.state(["towel"])["towel.points"][:, :, 0].mean(axis=1) - x0
        assert abs(float(drift[1])) < 2e-4, f"towel dragged {drift[1] * 1e3:.3f} mm (1.81 mm before the fix)"
        assert sim.health()["ok"]
    finally:
        sim.close()
