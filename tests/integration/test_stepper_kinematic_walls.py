# SPDX-License-Identifier: Apache-2.0
"""SCR-OSS3-5 / SCR-OSS22-6 (WP9's fix, WP4): unowned KINEMATIC bodies join the rigid entry in a coupled scene, so
the liquid sees a kinematic container as a wall, still or driven. Before, a free box beside a kinematic tray made the
scene coupled (mujoco + sph), the tray belonged to no entry, and the water fell through it (median z 0.115 -> 0.0009).
CPU, one world."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.integration.stepper import adopt_kinematic, dynamic_entries


def _tray_scene() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("tray", inner=(0.06, 0.06, 0.05), wall=0.004, pos=(0, 0, 0.1), kinematic=True)
    scene.add_liquid("water", size=(0.056, 0.056, 0.02), pos=(0, 0, 0.115))
    scene.add_box("free", size=(0.04, 0.04, 0.04), pos=(0.2, 0, 0.02), mass=0.1)       # makes the scene coupled
    return scene


def _median_z(sim) -> float:
    return float(np.median(sim.state(world=0)["water.points"][:, 2]))


def test_the_rigid_entry_owns_the_kinematic_tray():
    sim = alloy.build(_tray_scene(), device="cpu")
    built = sim.engine.built
    dyn = dynamic_entries(built)
    flags = built.model.body_flags.numpy()
    from alloy._vendor import newton

    tray = [b for b in range(built.model.body_count) if int(flags[b]) & int(newton.BodyFlags.KINEMATIC)]
    assert tray and not any(set(tray) & set(e.bodies) for e in dyn)                 # unowned before
    hub = next(e for e in adopt_kinematic(dyn, built.model) if e.capability == "mujoco")
    assert set(tray) <= set(hub.bodies)


@pytest.mark.parametrize("drive", [False, True], ids=["still", "driven"])
def test_kinematic_tray_holds_water_beside_a_free_box(drive):
    sim = alloy.build(_tray_scene(), device="cpu")
    z0 = _median_z(sim)
    if drive:
        sim.drive("tray", velocity=(0.0, 0.0, 0.05), duration=0.3)
    sim.step(30)
    dz = _median_z(sim) - z0
    if drive:
        assert dz > 0.8 * 0.05 * 0.3, dz             # the water rides up with the tray (0.015 m)
    else:
        assert dz > -0.003, dz                       # settles a little (SPH alone: -0.0011), never falls through
    assert sim.check()["ok"], sim.check()["problems"]


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_driven_kinematic_plate_carries_cloth_beside_a_free_box():
    """WP9's caveat (1), VBD as the destination: a cloth on a kinematic plate driven up at 0.05 m/s for 0.3 s rides
    up with it (CPU probe 11-06: +0.011 m after the 4 mm settle; the plate +0.015 m)."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("plate", size=(0.2, 0.2, 0.01), pos=(0, 0, 0.1), kinematic=True)
    scene.add_cloth("sheet", size=(0.15, 0.15), pos=(0, 0, 0.11), resolution=10, mass=0.02)
    scene.add_box("free", size=(0.04, 0.04, 0.04), pos=(0.3, 0, 0.02), mass=0.1)
    sim = alloy.build(scene, device="cpu")
    z0 = float(np.median(sim.state(world=0)["sheet.points"][:, 2]))
    sim.drive("plate", velocity=(0.0, 0.0, 0.05), duration=0.3)
    sim.step(30)
    s = sim.state(world=0)
    assert float(s["plate.pos"][2]) == pytest.approx(0.115, abs=1.0e-3)
    assert float(np.median(s["sheet.points"][:, 2])) - z0 > 0.008
    assert sim.check()["ok"], sim.check()["problems"]
