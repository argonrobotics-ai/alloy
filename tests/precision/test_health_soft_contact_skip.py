# SPDX-License-Identifier: Apache-2.0
"""Health does not judge the soft contacts of SPH / MPM particles (OSS-17's cup, 2026-11-07; CPU).

SPH and implicit MPM handle shapes themselves and never read Newton's soft contacts. One of those unused rows, signed
wrongly by a Warp parity query at a degenerate point (WP2's trace), put a particle 4.3 mm clear of a mesh cup's wall
6.28 mm "deep": liquid at rest went INVALID (PEN_FAIL) at t = 0.119 s. VBD-family particles, whose solvers do read
the rows, are still judged.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import warp as wp

from alloy.integration.wiring import _health_soft_contacts
from alloy.precision.health_setup import unjudged_world


def test_unjudged_world_marks_only_the_skipped_particles() -> None:
    world = wp.array(np.array([0, 0, 1, 1, -1], np.int32), dtype=wp.int32, device="cpu")
    assert unjudged_world(world, np.array([1, 2])).numpy().tolist() == [0, -1, -1, 1, -1]
    assert world.numpy().tolist() == [0, 0, 1, 1, -1]                  # the model's own array is untouched


def test_wiring_skips_the_particles_of_sph_and_mpm_entries_only() -> None:
    entry = lambda cap, parts: SimpleNamespace(capability=cap, particles=parts)     # noqa: E731
    built = SimpleNamespace(entries=[entry("mujoco", ()), entry("sph", (0, 1)), entry("vbd", (2, 3)),
                                     entry("mpm", (4,))])
    got = []
    _health_soft_contacts(SimpleNamespace(skip_soft_contacts=got.extend), built)
    assert got == [0, 1, 4]
    _health_soft_contacts(None, built)                                 # no health: nothing to do
    _health_soft_contacts(SimpleNamespace(skip_soft_contacts=got.extend), SimpleNamespace(entries=[entry("vbd", (2,))]))
    assert got == [0, 1, 4]                                            # no SPH / MPM: nothing skipped


def test_liquid_at_rest_in_a_mesh_cup_is_not_pen_fail() -> None:
    import alloy
    from alloy.api import meshes

    v, f = meshes.open_cylinder(inner_radius=0.03, depth=0.06, wall=0.004)
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_mesh("cup", vertices=v, faces=f, collision="sdf", static=True)
    scene.add_liquid("water", size=(0.05, 0.05, 0.03), pos=(0, 0, 0.02), shape="cylinder")
    sim = alloy.build(scene, device="cpu")
    assert (sim.engine.health._soft_world.numpy() < 0).all()           # every particle here is SPH
    sim.run(0.15)                                                      # INVALID at t = 0.119 s before
    assert sim.health()["worlds"][0]["severity"] == "OK" and sim.check()["ok"]
