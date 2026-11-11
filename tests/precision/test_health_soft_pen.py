# SPDX-License-Identifier: Apache-2.0
"""SCR-SOFT-3-1 (coordinator ruling 2026-11-06 18:07Z): a soft body inside another soft body is PEN_FAIL (INVALID), so
``check()`` cannot say ok when a stack of soft pads falls through (CPU)."""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.precision.health import HealthMonitor
from alloy.precision.health_soft_pen import SoftPen
from alloy.precision.types import CheckPolicy, HealthFlags

CELL = 0.01   # [m]


def _block(b: newton.ModelBuilder, z: float, xy: float = 0.0) -> None:
    b.add_soft_grid(pos=wp.vec3(xy, xy, z), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=3, dim_y=3, dim_z=2,
                    cell_x=CELL, cell_y=CELL, cell_z=CELL, density=1000.0, k_mu=1.0e4, k_lambda=1.0e4, k_damp=0.0,
                    particle_radius=1.0e-3)


def _two_blocks(top_z: float) -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    _block(b, 0.0)
    _block(b, top_z, xy=0.5 * CELL)   # the lower block spans z 0..2 cm; half a cell off its lattice
    return b


def _poll(model: newton.Model):
    monitor = HealthMonitor(model, CheckPolicy(skip=frozenset({"energy"})))
    s = model.state()
    monitor.update(0, s, s, None)
    return monitor.poll()


def test_a_soft_body_inside_another_is_pen_fail_in_that_world_only() -> None:
    s = newton.ModelBuilder()
    s.add_world(_two_blocks(0.5 * CELL))      # world 0: the top block's particles at the lower block's cell centres
    s.add_world(_two_blocks(0.025))           # world 1: 5 mm above it
    model = s.finalize(device="cpu")
    rep = _poll(model)
    pen = [bool(int(f) & int(HealthFlags.PEN_FAIL)) for f in rep.flags]
    assert pen == [True, False] and rep.invalid_worlds == (0,)


def test_one_soft_body_per_world_adds_no_check() -> None:
    b = newton.ModelBuilder()
    _block(b, 0.0)
    s = newton.ModelBuilder()
    s.replicate(b, 2)
    assert SoftPen.build(s.finalize(device="cpu")) is None
    assert SoftPen.build(_two_blocks(0.025).finalize(device="cpu")) is not None


def _pads(health: bool):
    """SOFT-3's repro (scr/SOFT.md): the 4 mm skin pad 1 mm above the 10 mm fat pad, 0.15 s on CPU. Without health
    the trajectory is not frozen at the first INVALID check (the reference geometry)."""
    import alloy

    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_object("fat", "tissue_pad_fat", pos=(0, 0, 0.0055))
    s.add_object("skin", "tissue_pad_skin", pos=(0, 0, 0.0135))
    sim = alloy.build(s, device="cpu")
    if not health:
        sim.engine.health = None
    tr = sim.run(0.15, record=["fat", "skin"], every=1).world(0)
    return sim, np.asarray(tr["fat.points"]), np.asarray(tr["skin.points"])


@pytest.mark.timeout(900)
def test_soft_3_s_fat_and_skin_pads_check_agrees_with_the_geometry() -> None:
    """Without soft-soft contact the pads pass through each other (the fat is pushed up off the ground into the
    skin within 5 ms) and check() must not say ok; with contact (WP2's part of the ruling) they stay stacked and
    check() must say ok."""
    _, fat, skin = _pads(health=False)
    # the skin's lowest particle below the fat's highest at any frame, by more than a particle (1 mm)
    through = bool((skin[..., 2].min(axis=1) < fat[..., 2].max(axis=1) - 1.0e-3).any())
    sim, _, _ = _pads(health=True)
    report = sim.check()
    assert report["ok"] == (not through), (through, report["problems"][:1])
    if through:
        assert int(sim.engine.health.poll().flags[0]) & int(HealthFlags.PEN_FAIL)
