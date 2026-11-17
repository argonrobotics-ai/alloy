# SPDX-License-Identifier: Apache-2.0
"""Liquid at rest in narrow round cups (liquid fix 2). ``add_open_cylinder`` builds a cup from 24 wall slabs; three
things made water in a narrow one start INVALID (ENERGY_JUMP within 0.04 s):
- the boundary counted the overlap of two nearly coplanar slabs as for orthogonal walls (the seams nearly twice);
- 5-6 slabs within the kernel of a wall particle filled its 4 boundary slots and left the floor out at the corner;
- the square lattice in a small disk held more particles than the disk's area (6 across: 32 against 28.3 a layer);
  3 % over (8 across: 52 against 50.3), or an outer ring 0.2 spacings from the side, stayed healthy but threw the
  top layer 11-40 mm up, over the rim of a 16 mm cup. The rings now hold at most the area, and the column adds
  layers on top for the volume asked.
CPU, one world."""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy
from alloy.api import _liquid
from alloy.cards.types import Primitive
from alloy.route.types import LIQUID_SPACING
from alloy.solvers.sph.build import DISK_CLEAR, liquid_points

WALL = 0.004


@pytest.mark.parametrize(("diameter", "spacing", "per_layer"), [
    (0.012, 0.002, 24), (0.016, 0.002, 44), (0.050, 0.002, 465), (0.032, 0.004, 44), (0.050, 0.004, 112),
    (0.015, 0.0015, 68)])
def test_a_round_fill_holds_about_the_disk_area_and_clears_the_side(diameter, spacing, per_layer):
    r = 0.5 * diameter
    pts = liquid_points(Primitive("cylinder", (r, 0.01)), spacing)
    layers = len(np.unique(pts[:, 2]))
    n = len(pts) // layers
    area = math.pi * r * r / spacing**2
    assert n == per_layer and n <= area
    assert r - float(np.hypot(pts[:, 0], pts[:, 1]).max()) >= DISK_CLEAR * spacing - 1e-9


@pytest.mark.parametrize(("diameter", "height", "spacing"), [(0.040, 0.020, 0.004), (0.040, 0.040, 0.004),
                                                         (0.012, 0.040, 0.002), (0.016, 0.040, 0.0015),
                                                         (0.0776, 0.0423, 0.002)])
def test_a_round_fill_adds_layers_on_top_to_the_asked_volume(diameter, height, spacing):
    """The rings hold less than the disk's area (a 40 mm disk at 4 mm: 68 against 78.5 a layer), so the column
    grows upward until its count is the closest to volume / spacing^3: within half a layer, its bottom where a box
    fill's is."""
    shape = Primitive("cylinder", (0.5 * diameter, 0.5 * height))
    pts = liquid_points(shape, spacing)
    box = liquid_points(Primitive("box", (0.5 * diameter, 0.5 * diameter, 0.5 * height)), spacing)
    per_layer = int(np.sum(np.isclose(pts[:, 2], pts[:, 2].min())))
    want = math.pi * (0.5 * diameter) ** 2 * height / spacing**3
    assert abs(len(pts) - want) <= 0.5 * per_layer
    assert float(pts[:, 2].min()) == pytest.approx(float(box[:, 2].min()), abs=1e-12)


def _cup(diameter: float, accuracy: str) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01, liquid_accuracy=accuracy)
    sc.add_ground()
    sc.add_open_cylinder("cup", inner_radius=0.5 * diameter, depth=0.06, wall=WALL)
    sc.add_liquid("water", size=(diameter, diameter, 0.04), pos=(0.0, 0.0, WALL + 0.02), shape="cylinder")
    return sc


@pytest.mark.compile_heavy
def test_the_floor_stays_in_a_corner_particles_slots():
    """In a 12 mm cup at 2 mm the bottom-ring particles see 5-6 wall slabs within h; the floor (its slot normal +z in
    its own frame) must be one of their 4 slots."""
    sim = alloy.build(_cup(0.012, "super"), device="cpu")
    try:
        sim.step(1)
        sph = _liquid.liquid_solvers(sim.engine)[0]
        x = sim.state(["water"])["water.points"][0]        # the only particles: the solver's own numbering
        shape, nbl = sph._slots.shape.numpy(), sph._slots.nbl.numpy()
        n = min(len(x), shape.shape[0])
        rho, z = np.hypot(x[:n, 0], x[:n, 1]), x[:n, 2]
        corner = (z < WALL + 0.002) & (rho > 0.003)
        assert corner.sum() >= 8
        has_floor = ((shape[:n] >= 0) & (nbl[:n, :, 2] > 0.9)).any(axis=1)
        assert has_floor[corner].all(), f"{int((~has_floor[corner]).sum())} corner particles without the floor"
    finally:
        sim.close()


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("diameter", "accuracy"), [(0.012, "super"), (0.016, "super"), (0.016, "ultra")])
def test_water_rests_in_a_narrow_round_cup(diameter, accuracy):
    """The super rows went INVALID with ENERGY_JUMP in the first 0.01 s before (main 56b1223d2, CPU). With 52 particles
    a layer in the 16 mm cup at super (3 % over its area) the top layer rose 40 mm and 9 particles left over the rim;
    at ultra, with the outer ring 0.19 spacings from the side, it rose 11 mm."""
    sim = alloy.build(_cup(diameter, accuracy), device="cpu")
    try:
        top = float(sim.state(["water"])["water.points"][0][:, 2].max())
        traj = sim.run(0.3, record=["water"], every=5)
        p = traj["water.points"][-1, 0]
        assert sim.check()["ok"], sim.check()["problems"][:2]
        assert float(traj["water.points"][..., 2].max()) < top + LIQUID_SPACING[accuracy]   # none rises a spacing
        assert np.all(np.hypot(p[:, 0], p[:, 1]) < 0.5 * diameter / math.cos(math.pi / 24) + 1e-4)
    finally:
        sim.close()
