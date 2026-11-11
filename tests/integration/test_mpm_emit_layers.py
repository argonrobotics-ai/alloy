# SPDX-License-Identifier: Apache-2.0
"""The MPM emitter fills a particle region a whole number of spacings tall with every layer
(alloy/scene/emitters/deformable.py emit_mpm).

(0.044 - 0.004) / 0.0025 is 15.999999999999998 in floating point, so floor() dropped soil_bed_damp's top layer: 37.5
of its 40 mm at a 5 mm voxel. The emitter now allows a 1e-6 slack. Of the library cards only soil_bed_damp changes
(15 -> 16 layers; checked over every particle-region and primitive card at 5, 2.5 and 1.25 mm voxels).
"""

from __future__ import annotations

import math

import pytest

from alloy.cards.types import ParticleSource
from alloy.core.registry import isolated_registry
from alloy.testing import route as T
from alloy.testing import scene as S
from alloy.testing.integration import assemble

pytestmark = [pytest.mark.integration]


def test_a_region_a_whole_number_of_spacings_tall_gets_every_layer() -> None:
    src = ParticleSource(((-0.01, -0.01, 0.004), (0.01, 0.01, 0.044)), 2.5e-3, grain_diameter=3e-4)
    floor = T.box("floor", T.material("wood", **T.WOOD), 0.1, 0.1, 0.01, motion="static")
    mat = T.material("sand", density=2650.0, E=1e7, nu=0.3, friction_angle=math.radians(33.0))
    card, lib = S.scene([floor, T.obj("fill", src, mat)], [S.place("floor", z=-0.01), S.place("fill", z=0.0005)],
                        worlds=1)
    with isolated_registry(empty=True):
        built = assemble(card, lib, "cpu", 1)
    voxel = float(built.report.decisions["fill"].numbers["voxel"])
    layers = len({round(float(v), 6) for v in built.model.particle_q.numpy()[:, 2]})
    assert layers == round(0.04 / (0.5 * voxel))                # 16 at a 5 mm voxel, not 15
