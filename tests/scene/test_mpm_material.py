# SPDX-License-Identifier: Apache-2.0
"""MPM material mapping (alloy.scene.emitters.deformable): granular bulk density and Mohr-Coulomb cohesion.

- WP1's card ruling (2026-11-06): a granular card's density is its grain (solid) density, so grain_mpm fills at
  bulk = GRAIN_PACKING x density (``mpm_density``); a paste keeps its card density. The readback compares the same.
- A granular card's ``plastic.cohesion`` is Newton's per-particle ``yield_stress`` (WP12 soil, docs/notes/
  soil-design.md: a free column's boundary at φ 0 is 1.03 ± 0.03 of 2c / (ρ_bulk g)); a card with both raises.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from alloy.cards.types import ParticleSource
from alloy.core.errors import AssemblyError
from alloy.core.registry import isolated_registry
from alloy.route.checks import GRAIN_PACKING
from alloy.scene.emitters.deformable import mpm_values
from alloy.testing import route as T
from alloy.testing import scene as S
from alloy.testing.integration import assemble

SPACING = 2.5e-3


def _built(mat):
    src = ParticleSource(((-0.02, -0.02, 0.0), (0.02, 0.02, 0.02)), SPACING, grain_diameter=3e-4)
    floor = T.box("floor", T.material("wood", **T.WOOD), 0.1, 0.1, 0.01, motion="static")
    card, lib = S.scene([floor, T.obj("fill", src, mat)], [S.place("floor", z=-0.01), S.place("fill", z=0.0005)],
                        worlds=1)
    with isolated_registry(empty=True):
        return assemble(card, lib, "cpu", 1)                # the readback runs here and must pass


@pytest.mark.parametrize("kind", ["grain", "paste"])
def test_mpm_mass_is_bulk_for_grains_and_card_density_for_pastes(kind: str) -> None:
    if kind == "grain":
        mat = T.material("sand", density=2650.0, E=1e7, nu=0.3, friction_angle=math.radians(33.0))
    else:
        mat = T.material("paste", density=1100.0, E=1e4, yield_stress=200.0)
    built = _built(mat)
    assert str(built.report.decisions["fill"].route) == f"{kind}_mpm"
    mass = built.model.particle_mass.numpy()
    rho = mat.density.value * (GRAIN_PACKING if kind == "grain" else 1.0)
    voxel = float(built.report.decisions["fill"].numbers["voxel"])
    assert mass.sum() == pytest.approx(rho * mass.size * (0.5 * voxel) ** 3, rel=1e-5)


def test_cohesion_is_yield_stress_and_excludes_it() -> None:
    mat = T.material("soil", density=2650.0, E=1e7, nu=0.3, friction_angle=math.radians(30.0))
    soil = dataclasses.replace(mat, plastic=dataclasses.replace(mat.plastic, cohesion=T.q(400.0)))
    assert mpm_values(soil)["yield_stress"] == pytest.approx(400.0)
    assert "yield_stress" not in mpm_values(mat)
    both = dataclasses.replace(soil, plastic=dataclasses.replace(soil.plastic, yield_stress=T.q(100.0)))
    with pytest.raises(AssemblyError, match="cohesion and plastic.yield_stress"):
        mpm_values(both)
