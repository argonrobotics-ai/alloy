# SPDX-License-Identifier: Apache-2.0
"""SolverDFSPH.emit (AlloySolverPlugin, SPEC §3.8 item 2): liquid cards -> lattice particles with sph:* attributes."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.cards.types import ParticleSource, Primitive
from alloy.core.errors import CapacityError
from alloy.route.types import Route, RouteDecision
from alloy.solvers.sph.config import PHASE_LIQUID, particle_mass, register_custom_attributes, DFSPHConfig
from alloy.solvers.sph.solver import SolverDFSPH
from alloy.testing.route import material, obj

R = 2.0e-3


def _decision(r: float = R) -> RouteDecision:
    return RouteDecision(route=Route.LIQUID_SPH, entry="sph", numbers={"particle_radius": r}, margin=1.0,
                         near_boundary=False, reasons=("test",))


def _emit(shape, mu: float = 1.0e-3, rho: float = 1000.0, budget: int | None = None) -> newton.ModelBuilder:
    o = obj("water", shape, material("water", rho, fluid_mu=mu))
    if budget is not None:
        o = dataclasses.replace(o, card=dataclasses.replace(o.card, budget=dataclasses.replace(o.card.budget,
                                                                                               particles=budget)))
    b = newton.ModelBuilder()
    register_custom_attributes(b)
    SolverDFSPH.emit(b, o, _decision())
    return b


def test_emit_box_fills_lattice_with_attributes():
    b = _emit(Primitive("box", (0.02, 0.01, 0.03)), mu=2.0e-3, rho=1200.0)
    n = b.particle_count
    assert n == 10 * 5 * 15  # 2r = 4 mm lattice in a 40 x 20 x 60 mm box
    q = np.asarray(b.particle_q)
    assert np.allclose(q.mean(axis=0), 0.0, atol=1e-9)  # centred in the object frame
    assert np.all(np.abs(q) <= np.array([0.02, 0.01, 0.03]))
    cfg = DFSPHConfig(particle_radius=R)
    assert np.allclose(b.particle_mass, particle_mass(cfg, 1200.0))
    assert set(b.custom_attributes["sph:phase"].values.values()) == {PHASE_LIQUID}
    assert np.allclose(list(b.custom_attributes["sph:viscosity"].values.values()), 2.0e-3)


def test_emit_cylinder_and_particle_source():
    cyl = _emit(Primitive("cylinder", (0.02, 0.02))).particle_count
    box = _emit(Primitive("box", (0.02, 0.02, 0.02))).particle_count
    assert 0.70 * box < cyl < 0.85 * box  # pi/4 of the bounding box
    src = _emit(ParticleSource(((0.0, 0.0, 0.0), (0.04, 0.04, 0.02)), spacing=0.004)).particle_count
    assert src == 10 * 10 * 5


def test_emit_respects_budget():
    with pytest.raises(CapacityError):
        _emit(Primitive("box", (0.02, 0.02, 0.02)), budget=100)


@pytest.mark.parametrize("width, spacing, n", [(0.3, 0.1, 3), (0.29, 0.1, 2), (0.012, 0.004, 3)])
def test_a_nozzle_a_whole_number_of_spacings_wide_keeps_every_column(width, spacing, n):
    """WP12's and WP2's find (layer_offsets): 0.3/0.1 = 2.9999999999999996 floored to 2 columns, so a nozzle a whole
    number of spacings wide emitted 1/n too little; with 1e-6 of a cell's slack it keeps n per side (a narrower one
    still rounds down)."""
    from alloy.solvers.sph.emitters import layer_offsets  # noqa: PLC0415

    half = np.array([0.5 * width, 0.5 * width, 0.05])
    pts = layer_offsets(half, np.array([0.0, 0.0, -1.0]), spacing)
    assert len(pts) == n * n
    assert np.unique(np.round(pts[:, 0], 9)).size == n and np.all(pts[:, 2] == 0.0)
