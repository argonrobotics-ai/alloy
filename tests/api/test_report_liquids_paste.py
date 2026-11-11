# SPDX-License-Identifier: Apache-2.0
"""``Sim._report_liquids`` with a paste: ``add_liquid`` of a paste material (mayonnaise, peanut butter) routes to MPM,
which has no SPH particle radius. The build's liquid-volume report skips it; before, ``alloy.build`` raised
SimulationError "'p' has no particle radius (not a liquid)". An SPH liquid is still reported. No build: a stand-in
for the built sim."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import alloy.api._liquid as L
from alloy.api.errors import SimulationError
from alloy.api.sim import Sim

PARTICLES = 240


def _built(radius: float | None) -> tuple[SimpleNamespace, list[str]]:
    """A built scene of one liquid item "p" asking 30 ml; ``radius`` None is a paste (MPM)."""
    numbers = {} if radius is None else {"particle_radius": radius}
    asked = []

    def particle_volume(name: str) -> float:
        if radius is None:
            raise SimulationError(f"{name!r} has no particle radius (not a liquid)", "call it on a liquid")
        return (2.0 * radius) ** 3

    def liquid_volume(name: str) -> float:
        asked.append(name)
        return PARTICLES * particle_volume(name)

    handle = SimpleNamespace(decision=SimpleNamespace(numbers=numbers))
    sim = SimpleNamespace(engine=None, scene=SimpleNamespace(items={"p": {"kind": "liquid", "volume": 3.0e-5}}),
                          index=SimpleNamespace(particles=["p"], particle_range=lambda name: range(PARTICLES)),
                          built=SimpleNamespace(handles={"p": handle}),
                          particle_volume=particle_volume, liquid_volume=liquid_volume)
    return sim, asked


@pytest.fixture(autouse=True)
def _no_kinematic(monkeypatch):
    monkeypatch.setattr(L, "unseen_kinematic", lambda engine: 0)


def test_a_paste_is_not_reported_as_a_liquid():
    sim, asked = _built(None)
    Sim._report_liquids(sim)
    assert asked == []


def test_an_sph_liquid_is_still_reported():
    sim, asked = _built(0.0025)                      # 240 particles of (5 mm)^3 = 30 ml, as asked
    Sim._report_liquids(sim)
    assert asked == ["p"]
