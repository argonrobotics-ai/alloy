# SPDX-License-Identifier: Apache-2.0
"""``alloy.Scene(liquid_tension=True)``: the liquid card's surface tension reaches its particles and the liquid solver
runs the csf model; off (the default) nothing changes (CPU, build and one step)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.api import _liquid

WATER_SIGMA = 0.07197          # library/materials/water.yaml fluid.surface_tension


def _scene(tension: bool | None) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01) if tension is None else alloy.Scene(dt=0.01, liquid_tension=tension)
    sc.add_ground()
    sc.add_open_box("tub", inner=(0.06, 0.06, 0.04), wall=0.004)
    sc.add_liquid("water", size=(0.06, 0.06, 0.016), pos=(0.0, 0.0, 0.012))
    sc.add_pour("pour", 4.0e-6, pos=(0.0, 0.0, 0.08))
    return sc


def _sph(sim):
    return next(s for s in _liquid.liquid_solvers(sim.engine))


def test_the_setting_is_checked_and_saved_only_when_on():
    assert alloy.Scene().liquid_tension is False and "liquid_tension" not in alloy.Scene().to_dict()
    data = alloy.Scene(liquid_tension=True).to_dict()
    assert data["liquid_tension"] is True and alloy.Scene.from_dict(data).liquid_tension is True
    with pytest.raises(alloy.SceneError) as err:
        alloy.Scene(liquid_tension="on")
    assert err.value.field == "liquid_tension"


@pytest.mark.compile_heavy
@pytest.mark.parametrize("tension", [None, False, True])
def test_tension_reaches_the_particles_only_when_on(tension):
    sim = alloy.build(_scene(tension), device="cpu")
    try:
        sph = _sph(sim)
        sigma = sim.engine.model.sph.surface_tension.numpy()
        phase = sim.engine.model.sph.phase.numpy()
        liquid = (phase == 1) | (phase == 0)                        # the block and the pour's spares
        assert liquid.sum() > 0
        if tension:
            assert sph.config.surface_tension == "csf"
            assert np.allclose(sigma[liquid], WATER_SIGMA, rtol=0.0, atol=1e-7)
            assert any("surface tension 0.07197 N/m" in r for r in sim.routes()["water"]["reasons"])
        else:
            assert sph.config.surface_tension == "none" and not sigma.any()
        sim.step(1)
        assert sim.check()["ok"]
    finally:
        sim.close()


@pytest.mark.compile_heavy
def test_still_water_with_tension_stays_still():
    """Water at rest in a tub with tension on settles: its kinetic energy falls over 2 s instead of growing (a free
    csf drop of low-viscosity liquid pumps its l = 2 mode; water at rest does not). Measured on this tub (CPU, 864
    particles): 3.4e-6 J/kg at 0.5 s, 4.3e-7 at 2 s, fastest particle 3 mm/s; tension off 3.7e-6 and 15 mm/s."""
    sc = alloy.Scene(dt=0.01, liquid_tension=True)
    sc.add_ground()
    sc.add_open_box("tub", inner=(0.048, 0.048, 0.04), wall=0.004)
    sc.add_liquid("water", size=(0.048, 0.048, 0.024), pos=(0.0, 0.0, 0.016))
    sim = alloy.build(sc, device="cpu")
    try:
        ke = []
        for _ in range(4):
            p = sim.run(0.5, record=["water"])["water.points"][:, 0]
            v = (p[-1] - p[-2]) / sim.dt
            ke.append((0.5 * float(np.mean(np.sum(v * v, axis=1))), float(np.linalg.norm(v, axis=1).max())))
        assert ke[-1][0] < 0.5 * ke[0][0] and ke[-1][0] < 2.0e-6, ke
        assert ke[-1][1] < 0.01, ke
        assert sim.check()["ok"]
    finally:
        sim.close()
