# SPDX-License-Identifier: Apache-2.0
"""SCR-JOB2-2 item 3 (WP12's share of alloy/scene/entries.py): the router's SPH particle radius reaches the sph
entry, so a card-path liquid runs at the radius its particles were seeded at, not DFSPHConfig's 2 mm (CPU)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.cards.types import ParticleSource
from alloy.core.errors import AssemblyError
from alloy.route.router import Router
from alloy.route.types import Route, RouterConfig
from alloy.scene import entries as E
from alloy.scene.assembler import SceneAssembler
from alloy.testing import integration as I
from alloy.testing import route as T
from alloy.testing import scene as S

R = 1.0e-3                     # liquid_accuracy "super": not DFSPHConfig's 2 mm default, so a lost kwarg shows


def _cup_of_water() -> object:
    water = T.obj("water", ParticleSource(((-0.01, -0.01, 0.0), (0.01, 0.01, 0.02)), 0.002),
                  T.material("water", density=1000.0, fluid_mu=1e-3))
    cup = T.box("cup", T.material("pp", density=900.0, E=1.5e9), 0.04, 0.04, 0.05,
                features=T.features(passage_width=0.02), motion="static")
    card, lib = S.scene([cup, water], [S.place("cup", z=0.05), S.place("water", z=0.06)], worlds=1)
    router = Router(RouterConfig(liquid_accuracy="super"), rod_self_test=lambda rod, h: dict(I.ROD_SELF_TEST_PASS))
    return SceneAssembler(lib, router).assemble(card, "cpu", world_count=1)


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_card_path_sph_runs_at_the_router_radius():
    from alloy.integration import build_simulation  # noqa: PLC0415

    built = _cup_of_water()
    sph = next(e for e in built.entries if e.capability == "sph")
    assert sph.solver_kwargs["particle_radius"] == pytest.approx(R)
    np.testing.assert_allclose(built.model.particle_radius.numpy(), R)          # seeded at the router's r
    sim = build_simulation(built, "standard", graph="off")
    cfg = sim.parts.solvers["sph"].config
    assert cfg.particle_radius == pytest.approx(R) and cfg.h == pytest.approx(4.0 * R)
    sim.step_period()
    assert np.isfinite(sim.state.particle_q.numpy()).all()


def _liquid(name: str, r: float) -> SimpleNamespace:
    return SimpleNamespace(name=name, decision=SimpleNamespace(route=Route.LIQUID_SPH, numbers={"particle_radius": r}))


def test_sph_kwargs_one_radius_or_refused():
    r = 1.25e-3
    assert E.sph_kwargs([_liquid("a", r), _liquid("b", r)]) == {"particle_radius": r}
    assert E.sph_kwargs([]) == {}
    with pytest.raises(AssemblyError, match=r"different particle radii \{'a': 0.00125, 'b': 0.002\}"):
        E.sph_kwargs([_liquid("a", r), _liquid("b", 2.0e-3)])
