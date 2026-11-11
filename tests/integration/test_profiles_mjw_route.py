# SPDX-License-Identifier: Apache-2.0
"""A profile may ask for Alloy's ordered MuJoCo route (coordinator ruling 2026-11-06 06:23Z, WP5).

``IntegrationProfile.mjw_route`` (default off) sets ``PrecisionConfig.mjw_route``. It also builds the MuJoCo entries
with a sparse Jacobian, the route's proven path: MuJoCo picks dense below nv 60, and the route refuses dense. The
default build is unchanged.
"""

from __future__ import annotations

import dataclasses

import pytest
import warp as wp

from alloy.cards.library import CardLibrary
from alloy.integration.profiles import profile
from alloy.integration.wiring import build_simulation
from alloy.precision import mjw_route
from alloy.scene.assembler import SceneAssembler

SCENE = "demo_peg_d12"   # a MuJoCo-only scene, nv 6: dense Jacobian by default


def _mujoco(sim):
    return next(s for s in sim.parts.solvers.values() if hasattr(s, "mjw_model"))


def _build():
    lib = CardLibrary()
    return SceneAssembler(lib).assemble(lib.scene(SCENE), "cpu", world_count=1)


@pytest.fixture(autouse=True)
def _cpu():
    with wp.ScopedDevice("cpu"):
        yield
    mjw_route.uninstall()


def test_default_profile_keeps_the_route_off_and_the_jacobian_as_mujoco_picks() -> None:
    assert profile("standard").mjw_route is False
    assert profile("standard").precision_config("cpu").mjw_route is False
    built = _build()
    sim = build_simulation(built, "standard")
    s = _mujoco(sim)
    assert not bool(s.mjw_model.is_sparse)
    assert "dense Jacobian" in (mjw_route.static_reason(s.mjw_model) or "")


def test_mjw_route_profile_asks_for_the_route_on_a_sparse_jacobian() -> None:
    p = profile("standard", mjw_route=True)
    assert p.precision_config("cpu").mjw_route is True
    built = _build()
    sim = build_simulation(built, "standard", mjw_route=True)
    s = _mujoco(sim)
    assert bool(s.mjw_model.is_sparse)
    assert mjw_route.static_reason(s.mjw_model) is None
    assert sim.precision.mjw_route is True
    assert any("jacobian=sparse" in n for n in sim.parts.notes)


def test_an_entry_that_sets_its_jacobian_keeps_it() -> None:
    built = _build()
    built = dataclasses.replace(built, entries=[
        dataclasses.replace(e, solver_kwargs={**dict(e.solver_kwargs), "jacobian": "dense"})
        if e.capability == "mujoco" else e for e in built.entries])
    sim = build_simulation(built, "standard", mjw_route=True)
    assert not bool(_mujoco(sim).mjw_model.is_sparse)
