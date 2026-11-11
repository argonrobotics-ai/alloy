# SPDX-License-Identifier: Apache-2.0
"""CLOTH_VBD hinge map and the opt-in accurate cloth profile (alloy.solvers.vbd.cloth_profile): every CLOTH_VBD sheet
gets edge_ke = c·D·L/(A1+A2) per hinge (0 on boundary edges, edge_kd = β·edge_ke; CARD-8), the readback expects it
(also per world under randomized Young's modulus); under `cloth_profile="accurate"` the scene's VBD entry runs
ACCURATE_ITERATIONS and the route report carries the effect and the cost caveat. CARD-8 itself is
tests/scene/test_sheet_route.py; the CARD-1 coupon rows are in docs/notes/wp11-results.md."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.cards.types import Randomization
from alloy.core.errors import AssemblyError, ReadbackError
from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene import cloth_accuracy, derive
from alloy.scene.assembler import SceneAssembler
from alloy.scene.emitters.deformable import sheet_thickness
from alloy.solvers.vbd import cloth_profile as CP
from alloy.testing import scene as S


def _build(profile: str, objs=None, worlds: int = 2, randomize=()):
    objs = objs if objs is not None else [S.cloth()]
    card, lib = S.scene(objs, [S.place(o.card.name, z=0.3, randomize=randomize) for o in objs], worlds=worlds)
    with isolated_registry(empty=True):
        return SceneAssembler(lib, Router(), cloth_profile=profile).assemble(card, "cpu")


def _vbd_kwargs(built) -> dict:
    (e,) = [e for e in built.entries if e.capability == "vbd"]
    return dict(e.solver_kwargs)


def test_unknown_cloth_profile_raises():
    with pytest.raises(AssemblyError, match="cloth profile"):
        SceneAssembler(S.library([S.cloth()]), Router(), cloth_profile="precise")


def test_plate_hinge_ke_on_a_square_cell():
    s, D, c = 0.01, 2.0e-3, CP.HINGE_C
    ke = derive.plate_hinge_ke(D, np.array([s, s * math.sqrt(2.0), s]), np.array([s * s, s * s, 0.0]), c)
    assert np.allclose(ke, [c * D / s, c * D * math.sqrt(2.0) / s, 0.0])


def _card_D() -> float:
    towel = S.cloth()
    el = towel.material.elastic
    return derive.cloth_stiffness(float(el.youngs.value), float(el.poisson.value), sheet_thickness(towel))[2]


def _assert_plate_map(m) -> None:
    edges, q = m.edge_indices.numpy(), m.particle_q.numpy().astype(np.float64)
    eb = m.edge_bending_properties.numpy()
    inner = (edges[:, 0] >= 0) & (edges[:, 1] >= 0)
    length, area = CP.hinge_geometry(q, edges)
    expected = CP.HINGE_C * _card_D() * length / np.where(inner, area, 1.0)
    assert inner.any() and (~inner).any()
    assert np.allclose(eb[inner, 0], expected[inner], rtol=1e-4) and not eb[~inner, 0].any()
    beta = float(S.cloth().material.elastic.rayleigh_beta.value)
    assert np.allclose(eb[:, 1], beta * eb[:, 0], rtol=1e-6)


def test_default_route_writes_the_plate_map():
    built = _build("default")
    d = built.handles["towel"].decision
    assert d.route == Route.CLOTH_VBD and CP.ACCURATE_EFFECT not in d.scene_effects
    default = dict(getattr(cloth_accuracy, "KWARGS", {}).get("default", {}))   # cloth_accuracy="default" (WP2)
    assert _vbd_kwargs(built)["iterations"] == default.get("iterations", 10)
    assert CP.ACCURATE_EFFECT not in built.report.scene_effects
    _assert_plate_map(built.model)


def test_accurate_profile_iterations_and_report():
    built, default = _build("accurate"), _build("default")
    assert CP.is_accurate(built.handles["towel"].decision)
    assert _vbd_kwargs(built)["iterations"] == CP.ACCURATE_ITERATIONS
    assert CP.ACCURATE_EFFECT in built.report.scene_effects
    assert any(w.startswith("accurate cloth profile") for w in built.report.warnings)
    np.testing.assert_array_equal(built.model.edge_bending_properties.numpy(),
                                  default.model.edge_bending_properties.numpy())      # the map is the route's


def test_accurate_profile_readback_per_world_youngs():
    """Randomize scales every row by E_w/E_0; the readback's per-world expectation c·D(E_w)·L/(A1+A2) holds."""
    E0 = float(S.cloth().material.elastic.youngs.value)
    built = _build("accurate", worlds=3, randomize=(Randomization("material.elastic.youngs", 0.5 * E0, 2.0 * E0),))
    eb = built.model.edge_bending_properties.numpy()
    per_world = eb[:, 0].reshape(3, -1).max(axis=1)
    assert np.unique(np.round(per_world, 12)).size == 3         # assembled (readback ran) with three different E


def test_accurate_profile_without_cloth_changes_nothing():
    a, b = _build("default", [S.box()]), _build("accurate", [S.box()])
    assert a.model_digest == b.model_digest
    assert CP.ACCURATE_EFFECT not in b.report.scene_effects


def test_readback_catches_d_in_every_hinge(monkeypatch):
    """The readback is live: a CLOTH_VBD sheet emitted with D in every hinge (the old map) does not assemble."""
    monkeypatch.setattr(CP, "rewrite_hinges", lambda b, e0, p0, D, beta: np.full(len(b.edge_indices) - e0, D))
    with pytest.raises(ReadbackError, match="poisson"):
        _build("default")
