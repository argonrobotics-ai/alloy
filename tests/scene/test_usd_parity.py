# SPDX-License-Identifier: Apache-2.0
"""CARD-4: card → compiled Model vs Newton's USD importer for the same material (tri_materials, edge_bending_properties,
tet_materials rel diff ≤ 1e-6). For CLOTH_VBD the importer's D is compared through the plate hinge map c·D·L/(A1+A2)
the emitter writes (CARD-8): the importer puts D itself into every hinge, ~60x soft on a 12.5 mm cell. The membrane
is compared on E·h and ν, not on the importer's tri_ke: the importer writes tri_ke = E·h/(1−ν²), tri_ka = 0, which in
Newton's own neo-Hookean VBD membrane (tri_ke = mu, tri_ka = lambda) is a 2.2x stiffer sheet with Poisson 0. The
emitter writes the 2D Lame pair that gives back E·h and ν: mu = stretch·(1−ν)/2, lambda = stretch·ν with the
importer's stretch = E·h/(1−ν²) (derive.cloth_stiffness; coordinator ruling 2026-11-08 08:40Z, [internal note]
notes/wp2-drape/diff).

`pxr` is not installed in this environment, so `ModelBuilder.add_usd` cannot run (the add_usd path is not_run). The
comparison uses the importer's own conversion functions that add_usd calls (newton@ac7f7830:
newton/_src/utils/import_usd_deformable_cloth.py:137-183 `_resolve_surface_structural_stiffnesses`, used at :569-573
with tri_ka forced to 0; newton/_src/usd/utils.py:805-809 `_deformable_lame_parameters`) against the arrays our
assembled Models hold. With pxr present, the add_usd branch also builds a .usda and compares arrays end to end."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.registry import isolated_registry
from alloy.route.router import Router
from alloy.scene.assembler import SceneAssembler
from alloy.solvers.vbd import cloth_profile as CP
from alloy.testing import route as T
from alloy.testing import scene as S

CASES = [(1e6, 0.3, 2.5e-4), (5e5, 0.45, 1e-3), (2e7, 0.2, 5e-4)]
RTOL = 1e-6


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):
        yield


def _importer():
    from alloy._vendor.newton._src.usd.utils import _deformable_lame_parameters
    from alloy._vendor.newton._src.utils.import_usd_deformable_cloth import _resolve_surface_structural_stiffnesses

    return _resolve_surface_structural_stiffnesses, _deformable_lame_parameters


@pytest.mark.parametrize("E,nu,h", CASES)
def test_card4_cloth_parity(E, nu, h):
    resolve, _ = _importer()
    cloth = T.box("towel", T.material("cl", density=1200.0, E=E, nu=nu, mu=0.5, e=0.1), 0.1, 0.1, 0.5 * h)
    card, lib = S.scene([S.table(), cloth], [S.place("table"), S.place("towel", z=0.03)], worlds=1)
    built = SceneAssembler(lib, Router()).assemble(card, "cpu")
    stretch, _shear, bend = resolve({"youngsModulus": E, "poissonsRatio": nu}, h, 1.0)
    m = built.model
    tm, eb = m.tri_materials.numpy(), m.edge_bending_properties.numpy()
    # the importer writes its D into every hinge; CLOTH_VBD maps the same D through the hinge geometry,
    # c·D·L/(A1+A2) (VBD's hinge energy is ½·edge_ke·L·θ², CARD-8), so the parity is on D
    plate = CP.hinge_ke(m.particle_q.numpy(), m.edge_indices.numpy(), bend)
    inner = plate > 0.0
    mu, lam = stretch * (1.0 - nu) / 2.0, stretch * nu            # the Lame pair of the importer's E·h, ν
    err = max(np.max(np.abs(tm[:, 0] / mu - 1)), np.max(np.abs(tm[:, 1] / lam - 1)),
              np.max(np.abs(eb[inner, 0] / plate[inner] - 1)), float(np.abs(eb[~inner, 0]).max(initial=0.0)))
    S.record("CARD-4.cloth", float(err), "pass" if err <= RTOL else "fail", E=E, nu=nu, h=h,
             path="importer conversion functions (add_usd not_run: pxr missing)")
    assert err <= RTOL


@pytest.mark.parametrize("E,nu", [(1e5, 0.3), (2e6, 0.45), (5e4, 0.1)])
def test_card4_tet_parity(E, nu):
    _, lame = _importer()
    from alloy.cards.types import Budget

    foam = T.box("sponge", T.material("fm", density=30.0, E=E, nu=nu, mu=0.8, e=0.1), 0.04, 0.03, 0.02,
                 budget=Budget(tets=200))
    card, lib = S.scene([S.table(), foam], [S.place("table"), S.place("sponge", z=0.045)], worlds=1)
    built = SceneAssembler(lib, Router()).assemble(card, "cpu")
    k_mu, k_lambda = lame(E, nu)
    tm = built.model.tet_materials.numpy()
    err = max(np.max(np.abs(tm[:, 0] / k_mu - 1)), np.max(np.abs(tm[:, 1] / k_lambda - 1)))
    S.record("CARD-4.tet", float(err), "pass" if err <= RTOL else "fail", E=E, nu=nu,
             path="importer conversion functions (add_usd not_run: pxr missing)")
    assert err <= RTOL


def test_card4_add_usd_path():
    pytest.importorskip("pxr", reason="CARD-4 add_usd end-to-end path not_run: pxr (usd-core) not installed")
