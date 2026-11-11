# SPDX-License-Identifier: Apache-2.0
"""A cloth sheet bends like its fabric at any contact thickness (R2-E1): the cloth route takes the hinge rigidity D from
the material's ``elastic.bending`` (KES B where measured, else derived from E at the card's fitted thickness) when the
card gives it, not from E·h³ at the sheet's thickness.

Each fabric card fits E at its own model thickness (cotton_woven 0.6 mm), so the plate value at another thickness
scales as h³: B3's 4 mm cotton_woven sheet (a fresh agent's fold, [internal note]) bent ~300x its fabric, like
card, and its fold sprang open on release. The cards' ``bending`` is exactly the plate value at each library object's
own thickness, so every library cloth object is unchanged bit for bit.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import yaml

import alloy
from alloy.cards import validate as V
from alloy.cards.library import CardLibrary, default_root
from alloy.scene import derive
from alloy.solvers.vbd import cloth_profile



def _plate(mat, h: float) -> float:
    return derive.cloth_stiffness(float(mat.elastic.youngs.value), float(mat.elastic.poisson.value), h)[2]


def test_library_cloth_objects_keep_their_bending_bit_for_bit():
    """Every library sheet object whose material carries ``bending`` sits at the thickness its value was derived at (a
    new one at another thickness fails here; model and state digests of these objects and demo_cloth_drop matched main
    bit for bit, [internal note] notes/wp11-fold/digest_compare.txt)."""
    lib = CardLibrary()
    elastic = {n: lib.material(n).elastic for n in lib.names("materials")}
    bent = {n for n, e in elastic.items() if e is not None and e.bending is not None}
    seen = set()
    for path in sorted((default_root() / "objects").glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("material") not in bent:
            continue
        obj, mat = lib.object(raw["name"]), lib.material(raw["material"])
        if obj.features.thickness is None:
            continue
        h = float(obj.features.thickness.value)
        E, nu = float(mat.elastic.youngs.value), float(mat.elastic.poisson.value)
        assert derive.sheet_bending(mat, E, nu, h) == _plate(mat, h), (mat.name, obj.name)
        seen.add(obj.name)
    assert {"cloth_cotton_sheet", "cloth_polyester_sheet", "towel_cotton", "towel_cotton_hand"} <= seen
    woven = lib.material("cotton_woven")                                   # add_cloth's default: fitted at 0.6 mm
    E, nu = float(woven.elastic.youngs.value), float(woven.elastic.poisson.value)
    assert derive.sheet_bending(woven, E, nu, 0.6e-3) == _plate(woven, 0.6e-3)
    assert derive.sheet_bending(woven, 0.5 * E, nu, 4.0e-3) == 0.5 * float(woven.elastic.bending.value)


def test_a_card_without_bending_keeps_the_plate_value_and_a_bad_one_is_refused():
    mat = CardLibrary().material("cotton_woven")
    bare = dataclasses.replace(mat, elastic=dataclasses.replace(mat.elastic, bending=None))
    E, nu = float(mat.elastic.youngs.value), float(mat.elastic.poisson.value)
    assert derive.sheet_bending(bare, E, nu, 4.0e-3) == derive.plate_rigidity(E, nu, 4.0e-3)
    bad = dataclasses.replace(mat, elastic=dataclasses.replace(mat.elastic, bending=dataclasses.replace(
        mat.elastic.bending, value=-1.0e-5)))
    iss = V.Issues(bad.name)
    V.material_issues(bad, iss)
    assert any(i.field == "elastic.bending" and i.level == "error" for i in iss.items)


def _hinges(thickness: float) -> tuple[np.ndarray, float]:
    """add_cloth's hinge stiffness (cotton_woven) at ``thickness`` and the particle radius."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_cloth("c", size=(0.2, 0.16), pos=(0.0, 0.0, 0.05), resolution=(11, 9), mass=0.03, thickness=thickness)
    sim = alloy.build(scene, device="cpu")
    m = sim.engine.model
    ke = m.edge_bending_properties.numpy()[:, 0].copy()
    radius = float(m.particle_radius.numpy().max())
    expected = cloth_profile.hinge_ke(m.particle_q.numpy(), m.edge_indices.numpy(),
                                      float(CardLibrary().material("cotton_woven").elastic.bending.value))
    np.testing.assert_allclose(ke, expected, rtol=1e-6, atol=0.0)          # the plate map of the fabric's D
    sim.close()
    return ke, radius


def test_add_cloth_bends_like_its_fabric_at_any_contact_thickness():
    """1, 2 and 4 mm sheets of the same grid: the same hinges (D = the card's 2e-5 N·m); only the contact radius
    (h/2) follows the thickness. E·h³ at 4 mm would have been ~300x the fabric."""
    ke, radii = {}, {}
    for h in (1.0e-3, 2.0e-3, 4.0e-3):
        ke[h], radii[h] = _hinges(h)
    np.testing.assert_array_equal(ke[1.0e-3], ke[2.0e-3])
    np.testing.assert_array_equal(ke[1.0e-3], ke[4.0e-3])
    assert radii[4.0e-3] == pytest.approx(2.0e-3, rel=1e-6) and radii[1.0e-3] == pytest.approx(0.5e-3, rel=1e-6)
    woven = CardLibrary().material("cotton_woven")
    assert _plate(woven, 4.0e-3) / float(woven.elastic.bending.value) == pytest.approx((4.0 / 0.6) ** 3, rel=1e-9)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_a_thick_contact_sheet_folded_in_half_stays_folded():
    """B3's case on laundry_crossings' fold rig (CPU): a 0.20 x 0.16 m terry sheet (437 particles) at a 4 mm contact
    thickness with the card's areal mass (x 0.15), one long edge carried round the fold line in 0.6 s and let go:
    1 s later it is still folded, with no crossings: its footprint across the fold is 0.76 of the flat sheet's (half
    of it plus the fold's loop, ~4 cm across: terry's bending length (B/w)^(1/3) is 1.6 cm). With E·h³ at 4 mm
    (~300x the fabric) the same fold opened flat: 1.00 ([internal note] notes/wp11-fold/law_plate.log)."""
    from alloy.tasks import laundry_crossings as lc

    r = lc.run(lc.FoldRig(card="towel_cotton", size=(0.20, 0.16), thickness=4.0e-3, particles=400, mass_scale=0.15,
                          fold_s=0.6, settle_s=1.0))
    assert r.finite and r.crossings_max == 0
    assert r.y_span < 0.85 * r.y_span0                                     # 0.76 measured; opened flat: 1.00
