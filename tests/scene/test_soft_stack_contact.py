# SPDX-License-Identifier: Apache-2.0
"""SCR-SOFT-3-1: two or more cloth or soft-solid objects in one VBD entry collide with each other by default
(``alloy.scene.vbd_contact``); one object without ``features.self_contact`` leaves the entry unchanged, bit for bit.
Law (CPU): SOFT-3's skin pad placed on its fat pad rests on it instead of falling through to the ground (control:
with the contact off it falls through)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import alloy.scene.entries as entries

from alloy.cards.types import Budget, Failure
from alloy.route.types import Route
from alloy.scene.vbd_contact import REST_EXCLUSION, vbd_self_contact
from alloy.solvers.vbd.reach import SELF_CONTACT_SPEED, SPEED_KEY
from alloy.testing import route as T
from alloy.testing import scene as S
from alloy.topology.types import TopologyBudget

H = 1.0e-3


def _entry(obj, route):
    return type("T", (), {"obj": obj, "name": obj.card.name, "decision": type("D", (), {"route": route})()})()


def _pad(name):
    return T.box(name, T.material("foam", **T.PU_FOAM, beta=1e-3), 0.02, 0.02, 0.005)


def _sheet(name, h, tearable=False):
    mat = T.material("cotton", **T.COTTON, beta=1e-3)
    topo = None
    if tearable:
        mat = dataclasses.replace(mat, failure=Failure(fracture_strain=T.q(0.3), tear_energy=T.q(500.0)))
        topo = TopologyBudget(spare_vertices=400)
    return T.box(name, mat, 0.05, 0.05, 0.5 * h, budget=Budget(particles=121), topology=topo,
                 features=T.features(thickness=h))


def test_one_object_leaves_the_entry_unchanged():
    assert vbd_self_contact([_entry(_pad("fat"), Route.SOFT_VBD)], radii={"fat": 0.5 * H}) == {}
    assert vbd_self_contact([_entry(_sheet("towel", H), Route.CLOTH_VBD)]) == {}


def test_two_objects_collide_at_the_thinner_ones_thickness():
    objs = [_entry(_pad("fat"), Route.SOFT_VBD), _entry(_pad("skin"), Route.SOFT_VBD)]
    kw = vbd_self_contact(objs, radii={"fat": 0.6 * H, "skin": 0.5 * H})
    assert kw == {"particle_enable_self_contact": True, "particle_self_contact_margin": H,
                  SPEED_KEY: SELF_CONTACT_SPEED}                            # 2 x the smaller radius; nothing excluded
    cloth = vbd_self_contact([objs[0], _entry(_sheet("towel", 0.4 * H), Route.CLOTH_VBD)], radii={"fat": 0.6 * H})
    assert cloth["particle_self_contact_margin"] == pytest.approx(0.4 * H)   # a sheet's thickness is its h


def test_a_presplit_object_excludes_its_coincident_seams():
    objs = [_entry(_pad("fat"), Route.SOFT_VBD), _entry(_sheet("towel", H, tearable=True), Route.CLOTH_VBD)]
    kw = vbd_self_contact(objs, radii={"fat": 0.6 * H})
    assert kw["particle_rest_shape_contact_exclusion_radius"] == pytest.approx(REST_EXCLUSION * H)


def _pads(seconds: float, health: bool = True) -> tuple[dict, bool]:
    """SOFT-3's repro (scr/SOFT.md SCR-SOFT-3-1): a 4 mm skin pad placed 1 mm above a 10 mm fat pad on the ground.
    ``health=False`` gives the true geometry: WP5's PEN_FAIL freezes a world whose pads pass through each other."""
    import alloy

    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_object("fat", "tissue_pad_fat", pos=(0, 0, 0.0055))
    s.add_object("skin", "tissue_pad_skin", pos=(0, 0, 0.0135))
    sim = alloy.build(s, device="cpu")
    if not health:
        sim.engine.health = None
    tr = sim.run(seconds, record=["fat", "skin"], every=5).world(0)
    return tr, bool(sim.check()["ok"]) if health else True


def _gaps(tr) -> np.ndarray:
    """Per frame: the skin's lowest particle above the fat's highest [m]."""
    return np.asarray([s[:, 2].min() - f[:, 2].max() for s, f in zip(tr["skin.points"], tr["fat.points"])])


@pytest.mark.slow
@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_skin_pad_rests_on_the_fat_pad():
    tr, ok = _pads(0.3)
    gaps = _gaps(tr)
    S.record("SCR-SOFT-3-1.pads", float(gaps[-1]), "pass" if gaps.min() >= 0.0 else "fail",
             min_gap_m=float(gaps.min()), end_gap_m=float(gaps[-1]), check_ok=ok)
    assert gaps.min() >= 0.0, f"the skin reached {gaps.min() * 1e3:.2f} mm into the fat"
    assert ok


@pytest.mark.compile_heavy
def test_without_inter_object_contact_the_skin_falls_through(monkeypatch):
    monkeypatch.setattr(entries, "vbd_self_contact", lambda objs, radii=None: {})
    tr, _ = _pads(0.15, health=False)
    assert _gaps(tr)[-1] < -5.0e-3                                          # onto the ground, through the fat
