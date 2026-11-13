# SPDX-License-Identifier: Apache-2.0
"""Audit of 0.1.2 (Gemini), WP4's rows: scenes whose solvers Alloy cannot step together are refused by
``alloy.estimate`` as by ``alloy.build``, naming the objects with a fix that exists (S1-001, S2-009, S2-011); a
cable beside a precision connector builds under contacts='auto' (S2-008). CPU, no stepping."""

from __future__ import annotations

import pytest

import alloy

pytestmark = pytest.mark.timeout(600)


def _cloth_sand(liquid: bool = False, rigid_box: bool = False) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_cloth("towel", size=(0.16, 0.16), pos=(0.0, 0.0, 0.10), resolution=8, mass=0.05)
    sc.add_box("sand", size=(0.06, 0.06, 0.04), pos=(0.30, 0.0, 0.03), material="sand_silica_dry")
    if liquid:
        sc.add_liquid("water", size=(0.06, 0.06, 0.04), pos=(-0.30, 0.0, 0.03))
    if rigid_box:
        sc.add_box("stand", size=(0.05, 0.05, 0.08), pos=(-0.30, 0.25, 0.04), mass=0.4)
    return sc


def _refusals(scene: alloy.Scene) -> list[alloy.BuildError]:
    out = []
    for fn in (lambda: alloy.estimate(scene, device="cpu"), lambda: alloy.build(scene, device="cpu")):
        with pytest.raises(alloy.BuildError) as err:
            fn()
        out.append(err.value)
    return out


def test_cloth_beside_sand_is_refused_by_estimate_as_by_build():
    """S1-001 case 1: estimate passed and build refused; both refuse now, with the objects and the same fix."""
    est, bld = _refusals(_cloth_sand())
    for e in (est, bld):
        assert "towel" in e.message and "sand" in e.message and "UnsupportedPairError" in e.message, e.message
        assert "rigid=True" in e.fix
    assert est.fix == bld.fix


@pytest.mark.parametrize("rigid_box", [False, True])
def test_a_liquid_beside_cloth_and_sand_is_refused_for_the_pair_with_its_fix(rigid_box):
    """S1-001 case 2 and S2-009 (with a free rigid body too): the cloth and the sand cannot step together whatever
    else is there, so that refusal comes first, named, with its own fix, never "check the object named"."""
    for e in _refusals(_cloth_sand(liquid=True, rigid_box=rigid_box)):
        assert "towel" in e.message and "sand" in e.message, e.message
        assert "check the object named" not in e.fix and "rigid=True" in e.fix, e.fix


def test_a_liquid_beside_two_other_deformable_solvers_names_every_solvers_objects():
    """S1-001 case 3: a foil sheet (film) beside sand steps, but not with a liquid as well; the refusal names the
    liquid, the sheet and the sand, and says what to keep (it was "SPH beside ['film', 'mpm']" with the fix "check
    the object named in the message")."""
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_cloth("wrap", size=(0.16, 0.16), pos=(0.0, 0.0, 0.10), material="aluminium_6061_t6", thickness=1e-4)
    sc.add_box("sand", size=(0.06, 0.06, 0.04), pos=(0.30, 0.0, 0.03), material="sand_silica_dry")
    sc.add_liquid("water", size=(0.06, 0.06, 0.04), pos=(-0.30, 0.0, 0.03))
    for e in _refusals(sc):
        assert all(n in e.message for n in ("water", "wrap", "sand")) and "SPH beside" not in e.message, e.message
        assert "keep only one of" in e.fix and "wrap" in e.fix and "sand" in e.fix, e.fix


def test_film_beside_cloth_is_refused_by_estimate_and_its_fix_names_a_real_material():
    """S2-011: a kraft sheet beside a cotton towel; estimate refuses as build does, and the fix's material exists."""
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_object("sheet", "paper_kraft_sheet", pos=(0.0, 0.0, 0.10))
    sc.add_cloth("wrap", size=(0.2, 0.2), pos=(0.4, 0.0, 0.10), material="cotton_woven", resolution=7)
    for e in _refusals(sc):
        assert "sheet" in e.message and "wrap" in e.message
        assert "paper_a4" not in e.fix and "cotton_woven" in e.fix
    assert "cotton_woven" in {m["name"] for m in alloy.materials()}


def test_a_cable_beside_a_precision_connector_builds_under_auto_contacts():
    """S2-008: a clamped cable (rod_vbd) can touch a USB port of a precision pair; under contacts='auto' the build
    takes Newton's contact pipeline and says so instead of failing; contacts='mujoco' still refuses."""
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_object("usb_port", "usb_a_receptacle", pos=(0.0, 0.0, 0.02), static=True)
    sc.add_object("usb_plug", "usb_a_plug", pos=(0.08, 0.0, 0.02))
    sc.add_rod("patch_cable", points=[(0.12, 0.0, 0.05), (0.22, 0.0, 0.06), (0.32, 0.0, 0.05)], radius=0.005,
               clamp="start")
    sim = alloy.build(sc, device="cpu", contacts="auto")
    try:
        assert any("contacts auto: Newton's contact pipeline" in n for n in sim.engine.parts.notes)
    finally:
        sim.close()
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(sc, device="cpu", contacts="mujoco")
    assert "precision" in err.value.message
