# SPDX-License-Identifier: Apache-2.0
"""Metal rods (copper and aluminium cards carry J2 plasticity since 2026-11-10), CPU.

A metal wire with ``route="chain"`` (also ``clamp="start", route="chain"``, the fast path beside a robot) failed its
health check in the first steps (ENERGY_JUMP, SPEED_FAIL): the router sent a yielding material to rod_vbd before the
chain's own link count and self-test ran, and the override then built a chain from rod_vbd's numbers. A chain asked
for now gets the chain's links (30 for a 0.16 m copper wire of 1.5 mm radius) and its check is ok; it stays
elastic.

A rod stiff enough at its size to count as rigid (a thick metal rod):

``add_rod(length=0.16, radius=0.006, material="copper_c110")`` failed to build with "AssemblyError: rod: shape source
CurveSource cannot be emitted as a rigid body": the router's stiffness check sent the rod to the rigid route, which
has no shape for a curve. ``route="rod_vbd"`` / ``"chain"`` then raised KeyError 'rod_length' (the rod numbers are
filled on the rod routes only). Now a stiff rod keeps the rod routes (and its ``"<name>.points"`` keys), and a rod
material with no elastic data is refused in plain words.
"""

from __future__ import annotations

import pytest

import alloy

RADIUS, LENGTH = 0.006, 0.16


def _rod_scene(material_without_elastic: str | None = None, **kw) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    if material_without_elastic:
        sc.add_material(material_without_elastic, density=1200.0, friction=0.5)
        kw["material"] = material_without_elastic
    sc.add_rod("bar", length=LENGTH, radius=RADIUS, pos=(0, -LENGTH / 2, 0.05), direction=(0, 1, 0), **kw)
    return sc


@pytest.mark.parametrize("route", [None, "rod_vbd"])
def test_thick_copper_rod_builds_and_lands(route: str | None) -> None:
    sim = alloy.build(_rod_scene(material="copper_c110", route=route), device="cpu")
    decision = sim.routes()["bar"]
    assert decision["route"] == "rod_vbd", decision
    assert any("a curve has no rigid shape" in r for r in decision["reasons"]), decision["reasons"]
    sim.step(50)
    z = sim.state(world=0)["bar.points"][:, 2]
    assert z.min() == pytest.approx(RADIUS, abs=0.5e-3), z          # resting on the ground, centre line one radius up
    assert z.max() == pytest.approx(RADIUS, abs=0.5e-3), z          # straight
    check = sim.check()
    assert check["ok"] and not check["problems"], check["problems"]


def test_rod_material_without_elastic_data_is_refused_in_plain_words() -> None:
    sc = _rod_scene(material_without_elastic="stiff_plastic")
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(sc, device="cpu")
    text = err.value.message + " " + (err.value.fix or "")
    assert "a rod needs a material that bends" in text and "add_capsule" in text, text
    assert "CurveSource" not in text and "AssemblyError" not in text, text


@pytest.mark.parametrize("material, clamp", [("copper_c110", None), ("copper_c110", "start"),
                                             ("aluminium_6061_t6", None)])
def test_metal_wire_on_the_chain_route_steps_clean(material: str, clamp: str | None) -> None:
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_rod("wire", length=LENGTH, radius=0.0015, pos=(0, -LENGTH / 2, 0.05), direction=(0, 1, 0),
               material=material, route="chain", clamp=clamp)
    sim = alloy.build(sc, device="cpu")
    decision = sim.routes()["wire"]
    assert decision["route"] == "chain", decision
    assert any("chain asked: the material yields" in r for r in decision["reasons"]), decision["reasons"]
    sim.step(50)
    z = sim.state(world=0)["wire.points"][:, 2]
    if clamp is None:
        assert z.max() == pytest.approx(0.0015, abs=0.5e-3), z      # on the ground
    else:
        assert z.max() == pytest.approx(0.05, abs=0.5e-3), z         # the welded first link holds the start
    check = sim.check()     # a free wire's landing dips 1.2 mm into the ground (PEN_WARN), as a cable on the chain does
    assert check["ok"] and all("PEN_WARN" in p["what"] for p in check["problems"]), check["problems"]
