# SPDX-License-Identifier: Apache-2.0
"""resolve: defaults filled, default leaves listed, worst-case table, card_hash scope, extents per source, errors."""

from __future__ import annotations

import dataclasses as dc
from pathlib import Path

import pytest

from alloy.cards import types as ct
from alloy.cards.identity import worst_end
from alloy.cards.library import CardLibrary
from alloy.core.errors import CardError
from alloy.core.quantity import Q

LIB = CardLibrary(assets=None)
POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def test_defaults_filled_and_listed() -> None:
    r = LIB.resolve(LIB.object("cube_beech_30"))
    assert r.material.surface.contact_stiffness == Q(2e6, "default")
    for leaf in ("loads.grip_force", "loads.bend_force", "loads.contact_speed", "loads.tolerance", "loads.design_force",
                 "material.surface.contact_stiffness"):
        assert leaf in r.default_leaves
    assert "material.density" not in r.default_leaves                    # sourced
    assert r.material.surface.torsional_lever is None                    # derived by WP2, not filled here


def test_sourced_loads_drop_out_of_default_leaves() -> None:
    r = LIB.resolve(LIB.object("pencil_wood"))
    assert "loads.bend_force" not in r.default_leaves and "loads.tolerance" not in r.default_leaves
    assert "loads.grip_force" in r.default_leaves


@pytest.mark.parametrize("path, expect", [
    ("material.elastic.youngs", "lo"), ("material.elastic.poisson", "lo"), ("part_materials.grip.elastic.youngs", "lo"),
    ("loads.tolerance", "lo"), ("features.thickness", "lo"), ("precision_features.0.clearance", "lo"),
    ("material.density", "hi"), ("loads.grip_force", "hi"), ("loads.bend_force", "hi"), ("loads.contact_speed", "hi"),
    ("material.fluid.viscosity", "hi"),
])
def test_worst_end_table(path: str, expect: str) -> None:
    assert worst_end(path, 1.0, 3.0, 2.0) == (1.0 if expect == "lo" else 3.0)


def test_worst_end_default_rule() -> None:
    assert worst_end("pose.0", -1.0, 3.0, 0.0) == 3.0           # farther from nominal
    assert worst_end("pose.0", -3.0, 1.0, 0.0) == -3.0
    assert worst_end("material.surface.mu_static", 0.4, 0.6, 0.5) == 0.6   # tie -> hi


def test_worst_case_on_placement() -> None:
    pl = ct.Placement("towel_cotton", POSE, (ct.Randomization("material.elastic.youngs", 5e5, 2e6),
                                              ct.Randomization("pose.2", 0.0, 0.1)))
    r = LIB.resolve(LIB.object("towel_cotton"), pl)
    assert r.worst_case == {"material.elastic.youngs": 5e5, "pose.2": 0.1}


def test_card_hash_scope() -> None:
    card = LIB.object("peg_d12")
    h0 = LIB.resolve(card).card_hash
    assert LIB.resolve(card).card_hash == h0
    pl = ct.Placement("peg_d12", POSE, (ct.Randomization("material.density", 7900.0, 8060.0),))
    assert LIB.resolve(card, pl).card_hash != h0                                   # ranges are routing inputs
    moved = ct.Placement("peg_d12", (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))
    assert LIB.resolve(card, moved).card_hash == h0                                # pose is not


def test_card_hash_follows_material_edits(tmp_library: Path) -> None:
    lib = CardLibrary([tmp_library])
    h0 = lib.resolve(lib.object("peg_d12")).card_hash
    m = lib.material("stainless_303")
    lib.put(dc.replace(m, density=Q(8100.0, "measured", None, "weighed")), reason="test")
    assert lib.resolve(lib.object("peg_d12")).card_hash != h0


def test_int_and_float_cards_hash_alike() -> None:
    a = ct.ObjectCard("cube_beech_30", ct.Primitive("box", (0.015, 0.015, 0.015)), "wood_beech",
                      features=ct.Features(precision_anchor=(0, 0, 0)))
    b = dc.replace(a, features=ct.Features(precision_anchor=(0.0, 0.0, 0.0)))
    assert LIB.resolve(a).card_hash == LIB.resolve(b).card_hash


@pytest.mark.parametrize("name, ext", [
    ("cube_beech_30", (0.03, 0.03, 0.03)), ("rolling_pin_beech", (0.2, 0.045, 0.045)), ("peg_d12", (0.05, 0.012, 0.012)),
    ("usb_cable_1m", (1.0, 3.5e-3, 3.5e-3)), ("water_cup", (0.06, 0.06, 0.05)), ("nut_m10", (17.0 / 3**0.5 * 2e-3, 0.017, 0.008)),
    ("tray_aluminium", (0.3, 0.22, 0.02)),
])
def test_extents(name: str, ext: tuple[float, float, float]) -> None:
    assert LIB.resolve(LIB.object(name)).extents == pytest.approx(ext, rel=1e-6)


def test_resolve_errors() -> None:
    with pytest.raises(CardError) as exc:
        LIB.resolve(dc.replace(LIB.object("peg_d12"), material="nope"))
    assert exc.value.field == "material"
    with pytest.raises(CardError) as exc:
        LIB.resolve(LIB.object("peg_d12"), ct.Placement("peg_d12", POSE, (ct.Randomization("material.density", 1.0, 2.0),)))
    assert exc.value.field == "placement.randomize.0.lo"
    with pytest.raises(CardError) as exc:
        LIB.resolve(LIB.object("peg_d12"), ct.Placement("nut_m10", POSE))
    assert exc.value.field == "object"
    usd = ct.ObjectCard("drawer", ct.USDSource("meshes/nut_m10_din934.obj", "/World"), "abs")
    with pytest.raises(CardError) as exc:
        LIB.resolve(usd)
    assert exc.value.field == "shape"


def test_mjcf_extents_through_newton() -> None:
    r = LIB.resolve(LIB.object("hinged_box_plywood"))          # base 100 x 80 x 60 mm + 6 mm lid on top
    assert r.extents == pytest.approx((0.1, 0.08, 0.066), rel=1e-5)
    lid = ct.ObjectCard("lid_only", ct.MJCFSource("mjcf/hinged_box.xml", "lid"), "plywood_marine")
    assert LIB.resolve(lid).extents == pytest.approx((0.1, 0.08, 0.006), rel=1e-5)
    with pytest.raises(CardError) as exc:
        LIB.resolve(ct.ObjectCard("x", ct.MJCFSource("mjcf/hinged_box.xml", "ghost"), "plywood_marine"))
    assert exc.value.field == "shape.body"
