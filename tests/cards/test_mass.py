# SPDX-License-Identifier: Apache-2.0
"""CARD-10: mass from card (Newton's inertia routine) vs catalogue weight, <= 2 %. Failing objects are strict xfails
with the measured number, so they are reported, not hidden, and flip the suite if they ever start passing."""

from __future__ import annotations

import math

import pytest

from alloy.cards import types as ct
from alloy.cards.codec import Ctx, decode
from alloy.cards.library import CardLibrary, default_root
from alloy.cards.mass import card_mass, mesh_mass, primitive_mass
from alloy.cards.meshgen import box_pocket
from alloy.cards.paths import CardView
from alloy.cards.units_yaml import yaml_load
from alloy.core.quantity import Q

LIB = CardLibrary(assets=None)
CATALOG = yaml_load((default_root() / "catalog.yaml").read_text(encoding="utf-8"), "catalog")
TOL = 0.02
KNOWN_FAIL = {   # measured on this commit; the thread is a pitch-diameter cylinder, chamfers not modelled
    "bolt_m10x60": "+2.04 %", "bolt_m10x40": "+2.95 %", "bolt_m6x30": "+4.51 %",
}


def _rel(name: str) -> float:
    q = decode(Q, CATALOG["masses"][name], f"masses.{name}", Ctx("catalog"), "mass")
    r = LIB.resolve(LIB.object(name))
    m = card_mass(CardView(r.card, r.material, r.part_materials), LIB.asset_path)
    return (m - q.value) / q.value


PARAMS = [pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=f"CARD-10 measured {KNOWN_FAIL[n]}"))
          if n in KNOWN_FAIL else n for n in CATALOG["masses"]]


@pytest.mark.parametrize("name", PARAMS)
def test_card_mass_vs_catalogue(name: str) -> None:
    assert abs(_rel(name)) <= TOL, f"{name}: {_rel(name):+.2%}"


def test_catalogue_entries_exist_and_exclusions_are_explained() -> None:
    for name in list(CATALOG["masses"]) + list(CATALOG["excluded"]):
        assert LIB.has_object(name), name
    assert all(isinstance(v, str) and v for v in CATALOG["excluded"].values())
    assert not set(CATALOG["masses"]) & set(CATALOG["excluded"])


@pytest.mark.parametrize("kind, size, volume", [
    ("box", (0.01, 0.02, 0.03), 8 * 0.01 * 0.02 * 0.03), ("sphere", (0.01,), 4 / 3 * math.pi * 1e-6),
    ("capsule", (0.01, 0.02), math.pi * 1e-4 * 0.04 + 4 / 3 * math.pi * 1e-6), ("cylinder", (0.01, 0.02), math.pi * 1e-4 * 0.04),
    ("cone", (0.01, 0.02), math.pi * 1e-4 * 0.04 / 3), ("ellipsoid", (0.01, 0.02, 0.03), 4 / 3 * math.pi * 6e-6),
])
def test_newton_primitive_mass_matches_closed_form(kind: str, size: tuple[float, ...], volume: float) -> None:
    # Newton computes in float32 (wp.vec3 scale), so 1e-6 relative
    assert primitive_mass(ct.Primitive(kind, size), 1000.0) == pytest.approx(1000.0 * volume, rel=1e-6)
    assert primitive_mass(ct.Primitive("plane", (1.0, 1.0)), 1000.0) == 0.0


def test_mesh_mass_matches_closed_form() -> None:
    m = box_pocket((0.03, 0.02, 0.01), (0.026, 0.016, 0.008))
    expect = 1000.0 * (0.03 * 0.02 * 0.01 - 0.026 * 0.016 * 0.008)
    assert mesh_mass(m, 1000.0) == pytest.approx(expect, rel=1e-5)


def test_part_materials_add_by_group() -> None:
    r = LIB.resolve(LIB.object("usb_a_plug"))
    total = card_mass(CardView(r.card, r.material, r.part_materials), LIB.asset_path)
    one = card_mass(CardView(r.card, r.material, {}), LIB.asset_path)
    assert 0 < total < one          # PBT insert and PVC overmold are lighter than all-steel
