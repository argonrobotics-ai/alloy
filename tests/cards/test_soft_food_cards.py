# SPDX-License-Identifier: Apache-2.0
"""Soft food and tissue library objects: cards validate, every value carries a sourcing tag, the key values keep
their evidence, each object takes its soft route, and the one-line scene builds and rests on CPU."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.cards.library import CardLibrary
from alloy.cards.paths import iter_q_leaves
from alloy.cards.validate import errors, validate_material
from alloy.sourcing.tags import has_link, tag_of

LIB = CardLibrary(assets=None)

#: object -> (material, route)
OBJECTS = {
    "cucumber_slicing": ("cucumber_flesh", "soft_vbd"),
    "dough_ball": ("dough_wheat", "paste_mpm"),
    "tissue_pad_skin": ("tissue_skin_silicone", "soft_vbd"),
    "tissue_pad_fat": ("tissue_fat_silicone", "soft_vbd"),
    "gel_block_ballistic": ("gel_gelatin_ballistic", "soft_vbd"),
    "foam_pad_pu": ("foam_pu_flexible", "soft_vbd"),
    "fruit_tomato": ("fruit_tomato_whole", "soft_vbd"),
    "fruit_peach": ("fruit_peach_whole", "soft_vbd"),
    "sponge_cellulose": ("sponge_cellulose_wet", "soft_vbd"),
}
#: values whose evidence is pinned: (material, card path) -> tag
EVIDENCE = {
    ("cucumber_flesh", "elastic.youngs"): "DOC",
    ("cucumber_flesh", "failure.fracture_stress"): "DOC",
    ("cucumber_flesh", "failure.fracture_strain"): "DOC",
    ("cucumber_flesh", "density"): "SNIPPET",
    ("dough_wheat", "density"): "DOC",
    ("dough_wheat", "plastic.yield_stress"): "DOC",           # measured G'/G'' crossover (was a simulation input)
    ("tissue_skin_silicone", "density"): "DOC",
    ("tissue_skin_silicone", "failure.fracture_stress"): "DOC",
    ("tissue_fat_silicone", "elastic.youngs"): "DOC",
    ("tissue_fat_silicone", "failure.fracture_strain"): "DOC",
    ("gel_gelatin_ballistic", "density"): "DOC",
    ("gel_gelatin_ballistic", "elastic.youngs"): "DOC",
    ("foam_pu_flexible", "density"): "DOC",
    ("foam_pu_flexible", "elastic.youngs"): "DOC",
    ("fruit_tomato_whole", "elastic.youngs"): "DOC",
    ("fruit_tomato_whole", "surface.mu_static"): "SNIPPET",
    ("fruit_peach_whole", "elastic.youngs"): "DOC",
    ("fruit_peach_whole", "surface.mu_static"): "DOC",
    ("fruit_peach_whole", "density"): "SNIPPET",
    ("sponge_cellulose_wet", "elastic.poisson"): "DOC",
    ("sponge_cellulose_wet", "elastic.youngs"): "ESTIMATE",          # two proxies; no kitchen-sponge modulus published
}


@pytest.mark.parametrize("name", sorted(OBJECTS))
def test_object_and_material_cards_validate(name: str) -> None:
    material, _ = OBJECTS[name]
    assert LIB.peek_object(name).material == material
    assert errors(LIB.object_issues(name)) == []
    assert errors(validate_material(LIB.material(material))) == []


@pytest.mark.parametrize("material", sorted({m for m, _ in OBJECTS.values()}))
def test_every_value_is_tagged_and_documents_carry_a_link(material: str) -> None:
    for path, q in iter_q_leaves(LIB.material(material)):
        tag = tag_of(q)
        assert tag in ("DOC", "SNIPPET", "UNVERIFIED", "ESTIMATE"), (material, path, tag)
        if tag in ("DOC", "SNIPPET"):
            assert has_link(q.note) and "'" in q.note, (material, path)       # a URL and a quote
        if q.unc is not None:
            assert q.unc[0] < q.unc[1] and q.unc[0] <= q.value <= q.unc[1], (material, path)


def test_key_values_keep_their_evidence() -> None:
    leaves = {(m, path): q for m in {m for m, _ in OBJECTS.values()} for path, q in iter_q_leaves(LIB.material(m))}
    for key, tag in EVIDENCE.items():
        assert tag_of(leaves[key]) == tag, key


@pytest.mark.parametrize("name", sorted(OBJECTS))
def test_route(name: str) -> None:
    from alloy.compose.catalog import Catalog  # noqa: PLC0415 - heavy

    assert Catalog().info(name).route == OBJECTS[name][1]


@pytest.mark.compile_heavy
@pytest.mark.parametrize("name, z", [("cucumber_slicing", 0.025), ("tissue_pad_fat", 0.0055),
                                         ("fruit_tomato", 0.033), ("sponge_cellulose", 0.0135)])
def test_one_line_use_builds_and_rests_on_cpu(name: str, z: float) -> None:
    import alloy  # noqa: PLC0415 - heavy

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_object("it", name, pos=(0.0, 0.0, z))
    sim = alloy.build(scene, device="cpu")
    try:
        pts = sim.run(0.1, record=["it"], every=5).world(0)["it.points"]
        assert np.isfinite(pts).all()
        assert abs(pts[-1][:, 2].min() - pts[0][:, 2].min()) < 0.005      # rests: moved under 5 mm
        assert sim.check()["ok"]
    finally:
        sim.close()
