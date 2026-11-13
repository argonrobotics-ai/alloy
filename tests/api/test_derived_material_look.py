# SPDX-License-Identifier: Apache-2.0
"""A per-object override (``mass=``, ``density=``, ``friction=``, ``restitution=``) keeps the base material's look
(CPU, no build).

The override makes a material ``<base>_for_<object>`` that had no look file, so ``alloy.vary`` left the object out of
its texture swaps (it resolved to the grey fallback, and the camera drew it in a raw palette colour beside its
twin's wood) and ``scene.save()`` wrote the derived card without a look. Law: the derived material takes its base's
look file, so the object varies as its base does and the saved folder carries that look.
"""

from __future__ import annotations

import json
from pathlib import Path

import alloy
from alloy.api import catalog


def _scene() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("wood_default", size=(0.12,) * 3, pos=(-0.12, 0.0, 0.06), material="wood_beech")
    scene.add_box("wood_with_mass", size=(0.12,) * 3, pos=(0.12, 0.0, 0.06), material="wood_beech", mass=0.8)
    scene.add_cloth("cloth_with_mass", size=(0.16, 0.16), pos=(0.12, 0.22, 0.005), resolution=4,
                    material="cotton_woven", mass=0.05)
    return scene


def _look(material: str) -> dict:
    root = catalog.library_root() / "looks" / "materials"
    return json.loads(next(root.rglob(f"{material}.json")).read_text(encoding="utf-8"))


def test_alloy_vary_swaps_the_texture_of_an_object_with_an_override(tmp_path: Path) -> None:
    alloy.vary(_scene(), 1, pos=0.0, seed=0, out=tmp_path / "variants")
    manifest = json.loads((tmp_path / "variants" / "variants.json").read_text(encoding="utf-8"))
    swapped = set(manifest["variants"][0]["textures"])
    assert {"wood_default", "wood_with_mass", "cloth_with_mass"} <= swapped


def test_saved_cards_carry_the_base_look_of_a_derived_material(tmp_path: Path) -> None:
    _scene().save(tmp_path / "cards")
    for base, derived in (("wood_beech", "wood_beech_for_wood_with_mass"),
                          ("cotton_woven", "cotton_woven_for_cloth_with_mass")):
        assert (tmp_path / "cards" / "materials" / f"{derived}.yaml").is_file()
        look = json.loads((tmp_path / "cards" / "looks" / "materials" / f"{derived}.json").read_text())
        assert look["material"] == derived
        kept = {k: v for k, v in _look(base).items() if k not in ("material", "note")}
        assert {k: look[k] for k in kept} == kept                       # preset, texture, scale: the base's own
