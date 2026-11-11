# SPDX-License-Identifier: Apache-2.0
"""library/labels.yaml (CARD-2 input): >= 60 expert routes over library objects, valid route names, required cases."""

from __future__ import annotations

from alloy.cards.library import CardLibrary, default_root
from alloy.cards.units_yaml import yaml_load
from alloy.route.types import Route

DOC = yaml_load((default_root() / "labels.yaml").read_text(encoding="utf-8"), "labels")
LIB = CardLibrary(assets=None)


def test_labels_well_formed() -> None:
    assert DOC["schema"] == "alloy.labels/1"
    labels = DOC["labels"]
    assert len(labels) >= 60
    names = [lab["object"] for lab in labels]
    assert len(set(names)) == len(names)
    routes = {r.value for r in Route}
    for lab in labels:
        assert set(lab) == {"object", "route", "reason"}
        assert LIB.has_object(lab["object"]), lab
        assert lab["route"] in routes, lab
        assert lab["reason"].strip()


def test_labels_cover_the_hard_cases() -> None:
    names = " ".join(lab["object"] for lab in DOC["labels"])
    for word in ("pencil", "chopstick", "spoon", "screwdriver", "tray", "lid"):
        assert word in names, word
    routes = {lab["route"] for lab in DOC["labels"]}
    assert {"rigid", "static", "chain", "cloth_vbd", "soft_vbd", "grain_mpm", "liquid_sph", "paste_mpm", "panels"} <= routes


def test_every_object_is_labelled() -> None:
    assert {lab["object"] for lab in DOC["labels"]} == set(LIB.names("objects"))
