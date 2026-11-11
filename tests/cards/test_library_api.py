# SPDX-License-Identifier: Apache-2.0
"""CardLibrary: lookup, unordered pairs, shadowing, name checks, put (validated, recorded), assets, scenes."""

from __future__ import annotations

import dataclasses as dc
import json
import shutil
from pathlib import Path

import pytest

from alloy.cards import types as ct
from alloy.cards.library import CardLibrary
from alloy.cards.loader import load_material
from alloy.core.errors import CardError
from alloy.core.protocols import CardLibraryLike
from alloy.core.quantity import Q


def test_protocol_and_lookups() -> None:
    lib = CardLibrary()
    assert isinstance(lib, CardLibraryLike)
    assert lib.pair("stainless_303", "pmma_cast") == lib.pair("pmma_cast", "stainless_303") is not None
    assert lib.pair("water", "honey") is None
    with pytest.raises(CardError) as exc:
        lib.material("unobtainium")
    assert exc.value.field == "name"
    assert lib.asset_path("meshes/nut_m10_din934.obj") is not None and lib.asset_path("meshes/nope.obj") is None
    assert lib.scene("demo_peg_d12").worlds == 16


def test_first_root_wins(tmp_path: Path, tmp_library: Path) -> None:
    over = tmp_path / "over"
    (over / "materials").mkdir(parents=True)
    m = load_material(tmp_library / "materials" / "water.yaml")
    text = (tmp_library / "materials" / "water.yaml").read_text().replace("997.05", "1000.0")
    (over / "materials" / "water.yaml").write_text(text)
    lib = CardLibrary([over, tmp_library])
    assert lib.material("water").density.value == 1000.0 != m.density.value
    assert lib.has_object("peg_d12")                         # still found in the second root


def test_file_name_must_match_card_name(tmp_library: Path) -> None:
    shutil.copy(tmp_library / "materials" / "water.yaml", tmp_library / "materials" / "agua.yaml")
    with pytest.raises(CardError) as exc:
        CardLibrary([tmp_library]).material("agua")
    assert exc.value.field == "name"


def test_invalid_library_card_raises_on_access(tmp_library: Path) -> None:
    p = tmp_library / "materials" / "water.yaml"
    p.write_text(p.read_text().replace("value: 997.05", "value: 5.0"))
    with pytest.raises(CardError) as exc:
        CardLibrary([tmp_library]).material("water")
    assert exc.value.field == "density"


def test_put_validates_writes_and_records(tmp_library: Path) -> None:
    lib = CardLibrary([tmp_library])
    m = lib.material("pp_homopolymer")
    fitted = dc.replace(m, surface=dc.replace(m.surface, mu_static=Q(0.33, "fit", (0.31, 0.35), "WP11 tilt rig run 7")))
    path = lib.put(fitted, reason="fit from tilt rig")
    assert path == tmp_library / "materials" / "pp_homopolymer.yaml"
    assert path.read_text().startswith("# written by CardLibrary.put ")
    assert lib.material("pp_homopolymer") == fitted
    row = json.loads((tmp_library / "history.jsonl").read_text().splitlines()[-1])
    assert row["name"] == "pp_homopolymer" and row["reason"] == "fit from tilt rig" and len(row["digest"]) == 64
    pair = ct.PairCard("water", "abs", Q(0.1, "model", None, "x"), Q(0.1, "model", None, "x"))
    assert lib.put(pair, reason="new pair").name == "abs__water.yaml"
    with pytest.raises(CardError):
        lib.put(dc.replace(m, density=Q(1.0, "fit", None, "bad fit")), reason="should be refused")
    with pytest.raises(CardError):
        lib.put(m, reason=" ")
    obj = dc.replace(lib.object("cube_beech_30"), name="cube_pp")
    lib.put(dc.replace(obj, material="pp_homopolymer"), reason="variant")
    assert lib.object("cube_pp").material == "pp_homopolymer"


def test_scene_validation_raises(tmp_library: Path) -> None:
    p = tmp_library / "scenes" / "demo_peg_d12.yaml"
    p.write_text(p.read_text().replace("object: peg_d12", "object: ghost"))
    with pytest.raises(CardError) as exc:
        CardLibrary([tmp_library]).scene("demo_peg_d12")
    assert exc.value.field == "objects.2.object"


def test_attachment_names_repeat_placements_like_the_assembler(tmp_library: Path) -> None:
    """The k-th repeat placement of an object is `name#k` (WP2 SceneAssembler); attachments may name it."""
    scene = ("schema: alloy.scene/1\nrobots: []\nobjects:\n"
             "  - {object: peg_d12, pose: [0, 0, 0.05, 0, 0, 0, 1]}\n"
             "  - {object: peg_d12, pose: [0.1, 0, 0.05, 0, 0, 0, 1]}\n"
             "attachments:\n  - {a: peg_d12, b: 'peg_d12#1', kind: weld}\n")
    (tmp_library / "scenes" / "two_pegs.yaml").write_text(scene)
    assert CardLibrary([tmp_library]).scene("two_pegs").attachments[0].b == "peg_d12#1"
    (tmp_library / "scenes" / "two_pegs.yaml").write_text(scene.replace("'peg_d12#1'", "'peg_d12#2'"))
    with pytest.raises(CardError) as exc:
        CardLibrary([tmp_library]).scene("two_pegs")
    assert exc.value.field == "attachments.0.b"


def test_attachment_solref_overrides_round_trip_and_validate(tmp_library: Path) -> None:
    """SCR-DC3-3: optional timeconst (s) and dampratio (-) on weld/joint attachments; > 0; not on springs."""
    from alloy.cards.loader import dump_card, loads_card

    scene = ("schema: alloy.scene/1\nrobots: []\nobjects:\n"
             "  - {object: peg_d12, pose: [0, 0, 0.05, 0, 0, 0, 1]}\n"
             "  - {object: block_bore_d12_2, pose: [0.1, 0, 0.05, 0, 0, 0, 1]}\n"
             "attachments:\n  - {a: peg_d12, b: block_bore_d12_2, kind: weld, timeconst: 2.5 ms,\n"
             "     dampratio: {value: 1.0, source: model, note: critical}}\n")
    path = tmp_library / "scenes" / "welded.yaml"
    path.write_text(scene)
    at = CardLibrary([tmp_library]).scene("welded").attachments[0]
    assert at.timeconst == Q(2.5e-3, "default") and at.dampratio == Q(1.0, "model", None, "critical")
    card = loads_card(dump_card(CardLibrary([tmp_library]).peek_scene("welded")), ct.SceneCard)
    assert card.attachments[0] == at
    assert ct.Attachment("a", "b").timeconst is None and ct.Attachment("a", "b").dampratio is None
    for bad, field in (("kind: weld", "attachments.0.timeconst"), ("kind: spring", "attachments.0.timeconst")):
        text = scene.replace("timeconst: 2.5 ms", "timeconst: -1 ms") if bad == "kind: weld" else scene.replace("kind: weld", bad)
        path.write_text(text)
        with pytest.raises(CardError) as exc:
            CardLibrary([tmp_library]).scene("welded")
        assert exc.value.field == field
    path.write_text(scene.replace("timeconst: 2.5 ms", "timeconst: 2.5 N"))
    with pytest.raises(CardError):
        CardLibrary([tmp_library]).scene("welded")


def test_chain_links_override_round_trips_and_validates(tmp_library: Path) -> None:
    """SCR-DC3-7: optional features.chain_links (CHAIN link count): an integer >= 3, not randomizable."""
    from alloy.cards.loader import dump_card, loads_card
    from alloy.cards.validate_object import validate_object
    from alloy.cards.validate_scene import placement_issues

    path = tmp_library / "objects" / "usb_cable_1m.yaml"
    text = path.read_text()
    assert ct.Features().chain_links is None and CardLibrary([tmp_library]).object("usb_cable_1m").features.chain_links is None
    path.write_text(text.replace("features:\n", "features:\n  chain_links: 12\n", 1))
    card = CardLibrary([tmp_library]).object("usb_cable_1m")
    assert card.features.chain_links == 12 and loads_card(dump_card(card), ct.ObjectCard) == card
    for bad in ("2", "12.5", "true", "12 mm"):
        path.write_text(text.replace("features:\n", f"features:\n  chain_links: {bad}\n", 1))
        with pytest.raises(CardError) as exc:
            CardLibrary([tmp_library]).object("usb_cable_1m")
        assert exc.value.field == "features.chain_links", bad
    lib = CardLibrary([tmp_library], assets=None)
    peg = dc.replace(lib.object("peg_d12"), features=dc.replace(lib.object("peg_d12").features, chain_links=8))
    assert [i.field for i in validate_object(peg, lib) if i.level == "warn"].count("features.chain_links") == 1
    pl = ct.Placement("usb_cable_1m", (0, 0, 0.05, 0, 0, 0, 1),
                      (ct.Randomization("features.chain_links", 10.0, 14.0),))
    path.write_text(text.replace("features:\n", "features:\n  chain_links: 12\n", 1))
    lib = CardLibrary([tmp_library])
    bad = [i for i in placement_issues(lib, lib.object("usb_cable_1m"), pl) if i.level == "error"]
    assert bad and "whole number" in bad[0].message
