# SPDX-License-Identifier: Apache-2.0
"""alloy.looks: look files resolve object -> material default -> grey, and every library card has a non-grey look."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pytest

from alloy.cards import CardLibrary
from alloy.cards.library import default_root
from alloy.looks import LookFileError, coverage, material_look, object_look, resolve, validate_look
from alloy.looks.resolve import look_dirs

LIB = default_root()
BALL = "rubber_ball_40mm"                       # primitive sphere, material silicone_40a


def _write(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def _mat(name: str, preset: str, **over) -> dict:
    return {"schema": "alloy.look/1", "material": name, "preset": preset, **over}


@pytest.fixture
def card_root(tmp_path: Path) -> Path:
    """A one-object card root (the ball and its material) with no look files yet."""
    for kind, name in (("objects", BALL), ("materials", "silicone_40a")):
        dst = tmp_path / kind / f"{name}.yaml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(LIB / kind / f"{name}.yaml", dst)
    return tmp_path


def _lib(*roots: Path) -> CardLibrary:
    return CardLibrary(roots=list(roots), assets=None)


def test_no_look_file_falls_back_to_grey_and_says_which_file(card_root, caplog):
    with caplog.at_level(logging.WARNING, logger="alloy.looks"):
        look = resolve(BALL, _lib(card_root))
    assert look.source == "fallback" and look.spec()["preset"] == "plastic_matte"
    assert "looks/materials/silicone_40a.json" in caplog.text
    assert coverage(_lib(card_root))["grey"] == [BALL]


def test_material_default_look_applies_to_its_objects(card_root):
    _write(card_root / "looks/materials/silicone_40a.json", _mat("silicone_40a", "rubber_grey", roughness=0.7))
    look = resolve(BALL, _lib(card_root))
    assert (look.source, look.name) == ("material", BALL)
    assert look.spec() == {"preset": "rubber_grey", "roughness": 0.7}
    assert material_look("silicone_40a", _lib(card_root)).path.endswith("silicone_40a.json")
    assert coverage(_lib(card_root)) == {"objects": 1, "object": 0, "material": 1, "fallback": 0, "grey": []}


def test_object_look_wins_and_assigns_parts_by_regex(card_root):
    _write(card_root / "looks/materials/silicone_40a.json", _mat("silicone_40a", "rubber_grey"))
    _write(card_root / "looks/objects/toys/rubber_ball_40mm.json", {
        "schema": "alloy.look/1", "object": BALL, "default": "body",
        "materials": {"body": {"preset": "plastic_gloss", "base_color": [0.8, 0.1, 0.1]}, "seam": {"preset": "abs_black"}},
        "assign": [{"match": "^seam", "material": "seam"}]})
    look = resolve(BALL, _lib(card_root))
    assert look.source == "object"
    assert look.spec("seam_0")["preset"] == "abs_black"
    assert look.spec("shell")["preset"] == look.spec()["preset"] == "plastic_gloss"
    assert object_look("nonexistent", _lib(card_root)) is None


def test_first_root_wins_like_cards(card_root, tmp_path_factory):
    over = tmp_path_factory.mktemp("over")
    _write(over / "looks/materials/silicone_40a.json", _mat("silicone_40a", "rubber_black"))
    _write(card_root / "looks/materials/silicone_40a.json", _mat("silicone_40a", "rubber_grey"))
    assert [d.parent for d in look_dirs(_lib(over, card_root))] == [over, card_root]
    assert resolve(BALL, _lib(over, card_root)).spec()["preset"] == "rubber_black"


@pytest.mark.parametrize(("data", "fragment"), [
    ({"material": "abs", "preset": "abs_black"}, "'schema' must be"),
    (_mat("abs", "no_such_preset"), "not a WP20 preset"),
    (_mat("abs", "abs_black", glow=1), "unknown key 'glow'"),
    ({"schema": "alloy.look/1", "preset": "abs_black"}, "names a 'material', an 'object'"),
    ({"schema": "alloy.look/1", "object": "x", "materials": {"a": {"preset": "abs_black"}}, "default": "b"}, "'default'"),
    ({"schema": "alloy.look/1", "object": "x", "materials": {"a": {"preset": "abs_black"}}, "default": "a",
      "assign": [{"match": "(", "material": "a"}]}, "valid 'match' regex"),
    ({"schema": "alloy.look/1", "object": "x", "materials": {"a": {"preset": "abs_black"}}, "default": "a",
      "assign": [{"match": "p", "material": "z"}]}, "not a key of 'materials'"),
])
def test_validation_names_the_problem(data, fragment):
    issues = validate_look(data)
    assert any(fragment in i for i in issues), issues


def test_file_name_must_match_the_key(card_root):
    path = _write(card_root / "looks/materials/silicone_40a.json", _mat("abs", "abs_black"))
    assert any("must match the file name" in i for i in validate_look(json.loads(path.read_text()), path))
    with pytest.raises(LookFileError, match="must match the file name"):
        resolve(BALL, _lib(card_root))


def test_library_look_files_are_valid():
    files = sorted((LIB / "looks").rglob("*.json"))
    assert files, "library/looks is empty"
    issues = [i for p in files for i in validate_look(json.loads(p.read_text()), p)]
    assert not issues, issues


def test_every_library_object_has_a_look_that_is_not_grey():
    report = coverage(CardLibrary(assets=None))
    assert report["objects"] > 50
    assert not report["grey"], (f"these library objects would render grey: {report['grey']}; add "
                                "library/looks/materials/<their material>.json (or an object look)")


def test_library_looks_render_in_the_path_tracer_too():
    pytest.importorskip("warp")
    from alloy.camera.looks.spec import check_supported, resolve_spec

    lib = CardLibrary(assets=None)
    bad = []
    for name in lib.names("objects"):
        for key, spec in resolve(name, lib).materials.items():
            try:
                check_supported(key, resolve_spec(key, spec))
            except Exception as exc:  # noqa: BLE001 - collect every unsupported look, then fail once
                bad.append(f"{name}.{key}: {exc}")
    assert not bad, bad


def test_fallback_look_gives_each_card_its_look_inside_its_own_labels(card_root):
    import re

    from alloy.looks.fallback import card_name, fallback_look

    _write(card_root / "looks/materials/silicone_40a.json", _mat("silicone_40a", "rubber_grey"))
    _write(card_root / "looks/objects/rubber_ball_40mm.json", {
        "schema": "alloy.look/1", "object": BALL, "default": "body",
        "materials": {"body": {"preset": "plastic_gloss"}, "seam": {"preset": "abs_black"}},
        "assign": [{"match": "seam", "material": "seam"}]})
    fb = fallback_look([f"{BALL}/hull_0", f"{BALL}#1", "not_a_card/seam"], _lib(card_root))

    def pick(label: str) -> str | None:
        return next((r["material"] for r in fb["assign"] if re.search(r["match"], label)), None)

    assert card_name(f"{BALL}#1") == card_name(f"{BALL}/x") == BALL
    assert pick(f"{BALL}/seam_2") == f"{BALL}.seam" and pick(BALL) == pick(f"{BALL}#1") == f"{BALL}.body"
    assert pick("not_a_card/seam") is None and pick(f"{BALL}x") is None
    assert fb["materials"][f"{BALL}.seam"] == {"preset": "abs_black"}
