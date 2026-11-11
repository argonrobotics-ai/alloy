# SPDX-License-Identifier: Apache-2.0
"""OSS-19 end to end on CPU with the engine: a sentence becomes a scene that passes alloy.check, variations differ,
the same seed gives the same scene, saved folders load and check on their own, the floor support writes its card,
objects that cannot be placed are reported, and the command line prints JSON with exit codes."""

from __future__ import annotations

import json

import pytest

import alloy.check
from alloy.cards.library import CardLibrary, default_root
from alloy.compose import cli
from alloy.compose.build import compose
from alloy.compose.catalog import Catalog
from alloy.compose.load import library, load_scene


@pytest.fixture(scope="module")
def cat() -> Catalog:
    return Catalog(library=CardLibrary([default_root()], assets=None))


def _poses(scene) -> list[tuple]:
    return [tuple(o["pose"]) for o in scene.objects]


def test_sentence_to_a_load_checked_scene_and_variations(cat, tmp_path):
    r = compose("a mug and two wooden cubes on a table", variations=2, check="load", device="cpu", catalog=cat)
    assert r.ok and len(r.scenes) == 3
    base = r.scenes[0]
    assert sorted(o["card"] for o in base.objects) == ["cube_beech_30", "cube_beech_30", "mug_stoneware"]
    assert {o["name"] for o in base.objects} == {"mug_stoneware", "cube_beech_30", "cube_beech_30#1"}
    assert all(o["on"] == "table_top_plywood" for o in base.objects)
    assert len({tuple(_poses(s)) for s in r.scenes}) == 3                     # variations differ
    again = compose("a mug and two wooden cubes on a table", check="none", device="cpu", catalog=cat)
    assert _poses(again.scenes[0]) == _poses(base)                           # same seed, same scene
    paths = r.save(tmp_path)
    assert (tmp_path / "scenes" / "scene_v02.yaml") in paths and (tmp_path / "compose.json").is_file()
    report = alloy.check.check_load(str(tmp_path / "scenes" / "scene.yaml"), device="cpu",
                                    library=library([tmp_path]))
    assert report.ok, report.summary()
    assert json.loads((tmp_path / "compose.json").read_text())["scenes"][0]["ok"] is True


def test_full_check_passes_and_reports(cat):
    r = compose("three cubes on a tidy table", check="full", device="cpu", worlds=1, seconds=0.2, catalog=cat)
    report = r.scenes[0].report
    assert r.ok and report is not None and report["ok"], r.summary()
    assert report["checks"]["overlap"] == "pass" and report["checks"]["rest_drift"] in ("pass", "warn")


def test_floor_support_writes_its_card_and_loads_as_a_scene(cat, tmp_path):
    r = compose("a cube on the floor", check="load", device="cpu", catalog=cat)
    assert r.ok and "compose_floor" in r.generated
    r.save(tmp_path)
    assert (tmp_path / "objects" / "compose_floor.yaml").is_file()
    scene = load_scene(tmp_path / "scenes" / "scene.yaml")
    assert sorted(scene.names()) == ["compose_floor", "cube_beech_30"]


def test_unplaceable_objects_are_reported_not_fatal(cat):
    r = compose("a bottle and a cube", check="none", catalog=cat)          # the PET bottle routes to cloth
    assert not r.ok and r.scenes[0].missing == ["1 x bottle"]
    assert any("bottle" in line and "cloth_vbd" in line for line in r.scenes[0].log)
    assert [o["card"] for o in r.scenes[0].objects] == ["cube_beech_30"]


def test_cli_json_and_exit_codes(tmp_path, capfd):
    assert cli.main(["parse", "two cubes on the floor"]) == 0
    assert json.loads(capfd.readouterr().out)["support"] == "floor"
    assert cli.main(["search", "cube", "--fast", "-k", "2"]) == 0
    assert json.loads(capfd.readouterr().out)["matches"][0]["name"]
    assert cli.main(["a cube on a table", "--check", "none", "--out", str(tmp_path)]) == 0
    out = json.loads(capfd.readouterr().out)
    assert out["ok"] and any(p.endswith("scene.yaml") for p in out["saved"])
    assert cli.main(["build", "a table"]) == 2
    err = json.loads(capfd.readouterr().out)
    assert err["error"] == "ComposeError" and err["fix"]
    assert cli.main(["build", "--variations", "x"]) == 2
    assert json.loads(capfd.readouterr().out)["error"] == "UsageError"


def test_an_object_that_does_not_build_is_swapped_for_another_match(monkeypatch):
    from alloy.compose import critic as C

    cat = Catalog(library=CardLibrary([default_root()], assets=None))      # its own: marks are per catalog
    real = C.Critic.after_place

    first = cat.search("peg")[0].name                                    # pegs: four equally good cards

    def fake(self, layout, key):
        return C.BROKEN if key.startswith(first) else real(self, layout, key)

    monkeypatch.setattr(C.Critic, "after_place", fake)
    r = compose("two pegs on a table", check="none", catalog=cat)
    cards = [o["card"] for o in r.scenes[0].objects]
    assert len(cards) == 2 and first not in cards and len(set(cards)) == 1 and cards[0].startswith("peg_")
    assert any(f"{first} does not build, using" in line for line in r.scenes[0].log)
    assert not cat.info(first).placeable


def test_jitter_spreads_free_objects_per_world_and_still_passes(cat):
    r = compose("three cubes on a table", jitter=True, scene_worlds=8, check="full", device="cpu", worlds=3,
                seconds=0.2, catalog=cat)
    card = r.scenes[0].card
    assert r.ok and card.worlds == 8, r.summary()
    for pl in card.objects[1:]:                                   # the table is static: no jitter
        x, y = pl.randomize
        assert (x.field, y.field) == ("pose.0", "pose.1")
        assert x.hi - x.lo == pytest.approx(2 * 0.45 * 0.015) and (x.lo + x.hi) / 2 == pytest.approx(pl.pose[0])
    assert card.objects[0].randomize == ()


def test_an_object_with_no_free_spot_is_reported(cat):
    r = compose("a tray and two cubes on a table", area=(0.0, 0.0, 0.05, 0.05), check="none", catalog=cat)
    assert not r.ok and r.scenes[0].missing                    # the 0.3 m tray cannot fit a 5 cm area... nor more
    assert any("not placed (no free spot" in line for line in r.scenes[0].log)


def test_no_fallback_to_a_weaker_meaning(monkeypatch):
    from alloy.compose import critic as C

    cat = Catalog(library=CardLibrary([default_root()], assets=None))
    monkeypatch.setattr(C.Critic, "after_place", lambda self, layout, key: C.BROKEN if key.startswith("mug")
                        else C.OK)
    r = compose("a mug and a cube", check="none", catalog=cat)              # "cup" synonyms score lower
    assert [o["card"] for o in r.scenes[0].objects] == ["cube_beech_30"] and r.scenes[0].missing


def test_a_gift_bag_stands_open_and_takes_its_bottom_board(cat):
    """SOFT-1's kraft gift bag, a film shell, is placed like an object and stands through the full check; its
    200 x 100 mm bottom board (a flat sheet, placed when named) goes in centred on the 203 x 114 mm bottom."""
    from alloy.compose.layout import GAP

    r = compose("a bottom board in a gift bag", check="full", device="cpu", worlds=1, seconds=0.2, catalog=cat)
    assert r.ok, r.scenes[0].log
    bag, board = r.scenes[0].objects
    assert (bag["card"], board["card"], board["on"]) == ("gift_bag_kraft", "gift_bag_board", bag["name"])
    assert board["pose"][:2] == pytest.approx(bag["pose"][:2], abs=1e-6)
    assert board["pose"][2] - bag["pose"][2] == pytest.approx(GAP + 0.000255, abs=1e-6)
    assert not any("film object" in line for line in r.scenes[0].log)          # the board is film, not a free body
    r = compose("a box in a gift bag", check="none", catalog=cat)
    assert any("rest on this film object" in line for line in r.scenes[0].log)


def test_a_build_error_keeps_its_fix_when_the_message_is_cut():
    """An engine error cut to 200 characters in the log still says what to change (its ``fix``)."""
    from alloy.compose.critic import _why

    class Refused(RuntimeError):
        fix = "make the cloth a film sheet"

    long = Refused("cloth_vbd (towel) with grain_mpm (sand) is not supported in one scene yet: " + "x" * 300)
    assert _why(long).endswith("...; to fix: make the cloth a film sheet")
    assert _why(Refused("short; nearest: make the cloth a film sheet")).count("film sheet") == 1   # not repeated
    assert _why(ValueError("plain")) == "ValueError: plain"
