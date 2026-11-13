# SPDX-License-Identifier: Apache-2.0
"""Audit S2-021: docs/guide/check.md saves a scene (``card = scene.save(...)``, a path) and calls
``alloy.check.apply_fixes(card, report)``; that raised a raw AttributeError ('PosixPath' object has no attribute
'objects'). apply_fixes now takes what alloy.check takes (a SceneCard, a .yaml path as Path or str, a library scene
name); a SceneCard comes back fixed, a saved .yaml as the path of a fixed copy beside it (its own cards still
resolve there); anything else raises alloy.SceneError with a fix. CPU, load checks only."""

from __future__ import annotations

from pathlib import Path

import pytest

import alloy
import alloy.check
from alloy.cards import types as ct


def _saved(tmp_path: Path) -> Path:
    scene = alloy.Scene(worlds=2, dt=0.01)                 # the guide's scene: the block 5 mm into the table top
    scene.add_ground()
    scene.add_box("table", size=(0.6, 0.4, 0.4), pos=(0, 0, 0.2), static=True)
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.42), mass=0.2)
    return scene.save(tmp_path / "checked_scene")


@pytest.mark.parametrize("as_text", [False, True])
def test_apply_fixes_takes_the_saved_path(tmp_path: Path, as_text: bool):
    card = _saved(tmp_path)
    report = alloy.check(str(card), worlds=2, device="cpu", run=False)
    assert report.checks["overlap"] == "fail"
    before = Path(card).read_text(encoding="utf-8")
    fixed, applied = alloy.check.apply_fixes(str(card) if as_text else card, report)
    assert type(fixed) is (str if as_text else type(card)) and len(applied) == 1 and "lift block" in applied[0]
    assert Path(fixed) == Path(card).with_name(f"{Path(card).stem}_fixed.yaml")
    assert Path(card).read_text(encoding="utf-8") == before                    # the file given is never changed
    assert alloy.check(fixed, worlds=2, device="cpu", run=False).checks["overlap"] == "pass"   # its own cards resolve


def test_apply_fixes_keeps_a_scene_card_in_memory(tmp_path: Path):
    from alloy.cards.loader import load_scene

    card = _saved(tmp_path)
    fixed, applied = alloy.check.apply_fixes(load_scene(card), alloy.check(card, worlds=2, device="cpu", run=False))
    assert isinstance(fixed, ct.SceneCard) and len(applied) == 1
    assert not Path(card).with_name(f"{Path(card).stem}_fixed.yaml").exists()


@pytest.mark.parametrize("card, report, field", [(42, "ok", "card"), ("ok", {"ok": False}, "report")])
def test_apply_fixes_wrong_inputs_say_what_to_pass(tmp_path: Path, card, report, field):
    path = _saved(tmp_path)
    good = alloy.check(path, worlds=2, device="cpu", run=False)
    card, report = (path if card == "ok" else card), (good if report == "ok" else report)
    with pytest.raises(alloy.SceneError) as exc:
        alloy.check.apply_fixes(card, report)
    assert isinstance(exc.value, alloy.AlloyAPIError) and exc.value.field == field and "pass " in exc.value.fix


def test_check_of_a_wrong_scene_argument_is_a_scene_error():
    with pytest.raises(alloy.SceneError) as exc:
        alloy.check(42, device="cpu", run=False)
    assert exc.value.field == "scene" and "YAML path" in exc.value.fix


def test_check_of_an_unknown_contacts_value_is_a_scene_error():
    with pytest.raises(alloy.SceneError) as exc:
        alloy.check("demo_peg_d12", device="cpu", run=False, contacts="bullet")
    assert exc.value.field == "contacts" and "'newton'" in exc.value.fix
