# SPDX-License-Identifier: Apache-2.0
"""OSS-7 report types and fix arithmetic: Finding validation, JSON shape, verdicts, mm wording, which object a
pose fix moves (no simulation)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alloy.check.advice import clearance, mm, separate, worlds_text
from alloy.check.contacts import PairDepth
from alloy.check.objects import ObjectIndex
from alloy.check.overlap import _finding
from alloy.check.report import SCHEMA, CheckReport, Finding, verdict


def _index(motion: dict[str, str], poses: dict[str, tuple] | None = None, worlds: int = 1) -> ObjectIndex:
    names = list(motion)
    empty = np.zeros(0, np.int32)
    placement = {n: i for i, n in enumerate(names) if motion[n] != "robot"}
    return ObjectIndex(names, motion, placement, empty, empty, empty, worlds, poses or {})


def test_finding_rejects_unknown_check_and_level():
    with pytest.raises(ValueError, match="use one of"):
        Finding("Gravity!", "error", "m", "f")          # extra check names are lowercase identifiers
    with pytest.raises(ValueError, match="use one of"):
        Finding("overlap", "fatal", "m", "f")


def test_report_json_round_trip_and_ok():
    r = CheckReport(scene="s", settings={"worlds": 2})
    r.add([Finding("overlap", "warn", "a and b overlap", "lift a", objects=("a", "b"), value=float("nan"))],
          "overlap")
    r.add([Finding("cards", "error", "bad density", "set density inside [10, 25000]")], "cards")
    r.skip("energy", "no monitors")
    d = json.loads(r.to_json())
    assert d["schema"] == SCHEMA and d["ok"] is False and d["errors"] == 1 and d["warnings"] == 1
    assert d["findings"][0]["level"] == "error"                 # errors sorted first
    assert d["findings"][1]["value"] is None                    # NaN -> null, JSON stays strict
    assert d["checks"] == {"overlap": "warn", "cards": "fail", "energy": "skipped: no monitors"}
    assert "lift a" in r.summary()


def test_verdict_levels():
    assert verdict([]) == "pass"
    assert verdict([Finding("energy", "info", "m", "f")]) == "pass"
    assert verdict([Finding("energy", "warn", "m", "f")]) == "warn"


def test_mm_and_clearance():
    assert mm(0.0032) == "3.2 mm" and mm(5e-6) == "5 um"
    assert clearance(0.0032) == pytest.approx(0.004)           # 3.2 mm + 0.5 mm, up to the next 0.5 mm
    assert clearance(0.000106) == pytest.approx(0.0002)        # under 1 mm: + 0.05 mm, up to the next 0.1 mm
    assert clearance(0.0) == pytest.approx(0.0001)
    assert worlds_text([3, 1, 7, 9], 64) == "in 4 of 64 worlds (e.g. 1, 3, 7)"
    assert worlds_text([0], 1) == ""


def test_separate_moves_the_dynamic_object_up():
    idx = _index({"table": "static", "mug": "dynamic"}, {"mug": (0.1, 0.0, 0.05, 0, 0, 0, 1)})
    mv = separate(idx, "mug", "table", (0.0, 0.0, -1.0), 0.0032)   # normal from mug toward table: down
    assert mv is not None and mv.name == "mug" and mv.verb() == "lift mug by 4 mm"
    assert mv.pose[2] == pytest.approx(0.054)
    assert mv.suggestion(idx) == {"object": "mug", "field": "objects.1.pose",
                                  "pose": [0.1, 0.0, 0.054, 0, 0, 0, 1]}


def test_separate_sideways_and_immovable():
    idx = _index({"a": "dynamic", "b": "static"}, {"a": (0, 0, 0, 0, 0, 0, 1)})
    mv = separate(idx, "a", "b", (1.0, 0.0, 0.0), 0.002)        # b is +x of a: move a toward -x
    assert mv.name == "a" and mv.verb().startswith("move a 2.5 mm along (-1.00")
    both_static = _index({"a": "static", "b": "kinematic"})
    assert separate(both_static, "a", "b", (0, 0, 1), 0.002) is None


def test_separate_prefers_the_upper_of_two_dynamic_objects():
    idx = _index({"box": "dynamic", "ball": "dynamic"})
    mv = separate(idx, "box", "ball", (0.0, 0.0, 1.0), 0.001)    # ball sits on box: lift the ball
    assert mv.name == "ball" and mv.pose is None and mv.suggestion(idx) is None


def _pair(a, b, depth=0.003, particles=None):
    return PairDepth(a, b, depth, 0, (0.0, 0.0, 1.0), {1e-4: [0], 1e-3: [0]}, particles)


def test_overlaps_the_simulation_ignores_are_info():
    idx = _index({"bay": "static", "pins": "kinematic", "plug": "dynamic", "arm": "robot"})
    idx.attached = {frozenset(("plug", "bay"))}
    assert _finding(_pair("bay", "pins"), idx).level == "info"            # neither moves
    assert _finding(_pair("arm", "arm"), idx).level == "info"             # a robot's own links
    assert _finding(_pair("bay", "plug"), idx).level == "info"            # attached
    f = _finding(_pair("arm", "bay"), idx)
    assert f.level == "error" and f.fix.startswith("lower arm")           # bay is above the robot: move it down


def test_particle_owner_is_named_after_sorting():
    idx = _index({"board": "static", "cucumber": "dynamic"}, {"cucumber": (0, 0, 0.06, 0, 0, 0, 1)})
    f = _finding(_pair("board", "cucumber", particles="cucumber"), idx)
    assert f.message.startswith("board and particles of cucumber overlap by 3 mm")
    f = _finding(_pair("board", "cucumber", particles="board"), idx)
    assert f.message.startswith("particles of board and cucumber")


class _Lib:
    """A stand-in library holding one imported card at ``path``."""

    def __init__(self, path):
        self.path = path

    def has_object(self, name):
        return name == "ycb_mug"

    def path_of(self, kind, name):
        return self.path


def test_imported_files_fetched_or_reported(tmp_path, monkeypatch):
    from alloy.check import fetch
    from alloy.compose import fetch as store

    card = tmp_path / "imports" / "ycb_gso" / "objects" / "ycb" / "ycb_mug.yaml"
    card.parent.mkdir(parents=True)
    card.write_text("x", encoding="utf-8")
    lib = _Lib(card)
    assert fetch.import_root_of(lib, "ycb_mug") == (tmp_path / "imports" / "ycb_gso").resolve()
    assert fetch.import_root_of(lib, "other") is None
    monkeypatch.setattr(store, "missing", lambda root, name: ["meshes/ycb_mug/physics.obj"])
    monkeypatch.delenv(store.STORE_ENV, raising=False)
    (f,) = fetch.fetch_imported(lib, ["ycb_mug", "other"])
    assert f.level == "error" and "ALLOY_ASSET_STORE" in f.fix and "alloy.importers.ycb_gso fetch" in f.fix
    got = []
    monkeypatch.setenv(store.STORE_ENV, str(tmp_path / "store"))
    monkeypatch.setattr(store, "fetch_object", lambda root, name, s: got.append(name) or 1)
    assert fetch.fetch_imported(lib, ["ycb_mug"]) == [] and got == ["ycb_mug"]
