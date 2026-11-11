# SPDX-License-Identifier: Apache-2.0
"""Sourcing records fill what Gym-Anything left open, and never replace its cited values (CPU)."""

from __future__ import annotations

import json

import pytest

from alloy.cards.library import CardLibrary
from alloy.importers.gymanything.pipeline import ImportOptions, import_specs
from alloy.testing.imp_gymanything import box_mesh, cited, write_spec

RECORD = """schema: alloy.sourcing.object/1
library: gymanything
object: {card}
values:
  mass:
    value: {mass}
    lo: {lo}
    hi: {hi}
    tag: ESTIMATE
    reason: two maker listings
    evidence:
    - {{url: "https://example.org/a", quote: "Weight: {mass} kg", tag: DOC, where: spec table, accessed: "2026-11-06"}}
  restitution:
    value: 0.6
    lo: 0.4
    hi: 0.8
    tag: ESTIMATE
    reason: category prior
"""


def _record(root, card, mass):
    d = root / "gymanything"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{card}.yaml").write_text(RECORD.format(card=card, mass=mass, lo=0.9 * mass, hi=1.1 * mass))


def test_records_fill_defaults_and_open_masses_only(tmp_path):
    specs, src, out = tmp_path / "specs", tmp_path / "sources", tmp_path / "out"
    write_spec(specs, box_mesh((0.05, 0.05, 0.05)), key="guessed", mass=cited(1.0, "estimate", "a.py:1"))
    write_spec(specs, box_mesh((0.05, 0.05, 0.05)), key="datasheet", mass=cited(1.0, "datasheet", "a.py:2"))
    _record(src, "gar_testsim_guessed", 0.4)
    _record(src, "gar_testsim_datasheet", 0.4)
    res = import_specs(specs, out, opt=ImportOptions(hulls=False, sources=src))
    lib = CardLibrary(roots=[out], assets=None)
    guessed, kept = lib.material("gar_testsim_guessed"), lib.material("gar_testsim_datasheet")
    assert guessed.density.value == pytest.approx(0.4 / 1e-3) and "example.org" in guessed.density.note
    assert kept.density.value == pytest.approx(1.0 / 1e-3) and "datasheet" in kept.density.note
    assert guessed.surface.restitution.value == 0.6 and kept.surface.restitution.value == 0.6
    assert res["checked_ok"] == 2 and res["sourced"] == {"mass": 1, "restitution": 2}
    rows = [json.loads(line) for line in (out / "queue.jsonl").read_text().splitlines()]
    assert not [r for r in rows if r["field"] == "material.surface.restitution"]
    assert not [r for r in rows if r["card"] == "gar_testsim_guessed" and r["field"] == "material.density"]
