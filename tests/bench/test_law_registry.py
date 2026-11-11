# SPDX-License-Identifier: Apache-2.0
"""Law registry and discovery; laws from unmerged producers are not_run (SPEC D23)."""

from __future__ import annotations

import dataclasses

import pytest

from alloy.bench import law as L
from alloy.bench.types import LawSpec, Tolerance
from alloy.core.errors import LawError


def spec(law_id: str = "T1.test", **kw) -> LawSpec:
    base = dict(id=law_id, suite="analytic", metrics={"m": Tolerance(abs=1.0, source="t")}, stat={"m": "mean"},
                dt=1e-3, substeps=1, duration_s=0.1)
    base.update(kw)
    return LawSpec(**base)


def test_register_and_list() -> None:
    s = spec()
    L.register_law(s, object)
    L.register_law(s, object)  # idempotent
    assert any(x.id == "T1.test" for x, _ in L.registered_laws("analytic"))
    with pytest.raises(LawError):
        L.register_law(dataclasses.replace(s, duration_s=0.2), object)
    L.unregister_law("T1.test")


def test_register_rejects_unsourced_tolerance() -> None:
    with pytest.raises(LawError):
        L.register_law(spec("T2.x", metrics={"m": Tolerance(abs=1.0)}), object)
    with pytest.raises(LawError):
        L.register_law(spec("T3.x", stat={"other": "mean"}), object)


def test_sort_order_numeric() -> None:
    ids = ["A10.x", "A2.y", "A1.z", "D1.a"]
    assert sorted(ids, key=L._sort_key) == ["A1.z", "A2.y", "A10.x", "D1.a"]


def test_discovery_registers_wp6_laws() -> None:
    status = L.discover(("alloy.bench.analytic", "alloy.bench.determinism", "alloy.bench.twins"))
    assert all(v is None for v in status.values()), status
    have = {L.law_prefix(s.id) for s, _ in L.registered_laws()}
    assert {"A1", "A2", "A3", "A4", "A5", "A6", "A15", "A16", "D1", "D2", "D3", "R3", "R4"} <= have


def test_unmerged_producer_is_not_run(monkeypatch) -> None:
    fake = L.CatalogueEntry("A99", "analytic", "WP99", "alloy.not_merged_pkg.laws", "x")
    monkeypatch.setattr(L, "CATALOGUE", L.CATALOGUE + (fake,))
    status = L.discover(("alloy.not_merged_pkg.laws",))
    assert "not merged" in status["alloy.not_merged_pkg.laws"]
    missing = dict((e.prefix, r) for e, r in L.missing_laws("analytic"))
    assert "A99" in missing and "not merged" in missing["A99"]
    res = L.not_run_result("A99", "any", "cpu", missing["A99"])
    assert res.verdict == "not_run" and res.verdicts[0].verdict == "not_run"
    assert res.monitors["not_run_reason"] == missing["A99"]


def test_broken_producer_reports_reason(monkeypatch, tmp_path) -> None:
    pkg = tmp_path / "brokenprod"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    status = L.discover(("brokenprod",))
    assert "failed to import" in status["brokenprod"] and "boom" in status["brokenprod"]
