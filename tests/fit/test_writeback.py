# SPDX-License-Identifier: Apache-2.0
"""alloy.fit.writeback and validate.held_out: refusal on held-out gap, provenance on put."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from alloy.bench.types import MetricVerdict, Quantity
from alloy.cards.library import CardLibrary
from alloy.core.errors import CardError
from alloy.core.quantity import Q
from alloy.fit.errors import FitError, HeldOutGapError
from alloy.fit.types import FitResult
from alloy.fit.validate import held_out
from alloy.fit.writeback import check_heldout, gap_violations, write_back


def _verdict(name: str, error: float, bound: float, verdict: str = "pass") -> MetricVerdict:
    q = Quantity(name, "1", 0.0, 0.0, 10, 0.0)
    return MetricVerdict(name, "1", q, q, error, bound, None, "mean", verdict)  # type: ignore[arg-type]


def _result(gap: float, bound: float = 0.03, params=None) -> FitResult:
    params = params or {"material:steel_carbon.surface.restitution": Q(0.66, "fit", (0.65, 0.67), note="fit-t: test")}
    v = _verdict("restitution@h5", 0.004 + gap, bound)
    return FitResult(params, {"restitution@h5": 0.004}, {"restitution@h5": 0.004 + gap},
                     {"restitution@h5": gap}, (v,), 1.0, 10, "fit-t")


@pytest.fixture
def lib(tmp_library: Path) -> CardLibrary:
    return CardLibrary([tmp_library])


def test_held_out_math():
    v = (_verdict("a@c", 0.02, 0.1), _verdict("b@c", 0.05, 0.1), _verdict("n@c", math.nan, 0.1, "not_run"))
    cal, valid, gap = held_out(v, {"a@c": 0.01, "b@c": None}, {"b@c": 0.03})
    assert valid == {"a@c": 0.02, "b@c": 0.05}
    assert cal == {"a@c": 0.01, "b@c": 0.03}                  # judge_rig's calib, else the search's error
    assert gap == pytest.approx({"a@c": 0.01, "b@c": 0.02})


def test_refuses_gap_above_half_bound(lib: CardLibrary):
    with pytest.raises(HeldOutGapError) as exc:
        write_back(_result(gap=0.016), lib, reason="test")     # 0.016 > 0.5 x 0.03
    assert exc.value.gaps == {"restitution@h5": (pytest.approx(0.016), 0.03)}
    assert lib.material("steel_carbon").surface.restitution.source != "fit"      # nothing written


def test_refuses_missing_or_nan_gap_and_no_evidence():
    r = _result(gap=0.0)
    nan_gap = FitResult(r.params, r.calib_error, r.valid_error, {"restitution@h5": math.nan}, r.verdicts, 1, 1, "f")
    assert gap_violations(nan_gap)
    no_gap = FitResult(r.params, r.calib_error, r.valid_error, {}, r.verdicts, 1, 1, "f")
    assert gap_violations(no_gap)
    empty = FitResult(r.params, {}, {}, {}, (), 1, 1, "f")
    with pytest.raises(HeldOutGapError, match="no judged"):
        check_heldout(empty)
    check_heldout(_result(gap=0.015))                          # exactly 0.5 x bound passes
    check_heldout(_result(gap=-0.01))                          # valid better than calib passes


def test_nan_validation_metric_is_refused():
    r = _result(gap=0.001)
    nan_v = _verdict("slip@b", math.nan, 0.01, "fail")       # compare_scalar: NaN sim mean -> fail, error NaN
    res = FitResult(r.params, {**r.calib_error, "slip@b": 0.0}, {**r.valid_error, "slip@b": math.nan},
                    {**r.heldout_gap, "slip@b": math.nan}, (*r.verdicts, nan_v), 1, 1, "f")
    with pytest.raises(HeldOutGapError) as exc:
        check_heldout(res)
    assert set(exc.value.gaps) == {"slip@b"}
    err = FitResult(r.params, r.calib_error, r.valid_error, r.heldout_gap,
                    (*r.verdicts, _verdict("x@b", 0.0, 0.01, "error")), 1, 1, "f")
    assert "x@b" in gap_violations(err)


def test_invalid_card_writes_nothing(lib: CardLibrary):
    pp = lib.material("pp_homopolymer").surface
    params = {"material:steel_carbon.surface.restitution": Q(0.66, "fit", note="fit-t"),
              "material:pp_homopolymer.surface.mu_dynamic": Q(pp.mu_static.value + 0.2, "fit", note="fit-t")}
    with pytest.raises(CardError):                            # mu_d > mu_s fails validation before any put
        write_back(_result(0.0, params=params), lib, reason="x")
    assert lib.material("steel_carbon").surface.restitution.source != "fit"


def test_writes_fitted_leaves_with_provenance(lib: CardLibrary):
    paths = write_back(_result(gap=0.001), lib, reason="R4 synthetic fit")
    q = lib.material("steel_carbon").surface.restitution
    assert (q.value, q.source, q.unc) == (0.66, "fit", (0.65, 0.67)) and q.note.startswith("fit-t")
    assert len(paths) == 1 and paths[0].name == "steel_carbon.yaml"
    hist = [json.loads(line) for line in (lib.roots[0] / "history.jsonl").read_text().splitlines()]
    assert "fit-t" in hist[-1]["reason"] and "R4 synthetic fit" in hist[-1]["reason"]


def test_pair_leaf_creates_the_pair_card(lib: CardLibrary):
    a, b = sorted(lib.names("materials"))[:2]
    assert lib.pair(a, b) is None
    leaf = f"pair:{a}__{b}.mu_static"
    write_back(_result(0.0, params={leaf: Q(0.5, "fit", (0.49, 0.51), note="fit-t")}), lib, reason="pair")
    card = lib.pair(a, b)
    assert card is not None and card.mu_static.value == 0.5 and card.mu_dynamic.source == "model"


def test_needs_reason_and_fitted_source(lib: CardLibrary):
    with pytest.raises(FitError):
        write_back(_result(0.0), lib, reason=" ")
    bad = {"material:steel_carbon.surface.restitution": Q(0.6, "measured", note="x")}
    with pytest.raises(FitError, match="fitted values only"):
        write_back(_result(0.0, params=bad), lib, reason="x")
