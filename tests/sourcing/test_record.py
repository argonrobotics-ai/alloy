# SPDX-License-Identifier: Apache-2.0
"""Record rules: every value has a range and a tag; DOC/SNIPPET need a URL and a quote; ESTIMATE needs a reason."""

from __future__ import annotations

import pytest

from alloy.core.quantity import Q
from alloy.sourcing.fields import SourcingError, parse_value
from alloy.sourcing.record import Evidence, SourcedValue, note_for, value_from_dict, value_to_dict
from alloy.sourcing.tags import tag_of

DOC = Evidence(url="https://example.org/mug", quote="Item weight: 12 ounces", tag="DOC", where="specs table")


def test_doc_value_round_trips_and_maps_to_catalog() -> None:
    v = SourcedValue(field="mass", value=0.3402, lo=0.33, hi=0.35, tag="DOC", evidence=(DOC,),
                     derivation="12 oz x 0.02835 = 0.340 kg")
    assert value_from_dict(value_to_dict(v), "mass") == v
    q = v.to_q()
    assert (q.source, q.unc) == ("catalog", (0.33, 0.35))
    assert q.note.startswith('DOC: "Item weight: 12 ounces" (https://example.org/mug)')


@pytest.mark.parametrize("kwargs, match", [
    ({"tag": "DOC"}, "needs evidence"),
    ({"tag": "ESTIMATE"}, "needs the reason"),
    ({"tag": "ESTIMATE", "reason": "r", "lo": 0.3, "hi": 0.3, "value": 0.3}, "real range"),
    ({"tag": "DOC", "evidence": (Evidence(url="https://x", quote="q", tag="SNIPPET"),)}, "stronger than"),
    ({"tag": "DOC", "evidence": (Evidence(url="https://x", quote="", tag="DOC"),)}, "exact quote"),
    ({"tag": "DOC", "evidence": (Evidence(url="Callister App. B", quote="q", tag="DOC"),)}, "http"),
    ({"tag": "BEST"}, "unknown tag"),
    ({"tag": "ESTIMATE", "reason": "r", "value": 5.0, "hi": 6.0}, "outside"),
    ({"tag": "ESTIMATE", "reason": "r", "value": 0.1}, "lo <= value"),
])
def test_rules_reject_with_what_to_change(kwargs: dict, match: str) -> None:
    base = {"field": "mu_static", "value": 0.5, "lo": 0.3, "hi": 0.7}
    with pytest.raises(SourcingError, match=match):
        SourcedValue(**{**base, **kwargs})


def test_unverified_citation_without_url_scheme_is_allowed() -> None:
    ev = Evidence(url="Callister & Rethwisch 9e, App. B.1", quote="", tag="UNVERIFIED")
    v = SourcedValue(field="density", value=1040.0, lo=1020.0, hi=1060.0, tag="UNVERIFIED", evidence=(ev,))
    assert v.to_q().source == "catalog"


def test_estimate_maps_to_model_and_note_carries_reason() -> None:
    v = SourcedValue(field="restitution", value=0.5, lo=0.3, hi=0.7, tag="ESTIMATE", evidence=(DOC,),
                     reason="ceramic on wood, proxy")
    assert v.to_q().source == "model"
    assert "proxy" in note_for(v) and "example.org" in note_for(v)


def test_unit_strings_parse_to_si() -> None:
    assert parse_value("1 lb", "mass") == pytest.approx(0.45359237)
    assert parse_value("12 oz", "mass") == pytest.approx(0.340194, rel=1e-5)
    assert parse_value("1.2 g/cm^3", "density") == pytest.approx(1200.0)
    assert parse_value("2.1 GPa", "youngs") == pytest.approx(2.1e9)
    with pytest.raises(SourcingError, match="unknown field"):
        parse_value(1.0, "colour")


def test_unknown_keys_are_errors() -> None:
    with pytest.raises(SourcingError, match="unknown keys"):
        value_from_dict({"value": 1, "lo": 1, "hi": 1, "tag": "DOC", "source": "x"}, "mass")


@pytest.mark.parametrize("q, tag", [
    (None, "MISSING"),
    (Q(0.5, "default"), "MISSING"),
    (Q(0.5, "model", note="tilt test"), "ESTIMATE"),
    (Q(0.5, "catalog", note="22 g ESTIMATE"), "ESTIMATE"),
    (Q(0.5, "catalog", note="JEDEC 1.27 mm [SNIPPET]"), "SNIPPET"),
    (Q(0.5, "catalog", note="Serway Table 5.1"), "UNVERIFIED"),
    (Q(0.5, "measured", note="YCB https://www.ycbbenchmarks.com/object-set/"), "DOC"),
    (Q(0.5, "catalog", note="doi:10.1016/j.wear.2013.01.001"), "DOC"),
])
def test_tag_of_card_quantities(q: Q | None, tag: str) -> None:
    assert tag_of(q) == tag


def test_card_paths_are_fields() -> None:
    v = value_from_dict({"value": 0.0012, "lo": 0.001, "hi": 0.0015, "tag": "ESTIMATE", "reason": "drawing"},
                        "features.min_wall")
    assert v.field == "features.min_wall"
    assert value_from_dict({"value": 0.4, "lo": 0.3, "hi": 0.5, "tag": "ESTIMATE", "reason": "r"},
                           "surface.mu_static").field == "mu_static"
    with pytest.raises(SourcingError, match="card path"):
        parse_value(1.0, "colour.red")
