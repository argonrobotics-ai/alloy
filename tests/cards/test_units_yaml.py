# SPDX-License-Identifier: Apache-2.0
"""YAML I/O: YAML 1.2 scalars, strict keys, exact float round trip, unit kinds of every numeric card field."""

from __future__ import annotations

import dataclasses
import math

import pytest
import yaml

from alloy.cards import types as ct
from alloy.cards.paths import non_none, type_hints, union_args
from alloy.cards.units_yaml import LEAF_KINDS, FlowList, kind_of, parse_number, yaml_dump, yaml_load
from alloy.core.errors import CardError
from alloy.core.quantity import Q


def test_yaml12_scalars() -> None:
    data = yaml_load("a: 1e-4\nb: on\nc: yes\nd: 2026-11-01\ne: 7850\nf: .inf\ng: true\nh: ~\ni: 0x1F\n", "t")
    assert data == {"a": 1e-4, "b": "on", "c": "yes", "d": "2026-11-01", "e": 7850, "f": math.inf, "g": True,
                    "h": None, "i": "0x1F"}
    assert isinstance(data["a"], float) and isinstance(data["e"], int)


def test_duplicate_and_non_string_keys() -> None:
    with pytest.raises(CardError) as exc:
        yaml_load("a: 1\nb: 2\na: 3\n", "dup")
    assert exc.value.field == "a" and "line 3" in exc.value.message
    with pytest.raises(CardError):
        yaml_load("1: x\n", "intkey")
    with pytest.raises(CardError) as exc:
        yaml_load("a: [1, 2\n", "syntax")
    assert exc.value.field == "<yaml>"


@pytest.mark.parametrize("value", [1e-05, 2e6, 0.1, 1 / 3, -0.0, 5e-324, 1.2345678901234568e16, 997.05, -1.5e-300])
def test_float_round_trip_bitwise(value: float) -> None:
    back = yaml_load(yaml_dump({"x": value}), "t")["x"]
    assert back == value and math.copysign(1.0, back) == math.copysign(1.0, value) and repr(back) == repr(value)


@pytest.mark.parametrize("text", ["on", "yes", "No", "1e-4", "2026-11-01", "~", "null", "true", "0x10", "1_000", "", "#x",
                                  "a: b", "- x", "multi\nline"])
def test_ambiguous_strings_round_trip_both_readers(text: str) -> None:
    dumped = yaml_dump({"s": text})
    assert yaml_load(dumped, "t")["s"] == text
    assert yaml.safe_load(dumped)["s"] == text          # stock YAML 1.1 readers see the same string


def test_flow_lists() -> None:
    assert yaml_dump({"p": FlowList([0.0, 1.5])}).strip() == "p: [0.0, 1.5]"


def test_parse_number() -> None:
    assert parse_number("0.2 mm", "length", "c", "f") == pytest.approx(2e-4)
    assert parse_number(3, "length", "c", "f") == 3.0 and isinstance(parse_number(3, "length", "c", "f"), float)
    assert parse_number("5 N", ("force", "torque"), "c", "f") == 5.0
    assert parse_number("5 N·m", ("force", "torque"), "c", "f") == 5.0
    for raw, kind in ((True, "length"), ("2 kg", "length"), ("1 mm", None), ([1], "length")):
        with pytest.raises(CardError) as exc:
            parse_number(raw, kind, "c", "f.g")
        assert exc.value.field == "f.g" and exc.value.card == "c"


def _numeric_fields(cls: type) -> list[str]:
    out = []
    for f in dataclasses.fields(cls):
        tp = non_none(type_hints(cls)[f.name])
        members = union_args(tp) or (tp,)
        if any(m is Q or m is float or (getattr(m, "__origin__", None) is tuple and "float" in repr(m)) for m in members):
            out.append(f.name)
    return out


CARD_CLASSES = [ct.Surface, ct.ClothAniso, ct.Elastic, ct.Plastic, ct.Fluid, ct.Failure, ct.Film, ct.MaterialCard, ct.PairCard,
                ct.Primitive, ct.MeshSource, ct.SDFSource, ct.TetSource, ct.CurveSource, ct.ParticleSource, ct.LoadEnvelope,
                ct.Crease, ct.Features, ct.BladeSpec, ct.PrecisionFeature, ct.Placement, ct.RobotRef, ct.Attachment,
                ct.EmitterSpec, ct.SceneCard]
UNITLESS = {"Randomization.lo", "Randomization.hi",          # kind of the target leaf (codec)
            "Features.bend_stiffness", "Features.twist_stiffness"}   # N·m² has no UnitKind: SI numbers only


@pytest.mark.parametrize("cls", CARD_CLASSES, ids=lambda c: c.__name__)
def test_every_numeric_field_has_a_unit_kind(cls: type) -> None:
    missing = [f for f in _numeric_fields(cls) if kind_of(cls.__name__, f) is None and f"{cls.__name__}.{f}" not in UNITLESS]
    assert missing == []


def test_leaf_kind_table_names_real_fields() -> None:
    classes = {c.__name__: c for c in CARD_CLASSES}
    for key in LEAF_KINDS:
        owner, field = key.split(".")
        assert field in {f.name for f in dataclasses.fields(classes[owner])}, key
