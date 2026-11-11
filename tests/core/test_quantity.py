# SPDX-License-Identifier: Apache-2.0
"""Q: sourced SI values and the note rule (SPEC §3.1)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from alloy.core.errors import AlloyError, QuantityError
from alloy.core.quantity import SOURCES, Q


def test_default_needs_no_note() -> None:
    q = Q(20.0, "default")
    assert q.is_default and q.note == "" and q.unc is None


@pytest.mark.parametrize("source", sorted(SOURCES - {"default"}))
def test_non_default_requires_note(source: str) -> None:
    with pytest.raises(QuantityError):
        Q(1.0, source)  # type: ignore[arg-type]
    with pytest.raises(QuantityError):
        Q(1.0, source, note="   ")  # type: ignore[arg-type]
    assert Q(1.0, source, note="catalog p. 3").note == "catalog p. 3"  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "1.0", None, True])
def test_value_must_be_finite_real(value: object) -> None:
    with pytest.raises(QuantityError):
        Q(value, "default")  # type: ignore[arg-type]


def test_int_value_is_stored_as_float() -> None:
    q = Q(2, "default")
    assert isinstance(q.value, float) and q.value == 2.0


def test_unknown_source_rejected() -> None:
    with pytest.raises(QuantityError):
        Q(1.0, "guess", note="x")  # type: ignore[arg-type]


def test_unc_interval_rules() -> None:
    assert Q(1.0, "measured", (0.9, 1.1), "rig R3").unc == (0.9, 1.1)
    for bad in [(1.1, 0.9), (0.0, math.nan), (1.0,)]:
        with pytest.raises(QuantityError):
            Q(1.0, "measured", bad, "rig R3")  # type: ignore[arg-type]


def test_frozen_hashable_and_with_value() -> None:
    q = Q(0.5, "catalog", note="Engineering Toolbox")
    with pytest.raises(dataclasses.FrozenInstanceError):
        q.value = 1.0  # type: ignore[misc]
    assert hash(q) == hash(Q(0.5, "catalog", note="Engineering Toolbox"))
    q2 = q.with_value(0.7)
    assert q2.value == 0.7 and q2.source == "catalog" and q2.note == q.note


def test_quantity_error_hierarchy() -> None:
    assert issubclass(QuantityError, AlloyError) and issubclass(QuantityError, ValueError)
