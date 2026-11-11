# SPDX-License-Identifier: Apache-2.0
"""Sourced SI quantity ``Q`` and the ``Vec3`` alias (SPEC §3.1, §4.1)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal, get_args

from alloy.core.errors import QuantityError

Vec3 = tuple[float, float, float]
Source = Literal["measured", "catalog", "fit", "model", "default"]
SOURCES: Final[frozenset[str]] = frozenset(get_args(Source))


@dataclass(frozen=True, slots=True)
class Q:
    """A value in canonical SI (float64 on host) with its provenance.

    ``note`` (citation or derivation) is required unless ``source == "default"``; ``unc`` is an absolute
    (lo, hi) interval in SI. Raises ``QuantityError`` on a non-finite value, an unknown source, a missing note or
    ``lo > hi``.
    """

    value: float
    source: Source
    unc: tuple[float, float] | None = None
    note: str = ""

    def __post_init__(self) -> None:
        _check_value(self.value)
        object.__setattr__(self, "value", float(self.value))
        if self.source not in SOURCES:
            raise QuantityError(f"unknown source {self.source!r}; expected one of {sorted(SOURCES)}")
        if self.source != "default" and not self.note.strip():
            raise QuantityError(f"Q({self.value}, {self.source!r}) needs a note (citation or derivation)")
        if self.unc is not None:
            object.__setattr__(self, "unc", _check_unc(self.unc))

    @property
    def is_default(self) -> bool:
        """True when the value is a library default (listed in ``ResolvedObject.default_leaves``)."""
        return self.source == "default"

    def with_value(self, value: float) -> Q:
        """Same provenance, new value (used by worst-case resolution and randomization)."""
        return Q(value, self.source, self.unc, self.note)


def _check_value(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuantityError(f"Q value must be a real number, got {type(value).__name__}")
    if not math.isfinite(value):
        raise QuantityError(f"Q value must be finite, got {value}")


def _check_unc(unc: tuple[float, float]) -> tuple[float, float]:
    if len(unc) != 2:
        raise QuantityError(f"unc must be (lo, hi), got {unc!r}")
    lo, hi = float(unc[0]), float(unc[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo > hi:
        raise QuantityError(f"unc must be finite with lo <= hi, got {unc!r}")
    return (lo, hi)
