# SPDX-License-Identifier: Apache-2.0
"""Fit-twin hooks (duck-typed, like WP6's optional SimTwin hooks; SPEC §3.12).

A twin joins a fit by providing:
- ``fit_fields: frozenset[str]``: card leaf fields it can realise per world (a leaf's last path segment, e.g.
  ``mu_static``, ``restitution``);
- ``bind_values(cond_id, values)``: use these card values (field -> SI) for condition ``cond_id``; the fitter calls
  it once per virtual condition ``"<cond>~<k>"`` before building;
- ``fix_values(values)``: use these values for every condition (final validation); ``bind_calib`` must then not
  derive its own values from calib trials.
`FitValues` holds that state for a twin: per-condition values with the fixed values (or the twin's own defaults)
as the fallback.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from alloy.fit.errors import FitError


@dataclass
class FitValues:
    """Per-condition card values for one twin instance."""

    defaults: dict[str, float] = field(default_factory=dict)
    by_condition: dict[str, dict[str, float]] = field(default_factory=dict)
    fixed: dict[str, float] | None = None

    def bind(self, cond_id: str, values: Mapping[str, float]) -> None:
        self.by_condition[cond_id] = {k: float(v) for k, v in values.items()}

    def fix(self, values: Mapping[str, float]) -> None:
        self.fixed = {k: float(v) for k, v in values.items()}

    def for_condition(self, cond_id: str) -> dict[str, float]:
        """Defaults, then the fixed values, then this condition's bound values (later wins)."""
        out = dict(self.defaults)
        out.update(self.fixed or {})
        out.update(self.by_condition.get(cond_id, {}))
        return out

    def get(self, cond_id: str, name: str, *fallbacks: str) -> float:
        """Value of `name` for a condition, else the first present fallback field; FitError if none is set."""
        vals = self.for_condition(cond_id)
        for key in (name, *fallbacks):
            if key in vals:
                return vals[key]
        raise FitError(f"no value for {name!r} (or {fallbacks}) on condition {cond_id!r}; bind or fix it first")


__all__ = ["FitValues"]
