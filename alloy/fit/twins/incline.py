# SPDX-License-Identifier: Apache-2.0
"""Fit-ready R3 incline twin: WP6's `InclineTwin` with card values per (virtual) condition (alloy.fit.twins.base).

Each world realises one friction coefficient (Newton has one mu per contact). The tilt-grid and creep worlds realise
`mu_static` (onset and holding are static friction); the kinetic world at theta_s + 5 deg realises `mu_dynamic`
(sliding). Both leaves are therefore identifiable on single-mu solvers; with only one of them bound, every world uses
it (the solver's single mu, `PairCard.single_mu`). The grid is centred on atan(mu_static) of the world's own
condition, so the measured slip angle is atan(mu_static) + the solver's onset error. The kinetic world tilts to
atan(max(mu_static, mu_dynamic)) + 5 deg (the rig's theta_s + 5 deg for physical cards, mu_d <= mu_s), so it slides
for every candidate and the kinetic metric never goes flat in mu_dynamic.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.twins.incline import DENSITY, HALF, LAW, ROLE_KINETIC, InclineTwin
from alloy.bench.types import Condition, RigCard, RigTrial
from alloy.fit.twins.base import FitValues

MU_FIELDS = ("mu_static", "mu_dynamic")


class _MuStatic(Mapping[str, float]):
    """`InclineTwin.mu_fit` view: condition id -> the mu_static the condition's worlds use."""

    def __init__(self, values: FitValues) -> None:
        self._values = values

    def __getitem__(self, cond_id: str) -> float:
        return self._values.get(cond_id, "mu_static", "mu_dynamic")

    def __iter__(self):
        return iter(self._values.by_condition)

    def __len__(self) -> int:
        return len(self._values.by_condition)


class FitInclineTwin(InclineTwin):
    """R3 twin whose friction comes from bound card values instead of the closed form tan(calib slip angle)."""

    law_id = LAW.id
    fit_fields = frozenset(MU_FIELDS)

    def __init__(self, mu: float | None = None) -> None:
        super().__init__()
        self.values = FitValues({} if mu is None else {f: float(mu) for f in MU_FIELDS})
        self.mu_fit = _MuStatic(self.values)  # type: ignore[assignment]

    def bind_values(self, cond_id: str, values: Mapping[str, float]) -> None:
        self.values.bind(cond_id, values)

    def fix_values(self, values: Mapping[str, float]) -> None:
        self.values.fix(values)

    def bind_calib(self, card: RigCard, calib: Sequence[RigTrial]) -> None:
        """With fixed or default values the twin judges those values; otherwise WP6's closed-form calib fit."""
        if self.values.fixed is None and not self.values.defaults:
            super().bind_calib(card, calib)
            return
        self.monitors["mu_fixed"] = dict(self.values.fixed or self.values.defaults)

    def mu_for(self, cond_id: str, kinetic: bool) -> float:
        if kinetic:
            return self.values.get(cond_id, "mu_dynamic", "mu_static")
        return self.values.get(cond_id, "mu_static", "mu_dynamic")

    def initial_values(self, calib_means: Mapping[str, float]) -> dict[str, float]:
        """Closed-form start from calib means: mu_s = tan(slip angle), mu_d = mu_kinetic (WP6's bind_calib rule)."""
        out = {}
        if math.isfinite(calib_means.get("slip_angle", math.nan)):
            out["mu_static"] = math.tan(calib_means["slip_angle"])
        if math.isfinite(calib_means.get("mu_kinetic", math.nan)):
            out["mu_dynamic"] = calib_means["mu_kinetic"]
        return out

    def _theta(self) -> np.ndarray:
        th = super()._theta()
        for i, (cond, sample) in enumerate(zip(self.worlds, self.samples)):
            if sample.get("role") == ROLE_KINETIC:
                mu = max(self.mu_for(cond.id, kinetic=False), self.mu_for(cond.id, kinetic=True))
                th[i] = math.atan(mu) + math.radians(sample["offset_deg"])
        return th

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        mu = self.mu_for(cond.id, kinetic=sample.get("role") == ROLE_KINETIC)
        cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=DENSITY)
        b.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.0), width=0.0, length=0.0, cfg=cfg)  # per world: its own mu
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, HALF), wp.quat_identity()), label="box")
        b.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)

    def describe(self) -> dict[str, Any]:
        return {"twin": type(self).__name__, "fields": sorted(self.fit_fields), "defaults": self.values.defaults}


__all__ = ["FitInclineTwin", "MU_FIELDS"]
