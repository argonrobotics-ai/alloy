# SPDX-License-Identifier: Apache-2.0
"""R3 incline-friction twin (bench §4.4 R3): slip angle (+/- 0.5 deg), kinetic mu (5 %), creep (<= 0.01 mm / 10 s).

The rig tilts slowly until the box slips > 0.5 mm, runs kinetic trials at theta_s + 5 deg and creep holds at
theta_s - 3 deg. The twin never steps the tilt: each world gets gravity as an eased body force (as in A5), on a
0.1 deg grid of tilts around the onset fitted from the `calib` trials (mu = tan(mean calib slip angle)), plus one
kinetic world and one creep world per condition. Values are fitted on calib only; `valid` judges (rigjudge).
Newton has one mu per contact, so the twin's kinetic and static mu are the same fitted value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench.analytic.coulomb import SLIP_M, T_EASE, _apply_tilted_gravity, onset_from_grid
from alloy.bench.analytic.common import G, BaseTwin, mujoco_law_settings, smoothstep
from alloy.bench.solvers import RIGID_SOLVERS
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, RigCard, RigTrial, Tolerance

HALF = 0.025  # [m]
DENSITY = 1000.0
ROLE_GRID, ROLE_KINETIC, ROLE_CREEP = 0, 1, 2
_SRC = "bench §4.4 R3"

LAW = LawSpec(
    id="R3.incline_friction", suite="rig",
    metrics={"slip_angle": Tolerance(abs=math.radians(0.5), source=f"{_SRC}: slip angle +/- 0.5 deg"),
             "mu_kinetic": Tolerance(rel=0.05, source=f"{_SRC}: mu_k within 5 %"),
             "creep": Tolerance(abs=1e-5, source=f"{_SRC}: creep <= max(real, 0.01 mm) per 10 s")},
    stat={"slip_angle": "mean", "mu_kinetic": "mean", "creep": "mean"},
    dt=1.25e-3, substeps=1, duration_s=2.0, worlds_per_condition=10, solvers=RIGID_SOLVERS,
)


class InclineTwin(BaseTwin):
    law_id = LAW.id
    units = {"slip_angle": "rad", "mu_kinetic": "1", "creep": "m"}
    settings = {"mujoco": mujoco_law_settings()}

    def __init__(self) -> None:
        super().__init__()
        self.mu_fit: dict[str, float] = {}

    def bind_calib(self, card: RigCard, calib: Sequence[RigTrial]) -> None:
        """mu per condition = tan(mean calib slip angle): the only data the twin sees."""
        by: dict[str, list[float]] = {}
        for t in calib:
            by.setdefault(t.condition_id, []).append(t.scalars["slip_angle"])
        self.mu_fit = {c: math.tan(float(np.mean(v))) for c, v in by.items()}
        self.monitors["mu_fit_calib"] = dict(self.mu_fit)

    def expand(self, cond: Condition, quick: bool, per: int) -> list[dict[str, float]]:
        n_grid = max(2, per - 2)
        grid = [{"role": ROLE_GRID, "offset_deg": 0.1 * (k - (n_grid - 1) / 2)} for k in range(n_grid)]
        return grid + [{"role": ROLE_KINETIC, "offset_deg": 5.0}, {"role": ROLE_CREEP, "offset_deg": -3.0}]

    def world_gravity(self, cond: Condition) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        cfg = newton.ModelBuilder.ShapeConfig(mu=self.mu_fit[cond.id], density=DENSITY)
        b.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.0), width=0.0, length=0.0, cfg=cfg)  # per-world: same mu
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, HALF), wp.quat_identity()), label="box")
        b.add_shape_box(body, hx=HALF, hy=HALF, hz=HALF, cfg=cfg)

    def _theta(self) -> np.ndarray:
        mu = np.array([self.mu_fit[c.id] for c in self.worlds])
        off = np.array([s["offset_deg"] for s in self.samples])
        return np.arctan(mu) + np.radians(off)

    def on_model(self, model: newton.Model) -> None:
        th = self._theta()
        d = np.stack([np.sin(th), np.zeros_like(th), -np.cos(th)], 1) * G
        self._dir = wp.array(d.astype(np.float32), dtype=wp.vec3, device=model.device)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        wp.launch(_apply_tilted_gravity, dim=len(self.worlds),
                  inputs=[self._first_body, self.model.body_mass, self._dir, smoothstep(t / T_EASE)],
                  outputs=[state.body_f], device=self.model.device)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        self.record_first_body(step, state, log)

    def _per_condition(self, cid: str, t: np.ndarray, s: np.ndarray, th: np.ndarray) -> dict[str, float]:
        idx = [i for i, c in enumerate(self.worlds) if c.id == cid]
        role = np.array([self.samples[i]["role"] for i in idx])
        off = np.array([self.samples[i]["offset_deg"] for i in idx])
        i0, i_half = int(np.searchsorted(t, T_EASE)), int(np.searchsorted(t, 0.5 * (T_EASE + t[-1])))
        hold = s[idx, -1] - s[idx, i0]
        grid = role == ROLE_GRID
        onset = onset_from_grid(off[grid], hold[grid] > SLIP_M)
        k = idx[int(np.flatnonzero(role == ROLE_KINETIC)[0])]
        win = t >= T_EASE + 0.05
        a = 2.0 * np.polyfit(t[win], s[k, win], 2)[0]
        c = idx[int(np.flatnonzero(role == ROLE_CREEP)[0])]
        creep = abs(s[c, -1] - s[c, i_half]) / max(t[-1] - t[i_half], 1e-9) * 10.0
        return {"slip_angle": math.atan(self.mu_fit[cid]) + math.radians(onset) if math.isfinite(onset) else math.nan,
                "mu_kinetic": math.tan(th[k]) - a / (G * math.cos(th[k])), "creep": creep}

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        t = log.times()
        xs = log.series("x").astype(np.float64)
        s = xs - xs[:, :1]
        th = self._theta()
        per = {cid: self._per_condition(cid, t, s, th) for cid in dict.fromkeys(c.id for c in self.worlds)}
        self.monitors["per_condition"] = per
        return {m: np.array([per[c.id][m] for c in self.worlds]) for m in ("slip_angle", "mu_kinetic", "creep")}


__all__ = ["LAW", "InclineTwin"]
