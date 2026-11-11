# SPDX-License-Identifier: Apache-2.0
"""Analytic law A12, Euler buckling P_cr = π²EI/(K·L)² (bench §4 A12, ± 5 %), on the CHAIN route.

A clamped-free column (K = 2) along +z, gravity off, carries a dead axial load f·P_E at its tip (force (0, 0, −P) at
the tip point plus its moment about the last link's COM, written to body_f every step). The column starts in a
stress-free first-mode imperfection w = δ·(1 − cos(πs/2L)), δ = 1e-4·L (the D6 joints' rest pose is the built shape).
Below P_cr the imperfection is amplified by 1/(1 − f) and oscillates; above it grows as exp(ω₀√(f−1)·t) into the
post-buckled elastica (tip angle α ≈ √(8(f−1)), so ≥ 0.1·L at f = 1.025). A world is unstable once its tip leaves
5 % of L. On the 2.5 % grid of f, P_cr is the midpoint between the largest stable and the smallest unstable f.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance
from alloy.route import rodref
from alloy.scene.analytic_rods import RodLawTwin, rod_card

GRID = (0.9, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.1)
OUT = 0.05                      # × L: tip excursion marking an unstable world
IMPERFECTION = 1e-4             # × L
K_CLAMPED_FREE = 2.0
L, RADIUS, DENSITY, EI = 0.2, 0.002, 1200.0, 1e-2     # ω₀ = 3.516·√(EI/(ρA·L⁴)) ≈ 72 rad/s, P_E = 0.617 N

LAW = LawSpec(
    id="A12.buckling", suite="analytic",
    metrics={"pcr_rel_err": Tolerance(abs=0.05, source="bench §4 A12: Euler buckling within 5 %")},
    stat={"pcr_rel_err": "mean"}, dt=1e-3, substeps=1, duration_s=1.5, worlds_per_condition=len(GRID),
    solvers=("mujoco",), production_dt=False, producer="WP2", quick={"worlds": len(GRID), "duration_s": 1.5},
)


@wp.kernel
def _tip_load(body_q: wp.array(dtype=wp.transform), bodies: wp.array(dtype=wp.int32),
              local: wp.array(dtype=wp.vec3), load: wp.array(dtype=float), body_f: wp.array(dtype=wp.spatial_vector)):
    """Dead load (0, 0, −P) at a body point: the force and its moment about the COM (Newton order: f, τ)."""
    w = wp.tid()
    b = bodies[w]
    xf = body_q[b]
    r = wp.quat_rotate(wp.transform_get_rotation(xf), local[w])
    f = wp.vec3(0.0, 0.0, -load[w])
    t = wp.cross(r, f)
    body_f[b] = wp.spatial_vector(f[0], f[1], f[2], t[0], t[1], t[2])


def euler_load() -> float:
    return rodref.euler_buckling_load(EI, L, K_CLAMPED_FREE)


def imperfect_column(n: int = 400) -> list[tuple[float, float, float]]:
    s = np.linspace(0.0, L, n)
    w = IMPERFECTION * L * (1.0 - np.cos(0.5 * math.pi * s / L))
    return [(float(a), 0.0, float(c)) for a, c in zip(w, s, strict=True)]


def critical_factor(f: np.ndarray, unstable: np.ndarray) -> float:
    """Midpoint between the largest stable and the smallest unstable load factor; NaN if not bracketed or
    non-monotone (an unstable world below a stable one)."""
    stable_f, unstable_f = f[~unstable], f[unstable]
    if stable_f.size == 0 or unstable_f.size == 0 or stable_f.max() > unstable_f.min():
        return math.nan
    return 0.5 * (float(stable_f.max()) + float(unstable_f.min()))


class BucklingTwin(RodLawTwin):
    law_id = LAW.id
    units = {"pcr_rel_err": "1"}
    dt = LAW.dt

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return (Condition("clamped_free_30", {"links": 30.0}),)

    def expand(self, cond: Condition, quick: bool, per: int) -> list[dict[str, float]]:
        return [{"factor": f} for f in GRID]

    def world_gravity(self, cond: Condition) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        rod = rod_card(L, EI, int(cond.params["links"]), radius=RADIUS, density=DENSITY)
        h = self.emit(b, rod, imperfect_column(), clamp=True)
        self.track([(h.bodies[-1], (0.5 * h.lengths[-1], 0.0, 0.0))])

    def on_model(self, model: newton.Model) -> None:
        super().on_model(model)
        loads = np.array([s["factor"] for s in self.samples]) * euler_load()
        self._load = wp.array(loads.astype(np.float32), dtype=float, device=model.device)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        wp.launch(_tip_load, dim=len(self.worlds), inputs=[state.body_q, self._pb[0], self._pl[0], self._load],
                  outputs=[state.body_f], device=self.model.device)

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        x, y, _ = self.point(log_, 0)
        excursion = np.nanmax(np.hypot(x, y), axis=1) / L
        f = np.array([s["factor"] for s in self.samples])
        unstable = excursion > OUT
        fc = critical_factor(f, unstable)
        disc = _chain_factor(int(self.worlds[0].params["links"]))
        self.monitors.update({"factor": f.tolist(), "max_excursion_over_L": excursion.tolist(),
                              "critical_factor": fc, "p_euler_n": euler_load(),
                              "chain_linear_factor": disc})
        return {"pcr_rel_err": np.full(len(f), abs(fc - 1.0))}


def _chain_factor(links: int) -> float:
    """Linear critical load of the clamped-free D7 chain / P_E (eigenvalue of K − P·K_g), a discretisation monitor."""
    lay = rodref.chain_layout(links, L, 1.0, EI)
    n = len(lay.lengths)
    k = lay.k
    K = np.zeros((n, n))
    for j in range(n):                                   # hinge j between link j−1 (or the clamp) and link j
        K[j, j] += k[j]
        if j > 0:
            K[j - 1, j - 1] += k[j]
            K[j - 1, j] -= k[j]
            K[j, j - 1] -= k[j]
    Kg = np.diag(lay.lengths)                            # dead tip load: V = −P·Σ l_i(1 − cos θ_i) ≈ −½P·Σ l_i θ_i²
    lam = np.linalg.eigvals(np.linalg.solve(Kg, K)).real
    return float(lam[lam > 0].min() / euler_load())


__all__ = ["LAW", "BucklingTwin", "critical_factor", "euler_load"]
