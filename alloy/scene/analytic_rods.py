# SPDX-License-Identifier: Apache-2.0
"""Analytic laws A11 (catenary sag, hanging-chain ω₁, elastica cantilever) and A12 (Euler buckling) on the CHAIN
route (bench §4 A11/A12, SPEC §5.3: producer WP2). Every twin builds its rod with the CHAIN emitter itself
(`alloy.scene.emitters.chain.build_chain`: D7 half-length ends, D6 joints k = EI/l and GJ/l through
`mujoco:dof_passive_stiffness`, implicitfast armature), so the laws judge exactly what the router emits.

- A11.hanging_chain: a chain pinned at the top (ball joint), started in the continuum first-mode shape
  u ∝ J₀(2.4048·√(z/L)) (z up from the free end); ω₁ from tip zero crossings vs 1.2024·√(g/L), 2 % at ≥ 40 links.
  Bending is made negligible (γ = 10⁶: bending shifts ω₁² by ~β⁴/γ ≈ 1e-5).
- A11.catenary: a chain of length L between two level supports S apart (ball joint at A, MuJoCo connect equality at
  B, started as a V of the same length) settles with joint damping; mid-span sag below the chord of the realised
  supports vs the inextensible catenary for the realised span, 1 %.
- A11.cantilever: the clamped self-weight cantilever settled with the self-test damping (τ = 1.4/ω₁); tip drop vs the
  exact large-deflection solution of the same chain (`rodref.static_tip_drop`, any γ), 1 %. The gap to the continuum
  elastica (the discretisation error) is a monitor.
- A12.buckling (alloy.scene.analytic_buckling).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import warp as wp
from scipy.special import j0

from alloy._vendor import newton

from alloy.bench.analytic.common import BaseTwin
from alloy.bench.solvers import SolverSettings
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance
from alloy.route import rodref
from alloy.route.types import RodCard
from alloy.scene.emitters.chain import ChainHandles, ChainOptions, build_chain, first_mode

log = logging.getLogger("alloy.scene")

G = rodref.G
H0 = 1.0                        # m, support / clamp height
RADIUS, DENSITY = 0.002, 1200.0
LIN = DENSITY * math.pi * RADIUS**2
J0_ZERO = 2.404825557695773
SETTLE_TAU = 1.4                # × 1/ω₁: the rod self-test's static-settle damping time (alloy/route/rodcheck.py)
_SRC = "bench §4 A11"
Point = tuple[int, tuple[float, float, float]]


def rod_card(L: float, EI: float, links: int, *, radius: float = RADIUS, density: float = DENSITY,
             damping_ratio: float = 0.0, integrator: str = "implicitfast") -> RodCard:
    """A RodCard for a law (GJ = EI/1.3, an isotropic ν = 0.3 rod; EA is unused by the chain)."""
    lin = density * math.pi * radius**2
    return RodCard(length=L, radius=radius, density=density, EI=EI, GJ=EI / 1.3, EA=1e9, links=links,
                   damping_ratio=damping_ratio, gamma=lin * G * L**3 / EI, integrator=integrator)


def ei_for_gamma(L: float, gamma: float, lin: float = LIN) -> float:
    return lin * G * L**3 / gamma


def mujoco_settings(integrator: str = "implicitfast") -> dict[str, SolverSettings]:
    return {"mujoco": SolverSettings("mujoco", {"integrator": integrator, "solver": "newton", "iterations": 50,
                                                "disable_contacts": True})}


@wp.kernel
def _rec_point(body_q: wp.array(dtype=wp.transform), bodies: wp.array(dtype=wp.int32),
               local: wp.array(dtype=wp.vec3), col: int, x: wp.array2d(dtype=float), y: wp.array2d(dtype=float),
               z: wp.array2d(dtype=float)):
    w = wp.tid()
    p = wp.transform_point(body_q[bodies[w]], local[w])
    x[w, col] = p[0]
    y[w, col] = p[1]
    z[w, col] = p[2]


def torsion_case(rod: Mapping[str, float], torque_nm: float, device: str = "cpu") -> dict[str, Any]:
    """XB4-10's rod-twist hook at the path RERUN.md names; runs on the ROD_VBD route
    (`alloy.scene.analytic_rods_vbd.torsion_case`)."""
    from alloy.scene.analytic_rods_vbd import torsion_case as on_rod_vbd

    return on_rod_vbd(rod, torque_nm, device)


def crossing_period(t: np.ndarray, x: np.ndarray) -> float:
    """Mean period from all zero crossings (2 × mean spacing, linear interpolation); NaN with fewer than 3."""
    s = np.signbit(x)
    idx = np.flatnonzero(s[1:] != s[:-1])
    if idx.size < 3:
        return math.nan
    tc = t[idx] - x[idx] * (t[idx + 1] - t[idx]) / (x[idx + 1] - x[idx])
    return float(2.0 * (tc[-1] - tc[0]) / (tc.size - 1))


class RodLawTwin(BaseTwin):
    """Shared CHAIN twin plumbing: chains built with the CHAIN emitter, K tracked points (body + local offset) per
    world recorded into channels p<k>x|y|z."""

    needs_contacts = False
    settings = mujoco_settings()
    dt = 1.25e-3

    def __init__(self) -> None:
        super().__init__()
        self.chains: list[ChainHandles] = []
        self.rods: list[RodCard] = []
        self._points: list[list[Point]] = []

    def emit(self, b: newton.ModelBuilder, rod: RodCard, points, **opts: Any) -> ChainHandles:
        opts.setdefault("collide", False)
        h = build_chain(b, rod, points, ChainOptions(capability="mujoco", dt=self.dt, **opts),
                        label=f"law{len(self.chains)}")
        self.chains.append(h)
        self.rods.append(rod)
        return h

    def track(self, points: Sequence[Point]) -> None:
        """The points recorded for the world being built (same count in every world)."""
        self._points.append([(int(b_), tuple(float(v) for v in loc)) for b_, loc in points])

    def on_model(self, model: newton.Model) -> None:
        dev = model.device
        k = len(self._points[0]) if self._points else 0
        self._pb = [wp.array([w[i][0] for w in self._points], dtype=wp.int32, device=dev) for i in range(k)]
        self._pl = [wp.array([wp.vec3(*w[i][1]) for w in self._points], dtype=wp.vec3, device=dev) for i in range(k)]

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log_: DeviceTwinLog) -> None:
        col = log_.column(step)
        if col < 0:
            return
        for i, (bodies, local) in enumerate(zip(self._pb, self._pl, strict=True)):
            outs = [log_.channel(f"p{i}{a}") for a in "xyz"]
            wp.launch(_rec_point, dim=len(self._points), inputs=[state.body_q, bodies, local, col], outputs=outs,
                      device=self.model.device)

    @staticmethod
    def point(log_: DeviceTwinLog, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, z = (log_.series(f"p{i}{a}").astype(np.float64) for a in "xyz")
        return x, y, z


# --- A11.hanging_chain ---------------------------------------------------------------------------------------------

LAW_HANGING = LawSpec(
    id="A11.hanging_chain", suite="analytic",
    metrics={"omega1_rel_err": Tolerance(abs=0.02, source=f"{_SRC}: hanging chain ω₁ within 2 % at ≥ 40 segments")},
    stat={"omega1_rel_err": "mean"}, dt=2e-3, substeps=1, duration_s=4.5, worlds_per_condition=1,
    solvers=("mujoco",), production_dt=False, producer="WP2", quick={"worlds": 2, "duration_s": 4.5},
)


def mode_points(L: float, amp: float, n: int = 400) -> list[tuple[float, float, float]]:
    """Centre line of the continuum first hanging mode, x = amp·J₀(2.4048·√((L−s)/L)), hanging from (0, 0, H0)."""
    s = np.linspace(0.0, L, n)
    x = amp * j0(J0_ZERO * np.sqrt(np.clip((L - s) / L, 0.0, None)))
    return [(float(a), 0.0, float(H0 - c)) for a, c in zip(x, s, strict=True)]


class HangingChainTwin(RodLawTwin):
    law_id = LAW_HANGING.id
    units = {"omega1_rel_err": "1"}
    dt = LAW_HANGING.dt
    L = 0.5

    LINKS = 40                  # SolverMuJoCo needs homogeneous worlds: one link count per run

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"L{int(L * 100)}cm", {"length": L}) for L in ((0.5, 0.8) if quick else (0.3, 0.5, 0.8)))

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        L = cond.params["length"]
        h = self.emit(b, rod_card(L, ei_for_gamma(L, 1e6), self.LINKS), mode_points(L, 0.01 * L), pin=True)
        self.track([(h.bodies[-1], (0.5 * h.lengths[-1], 0.0, 0.0))])

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        t = log_.times()
        x, _, _ = self.point(log_, 0)
        ref = np.array([rodref.hanging_chain_w1(L) for L in self.param("length")])
        w = np.array([2.0 * math.pi / crossing_period(t, x[i] - np.mean(x[i])) for i in range(len(self.worlds))])
        self.monitors.update({"omega1_sim": w.tolist(), "omega1_ref": ref.tolist()})
        return {"omega1_rel_err": np.abs(w / ref - 1.0)}


# --- A11.catenary --------------------------------------------------------------------------------------------------

LAW_CATENARY = LawSpec(
    id="A11.catenary", suite="analytic",
    metrics={"sag_rel_err": Tolerance(abs=0.01, source=f"{_SRC}: catenary sag within 1 %")},
    stat={"sag_rel_err": "mean"}, dt=2e-3, substeps=1, duration_s=8.0, worlds_per_condition=1,
    solvers=("mujoco",), production_dt=False, producer="WP2", quick={"worlds": 2, "duration_s": 8.0},
)


class CatenaryTwin(RodLawTwin):
    law_id = LAW_CATENARY.id
    units = {"sag_rel_err": "1"}
    dt = LAW_CATENARY.dt
    L = 1.0
    # N·m·s/rad per joint dof. Measured (span 0.8/0.6, 40 links, dt 2 ms): 2e-5 still swings at 8 s (−2.4 %/+1.0 %),
    # 2e-4 settles by 6 s (+0.09 %/+0.16 %), 2e-3 is overdamped (+0.26 %/+0.80 % still creeping at 12 s)
    SETTLE_DAMPING = 2e-4

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        spans = (0.8, 0.6) if quick else (0.9, 0.8, 0.6)
        return tuple(Condition(f"span{int(s * 100)}", {"span": s, "links": 40.0}) for s in spans)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        from alloy._vendor.newton.solvers import mujoco as nmj

        S, L = cond.params["span"], self.L
        d = math.sqrt(max(0.25 * L * L - 0.25 * S * S, 0.0))
        q0 = b.joint_dof_count
        h = self.emit(b, rod_card(L, ei_for_gamma(L, 1e6), int(cond.params["links"])),
                      [(-0.5 * S, 0.0, H0), (0.0, 0.0, H0 - d), (0.5 * S, 0.0, H0)], pin=True)
        for dof in range(q0, b.joint_dof_count):
            b.joint_damping[dof] = self.SETTLE_DAMPING      # armature: the route's CHAIN floor (emitters.chain)
        end = (0.5 * h.lengths[-1], 0.0, 0.0)
        nmj.add_equality_connect(b, h.bodies[-1], -1, anchor=wp.vec3(*end), solref=(2.0 * self.dt, 1.0))
        mid = len(h.bodies) // 2 - 1                       # node L/2 = outer end of link N/2 − 1 (N even)
        self.track([(h.bodies[mid], (0.5 * h.lengths[mid], 0.0, 0.0)), (h.bodies[-1], end)])

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        _, _, mid_z = self.point(log_, 0)
        end_x, _, end_z = self.point(log_, 1)
        # The reference is the law's own geometry: supports at +-S/2 and H0 (audit AB-laws-2: computed at the
        # realised span and far-support height it followed a far support the emitter or the equality put elsewhere).
        span = np.abs(end_x[:, -1] + 0.5 * self.param("span"))      # realised: monitors below
        sag = H0 - mid_z[:, -1]
        ref = np.array([rodref.catenary_sag(s, self.L) for s in self.param("span")])
        self.monitors.update({"sag_sim_m": sag.tolist(), "sag_ref_m": ref.tolist(), "span_realised_m": span.tolist(),
                              "support_drop_m": (H0 - end_z[:, -1]).tolist(),
                              "sag_drift_last_s_m": (mid_z[:, -1] - mid_z[:, -1 - _last_second(log_)]).tolist()})
        return {"sag_rel_err": np.abs(sag / ref - 1.0)}


def _last_second(log_: DeviceTwinLog) -> int:
    return max(1, int(round(1.0 / (log_.stride * log_.dt))))


# --- A11.cantilever ------------------------------------------------------------------------------------------------

LAW_CANTILEVER = LawSpec(
    id="A11.cantilever", suite="analytic",
    metrics={"tip_drop_rel_err": Tolerance(abs=0.01, source=f"{_SRC}: elastica cantilever tip drop within the "
                                                            "discretisation error (vs the exact chain, 1 %)")},
    stat={"tip_drop_rel_err": "mean"}, dt=1.25e-3, substeps=1, duration_s=5.5, worlds_per_condition=1,
    solvers=("mujoco",), production_dt=True, producer="WP2", quick={"worlds": 2, "duration_s": 5.5},
)


class CantileverTwin(RodLawTwin):
    law_id = LAW_CANTILEVER.id
    units = {"tip_drop_rel_err": "1"}
    dt = LAW_CANTILEVER.dt
    L = 0.5

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"gamma{g:g}", {"gamma": g, "links": 30.0}) for g in (0.5, 5.0))

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        rod = rod_card(self.L, ei_for_gamma(self.L, cond.params["gamma"]), int(cond.params["links"]))
        rod = rod_card(rod.length, rod.EI, rod.links, damping_ratio=0.5 * SETTLE_TAU)   # ζ·2/ω₁ = τ = 1.4/ω₁
        h = self.emit(b, rod, [(0.0, 0.0, H0), (self.L, 0.0, H0)], clamp=True)
        self.track([(h.bodies[-1], (0.5 * h.lengths[-1], 0.0, 0.0))])

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        _, _, z = self.point(log_, 0)
        drop = H0 - z[:, -1]
        disc = np.array([rodref.static_tip_drop(rodref.chain_layout(r.links, r.length, LIN, r.EI, r.radius))
                         for r in self.rods])
        cont = np.array([rodref.elastica_tip_drop(r.length, LIN, r.EI) for r in self.rods])
        self.monitors.update({"tip_drop_m": drop.tolist(), "chain_exact_m": disc.tolist(),
                              "elastica_m": cont.tolist(), "discretisation_err": (disc / cont - 1.0).tolist(),
                              "omega1": [first_mode(r) for r in self.rods]})
        return {"tip_drop_rel_err": np.abs(drop / disc - 1.0)}


__all__ = ["LAW_HANGING", "LAW_CATENARY", "LAW_CANTILEVER", "HangingChainTwin", "CatenaryTwin", "CantileverTwin",
           "RodLawTwin", "rod_card", "ei_for_gamma", "crossing_period", "mode_points"]
