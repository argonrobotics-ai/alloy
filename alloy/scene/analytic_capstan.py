# SPDX-License-Identifier: Apache-2.0
"""Analytic law A10, capstan T_out/T_in = exp(μθ) (bench §4 A10, ± 2 %), on the CHAIN route as the router emits it.

A limp rope (radius 1 mm, 7,850 kg/m³, EI 3e-6 N·m²) is wrapped over the top half of a fixed horizontal capsule drum
(R = 0.2 m, θ = π), gravity off. Both ends carry dead loads along −z at their outer end points: T_in = 1 N on link 0
and T_out = r·T_in·exp(μθ) on the last link, eased in over 50 ms at a fixed ratio. The measured threshold is the
midpoint between the largest holding and the smallest slipping r; its error is |r_c − 1|.

The rope is the default route's (WP2 A10 diagnosis, docs/notes/wp2-results.md "A10"):
- links: the router's CHAIN rule (`alloy.route.checks.chain_links` on the rope's bending length: 34 links), laid as
  a polygon with chords `GAP_M` off the drum and hinges on both leg/drum junctions (turn α/2 there), legs of
  `LEG_LINKS` + ½ links. The old 80-link rope had 2.3 mm links on a 2 mm radius, a bead chain the router never emits
  (segments are never shorter than a diameter), and slipped at 0.875 / 0.65 of exp(μθ) (μ 0.2 / 0.4).
- contacts: MuJoCo's own (capsule–capsule, elliptic, condim 3), with gains as the assembler writes them: links
  `native_gains(K, m, 0, dt)` (apply_native_gains), the static drum the card's force-space K with kd 0, which Newton
  turns into MuJoCo's default solref (emitters/common.shape_config). Same μ on both (MuJoCo takes the max).
- MuJoCo settings: the default route's for this model (`alloy.bench.default_route.routed_mujoco_settings`: newton,
  elliptic, implicitfast; impratio and solimp at their defaults; the route's impratio fill-in is held back as for any
  CHAIN on Newton, SCR-WP3-10), plus contact capacity. On this rope the old A5
  settings (impratio 300, flat imp 0.99; MuJoCo C) creep 33-81 mm/s at μ 0.4 between 0.95 and 0.99, rising into
  the slip speed, so creep and slip do not separate; the default route's creep at most 14 mm/s.
- thin-rope limit: friction acts at the drum radius R, tension along the centre line ρ = R + r, so a rope holds
  exp(μθ·R/ρ); R/ρ = 0.993 here (−0.4 % / −0.9 %, inside the 0.99-1.01 bracket). The old 2 mm rope on a 30 mm drum
  had R/ρ = 0.936 (−3.9 % / −7.7 %).
- chain armature: the route's CHAIN floor (`emitters.chain.ARMATURE_FLOOR`, 1e-6 kg·m² on every rotational dof);
  the twin sets none of its own. Without it MuJoCo Warp's float32 Newton goes non-finite on these links.

Slip is judged by speed, not by a rise: below the threshold MuJoCo's soft friction creeps (SPEC §3.5 creep rule, the
CHAIN known gap; up to ~14 mm/s at 0.99 here); above it the rope accelerates at (r − 1)·T_in·e^{μθ}/m_rope. A world
slips when its T_in end moves faster, over the last `SLIP_WINDOW_S`, than `SLIP_FRACTION` of the speed the smallest
overload in `RATIOS` (1 %) would reach by then. The creep speed of the holding worlds is a monitor.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.bench.analytic.common import smoothstep
from alloy.bench.solvers import SolverSettings
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance
from alloy.core.errors import LawError
from alloy.route import checks as ck
from alloy.route.types import RouterConfig
from alloy.scene.analytic_rods import RodLawTwin, rod_card
from alloy.scene.native_gains import native_gains

DRUM_R, ROPE_R = 0.2, 1e-3                        # m
DENSITY = 7850.0                                  # kg/m³ (a steel wire rope's mass; gravity is off)
EI = 3e-6                                         # N·m², limp: bending's capstan offset ~EI/(T·R²) < 1e-4
LEG_LINKS = 2                                     # full links per leg below the half-length end link
T_IN = 1.0                                        # N
RATIOS = (0.96, 0.98, 0.99, 1.01, 1.02, 1.04)
GAP_M = 5e-5                                      # m, initial chord clearance
EASE_S = 0.05
K_CONTACT = 1e4                                   # N/m, force-space contact stiffness (links and drum)
SLIP_WINDOW_S = 0.05
SLIP_FRACTION = 0.25
CAPACITY = {"nconmax": 256, "njmax": 1536}        # per-world contact / constraint rows (not physics)

LAW = LawSpec(
    id="A10.capstan", suite="analytic",
    metrics={"ratio_rel_err": Tolerance(abs=0.02, source="bench §4 A10: capstan T_out/T_in = exp(μθ) within 2 %")},
    stat={"ratio_rel_err": "mean"}, dt=5e-4, substeps=1, duration_s=0.2, worlds_per_condition=len(RATIOS),
    solvers=("mujoco",), production_dt=False, producer="WP2",
    quick={"worlds": 2 * len(RATIOS), "duration_s": 0.2},
)


@dataclass(frozen=True)
class Wrap:
    """The routed rope's polygon: links, arc links, turn per arc hinge, link length, rope length, hinge radius, leg."""
    links: int
    arc: int
    alpha: float
    link: float
    length: float
    rw: float
    leg: float


@functools.lru_cache(maxsize=1)
def wrap() -> Wrap:
    """Links from the router's CHAIN rule for this rope (a fixed point: the rope length follows the link count)."""
    cfg = RouterConfig()
    c = ck.rod_bending_length(DENSITY, math.pi * ROPE_R**2, EI)
    d = DRUM_R + ROPE_R + GAP_M                     # chord distance from the drum axis
    n = 30                                          # start of the fixed point
    for _ in range(10):
        m = n - 2 - 2 * LEG_LINKS
        alpha = math.pi / m
        link = 2.0 * d * math.tan(0.5 * alpha)
        length = (n - 1) * link                     # D7 half-length ends
        routed, required = ck.chain_links(length, c, ROPE_R)
        if required > cfg.chain_links_max:
            raise LawError(f"{LAW.id}: rope needs {required} links > cap {cfg.chain_links_max} (not a CHAIN rope)")
        if routed == n:
            return Wrap(n, m, alpha, link, length, d / math.cos(0.5 * alpha), (LEG_LINKS + 0.5) * link)
        n = routed
    raise LawError(f"{LAW.id}: routed link count did not settle (last {n})")


def wrap_points() -> list[tuple[float, float, float]]:
    """Left leg up, the polygon's hinges over the top (drum axis along y at the origin), right leg down."""
    w = wrap()
    up = [(-w.rw, 0.0, -w.leg + w.leg * k / 20) for k in range(20)]
    arc = [(-w.rw * math.cos(w.alpha * k), 0.0, w.rw * math.sin(w.alpha * k)) for k in range(w.arc + 1)]
    down = [(w.rw, 0.0, -w.leg * k / 20) for k in range(1, 21)]
    return up + arc + down


def rope_mass() -> float:
    return DENSITY * math.pi * ROPE_R**2 * wrap().length


def slip_speed(mu: float, t_end: float) -> float:
    """SLIP_FRACTION of the speed the smallest overload in RATIOS reaches by t_end (m/s)."""
    r1 = min(r for r in RATIOS if r > 1.0)
    accel = (r1 - 1.0) * T_IN * math.exp(mu * math.pi) / rope_mass()
    return SLIP_FRACTION * accel * max(t_end - EASE_S, 0.0)


def threshold_ratio(r: np.ndarray, slipped: np.ndarray) -> float:
    """Midpoint between the largest holding and the smallest slipping ratio; NaN if not bracketed or non-monotone."""
    hold, slip = r[~slipped], r[slipped]
    if hold.size == 0 or slip.size == 0 or hold.max() > slip.min():
        return math.nan
    return 0.5 * (float(hold.max()) + float(slip.min()))


@wp.kernel
def _end_loads(body_q: wp.array(dtype=wp.transform), bodies: wp.array(dtype=wp.int32),
               local: wp.array(dtype=wp.vec3), load: wp.array(dtype=float), scale: float,
               body_f: wp.array(dtype=wp.spatial_vector)):
    """Dead load (0, 0, −P·scale) at a body point: the force and its moment about the COM (Newton order f, τ)."""
    i = wp.tid()
    b = bodies[i]
    r = wp.quat_rotate(wp.transform_get_rotation(body_q[b]), local[i])
    f = wp.vec3(0.0, 0.0, -load[i] * scale)
    t = wp.cross(r, f)
    body_f[b] = wp.spatial_vector(f[0], f[1], f[2], t[0], t[1], t[2])


def _chain_newton_held() -> set[str]:
    """Route fill-ins a plain scene gets and a CHAIN stepped by Newton does not (SCR-WP3-10: impratio)."""
    from alloy.scene.entries import mujoco_overrides

    held = mujoco_overrides(frozenset(), False, None, chain_newton=True)
    return set(mujoco_overrides(frozenset(), False, None)) - set(held)


class CapstanTwin(RodLawTwin):
    law_id = LAW.id
    units = {"ratio_rel_err": "1"}
    dt = LAW.dt
    settings: Mapping[str, SolverSettings] = {}       # MuJoCo as Alloy routes this model (settings_for)

    def __init__(self) -> None:
        super().__init__()
        self._loads: list[float] = []

    def settings_for(self, solver: str) -> SolverSettings:
        """The default route's MuJoCo settings for this model plus contact capacity. The rope is a CHAIN on the
        Newton solver, so the route's impratio fill-in is held back as for any such scene (`entries.chain_on_newton`,
        SCR-WP3-10): at impratio 300 the held worlds went non-finite (WP6, main 27da8424)."""
        s = super().settings_for(solver)
        if solver != "mujoco":
            return s
        kw = dict(s.kwargs)
        if str(kw.get("solver", "newton")) == "newton":
            kw = {k: v for k, v in kw.items() if k not in _chain_newton_held()}
        return SolverSettings("mujoco", {**kw, **CAPACITY}, mujoco_solimp=s.mujoco_solimp)

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"mu{m:g}", {"mu": m}) for m in (0.2, 0.4))

    def expand(self, cond: Condition, quick: bool, per: int) -> list[dict[str, float]]:
        return [{"ratio": r} for r in RATIOS]

    def world_gravity(self, cond: Condition) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        w, mu = wrap(), cond.params["mu"]
        rod = rod_card(w.length, EI, w.links, radius=ROPE_R, density=DENSITY)
        ke, kd = native_gains(K_CONTACT, DENSITY * math.pi * ROPE_R**2 * w.link, 0.0, dt=self.dt)
        link = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu, ke=ke, kd=kd, mu_torsional=0.0, mu_rolling=0.0)
        drum = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu, ke=K_CONTACT, kd=0.0, mu_torsional=0.0,
                                               mu_rolling=0.0)
        axis_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
        b.add_shape_capsule(-1, radius=DRUM_R, half_height=0.05, xform=wp.transform(wp.vec3(), axis_y), cfg=drum)
        h = self.emit(b, rod, wrap_points(), cfg=link, collide=True)
        self._loads += [T_IN, T_IN * sample["ratio"] * math.exp(mu * math.pi)]
        self.track([(h.bodies[0], (-0.5 * h.lengths[0], 0.0, 0.0)), (h.bodies[-1], (0.5 * h.lengths[-1], 0.0, 0.0))])

    def on_model(self, model: newton.Model) -> None:
        super().on_model(model)
        dev = model.device
        pts = [p for w in self._points for p in w]                        # (in, out) per world, interleaved
        self._lb = wp.array([p[0] for p in pts], dtype=wp.int32, device=dev)
        self._ll = wp.array([wp.vec3(*p[1]) for p in pts], dtype=wp.vec3, device=dev)
        self._lp = wp.array(np.asarray(self._loads, np.float32), dtype=float, device=dev)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        wp.launch(_end_loads, dim=len(self._loads), inputs=[state.body_q, self._lb, self._ll, self._lp,
                                                            smoothstep(t / EASE_S)],
                  outputs=[state.body_f], device=self.model.device)

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        _, _, z_in = self.point(log_, 0)
        t = log_.times()
        k = max(1, int(round(SLIP_WINDOW_S / (t[1] - t[0]))))
        speed = (z_in[:, -1] - z_in[:, -1 - k]) / (t[-1] - t[-1 - k])     # T_in end, up = toward T_out
        r = np.array([s["ratio"] for s in self.samples])
        ids = [c.id for c in self.worlds]
        err = np.empty(len(r))
        per = {}
        for cid in dict.fromkeys(ids):
            m = np.array([i == cid for i in ids])
            mu = next(c.params["mu"] for c in self.worlds if c.id == cid)
            v_slip = slip_speed(mu, float(t[-1]))
            slipped = speed[m] > v_slip
            rc = threshold_ratio(r[m], slipped) if np.all(np.isfinite(speed[m])) else math.nan   # NaN is not a hold
            err[m] = abs(rc - 1.0)
            hold = speed[m][~slipped]
            per[cid] = {"threshold_ratio": rc, "slip_speed_m_s": v_slip, "speed_m_s": speed[m].tolist(),
                        "rise_m": (z_in[m, -1] - z_in[m, 0]).tolist(),
                        "creep_max_m_s": float(np.abs(hold).max()) if hold.size else math.nan,
                        "thin_rope_ratio": math.exp(-mu * math.pi * (1.0 - DRUM_R / wrap().rw))}
        w = wrap()
        self.monitors.update({"per_condition": per,
                              "exp_mu_theta": {c.id: math.exp(c.params["mu"] * math.pi) for c in self.worlds},
                              "rope": {"links": w.links, "arc_links": w.arc, "link_m": w.link, "R_over_rho":
                                       DRUM_R / w.rw}})
        return {"ratio_rel_err": err}


__all__ = ["LAW", "CapstanTwin", "Wrap", "slip_speed", "threshold_ratio", "wrap", "wrap_points"]
