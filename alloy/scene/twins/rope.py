# SPDX-License-Identifier: Apache-2.0
"""R9 rope sag and period twin (bench §4.4 R9): sag over a 0.8 m span (± 3 mm), hanging pluck period (3 %) and log
decrement (20 %), cantilever tip drop (3 %), on the CHAIN route.

One world per rope (condition). Each world carries three 40-link chains built by the CHAIN emitter, the same in every
world (SolverMuJoCo needs homogeneous worlds): a 1.0 m rope pinned at both ends 0.8 m apart (ball + MuJoCo connect,
settled with joint damping), a 1.0 m rope pinned at the top and released from its first mode, and a 0.45 m clamped
cantilever (self-test settle). Fitted on `calib` trials only: EI from the mean calib tip drop (inverting the
large-deflection cantilever, `rodref.elastica_tip_drop`), and a mass-proportional drag rate α (F = −α·m·v on every
pluck link, so ζ = α/(2ω₁) exactly) from the mean calib log decrement and the chain's own ω₁. Joint dampers do not
work here: on 0.3 g links they lock the hinge they sit on (hinge dampers giving δ ≈ 0.05 made the rope swing as a
rigid pendulum, δ 0.006; a pivot damper gave 43-64 % of the projected δ on limp ropes). Geometry (radius,
mass per length) comes from the condition, as a rig measures it. Initial shapes are set through joint_q / the MuJoCo
spring reference so every rope's stress-free shape is straight.

The synthetic truth (`dataset`) is the reference physics with bending: `roperef.supported_sag`, the hanging first mode
with hinge damping, and the elastica cantilever, plus Gaussian trial noise.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from alloy._vendor import newton
import warp as wp

from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, RigCard, RigDataset, RigTrial, Tolerance
from alloy.route import rodref, roperef
from alloy.scene.analytic_rods import RodLawTwin, crossing_period, rod_card

_SRC = "bench §4.4 R9"
LINKS = 40
ROPE_L, SPAN, CANT_L = 1.0, 0.8, 0.45
SAG_DAMPING = 2e-4              # N·m·s/rad per dof (the A11.catenary settle)
PLUCK_AMP = 0.01                # m, tip excursion of the released first mode
STRAIGHT_SIGN = 1.0             # sign of the corner hinge angle that straightens the V (checked in tests)
METRICS = ("sag", "period", "log_decrement", "tip_drop")

LAW = LawSpec(
    id="R9.rope", suite="rig",
    metrics={"sag": Tolerance(abs=3e-3, source=f"{_SRC}: sag ± 3 mm"),
             "period": Tolerance(rel=0.03, source=f"{_SRC}: period 3 %"),
             "log_decrement": Tolerance(rel=0.20, source=f"{_SRC}: log decrement 20 %"),
             "tip_drop": Tolerance(rel=0.03, source=f"{_SRC}: tip drop 3 %")},
    stat={m: "mean" for m in METRICS}, dt=2e-3, substeps=1, duration_s=10.0, worlds_per_condition=1,
    solvers=("mujoco",), production_dt=False, producer="WP2", quick={"worlds": 3, "duration_s": 10.0},
)

# synthetic ropes: geometry as measured, EI and the pluck log decrement as the hidden truth
ROPES: dict[str, dict[str, float]] = {
    "paracord": {"radius": 0.002, "lin_mass": 0.0138, "EI": 4e-4, "log_decrement": 0.15},
    "usb_cable": {"radius": 0.00175, "lin_mass": 0.01124, "EI": 5.67e-5, "log_decrement": 0.10},
    "steel_wire": {"radius": 0.0005, "lin_mass": 7850.0 * math.pi * 0.0005**2, "EI": 9.8e-3, "log_decrement": 0.05},
}
NOISE = {"sag": 3e-4, "period": 1e-3, "log_decrement": 0.02, "tip_drop": 1e-4}       # sag/tip m; period/δ relative
INSTRUMENTS = {"sag": {"u95": 5e-4, "what": "photo + scale bar, 0.5 mm"},
               "period": {"u95": 3e-3, "what": "240 fps tip track, 3 ms"},
               "log_decrement": {"u95": 3e-3, "what": "peak fit"},
               "tip_drop": {"u95": 2e-4, "what": "laser distance, 0.2 mm"}}


def drag_rate_for(log_dec: float, omega: float) -> float:
    """Mass-proportional drag rate α [1/s] (F = −α·m·v on every link) giving the mode at ω the log decrement δ:
    proportional damping, so ζ = α/(2ω) for every mode and every discretisation."""
    zeta = log_dec / math.sqrt(4.0 * math.pi**2 + log_dec**2)
    return 2.0 * zeta * omega


def chain_lengths(L: float, links: int = LINKS) -> np.ndarray:
    """D7 link lengths, top to bottom: half-length ends."""
    l = np.full(links, L / (links - 1))
    l[0] = l[-1] = 0.5 * L / (links - 1)
    return l


def fit_ei(tip_drop: float, L: float, lin: float) -> float:
    """EI whose large-deflection self-weight cantilever drops `tip_drop` (bisection in log EI; drop falls with EI)."""
    lo, hi = -9.0, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if rodref.elastica_tip_drop(L, lin, 10.0**mid) < tip_drop else (mid, hi)
    return 10.0 ** (0.5 * (lo + hi))


def truth(params: Mapping[str, float], cond: Condition) -> dict[str, float]:
    p = ROPES[cond.id]
    mode = roperef.hanging_modes(ROPE_L, p["lin_mass"], p["EI"], 0.0, p["radius"])
    return {"sag": roperef.supported_sag(ROPE_L, SPAN, p["lin_mass"], p["EI"]), "period": mode.period,
            "log_decrement": p["log_decrement"], "tip_drop": rodref.elastica_tip_drop(CANT_L, p["lin_mass"], p["EI"])}


def dataset(seed: int = 0) -> RigDataset:
    """Synthetic R9 dataset: 15 trials per rope (5 calib, 10 valid), noise as `NOISE` (relative for period/δ)."""
    from alloy.testing.fakes import synthetic_rig

    conds = tuple(Condition(name, {"radius": p["radius"], "lin_mass": p["lin_mass"]}) for name, p in ROPES.items())
    data = synthetic_rig(LAW.id, {}, 0.0, conditions=conds, trials_per_condition=15, calib_per_condition=5,
                         seed=seed, model=truth)
    rng = np.random.default_rng(seed + 1)
    trials = tuple(_noisy(t, rng) for t in data.trials)
    card = dataclasses.replace(data.card, dataset_id="synthetic-v1", protocol_version="synthetic/1",
                               instruments=dict(INSTRUMENTS),
                               objects={n: {"radius_m": p["radius"], "lin_mass_kg_m": p["lin_mass"]}
                                        for n, p in ROPES.items()})
    return dataclasses.replace(data, card=card, trials=trials)


def _noisy(t: RigTrial, rng: np.random.Generator) -> RigTrial:
    out = {}
    for k, v in t.scalars.items():
        sigma = NOISE[k] * (abs(v) if k in ("period", "log_decrement") else 1.0)
        out[k] = float(v + sigma * rng.standard_normal())
    return dataclasses.replace(t, scalars=out)


@wp.kernel
def _drag(body_qd: wp.array(dtype=wp.spatial_vector), mass: wp.array(dtype=float), bodies: wp.array(dtype=wp.int32),
          rate: wp.array(dtype=float), body_f: wp.array(dtype=wp.spatial_vector)):
    """F = −α·m·v at the COM (Newton order: linear first)."""
    i = wp.tid()
    b = bodies[i]
    q = body_qd[b]
    v = wp.vec3(q[0], q[1], q[2])
    f = -rate[i] * mass[b] * v
    body_f[b] = wp.spatial_vector(f[0], f[1], f[2], 0.0, 0.0, 0.0)


def _bend(b: newton.ModelBuilder, h, root_angle: float, hinges: Mapping[int, float]) -> None:
    """Initial joint_q of a straight chain: the root ball turned by `root_angle` about the link y axis, and hinge i
    (between links i − 1 and i) bent by hinges[i] about its first (y) axis."""
    j0 = h.joints[0]
    qs = b.joint_q_start[j0]
    b.joint_q[qs:qs + 4] = list(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), root_angle))
    for i, a in hinges.items():
        b.joint_q[b.joint_q_start[h.joints[i]]] = a


def _chain_mode_angles(mode: roperef.HangingMode, lengths: Sequence[float], amp: float = PLUCK_AMP) -> np.ndarray:
    """Reference mode angles at the chain's link centres, scaled to a tip excursion `amp`."""
    n = len(mode.angles)
    s_ref = (np.arange(n) + 0.5) * (ROPE_L / n)
    ends = np.cumsum(lengths)
    centres = ends - 0.5 * np.asarray(lengths)
    phi = np.interp(centres, s_ref, mode.angles)
    return phi * (amp / float(np.sum(np.asarray(lengths) * np.sin(phi))))


def peak_log_decrement(t: np.ndarray, x: np.ndarray) -> float:
    """Log decrement from successive positive peaks of x(t) (slope of ln peak vs peak index); NaN with < 3 peaks."""
    i = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > 0.0)) + 1
    if i.size < 3:
        return math.nan
    return float(-np.polyfit(np.arange(i.size), np.log(x[i]), 1)[0])


class RopeTwin(RodLawTwin):
    law_id = LAW.id
    units = {"sag": "m", "period": "s", "log_decrement": "1", "tip_drop": "m"}
    dt = LAW.dt

    def __init__(self) -> None:
        super().__init__()
        self.fit: dict[str, dict[str, float]] = {}
        self._drag: list[tuple[int, float]] = []

    def on_model(self, model: newton.Model) -> None:
        super().on_model(model)
        dev = model.device
        self._drag_bodies = wp.array([d[0] for d in self._drag], dtype=wp.int32, device=dev)
        self._drag_rate = wp.array(np.asarray([d[1] for d in self._drag], np.float32), dtype=float, device=dev)

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        wp.launch(_drag, dim=len(self._drag), inputs=[state.body_qd, self.model.body_mass, self._drag_bodies,
                                                       self._drag_rate], outputs=[state.body_f],
                  device=self.model.device)

    def bind_calib(self, card: RigCard, calib: Sequence[RigTrial]) -> None:
        by: dict[str, list[RigTrial]] = {}
        for tr in calib:
            by.setdefault(tr.condition_id, []).append(tr)
        for cid, trials in by.items():
            geo = card.objects[cid]
            lin, r = geo["lin_mass_kg_m"], geo["radius_m"]
            ei = fit_ei(float(np.mean([t.scalars["tip_drop"] for t in trials])), CANT_L, lin)
            w = roperef.hanging_modes(ROPE_L, lin, ei, 0.0, r, lengths=chain_lengths(ROPE_L)).omega
            alpha = drag_rate_for(float(np.mean([t.scalars["log_decrement"] for t in trials])), w)
            self.fit[cid] = {"EI": ei, "drag_rate": alpha, "lin_mass": lin, "radius": r}
        self.monitors["fit_calib"] = {k: dict(v) for k, v in self.fit.items()}

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        f = self.fit[cond.id]
        density = f["lin_mass"] / (math.pi * f["radius"] ** 2)
        card = lambda L, **kw: rod_card(L, f["EI"], LINKS, radius=f["radius"], density=density, **kw)  # noqa: E731
        pts: list = []
        pts += self._sag_rope(b, card(ROPE_L))
        pts += self._pluck_rope(b, card(ROPE_L), f)
        h = self.emit(b, card(CANT_L, damping_ratio=0.7), [(0.0, 1.0, 1.0), (CANT_L, 1.0, 1.0)], clamp=True)
        pts.append((h.bodies[-1], (0.5 * h.lengths[-1], 0.0, 0.0)))
        self.track(pts)

    def _sag_rope(self, b: newton.ModelBuilder, rod) -> list:
        """Built as the V of the rope length between the pins (so the connect anchor at B is inferred there), with
        the corner hinge's MuJoCo spring reference at the straight angle: the stress-free shape stays straight."""
        from alloy._vendor.newton.solvers import mujoco as nmj

        d = math.sqrt(0.25 * (ROPE_L**2 - SPAN**2))
        q0 = b.joint_dof_count
        h = self.emit(b, rod, [(-0.5 * SPAN, 0.0, 1.0), (0.0, 0.0, 1.0 - d), (0.5 * SPAN, 0.0, 1.0)], pin=True)
        mid = LINKS // 2 - 1                                    # node L/2 = outer end of link N/2 − 1 (N even)
        corner = b.joint_qd_start[h.joints[mid + 1]]
        b.custom_attributes["mujoco:dof_springref"].values[corner] = STRAIGHT_SIGN * 2.0 * math.acos(SPAN / ROPE_L)
        for dof in range(q0, b.joint_dof_count):
            b.joint_damping[dof] = SAG_DAMPING               # armature: the route's CHAIN floor (emitters.chain)
        end = (0.5 * h.lengths[-1], 0.0, 0.0)
        nmj.add_equality_connect(b, h.bodies[-1], -1, anchor=wp.vec3(*end), solref=(2.0 * self.dt, 1.0))
        return [(h.bodies[mid], (0.5 * h.lengths[mid], 0.0, 0.0)), (h.bodies[-1], end)]

    def _pluck_rope(self, b: newton.ModelBuilder, rod, f: Mapping[str, float]) -> list:
        """Built hanging straight, released from the reference first mode (link angles set through joint_q)."""
        mode = roperef.hanging_modes(ROPE_L, f["lin_mass"], f["EI"], 0.0, f["radius"], n=200)
        h = self.emit(b, rod, [(0.0, 0.5, 1.5), (0.0, 0.5, 1.5 - ROPE_L)], pin=True)
        phi = _chain_mode_angles(mode, h.lengths)
        _bend(b, h, float(phi[0]), {i: float(phi[i] - phi[i - 1]) for i in range(1, len(phi))})
        self._drag += [(body, f["drag_rate"]) for body in h.bodies]
        return [(h.bodies[-1], (0.5 * h.lengths[-1], 0.0, 0.0))]

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        t = log_.times()
        _, _, mid_z = self.point(log_, 0)
        end_x, _, end_z = self.point(log_, 1)
        pluck_x, _, _ = self.point(log_, 2)
        _, _, tip_z = self.point(log_, 3)
        sag = 0.5 * (1.0 + end_z[:, -1]) - mid_z[:, -1]
        x = pluck_x                                             # hangs at x = 0 (straight rest, pinned at x = 0)
        period = np.array([crossing_period(t, row) for row in x])
        logdec = np.array([peak_log_decrement(t, row) for row in x])
        drop = 1.0 - tip_z[:, -1]
        self.monitors["per_condition"] = {c.id: {"sag": sag[i], "period": period[i], "log_decrement": logdec[i],
                                                 "tip_drop": drop[i], "span_realised": float(end_x[i, -1] + 0.4)}
                                          for i, c in enumerate(self.worlds)}
        return {"sag": sag, "period": period, "log_decrement": logdec, "tip_drop": drop}


__all__ = ["LAW", "RopeTwin", "dataset", "truth", "fit_ei", "drag_rate_for", "chain_lengths", "peak_log_decrement"]
