# SPDX-License-Identifier: Apache-2.0
"""Rig twin R2 (bench §4 R2: M10x1.5 torque vs angle) and a synthetic R2 dataset from the closed forms.

Protocol (bench.md R2): nut at π rad/s against a 5 N axial spring, run-down turns, then seat on a hardened washer to
10 N·m. Metrics: advance per revolution (1.500 ± 0.01 mm), run-down torque (10 % or 0.01 N·m, taken as the sum
0.01 N·m + 10 %), seating slope dT/dθ (10 %). Cross-threading (entry tilt ± 1°) is not covered: the twin's D6 drive
locks tilt, so the metric is left out of ``R2_LAW`` rather than reported as a pass.

Synthetic data (no rig yet; ``protocol_version`` "synthetic/..."): per trial the thread and bearing friction scatter
1.5 % (1σ) around the condition's truth (dry 0.14/0.14, oiled 0.10/0.10, our choice), the torque cell adds 1 mN·m and
the dial indicator 1 µm white noise at 1 kHz. Run-down torque is F·(d2/2)·tan(λ + ρ'), tan ρ' = μ·sec 30° (A9);
after the nut face meets the washer the clamp force grows as k_c·p·Δθ/2π (card ``holder.k_axial`` 2e8 N/m, a typical
M10 joint, not measured) and T = (F + F_c)·(d2/2)·tan(λ + ρ') + μ_b·F_c·r_b, r_b the uniform-pressure radius of the
nut face on the washer (5.25..8.5 mm). Trials alternate calib/valid.

Twin: ``ThreadTwinBase`` with the washer as a modelled stop: after ``turns`` run-down turns the nut's axial slide meets
a one-sided spring of the card's k_c (damping ratio 0.5) and the spin a bearing friction torque μ_b·F_c·r_b; the thread
contact that carries F_c up to ~5 kN is simulated (analytic pair, every row the stiff row K). Washer contact rows
were tried first: a K-split face patch added MuJoCo's row damping 2τ·K·v (τ 4 ms at 0.55 mm/s) and raised the slope
70 %; a spring-mounted washer body rang at 0.7 kHz and unloaded the thread mid-fit.
``bind_calib`` fits μ per condition from the calib run-down torques and μ_b from the calib seating slopes, inverting
the closed forms (k_c from the card); ``valid`` judges. Seating stops the drive (target 0) once the torque reaches
10 N·m. The run-down count is 1 turn in quick mode (6 on the rig): the metrics are per turn.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, RigCard, RigDataset, RigTrial, Tolerance
from alloy.contact.thread_sdf import SEC30, parse_thread_id, rundown_torque
from alloy.contact.thread_twins import (BEARING_R_IN, BOLT_ID, CONTROL_DT, NUT_OUTER, SETTLE_S, ThreadRig,
                                        ThreadTwinBase, bearing_radius, nut_mass, thread_dt, thread_metrics)
from alloy.core.timebase import snap_to_grid
from alloy.core.errors import LawError
from alloy.core.log import get_logger

log = get_logger("contact.thread_rig")

T_STOP = 10.0                       # N·m seating torque
F_AXIAL, OMEGA = 5.0, math.pi
K_CLAMP = 2.0e8                     # N/m synthetic card value
RATE_HZ = 1000.0
TRIAL_SPREAD, TORQUE_NOISE, Z_NOISE = 0.015, 1.0e-3, 1.0e-6
METRICS = ("advance_per_rev", "rundown_torque", "seating_slope")
_SRC = "bench §4 R2"


@dataclass(frozen=True)
class R2Truth:
    """Synthetic ground truth of one condition."""

    mu_thread: float
    mu_bearing: float


TRUTH: dict[str, R2Truth] = {"dry": R2Truth(0.14, 0.14), "oiled": R2Truth(0.10, 0.10)}


def thread_factor(mu: float) -> float:
    """T/F during run-down [m]: (d2/2)·tan(λ + ρ') (A9)."""
    return rundown_torque(parse_thread_id(BOLT_ID), 1.0, mu)


def seating_slope(mu: float, mu_b: float, k_clamp: float) -> float:
    """dT/dθ once seated [N·m/rad]: k_c·(p/2π)·((d2/2)·tan(λ + ρ') + μ_b·r_b)."""
    p = parse_thread_id(BOLT_ID).pitch
    return k_clamp * p / (2.0 * math.pi) * (thread_factor(mu) + mu_b * bearing_radius())


def fit_mu(torque: float, axial: float) -> float:
    """Inverse of the run-down formula: μ = tan(atan(2T/(F·d2)) - λ)·cos 30°."""
    spec = parse_thread_id(BOLT_ID)
    lam = math.atan(spec.pitch / (math.pi * spec.d2))
    return math.tan(math.atan(2.0 * torque / (axial * spec.d2)) - lam) / SEC30


def fit_mu_bearing(slope: float, mu: float, k_clamp: float) -> float:
    """Inverse of ``seating_slope`` for μ_b."""
    p = parse_thread_id(BOLT_ID).pitch
    return (slope / (k_clamp * p / (2.0 * math.pi)) - thread_factor(mu)) / bearing_radius()


# ---------------------------------------------------------------- synthetic dataset ---------------------------------

def _trial_series(truth: R2Truth, turns: float, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Closed-form run-down then seating to T_STOP, sampled at RATE_HZ with sensor noise."""
    p = parse_thread_id(BOLT_ID).pitch
    mu = truth.mu_thread * (1.0 + TRIAL_SPREAD * rng.standard_normal())
    mu_b = truth.mu_bearing * (1.0 + TRIAL_SPREAD * rng.standard_normal())
    seat = 2.0 * math.pi * turns
    t = np.arange(0.0, SETTLE_S + seat / OMEGA + 1.0, 1.0 / RATE_HZ)
    theta = OMEGA * t
    clamp = K_CLAMP * p / (2.0 * math.pi) * np.maximum(theta - seat, 0.0)
    tau = (F_AXIAL + clamp) * thread_factor(mu) + mu_b * clamp * bearing_radius()
    stop = int(np.argmax(tau >= T_STOP)) + 1 if np.any(tau >= T_STOP) else len(t)
    t, theta, tau, clamp = t[:stop], theta[:stop], tau[:stop], clamp[:stop]
    z = p * theta / (2.0 * math.pi) - clamp / K_CLAMP
    return {"t": t, "theta": theta, "torque": tau + TORQUE_NOISE * rng.standard_normal(len(t)),
            "z": z + Z_NOISE * rng.standard_normal(len(t))}


def _card(seed: int) -> RigCard:
    bolt = parse_thread_id(BOLT_ID)
    return RigCard(
        law_id="R2.thread_rundown", dataset_id=f"synthetic-seed{seed}", protocol_version="synthetic/closed-form-1",
        objects={"bolt": {"d": bolt.d, "pitch": bolt.pitch, "d2": bolt.d2},
                 "washer": {"r_in": BEARING_R_IN, "r_out": NUT_OUTER}},
        instruments={"advance_per_rev": {"u95": 2.0e-6}, "rundown_torque": {"u95": 2.0e-3},
                     "seating_slope": {"u95": 1.0}},
        holder={"k_axial": K_CLAMP, "spring_force": F_AXIAL}, environment={"temperature_K": 293.15})


def _digest(card: RigCard, trials: Sequence[RigTrial]) -> str:
    h = hashlib.sha256(json.dumps({"card": card.dataset_id, "law": card.law_id}, sort_keys=True).encode())
    for tr in trials:
        h.update(f"{tr.condition_id}/{tr.index}/{tr.split}".encode())
        for key in sorted(tr.series):
            h.update(np.ascontiguousarray(tr.series[key], np.float64).tobytes())
    return h.hexdigest()


def synthetic_r2_dataset(seed: int = 0, trials_per_condition: int = 10, turns: float = 6.0) -> RigDataset:
    """In-memory synthetic R2 dataset (dry and oiled), scalars from ``thread_metrics`` on the noisy series."""
    rng = np.random.default_rng(seed)
    conds = tuple(Condition(c, {"F_axial": F_AXIAL, "omega": OMEGA, "oiled": float(c == "oiled")}) for c in TRUTH)
    trials = []
    for cond in conds:
        for k in range(trials_per_condition):
            series = _trial_series(TRUTH[cond.id], turns, rng)
            scalars = thread_metrics(series["t"], series["theta"], series["z"], series["torque"])
            trials.append(RigTrial(cond.id, k, "calib" if k % 2 == 0 else "valid", series, scalars))
    card = _card(seed)
    return RigDataset(card, conds, tuple(trials), _digest(card, trials))


# ---------------------------------------------------------------- law and twin --------------------------------------

R2_ARMATURE = 30.0                  # kg axial: the explicit washer spring needs sqrt(k_c/m)·dt <= 1/2
R2_DT = snap_to_grid(min(thread_dt(6000.0, R2_ARMATURE), 0.5 * math.sqrt((nut_mass() + R2_ARMATURE) / K_CLAMP)),
                     CONTROL_DT)
R2_LAW = LawSpec(
    id="R2.thread_rundown", suite="rig",
    metrics={"advance_per_rev": Tolerance(abs=1.0e-5, source=f"{_SRC}: 1.500 ± 0.01 mm per rev"),
             "rundown_torque": Tolerance(abs=0.01, rel=0.10, source=f"{_SRC}: run-down torque 10 % or 0.01 N·m"),
             "seating_slope": Tolerance(rel=0.10, source=f"{_SRC}: seating slope dT/dθ 10 %")},
    stat={m: "mean" for m in METRICS}, dt=R2_DT, substeps=1, duration_s=SETTLE_S + 12.0 + 0.4,
    worlds_per_condition=1, solvers=("mujoco",), producer="WP3",
    quick={"worlds": 2, "duration_s": SETTLE_S + 2.0 + 0.4})


class ThreadR2Twin(ThreadTwinBase):
    """R2 twin; see the module docstring."""

    law_id = R2_LAW.id

    def __init__(self) -> None:
        super().__init__(R2_LAW.dt)
        self.fit: dict[str, R2Truth] = {}
        self.k_clamp = K_CLAMP

    def bind_calib(self, card: RigCard, calib: Sequence[RigTrial]) -> None:
        """μ and μ_b per condition from calib trials only (the closed forms inverted); k_c from the card."""
        k_c = float(card.holder["k_axial"])
        self.k_clamp = k_c
        by: dict[str, list[RigTrial]] = {}
        for tr in calib:
            by.setdefault(tr.condition_id, []).append(tr)
        for cid, trs in by.items():
            mu = fit_mu(float(np.mean([t.scalars["rundown_torque"] for t in trs])), F_AXIAL)
            slope = float(np.mean([t.scalars["seating_slope"] for t in trs]))
            self.fit[cid] = R2Truth(mu, fit_mu_bearing(slope, mu, k_c))
        self.monitors["fit_calib"] = {c: (f.mu_thread, f.mu_bearing) for c, f in self.fit.items()}

    def expand(self, cond: Condition, quick: bool, per: int) -> list[dict[str, float]]:
        return [{"turns": 1.0 if quick else 6.0}]

    def rig_for(self, cond: Condition, sample: Mapping[str, float]) -> ThreadRig:
        if cond.id not in self.fit:
            raise LawError(f"R2 twin: no calib fit for condition {cond.id!r} (bind_calib first)")
        fit = self.fit[cond.id]
        gap = parse_thread_id(BOLT_ID).pitch * sample["turns"]
        return ThreadRig(mu_thread=fit.mu_thread, mu_bearing=fit.mu_bearing, axial_load=cond.params["F_axial"],
                         omega=cond.params["omega"], seat_gap=gap, k_clamp=self.k_clamp, axial_armature=R2_ARMATURE,
                         torque_stop=T_STOP)

    def scalars(self, log_: DeviceTwinLog) -> dict[str, np.ndarray]:
        rows = self.per_world(log_)
        return {m: np.array([r[m] for r in rows]) for m in METRICS}


__all__ = ["R2_LAW", "ThreadR2Twin", "synthetic_r2_dataset", "seating_slope", "fit_mu", "fit_mu_bearing",
           "thread_factor", "TRUTH", "K_CLAMP"]
