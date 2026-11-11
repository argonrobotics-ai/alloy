# SPDX-License-Identifier: Apache-2.0
"""Synthetic R1 dataset (bench §4.4 R1) from Whitney's quasi-static two-point-contact theory plus seeded noise.

Model (Whitney 1982, "Quasi-static assembly of compliantly supported rigid parts", J. Dyn. Sys. Meas. Control 104:
65-77), planar and quasi-static, rigid parts, Coulomb friction ``mu``, gravity tared. The holder acts at a compliance
centre ``l_g`` above the peg tip: lateral force ``F_x = -k_x (x_c - x_c0)`` and moment ``M = -k_theta (θ - θ0)``,
where the unloaded peg has tip offset ``ε0`` and tilt ``θ0`` (lean +x) and ``x_c = x_tip + l_g sin θ``. The stage
drives the tip depth ``z`` (axially rigid). Contacts: A, the hole edge (at the chamfer's foot, depth ``e`` below the
top face) on the peg's +x side, ``h = z - e`` above the tip; B, the peg's -x tip corner on the -x bore wall. Each
depth solves force (x, z) and moment balance with sliding friction (peg moving down) for the active set:
- free: no gap closed, force 0;
- one-point (A or B): one geometric constraint; the tilt is the root of the moment balance;
- two-point (A and B): the tilt is fixed by geometry (``(2R - r cos θ) cos θ - h sin θ = r``, the A8 clearance law
  for small θ: θ ≈ c/h); the three balances are linear in f_A, f_B and the insertion force P;
- jam (wedged): two-point with no sliding solution (f_A, f_B, P >= 0), i.e. the axial force's moment about the
  contacts, ``l_g sin θ · P``, outgrows the friction couple. The peg then stops and the stage compresses the holder at
  ``k_z v`` until the protocol's stop force.
Two-point forces scale roughly as (a/h - b/h²), so the peak sits at h of order 2c/θ0 (9.5 mm above the edge at 2°
with the default holder; past the 19 mm stroke at 1° and below, where the force only rises). Validity: tip offset in
the lean plane with 0 <= ε0 <= c/2 (the tip clears the edge, no chamfer crossing), θ0 >= 0; stage-frame springs.
Tilt and offset are the hanging peg's, so gravity (tared in the force) only enters through them.

Noise (seeded): per trial, tilt and offset jitter from the card's setting repeatability and mu × (1 + N(0, 0.02));
per sample, force N(0, 0.05 N) and depth N(0, 2 µm), at 100 Hz (the 1 kHz cell averaged over 10 samples).
Trials: the first half of each condition is ``calib``, the rest ``valid``. Metrics are reduced with
``alloy.contact.rig_r1.insertion_metrics``, the same code the twin uses.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from alloy.bench.types import Condition, RigCard, RigDataset, RigTrial
from alloy.contact.rig_r1 import (
    DEFAULT_MU, LAW_ID, R1Geometry, R1Holder, R1Protocol, insertion_metrics, r1_conditions,
)
from alloy.core.log import get_logger

log = get_logger("contact.rig")

FREE, ONE_A, ONE_B, TWO, JAM = range(5)
SECANT_ITERS = 40


@dataclass(frozen=True)
class WhitneyPair:
    """One trial's planar parameters (SI): bore and peg radii, edge depth, friction, holder, initial pose."""

    R: float
    r: float
    edge: float
    mu: float
    k_x: float
    k_theta: float
    l_g: float
    tilt: float
    offset: float

    @classmethod
    def build(cls, g: R1Geometry, holder: R1Holder, tilt: float, offset: float, chamfer: float,
              mu: float) -> WhitneyPair:
        if tilt < 0.0 or not 0.0 <= offset <= 0.5 * g.clearance + 1e-12:
            raise ValueError(f"outside the model's validity: tilt {tilt} rad, offset {offset} m (0..c/2)")
        return cls(0.5 * g.bore_d, 0.5 * g.peg_d, chamfer, mu, holder.k_x, holder.k_theta, holder.l_g, tilt, offset)


@dataclass(frozen=True)
class _Solution:
    th: np.ndarray
    x_t: np.ndarray
    fa: np.ndarray
    fb: np.ndarray
    P: np.ndarray


def _secant(fun: Callable[[np.ndarray], np.ndarray], x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Vectorised secant root of ``fun``; converged entries stop moving."""
    f0, f1 = fun(x0), fun(x1)
    for _ in range(SECANT_ITERS):
        d = f1 - f0
        safe = np.abs(d) > 1e-300
        step = np.where(safe, f1 * (x1 - x0) / np.where(safe, d, 1.0), 0.0)
        x0, f0 = x1, f1
        x1 = x1 - step
        f1 = fun(x1)
    return x1


def _wrench(p: WhitneyPair, x_t: np.ndarray, th: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Holder lateral force and moment on the peg."""
    fx = -p.k_x * (x_t + p.l_g * np.sin(th) - p.offset - p.l_g * math.sin(p.tilt))
    return fx, -p.k_theta * (th - p.tilt)


def _gap_a(p: WhitneyPair, x_t: np.ndarray, th: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Edge to the peg's +x side, along the side's normal."""
    return (p.R - x_t) * np.cos(th) - h * np.sin(th) - p.r


def _wall(p: WhitneyPair, z: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Radius of the -x wall at the -x tip corner's depth (the chamfer cone above the edge)."""
    return p.R + np.maximum(0.0, p.edge - (z - p.r * np.sin(th)))


def _gap_b(p: WhitneyPair, x_t: np.ndarray, th: np.ndarray, z: np.ndarray) -> np.ndarray:
    return x_t - p.r * np.cos(th) + _wall(p, z, th)


def _dirs(p: WhitneyPair, th: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Force of contact A on the peg per unit normal load: normal -n plus friction mu along +axis."""
    return -np.cos(th) + p.mu * np.sin(th), np.sin(th) + p.mu * np.cos(th)


def _torque(p: WhitneyPair, th, fx, m, P, fa, fb, x_t, h) -> np.ndarray:
    """Moment balance about the tip centre (lean +x positive): holder, contact A, contact B."""
    ax, ay = _dirs(p, th)
    return (p.l_g * np.cos(th) * fx + p.l_g * np.sin(th) * P + m + fa * (h * ax - (p.R - x_t) * ay)
            + fb * p.r * (np.sin(th) + p.mu * np.cos(th)))


def _one_a(p: WhitneyPair, th: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, ...]:
    x_t = p.R - (p.r + h * np.sin(th)) / np.cos(th)
    fx, m = _wrench(p, x_t, th)
    ax, ay = _dirs(p, th)
    fa = -fx / ax
    P = fa * ay
    return x_t, fa, P, _torque(p, th, fx, m, P, fa, 0.0, x_t, h)


def _one_b(p: WhitneyPair, th: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, ...]:
    x_t = p.r * np.cos(th) - _wall(p, z, th)
    fx, m = _wrench(p, x_t, th)
    fb = -fx
    P = p.mu * fb
    return x_t, fb, P, _torque(p, th, fx, m, P, 0.0, fb, x_t, 0.0)


def _solve_one_a(p: WhitneyPair, h: np.ndarray) -> _Solution:
    th0 = np.full_like(h, p.tilt)
    th = _secant(lambda t: _one_a(p, t, h)[3], th0, th0 + 1e-4)
    x_t, fa, P, _ = _one_a(p, th, h)
    return _Solution(th, x_t, fa, np.zeros_like(h), P)


def _solve_one_b(p: WhitneyPair, z: np.ndarray) -> _Solution:
    th0 = np.full_like(z, p.tilt)
    th = _secant(lambda t: _one_b(p, t, z)[3], th0, th0 + 1e-4)
    x_t, fb, P, _ = _one_b(p, th, z)
    return _Solution(th, x_t, np.zeros_like(z), fb, P)


def _solve_two(p: WhitneyPair, h: np.ndarray) -> _Solution:
    """Two-point tilt from geometry, then the linear force/moment balance in (f_A, f_B, P)."""
    guess = 2.0 * (p.R - p.r) / h
    th = _secant(lambda t: _gap_a(p, p.r * np.cos(t) - p.R, t, h), guess, 1.01 * guess)
    x_t = p.r * np.cos(th) - p.R
    fx, m = _wrench(p, x_t, th)
    ax, ay = _dirs(p, th)
    A = np.zeros(h.shape + (3, 3))
    A[..., 0, :] = np.stack([ax, np.ones_like(h), np.zeros_like(h)], -1)
    A[..., 1, :] = np.stack([ay, np.full_like(h, p.mu), -np.ones_like(h)], -1)
    A[..., 2, :] = np.stack([h * ax - (p.R - x_t) * ay, p.r * (np.sin(th) + p.mu * np.cos(th)),
                             p.l_g * np.sin(th)], -1)
    b = np.stack([-fx, np.zeros_like(h), -(p.l_g * np.cos(th) * fx + m)], -1)
    det = np.linalg.det(A)
    ok = np.abs(det) > 1e-18
    A[~ok] = np.eye(3)
    sol = np.linalg.solve(A, b[..., None])[..., 0]
    sol[~ok] = -1.0                                  # singular: no sliding solution
    return _Solution(th, x_t, sol[..., 0], sol[..., 1], sol[..., 2])


@dataclass(frozen=True)
class QuasiStatic:
    """Per depth: insertion force P [N], phase, tilt [rad], tip offset [m], holder lateral force and moment."""

    P: np.ndarray
    phase: np.ndarray
    th: np.ndarray
    x_t: np.ndarray
    fx: np.ndarray
    m: np.ndarray


def quasi_static_force(p: WhitneyPair, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Insertion force P [N] and contact phase (FREE, ONE_A, ONE_B, TWO, JAM) at tip depths ``z``."""
    qs = quasi_static(p, z)
    return qs.P, qs.phase


def quasi_static(p: WhitneyPair, z: np.ndarray) -> QuasiStatic:
    """The quasi-static solution at tip depths ``z`` (JAM rows carry the free pose and zero force)."""
    z = np.asarray(z, np.float64)
    h = z - p.edge
    engaged = h > 0.0
    hs = np.where(engaged, h, 1.0)                   # placeholder where the edge is not reached (masked below)
    th0, x0 = np.full_like(z, p.tilt), np.full_like(z, p.offset)
    with np.errstate(all="ignore"):
        free = (~engaged | (_gap_a(p, x0, th0, hs) >= 0.0)) & (_gap_b(p, x0, th0, z) >= 0.0)
        a, b, t = _solve_one_a(p, hs), _solve_one_b(p, z), _solve_two(p, hs)
        ok_a = ~free & engaged & (a.fa >= 0.0) & (_gap_b(p, a.x_t, a.th, z) >= 0.0)
        ok_b = ~free & ~ok_a & (b.fb >= 0.0) & (~engaged | (_gap_a(p, b.x_t, b.th, hs) >= 0.0))
        ok_t = ~free & ~ok_a & ~ok_b & engaged & (t.fa >= 0.0) & (t.fb >= 0.0) & (t.P >= 0.0)
    pick = [ok_a, ok_b, ok_t]
    th = np.select(pick, [a.th, b.th, t.th], p.tilt)
    x_t = np.select(pick, [a.x_t, b.x_t, t.x_t], p.offset)
    fx, m = _wrench(p, x_t, th)
    phase = np.select([free] + pick, [FREE, ONE_A, ONE_B, TWO], JAM).astype(np.int8)
    return QuasiStatic(np.select(pick, [a.P, b.P, t.P], 0.0), phase, th, x_t, fx, m)


@dataclass(frozen=True)
class SynthNoise:
    mu_rel: float = 0.02
    force_n: float = 0.05
    depth_m: float = 2.0e-6
    rate_hz: float = 100.0


@dataclass(frozen=True)
class SynthConfig:
    trials_per_condition: int = 10
    mu: float = DEFAULT_MU
    geometry: R1Geometry = field(default_factory=R1Geometry)
    holder: R1Holder = field(default_factory=R1Holder)
    proto: R1Protocol = field(default_factory=R1Protocol)
    noise: SynthNoise = field(default_factory=SynthNoise)
    dataset_id: str = "synthetic-v1"


def _stop_tail(t, z, P, travel, stop_at: int, k_z: float, proto: R1Protocol):
    """After a jam (or the force limit) the peg stays put and the holder compresses at k_z·v until f_stop, where the
    stage stops; the trial ends after a dwell."""
    j = max(stop_at - 1, 0)
    ramp = P[j] + k_z * (travel - travel[j])
    after = np.arange(len(t)) >= stop_at
    P = np.where(after, np.minimum(ramp, proto.f_stop), P)
    z = np.where(after, z[j], z)
    hit = np.flatnonzero(after & (ramp >= proto.f_stop))
    end = len(t) if hit.size == 0 else min(len(t), int(hit[0]) + int(round(proto.t_dwell / (t[1] - t[0]))) + 1)
    return t[:end], z[:end], P[:end]


def trial_series(p: WhitneyPair, cfg: SynthConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """One trial at the protocol speed: ``t``, ``depth`` (tip) and ``force`` channels with sample noise (SI)."""
    proto, noise = cfg.proto, cfg.noise
    t = np.arange(0.0, proto.duration(proto.speed), 1.0 / noise.rate_hz)
    travel = np.clip(proto.speed * (t - proto.t_settle), 0.0, proto.stroke)
    z = travel - proto.approach
    P, phase = quasi_static_force(p, z)
    stop = np.flatnonzero((phase == JAM) | (P >= proto.f_stop))
    if stop.size:
        t, z, P = _stop_tail(t, z, P, travel, int(stop[0]), cfg.holder.k_z, proto)
    return {"t": t, "depth": z + rng.normal(0.0, noise.depth_m, t.shape),
            "force": P + rng.normal(0.0, noise.force_n, t.shape)}


def _trial(cfg: SynthConfig, cond: Condition, k: int, rng: np.random.Generator) -> RigTrial:
    g, hd = cfg.geometry, cfg.holder
    tilt = abs(cond.params["tilt"] + rng.normal(0.0, hd.tilt_sigma))
    offset = float(np.clip(cond.params["offset"] + rng.normal(0.0, hd.offset_sigma), 0.0, 0.5 * g.clearance))
    mu = cfg.mu * (1.0 + rng.normal(0.0, cfg.noise.mu_rel))
    pair = WhitneyPair.build(g, hd, tilt, offset, cond.params["chamfer"], mu)
    series = trial_series(pair, cfg, rng)
    scalars = insertion_metrics(series["depth"], series["force"], cfg.proto).scalars()
    split = "calib" if k < cfg.trials_per_condition // 2 else "valid"
    return RigTrial(cond.id, k, split, series, scalars)


def r1_card(cfg: SynthConfig) -> RigCard:
    cell = {"u95": 0.05, "what": "20 kg HX711 cell at 1 kHz, M1-mass calibrated (synthetic)"}
    inst = {"sliding_force": dict(cell), "peak_force": dict(cell),
            "peak_depth": {"u95": 1.0e-5, "what": "lead-screw stage encoder, 0.01 mm"},
            "load_cell": {"range_n": 196.1, "rate_hz": 1000.0, "u95_n": 0.05, "calibrated": "synthetic"}}
    return RigCard(LAW_ID, cfg.dataset_id, "synthetic/1", cfg.geometry.objects(cfg.mu), inst,
                   cfg.holder.card(cfg.proto), {"temperature_c": 21.0, "humidity_pct": 45.0})


def _sha(card: RigCard, conds: tuple[Condition, ...], trials: list[RigTrial]) -> str:
    h = hashlib.sha256(json.dumps({"card": asdict(card), "conditions": [(c.id, dict(c.params)) for c in conds]},
                                  sort_keys=True, default=str).encode())
    for t in trials:
        h.update(json.dumps([t.condition_id, t.index, t.split, dict(t.scalars)], sort_keys=True).encode())
        for name in sorted(t.series):
            h.update(name.encode() + np.ascontiguousarray(t.series[name], np.float64).tobytes())
    return h.hexdigest()


def r1_dataset(seed: int = 0, quick: bool = False, cfg: SynthConfig = SynthConfig()) -> RigDataset:
    """The synthetic R1 dataset (full grid, or the quick 2-condition set); deterministic in ``seed``."""
    if cfg.trials_per_condition < 2:
        raise ValueError("need >= 2 trials per condition (calib and valid halves)")
    conds = r1_conditions(quick, cfg.geometry)
    trials = [_trial(cfg, c, k, np.random.default_rng([seed, ci, k]))
              for ci, c in enumerate(conds) for k in range(cfg.trials_per_condition)]
    card = r1_card(cfg)
    log.info("synthetic R1 dataset: %d conditions x %d trials (seed %d, quick=%s)", len(conds),
             cfg.trials_per_condition, seed, quick)
    return RigDataset(card, conds, tuple(trials), _sha(card, conds, trials))


__all__ = ["WhitneyPair", "QuasiStatic", "SynthNoise", "SynthConfig", "quasi_static", "quasi_static_force",
           "trial_series", "r1_card",
           "r1_dataset", "FREE", "ONE_A", "ONE_B", "TWO", "JAM"]
