# SPDX-License-Identifier: Apache-2.0
"""Gauss-Newton (Levenberg-Marquardt) refinement with batched central-difference Jacobians (`method="hybrid"`).

The objective is a sum of squared normalised residuals r (objective.py), so near the optimum a Gauss-Newton step
-(J^T J + lambda diag(J^T J))^-1 J^T r converges much faster than CMA-ES. J is the central-difference Jacobian of r
in the unit box, from one batch of 2n + 1 candidates per iteration (the centre and +-h along each axis), so it works
on every solver (Newton's autodiff through rigid contact fails its finite-difference check at this pin, grad.py).
Prior z terms are extra residual rows. The step is accepted if the objective drops (lambda /= 3), else lambda *= 4.

From the final Jacobian it also gives a data-driven 95 % interval: with residuals in law units over sigma_t =
U95_t / 1.96 (the calib mean's uncertainty, alloy.bench.compare.summarize), cov_u = (J_s^T J_s)^-1 and the interval is
u +- 1.96 sqrt(cov_u_jj), mapped to SI. It is reported (`report["gn_unc95"]`); Q.unc stays the population spread
(SPEC §3.12).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from alloy.bench import compare as C
from alloy.fit.batch import BatchResult
from alloy.fit.objective import Score
from alloy.fit.params import ParamSpace
from alloy.fit.problem import Target
from alloy.fit.search import Budget, Evaluate, Generation, SearchOutcome

log = logging.getLogger("alloy.fit")
FD_STEP = 1e-3            # unit-box central-difference step
LM_LAMBDA0 = 1e-3
MIN_STEP = 1e-7


@dataclass(frozen=True)
class Linearisation:
    u: np.ndarray             # centre, unit box
    f: float                  # objective at the centre
    r: np.ndarray             # residual vector at the centre (targets, then prior rows)
    jac: np.ndarray           # d r / d u, [m, n]
    score: Score


def residual_vector(score: Score, keys: Sequence[str], prior_z: np.ndarray) -> np.ndarray:
    r = np.array([score.residuals.get(k, math.nan) for k in keys], np.float64)
    return np.concatenate([r, prior_z])


def _prior_z(space: ParamSpace, x: np.ndarray) -> np.ndarray:
    s = space.scale_of(np.atleast_2d(x))
    rows = [(s[:, j] - p.prior[0]) / p.prior[1] for j, p in enumerate(space.params) if p.prior is not None]
    return np.stack(rows, axis=1) if rows else np.zeros((s.shape[0], 0))


def stencil(u: np.ndarray, h: float = FD_STEP) -> np.ndarray:
    """Centre plus +-h along each axis (reflected inside [0, 1] at the edges), shape [2n + 1, n]."""
    n = u.size
    rows = [u.copy()]
    for j in range(n):
        step = np.zeros(n)
        step[j] = h
        rows += [u + step, u - step]
    return np.clip(np.array(rows), 0.0, 1.0)


def linearise(space: ParamSpace, evaluate: Evaluate, u: np.ndarray, keys: Sequence[str]) -> tuple[Linearisation,
                                                                                                    BatchResult]:
    pts = stencil(u)
    batch = evaluate(space.from_unit(pts))
    z = _prior_z(space, batch.x)
    rs = np.array([residual_vector(sc, keys, z[i]) for i, sc in enumerate(batch.scores)])
    n = u.size
    jac = np.zeros((rs.shape[1], n))
    for j in range(n):
        du = pts[1 + 2 * j, j] - pts[2 + 2 * j, j]
        jac[:, j] = (rs[1 + 2 * j] - rs[2 + 2 * j]) / du if du > 0 else 0.0
    return Linearisation(pts[0], float(batch.objective[0]), rs[0], jac, batch.scores[0]), batch


def lm_step(lin: Linearisation, lam: float) -> np.ndarray:
    """Levenberg-Marquardt step in the unit box (NaN residual rows are dropped)."""
    ok = np.isfinite(lin.r) & np.all(np.isfinite(lin.jac), axis=1)
    jac, r = lin.jac[ok], lin.r[ok]
    jtj = jac.T @ jac
    damp = lam * np.diag(np.maximum(np.diag(jtj), 1e-12))
    try:
        return -np.linalg.solve(jtj + damp, jac.T @ r)
    except np.linalg.LinAlgError:
        return np.zeros_like(lin.u)


def unc95(space: ParamSpace, lin: Linearisation, targets: Sequence[Target]) -> dict[str, tuple[float, float]]:
    """95 % interval per leaf from the Jacobian: scalar-mean residuals rescaled to law units over each target's
    calib-mean uncertainty (U95 / 1.96), plus the prior rows (already z units), so the interval is the posterior."""
    rows = []
    for i, t in enumerate(targets):
        if t.stat != "mean":
            continue                                           # other residuals are not mean errors
        q = C.summarize(t.key, "1", t.real, t.u_instr)
        sigma = q.u95 / 1.96 if q.u95 > 0 else math.nan
        rows.append(lin.jac[i] * t.bound / sigma)          # d(sim, law units)/du over sigma_t
    rows += list(lin.jac[len(targets):])                   # prior z rows
    js = np.array(rows) if rows else np.zeros((0, lin.u.size))
    js = js[np.all(np.isfinite(js), axis=1)]
    try:
        cov = np.linalg.inv(js.T @ js)
    except np.linalg.LinAlgError:
        return {}
    half = 1.96 * np.sqrt(np.maximum(np.diag(cov), 0.0))
    lo, hi = space.from_unit(lin.u - half), space.from_unit(lin.u + half)
    return {leaf: (float(lo[j]), float(hi[j])) for j, leaf in enumerate(space.leaves)}


def _record(out: SearchOutcome, batch: BatchResult, it: int, step: float, space: ParamSpace,
            u: np.ndarray) -> None:
    out.improve(batch, out.best_run)
    out.evaluations += len(batch.x)
    out.worlds += batch.worlds
    finite = batch.objective[np.isfinite(batch.objective)]
    out.history.append(Generation(0, it, len(batch.x), float(np.min(batch.objective)),
                                  float(np.median(finite)) if finite.size else math.nan, step,
                                  tuple(float(v) for v in space.from_unit(u)), batch.wall_s, phase="refine"))


def refine(space: ParamSpace, evaluate: Evaluate, out: SearchOutcome, targets: Sequence[Target], steps: int,
           budget: Budget) -> SearchOutcome:
    """Refine `out.best_x` in place: one stencil batch per iteration, centred on the trial point, so a step costs
    one batch whether it is accepted or not; adds `gn_unc95` to `out.extra` (history phase "refine")."""
    keys = [t.key for t in targets]
    lin, batch = linearise(space, evaluate, space.to_unit(out.best_x[None, :])[0], keys)
    _record(out, batch, 0, 0.0, space, lin.u)
    lam, reason = LM_LAMBDA0, f"refine: {steps} iterations"
    for it in range(1, steps + 1):
        step = lm_step(lin, lam)
        size = float(np.max(np.abs(step)))
        if size < MIN_STEP:
            reason = f"refine: converged at iteration {it}"
            break
        if not budget.allows(batch.wall_s):
            reason = "refine: wall_budget"
            break
        trial, batch = linearise(space, evaluate, np.clip(lin.u + step, 0.0, 1.0), keys)
        _record(out, batch, it, size, space, trial.u)
        if trial.f < lin.f:
            lin, lam = trial, max(lam / 3.0, 1e-9)
        else:
            lam *= 4.0
    out.stop.append(reason)
    out.extra["gn_unc95"] = unc95(space, lin, targets)
    out.extra["gn_objective"] = lin.f
    out.method = "hybrid"
    out.wall_s = budget.elapsed()
    return out
__all__ = ["refine", "linearise", "lm_step", "stencil", "unc95", "residual_vector", "Linearisation"]
