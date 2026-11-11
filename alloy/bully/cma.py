# SPDX-License-Identifier: Apache-2.0
"""Batched CMA-ES in numpy for the B1 search (SPEC §3.11, bench §4.7).

One candidate per simulated world: ``ask`` returns the whole population, the caller evaluates it in one batched
episode and passes the fitness back to ``tell``. The search space is the unit box [0, 1]^n (normalised action
knots); candidates are repaired by clipping and the squared repair distance is added to the fitness, so the mean
stays inside the box (Hansen 2016, "The CMA Evolution Strategy: A Tutorial", arXiv:1604.00772, App. B and the
boundary handling of §B.3). Minimisation; the bully passes ``-badness``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from alloy.core.errors import BullyError

log = logging.getLogger("alloy.bully")

_REPAIR_WEIGHT = 1.0  # fitness penalty per unit squared distance outside the box


@dataclass(frozen=True)
class CMAParams:
    """Strategy constants derived from the dimension ``n`` and population ``lam`` (Hansen 2016, Table 1)."""

    n: int
    lam: int
    mu: int
    weights: np.ndarray  # [mu], positive, sum 1
    mu_eff: float
    c_sigma: float
    d_sigma: float
    c_c: float
    c_1: float
    c_mu: float
    chi_n: float

    @classmethod
    def default(cls, n: int, lam: int) -> CMAParams:
        if n < 1 or lam < 2:
            raise BullyError(f"CMA-ES needs n >= 1 and lam >= 2, got n={n}, lam={lam}")
        mu = lam // 2
        raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = raw / raw.sum()
        mu_eff = float(1.0 / np.sum(weights**2))
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1.0 - c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        return cls(n, lam, mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, float(chi_n))


@dataclass
class CMAState:
    """Mutable search state; arrays are float64."""

    mean: np.ndarray
    sigma: float
    cov: np.ndarray
    p_sigma: np.ndarray
    p_c: np.ndarray
    generation: int = 0
    best_x: np.ndarray | None = None
    best_f: float = np.inf
    history: list[float] = field(default_factory=list)  # best fitness per generation


class BatchedCMAES:
    """CMA-ES over the unit box with population = world count."""

    def __init__(self, mean0: np.ndarray, sigma0: float, lam: int, seed: int = 0) -> None:
        mean0 = np.clip(np.asarray(mean0, dtype=np.float64).ravel(), 0.0, 1.0)
        if sigma0 <= 0.0:
            raise BullyError(f"sigma0 must be positive, got {sigma0}")
        n = mean0.size
        self.params = CMAParams.default(n, lam)
        self.state = CMAState(mean0, float(sigma0), np.eye(n), np.zeros(n), np.zeros(n))
        self._rng = np.random.default_rng(seed)
        self._last_y: np.ndarray | None = None
        self._last_raw: np.ndarray | None = None
        self._update_eigen()

    def _update_eigen(self) -> None:
        cov = self.state.cov
        cov = np.triu(cov) + np.triu(cov, 1).T  # enforce symmetry
        evals, evecs = np.linalg.eigh(cov)
        evals = np.maximum(evals, 1e-20)
        self._b = evecs
        self._d = np.sqrt(evals)
        self._inv_sqrt = evecs @ np.diag(1.0 / self._d) @ evecs.T
        self.state.cov = cov

    def ask(self) -> np.ndarray:
        """Return the population [lam, n] repaired into [0, 1]; the raw samples are kept for ``tell``."""
        p = self.params
        z = self._rng.standard_normal((p.lam, p.n))
        y = (z * self._d) @ self._b.T  # rows: B·D·z
        raw = self.state.mean + self.state.sigma * y
        self._last_y, self._last_raw = y, raw
        return np.clip(raw, 0.0, 1.0)

    def tell(self, fitness: np.ndarray) -> None:
        """Update from the fitness of the last ``ask`` population (lower is better, NaN counts as +inf)."""
        if self._last_y is None or self._last_raw is None:
            raise BullyError("tell() called before ask()")
        p, s = self.params, self.state
        f = np.where(np.isfinite(fitness), np.asarray(fitness, dtype=np.float64), np.inf)
        repaired = np.clip(self._last_raw, 0.0, 1.0)
        f = f + _REPAIR_WEIGHT * np.sum((self._last_raw - repaired) ** 2, axis=1)
        order = np.argsort(f, kind="stable")
        self._record_best(repaired[order[0]], float(f[order[0]]))
        y_sel = self._last_y[order[: p.mu]]
        y_w = p.weights @ y_sel
        s.mean = s.mean + s.sigma * y_w
        self._update_paths(y_w)
        self._update_cov(y_sel)
        s.sigma *= float(np.exp((p.c_sigma / p.d_sigma) * (np.linalg.norm(s.p_sigma) / p.chi_n - 1.0)))
        s.generation += 1
        self._update_eigen()
        self._last_y = self._last_raw = None

    def _record_best(self, x: np.ndarray, f: float) -> None:
        s = self.state
        s.history.append(f)
        if f < s.best_f:
            s.best_f, s.best_x = f, x.copy()

    def _update_paths(self, y_w: np.ndarray) -> None:
        p, s = self.params, self.state
        s.p_sigma = (1.0 - p.c_sigma) * s.p_sigma + np.sqrt(p.c_sigma * (2.0 - p.c_sigma) * p.mu_eff) * (
            self._inv_sqrt @ y_w
        )
        denom = np.sqrt(1.0 - (1.0 - p.c_sigma) ** (2 * (s.generation + 1)))
        self._h_sigma = float(np.linalg.norm(s.p_sigma) / denom < (1.4 + 2.0 / (p.n + 1.0)) * p.chi_n)
        s.p_c = (1.0 - p.c_c) * s.p_c + self._h_sigma * np.sqrt(p.c_c * (2.0 - p.c_c) * p.mu_eff) * y_w

    def _update_cov(self, y_sel: np.ndarray) -> None:
        p, s = self.params, self.state
        delta_h = (1.0 - self._h_sigma) * p.c_c * (2.0 - p.c_c)
        rank_one = np.outer(s.p_c, s.p_c) + delta_h * s.cov
        rank_mu = (y_sel * p.weights[:, None]).T @ y_sel
        s.cov = (1.0 - p.c_1 - p.c_mu) * s.cov + p.c_1 * rank_one + p.c_mu * rank_mu
