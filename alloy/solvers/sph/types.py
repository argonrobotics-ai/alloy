# SPDX-License-Identifier: Apache-2.0
"""SPH report type (SPEC §3.8; ``SolverDFSPH`` is WP9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class SPHReport:                                   # host copy, [W] arrays
    density_err: np.ndarray
    divergence_err: np.ndarray
    iters_density: np.ndarray
    iters_divergence: np.ndarray
    mass_active: np.ndarray                        # kg
    mass_in: np.ndarray                            # kg
    mass_out: np.ndarray                           # kg
    escaped: np.ndarray
    neighbor_overflow: np.ndarray
    max_speed: np.ndarray
    unconverged: np.ndarray | None = None          # [W] bool, hit max_iterations this step (solvers §4.2)
    tension: str = "none"                          # Config.surface_tension (SCR-WP9-7)
    tension_qualitative: bool = False              # akinci2013: no gated law
    tension_rh_range: tuple[float, float] | None = None  # csf: drop R/h range of the laws; below it, qualitative
    capillary_dt_violations: int = 0               # eager steps run with dt above the capillary limit
    density_tol: np.ndarray | None = None          # [W] float, effective density tolerance (SCR-WP9-8)
