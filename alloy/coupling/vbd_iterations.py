# SPDX-License-Identifier: Apache-2.0
"""VBD iterations and Chebyshev rho scaled to the coupled step (speed-now S2 lever, WP10 2026-11-09; opt-in:
``ClockPolicy.vbd_step_iterations``).

WARNING (experimental, not advertised): this lever FAILS the settled-drape law. A towel resting on a still arm slides
off: a Panda at rest with a draped 0.3 m towel, 8 worlds with nudged starts, 2 s at the 78 us exchange step, kept the
towel on the arm in 1 of 8 worlds with this lever (15 iterations, rho 0.123), against 8 of 8 with the defaults (40,
rho 0.9) and with 40 iterations and Chebyshev off. Its rho part fails the law too: rho 0.123 at the full 40 iterations
kept the towel in 5 of 8. Both parts lose the friction hold. Keep it off until a law passes ([internal note] scr/WP10.md,
speed-now S2, 2026-11-09T14:20Z and 15:11Z).

A cloth entry runs the solve its route chose (``alloy.scene.cloth_accuracy``: 40 iterations by default, 60 at "super",
Chebyshev-accelerated with rho 0.9), measured on drapes stepped at ``H_MEASURED`` (16 substeps per 10 ms: 0.625 ms;
``cloth_accuracy.ACCURACY``, docs/notes/wp2-results.md SCR-WP2-13). Coupled to a robot, the entry steps at the
exchange clock instead, ``h = dt_x / substeps``: 78 us at j = 7, and every one of those steps runs the full solve
(128 x 40 = 5,120 iterations per 10 ms period instead of 16 x 40 = 640). Per vertex VBD solves m/h^2 + K, so a sweep
contracts the smooth error by about e/(1 + e), e = K h^2 / m: e falls 64x from 0.625 ms to 78 us.

- Rho (``step_rho``, WP11's model): Chebyshev's rho estimates that spectral radius. Fixed at 0.9 (e = 9) it is far above
  the true 0.12 at 78 us and over-relaxes: on a Panda turning with a draped towel (robot_cloth, 64 worlds, L4) the
  towel's particles reached 8-14 m/s at 40 iterations and 42-44 m/s at 15, and the contact band took the clock to
  256-1024 exchanges per period. With rho = e/(1 + e), e = e_route (h / H_MEASURED)^2 (0.123 at 78 us), 10 iterations
  kept 128 exchanges in every period, particles <= 5.6 m/s and no INVALID world, at 9.0 s per period against 23.7.
  It is the route's rho at or above H_MEASURED and never above it. It removes the trips but not the cost (17.8 s per
  period at 40 iterations), and it fails the settled-drape law (WARNING above).
- Count (``step_iterations``): the route's count at or above H_MEASURED; below it n_route sqrt(h / H_MEASURED), rounded
  up, never under ``FLOOR`` (VBD's own count, ``cloth_accuracy.FAST_ITERATIONS``) scaled by n_route / 40 (so "super"
  keeps 1.5x the default's count: WP2) and never over n_route. 40 gives 15 at 78 us (j = 7) and 10 from 39 us down;
  60 gives 22 and 15.

Only entries whose count is above FLOOR change (cloth at "default" or "super" and soft solids sharing their entry);
count and rho are pure functions of (dt_x, substeps), which the graph key already carries, so a captured exchange
always replays what it was captured with (rho goes through the solver's public ``set_chebyshev_rho``, validated, and
only on entries built with Chebyshev on). ``restore`` puts back the entry's own count and rho.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from alloy.scene.cloth_accuracy import DEFAULT_ITERATIONS, FAST_ITERATIONS

log = logging.getLogger("alloy.coupling")

FLOOR = FAST_ITERATIONS
H_MEASURED = 0.01 / 16              # [s] the step cloth_accuracy's counts and rho were measured at

__all__ = ["FLOOR", "H_MEASURED", "VbdIterationScaler", "VbdStepIterationsMixin", "step_iterations", "step_rho"]


def step_iterations(n_route: int, h: float, h_route: float = H_MEASURED, floor: int = FLOOR) -> int:
    """Iterations for a VBD step of ``h`` seconds whose route chose ``n_route`` for steps of ``h_route`` (module
    docstring)."""
    if n_route <= floor or not (h > 0.0 and h_route > 0.0) or h >= h_route:
        return int(n_route)
    return int(min(n_route, max(floor, math.ceil(n_route * math.sqrt(h / h_route) - 1e-9))))


def step_rho(rho_route: float, h: float, h_route: float = H_MEASURED) -> float:
    """Chebyshev rho for a VBD step of ``h`` seconds whose route chose ``rho_route`` for steps of ``h_route``:
    e/(1 + e) with e = e_route (h / h_route)^2, e_route = rho_route / (1 - rho_route) (module docstring)."""
    if not (h > 0.0 and h_route > 0.0) or h >= h_route:
        return float(rho_route)
    e = rho_route / (1.0 - rho_route) * (h / h_route) ** 2
    return float(min(rho_route, e / (1.0 + e)))


@dataclass
class _Scaled:
    solver: Any
    n_route: int
    floor: int
    rho_route: float | None         # Chebyshev rho as built (None: off, or a solver without the setter)


class VbdIterationScaler:
    """Per coupled entry: the VBD solver, its route count and Chebyshev config; ``apply`` sets both for an attempt."""

    def __init__(self, entries: Mapping[str, Any]) -> None:
        self.entries: dict[str, _Scaled] = {}
        for name, entry in entries.items():
            solver = getattr(entry, "solver", None)
            n = getattr(solver, "iterations", None)
            if not _is_vbd(solver) or not isinstance(n, int) or n <= FLOOR:
                continue
            rho = getattr(solver, "chebyshev_rho", None) if hasattr(solver, "set_chebyshev_rho") else None
            floor = max(FLOOR, math.ceil(FLOOR * n / DEFAULT_ITERATIONS - 1e-9))   # "super" stays above the default
            self.entries[name] = _Scaled(solver, n, floor, None if rho is None else float(rho))
        self.last: dict[str, tuple[int, float | None]] = {}

    def apply(self, dt_x: float, substeps: Mapping[str, int]) -> dict[str, tuple[int, float | None]]:
        """Set each scaled entry's iterations and rho for steps of dt_x / substeps[name]; returns {name: (n, rho)}."""
        out = {}
        for name, s in self.entries.items():
            h = float(dt_x) / max(int(substeps.get(name, 1)), 1)
            s.solver.iterations = n = step_iterations(s.n_route, h, floor=s.floor)
            rho = None if s.rho_route is None else step_rho(s.rho_route, h)
            if rho is not None and s.solver.chebyshev_rho != rho:
                s.solver.set_chebyshev_rho(rho)
            out[name] = (n, rho)
        if out != self.last:
            log.debug("VBD iterations and rho at dt_x %.3g s: %s", dt_x, out)
        self.last = out
        return out

    def restore(self) -> None:
        for s in self.entries.values():
            s.solver.iterations = s.n_route
            if s.rho_route is not None:
                s.solver.set_chebyshev_rho(s.rho_route)


def _is_vbd(solver: Any) -> bool:
    from alloy._vendor.newton.solvers import SolverVBD  # noqa: PLC0415 - the coupling package imports without it

    return isinstance(solver, SolverVBD)


class VbdStepIterationsMixin:
    """For the Scheduler (before SchedulerMixin in its bases): after the clock picks an attempt's dt_x and substeps,
    set the scaled VBD entries' iterations and rho (``ClockPolicy.vbd_step_iterations``)."""

    _vbd_scaler: VbdIterationScaler | None = None

    def choose_clock(self, control_dt: float, last: Any) -> tuple[float, dict[str, int]]:
        dt_x, substeps = super().choose_clock(control_dt, last)
        if self._alloy_policy.clock.vbd_step_iterations:
            if self._vbd_scaler is None:
                self._vbd_scaler = VbdIterationScaler(self._entries)
            self._vbd_scaler.apply(dt_x, substeps)
        return dt_x, substeps

    def graph_key(self) -> tuple:
        """What a capture bakes in: SchedulerMixin's, plus the scaled entries' (iterations, rho) when the lever is on.
        Both are pure functions of (dt_x, substeps) already; keyed as well, so a captured Chebyshev schedule (host w_n
        weights) can never replay under another rho (WP2's condition)."""
        base = super().graph_key()
        if self._vbd_scaler is None or not self._vbd_scaler.last:
            return base
        return base + (("vbd", tuple(sorted(self._vbd_scaler.last.items()))),)

    def vbd_iterations(self) -> dict[str, tuple[int, float | None]]:
        """The scaled entries' (iterations, Chebyshev rho) for the current attempt (empty when off or nothing scales)."""
        return dict(self._vbd_scaler.last) if self._vbd_scaler is not None else {}
