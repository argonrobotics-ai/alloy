# SPDX-License-Identifier: Apache-2.0
"""What a per-world exchange clock would save: P0 of gap #10b ([internal note] notes/wp4-perworld-clock/DESIGN.md).

A Scheduler steps every world at one dt_x. The clock takes the finest bound over all worlds' proxies, and a re-run
for one world's escalation re-runs every world. This census changes nothing. Once per attempt it reads what the clock
already knows, world by world, and counts:

- ``j_world``: the grid level each world would pick from its own bound. With the contact gate (lever #2), that is
  the minimum over its own armed proxies (``GatedEndpoints.bound_s``) and the always-on pairs. Without it, it is each
  pair's stiffness and damping bound at that world's lightest proxy (``endpoints.clock_masses``). The contact band's
  bound (R2-E3) and the ladder's sticky floor (``ClockController.j_floor``) stay shared: they bind no tighter per
  world than they do now.
- ``world_exchanges``: worlds x exchanges the attempts ran (W 2^j each);
- ``per_world_exchanges``: what each world's own rung would have run (sum_w 2^j_w, the attempt's own refinement on
  top: an onset or ladder re-run's extra halvings stay);
- ``rerun_kept_exchanges``: exchanges re-run for worlds whose result the re-run did not keep (a per-world rollback's
  kept worlds; every non-escalated world of a whole re-run).
Reported in each CouplingReport's ``notes["census"]`` and over the run by ``clock_census()``.

The census reads the bound, band and armed set the attempt's clock used (snapshotted in ``choose_clock``), with the
clock's own build-time proxy masses (as ``Scheduler._pair_mass``). The minimum over its per-world bounds must equal
the clock's shared bound. If it does not, ``drift`` is set and a warning logs once: the census and the clock no longer
share one bound (WP10's review). Each attempt is counted once, however often ``report`` runs.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from alloy.coupling.clocks import damping_bound, interface_bound
from alloy.coupling.endpoints import clock_masses
from alloy.coupling.world_clock_need import AttemptLevels, ClockGrid, per_world_rungs

log = logging.getLogger("alloy.coupling")

PARKED_RUNG = -1          # ``j_world`` of a parked world: held at the pack's clock, it states no need (parked.py)


class ClockCensusMixin:
    """For the Scheduler (before SchedulerMixin in its bases): the census of the module docstring."""

    _census_totals: dict[str, int] | None = None
    _census_kept: int = 0                      # worlds the next attempt re-runs without keeping (on_rollback)
    _census_mass: dict[tuple[str, str], np.ndarray] | None = None
    _census_attempt: int = 0                   # attempts begun (choose_clock)
    _census_counted: int = -1                  # the attempt the totals last counted
    _census_used: tuple[float, float, np.ndarray | None] | None = None   # the attempt's shared bound, band, armed
    _census_band: tuple[np.ndarray, np.ndarray] | None = None   # the attempt's per-world band (shared rule, need)
    _census_last: dict[str, Any] | None = None
    _census_warned: bool = False

    def choose_clock(self, control_dt: float, last: Any) -> tuple[float, dict[str, int]]:
        out = super().choose_clock(control_dt, last)
        gate = getattr(self, "_gate", None)
        armed = None if gate is None else np.asarray(gate.armed_host, dtype=bool).copy()
        self._census_used = (float(self._interface_bound()), float(self._band_bound), armed)
        self._census_band = self._world_band(float(self._band_bound))
        self._census_attempt += 1
        return out

    def clock_census(self) -> dict[str, int]:
        """Totals over the run: world_exchanges, per_world_exchanges, rerun_kept_exchanges, attempts."""
        return dict(self._census_totals or {})

    def on_rollback(self, world_mask: Any = None) -> None:
        w = int(self._watchdog.world_count)
        if world_mask is not None:
            self._census_kept = int(w - np.count_nonzero(np.asarray(world_mask)))
        else:
            report = getattr(self, "_last_report", None)
            ws = np.asarray(getattr(report, "watch_state", ()))
            self._census_kept = int(w - np.count_nonzero(ws == 2)) if ws.size == w else 0
        super().on_rollback(world_mask)

    def report(self) -> Any:
        report = super().report()
        census = self._census(report)
        if census is not None and report.notes is not None:
            report.notes["census"] = census
        return report

    def _census(self, report: Any) -> dict[str, Any] | None:
        dt_x, control_dt = float(report.dt_x), float(getattr(self, "_control_dt", 0.0) or 0.0)
        if not (dt_x > 0.0 and control_dt > 0.0):
            return None
        if self._census_counted == self._census_attempt and self._census_last is not None:
            return self._census_last                              # this attempt is counted already
        w = int(self._watchdog.world_count)
        clock = self._alloy_policy.clock
        shared, band, armed = self._census_used or (math.nan, float(self._band_bound), None)
        j_attempt = int(round(math.log2(control_dt / dt_x)))
        extra = max(j_attempt - int(getattr(self._clock, "j", j_attempt)), 0)   # onset / ladder halvings this attempt
        floor = int(getattr(self._clock, "j_floor", 0))
        own = np.minimum(self._world_bounds(w, armed), band)
        parked = getattr(self, "_parked_mask", lambda: None)()          # worlds the clock's bound leaves out
        parked = np.zeros(w, dtype=bool) if parked is None else parked
        low = float(np.min(own[~parked], initial=math.inf))
        drift = bool(math.isfinite(shared) and not math.isclose(low, shared, rel_tol=1e-9, abs_tol=0.0)
                     and not (math.isinf(shared) and math.isinf(low)))
        if drift and not self._census_warned:
            self._census_warned = True
            log.warning("clock census: min over worlds %.6g s != the clock's shared bound %.6g s (the census no "
                        "longer reads the clock's bound)", low, shared)
        rows = self._census_band is not None and not drift       # each world's own band need, else the shared scalar
        need = per_world_rungs(self._world_bounds(w, armed), self._census_band[1] if rows else band,
                               ClockGrid(control_dt, clock.dt_x_max, clock.j_max),
                               AttemptLevels(floor=floor, extra=extra, j_attempt=j_attempt))
        j_w = np.where(parked, j_attempt, need.j)                # parked: counted as it ran
        n = 2 ** j_attempt
        kept, self._census_kept = self._census_kept, 0
        out = {"j_world": np.where(parked, PARKED_RUNG, j_w).tolist(), "world_exchanges": w * n,
               "per_world_exchanges": int(np.sum(2 ** j_w)), "rerun_kept_exchanges": kept * n, "drift": drift,
               "term_world": need.terms(), "band_rule": "world rows" if rows else "shared"}
        totals = self._census_totals or {"world_exchanges": 0, "per_world_exchanges": 0, "rerun_kept_exchanges": 0,
                                         "attempts": 0}
        for k in ("world_exchanges", "per_world_exchanges", "rerun_kept_exchanges"):
            totals[k] += out[k]
        totals["attempts"] += 1
        self._census_totals, self._census_counted, self._census_last = totals, self._census_attempt, out
        return out

    def keep_world_rows(self, on: bool = True) -> None:
        """Opt in to (or out of) the contact band's per-world rows: the census's ``j_world`` then carries each world's
        own band need (``band_rule``: "world rows"), not the shared scalar. Off by default: no extra kernel work."""
        band = getattr(self, "_band", None)
        if band is not None and hasattr(band, "keep_world_rows"):
            band.keep_world_rows(on)

    def _world_band(self, shared: float) -> tuple[np.ndarray, np.ndarray] | None:
        """Before an attempt: the per-world band bounds behind the clock's ``shared`` band bound, as (the shared rule
        world by world, each world's own need); None when the band keeps no per-world rows, or when their minimum is
        not the shared bound (a bound adopted from elsewhere, e.g. the fine re-run pack: the scalar then stands for
        every world). The rows exist only once asked for (``keep_world_rows``); the resize is on the host here, outside
        any captured period."""
        band = getattr(self, "_band", None)
        if band is None or not hasattr(band, "ensure_worlds"):
            return None
        band.ensure_worlds(int(self._watchdog.world_count))      # rows only when asked (band.keep_world_rows)
        rule, need = band.world_bounds(need=False), band.world_bounds(need=True)
        if rule is None or float(rule.min()) != shared:
            return None
        return rule, need

    def _world_bounds(self, w: int, armed: np.ndarray | None = None) -> np.ndarray:
        """Per world [s]: the gated bound over its armed proxies (``armed``: the attempt's), else each pair's bound at
        its lightest proxy."""
        gate = getattr(self, "_gate", None)
        if gate is not None:
            ends = gate.ends
            armed = np.asarray(gate.armed_host if armed is None else armed, dtype=bool)
            out = np.full(w, float(ends.fixed_bound_s))
            np.minimum.at(out, np.asarray(ends.world, dtype=np.int64)[armed], np.asarray(ends.bound_s)[armed])
            return out
        clock = self._alloy_policy.clock
        masses = self._census_masses(w)
        out = np.full(w, math.inf)
        for pair in self._alloy_pairs:
            m = masses.get((pair.source, pair.destination))
            if m is None:
                continue
            k_c = self._pair_k_c(pair)
            c_d = self._pair_c_d(pair) if clock.damping_bound else None
            vals, inv = np.unique(m, return_inverse=True)        # distinct masses: the clocks.py bounds once each
            b = np.array([min(math.inf if k_c is None else interface_bound(k_c, float(v), clock.safety,
                                                                            clock.omega_dt_max),
                              math.inf if c_d is None else damping_bound(c_d, float(v), clock.safety,
                                                                         clock.omega_dt_max)) for v in vals])
            out = np.minimum(out, b[inv.reshape(-1)])
        return out

    def _census_masses(self, w: int) -> dict[tuple[str, str], np.ndarray]:
        """Per pair, per world: its lightest proxy's clock mass at build, as the clock's own (``Scheduler._pair_mass``,
        computed once at build)."""
        if self._census_mass is None:
            out: dict[tuple[str, str], np.ndarray] = {}
            for es in self._sets:
                m, world = clock_masses(es), es.world.numpy()
                key = (es.src_name, es.dst_name)
                per = out.setdefault(key, np.full(w, math.inf))
                use = m > 0
                np.minimum.at(per, world[use], m[use])
            self._census_mass = out
        return self._census_mass


__all__ = ["PARKED_RUNG", "ClockCensusMixin"]
