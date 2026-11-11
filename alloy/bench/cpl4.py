# SPDX-License-Identifier: Apache-2.0
"""CPL-4 under the amended protocol (§5.2 CPL-4, §5.1 SB-8; coordinator ruling 2026-11-04 20:46Z, [internal note];
docs/notes/wp0-scr-log.md "SB-8 / CPL-4 protocol"): box on cloth with the box raised 0-3.5 mm in 0.5 mm steps, one
world per height, at least 60 s of sim. Per height and bounded quantity: the per-sim-minute rate over the whole run (no
extrapolation from a shorter window), the largest increase over every 1 s window and the largest single exchange, each
as a fraction of E_scale. The row is the worst height for each bound, with all 8 heights in the detail.

    python -m alloy.bench.cpl4 --scene dissipative|conservative [--seconds 60] [--device cpu] --out <file.json>

- dissipative: μ 0.5, damped; bound on the coupling injection ΔE_int.
- conservative: μ 0, no cloth or contact damping; bounds on ΔE_int and on the passive gain G = E - E0 - W_ext.
- Bounds (unchanged by the amendment): <= 1e-2 of E_scale per sim-minute, never > 5e-2 in any 1 s window.

Exchanges of a rolled-back attempt are dropped, so the time axis and the ledger hold the accepted run only. A run cut
short leaves a checkpoint marked partial; a partial result never has a verdict.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from alloy.bench.provenance import code_versions

log = logging.getLogger("alloy.bench")

Scene = Literal["dissipative", "conservative"]
RATE_BOUND = 1e-2           # of E_scale per sim-minute
WINDOW_BOUND = 5e-2         # of E_scale in any 1 s window
WINDOW_S = 1.0
MIN_SECONDS = 60.0          # the protocol's shortest run
HEIGHTS_M = tuple(0.0005 * k for k in range(8))
QUANTITIES: dict[str, tuple[str, ...]] = {"dissipative": ("interface_work",),
                                          "conservative": ("interface_work", "gain")}


@dataclass(frozen=True)
class SweepSpec:
    """One CPL-4 sweep: scene, sim length, box heights, device."""

    scene: Scene
    seconds: float = MIN_SECONDS
    heights_m: tuple[float, ...] = HEIGHTS_M
    device: str = "cpu"
    box_mass: float = 0.2          # kg (coupling-time scene, §3.2)
    control_dt: float = 0.02       # s, one PeriodRunner period
    checkpoint_s: float = 5.0      # s of sim between partial writes


@dataclass(frozen=True)
class SeriesStats:
    """One quantity at one height, as fractions of E_scale."""

    per_min: float                 # value at the end / E_scale x 60 / t_end
    max_1s: float                  # largest increase over every 1 s window
    max_step: float                # largest single-exchange increase

    def passes(self) -> bool:
        return self.per_min <= RATE_BOUND and self.max_1s <= WINDOW_BOUND


@dataclass(frozen=True)
class HeightRow:
    dz_mm: float
    e_scale: float                 # J, from the run's ledger (D20 as on the tree that ran)
    stats: dict[str, SeriesStats]


@dataclass
class SweepResult:
    spec: SweepSpec
    t_end: float
    rows: list[HeightRow]
    exchanges: int
    rollbacks: int                 # re-runs of a period: watchdog rollbacks plus onset re-runs (`attempts`)
    wall_s: float
    partial: bool
    versions: dict[str, str] = field(default_factory=code_versions)

    def worst(self, quantity: str, bound: Literal["per_min", "max_1s"]) -> tuple[float, float]:
        """(value, dz_mm) of the worst height for one quantity and bound."""
        row = max(self.rows, key=lambda r: getattr(r.stats[quantity], bound))
        return getattr(row.stats[quantity], bound), row.dz_mm

    def verdict(self) -> str:
        """pass only when every height passes every bound over >= 60 s; a partial or short run has none."""
        if self.partial or self.t_end < MIN_SECONDS - 1e-9:
            return "incomplete"
        return "pass" if all(s.passes() for r in self.rows for s in r.stats.values()) else "fail"


def series_stats(t: np.ndarray, v: np.ndarray, scale: float, window: float = WINDOW_S) -> SeriesStats:
    """Rate over the whole run, worst window increase and worst single step of a ledger series that starts at 0.

    The window base is the last sample before t_k - window (0 before the first sample), as in
    tests/coupling/test_energy.py, so each window spans at least `window` seconds.
    """
    j = np.searchsorted(t, t - window)
    base = np.where(j > 0, v[np.maximum(j - 1, 0)], 0.0)
    steps = np.diff(v, prepend=0.0)
    return SeriesStats(per_min=float(v[-1] / scale * 60.0 / t[-1]), max_1s=float(np.max(v - base) / scale),
                       max_step=float(np.max(steps) / scale))


def height_rows(t: np.ndarray, series: dict[str, np.ndarray], e_scale: np.ndarray,
                heights_m: Sequence[float]) -> list[HeightRow]:
    """Per-world stats; `series[q]` is (exchanges, worlds)."""
    return [HeightRow(dz_mm=round(dz * 1e3, 6), e_scale=float(e_scale[w]),
                      stats={q: series_stats(t, s[:, w], float(e_scale[w])) for q, s in series.items()})
            for w, dz in enumerate(heights_m)]


def attempts(runner: Any) -> int:
    """Re-runs of a period so far: the watchdog ladder's rollbacks plus the onset refiner's finer re-runs, which
    PeriodRunner counts apart (`onset_reruns`) since main 61d3b39a; on earlier trees `rollbacks` counts both."""
    return int(runner.rollbacks) + int(getattr(runner, "onset_reruns", 0))


def _ledger_runner(spec: SweepSpec) -> Any:
    """A PeriodRunner on the sweep's model that keeps only the accepted exchanges' ledger values."""
    from alloy.testing.coupling import ClothSceneSpec, PeriodRunner, cloth_scheduler, coupling_box_on_cloth

    scene = (ClothSceneSpec(box_mass=spec.box_mass) if spec.scene == "dissipative"
             else ClothSceneSpec(box_mass=spec.box_mass, contact_mu=0.0, damped=False))
    model = coupling_box_on_cloth(len(spec.heights_m), scene, drop_offsets=list(spec.heights_m), device=spec.device)
    keys = QUANTITIES[spec.scene]

    class LedgerRunner(PeriodRunner):
        def __init__(self) -> None:
            super().__init__(model, cloth_scheduler(model, scene), spec.control_dt, log_exchanges=True)
            self.pending: list[tuple[int, float, dict[str, np.ndarray]]] = []
            self.t: list[float] = []
            self.values: dict[str, list[np.ndarray]] = {k: [] for k in (*keys, "e_scale")}

        def _record(self) -> None:          # every exchange, tagged with the attempt it belongs to
            lg = self.stepper.energy_ledger()
            self.pending.append((attempts(self), self.log.dt.pop(), {k: np.array(lg[k]) for k in self.values}))

        def run_period(self) -> Any:
            report = super().run_period()
            kept = [p for p in self.pending if p[0] == attempts(self)]   # the accepted attempt only
            self.pending.clear()
            now = self.t[-1] if self.t else 0.0
            for _, dt, vals in kept:
                now += dt
                self.t.append(now)
                for k, v in vals.items():
                    self.values[k].append(v)
            return report

    return LedgerRunner()


def _result(spec: SweepSpec, runner: Any, wall_s: float, partial: bool) -> SweepResult:
    t = np.asarray(runner.t)
    scale = np.stack(runner.values["e_scale"])
    if not np.all(scale == scale[0]):
        log.warning("cpl4: E_scale changed during the run (max rel %.3e); stats use the reset value",
                    float(np.max(np.abs(scale / scale[0] - 1.0))))
    series = {k: np.stack(runner.values[k]) for k in QUANTITIES[spec.scene]}
    return SweepResult(spec=spec, t_end=float(t[-1]), rows=height_rows(t, series, scale[0], spec.heights_m),
                       exchanges=len(t), rollbacks=attempts(runner), wall_s=wall_s, partial=partial)


def run_sweep(spec: SweepSpec, out: Path | None = None) -> SweepResult:
    """Run the sweep, writing a partial checkpoint to `out` every `checkpoint_s` of sim, then the final result."""
    runner = _ledger_runner(spec)
    start = time.time()
    total = int(round(spec.seconds / spec.control_dt))
    chunk = max(1, int(round(spec.checkpoint_s / spec.control_dt)))
    done = 0
    while done < total:
        n = min(chunk, total - done)
        runner.run(n)
        done += n
        res = _result(spec, runner, time.time() - start, partial=done < total)
        if out is not None:
            write_result(res, out)
        log.info("cpl4 %s: %.1f / %.1f s sim, %.0f s wall", spec.scene, res.t_end, spec.seconds, res.wall_s)
    return res


def result_json(res: SweepResult) -> dict[str, Any]:
    worst = {q: {b: dict(zip(("value", "dz_mm"), res.worst(q, b))) for b in ("per_min", "max_1s")}
             for q in QUANTITIES[res.spec.scene]}
    return {"schema": "alloy.bench.cpl4/1", "verdict": res.verdict(), "worst": worst, **asdict(res)}


def write_result(res: SweepResult, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result_json(res), indent=1))
    tmp.replace(out)


def summary(res: SweepResult) -> str:
    """One line per quantity: worst rate and worst window (as % of E_scale), then every height's rate."""
    out = []
    for q in QUANTITIES[res.spec.scene]:
        (rate, dz_r), (win, dz_w) = res.worst(q, "per_min"), res.worst(q, "max_1s")
        each = ", ".join(f"{r.dz_mm:g} mm {100 * r.stats[q].per_min:.3g}" for r in res.rows)
        step = max(r.stats[q].max_step for r in res.rows)
        out.append(f"{q}: worst {100 * rate:.3g} %/sim-min (dz {dz_r:g} mm), worst 1 s window {100 * win:.3g} % "
                   f"(dz {dz_w:g} mm), largest exchange {100 * step:.3g} %; per height %/sim-min: {each}")
    return f"{res.spec.scene} {res.t_end:.1f} s, {res.verdict()}: " + "; ".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.bench.cpl4", description=__doc__.split("\n\n")[0])
    ap.add_argument("--scene", choices=("dissipative", "conservative"), required=True)
    ap.add_argument("--seconds", type=float, default=MIN_SECONDS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = run_sweep(SweepSpec(scene=args.scene, seconds=args.seconds, device=args.device), args.out)
    log.info("cpl4: %s", summary(res))
    return 0 if res.verdict() == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
