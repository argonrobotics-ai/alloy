# SPDX-License-Identifier: Apache-2.0
"""Speed-ladder sweep over raw-Newton rungs, built on the probe (parallel §4.9; SPEC §3.9 record fields).

One JSON row per (scene, param, config, N) with the harness's speed fields (``alloy.bench.run/1`` names) plus the
ladder extras, and one ``fit`` row per sweep (``t_floor_s, c_w_s, n_knee, bytes_fixed, bytes_per_world`` over the
rungs whose quality gate passed: no gate, no result). ``bench/scenarios/ladder.py`` exposes the same rungs to the
bench harness; this module runs without it.

    python -m alloy.parallel.ladder_run --scene cable_chain --param 20 --worlds 1,16,256,1024 --device cuda:0 \
        --out bench/results/ladder/L4_cable_chain.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from alloy.parallel.costmodel import capacities_from_counts, fit_cost_model
from alloy.parallel.counters import kernels_per_call, live_bytes, trim_graph_pool
from alloy.parallel.probe import ProbeRun, TimingOptions, fit_memory, stat_p, time_target
from alloy.parallel.rungs import RawConfig, RawRung
from alloy.parallel.types import PackPolicy

log = logging.getLogger("alloy.parallel")
SCHEMA = "alloy.bench.run/1"
RUNG_OF = {"arm": "L0", "box_pile": "L1", "hull_pile": "L2", "t12": "L3", "t09": "L3", "cable_chain": "L4"}
TASK_SCENES = ("t12", "t09")  # alloy.parallel.rung_tasks.TASKS (imported lazily: it pulls in WP1/WP2/WP13)
EMB_PREFIX = "emb_"           # P3-2b embodiment rungs (rung_tasks TASKS, WP13's targets), all L3


def is_task(scene: str) -> bool:
    return scene in TASK_SCENES or scene.startswith(EMB_PREFIX)


def rung_of(scene: str) -> str | None:
    return "L3" if scene.startswith(EMB_PREFIX) else RUNG_OF.get(scene)
TASK_CONFIG = "alloy_stepper"


@dataclass(frozen=True)
class CapacityCounts:
    """Per-world counts a pack's capacities are sized from (``costmodel.capacities``: 1.5x p99 contacts and rows)."""

    contacts_p99: float
    rows_p99: float
    pairs_peak: float = 0.0  # native MuJoCo contacts: no Newton shape-pair buffer


def run_point(scene: str, param: int, n: int, device: str, opts: TimingOptions, cfg: RawConfig = RawConfig(),
              usd_per_h: float | None = None, counts: CapacityCounts | None = None) -> dict[str, Any]:
    """Build, time and gate one rung point; failures become status=error rows (never raises). ``counts`` sizes a
    task pack's capacities the WP8 way instead of the router's defaults."""
    row = _base_row(scene, param, n, device, cfg, counts)
    try:
        gc.collect()  # release the previous point's pack and graph before taking the memory baseline
        trim_graph_pool(device)
        mem0 = live_bytes(device)
        t0 = time.perf_counter()
        rung = make_rung(scene, param, n, device, cfg, counts)
        row.update(setup_s=time.perf_counter() - t0, dt=rung.dt, control_dt=rung.control_dt,
                   substeps=rung.steps_per_period, periods_per_call=rung.periods_per_call)
        if scene.startswith(EMB_PREFIX):   # the SB-7 target's capacities and route (rung_tasks)
            cap = rung.capacities
            row.update(config=f"{TASK_CONFIG}_sb7", mjw_route=rung.route, capacities={
                "nconmax": cap.nconmax, "njmax": cap.njmax, "rigid_contact_max": cap.rigid_contact_max})
        run = time_target(rung.target(), opts, mem0)
        _fill_speed(row, run, n, rung, usd_per_h)
        gc.collect()
        trim_graph_pool(device)  # the timing graph is gone; its scratch must not block the eager gate run
        row["quality_gate"] = rung.gate()
        row["kernels_per_substep"] = _kernels(rung)
    except Exception as exc:  # noqa: BLE001 - the row is the error report
        row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000])
        log.warning("ladder %s:%d n=%d failed: %s", scene, param, n, row["error"])
    return row


def make_rung(scene: str, param: int, n: int, device: str, cfg: RawConfig = RawConfig(),
              counts: CapacityCounts | None = None) -> Any:
    """RawRung for the raw scenes, TaskRung (L3: assembled WP13 scene, Alloy entry stepper) for t12 / t09."""
    if not is_task(scene):
        return RawRung(scene, param, n, device, cfg)
    from alloy.parallel.rung_tasks import TaskRung, pack_spec  # noqa: PLC0415 - needs cards, scene, sim, tasks

    spec = None if counts is None else pack_spec(n, device, _capacities(counts, n))
    return TaskRung(scene, n, device, capacities=spec)


def _capacities(counts: CapacityCounts, n: int) -> dict[str, int]:
    return capacities_from_counts(counts.contacts_p99, counts.rows_p99, counts.pairs_peak, n, PackPolicy())


def _base_row(scene: str, param: int, n: int, device: str, cfg: RawConfig,
              counts: CapacityCounts | None = None) -> dict[str, Any]:
    dev = wp.get_device(device)
    task = is_task(scene)
    config = (TASK_CONFIG + ("_wp8cap" if counts else "")) if task else cfg.name
    caps = None if counts is None else {"counts": asdict(counts), **_capacities(counts, max(n, 1))}
    return {"schema": SCHEMA, "family": "ladder", "rung": rung_of(scene), "scenario": f"{rung_of(scene)}_{scene}",
            "param": param, "config": config, "capacities": caps, "solver": "mujoco", "device": device,
            "device_name": dev.name, "nworld": n, "dt": None if task else cfg.dt,
            "control_dt": None if task else cfg.control_dt, "substeps": None if task else cfg.substeps, "status": "ok",
            "error": None, "versions": _versions(), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "command": " ".join(sys.argv)}


def _fill_speed(row: dict[str, Any], run: ProbeRun, n: int, rung: Any, usd_per_h: float | None) -> None:
    """Speed fields from one timed run; a timed call is ``rung.periods_per_call`` control periods."""
    per_call = rung.steps_per_period * rung.periods_per_call
    period_s = run.step_s / rung.periods_per_call
    thr = n * rung.control_dt / period_s
    row.update(steps=run.steps * per_call, windows=run.windows, cuda_graph=run.graph,
               ms_per_step=1e3 * run.step_s / per_call, ms_per_period=1e3 * period_s,
               ms_per_period_min=1e3 * run.step_s_min / rung.periods_per_call, spread=run.step_s_spread,
               world_steps_per_s=n * per_call / run.step_s, sim_seconds_per_gpu_second=thr,
               single_world_realtime_factor=(rung.control_dt / period_s) if n == 1 else None,
               gpu_mem_mb=None if run.mem_bytes is None else run.mem_bytes / 2**20,
               iters_p50=stat_p(run.samples, "iters", 50), iters_p99=stat_p(run.samples, "iters", 99),
               contacts_p99=stat_p(run.samples, "contacts", 99), rows_p99=stat_p(run.samples, "rows", 99),
               overflow=max((s.overflow for s in run.samples), default=0),
               iter_limit=max((s.iter_limit for s in run.samples), default=0),
               usd_per_million_sim_s=None if usd_per_h is None else usd_per_h / (3600.0 * thr) * 1e6)
    row["mem_bytes"] = run.mem_bytes


def _kernels(rung: Any) -> float | None:
    """Warp kernels per substep from one eager period (information). Eager scratch comes on top of the graph's, so
    at the largest N it can run out of memory after the timing is done: that leaves the row ok, kernels None."""
    try:
        k = kernels_per_call(rung.step, rung.device)
    except RuntimeError as exc:
        log.warning("kernel count skipped at n=%d: %s", rung.n, exc)
        return None
    return None if k is None else k / (rung.steps_per_period * rung.periods_per_call)


def fit_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Cost-model and memory fit over the gate-passing rows of one sweep."""
    ok = [r for r in rows if r["status"] == "ok" and r.get("quality_gate", {}).get("pass")]
    out: dict[str, Any] = {"kind": "fit", "points": [r["nworld"] for r in ok],
                           "excluded": [r["nworld"] for r in rows if r not in ok]}
    if len({r["nworld"] for r in ok}) < 2:
        out["status"] = "not_run"
        out["reason"] = "fewer than two gate-passing world counts"
        return out
    t_floor, c_w = fit_cost_model([r["nworld"] for r in ok], [r["ms_per_period"] * 1e-3 for r in ok])
    out.update(status="ok", t_floor_s=t_floor, c_w_s=c_w, n_knee=(t_floor / c_w) if c_w > 0 else None,
               t_floor_unit="s per control period")
    mem = [r for r in ok if r.get("mem_bytes") is not None]
    if len({r["nworld"] for r in mem}) >= 2:
        fixed, per_world = fit_memory([(r["nworld"], r["mem_bytes"]) for r in mem])
        out.update(bytes_fixed=fixed, bytes_per_world=per_world)
    return out


def sweep(scene: str, param: int, worlds: Sequence[int], device: str, opts: TimingOptions,
          cfg: RawConfig = RawConfig(), out: Path | None = None, usd_per_h: float | None = None,
          counts: CapacityCounts | None = None) -> list[dict]:
    """Run every N ascending; after an error the larger N are recorded as skipped. Rows are appended to ``out``."""
    rows: list[dict[str, Any]] = []
    failed = None
    for n in sorted(set(worlds)):
        if failed is not None:
            row = _base_row(scene, param, n, device, cfg, counts)
            row.update(status="skipped", error=f"skipped: error at nworld={failed}")
        else:
            row = run_point(scene, param, n, device, opts, cfg, usd_per_h, counts)
            failed = n if row["status"] == "error" else None
        rows.append(row)
        _append(out, row)
        log.info("ladder %s:%d n=%d %s %.4g ms/period gate=%s", scene, param, n, row["status"],
                 row.get("ms_per_period") or float("nan"), row.get("quality_gate", {}).get("pass"))
    fit = {**_base_row(scene, param, 0, device, cfg, counts), **fit_row(rows)}
    _append(out, fit)
    return rows + [fit]


def _append(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


def _versions() -> dict[str, Any]:
    from alloy.core.newton_compat import installed_newton_commit

    return {"newton_commit": installed_newton_commit(), "warp": wp.config.version, "python": platform.python_version(),
            "host": platform.node()}


def probe_counts(scene: str, device: str, opts: TimingOptions, out: Path | None = None) -> CapacityCounts:
    """Contacts and rows p99 of one world at the router's default capacities (the row is appended to ``out``)."""
    row = run_point(scene, 0, 1, device, opts)
    _append(out, row)
    if row["status"] != "ok" or row.get("contacts_p99") is None or row.get("rows_p99") is None:
        raise RuntimeError(f"capacity probe of {scene} failed: {row.get('error')}")
    counts = CapacityCounts(float(row["contacts_p99"]), float(row["rows_p99"]))
    log.info("ladder %s: WP8 capacities from %s -> %s", scene, counts, _capacities(counts, 1))
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Alloy speed ladder (raw-Newton rungs L0-L2, L4; task scenes L3)")
    ap.add_argument("--scene", required=True, help=f"{sorted(RUNG_OF)} or a P3-2b embodiment {EMB_PREFIX}<key>")
    ap.add_argument("--param", type=int, default=0, help="objects (piles) or links (cable); ignored for arm, t12, t09")
    ap.add_argument("--worlds", default="1,4,16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--windows", type=int, default=5)
    ap.add_argument("--min-sim-s", type=float, default=0.0, help="floor on simulated seconds per window")
    ap.add_argument("--usd-per-h", type=float, default=None, help="VM spot price for usd_per_million_sim_s")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--wp8-capacities", action="store_true",
                    help="task scenes: size njmax/nconmax from a 1-world probe at the router's defaults (1.5x p99)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    opts = TimingOptions(window_s=a.window_s, windows=a.windows, min_window_sim_s=a.min_sim_s)
    if a.wp8_capacities and a.scene.startswith(EMB_PREFIX):
        log.warning("ladder %s: --wp8-capacities ignored; embodiment rungs use the SB-7 target's capacities", a.scene)
    probe = a.wp8_capacities and not a.scene.startswith(EMB_PREFIX)
    counts = probe_counts(a.scene, a.device, opts, a.out) if probe else None
    rows = sweep(a.scene, a.param, [int(x) for x in a.worlds.split(",")], a.device, opts, out=a.out,
                 usd_per_h=a.usd_per_h, counts=counts)
    return 0 if all(r["status"] in ("ok", "not_run") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
