# SPDX-License-Identifier: Apache-2.0
"""Alloy bench harness: Scenario base class, world-count ladder runner and JSON-lines records.

A scenario builds one Newton model holding `nworld` copies of its scene, steps it, and reports host-side
measurements between timed chunks. The runner sweeps the ladder per config, times with warm-up and device
sync (CUDA graph when allowed), turns every failure into a record, and appends one JSON line per
(scenario, config, nworld) to bench/results/<family>/<scenario>.jsonl. Field definitions: bench/SCENARIOS.md §2.
"""
from __future__ import annotations

import gc
import json
import logging
import math
import platform
import resource
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from bench import metrics as M

log = logging.getLogger("alloy.bench")
REPO = Path(__file__).resolve().parents[1]
DEFAULT_LADDER = (1, 16, 256, 1024, 4096)
SCHEMA = "alloy.bench.run/1"


@dataclass(frozen=True)
class Config:
    """One solver configuration of a scenario. dt is the outer (frame) step; the solver sees dt / substeps."""

    name: str
    solver: str
    dt: float = 1.0 / 60.0
    substeps: int = 1
    params: dict[str, Any] = field(default_factory=dict)
    cuda_graph: bool = True
    max_nworld: int | None = None  # cap the ladder for this config (memory or speed)


@dataclass
class Built:
    """Everything a scenario needs to step. Unpacks as (model, solver, state, control, contacts)."""

    model: Any
    solver: Any
    state_0: Any
    state_1: Any
    control: Any
    contacts: Any = None
    pipeline: Any = None  # newton.CollisionPipeline or None (solver-internal collision)
    nworld: int = 1
    extra: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Any]:
        return iter((self.model, self.solver, self.state_0, self.control, self.contacts))


class Scenario:
    """Base class. Subclasses set the class attributes and implement build(); the rest has working defaults.

    Hooks (all host side, outside the timed window):
      on_start(b, rec)        after warm-up, before timing
      measure(b, t, rec)      every sample_every outer steps; t is sim time [s]
      finish(b, rec)          after the horizon; set rec["accuracy_error"], rec["metrics"][...]
      energy(b) -> float      total energy [J] for energy_drift_pct (None = not reported)
    If host_control is True, control(b, t) runs before every outer step and CUDA graphs are disabled.
    """

    id: str = ""
    family: str = ""
    title: str = ""
    horizon_s: float = 1.0
    configs: tuple[Config, ...] = ()
    ladder: tuple[int, ...] = DEFAULT_LADDER
    has_contacts: bool = False
    host_control: bool = False
    accuracy_unit: str = ""
    accuracy_ref: str = ""  # law or source, e.g. "z0 - g t^2 / 2 (exact)"
    sample_every: int = 0  # outer steps between measure() calls; 0 = about 20 samples per horizon
    warmup_steps: int = 2

    # ---- to implement -------------------------------------------------------------------------------------
    def build(self, nworld: int, device: str, config: Config) -> Built:
        raise NotImplementedError

    # ---- stepping -----------------------------------------------------------------------------------------
    def substep(self, b: Built, dt: float) -> None:
        """One solver substep with Newton's standard ping-pong states."""
        b.state_0.clear_forces()
        if b.pipeline is not None:
            b.pipeline.collide(b.state_0, b.contacts)
        b.solver.step(b.state_0, b.state_1, b.control, b.contacts, dt)
        b.state_0, b.state_1 = b.state_1, b.state_0

    def step(self, b: Built, config: Config) -> None:
        """One outer step (config.substeps substeps)."""
        h = config.dt / config.substeps
        for _ in range(config.substeps):
            self.substep(b, h)

    def control(self, b: Built, t: float) -> None:
        """Host-side control update (only called when host_control is True)."""

    # ---- measurement hooks --------------------------------------------------------------------------------
    def on_start(self, b: Built, rec: dict) -> None:
        pass

    def measure(self, b: Built, t: float, rec: dict) -> None:
        if self.has_contacts and b.contacts is not None:
            pen = M.rigid_penetration_mm(b.model, b.state_0, b.contacts)
            if pen.size:
                b.extra.setdefault("_pen_mm", []).append(pen)

    def finish(self, b: Built, rec: dict) -> None:
        pass

    def energy(self, b: Built) -> float | None:
        return None

    def state_arrays(self, b: Built) -> dict[str, np.ndarray]:
        """State snapshot used for the finite check and repeat_identical."""
        out = {}
        for name in ("body_q", "body_qd", "particle_q", "particle_qd", "joint_q"):
            arr = getattr(b.state_0, name, None)
            if arr is not None and arr.shape[0] > 0:
                out[name] = arr.numpy()
        return out


# ---- provenance -------------------------------------------------------------------------------------------
def _dist_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _newton_commit() -> str | None:
    """Newton is vendored in Alloy (alloy/_vendor): its commit is the pin."""
    from alloy.core.newton_compat import NEWTON_PIN

    return NEWTON_PIN


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=1)
def versions() -> dict[str, Any]:
    from alloy._vendor import VERSIONS  # Newton and MuJoCo Warp are vendored: the versions of the pins

    sha = _git("rev-parse", "--short=10", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--", "alloy", "bench"))
    return {
        "alloy": f"{sha}{'-dirty' if dirty else ''}" if sha else None,
        "newton": VERSIONS["newton"],
        "newton_commit": _newton_commit(),
        "warp": _dist_version("warp-lang"),
        "mujoco": _dist_version("mujoco"),
        "mujoco_warp": VERSIONS["mujoco_warp"],
        "numpy": np.__version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "driver": _nvidia_smi("driver_version"),
    }


def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except Exception:  # noqa: BLE001
        return None


def _host_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10  # bytes on macOS, KiB on Linux


def _device_mem_mb(device: str) -> float | None:
    import warp as wp

    dev = wp.get_device(device)
    if not dev.is_cuda:
        return None
    if wp.is_mempool_enabled(dev):
        return wp.get_mempool_used_mem_current(dev) / 2**20
    used = _nvidia_smi("memory.used")
    return float(used) if used else None


# ---- runner -----------------------------------------------------------------------------------------------
@dataclass
class RunOptions:
    device: str = "cpu"
    ladder: tuple[int, ...] | None = None  # None = each scenario's own ladder
    horizon_s: float | None = None  # override every scenario's horizon
    budget_s: float = 120.0  # wall-clock cap on the timed loop per run (record gets complete=False)
    repeat_max_nworld: int = 16  # run a second time and compare bitwise when nworld <= this (0 = never)
    out_dir: Path = REPO / "bench" / "results"
    command: str = ""


def _base_record(scn: Scenario, cfg: Config, nworld: int, opts: RunOptions) -> dict:
    return {
        "schema": SCHEMA, "scenario": scn.id, "family": scn.family, "config": cfg.name, "solver": cfg.solver,
        "params": cfg.params, "device": opts.device, "device_name": None, "nworld": nworld, "dt": cfg.dt,
        "substeps": cfg.substeps, "horizon_s": opts.horizon_s or scn.horizon_s, "steps": 0, "cuda_graph": False,
        "status": "ok", "error": None, "setup_s": None, "first_steps_s": None, "ms_per_step": None,
        "world_steps_per_s": None, "sim_seconds_per_gpu_second": None, "single_world_realtime_factor": None,
        "gpu_mem_mb": None, "gpu_mem_mb_per_world": None, "host_rss_mb": None, "stable": None, "complete": None,
        "energy_drift_pct": None, "penetration_p99_mm": None, "accuracy_error": None,
        "accuracy_unit": scn.accuracy_unit, "accuracy_ref": scn.accuracy_ref, "repeat_identical": None,
        "metrics": {}, "notes": "", "versions": versions(), "command": opts.command,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _sync(device: str) -> None:
    import warp as wp

    wp.synchronize_device(device)


def _execute(scn: Scenario, cfg: Config, nworld: int, opts: RunOptions, rec: dict) -> dict[str, np.ndarray]:
    """Build, warm up, time and measure one run in place on rec; returns the final state snapshot."""
    import warp as wp

    dev = wp.get_device(opts.device)
    rec["device_name"] = dev.name
    horizon = opts.horizon_s or scn.horizon_s
    total = max(1, int(round(horizon / cfg.dt)))
    use_graph = dev.is_cuda and cfg.cuda_graph and not scn.host_control
    graph_steps = 1 if cfg.substeps % 2 == 0 else 2  # even number of ping-pong swaps per graph launch
    every = scn.sample_every or max(1, total // 20)
    if use_graph:
        every = max(graph_steps, every - every % graph_steps)

    _sync(opts.device)
    mem0 = _device_mem_mb(opts.device)
    t0 = time.perf_counter()
    with wp.ScopedDevice(opts.device):
        b = scn.build(nworld, opts.device, cfg)
    b.nworld = nworld
    _sync(opts.device)
    rec["setup_s"] = time.perf_counter() - t0

    def advance(n: int, t_sim: float) -> None:
        for k in range(n):
            if scn.host_control:
                scn.control(b, t_sim + k * cfg.dt)
            scn.step(b, cfg)

    warm = min(max(scn.warmup_steps, graph_steps if use_graph else 1), total)
    t0 = time.perf_counter()
    advance(warm, 0.0)
    _sync(opts.device)
    rec["first_steps_s"] = time.perf_counter() - t0
    done = warm

    graph = None
    if use_graph:
        with wp.ScopedCapture(device=opts.device) as cap:
            advance(graph_steps, 0.0)
        graph = cap.graph
        rec["cuda_graph"] = True

    e0 = scn.energy(b)
    energies = [] if e0 is None else [e0]
    scn.on_start(b, rec)
    timed, stable = 0.0, True
    while done < total and timed < opts.budget_s:
        n = min(every, total - done)
        if graph is not None:
            n = max(graph_steps, n - n % graph_steps)
        t0 = time.perf_counter()
        if graph is not None:
            for _ in range(n // graph_steps):
                wp.capture_launch(graph)
        else:
            advance(n, done * cfg.dt)
        _sync(opts.device)
        timed += time.perf_counter() - t0
        done += n
        snap = scn.state_arrays(b)
        if not M.finite_state(snap):
            stable = False
            break
        scn.measure(b, done * cfg.dt, rec)
        if e0 is not None:
            energies.append(scn.energy(b))

    timed_steps = done - warm
    rec["steps"] = done
    rec["complete"] = done >= total
    rec["stable"] = stable
    if timed_steps > 0 and timed > 0:
        step_s = timed / timed_steps
        rec["ms_per_step"] = step_s * 1e3
        rec["world_steps_per_s"] = nworld / step_s
        rec["sim_seconds_per_gpu_second"] = nworld * cfg.dt / step_s
        if nworld == 1:
            rec["single_world_realtime_factor"] = cfg.dt / step_s
    mem1 = _device_mem_mb(opts.device)
    if mem0 is not None and mem1 is not None:
        rec["gpu_mem_mb"] = mem1 - mem0
        rec["gpu_mem_mb_per_world"] = (mem1 - mem0) / nworld
    rec["host_rss_mb"] = _host_rss_mb()
    if len(energies) > 1 and stable:
        ref = max(abs(energies[0]), 1e-12)
        rec["energy_drift_pct"] = 100.0 * max(abs(e - energies[0]) for e in energies) / ref
    pens = b.extra.pop("_pen_mm", None)
    if pens:
        rec["penetration_p99_mm"] = float(np.percentile(np.concatenate(pens), 99))
    if stable:
        scn.finish(b, rec)
    final = scn.state_arrays(b)
    del b
    return final


def run_one(scn: Scenario, cfg: Config, nworld: int, opts: RunOptions) -> dict:
    """One ladder rung; never raises (failures become status=error records)."""
    rec = _base_record(scn, cfg, nworld, opts)
    try:
        final = _execute(scn, cfg, nworld, opts, rec)
        if opts.repeat_max_nworld and nworld <= opts.repeat_max_nworld and rec["stable"]:
            rec2 = _base_record(scn, cfg, nworld, opts)
            rec["repeat_identical"] = M.arrays_identical(final, _execute(scn, cfg, nworld, opts, rec2))
    except Exception as exc:  # noqa: BLE001 - the record is the error report
        rec["status"] = "error"
        rec["stable"] = False if rec["stable"] is None else rec["stable"]
        rec["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        rec["notes"] = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])[-2000:]
    finally:
        gc.collect()
    return rec


def _ladder_for(scn: Scenario, cfg: Config, opts: RunOptions) -> list[int]:
    ladder = list(opts.ladder or scn.ladder)
    cap = cfg.max_nworld
    return [n for n in ladder if cap is None or n <= cap]


def write_record(rec: dict, out_dir: Path) -> Path:
    path = out_dir / rec["family"] / f"{rec['scenario']}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=_json_default) + "\n")
    return path


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def run_scenario(scn: Scenario, opts: RunOptions, config_names: set[str] | None = None) -> list[dict]:
    """Sweep the ladder for every (selected) config. After an error, larger rungs of that config are skipped."""
    records = []
    for cfg in scn.configs:
        if config_names and cfg.name not in config_names:
            continue
        failed_at = None
        for n in _ladder_for(scn, cfg, opts):
            if failed_at is not None:
                rec = _base_record(scn, cfg, n, opts)
                rec.update(status="skipped", notes=f"skipped: error at nworld={failed_at}")
            else:
                rec = run_one(scn, cfg, n, opts)
                if rec["status"] == "error":
                    failed_at = n
            path = write_record(rec, opts.out_dir)
            records.append(rec)
            log.info("%s/%s n=%d %s ms/step=%s stable=%s acc=%s -> %s", scn.id, cfg.name, n, rec["status"],
                     _fmt(rec["ms_per_step"]), rec["stable"], _fmt(rec["accuracy_error"]), path)
            if rec["error"]:
                log.warning("  %s", rec["error"])
    return records


def _fmt(x: Any) -> str:
    return "-" if x is None else (f"{x:.4g}" if isinstance(x, float) and math.isfinite(x) else str(x))


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)


__all__ = ["Built", "Config", "RunOptions", "Scenario", "run_one", "run_scenario", "write_record", "versions",
           "setup_logging", "DEFAULT_LADDER"]
