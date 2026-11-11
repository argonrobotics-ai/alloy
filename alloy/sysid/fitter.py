# SPDX-License-Identifier: Apache-2.0
"""The fit: replay the recording in many worlds with candidate parameters and shrink the tracking error.

Every world holds one candidate for every joint. Each joint has its own small search (CMA-ES in the unit box of its
variables, ``alloy.fit.cma``) scored by that joint's own tracking error, and all searches share the same worlds, so
a generation costs one batched replay whatever the number of joints. World 0 always carries the current estimate
of every joint. After the last generation one more replay compares the robot file's values with the fit, and
measures how well each parameter is determined (``alloy.sysid.identify``).

Deterministic: the same recording, options and ``seed`` give the same file on the CPU (on a GPU, to the last digits).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alloy.sysid import identify, units
from alloy.sysid.errors import SysidError
from alloy.sysid.params import MAX_DELAY, Box, to_fields, variables
from alloy.sysid.recording import Recording
from alloy.sysid.replay import Bench, Track, control_dt, resample, squared_error
from alloy.sysid.result import SysidResult

log = logging.getLogger("alloy.sysid")
SIGMA0 = 0.25           # first search step, in the unit box
MIN_WORLDS = 16


@dataclass
class FitConfig:
    """Options of ``fit``. ``parameters``: names from ``alloy.sysid.PARAMETERS`` (None: gains, damping, friction,
    delay); ``joints``: a subset of the recording's (None: all); ``worlds``: candidates per generation."""

    parameters: tuple[str, ...] | None = None
    joints: tuple[str, ...] | None = None
    worlds: int = 64
    generations: int = 30
    seed: int = 0
    max_delay: float = MAX_DELAY
    angle_unit: str = "rad"
    time_unit: str = "s"
    device: str | None = None


@dataclass
class _Problem:
    bench: Bench
    tracks: list[Track]
    names: list[str]
    box: Box
    moved: np.ndarray

    def evaluate(self, u: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Unit-box candidates ``u`` (worlds, V, J) -> (mean squared error (worlds, J), simulated positions)."""
        fields = to_fields(self.names, self.box.decode(u), self.bench.nominal)
        sim_q = self.bench.rollout(fields, self.tracks)
        return squared_error(sim_q, self.tracks), sim_q


def fit(robot: Any, data: Any, config: FitConfig | None = None, **options: Any) -> SysidResult:
    """Fit ``robot``'s joint parameters to a recording. ``robot``: a shipped robot's name or a robot file;
    ``data``: a recording file (CSV, HDF5, LeRobot dataset folder) or a ``Recording``; options as ``FitConfig``.

    Example: ``alloy.sysid.fit("panda", "take.csv", worlds=64).save("my_panda.sysid.json")``."""
    cfg = config or FitConfig(**options)
    if cfg.worlds < MIN_WORLDS or cfg.generations < 1:
        raise SysidError(f"fit needs worlds >= {MIN_WORLDS} and generations >= 1", "the defaults are worlds=64, "
                         "generations=30", field="worlds")
    started = time.perf_counter()
    problem, rec = _prepare(robot, data, cfg)
    means, history = _search(problem, cfg)
    result = _finish(problem, means, cfg, robot, rec)
    result.meta.update({"seed": cfg.seed, "worlds": cfg.worlds, "generations": len(history),
                        "wall_s": round(time.perf_counter() - started, 1), "device": str(problem.bench.sim.device),
                        "recording": {"source": rec.source, "seconds": round(rec.duration, 3),
                                      "episodes": len(rec.episodes), "sample_dt": rec.dt},
                        "history_rms": [[float(x) for x in row] for row in history]})
    return result


def _prepare(robot: Any, data: Any, cfg: FitConfig) -> tuple[_Problem, Recording]:
    from alloy.sysid.readers import load_recording  # noqa: PLC0415

    rec = data if isinstance(data, Recording) else load_recording(data)
    if cfg.joints:
        rec = rec.select(list(cfg.joints))
    names = variables(cfg.parameters)
    dt = control_dt(rec if cfg.time_unit == "s" else units.convert(rec, time=cfg.time_unit))
    bench = Bench(robot, rec.joints, cfg.worlds, dt, cfg.device)
    rec = units.convert(rec, cfg.angle_unit, cfg.time_unit, bench.hinge)
    units.check(rec, bench.lower, bench.upper, bench.hinge)
    tracks = [resample(e, bench.dt) for e in rec.episodes]
    has_moved = identify.moved(np.concatenate([t.command for t in tracks]),
                               np.concatenate([t.position for t in tracks]), bench.hinge)
    if not has_moved.any():
        raise SysidError("no joint moves in this recording (every command and position spans under 0.02 rad)",
                         "record the excitation from `python -m alloy.sysid excite`", field="data")
    box = Box.around(names, bench.nominal, cfg.max_delay)
    return _Problem(bench, tracks, names, box, has_moved), rec


def _search(problem: _Problem, cfg: FitConfig) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run the per-joint searches; returns the estimate (V, J) in the unit box and the RMS of world 0 per
    generation."""
    from alloy.fit.cma import CMAES  # noqa: PLC0415

    u0 = problem.box.encode(problem.box.start(problem.bench.nominal))            # (V, J)
    joints = np.flatnonzero(problem.moved)
    searches = {int(j): CMAES(u0[:, j], SIGMA0, cfg.worlds, cfg.seed + 7919 * int(j)) for j in joints}
    means, history = u0.copy(), []
    for generation in range(cfg.generations):
        u = np.repeat(u0[None], cfg.worlds, axis=0)
        for j, search in searches.items():
            u[:, :, j] = search.ask(inject=means[:, j])[: cfg.worlds]
        mse, _ = problem.evaluate(u)
        for j, search in searches.items():
            lam = search.lam
            search.tell(np.resize(u[:, :, j], (lam, u.shape[1])) if lam != cfg.worlds else u[:, :, j],
                        np.resize(mse[:, j], lam) if lam != cfg.worlds else mse[:, j])
            means[:, j] = search.mean
        history.append(np.sqrt(mse[0]))
        log.info("sysid generation %d/%d: worst joint RMS %.4g", generation + 1, cfg.generations,
                 float(np.max(np.sqrt(mse[0, joints]))))
        if all(s.stop_reason() for s in searches.values()):
            break
    return means, history


def _finish(problem: _Problem, means: np.ndarray, cfg: FitConfig, robot: Any, rec: Recording) -> SysidResult:
    """One more replay: world 0 the robot file's values, world 1 the fit, then the nudged worlds."""
    bench, box, names = problem.bench, problem.box, problem.names
    u0 = box.encode(box.start(bench.nominal))
    points, steps = identify.nudged(means)
    u = np.repeat(means[None], cfg.worlds, axis=0)
    u[0] = u0
    u[2:2 + len(points)] = points
    mse, sim_q = problem.evaluate(u)
    worse = (mse[1] >= mse[0]) | ~problem.moved
    final = np.where(worse[None, :], u0, means)
    measured = np.concatenate([t.position for t in problem.tracks])
    joined = np.concatenate(sim_q, axis=0)                                       # (N, worlds, J)
    sigma_u = identify.unit_sigma(joined[:, 1] - measured, np.moveaxis(joined[:, 2:2 + len(points)], 1, 0), steps,
                                  bench.dt)
    ok = identify.determined(sigma_u, problem.moved & ~worse)
    x = box.decode(final)
    values = {f: np.asarray(v, np.float64) for f, v in to_fields(names, x, bench.nominal).items()}
    rel = _relative_sigma(box, final, sigma_u)
    result = SysidResult(
        robot=str(Path(str(robot)).name if Path(str(robot)).is_file() else robot), joints=list(rec.joints),
        control_dt=bench.dt, values=values, variables=list(names), nominal=dict(bench.nominal),
        rms_before=np.sqrt(mse[0]), rms_after=np.sqrt(np.where(worse, mse[0], mse[1])), moved=problem.moved.copy(),
        sigma={n: rel[v] for v, n in enumerate(names)}, determined={n: ok[v] for v, n in enumerate(names)},
        hinge=bench.hinge.copy(), notes=[*bench.notes, *rec.notes, *_notes(problem, final, worse, rec)])
    result.curves = {"dt": bench.dt, "measured": measured, "before": joined[:, 0], "after": joined[:, 1],
                     "command": np.concatenate([t.command for t in problem.tracks])}
    return result


def _relative_sigma(box: Box, u: np.ndarray, sigma_u: np.ndarray) -> np.ndarray:
    """1-sigma of each value as a share of the value (of the bound's width for a value at 0), (V, J)."""
    s = np.where(np.isfinite(sigma_u), sigma_u, 1.0)
    span = box.decode(np.clip(u + s, 0.0, 1.0)) - box.decode(np.clip(u - s, 0.0, 1.0))
    x = box.decode(u)
    return np.where(np.isfinite(sigma_u), 0.5 * span / np.where(x > 1e-12, x, box.hi - box.lo + 1e-300), np.inf)


def _notes(problem: _Problem, u: np.ndarray, worse: np.ndarray, rec: Recording) -> list[str]:
    out = []
    for j, name in enumerate(rec.joints):
        if not problem.moved[j]:
            out.append(f"{name}: never moved in the recording (under 0.02 rad); its values are the robot file's")
        elif worse[j]:
            out.append(f"{name}: the search found nothing better than the robot file's values; they are kept")
        else:
            at = [n for v, n in enumerate(problem.names)
                  if u[v, j] > 0.98 or (u[v, j] < 0.02 and bool(problem.box.log[v, j]))]
            if at:
                out.append(f"{name}: {', '.join(at)} ended at a bound of the search; the true value may lie outside")
    if "vel" in problem.names:
        out.append("kd and joint damping act alike on a position servo: only their sum is determined, and it is "
                   "split in the robot file's ratio")
    return out


__all__ = ["FitConfig", "fit"]
