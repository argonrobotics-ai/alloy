# SPDX-License-Identifier: Apache-2.0
"""The harness's generic checks: contract, crash, determinism, world independence, speed, float-atomic lint,
action-reaction and object sanity. Each returns ``CheckResult``s whose ``fix`` says what to change."""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import numpy as np

from alloy.core.defaults import SPEED_FAIL_MPS
from alloy.plugins.base import ForcePlugin, ObjectPlugin, Plugin, SensorPlugin, SolverPlugin
from alloy.plugins.harness.report import CheckResult
from alloy.plugins.harness.run import Run
from alloy.plugins.npgeom import quat_rotate

log = logging.getLogger("alloy.plugins")

POSITION_LIMIT = 100.0       # m from the origin
BALANCE_BOUND = 1.0e-4       # relative net force / moment of internal forces (float32 sums)
REQUIRED_HOOK = {"force": (ForcePlugin, "apply"), "sensor": (SensorPlugin, "measure"),
                 "object": (ObjectPlugin, "build"), "solver": (SolverPlugin, "capability")}


def contract(cls: type[Plugin]) -> CheckResult:
    """Class-level rules: a name, a known kind, a docstring, its required hook overridden."""
    problems = []
    if not cls.name or not cls.name.replace("_", "").isalnum():
        problems.append("set name = 'snake_case_name'")
    if cls.kind not in REQUIRED_HOOK:
        problems.append("subclass ForcePlugin, SensorPlugin, ObjectPlugin or SolverPlugin")
    elif cls.kind == "force":
        if cls.apply is ForcePlugin.apply and cls.update is ForcePlugin.update:
            problems.append("implement apply() (forces) and/or update() (own state)")
    else:
        base, hook = REQUIRED_HOOK[cls.kind]
        if getattr(cls, hook) is getattr(base, hook):
            problems.append(f"implement {hook}()")
    if not (cls.__doc__ or "").strip():
        problems.append("add a class docstring: what it models, units, an example")
    if not cls.summary:
        problems.append("set summary = 'one line for alloy.plugins list'")
    detail = "; ".join(problems) or "name, kind, docstring, hook"
    return CheckResult("contract", "contract", not problems, detail=detail,
                       fix="; ".join(problems))


def _finite_reads(run: Run) -> list[str]:
    bad = []
    for k, reads in enumerate(run.trace.reads):
        for label, arrays in reads.items():
            for name, a in arrays.items():
                if np.issubdtype(np.asarray(a).dtype, np.number) and not np.isfinite(a).all():
                    bad.append(f"{label}.{name} after period {k}")
    return bad


def crash(run: Run) -> CheckResult:
    """The run finished; state, plug-in forces and plug-in arrays stayed finite, in range and below 20 m/s."""
    fix = "look at the traceback/first bad period; lower forces or check_dt-limit the step, guard divisions"
    if run.error is not None:
        last = run.error.strip().splitlines()[-1]
        return CheckResult("crash", "crash", False, detail=f"exception: {last}", fix=f"{fix}\n{run.error[-1500:]}")
    tr = run.trace
    for name in ("body_q", "body_qd", "particle_q", "particle_qd", "body_f"):
        a = getattr(tr, name)
        if a.size and not np.isfinite(a).all():
            k = int(np.argmax(~np.isfinite(a).reshape(a.shape[0], -1).all(axis=1)))
            return CheckResult("crash", "crash", False, detail=f"{name} not finite after period {k}", fix=fix)
    bad = _finite_reads(run)
    if bad:
        return CheckResult("crash", "crash", False, detail=f"plug-in array not finite: {bad[0]}", fix=fix)
    reach = max(float(np.abs(tr.body_q[..., :3]).max(initial=0.0)), float(np.abs(tr.particle_q).max(initial=0.0)))
    speed = max(float(np.linalg.norm(tr.body_qd[..., :3], axis=-1).max(initial=0.0)),
                float(np.linalg.norm(tr.particle_qd, axis=-1).max(initial=0.0)))
    ok = reach <= POSITION_LIMIT and speed <= SPEED_FAIL_MPS
    return CheckResult("crash", "crash", ok, speed, SPEED_FAIL_MPS,
                       f"{run.exchanges} steps, max speed {speed:.3g} m/s, max |x| {reach:.3g} m",
                       "" if ok else f"a body left the scene or exceeded {SPEED_FAIL_MPS} m/s: {fix}")


def determinism(a: Run, b: Run) -> CheckResult:
    """Two identical runs give bit-identical per-world digests (State + every plug-in/solver state array)."""
    if a.error or b.error:
        return CheckResult.skip("determinism", "determinism", "a run crashed (see crash)")
    da, db = a.digests(), b.digests()
    diff = int((da != db).any(axis=1).sum())
    return CheckResult("determinism", "determinism", diff == 0, float(diff), 0.0,
                       f"{diff} of {da.shape[0]} worlds differ between two identical runs",
                       "no float atomics (write each element from one thread), no host randomness or time, "
                       "seed device RNG with the world id and step")


def graph_replay(eager: Run, graph: Run) -> CheckResult:
    """The same run with every control period captured as a graph and replayed matches the eager run bit for bit."""
    if eager.error or graph.error:
        why = graph.error.strip().splitlines()[-1] if graph.error else "the eager run crashed (see crash)"
        if eager.error:
            return CheckResult.skip("graph_replay", "graph", why)
        return CheckResult("graph_replay", "graph", False, detail=f"graph run failed: {why}", fix=graph.error[-1500:])
    diff = int((eager.digests() != graph.digests()).any(axis=1).sum())
    return CheckResult("graph_replay", "graph", diff == 0, float(diff), 0.0,
                       f"{diff} of {eager.model.world_count} worlds differ between eager and graph-replayed periods",
                       "a hook used a Python value that changes between steps (a counter, time, a float argument "
                       "computed each step); a captured graph freezes it. Keep changing values in device arrays "
                       "(add_state / add_input / ctx.time) and update them inside kernels")


def _world_rows(run: Run, w: int) -> np.ndarray:
    return np.nonzero(run.model.body_world.numpy() == w)[0]


def _world_diff(ta: Run, wa: int, tb: Run, wb: int) -> float:
    """Max abs difference between world ``wa`` of run ``ta`` and world ``wb`` of ``tb``: bodies and plug-in arrays."""
    ra, rb = _world_rows(ta, wa), _world_rows(tb, wb)
    diff = 0.0
    if ra.size:
        for name in ("body_q", "body_qd", "body_f"):
            x, y = getattr(ta.trace, name)[:, ra], getattr(tb.trace, name)[:, rb]
            diff = max(diff, float(np.abs(x.astype(np.float64) - y).max(initial=0.0)))
    for reads_a, reads_b in zip(ta.trace.reads, tb.trace.reads, strict=True):
        for label, arrays in reads_a.items():
            for name, x in arrays.items():
                x, y = np.asarray(x), np.asarray(reads_b[label][name])
                if np.issubdtype(x.dtype, np.number) and x.shape[0] > max(wa, wb):
                    diff = max(diff, float(np.abs(x[wa].astype(np.float64) - y[wb]).max(initial=0.0)))
    return diff


def independence(a: Run, p: Run, tol: float = 1.0e-6) -> list[CheckResult]:
    """Identical worlds stay identical, and perturbing worlds 1.. leaves world 0 unchanged."""
    if a.error or p.error:
        return [CheckResult.skip("identical_worlds", "independence", "a run crashed (see crash)")]
    worlds = a.model.world_count
    if worlds < 2:
        return [CheckResult.skip("identical_worlds", "independence", "scene has one world")]
    same = max(_world_diff(a, w, a, 0) for w in range(1, worlds))
    leak = _world_diff(a, 0, p, 0)
    fix = "index every array with the world id from wp.tid(); look items up per world with ctx.bodies(label)"
    return [
        CheckResult("identical_worlds", "independence", same <= tol, same, tol,
                    f"max difference between identical worlds over {worlds} worlds", fix),
        CheckResult("isolated_worlds", "independence", leak <= tol, leak, tol,
                    "max change of world 0 when worlds 1.. are perturbed", fix),
    ]


def speed(with_plugin: float, without: float, budget: float) -> CheckResult:
    """Step time with the plug-in at most ``budget`` times the time without it."""
    ratio = with_plugin / max(without, 1e-12)
    return CheckResult("speed", "speed", ratio <= budget, ratio, budget,
                       f"{with_plugin * 1e3:.3g} ms/step with, {without * 1e3:.3g} ms without",
                       "fewer kernel launches per step (fuse loops into one kernel), no host syncs, no allocation "
                       "in apply/update/measure; or raise the class attribute speed_budget with a reason")


def _source_files(cls: type) -> list[Path]:
    file = inspect.getsourcefile(cls)
    if file is None:
        return []
    path = Path(file)
    if (path.parent / "__init__.py").exists() and path.parent.name not in ("site-packages",):
        return sorted(path.parent.glob("*.py"))
    return [path]


def lint(cls: type) -> CheckResult:
    """No float atomics in the plug-in's kernels (they make GPU runs non-deterministic)."""
    try:
        from alloy.tools.lint_atomics import lint as lint_paths  # noqa: PLC0415
    except ImportError:
        return CheckResult.skip("float_atomics", "lint", "alloy.tools.lint_atomics not available")
    files = _source_files(cls)
    if not files:
        return CheckResult.skip("float_atomics", "lint", "no source file")
    findings, errors = lint_paths(files)
    hard = [f for f in findings if f.rule != "DET002"]
    soft = [f for f in findings if f.rule == "DET002"]
    detail = "; ".join(f.format() for f in hard[:5]) or f"{len(files)} file(s) clean"
    if soft:
        detail += f" ({len(soft)} unresolved dtype note(s): annotate kernel parameters)"
    return CheckResult("float_atomics", "lint", not hard and not errors, float(len(hard)), 0.0, detail,
                       "replace atomic_add / a[i] += x on floats with one writer per element: loop over the "
                       "contributions inside one thread (e.g. one thread per world) and write a[i] = a[i] + x")


def action_reaction(run: Run) -> CheckResult:
    """Internal forces: per world, net plug-in force and net moment (about the origin) vanish."""
    if run.error:
        return CheckResult.skip("action_reaction", "law", "the run crashed (see crash)")
    q = run.trace.applied_q if getattr(run.trace, "applied_q", None) is not None else run.trace.body_q
    f = run.trace.body_f.astype(np.float64)
    com = q[..., :3] + quat_rotate(q[..., 3:7].astype(np.float64), run.model.body_com.numpy()[None].astype(np.float64))
    moment = f[..., 3:] + np.cross(com, f[..., :3])
    worst = 0.0
    for w in range(run.model.world_count):
        rows = _world_rows(run, w)
        fs, ms = f[:, rows, :3], moment[:, rows]
        scale_f = np.linalg.norm(fs, axis=-1).sum(axis=1) + 1e-30
        scale_m = np.linalg.norm(ms, axis=-1).sum(axis=1) + np.linalg.norm(f[:, rows, 3:], axis=-1).sum(axis=1) + 1e-30
        worst = max(worst, float((np.linalg.norm(fs.sum(axis=1), axis=-1) / scale_f).max(initial=0.0)),
                    float((np.linalg.norm(ms.sum(axis=1), axis=-1) / scale_m).max(initial=0.0)))
    return CheckResult("action_reaction", "law", worst <= BALANCE_BOUND, worst, BALANCE_BOUND,
                       "relative net force / moment of the plug-in's forces per world",
                       "apply equal and opposite forces at the same point (torque = (point - COM) x force); or set "
                       "internal = False if the plug-in pushes against something outside the scene")


def object_mass(run: Run, rows: np.ndarray) -> CheckResult:
    """Every body the object built has mass > 0 and a physical inertia (positive, triangle inequality)."""
    mass = run.model.body_mass.numpy()[rows]
    inertia = run.model.body_inertia.numpy()[rows].reshape(-1, 3, 3)
    bad = []
    for i, (m, inn) in enumerate(zip(mass, inertia, strict=True)):
        ev = np.linalg.eigvalsh(0.5 * (inn + inn.T))
        if m <= 0 or ev.min() <= 0 or ev[0] + ev[1] < ev[2] * (1 - 1e-5):
            bad.append(f"body {int(rows[i])}: mass {m:.3g}, inertia eigenvalues {np.round(ev, 9).tolist()}")
    return CheckResult("mass_inertia", "law", not bad, float(len(bad)), 0.0, "; ".join(bad) or f"{len(rows)} bodies",
                       "give every body a shape with density > 0 (or set mass and inertia explicitly)")


def settles(run: Run, rows: np.ndarray, v_tol: float = 0.02, w_tol: float = 0.5) -> CheckResult:
    """Dropped on the ground, the object comes to rest (no jitter, no drift)."""
    if run.error:
        return CheckResult.skip("settles", "law", "the run crashed (see crash)")
    qd = run.trace.body_qd[-1, rows]
    v, w = float(np.linalg.norm(qd[:, :3], axis=-1).max()), float(np.linalg.norm(qd[:, 3:], axis=-1).max())
    ok = v <= v_tol and w <= w_tol
    return CheckResult("settles", "law", ok, v, v_tol, f"end speed {v:.3g} m/s, spin {w:.3g} rad/s after the drop",
                       "check joint limits/damping and shape overlaps inside the object (collision filtering)")


__all__ = ["action_reaction", "contract", "crash", "determinism", "graph_replay", "independence", "lint",
           "object_mass", "settles", "speed"]
