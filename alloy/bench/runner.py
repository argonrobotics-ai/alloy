# SPDX-License-Identifier: Apache-2.0
"""`run_law`: build one world per (condition, perturbation sample), step, record on device, judge (SPEC §3.10).

Twin hooks beyond the `SimTwin` protocol are optional and looked up by name (pending SCR-WP6-1):
- `conditions(quick) -> tuple[Condition, ...]`: conditions when there is no rig dataset (analytic laws);
- `make_solver(model, solver)`: a second positional parameter receives the solver name chosen by `run_law`;
- `make_pipeline(model, solver)`: contact source (default: Newton pipeline unless MuJoCo-native contacts);
- `make_contacts(pipeline) -> Contacts` (default `pipeline.contacts()`) and `after_collide(state, contacts)`, called
  after every `collide`: per-contact properties such as Alloy's PairTable stiffness/damping (A6, SPEC §5.3);
- `init_state(model, state)`, `prepare_builder(builder, solver)`, `units`, `references(log)`, `record_stride`;
- `evaluate(ctx) -> tuple[MetricVerdict, ...]`: a law that is not a single stepped run (A16, D1-D3);
- `bind_calib(card, calib_trials)`: rig twins receive only the `calib` split (material values are fitted there);
- `expand(cond, quick, per_condition) -> list[sample]`: the twin's own worlds per condition (e.g. a tilt grid);
- `reference(cond) -> Mapping[str, float]`: closed-form value per metric for analytic laws whose `scalars` returns
  raw simulated values; the judged error is then |value - reference| and `Tolerance.rel` scales with the reference.
  Without it, analytic `scalars` must already be per-world errors.
A twin missing a required protocol method (`conditions` when there is no dataset, `build_world`, `make_solver`,
`drive`, `record`, `scalars`) yields `not_run` with the missing names: the law cannot run until its producer
implements SPEC §3.10; it never counts as a pass.
A failure inside the simulation becomes a LawResult with verdict `error` and the exception text in `monitors`;
`run_law` raises `LawError` only for misuse (a rig law without data, an unknown solver).
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench import compare as C
from alloy.bench import solvers as SV
from alloy.bench.provenance import provenance
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawResult, LawSpec, MetricVerdict, RigDataset
from alloy.core.defaults import RIGID_GAP_M
from alloy.core.errors import LawError

log = logging.getLogger("alloy.bench")


@dataclass(frozen=True)
class RunContext:
    """What a direct law (`evaluate`) or a stepped run needs to know about this invocation."""

    spec: LawSpec
    device: str
    seed: int
    solver: str
    quick: bool

    @property
    def duration_s(self) -> float:
        return float(self.spec.quick.get("duration_s", self.spec.duration_s)) if self.quick else self.spec.duration_s

    @property
    def max_worlds(self) -> int | None:
        return int(self.spec.quick.get("worlds", 16)) if self.quick else None


@dataclass
class _World:
    condition: Condition
    sample: dict[str, float]


@dataclass
class SteppedRun:
    """Everything one stepped law run produced (kept for determinism laws that compare runs)."""

    model: Any
    state: Any
    log: DeviceTwinLog
    worlds: list[_World]
    monitors: dict[str, Any] = field(default_factory=dict)


def _samples(spec: LawSpec, n: int, seed: int, cond_index: int) -> list[dict[str, float]]:
    """Perturbation deltas per world: sample 0 is nominal, the rest N(0, sigma) per parameter, seeded per condition."""
    out = [dict.fromkeys(spec.perturb, 0.0)]
    rng = np.random.default_rng([seed, cond_index])
    for _ in range(1, n):
        out.append({k: float(rng.normal(0.0, s)) for k, s in sorted(spec.perturb.items())})
    return out


def plan_worlds(spec: LawSpec, conds: Sequence[Condition], ctx: RunContext,
                expand: Callable[[Condition, bool, int], list[dict[str, float]]] | None = None) -> list[_World]:
    """One world per (condition, sample); quick mode caps the total at `spec.quick['worlds']`.

    `expand(cond, quick, per_condition)` (a twin hook) replaces the random perturbation samples with the twin's own
    per-world settings, e.g. a tilt grid per rig condition."""
    if not conds:
        raise LawError(f"{spec.id}: no conditions")
    per = spec.worlds_per_condition
    if ctx.max_worlds is not None:
        per = max(1, min(per, ctx.max_worlds // len(conds)))
        conds = conds[: ctx.max_worlds] if len(conds) > ctx.max_worlds else conds
    worlds = []
    for i, cond in enumerate(conds):
        if expand is not None:
            worlds += [_World(cond, s) for s in expand(cond, ctx.quick, per)]
            continue
        worlds += [_World(cond, s) for s in _samples(spec, per, ctx.seed, i)]
    return worlds


def _call_make_solver(twin: Any, model: Any, solver: str) -> Any:
    params = list(inspect.signature(twin.make_solver).parameters)
    return twin.make_solver(model, solver) if len(params) >= 2 else twin.make_solver(model)


def _make_pipeline(twin: Any, model: Any, solver: str) -> Any:
    if hasattr(twin, "make_pipeline"):
        return twin.make_pipeline(model, solver)
    return SV.make_pipeline(model, SV.SolverSettings(solver))


def build_model(spec: LawSpec, twin: Any, worlds: list[_World], ctx: RunContext) -> Any:
    """Finalize one Model with every planned world (`begin_world`/`end_world`, SPEC §3.10)."""
    b = newton.ModelBuilder()
    b.rigid_gap = RIGID_GAP_M  # never Newton's 0.1 m default (SPEC §4.2 builder gaps)
    SV.prepare_builder(ctx.solver, b)
    if hasattr(twin, "prepare_builder"):
        twin.prepare_builder(b, ctx.solver)
    gravity = getattr(twin, "world_gravity", lambda cond: None)
    for wd in worlds:
        b.begin_world(label=f"{wd.condition.id}", gravity=gravity(wd.condition))
        twin.build_world(b, wd.condition, wd.sample)
        b.end_world()
    SV.before_finalize(ctx.solver, b)
    return b.finalize(device=ctx.device)


def step_twin(spec: LawSpec, twin: Any, worlds: list[_World], ctx: RunContext) -> SteppedRun:
    """Build, initialise and step the twin for the run duration; records via `twin.record` every outer step."""
    model = build_model(spec, twin, worlds, ctx)
    solver = _call_make_solver(twin, model, ctx.solver)
    pipeline = _make_pipeline(twin, model, ctx.solver)
    contacts = _make_contacts(twin, pipeline)
    after_collide = getattr(twin, "after_collide", None)
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    if hasattr(twin, "init_state"):
        twin.init_state(model, s0)
    n_steps = max(1, int(round(ctx.duration_s / spec.dt)))
    rs = getattr(twin, "record_stride", 1)
    stride = int(rs(n_steps) if callable(rs) else rs)
    tlog = DeviceTwinLog(len(worlds), n_steps, spec.dt, ctx.device, stride=stride)
    obs = _observables(twin, solver, contacts)
    twin.record(0, s0, contacts, obs, tlog)
    h = spec.dt / spec.substeps
    for step in range(1, n_steps + 1):
        for sub in range(spec.substeps):
            s0.clear_forces()
            twin.drive(step - 1, (step - 1) * spec.dt + sub * h, s0, control)
            if pipeline is not None:
                pipeline.collide(s0, contacts)
                if after_collide is not None:
                    after_collide(s0, contacts)
            _solver_step(solver, s0, s1, control, contacts, h, obs)
            s0, s1 = s1, s0
        twin.record(step, s0, contacts, obs, tlog)
    wp.synchronize_device(ctx.device)
    return SteppedRun(model, s0, tlog, worlds, {"steps": n_steps, "worlds": len(worlds)})


def _make_contacts(twin: Any, pipeline: Any) -> Any:
    if pipeline is None:
        return None
    make = getattr(twin, "make_contacts", None)
    return make(pipeline) if make is not None else pipeline.contacts()


def _observables(twin: Any, solver: Any, contacts: Any) -> Any:
    kinds = getattr(twin, "observables", None)
    return solver.observables(kinds=kinds, contacts=contacts) if kinds else None


def _solver_step(solver: Any, s0: Any, s1: Any, control: Any, contacts: Any, h: float, obs: Any) -> None:
    if obs is None:
        solver.step(s0, s1, control, contacts, h)
    else:
        solver.step(s0, s1, control, contacts, h, observables=obs)


def _state_finite(state: Any) -> bool:
    for name in ("body_q", "body_qd", "particle_q", "particle_qd"):
        arr = getattr(state, name, None)
        if arr is not None and arr.shape[0] > 0 and not np.all(np.isfinite(arr.numpy())):
            return False
    return True


def judge_analytic(spec: LawSpec, twin: Any, values: Mapping[str, np.ndarray], refs: Mapping[str, np.ndarray],
                   conds: Sequence[Condition] = ()) -> tuple[MetricVerdict, ...]:
    """Per-metric `compare_errors` over worlds; a metric the twin did not produce is an `error` verdict.

    With a `reference(cond)` hook, `values` are raw simulated values and the errors are |value - reference|."""
    units: Mapping[str, str] = getattr(twin, "units", {})
    expected = _expected(twin, conds)
    out = []
    for name, tol in spec.metrics.items():
        unit, stat = units.get(name, "1"), spec.stat.get(name, "mean")
        if name not in values:
            out.append(C.placeholder_verdict(name, unit, "error", stat))
            continue
        vals, ref = np.asarray(values[name], np.float64), refs.get(name)
        if name in expected:
            vals, ref = np.abs(vals - expected[name]), expected[name]
        out.append(C.compare_errors(name, unit, vals, tol, ref, stat))
    return tuple(out)


def _expected(twin: Any, conds: Sequence[Condition]) -> dict[str, np.ndarray]:
    """Per-world closed-form values from the optional `reference(cond)` hook ({} without it)."""
    ref = getattr(twin, "reference", None)
    if ref is None or not conds:
        return {}
    rows = [dict(ref(c)) for c in conds]
    return {k: np.array([r[k] for r in rows], np.float64) for k in rows[0]}


_REQUIRED = ("build_world", "make_solver", "drive", "record", "scalars")


def missing_hooks(twin: Any, data: RigDataset | None) -> list[str]:
    """Required SimTwin methods the twin lacks (SPEC §3.10, SCR-WP6-1); `evaluate` twins need none."""
    if hasattr(twin, "evaluate"):
        return []
    need = _REQUIRED + (("conditions",) if data is None else ())
    return [n for n in need if not callable(getattr(twin, n, None))]


def _error_result(spec: LawSpec, ctx: RunContext, exc: BaseException, wall: float, extra: dict) -> LawResult:
    units: Mapping[str, str] = {}
    verdicts = tuple(C.placeholder_verdict(n, units.get(n, "1"), "error", spec.stat.get(n, "mean"))
                     for n in spec.metrics)
    msg = f"{type(exc).__name__}: {exc}"[:2000]
    log.warning("%s on %s/%s failed: %s", spec.id, ctx.solver, ctx.device, msg)
    return LawResult(spec.id, ctx.solver, ctx.device, verdicts, {"error": msg}, provenance(ctx.device, extra), wall,
                     not spec.production_dt)


def _incomplete_twin(spec: LawSpec, twin: Any, ctx: RunContext, missing: list[str], extra: dict) -> LawResult:
    reason = f"twin {type(twin).__name__} lacks SimTwin.{', '.join(missing)} (SPEC §3.10, SCR-WP6-1)"
    log.warning("%s: not_run: %s", spec.id, reason)
    verdicts = tuple(C.placeholder_verdict(n, "1", "not_run", spec.stat.get(n, "mean")) for n in spec.metrics)
    return LawResult(spec.id, ctx.solver, ctx.device, verdicts, {"not_run_reason": reason},
                     provenance(ctx.device, extra), 0.0, not spec.production_dt)


def run_law(spec: LawSpec, twin: Any, data: RigDataset | None, *, device: str, seed: int = 0,
            solver: str | None = None, quick: bool = False) -> LawResult:
    """Run one law on one solver and device; never raises for simulation failures (they become `error`)."""
    name = solver or (spec.solvers[0] if spec.solvers else "mujoco")
    ctx = RunContext(spec, device, seed, name, quick)
    extra = {"law": spec.id, "suite": spec.suite, "solver": name, "quick": str(quick), "seed": str(seed)}
    config = {"solver": name, "dt": spec.dt, "substeps": spec.substeps, "duration_s": ctx.duration_s,
              "quick": quick, "seed": seed}
    if hasattr(twin, "settings_record"):
        config["solver_settings"] = twin.settings_record(name)
    data_rec = {} if data is None else {"sha256": data.sha256, "dataset_id": data.card.dataset_id,
                                        "protocol": data.card.protocol_version,
                                        "path": str(getattr(data, "path", "") or "")}
    if spec.suite == "rig" and data is None and not hasattr(twin, "evaluate"):
        raise LawError(f"{spec.id}: rig law needs a RigDataset")
    missing = missing_hooks(twin, data)
    if missing:
        res = _incomplete_twin(spec, twin, ctx, missing, extra)
        return dataclasses.replace(res, config=config, data=data_rec)
    t0 = time.perf_counter()
    try:
        verdicts, monitors = _run_inner(spec, twin, data, ctx)
    except Exception as exc:  # noqa: BLE001 - the result records the failure; it is never hidden
        res = _error_result(spec, ctx, exc, time.perf_counter() - t0, extra)
        return dataclasses.replace(res, config=config, data=data_rec)
    config["worlds"] = monitors.get("worlds", 0)
    if hasattr(twin, "settings_record"):
        config["solver_settings"] = twin.settings_record(name)  # as run (e.g. the routed integrator)
    return LawResult(spec.id, name, device, verdicts, monitors, provenance(device, extra),
                     time.perf_counter() - t0, not spec.production_dt, config=config, data=data_rec)


def _run_inner(spec: LawSpec, twin: Any, data: RigDataset | None, ctx: RunContext
               ) -> tuple[tuple[MetricVerdict, ...], dict[str, Any]]:
    if hasattr(twin, "evaluate"):
        verdicts = tuple(twin.evaluate(ctx))
        return verdicts, dict(getattr(twin, "monitors", {}))
    conds = data.conditions if data is not None else tuple(twin.conditions(ctx.quick))
    if data is not None and hasattr(twin, "bind_calib"):
        twin.bind_calib(data.card, data.split("calib"))  # fit-vs-validate split enforced here: calib trials only
    worlds = plan_worlds(spec, conds, ctx, getattr(twin, "expand", None))
    run = step_twin(spec, twin, worlds, ctx)
    run.monitors["finite"] = _state_finite(run.state)
    values = twin.scalars(run.log)
    if data is None:
        refs = twin.references(run.log) if hasattr(twin, "references") else {}
        verdicts = judge_analytic(spec, twin, values, refs, [w.condition for w in worlds])
    else:
        from alloy.bench.rigjudge import judge_rig  # local import keeps the analytic path light
        verdicts, calib = judge_rig(spec, twin, values, data, [w.condition.id for w in worlds], run.log)
        run.monitors["calib_error"] = calib
    run.monitors.update(dict(getattr(twin, "monitors", {})))
    return verdicts, run.monitors


TwinFactory = Callable[[], Any]

__all__ = ["run_law", "RunContext", "SteppedRun", "plan_worlds", "build_model", "step_twin", "judge_analytic",
           "TwinFactory"]
