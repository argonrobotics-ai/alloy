# SPDX-License-Identifier: Apache-2.0
"""P3-2 SPH liquid laws A13.hydrostatic, A13.archimedes, A14.dam_break: port packs and scorer (Alloy venv).

    PYTHONPATH=. python bench/xbench/ours/fluid_pack.py pack [OUT=bench/xbench/ours/packs/laws]
    PYTHONPATH=. python bench/xbench/ours/fluid_pack.py score RUN_DIR [--laws ...]
    PYTHONPATH=. python bench/xbench/ours/fluid_pack.py alloy RUN_DIR [--device cuda:0]   # Alloy beside

``--quick`` uses the twins' quick conditions (tests). Pack ``packs/laws/<law>/<condition>/``, each condition built as
Alloy's runner builds it (``plan_worlds`` + ``build_model`` + the twin's ``make_solver``, route ``sph``):
``particles.npz`` (``particle_q`` [P, 3] float32: world 0's liquid particles, exactly the model's float32 values) and
``fluid.json``: SPH parameters with their meaning (``sph``), particles (count, mass, rest density, viscosity), the
tank's static wall boxes and inner box, the floating body (mass, COM, inertia, box, pose), gravity, dt / steps,
``record_every``, the particle frames the scorer reads, and ``export``: the npz keys per law and what they mean.
Engines run their own SPH and map Alloy's parameters (record the mapping in ``meta.settings``); they place exactly
these particles at rest (no resampling) and export ``RUN_DIR/<law>/<condition>.npz`` (README contract, W = 1).

Score: Alloy's twin on the engine's data. The log runs on the engine's frames (dt = meta dt x record_every, the
twin's ``record_stride``); per column the frame goes into a Newton state (particles, the archimedes box) and the
liquid's wall impulse into the solver's shape meter (fixed point 2^50, as Alloy accumulates it); then the twin's own
``record``, ``scalars`` and ``judge_analytic``. Liquid volume (LIQ-2) is Alloy's SPH estimate of the engine's
particles (``SolverDFSPH.density_ratio``, then the twin's ``liquid_volume``). At column k Alloy's volume uses the
density its step k computed at the step's start, i.e. frame k - 1 (frame 0 for k = 0); the scorer applies that lag
to every engine. Frame 0 must reproduce the pack (count, positions within 1 um; a permuted order is accepted, every
metric is order-free) or the law is ``port_mismatch``. Columns whose particle frame was not exported get NaN.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]

from fluid_spec import EXPORT, SCENE, sph_params  # noqa: E402
from law_pack import law_twin  # noqa: E402
from score_laws import _frames, _load, _verdict_rows, _world_bodies  # noqa: E402

log = logging.getLogger("xbench.ours")
LAWS = ("A13.hydrostatic", "A13.archimedes", "A14.dam_break")
SOLVER = "sph"  # the laws' only capability (LawSpec.solvers)
OUT = HERE / "packs" / "laws"
SCHEMA = "alloy.xbench.ours.fluid/1"
TOL_M = 1.0e-6  # frame 0 vs the pack [m]
CHANNEL = {"A14.dam_break": "front", "A13.archimedes": "volume", "A13.hydrostatic": "volume"}  # from particle_q
KEYS = {"A14.dam_break": ("t", "particle_q"), "A13.archimedes": ("t", "particle_q", "body_q", "body_qd", "body_names"),
        "A13.hydrostatic": ("t", "particle_q", "wall_impulse")}  # npz keys the scorer needs


# ----------------------------------------------------------------------------------------------------- building
def _setup(law_id: str, device: str = "cpu", quick: bool = False, duration_s: float | None = None,
           cond_index: int | None = None):
    """(spec, twin, ctx, worlds) as Alloy's runner plans them; ``duration_s`` shortens the run (tests)."""
    from alloy.bench.runner import RunContext, plan_worlds

    spec, twin = law_twin(law_id)
    if duration_s is not None:
        spec = dataclasses.replace(spec, duration_s=duration_s, quick={**spec.quick, "duration_s": duration_s})
    ctx = RunContext(spec, device, 0, SOLVER, quick)
    conds = twin.conditions(quick)
    conds = conds if cond_index is None else conds[cond_index:cond_index + 1]
    return spec, twin, ctx, plan_worlds(spec, conds, ctx, getattr(twin, "expand", None))


def _build(law_id: str, device: str = "cpu", quick: bool = False, cond_index: int | None = None):
    """Setup plus model, solver and the initial state (before the twin's pressure prime, which moves nothing)."""
    from alloy._vendor import newton

    from alloy.bench.runner import _call_make_solver, build_model

    spec, twin, ctx, worlds = _setup(law_id, device, quick, cond_index=cond_index)
    model = build_model(spec, twin, worlds, ctx)
    solver = _call_make_solver(twin, model, SOLVER)
    s0 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    return spec, twin, ctx, worlds, model, solver, s0


def _steps(spec, ctx) -> int:
    return max(1, int(round(ctx.duration_s / spec.dt)))  # step_twin's count


def _plan(law_id: str, n_steps: int, stride: int) -> dict:
    """Engine cadence: ``record_every`` (physics steps per frame) and the particle frames the scorer reads."""
    if law_id == "A14.dam_break":
        return {"record_every": stride, "particle_frames": "all"}  # the twin's own columns
    if law_id == "A13.archimedes":
        return {"record_every": 1, "particle_frames": [0]}
    last = n_steps // stride * stride  # the twin's last column (volume read on the frame before it)
    return {"record_every": 1, "particle_frames": sorted({0, max(last - 1, 0), last})}


# ----------------------------------------------------------------------------------------------------- pack
def _named_walls(model, w: int) -> dict[str, int]:
    """Static boxes of world w (an open box tank) by side: floor, x_lo, x_hi, y_lo, y_hi -> shape index."""
    sw, sb, xf = model.shape_world.numpy(), model.shape_body.numpy(), model.shape_transform.numpy()
    ids = [s for s in range(int(model.shape_count)) if sw[s] == w and sb[s] < 0]
    if len(ids) != 5:
        raise ValueError(f"world {w}: expected the 5 boxes of an open tank, found {len(ids)} static shapes")
    floor = min(ids, key=lambda s: xf[s, 2])
    xs = sorted((s for s in ids if s != floor), key=lambda s: xf[s, 0])
    ys = sorted(xs[1:-1], key=lambda s: xf[s, 1])
    return {"floor": floor, "x_lo": xs[0], "x_hi": xs[-1], "y_lo": ys[0], "y_hi": ys[1]}


def _box(model, s: int) -> dict:
    xf, half = model.shape_transform.numpy()[s].astype(np.float64), model.shape_scale.numpy()[s].astype(np.float64)
    return {"shape": s, "type": "box", "center": xf[:3].tolist(), "quat_xyzw": xf[3:].tolist(),
            "half_extents": half.tolist()}


def _tank(model, w: int) -> dict:
    names = _named_walls(model, w)
    walls = {k: _box(model, s) for k, s in names.items()}
    c = {k: np.array(v["center"]) for k, v in walls.items()}
    hx = {k: np.array(v["half_extents"]) for k, v in walls.items()}
    lo = [c["x_lo"][0] + hx["x_lo"][0], c["y_lo"][1] + hx["y_lo"][1], c["floor"][2] + hx["floor"][2]]
    hi = [c["x_hi"][0] - hx["x_hi"][0], c["y_hi"][1] - hx["y_hi"][1], c["x_lo"][2] + hx["x_lo"][2]]
    return {"walls": walls, "inner_lo": lo, "inner_hi": hi, "open_top": True, "wall_thickness": 2 * hx["floor"][2],
            "note": "static boxes (float32 model values), SDF boundaries; they tile the solid without overlap. The "
                    "liquid sees only these walls (and the box); there is no gate and no ground plane"}


def _bodies(model, s0, w: int) -> list[dict]:
    """The free bodies of world w: mass, COM, inertia (body frame, about the COM), box shapes, initial pose."""
    if not int(model.body_count):
        return []
    sb = model.shape_body.numpy()
    q, qd = s0.body_q.numpy(), s0.body_qd.numpy()
    mass, com, inertia = model.body_mass.numpy(), model.body_com.numpy(), model.body_inertia.numpy()
    out = []
    for b, name in _world_bodies(model).get(w, []):
        shapes = [_box(model, int(s)) for s in np.flatnonzero(sb == b)]
        vol = sum(8.0 * float(np.prod(sh["half_extents"])) for sh in shapes)
        out.append({"name": name, "mass": float(mass[b]), "com": com[b].tolist(), "inertia": inertia[b].tolist(),
                    "density": float(mass[b]) / vol, "shapes": shapes, "q": q[b].tolist(), "qd": qd[b].tolist(),
                    "motion": "free 6-DOF: gravity + the liquid's wrench (two-way coupling, integrated inside the "
                              "SPH step); no rigid contact with anything (it touches only the liquid)"})
    return out


def _lattice_offset(q: np.ndarray, s: float) -> float:
    """Largest distance of a packed particle from its 2r-lattice cell centre (float32 rounding only, ~1e-9 m)."""
    x = q.astype(np.float64)
    return float(np.abs(x - (np.round(x / s - 0.5) + 0.5) * s).max())


def _particles(model, q: np.ndarray, spacing: float) -> dict:
    return {"file": "particles.npz", "key": "particle_q", "count": int(len(q)),
            "mass": float(model.particle_mass.numpy()[0]), "rest_density": float(model.sph.rest_density.numpy()[0]),
            "viscosity_Pa_s": float(model.sph.viscosity.numpy()[0]), "velocity": [0.0, 0.0, 0.0],
            "lattice_offset_max_m": _lattice_offset(q, spacing),
            "note": "place exactly these (Alloy's float32 values, pack order), at rest; never resample. They are "
                    "2r-lattice cell centres (i + 1/2) 2r from the tank's inner corner, minus the cells under the box "
                    "(archimedes); offsets are float32 rounding only"}


def write_condition(law_id: str, out: Path, idx: int = 0, quick: bool = False) -> dict:
    spec, twin, ctx, worlds, model, solver, s0 = _build(law_id, quick=quick, cond_index=idx)
    cond, cfg, n = worlds[0].condition, twin.config, _steps(spec, ctx)
    plan = _plan(law_id, n, twin.record_stride(n))
    d = out / law_id / cond.id
    d.mkdir(parents=True, exist_ok=True)
    q = model.particle_q.numpy()[model.particle_world.numpy() == 0]
    np.savez_compressed(d / "particles.npz", particle_q=q)
    rec = {"schema": SCHEMA, "law": law_id, "condition": cond.id, "scene": SCENE[law_id],
           "params": dict(cond.params), "dt": spec.dt, "substeps": spec.substeps, "duration_s": ctx.duration_s,
           "steps": n, **plan, "frames": n // plan["record_every"] + 1, "gravity": model.gravity.numpy()[0].tolist(),
           "worlds": len(worlds), "alloy_solver": SOLVER, "sph": sph_params(cfg, solver, model, spec.dt),
           "particles": _particles(model, q, cfg.spacing),
           "tank": _tank(model, 0), "bodies": _bodies(model, s0, 0), "export": EXPORT[law_id],
           "export_common": "npz per README contract: meta (dt, record_every, settings incl. the SPH parameter "
                            "mapping), t [T]; particle_frame [Tp] int (indices into t) when particle_q holds only "
                            "some frames",
           "alloy_settings": twin.settings_record(SOLVER)}
    if law_id == "A13.hydrostatic":
        rec["export"] = dict(EXPORT[law_id], wall_shape=_named_walls(model, 0)["x_lo"])
    (d / "fluid.json").write_text(json.dumps(rec, indent=1, default=str))
    return {"id": cond.id, "params": dict(cond.params), "particles": int(len(q))}


def pack(out: Path, quick: bool = False, laws=LAWS) -> None:
    for law_id in laws:
        spec, twin = law_twin(law_id)
        conds = [write_condition(law_id, out, i, quick) for i in range(len(twin.conditions(quick)))]
        (out / law_id / "law.json").write_text(json.dumps({
            "law": law_id, "dt": spec.dt, "duration_s": spec.duration_s, "quick": quick, "stat": dict(spec.stat),
            "metrics": {k: {"abs": t.abs, "rel": t.rel, "source": t.source} for k, t in spec.metrics.items()},
            "conditions": conds}, indent=1))
        log.info("%s: packs in %s", law_id, out / law_id)


# ----------------------------------------------------------------------------------------------------- score
def _per_world(worlds, runs: dict[str, dict]) -> list[tuple[dict, int]]:
    """(run, index on its W axis) per model world, in plan order; adds ``_pslot`` (frame -> particle slot)."""
    seen: dict[str, int] = {}
    out = []
    for wd in worlds:
        cid = wd.condition.id
        j = seen[cid] = seen.get(cid, -1) + 1
        r = runs[cid]
        if "particle_q" in r and "_pslot" not in r:
            frames = r.get("particle_frame", np.arange(r["particle_q"].shape[0]))
            r["_pslot"] = {int(f): i for i, f in enumerate(frames)}
        out.append((r, j))
    return out


def _particle_dev(got: np.ndarray, ref: np.ndarray) -> tuple[float, str]:
    """Max |got - ref| [m] in pack order, or as sets when only the order differs (sorted by 0.1 mm bins, so the
    2 mm lattice sites decide the order and um noise does not)."""
    dev = float(np.abs(got - ref).max())
    if dev <= TOL_M:
        return dev, "pack"
    key = lambda a: a[np.lexsort(np.round(a / 1.0e-4).T[::-1])]  # noqa: E731
    dev_set = float(np.abs(key(got) - key(ref)).max())
    return (dev_set, "permuted") if dev_set <= TOL_M else (dev, "pack")


def _port_check(model, s0, per_world, bq: np.ndarray | None) -> dict:
    """Frame 0 vs Alloy's initial state: particle count and positions, body poses (within TOL_M)."""
    pw, q0 = model.particle_world.numpy(), model.particle_q.numpy().astype(np.float64)
    rows = []
    for g, (run, j) in enumerate(per_world):
        slot = run.get("_pslot", {}).get(0)
        if slot is None:
            return {"ok": False, "reason": f"world {g}: no particle_q at frame 0"}
        got, ref = run["particle_q"][slot, j].astype(np.float64), q0[pw == g]
        if got.shape != ref.shape:
            return {"ok": False, "reason": f"world {g}: {len(got)} particles, the pack has {len(ref)}"}
        dev, order = _particle_dev(got, ref)
        rows.append({"particles": len(got), "max_dev_m": dev, "order": order})
    body_dev = 0.0
    if bq is not None and len(bq[0]):
        a, b = bq[0].astype(np.float64), s0.body_q.numpy().astype(np.float64)
        quat = np.minimum(np.abs(a[:, 3:] - b[:, 3:]).max(1), np.abs(a[:, 3:] + b[:, 3:]).max(1))
        body_dev = float(max(np.abs(a[:, :3] - b[:, :3]).max(), quat.max()))
    ok = all(r["max_dev_m"] <= TOL_M for r in rows) and body_dev <= TOL_M
    return {"ok": ok, "worlds": rows, "body_max_dev": body_dev, "tol_m": TOL_M}


class _Replay:
    """Writes engine frames into a Newton state / the solver and calls the twin's own ``record``."""

    def __init__(self, law_id, twin, model, solver, per_world, bq, bqd, tlog):
        from alloy._vendor import newton

        self.law_id, self.twin, self.model, self.solver = law_id, twin, model, solver
        self.per_world, self.bq, self.bqd, self.tlog = per_world, bq, bqd, tlog
        self.state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state)

    def particles(self, frame: int) -> bool:
        parts = []
        for run, j in self.per_world:
            slot = run.get("_pslot", {}).get(frame)
            if slot is None:
                return False
            parts.append(run["particle_q"][slot, j])
        self.state.particle_q.assign(np.concatenate(parts).astype(np.float32))
        return True

    def density(self, frame: int) -> bool:
        """Alloy's density ratio of ``frame`` into the solver (what ``liquid_volume`` reads)."""
        if not self.particles(frame):
            return False
        self.solver.density_ratio(self.state)
        return True

    def bodies(self, frame: int) -> None:
        self.state.body_q.assign(self.bq[frame])
        self.state.body_qd.assign(self.bqd[frame])

    def meter(self, frame: int) -> None:
        from alloy.solvers.sph.bodies import IMP_SCALE_HOST

        m = np.zeros((int(self.model.shape_count), 3), np.int64)
        for g, (run, j) in enumerate(self.per_world):
            m[self.twin._walls[g]] = np.round(run["wall_impulse"][frame, j].astype(np.float64) * IMP_SCALE_HOST)
        self.solver._meter_j.assign(m)

    def feed(self, k: int) -> bool:
        """Frame k into state / solver per law; False when its particle-derived channel has no data."""
        if self.law_id == "A14.dam_break":
            return self.particles(k)
        if self.law_id == "A13.archimedes":
            self.bodies(k)
            return self.density(0) if k == 0 else True  # the twin reads the volume of column 0 only
        self.meter(k)
        return self.density(max(k - 1, 0))

    def run(self) -> list[int]:
        """Every logged column through the twin's ``record``; returns the columns blanked to NaN."""
        bad = []
        for k in range(self.tlog.n_steps + 1):
            col = self.tlog.column(k)
            if col < 0:
                continue
            ok = self.feed(k)
            self.twin.record(k, self.state, None, None, self.tlog)
            if not ok:
                bad.append(col)
        if bad:
            ch = self.tlog.channel(CHANNEL[self.law_id])
            a = ch.numpy().copy()
            a[:, bad] = np.nan
            ch.assign(a)
        return bad


def score_law(law_id: str, run_dir: Path, device: str = "cpu", quick: bool = False) -> dict:
    """Verdicts of one law from the engine's per-condition npz (``not_run`` if any condition is missing)."""
    from alloy.bench.runner import judge_analytic
    from alloy.bench.twin import DeviceTwinLog

    _, twin0 = law_twin(law_id)
    paths = {c.id: run_dir / law_id / f"{c.id}.npz" for c in twin0.conditions(quick)}
    missing = [p.name for p in paths.values() if not p.exists()]
    if missing:
        return {"law": law_id, "status": "not_run", "missing": missing}
    runs = {cid: _load(p) for cid, p in paths.items()}
    lacks = {cid: [k for k in KEYS[law_id] if k not in r] for cid, r in runs.items()}
    if any(lacks.values()):
        return {"law": law_id, "status": "error", "error": f"npz lacks keys {lacks} (fluid.json 'export')"}
    spec, twin, ctx, worlds, model, solver, s0 = _build(law_id, device, quick)
    per_world = _per_world(worlds, runs)
    bq, bqd = _frames(runs, worlds, model) if int(model.body_count) else (None, None)
    port = _port_check(model, s0, per_world, bq)
    first = per_world[0][0]
    if not port["ok"]:
        return {"law": law_id, "status": "port_mismatch", "port_check": port, "engine": first["meta"].get("engine")}
    n = len(first["t"]) - 1
    dt = float(first["meta"]["dt"]) * int(first["meta"].get("record_every", 1))
    tlog = DeviceTwinLog(len(worlds), n, dt, device, stride=twin.record_stride(n))
    bad = _Replay(law_id, twin, model, solver, per_world, bq, bqd, tlog).run()
    rows = _verdict_rows(judge_analytic(spec, twin, twin.scalars(tlog), {}, [w.condition for w in worlds]))
    return {"law": law_id, "status": "pass" if all(r["verdict"] == "pass" for r in rows) else "fail",
            "metrics": rows, "dt": dt, "law_dt": spec.dt, "frames": n + 1, "columns_without_particles": len(bad),
            "port_check": port, "engine": first["meta"].get("engine"), "settings": first["meta"].get("settings"),
            "monitors": json.loads(json.dumps(twin.monitors, default=str))}


def score(run_dir: Path, device: str = "cpu", quick: bool = False, laws=LAWS) -> list[dict]:
    out = []
    for law_id in laws:
        try:
            res = score_law(law_id, run_dir, device, quick)
        except Exception as exc:  # noqa: BLE001 - a broken run is a row, not a crash of the whole scoring
            log.exception("%s: scoring failed", law_id)
            res = {"law": law_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        log.info("%s: %s", law_id, res["status"])
        out.append(res)
    return out


# ----------------------------------------------------------------------------------------------------- Alloy beside
class _Capture:
    """Wraps the twin's ``record`` to keep the frames an engine exports (``_plan``) from Alloy's own run."""

    def __init__(self, law_id: str, twin, plan: dict):
        self.law_id, self.twin, self.every, self.pf = law_id, twin, plan["record_every"], plan["particle_frames"]
        self.frames, self.q, self.pframes, self.bq, self.bqd, self.jw = 0, [], [], [], [], []

    def wrap(self, inner):
        def record(step, state, contacts, obs, tlog):
            if step % self.every == 0:
                self._grab(step // self.every, state)
            inner(step, state, contacts, obs, tlog)
        return record

    def _grab(self, f: int, state) -> None:
        self.frames = f + 1
        if self.pf == "all" or f in self.pf:
            self.pframes.append(f)
            self.q.append(state.particle_q.numpy().copy())
        if self.law_id == "A13.archimedes":
            self.bq.append(state.body_q.numpy().copy())
            self.bqd.append(state.body_qd.numpy().copy())
        if self.law_id == "A13.hydrostatic":
            self.jw.append(self.twin.solver.shape_impulse()[self.twin._walls])

    def write(self, d: Path, worlds, model, meta: dict) -> None:
        """One npz per condition, W axis = that condition's worlds in plan order."""
        pw, by_world = model.particle_world.numpy(), _world_bodies(model)
        groups: dict[str, list[int]] = {}
        for g, w in enumerate(worlds):
            groups.setdefault(w.condition.id, []).append(g)
        d.mkdir(parents=True, exist_ok=True)
        for cid, gs in groups.items():
            arrays = {"meta": np.array(json.dumps(dict(meta, condition=cid, worlds=len(gs)))),
                      "t": np.arange(self.frames) * meta["dt"] * self.every,
                      "particle_q": np.stack([np.stack([q[pw == g] for g in gs]) for q in self.q]).astype(np.float32)}
            if self.pf != "all":
                arrays["particle_frame"] = np.array(self.pframes, np.int32)
            if self.bq:
                idx = np.array([[b for b, _ in by_world[g]] for g in gs])
                arrays.update(body_names=np.array([n for _, n in by_world[gs[0]]]),
                              body_q=np.stack(self.bq)[:, idx].astype(np.float32),
                              body_qd=np.stack(self.bqd)[:, idx].astype(np.float32))
            if self.jw:
                arrays["wall_impulse"] = np.stack(self.jw)[:, gs].astype(np.float64)
            np.savez_compressed(d / f"{cid}.npz", **arrays)


def alloy_law(law_id: str, out_dir: Path, device: str, quick: bool = False, duration_s: float | None = None) -> dict:
    """Alloy beside: the law through ``step_twin`` as ``python -m alloy.bench run`` does, frames captured into the
    engine npz format; returns the direct verdicts."""
    from alloy.bench.runner import judge_analytic, step_twin
    from alloy_t12 import versions

    spec, twin, ctx, worlds = _setup(law_id, device, quick, duration_s)
    n = _steps(spec, ctx)
    plan = _plan(law_id, n, twin.record_stride(n))
    cap = _Capture(law_id, twin, plan)
    twin.record = cap.wrap(twin.record)
    t0 = time.perf_counter()
    run = step_twin(spec, twin, worlds, ctx)
    wall = time.perf_counter() - t0
    rows = _verdict_rows(judge_analytic(spec, twin, twin.scalars(run.log), {}, [w.condition for w in worlds]))
    meta = {"engine": "alloy", "engine_version": versions()["alloy"], "versions": versions(), "test": law_id,
            "dt": spec.dt, "substeps": spec.substeps, "record_every": plan["record_every"], "row": "recommended",
            "settings": twin.settings_record(SOLVER), "wall_s": wall, "quick": quick,
            "notes": "Alloy default route (SolverDFSPH) via alloy.bench.runner.step_twin"}
    cap.write(out_dir / law_id, worlds, run.model, meta)
    return {"law": law_id, "status": "pass" if all(r["verdict"] == "pass" for r in rows) else "fail",
            "metrics": rows, "wall_s": wall, "steps": n, "monitors": json.loads(json.dumps(twin.monitors, default=str))}


def alloy_run(out_dir: Path, device: str, quick: bool = False, duration_s: float | None = None,
              laws=LAWS) -> list[dict]:
    out = []
    for law_id in laws:
        out.append(alloy_law(law_id, out_dir, device, quick, duration_s))
        log.info("%s: %s (%.1f s)", law_id, out[-1]["status"], out[-1]["wall_s"])
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("pack", "score", "alloy"))
    p.add_argument("path", type=Path, nargs="?", default=OUT)
    p.add_argument("--laws", default=",".join(LAWS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--quick", action="store_true", help="the twins' quick conditions and durations")
    a = p.parse_args()
    laws = tuple(a.laws.split(","))
    if a.cmd == "pack":
        pack(a.path, a.quick, laws)
        return 0
    a.path.mkdir(parents=True, exist_ok=True)
    if a.cmd == "alloy":
        res = alloy_run(a.path, a.device, a.quick, laws=laws)
        (a.path / "fluid_alloy_direct.json").write_text(json.dumps(res, indent=1, default=str))
    else:
        res = score(a.path, a.device, a.quick, laws)
        (a.path / "fluid_scored.json").write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
