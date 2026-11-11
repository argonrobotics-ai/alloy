# SPDX-License-Identifier: Apache-2.0
"""P3-2 A15 (uniaxial Neo-Hookean) port pack and scorer (Alloy venv).

    PYTHONPATH=. python bench/xbench/ours/soft_pack.py pack [--out bench/xbench/ours/packs/laws]
    PYTHONPATH=. python bench/xbench/ours/soft_pack.py score RUN_DIR
    PYTHONPATH=. python bench/xbench/ours/soft_pack.py alloy RUN_DIR [--device cuda:0]   # Alloy beside

Pack ``packs/laws/A15.neohookean/<condition>/soft.json``: the pad Alloy runs on its default route (VBD): rest node
positions and tets (Newton's soft grid, world 0), top / bottom node ids, the dead load per top node (eased by
smoothstep(t / 0.25 s)), the material (Lame mu, lambda; E, nu; density; Newton VBD's stable Neo-Hookean), a
frictionless floor at z = 0 and zero gravity, dt and steps. An engine with FEM tets builds the same pad (explicit
tets if it accepts them, else its own mesher on the 40 x 40 x 10 mm box), applies the same total load spread over
its top face by area, and exports ``particle_q`` [T, W, P, 3] (``particle_rest`` [P, 3] too if its nodes differ).

Score: Alloy's ``NeoHookeanTwin`` (model, ``on_model`` indices, ``scalars``). When the engine's nodes are Newton's
(same count, rest positions within 1 um) the frames go through the twin's own ``record``; otherwise the twin's two
log channels are filled from the engine's own top / bottom faces (nodes within 1 um of the rest max / min z) and its
rest height replaces the twin's, and the twin's ``scalars`` judges them (same formula and tolerance).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]

from law_pack import law_twin  # noqa: E402

log = logging.getLogger("xbench.ours")
LAW_ID = "A15.neohookean"
SOLVER = "vbd"  # Alloy's default route for soft solids (alloy.bench.default_route)
OUT = HERE / "packs" / "laws"


def _run(conds, device: str = "cpu"):
    from alloy.bench.runner import RunContext, _call_make_solver, build_model, plan_worlds

    spec, twin = law_twin(LAW_ID)
    ctx = RunContext(spec, device, 0, SOLVER, False)
    worlds = plan_worlds(spec, conds, ctx, getattr(twin, "expand", None))
    model = build_model(spec, twin, worlds, ctx)
    _call_make_solver(twin, model, SOLVER)
    return spec, twin, worlds, model


def _world0(model, twin) -> dict:
    from alloy.bench.analytic import neohookean as NH

    pw = model.particle_world.numpy()
    ids = np.flatnonzero(pw == 0)
    x = model.particle_q.numpy()[ids].astype(np.float64)
    tets = model.tet_indices.numpy()
    tets = tets[np.isin(tets[:, 0], ids)] - ids[0]
    top = twin._top.numpy()[twin._top_rng[0].numpy()[0]:twin._top_rng[1].numpy()[0]] - ids[0]
    bot = twin._bot.numpy()[twin._bot_rng[0].numpy()[0]:twin._bot_rng[1].numpy()[0]] - ids[0]
    force = twin._force.numpy()[twin._top_rng[0].numpy()[0]:twin._top_rng[1].numpy()[0]]
    mu, lam = NH.MU_L, NH.LAM_L
    return {"rest": x.tolist(), "tets": tets.tolist(), "top": top.tolist(), "bottom": bot.tolist(),
            "top_force_z": force.tolist(), "material": {"mu": mu, "lambda": lam, "E": mu * (3 * lam + 2 * mu) / (lam + mu),
            "nu": lam / (2 * (lam + mu)), "density": NH.DENSITY, "model": "stable Neo-Hookean (Smith et al. 2018) as "
            "Newton VBD (newton@ac7f7830 particle_vbd_kernels.py:228-262)"}, "box_m": [NH.WIDTH, NH.WIDTH, NH.HEIGHT],
            "cells": list(NH.CELLS), "particle_radius": NH.RADIUS, "ease_s": NH.T_EASE,
            "load": "dead load per top node = top_force_z * smoothstep(t / ease_s), -z (P < 0 compresses)",
            "floor": {"z": 0.0, "mu": 0.0, "note": "frictionless; Alloy's floor penalty ke 1e4 N/m, kd 1 N s/m"}}


def pack(out: Path) -> None:
    spec, twin = law_twin(LAW_ID)
    for cond in twin.conditions(False):
        _, tw, worlds, model = _run([cond])
        d = out / LAW_ID / cond.id
        d.mkdir(parents=True, exist_ok=True)
        rec = {"schema": "alloy.xbench.ours.soft/1", "law": LAW_ID, "condition": cond.id, "params": dict(cond.params),
               "dt": spec.dt, "substeps": spec.substeps, "duration_s": spec.duration_s,
               "steps": int(round(spec.duration_s / spec.dt)), "gravity": [0.0, 0.0, 0.0], "worlds": len(worlds),
               "alloy_solver": SOLVER, **_world0(model, tw)}
        (d / "soft.json").write_text(json.dumps(rec, indent=1))
    (out / LAW_ID / "law.json").write_text(json.dumps({"law": LAW_ID, "dt": spec.dt, "duration_s": spec.duration_s,
        "metrics": {k: {"abs": t.abs, "rel": t.rel, "source": t.source} for k, t in spec.metrics.items()},
        "conditions": [{"id": c.id, "params": dict(c.params)} for c in twin.conditions(False)]}, indent=1))
    log.info("%s: packs in %s", LAW_ID, out / LAW_ID)


def _faces(rest: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Mean z of the top and bottom rest faces over frames ([T] each) and the rest height."""
    z0 = rest[:, 2]
    top, bot = np.abs(z0 - z0.max()) < 1e-6, np.abs(z0 - z0.min()) < 1e-6
    return q[:, top, 2].mean(axis=1), q[:, bot, 2].mean(axis=1), float(z0.max() - z0.min())


def score(run_dir: Path, device: str = "cpu") -> dict:
    from alloy._vendor import newton

    from alloy.bench.runner import judge_analytic
    from alloy.bench.twin import DeviceTwinLog
    from score_laws import _load, _verdict_rows

    spec, twin = law_twin(LAW_ID)
    conds = twin.conditions(False)
    paths = [run_dir / LAW_ID / f"{c.id}.npz" for c in conds]
    if not all(p.exists() for p in paths):
        return {"law": LAW_ID, "status": "not_run", "missing": [p.name for p in paths if not p.exists()]}
    runs = [_load(p) for p in paths]
    spec, twin, worlds, model = _run(conds, device)
    n = runs[0]["particle_q"].shape[0] - 1
    dt = float(runs[0]["meta"]["dt"]) * int(runs[0]["meta"].get("record_every", 1))
    tlog = DeviceTwinLog(len(worlds), n, dt, device, stride=twin.record_stride(n))
    pw = model.particle_world.numpy()
    rest_newton = model.particle_q.numpy()
    exact = all(r["particle_q"].shape[2] == int(np.sum(pw == w)) and np.allclose(
        r["particle_q"][0, 0], rest_newton[pw == w], atol=1e-6) for w, r in enumerate(runs))
    if exact:
        state = model.state()
        for k in range(n + 1):
            state.particle_q.assign(np.concatenate([r["particle_q"][k, 0] for r in runs]).astype(np.float32))
            twin.record(k, state, None, None, tlog)
    else:
        import warp as wp

        top, bot = tlog.channel("z_top", wp.float64), tlog.channel("z_bot", wp.float64)
        cols = np.array([k for k in range(n + 1) if tlog.column(k) >= 0])
        faces = [_faces(r.get("particle_rest", r["particle_q"][0, 0]), r["particle_q"][cols, 0]) for r in runs]
        top.assign(np.stack([f[0] for f in faces]))
        bot.assign(np.stack([f[1] for f in faces]))
        twin._h0 = faces[0][2]
    values = twin.scalars(tlog)
    rows = _verdict_rows(judge_analytic(spec, twin, values, {}, [w.condition for w in worlds]))
    return {"law": LAW_ID, "status": "pass" if all(r["verdict"] == "pass" for r in rows) else "fail",
            "metrics": rows, "path": "twin record (same nodes)" if exact else "engine faces (own mesh)", "dt": dt,
            "law_dt": spec.dt, "monitors": json.loads(json.dumps(twin.monitors, default=str)),
            "engine": runs[0]["meta"].get("engine")}


def alloy_run(out_dir: Path, device: str) -> dict:
    """Alloy beside: the law on its default route (VBD), ``particle_q`` every step per condition npz; direct verdict."""
    import time

    from alloy.bench.runner import RunContext, judge_analytic, plan_worlds, step_twin
    from alloy_t12 import versions
    from score_laws import _verdict_rows

    spec, twin = law_twin(LAW_ID)
    ctx = RunContext(spec, device, 0, SOLVER, False)
    worlds = plan_worlds(spec, twin.conditions(False), ctx, getattr(twin, "expand", None))
    frames: list[np.ndarray] = []
    inner = twin.record

    def record(step, state, contacts, obs, tlog):
        frames.append(state.particle_q.numpy().copy())
        inner(step, state, contacts, obs, tlog)

    twin.record = record
    t0 = time.perf_counter()
    run = step_twin(spec, twin, worlds, ctx)
    wall = time.perf_counter() - t0
    rows = _verdict_rows(judge_analytic(spec, twin, twin.scalars(run.log), {}, [w.condition for w in worlds]))
    q = np.stack(frames)
    pw = run.model.particle_world.numpy()
    d = out_dir / LAW_ID
    d.mkdir(parents=True, exist_ok=True)
    for w, wd in enumerate(worlds):
        meta = {"engine": "alloy", "versions": versions(), "test": LAW_ID, "condition": wd.condition.id,
                "dt": spec.dt, "record_every": 1, "worlds": 1, "settings": {"solver": SOLVER}, "wall_s": wall}
        np.savez_compressed(d / f"{wd.condition.id}.npz", meta=np.array(json.dumps(meta)),
                            t=np.arange(q.shape[0]) * spec.dt, particle_q=q[:, None, pw == w].astype(np.float32))
    return {"law": LAW_ID, "status": "pass" if all(r["verdict"] == "pass" for r in rows) else "fail",
            "metrics": rows, "wall_s": wall}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("pack", "score", "alloy"))
    p.add_argument("path", type=Path, nargs="?", default=OUT)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.cmd == "pack":
        pack(a.path)
    elif a.cmd == "alloy":
        res = alloy_run(a.path, a.device)
        (a.path / f"{LAW_ID}.alloy_direct.json").write_text(json.dumps(res, indent=1, default=str))
    else:
        res = score(a.path, a.device)
        (a.path / f"{LAW_ID}.scored.json").write_text(json.dumps(res, indent=1, default=str))
        print(json.dumps({k: res[k] for k in ("law", "status") if k in res}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
