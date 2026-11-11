# SPDX-License-Identifier: Apache-2.0
"""Score an engine's law runs with Alloy's own law code (Alloy venv).

    PYTHONPATH=. python bench/xbench/ours/score_laws.py RUN_DIR [--laws ...] [--device cpu] [--out laws.json]

``RUN_DIR/<law>/<condition>.npz`` are the engine's trajectories (contract npz, a frame every physics step; W axis =
the condition's ``worlds`` in ``cond.json`` order) for the conditions of ``packs/laws/<law>``;
``RUN_DIR/D1.rerun/<condition>_run{0,1}.npz`` are two runs of the A3.contact_energy
conditions in one process. For each law the full multi-world model is built exactly as Alloy's runner builds it
(``plan_worlds`` + ``build_model`` + the twin's ``make_solver``, which sets the twin's indices), every frame is written
into a Newton state (world w = condition w, bodies matched by Newton label, or by order for an npz named like the
pack's ``cond.json``), the twin's own ``record`` fills its log, and
the twin's ``scalars`` + ``alloy.bench.runner.judge_analytic`` give the verdicts with the LawSpec tolerances. The
log's dt is the engine's dt, so an engine that cannot run the law's dt is judged by the same formulas at its own.
D1 is ``alloy.bench.determinism.ulp_distance`` over every frame of body_q and body_qd (0 ulp to pass).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]

from law_pack import RIGID, RODS, law_twin  # noqa: E402

log = logging.getLogger("xbench.ours")


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out


def _world_bodies(model) -> dict[int, list[tuple[int, str]]]:
    """World -> [(Newton body index, name)] (name = last label component, the MJCF name)."""
    bw = model.body_world.numpy()
    out: dict[int, list[tuple[int, str]]] = {}
    for b, w in enumerate(bw):
        out.setdefault(int(w), []).append((b, str(model.body_label[b]).rsplit("/", 1)[-1]))
    return out


def _frames(runs: dict[str, dict], worlds, model, law_id: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Per-condition runs (W axis = the condition's worlds in plan order) -> Newton-ordered [T, body_count, 7 / 6]."""
    T = {r["body_q"].shape[0] for r in runs.values()}
    if len(T) != 1:
        raise ValueError(f"conditions have different frame counts {sorted(T)}")
    q = np.zeros((T.pop(), model.body_count, 7), np.float32)
    qd = np.zeros((q.shape[0], model.body_count, 6), np.float32)
    seen: dict[str, int] = {}
    for g, bodies in sorted(_world_bodies(model).items()):
        cid = worlds[g].condition.id
        j = seen[cid] = seen.get(cid, -1) + 1
        r = runs[cid]
        if j >= r["body_q"].shape[1]:
            raise ValueError(f"{cid}: npz has {r['body_q'].shape[1]} worlds, the law plans more")
        idx = _npz_index(r, bodies, law_id, cid)
        for (b, _), i in zip(bodies, idx, strict=True):
            q[:, b], qd[:, b] = r["body_q"][:, j, i], r["body_qd"][:, j, i]
    return q, qd


def _pack_names(law_id: str, cid: str) -> list[str]:
    cond = HERE / "packs" / "laws" / law_id / cid / "cond.json"
    return [b["name"] for b in json.loads(cond.read_text())["bodies"]] if law_id and cond.exists() else []


def _npz_index(run: dict, bodies: list[tuple[int, str]], law_id: str, cid: str) -> list[int]:
    """npz B index of each Newton body of one world: by Newton label, or, for an npz named like the pack's
    ``cond.json`` (MJCF names, e.g. ``law0_3`` / ``body_0`` where Newton has ``3`` / ``body_5``), by order within
    the world (the pack's body order is Newton's, tested)."""
    names = [str(n) for n in run["body_names"]]
    pos = {n: i for i, n in enumerate(names)}
    if all(name in pos for _, name in bodies):
        return [pos[name] for _, name in bodies]
    if names == _pack_names(law_id, cid) and len(names) == len(bodies):
        return list(range(len(bodies)))
    missing = sorted(name for _, name in bodies if name not in pos)
    raise ValueError(f"{law_id}/{cid}: npz bodies {names} are neither Newton's (lacks {missing}) nor the pack's")


def _verdict_rows(verdicts) -> list[dict]:
    return [{k: (str(v) if k in ("verdict", "stat") else v) for k, v in dataclasses.asdict(mv).items()
             if k in ("name", "unit", "error", "bound", "verdict", "stat")} for mv in verdicts]


def score_law(law_id: str, run_dir: Path, device: str = "cpu") -> dict:
    """Verdicts of one law from the engine's per-condition npz (``not_run`` if any condition is missing)."""
    from alloy._vendor import newton

    from alloy.bench.runner import RunContext, _call_make_solver, build_model, judge_analytic, plan_worlds
    from alloy.bench.twin import DeviceTwinLog

    spec, twin = law_twin(law_id)
    conds = twin.conditions(False)
    paths = [run_dir / law_id / f"{c.id}.npz" for c in conds]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        return {"law": law_id, "status": "not_run", "missing": missing}
    runs = {c.id: _load(p) for c, p in zip(conds, paths, strict=True)}
    ctx = RunContext(spec, device, 0, "mujoco", False)
    worlds = plan_worlds(spec, conds, ctx, getattr(twin, "expand", None))
    model = build_model(spec, twin, worlds, ctx)
    _call_make_solver(twin, model, "mujoco")
    q, qd = _frames(runs, worlds, model, law_id)
    first = runs[conds[0].id]
    dt = float(first["meta"]["dt"]) * int(first["meta"].get("record_every", 1))
    n = q.shape[0] - 1
    rs = getattr(twin, "record_stride", 1)
    tlog = DeviceTwinLog(len(worlds), n, dt, device, stride=int(rs(n) if callable(rs) else rs))
    state = model.state()
    for k in range(q.shape[0]):
        state.body_q.assign(q[k])
        state.body_qd.assign(qd[k])
        newton.eval_ik(model, state, state.joint_q, state.joint_qd)
        twin.record(k, state, None, None, tlog)
    values = twin.scalars(tlog)
    refs = twin.references(tlog) if hasattr(twin, "references") else {}
    verdicts = judge_analytic(spec, twin, values, refs, [w.condition for w in worlds])
    rows = _verdict_rows(verdicts)
    return {"law": law_id, "status": "pass" if all(r["verdict"] == "pass" for r in rows) else "fail",
            "metrics": rows, "dt": dt, "law_dt": spec.dt, "frames": int(q.shape[0]), "finite": bool(
                np.all(np.isfinite(q)) and np.all(np.isfinite(qd))), "engine": first["meta"].get("engine"),
            "settings": first["meta"].get("settings"), "monitors": json.loads(json.dumps(
                getattr(twin, "monitors", {}), default=str))}


def score_d1(run_dir: Path) -> dict:
    """D1.rerun: max ulp between two same-process runs of every A3.contact_energy condition (0 to pass)."""
    from alloy.bench.determinism import ulp_distance

    _, twin = law_twin("A3.contact_energy")
    ulps, missing = [], []
    for c in twin.conditions(False):
        a, b = (run_dir / "D1.rerun" / f"{c.id}_run{i}.npz" for i in (0, 1))
        if not (a.exists() and b.exists()):
            missing.append(c.id)
            continue
        ra, rb = _load(a), _load(b)
        ulps += [ulp_distance(ra[k], rb[k]) for k in ("body_q", "body_qd")]
    if missing:
        return {"law": "D1.rerun", "status": "not_run", "missing": missing}
    worst = max(ulps)
    return {"law": "D1.rerun", "status": "pass" if worst == 0 else "fail",
            "metrics": [{"name": "max_ulp", "unit": "ulp", "error": float(worst), "bound": 0.0,
                         "verdict": "pass" if worst == 0 else "fail", "stat": "mean"}]}


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--laws", default=",".join((*RIGID, *RODS, "D1.rerun")))
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    out = []
    for law_id in a.laws.split(","):
        try:
            res = score_d1(a.run_dir) if law_id == "D1.rerun" else score_law(law_id, a.run_dir, a.device)
        except Exception as exc:  # noqa: BLE001 - a broken run is a row, not a crash of the whole scoring
            res = {"law": law_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        log.info("%s: %s", law_id, res["status"])
        out.append(res)
    (a.out or a.run_dir / "laws_scored.json").write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
