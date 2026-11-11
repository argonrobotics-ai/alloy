# SPDX-License-Identifier: Apache-2.0
"""Alloy beside, laws: run Alloy's rigid laws as its runner does and write the P3-2 per-condition npz (Alloy venv).

    PYTHONPATH=. python bench/xbench/ours/alloy_laws.py OUT_DIR [--laws ...] [--device cuda:0] [--d1-route on|off]

Each law runs once through ``alloy.bench.runner.step_twin`` on the default route (SolverMuJoCo with the twin's own
settings, every condition a world of one model, as ``python -m alloy.bench run`` does). The twin's ``record`` is
wrapped to copy ``body_q`` / ``body_qd`` after every step, and the frames are split per world into
``OUT_DIR/<law>/<condition>.npz``, the same files an engine writes. ``OUT_DIR/alloy_direct.json`` keeps the verdicts
of the direct run, so ``score_laws.py OUT_DIR`` can be checked against them (same frames -> same numbers).
D1.rerun writes two fresh runs of the A3.contact_energy conditions, on WP5's ordered route by default (the
determinism Alloy offers, as SB-11 and the SB-7 targets run it; Alloy's D1 spec asks 0 ulp on CPU only, and the default
route races on GPU); ``--d1-route off`` writes the default-route pair for the beside row. The npz meta ``mjw_route``
records the route in force: this law scene (RK4, dense Jacobian) is outside the route's proven coverage, so a GPU run
reads ``refused: <reason>`` and is not bitwise; ``--device cpu`` is the D1 verdict row (Alloy's spec).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2]), str(HERE)]

from alloy_t12 import versions  # noqa: E402
from law_pack import RIGID, RODS, law_twin  # noqa: E402
from score_laws import _verdict_rows, _world_bodies  # noqa: E402

log = logging.getLogger("xbench.ours")


def _route_on(twin, law_id: str) -> list[str]:
    """WP5's ordered MuJoCo Warp route on the twin's solver (what ``alloy_repeat.py`` and the SB-7 targets run).
    Returns the list the built solvers' refusal reasons land in (empty = every solver is on the route)."""
    from alloy.precision.mjw_route import enable_route

    inner = twin.make_solver
    refused: list[str] = []

    def make_solver(model, solver):
        built = inner(model, solver)
        why = enable_route(built)
        if why is not None:
            refused.append(why)
            log.warning("%s: ordered route off (%s)", law_id, why)
        return built

    twin.make_solver = make_solver
    return refused


def _route_state(det: bool, refused: list[str]) -> str:
    """What the D1 meta records: the route actually in force, not the one asked for."""
    if not det:
        return "off"
    return "on" if not refused else f"refused: {refused[0]}"


def _recording_run(law_id: str, device: str, det: bool = False):
    """(spec, twin, worlds, run, q [T, N, 7], qd [T, N, 6], route) of one direct runner pass with frame capture;
    ``det`` asks for the ordered route, ``route`` says whether it was in force."""
    from alloy.bench.runner import RunContext, plan_worlds, step_twin

    spec, twin = law_twin(law_id)
    refused = _route_on(twin, law_id) if det else []
    ctx = RunContext(spec, device, 0, "mujoco", False)
    worlds = plan_worlds(spec, twin.conditions(False), ctx, getattr(twin, "expand", None))
    qs, qds = [], []
    inner = twin.record

    def record(step, state, contacts, obs, tlog):
        qs.append(state.body_q.numpy().copy())
        qds.append(state.body_qd.numpy().copy())
        inner(step, state, contacts, obs, tlog)

    twin.record = record
    run = step_twin(spec, twin, worlds, ctx)
    return spec, twin, worlds, run, np.stack(qs), np.stack(qds), _route_state(det, refused)


def _write_worlds(out: Path, worlds, model, q: np.ndarray, qd: np.ndarray, meta: dict, suffix: str = "") -> None:
    """One npz per condition, W axis = that condition's worlds in plan order."""
    by_world = _world_bodies(model)
    groups: dict[str, list[int]] = {}
    for g, w in enumerate(worlds):
        groups.setdefault(w.condition.id, []).append(g)
    for cid, gs in groups.items():
        bodies = by_world[gs[0]]
        idx = np.array([[b for b, _ in by_world[g]] for g in gs])          # [W, B] Newton body ids
        m = dict(meta, condition=cid, worlds=len(gs))
        np.savez_compressed(out / f"{cid}{suffix}.npz", meta=np.array(json.dumps(m)),
                            body_names=np.array([n for _, n in bodies]), t=np.arange(q.shape[0]) * meta["dt"],
                            body_q=q[:, idx].astype(np.float32), body_qd=qd[:, idx].astype(np.float32))


def _meta(law_id: str, spec, twin, wall: float) -> dict:
    return {"engine": "alloy", "engine_version": versions()["alloy"], "versions": versions(), "test": law_id,
            "dt": spec.dt, "substeps": spec.substeps, "record_every": 1, "row": "recommended", "wall_s": wall,
            "settings": twin.settings_record("mujoco") if hasattr(twin, "settings_record") else {},
            "notes": "Alloy default route via alloy.bench.runner.step_twin, one model with every condition"}


def run_law(law_id: str, out_dir: Path, device: str) -> dict:
    from alloy.bench.runner import judge_analytic

    t0 = time.perf_counter()
    spec, twin, worlds, run, q, qd, _ = _recording_run(law_id, device)
    wall = time.perf_counter() - t0
    values = twin.scalars(run.log)
    refs = twin.references(run.log) if hasattr(twin, "references") else {}
    rows = _verdict_rows(judge_analytic(spec, twin, values, refs, [w.condition for w in worlds]))
    d = out_dir / law_id
    d.mkdir(parents=True, exist_ok=True)
    _write_worlds(d, worlds, run.model, q, qd, _meta(law_id, spec, twin, wall))
    return {"law": law_id, "status": "pass" if all(r["verdict"] == "pass" for r in rows) else "fail",
            "metrics": rows, "wall_s": wall}


def run_d1(out_dir: Path, device: str, det: bool = True) -> None:
    """D1: two fresh builds of the A3.contact_energy conditions in one process, on the ordered route (``det``, the
    determinism Alloy offers; its D1 spec asks 0 ulp on CPU only) or the default route. meta ``mjw_route`` is the
    route in force: the law scene's RK4 + dense Jacobian is outside the route's proven coverage, so on GPU it reads
    ``refused: ...`` and the runs race."""
    d = out_dir / "D1.rerun"
    d.mkdir(parents=True, exist_ok=True)
    for i in (0, 1):
        spec, twin, worlds, run, q, qd, route = _recording_run("A3.contact_energy", device, det)
        meta = dict(_meta("D1.rerun", spec, twin, 0.0), mjw_route=route, device=device)
        _write_worlds(d, worlds, run.model, q, qd, meta, suffix=f"_run{i}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    p.add_argument("--laws", default=",".join((*RIGID, *RODS, "D1.rerun")))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--d1-route", choices=("on", "off"), default="on", help="D1 on WP5's ordered route (default)")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    direct = []
    for law_id in a.laws.split(","):
        if law_id == "D1.rerun":
            run_d1(a.out_dir, a.device, a.d1_route == "on")
            continue
        direct.append(run_law(law_id, a.out_dir, a.device))
        log.info("%s: %s (%.1f s)", law_id, direct[-1]["status"], direct[-1]["wall_s"])
    (a.out_dir / "alloy_direct.json").write_text(json.dumps(direct, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
