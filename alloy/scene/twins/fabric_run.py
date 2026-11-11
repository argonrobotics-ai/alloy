# SPDX-License-Identifier: Apache-2.0
"""Run the measured-fabric drape rigs (``fabric_rig``): build once, drape with the gravity ramp, score every rig.

Used by the D03M bench runner (``bench/scenarios/fabric_drape_run.py``, its CLI) and the R6 law's production-route
twin (``alloy.scene.twins.drape_route``). ``RunOpts.iterations`` is a convergence probe: it merges the iteration
count into the VBD entry's kwargs on top of the cloth_accuracy setting (``alloy.integration.throughput
.with_entry_kwargs``, as XB4's drape harness does). Gravity eases in by writing the model's gravity each control
period (smoothstep over ``fabric_rig.G_RAMP_S``).
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np

from alloy.scene.twins import fabric_rig as FR
from alloy.scene.twins import fabric_score as FS


@dataclasses.dataclass(frozen=True)
class RunOpts:
    out: Path                       # cards are written under out/cards
    worlds: int = 8
    device: str = "cuda:0"
    accuracy: str = "super"
    iterations: int = 0             # VBD iterations above the setting's (0: as set)
    edge: float = 0.008             # m: mesh cell
    seconds: float = 4.5
    seed: int = 0
    mesh_kind: str = "r6"           # recorded in the rows; the mesh itself comes from the caller or fabric_rig
    clamp: bool = True              # False: resting on the support (a diagnostic, fabric_rig.add_rig)


def build(scene, device: str, iterations: int = 0):
    """alloy.build, with ``iterations`` merged into the VBD entry's kwargs when non-zero."""
    import alloy

    if not iterations:
        return alloy.build(scene, device=device, seed=0)
    import alloy.api._plugins as B          # alloy.api._build imports build_engine from here at call time
    from alloy.integration.throughput import with_entry_kwargs

    inner = B.build_engine

    def with_iterations(built, *a, **k):
        return inner(with_entry_kwargs(built, {"iterations": int(iterations)}, "vbd"), *a, **k)

    B.build_engine = with_iterations
    try:
        return alloy.build(scene, device=device, seed=0)
    finally:
        B.build_engine = inner


def set_gravity(model, t: float, g: float = 9.81) -> None:
    s = min(max(t / FR.G_RAMP_S, 0.0), 1.0)
    vec = np.array([0.0, 0.0, -g * s * s * (3.0 - 2.0 * s)], np.float32)
    model.gravity.assign(np.tile(vec, (model.gravity.shape[0], 1)))


def solver_info(sim) -> dict:
    from alloy.integration.stepper import entry_solvers
    from alloy.solvers.vbd.base import SolverVBDAlloy

    eng = sim.engine
    vbd = [s for s in entry_solvers(getattr(eng, "stepper", eng)).values() if isinstance(s, SolverVBDAlloy)]
    if not vbd:
        return {"vbd_iterations": None}
    cfg = getattr(getattr(vbd[0], "_accel", None), "cfg", None)
    return {"vbd_iterations": int(vbd[0].iterations), "initial_guess": getattr(cfg, "initial_guess", None),
            "chebyshev_rho": getattr(cfg, "chebyshev_rho", None)}


def run_frames(sim, names: list[str], seconds: float) -> dict:
    """Step with the gravity ramp; points every FRAME_S over the last SCORE_S. Wall time excludes reads and frame 0."""
    model = sim.engine.model
    every, n = max(1, int(round(FR.FRAME_S / sim.dt))), int(round(seconds / sim.dt))
    set_gravity(model, 0.0)
    sim.step(1)
    sim.state(names)
    t0, read_s, ts, qs = time.perf_counter(), 0.0, [], {k: [] for k in names}
    for k in range(1, n):
        set_gravity(model, k * sim.dt)
        sim.step(1)
        t = (k + 1) * sim.dt
        if (k + 1) % every == 0 and t >= seconds - FR.SCORE_S - 1e-6:
            r0 = time.perf_counter()
            st = sim.state(names)
            for name in names:
                qs[name].append(np.asarray(st[f"{name}.points"], np.float64))
            ts.append(t)
            read_s += time.perf_counter() - r0
    sim.state(names)
    return {"t": np.array(ts), "q": {k: np.stack(v) for k, v in qs.items()},
            "wall_s": time.perf_counter() - t0 - read_s, "sim_s": (n - 1) * sim.dt}


def score_rig(rig: FR.Rig, rec: dict, mesh, radius: float, r_sup: float) -> dict:
    q = rec["q"][rig.name].copy()
    q[..., 0] -= rig.xy[0]
    q[..., 1] -= rig.xy[1]
    row = FS.Scorer(mesh[0], mesh[1], radius, r_sup).frames(rec["t"], q)
    dc_m = rig.fab.get("dc_measured")
    return {"fabric": rig.fab["id"], "arm": rig.arm, "bend": rig.bend, "card": rig.name, **row, "dc_measured": dc_m,
            "dc_minus_measured": None if dc_m is None else row["dc_mean"] - float(dc_m)}


def run_rigs(pack: dict, rigs: list[FR.Rig], opts: RunOpts, mesh=None) -> tuple[list[dict], dict]:
    """One build of ``rigs`` (``fabric_rig.rigs_for``) on ``mesh`` (default: the R6 mesh at ``opts.edge``): one row
    per rig, and the per-world DC / node arrays per rig name (for a law's judge)."""
    root = FR.write_cards(opts.out, rigs, pack["id"])
    radius, r_sup = 0.5 * pack["rig"]["specimen_diameter_m"], 0.5 * pack["rig"]["support_diameter_m"]
    mesh = mesh if mesh is not None else FR.specimen_mesh(radius, opts.edge, opts.seed)
    shortest = FR.check_mesh(mesh, rigs)
    scene = FR.make_scene(rigs, opts.worlds, opts.accuracy, root, mesh, r_sup, opts.seed, opts.clamp)
    sim = build(scene, opts.device, opts.iterations)
    routes = {k: r.get("route") for k, r in sim.routes().items()}
    info = solver_info(sim)
    rec = run_frames(sim, [r.name for r in rigs], opts.seconds)
    check = sim.check()
    sim.close()
    common = {"pack": pack["id"], "accuracy": opts.accuracy, "iterations_flag": opts.iterations, **info,
              "edge_m": opts.edge, "mesh": opts.mesh_kind, "mesh_min_edge_m": shortest, "mesh_nodes": len(mesh[0]),
              "worlds": opts.worlds, "clamp": opts.clamp, "seed": opts.seed, "device": opts.device,
              "seconds": opts.seconds,
              "rigs_in_build": len(rigs), "check_ok": bool(check["ok"]),
              "problems": check.get("problems", [])[:5],
              "build_wall_s": rec["wall_s"], "build_sim_s_per_s": opts.worlds * rec["sim_s"] / rec["wall_s"]}
    rows = [{**score_rig(r, rec, mesh, radius, r_sup), "route": routes.get(r.name), **common} for r in rigs]
    per_world = {row["card"]: {"dc": np.asarray(row["dc_world_mean"], np.float64),
                               "nodes": np.asarray(row["nodes_world"], np.float64)} for row in rows}
    return rows, per_world


__all__ = ["RunOpts", "build", "run_frames", "run_rigs", "score_rig", "set_gravity", "solver_info"]
