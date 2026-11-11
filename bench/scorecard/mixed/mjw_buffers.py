# SPDX-License-Identifier: Apache-2.0
"""Measure the constraint and contact buffers Newton's SolverMuJoCo needs on each hold task.

    PYTHONPATH=. python bench/scorecard/mixed/mjw_buffers.py --tasks a,b,c --worlds 1024 --out buffers.json

Newton sizes MuJoCo Warp's ``njmax`` / ``nconmax`` from the start state when none is given; a robot that makes more
contacts later overflows them (MuJoCo Warp then drops constraints and prints a warning every step, which also
slows the step). For each task this builds with generous buffers, runs the 4 s hold loop, and records the default
sizes, the peak constraint rows in any world and the peak contacts per world on average, and the sizes to use
(1.5 x the peak, never under the default).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mixed  # noqa: E402

GENEROUS = {"njmax": 4096, "nconmax": 1024}
MARGIN = 1.5


def probe(mods: tuple[Any, ...], name: str, a: argparse.Namespace) -> dict[str, Any]:
    import importlib  # noqa: PLC0415

    newton = importlib.import_module(a.newton_module)
    import warp as wp  # noqa: PLC0415

    S, T, _, NH = mods
    path, _, _ = T.model(T.TASKS[name], Path(a.out).parent / "models")
    _, model, s0, s1, ctrl, solver = NH.build(path, a.worlds, a.device, a.newton_module)
    default = {"njmax": int(solver.mjw_data.njmax), "nconmax": int(solver.mjw_data.naconmax) // a.worlds}
    del solver
    solver = newton.solvers.SolverMuJoCo(model, **GENEROUS)
    bank, dt, peak_efc, peak_con = NH.ctrl_bank(path, ctrl, a.worlds, 0), S.DT / a.substeps, 0, 0
    for k in range(a.steps):
        wp.copy(ctrl.mujoco.ctrl, bank[k % mixed.NOISE_BANK])
        for _ in range(a.substeps):
            s0.clear_forces()
            solver.step(s0, s1, ctrl, None, dt)
            s0, s1 = s1, s0
        if k % a.every == 0 or k == a.steps - 1:
            peak_efc = max(peak_efc, int(solver.mjw_data.nefc.numpy().max()))
            peak_con = max(peak_con, int(solver.mjw_data.nacon.numpy().max()))
    per_world = peak_con / a.worlds
    return {"default": default, "peak_nefc_any_world": peak_efc, "peak_contacts_per_world_mean": per_world,
            "generous": GENEROUS, "overflows_default": peak_efc > default["njmax"] or per_world > default["nconmax"],
            "use": {"njmax": max(default["njmax"], math.ceil(MARGIN * peak_efc)),
                    "nconmax": max(default["nconmax"], math.ceil(MARGIN * per_world))}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--worlds", type=int, default=1024)
    ap.add_argument("--substeps", type=int, default=2)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--newton-module", default="alloy._vendor.newton")
    ap.add_argument("--pxs", default=str(mixed.DEFAULT_PXS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mods = mixed.load_pxs(Path(a.pxs))
    out = {"newton_module": a.newton_module, "worlds": a.worlds, "substeps": a.substeps, "steps": a.steps, "margin": MARGIN, "tasks": {}}
    for name in a.tasks.split(","):
        out["tasks"][name] = probe(mods, name, a)
        print("BUF", name, json.dumps(out["tasks"][name]), flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
