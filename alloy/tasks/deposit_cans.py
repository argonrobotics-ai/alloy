# SPDX-License-Identifier: Apache-2.0
"""task_deposit_cans: cans dropped into a recycling bag the robot holds by its straps, then the bag carried (WP13;
the Gym-Anything deposit-cans gap, xb4 REPORT.md §2 "thin films / bags (C16 strap slip)"; docs/notes/film-grip-c16.md).

Rig: alloy.tasks.deposit_bag. The bag stands on the counter, its two straps in the gantry's pinch heads. The gantry
lowers ``SLACK`` so the bag body is loose (held taut between the counter and the jaws, the inextensible panels cannot
open around a can: measured, the cans wedged 20 cm up). The cans, released at rest above the open mouth, fall in and
settle. Then the gantry lifts the bag off the counter, carries it sideways and holds it.

Phases (s, ``PHASES``): slack and drop, settle until 0.8; lift to ``LIFT`` above the start until 1.8; carry
``CARRY_X`` until 3.3; hold until 4.3. Gantry moves are cosine blends.
Success per world (Gym-Anything's rules: targets wholly inside the bag, nothing lost to the floor):
- every can inside the bag: its centre below the mouth, above the bag's lowest point, and inside the x / y bounds of
  the bag body's vertices;
- both straps held: the grip bands moved with the jaws, slide <= ``SLIP_TOL``;
- the bag carried, not dragged: its lowest point >= ``AIRBORNE`` above the counter;
- finite state.
Before the film grip fix (SCR-WP13-GRIP-1) the straps walk out of the jaws once the bag carries the cans.

python -m alloy.tasks.deposit_cans [--worlds W] [--cans N] [--device cpu|cuda:0] [--graph] [--out rows.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from alloy.tasks.deposit_bag import DepositRig, simulation

PHASES = (0.8, 1.8, 3.3, 4.3)          # s: drop and settle end, lift end, carry end, hold end
SLACK = 0.05                           # m the gantry lowers while the cans go in (over the first 0.15 s)
LIFT = 0.10                            # m above the start at the end of the lift
CARRY_X = 0.15                         # m
SLIP_TOL = 5.0e-3                      # m: a strap band's slide in its jaws
AIRBORNE = 5.0e-3                      # m: the bag's lowest point above the counter at the end
CLOSE = 0.01                           # m: jaw servo target beyond the start (the force limit sets the squeeze)


def _blend(a: float, b: float, s: float) -> float:
    return a + (b - a) * 0.5 * (1.0 - math.cos(math.pi * min(max(s, 0.0), 1.0)))


def targets(t: float) -> tuple[float, float]:
    """Gantry (x, z) targets relative to the start at time t."""
    if t < PHASES[0]:
        z = _blend(0.0, -SLACK, t / 0.15)
    else:
        z = _blend(-SLACK, LIFT, (t - PHASES[0]) / (PHASES[1] - PHASES[0]))
    x = _blend(0.0, CARRY_X, (t - PHASES[1]) / (PHASES[2] - PHASES[1]))
    return x, z


def _columns(sim) -> dict[str, list[int]]:
    """joint_q columns per world: gantry x, gantry z and the jaw sliders."""
    labels, starts = sim.model.joint_label, sim.model.joint_q_start.numpy()
    out: dict[str, list[int]] = {"x": [], "z": [], "jaw": []}
    for j, lab in enumerate(labels):
        name = lab.split("/")[-1]
        key = {"gantry_x": "x", "gantry_z": "z"}.get(name, "jaw" if name.endswith("_slide") else None)
        if key is not None:
            out[key].append(int(starts[j]))
    return out


def _can_bodies(sim) -> np.ndarray:
    return np.asarray([i for i, lab in enumerate(sim.model.body_label) if lab.split("/")[-1].startswith("can")])


def run(worlds: int = 1, device: str = "cpu", cans: int = 3, graph: bool = False, rig: DepositRig | None = None,
        seconds: float | None = None) -> dict:
    rig = replace(rig or DepositRig(), cans=cans)
    sim, groups = simulation(rig, worlds, device, graph)
    cols = _columns(sim)
    tgt = sim.control.joint_target_q.numpy()
    tgt[cols["jaw"]] = CLOSE
    q0 = sim.state.particle_q.numpy().reshape(worlds, -1, 3).copy()      # .numpy() on CPU is a live view
    end = PHASES[3] if seconds is None else seconds
    for k in range(int(round(end / rig.control_dt))):
        x, z = targets((k + 1) * rig.control_dt)
        tgt[cols["x"]], tgt[cols["z"]] = x, z
        sim.control.joint_target_q.assign(tgt.astype(np.float32))
        sim.step_period()
    q = sim.state.particle_q.numpy().reshape(worlds, -1, 3)
    cans_xyz = sim.state.body_q.numpy()[_can_bodies(sim), :3].reshape(worlds, -1, 3)
    lift = sim.state.joint_q.numpy()[cols["z"]]
    row = _score(q, q0, cans_xyz, lift, groups)
    return {"task": "task_deposit_cans", "worlds": worlds, "device": device, "graph": graph, "cans": cans,
            "seconds": end, **row}


def _score(q: np.ndarray, q0: np.ndarray, cans: np.ndarray, lift: np.ndarray, groups: dict) -> dict:
    """q, q0 [W, P, 3] particles now / at the start (world-local ids in ``groups``), cans [W, C, 3] centres, lift [W]
    the gantry's z travel."""
    body, mouth = groups["body"], groups["mouth"]
    lo, hi = q[:, body, :].min(axis=1), q[:, body, :].max(axis=1)
    mouth_z = q[:, mouth, 2].mean(axis=1)
    inside = ((cans[..., 2] < mouth_z[:, None]) & (cans[..., 2] > lo[:, None, 2])
              & (cans[..., 0] > lo[:, None, 0]) & (cans[..., 0] < hi[:, None, 0])
              & (cans[..., 1] > lo[:, None, 1]) & (cans[..., 1] < hi[:, None, 1]))
    slide = np.max([np.abs((q[:, b, 2] - q0[:, b, 2]).mean(axis=1) - lift)
                    for b in (groups["band_l"], groups["band_r"])], axis=0)        # the worse strap
    zmin = q[:, :, 2].min(axis=1)
    finite = np.isfinite(q).all(axis=(1, 2)) & np.isfinite(cans).all(axis=(1, 2))
    ok = inside.all(axis=1) & (slide <= SLIP_TOL) & (zmin >= AIRBORNE) & finite
    return {"success": int(ok.sum()), "cans_inside": inside.sum(axis=1).tolist(),
            "strap_slide_mm_max": float(1e3 * slide.max()), "bag_zmin_mm_min": float(1e3 * zmin.min()),
            "finite": bool(finite.all())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--cans", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    row = run(a.worlds, a.device, a.cans, a.graph)
    print(json.dumps(row))
    if a.out is not None:
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0 if row["success"] == a.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())
