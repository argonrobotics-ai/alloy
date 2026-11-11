# SPDX-License-Identifier: Apache-2.0
"""task_suction_pick (engine item 10): a gantry suction cup picks a box off the ground, carries it, vents and places it.

The gantry is three prismatic joints (x, y, z) on position drives; its z link carries the cup's lip pad (a rigid cup:
the z link is both the lip body and the mount of the ``suction`` plug-in). The box (80 x 80 x 40 mm, 0.5 kg) starts on
the ground under the cup. Script (joint targets relative to the start, cosine-blended; pump from t = 0):

    0.0-0.4 s  down 30.5 mm onto the box top         1.0-1.6 s  carry +0.20 m in x
    0.4-0.6 s  hold (the cup evacuates, τ ≈ 50 ms)     1.6-2.0 s  down onto the ground
    0.6-1.0 s  lift 0.10 m                            2.0 s      pump off, vent open; 2.2-2.6 s  up 50 mm

Success per world: the box was lifted >= 60 mm and ends within 10 mm of (0.20, 0) on the ground. The pad and box
contacts are critically damped (kd = 2·sqrt(ke)); with Newton's default kd the contact chattered at ~100 Hz and walked
the box 7.6 mm.

``--gripper adhesion`` runs the same script with an ``adhesion`` pad (pull 20 N, MuJoCo's rule) on the z link instead
of the cup: ctrl 1 from t = 0, 0 at 2.0 s.

    python -m alloy.tasks.suction_pick --worlds 64 --device cuda:0 --graph --out rows.jsonl
writes one row with the success count and the cost per step with and without the plug-in.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform
from alloy.plugins.library.adhesion import Adhesion
from alloy.plugins.library.suction import Suction

BOX_HALF = (0.04, 0.04, 0.02)        # m
BOX_MASS = 0.5                       # kg
LIP_R = 0.02                         # m, lip radius (a square pad of half side R carries the contacts)
LIP_DROP = 0.02                      # m, z-link frame to the lip plane
CLEAR = 0.03                         # m, lip above the box top at the start
Z0 = 2.0 * BOX_HALF[2] + CLEAR + LIP_DROP
CONTACT_KE = 1.0e5                   # lip and box contact stiffness (MuJoCo, mass-normalised)
CONTACT_KD = 2.0 * CONTACT_KE ** 0.5  # critical damping: Newton's default kd 100 (dampratio ~0.15) made the pad/box contact
                                     # chatter at ~100 Hz and walked the box 7.6 mm (WP13 11-06)
DRIVE_KE, DRIVE_KD = 5.0e3, 200.0    # N/m, N·s/m per gantry axis
PLACE = (0.20, 0.0)                  # m, where the box goes
PLACE_TOL = 0.010                    # m; reported as place_err_mm_max (1.5e-5 mm with critical contact damping)
PERIODS = 140                        # 2.8 s at 20 ms
CONTROL_DT = 0.02                    # s, the harness scene's control period
SOLVER = {"impratio": 100.0, "cone": "elliptic"}
# (t0, t1, axis, start, end): cosine blends of the joint targets
MOVES = ((0.0, 0.4, 2, 0.0, -(CLEAR + 5e-4)), (0.6, 1.0, 2, -(CLEAR + 5e-4), 0.10), (1.0, 1.6, 0, 0.0, PLACE[0]),
         (1.6, 2.0, 2, 0.10, -CLEAR + 5e-4), (2.2, 2.6, 2, -CLEAR + 5e-4, 0.05))
VENT_AT = 2.0
ADH_GAIN = 20.0                      # N, the adhesion pad's pull at ctrl 1 (4x the box weight)
GRIPPERS = ("suction", "adhesion")


def build(b: newton.ModelBuilder) -> None:
    """One world: the gantry with the lip pad, and the box on the ground."""
    start = body_xform(0.0, 0.0, Z0)
    links, joints = [], []
    for k, axis in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
        name = "gantry_" + "xyz"[k]
        mass = 0.5 if k < 2 else 1.0
        link = b.add_link(xform=start, mass=mass, inertia=((1e-3, 0, 0), (0, 1e-3, 0), (0, 0, 1e-3)), label=name)
        parent = links[-1] if links else -1
        joints.append(b.add_joint_prismatic(parent, link, parent_xform=start if parent < 0 else body_xform(0, 0, 0),
                                            axis=axis, target_pos=0.0, target_ke=DRIVE_KE, target_kd=DRIVE_KD,
                                            label=name))
        links.append(link)
    pad = newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.8, ke=CONTACT_KE, kd=CONTACT_KD)
    b.add_shape_box(links[2], xform=body_xform(0.0, 0.0, -0.5 * LIP_DROP), hx=LIP_R, hy=LIP_R, hz=0.5 * LIP_DROP,
                    cfg=pad, label="lip")
    b.add_articulation(joints, label="gantry")
    box = b.add_body(xform=body_xform(0.0, 0.0, BOX_HALF[2]), label="box")
    b.add_shape_box(box, hx=BOX_HALF[0], hy=BOX_HALF[1], hz=BOX_HALF[2], cfg=newton.ModelBuilder.ShapeConfig(
        density=float(BOX_MASS / (8.0 * np.prod(BOX_HALF))), mu=0.8, ke=CONTACT_KE, kd=CONTACT_KD))


def scene(worlds: int = 3, periods: int = PERIODS) -> TestScene:
    return TestScene(build=build, worlds=worlds, periods=periods, solver_kwargs=SOLVER, ground_mu=0.8)


def plugin(gripper: str = "suction") -> Suction | Adhesion:
    """The cup (rigid: the z link is lip body and mount) or the adhesion pad, on the gantry's z link."""
    if gripper == "adhesion":
        return Adhesion(body="gantry_z", center=(0.0, 0.0, -LIP_DROP), radius=LIP_R, gain=ADH_GAIN, targets=("box",))
    return Suction(cup="gantry_z", mount="gantry_z", lip_center=(0.0, 0.0, -LIP_DROP), radius=LIP_R,
                   targets=("box",))


def set_inputs(plug: Suction | Adhesion, t: float) -> None:
    """Grip until VENT_AT, then let go (suction: pump off and vent open; adhesion: ctrl 0)."""
    on = t < VENT_AT
    if isinstance(plug, Adhesion):
        plug.set_input("ctrl", 1.0 if on else 0.0)
        return
    plug.set_input("pump", 1.0 if on else 0.0)
    plug.set_input("vent", 0.0 if on else 1.0)


def targets(t: float) -> np.ndarray:
    """Joint targets (x, y, z) at time t [m, relative to the start]."""
    q = np.zeros(3)
    for t0, t1, axis, a, b in MOVES:                  # in time order: a later move starts where the last ended
        if t >= t1:
            q[axis] = b
        elif t > t0:
            q[axis] = a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * (t - t0) / (t1 - t0)))
    return q


def _dofs(model: newton.Model) -> np.ndarray:
    """[W, 3] joint_q indices of the gantry joints x, y, z in every world (one coordinate each)."""
    labels = list(model.joint_label)
    q0 = model.joint_q_start.numpy()
    cols = [[q0[j] for j, lab in enumerate(labels) if lab.split("/")[-1] == f"gantry_{a}"] for a in "xyz"]
    return np.asarray(cols, np.int64).T


def run(worlds: int, device: str, with_plugin: bool = True, graph: bool = False, periods: int = PERIODS,
        gripper: str = "suction") -> dict:
    """Run the script in ``worlds`` worlds: per-world success, place error, lift; wall ms per step (after a warm-up
    period, the graph captured)."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415

    plug = plugin(gripper) if with_plugin else None
    sim = build_simulation(scene(worlds, periods), [plug] if plug else [], device, worlds, graph=graph)
    dofs = _dofs(sim.model)
    box = np.array([i for i, lab in enumerate(sim.model.body_label) if lab.split("/")[-1] == "box"])
    tgt = sim.control.joint_target_q.numpy()
    lift = np.zeros(worlds)
    t_wall, steps = 0.0, 0
    for k in range(periods):
        t = k * CONTROL_DT
        q = targets(t + CONTROL_DT)
        for a in range(3):
            tgt[dofs[:, a]] = q[a]
        sim.control.joint_target_q.assign(tgt.astype(np.float32))
        if plug is not None:
            set_inputs(plug, t)
        t0 = time.perf_counter()
        sim.step_period()
        if k >= 1:                                   # period 0 compiles / captures
            sim_sync(sim)
            t_wall += time.perf_counter() - t0
            steps += 2 ** int(sim.clock.j)               # exchanges per control period
        lift = np.maximum(lift, sim.state.body_q.numpy()[box, 2] - BOX_HALF[2])
    end = sim.state.body_q.numpy()[box]
    err = np.hypot(end[:, 0] - PLACE[0], end[:, 1] - PLACE[1])
    ok = (lift >= 0.06) & (err <= PLACE_TOL) & (np.abs(end[:, 2] - BOX_HALF[2]) <= 2e-3)
    return {"task": "task_suction_pick", "gripper": gripper, "plugin": with_plugin, "worlds": worlds, "device": device, "graph": graph,
            "success": int(ok.sum()), "place_err_mm_max": float(err.max() * 1e3), "lift_mm_min": float(lift.min() * 1e3),
            "ms_per_step": 1e3 * t_wall / max(steps, 1), "steps_timed": steps}


def sim_sync(sim: Any) -> None:
    import warp as wp  # noqa: PLC0415

    wp.synchronize_device(sim.model.device)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--gripper", choices=GRIPPERS, default="suction")
    a = ap.parse_args(argv)
    on = run(a.worlds, a.device, True, a.graph, gripper=a.gripper)
    off = run(a.worlds, a.device, False, a.graph)
    row = {**on, "ms_per_step_without": off["ms_per_step"], "cost_ratio": on["ms_per_step"] / off["ms_per_step"]}
    print(json.dumps(row))
    if a.out is not None:
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0 if row["success"] == a.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())
