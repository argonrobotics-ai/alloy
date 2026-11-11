# SPDX-License-Identifier: Apache-2.0
"""task_pneumatic_grip (engine item 10): a gantry with a two-finger pneumatic gripper picks, carries and places a box.

The gantry is three prismatic joints (x, y, z) on position drives. Its z link carries two fingers on prismatic joints
(along -y and +y), each driven by a double-acting pneumatic cylinder of the ``fluid_network`` plug-in (10 mm bore, 4 mm
rod, 20 mm stroke, 6 bar supply). One 5/3 closed-centre valve feeds both: 4 ISO 6358 orifices per cylinder (supply and
exhaust of the cap and of the rod side), the exhausts throttled (meter-out flow control, the usual speed control).

    valve "open":  supply -> rod sides, cap sides -> exhaust (fingers retract)
    valve "close": supply -> cap sides, rod sides -> exhaust (fingers squeeze)

The box (50 x 40 x 60 mm, 0.15 kg) stands on the ground between the open fingers. Script (gantry targets relative to
the start, cosine-blended):

    0.0-0.4 s  down 50 mm (pads at the box's mid-height)   1.2-1.8 s  carry +0.20 m in x
    0.4 s      valve "close"; 0.4-0.8 s the fingers close   1.8-2.2 s  down onto the ground
    0.8-1.2 s  lift 0.10 m                                  2.2 s      valve "open"; 2.4-2.8 s  up 50 mm

Success per world: the box was lifted >= 60 mm, ends on the ground within 10 mm of (0.20, 0), the fingers end open,
and during the carry each cylinder's force is the static force at the supply pressure (rod side exhausted) to within
its seal friction + 2 %. That shows the chambers sit at supply and exhaust while carrying; the finger joints have no
drive, so the network's pressure holds the grip.

    python -m alloy.tasks.pneumatic_grip --worlds 64 --device cuda:0 --graph --out rows.jsonl
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

from alloy.plugins.fluid import P_ATM, Edge, Node
from alloy.plugins.fluid.actuators import Cylinder
from alloy.plugins.harness.scene import TestScene, body_xform
from alloy.plugins.library.fluid_network import FluidNetwork

BOX_HALF = (0.025, 0.02, 0.03)        # m
BOX_MASS = 0.15                       # kg
PAD_HALF = (0.012, 0.005, 0.02)       # m, finger pads
FINGER_MASS = 0.03                    # kg
GAP = 0.012                           # m, pad to box at the start (fingers open)
PAD_Y = BOX_HALF[1] + GAP + PAD_HALF[1]
DESCENT = 0.05                        # m, start height of the pads' centre above the box's centre
Z0 = BOX_HALF[2] + DESCENT            # m, z-link frame (= pads' centre) at the start
CONTACT_KE = 1.0e5                    # pads and box contact stiffness (MuJoCo, mass-normalised)
CONTACT_KD = 2.0 * math.sqrt(CONTACT_KE)   # critical: Newton gives MuJoCo dampratio kd/2·sqrt(d/ke) ~ 1 (the default
                                           # kd 100 gives 0.15: the squeezed fingers bounce on the box at ~100 Hz)
DRIVE_KE, DRIVE_KD = 5.0e3, 200.0     # N/m, N·s/m per gantry axis
LIMIT_KE = 1.0e5                      # finger end stops
LIMIT_KD = 2.0 * math.sqrt(LIMIT_KE)  # critical, as the contacts (kd 100 chatters on the open stop)
SUPPLY = 6.0e5                        # Pa abs
SUPPLY_C, EXHAUST_C = 2.0e-9, 2.0e-10  # m³/(s·Pa): valve ports (0.2 l/(s·bar)) and the meter-out throttles
SEAL_FC = 1.0                         # N, cylinder seal friction
CYLINDER = {"bore": 0.010, "rod": 0.004, "stroke": 0.020, "coulomb": SEAL_FC}
PLACE = (0.20, 0.0)                   # m
PLACE_TOL = 0.010                     # m
PERIODS = 150                         # 3.0 s at 20 ms
CONTROL_DT = 0.02                     # s, the harness scene's control period
SOLVER = {"impratio": 100.0, "cone": "elliptic"}   # no noslip: with critical contact damping the grip holds without it
LIFT = 0.10
# (t0, t1, axis, start, end): cosine blends of the gantry targets
MOVES = ((0.0, 0.4, 2, 0.0, -DESCENT), (0.8, 1.2, 2, -DESCENT, -DESCENT + LIFT), (1.2, 1.8, 0, 0.0, PLACE[0]),
         (1.8, 2.2, 2, -DESCENT + LIFT, -DESCENT), (2.4, 2.8, 2, -DESCENT, -DESCENT + 0.05))
VALVE = ((0.0, "open"), (0.4, "close"), (2.2, "open"))    # (from t, position)
CARRY = (1.3, 1.8)                    # s, the grip force is judged here
SIDES = {"left": 1.0, "right": -1.0}  # finger at +y closes along -y and the other way round
PORTS = {"open": ("supply_rod", "exhaust_cap"), "close": ("supply_cap", "exhaust_rod")}


def gantry(b: newton.ModelBuilder, start: Any, masses: tuple[float, ...] = (0.5, 0.5, 1.0)) -> tuple[int, list]:
    """Three prismatic links x, y, z (labels ``gantry_x/y/z``) on position drives, from world pose ``start``; returns
    the z link and the joints (for the articulation)."""
    links, joints = [], []
    for k, axis in enumerate(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))):
        name = "gantry_" + "xyz"[k]
        link = b.add_link(xform=start, mass=masses[k], inertia=((1e-3, 0, 0), (0, 1e-3, 0), (0, 0, 1e-3)), label=name)
        parent = links[-1] if links else -1
        joints.append(b.add_joint_prismatic(parent, link, parent_xform=start if parent < 0 else body_xform(0, 0, 0),
                                            axis=axis, target_pos=0.0, target_ke=DRIVE_KE, target_kd=DRIVE_KD,
                                            label=name))
        links.append(link)
    return links[2], joints


def build(b: newton.ModelBuilder) -> None:
    """One world: the gantry with two fingers on prismatic joints, and the box on the ground."""
    zlink, joints = gantry(b, body_xform(0.0, 0.0, Z0))
    pad = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.8, ke=CONTACT_KE, kd=CONTACT_KD)
    for side, s in SIDES.items():
        f = b.add_link(xform=body_xform(0.0, s * PAD_Y, Z0), mass=FINGER_MASS,
                       inertia=((1e-6, 0, 0), (0, 1e-6, 0), (0, 0, 1e-6)), label=f"finger_{side}")
        b.add_shape_box(f, hx=PAD_HALF[0], hy=PAD_HALF[1], hz=PAD_HALF[2], cfg=pad, label=f"pad_{side}")
        joints.append(b.add_joint_prismatic(zlink, f, parent_xform=body_xform(0.0, s * PAD_Y, 0.0), axis=(0.0, -s, 0.0),
                                            limit_lower=0.0, limit_upper=CYLINDER["stroke"], limit_ke=LIMIT_KE,
                                            limit_kd=LIMIT_KD, label=f"finger_{side}"))
    b.add_articulation(joints, label="gripper")
    box = b.add_body(xform=body_xform(0.0, 0.0, BOX_HALF[2]), label="box")
    b.add_shape_box(box, hx=BOX_HALF[0], hy=BOX_HALF[1], hz=BOX_HALF[2], cfg=newton.ModelBuilder.ShapeConfig(
        density=float(BOX_MASS / (8.0 * np.prod(BOX_HALF))), mu=0.8, ke=CONTACT_KE, kd=CONTACT_KD))


def scene(worlds: int = 3, periods: int = PERIODS) -> TestScene:
    return TestScene(build=build, worlds=worlds, periods=periods, solver_kwargs=SOLVER, ground_mu=0.8)


def cylinder(side: str) -> Cylinder:
    """The finger's cylinder: cap side extends it (closes the finger); it starts retracted (fingers open)."""
    return Cylinder(f"cyl_{side}", joint=f"finger_{side}", a=f"{side}_cap", b=f"{side}_rod", **CYLINDER)


def valve_openings(position: str) -> dict[str, float]:
    """The 5/3 valve's edge openings for ``position`` ("open", "close" or "centre": every port shut)."""
    if position not in (*PORTS, "centre"):
        raise ValueError(f"valve position must be one of {[*PORTS, 'centre']}, got {position!r}")
    on = PORTS.get(position, ())
    return {f"{side}_{port}": float(port in on) for side in SIDES for port in PORTS["open"] + PORTS["close"]}


def plugin() -> FluidNetwork:
    """Both cylinders, their chambers (open: rod sides at the supply) and the valve's 8 orifices."""
    nodes, edges = [], []
    for side in SIDES:
        nodes += [Node(f"{side}_cap", pressure=P_ATM), Node(f"{side}_rod", pressure=SUPPLY)]
        for chamber in ("cap", "rod"):
            edges += [Edge(f"{side}_supply_{chamber}", "orifice_gas", f"{side}_{chamber}", reservoir=SUPPLY,
                           sonic_conductance=SUPPLY_C),
                      Edge(f"{side}_exhaust_{chamber}", "orifice_gas", f"{side}_{chamber}", reservoir=P_ATM,
                           sonic_conductance=EXHAUST_C)]
    return FluidNetwork(nodes=tuple(nodes), edges=tuple(edges), actuators=tuple(cylinder(s) for s in SIDES),
                        openings=tuple(valve_openings("open").items()))


def grip_force() -> float:
    """The static squeeze per finger [N]: cap at the supply, rod side exhausted, no friction."""
    return cylinder("left").force(SUPPLY, P_ATM)


def valve(t: float) -> str:
    """The valve position at time t."""
    return [pos for t0, pos in VALVE if t >= t0][-1]


def targets(t: float) -> np.ndarray:
    """Gantry joint targets (x, y, z) at time t [m, relative to the start]."""
    q = np.zeros(3)
    for t0, t1, axis, a, b in MOVES:                  # in time order: a later move starts where the last ended
        if t >= t1:
            q[axis] = b
        elif t > t0:
            q[axis] = a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * (t - t0) / (t1 - t0)))
    return q


def _index(model: newton.Model) -> tuple[np.ndarray, np.ndarray]:
    """[W, 3] joint_q indices of the gantry joints x, y, z, and [W] body indices of the box."""
    labels = [lab.split("/")[-1] for lab in model.joint_label]
    q0 = model.joint_q_start.numpy()
    cols = [[q0[j] for j, lab in enumerate(labels) if lab == f"gantry_{a}"] for a in "xyz"]
    box = np.array([i for i, lab in enumerate(model.body_label) if lab.split("/")[-1] == "box"])
    return np.asarray(cols, np.int64).T, box


class _Script:
    """Writes the gantry targets and the valve position before each period."""

    def __init__(self, sim: Any, plug: FluidNetwork | None) -> None:
        self.sim, self.plug, self.position = sim, plug, "open"
        self.dofs, _ = _index(sim.model)
        self.tgt = sim.control.joint_target_q.numpy()

    def __call__(self, t: float) -> None:
        q = targets(t + CONTROL_DT)
        for a in range(3):
            self.tgt[self.dofs[:, a]] = q[a]
        self.sim.control.joint_target_q.assign(self.tgt.astype(np.float32))
        if self.plug is not None and valve(t) != self.position:
            self.position = valve(t)
            for edge, value in valve_openings(self.position).items():
                self.plug.set_opening(edge, value)


def run(worlds: int, device: str, with_plugin: bool = True, graph: bool = False, periods: int = PERIODS) -> dict:
    """Run the script in ``worlds`` worlds: per-world success, place error, lift, grip force; wall ms per step (after a
    warm-up period, the graph captured)."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415

    plug = plugin() if with_plugin else None
    sim = build_simulation(scene(worlds, periods), [plug] if plug else [], device, worlds, graph=graph)
    script, (_, box) = _Script(sim, plug), _index(sim.model)
    lift, force = np.zeros(worlds), []
    t_wall, steps = 0.0, 0
    for k in range(periods):
        t = k * CONTROL_DT
        script(t)
        t0 = time.perf_counter()
        sim.step_period()
        if k >= 1:                                       # period 0 compiles / captures
            _sync(sim)
            t_wall += time.perf_counter() - t0
            steps += 2 ** int(sim.clock.j)               # exchanges per control period
        lift = np.maximum(lift, sim.state.body_q.numpy()[box, 2] - BOX_HALF[2])
        if plug is not None and CARRY[0] <= t + CONTROL_DT <= CARRY[1]:
            force.append(plug.force.numpy()[:, :2].astype(np.float64))
    end = sim.state.body_q.numpy()[box]
    travel = plug.travel.numpy()[:, :2] if plug is not None else np.full((worlds, 2), np.nan)
    row = _score(end, lift, np.asarray(force) if force else np.full((1, worlds, 2), np.nan), travel)
    return {"task": "task_pneumatic_grip", "plugin": with_plugin, "worlds": worlds, "device": device, "graph": graph,
            **row, "ms_per_step": 1e3 * t_wall / max(steps, 1), "steps_timed": steps}


def _score(end: np.ndarray, lift: np.ndarray, force: np.ndarray, travel: np.ndarray) -> dict:
    """Per-world success from the box's end pose, its lift, the cylinder forces over the carry [periods, W, 2] and the
    end travel [W, 2]."""
    err = np.hypot(end[:, 0] - PLACE[0], end[:, 1] - PLACE[1])
    f_err = np.abs(force.mean(axis=0) - grip_force()).max(axis=1)          # [W], the worse finger
    f_tol = SEAL_FC + 0.02 * grip_force()
    ok = ((lift >= 0.06) & (err <= PLACE_TOL) & (np.abs(end[:, 2] - BOX_HALF[2]) <= 2e-3) & (f_err <= f_tol)
          & (travel.max(axis=1) <= 1e-3))
    return {"success": int(ok.sum()), "place_err_mm_max": float(err.max() * 1e3),
            "lift_mm_min": float(lift.min() * 1e3),
            "grip_force_n": grip_force(), "grip_force_err_n_max": float(f_err.max()), "grip_force_tol_n": f_tol,
            "end_travel_mm_max": float(travel.max() * 1e3)}


def _sync(sim: Any) -> None:
    import warp as wp  # noqa: PLC0415

    wp.synchronize_device(sim.model.device)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    on = run(a.worlds, a.device, True, a.graph)
    off = run(a.worlds, a.device, False, a.graph)
    row = {**on, "ms_per_step_without": off["ms_per_step"], "cost_ratio": on["ms_per_step"] / off["ms_per_step"]}
    print(json.dumps(row))
    if a.out is not None:
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0 if row["success"] == a.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())
