# SPDX-License-Identifier: Apache-2.0
"""task_forklift_pick: drive to a loaded pallet, fork it, lift it, carry it back and set it down.

Gym-Anything's forklift tasks (drive-to-pallet, carry-to-floor-bay, put-away; XB4-7 REPORT §2) need tyres, a
drivetrain and a hydraulic mast. The scene is ``forklift_rig``: a 2.9 t counterbalance truck on the ``vehicle``
plug-in's wheels, a mast on a ``fluid_network`` cylinder, and a 725 kg pallet. A scripted operator (host side,
between control periods) runs four phases:
1. approach: drive forward until the carriage meets the pallet's near edge (speed profile up to 1 m/s, decelerating
   at 0.5 m/s², throttle feedback);
2. lift: run the pump until the forks are ``lift`` up;
3. carry: reverse ``carry`` metres;
4. set down: open the lowering valve until the mast is back down.

Metrics (world 0): the stop error at the pallet, the lift height, how far the pallet came along, its slip on the
forks, the set-down height, the peak lift pressure and the worst drive-wheel slip.

Run: ``python -m alloy.tasks.forklift_pick [--worlds W] [--device cuda:0]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.tasks.forklift_rig import FORKS, PALLET, TRUCK, Placement, build, lift_network, vehicle

log = logging.getLogger("alloy.tasks.forklift")

CONTROL_DT = 0.02
V_MAX, DECEL, GAIN = 1.0, 0.5, 1.5           # m/s, m/s², throttle per m/s of speed error
SOLVER = {"nconmax": 256, "njmax": 1024}     # the pallet's boxes on the ground and the forks: room for every contact


@dataclass(frozen=True)
class Pick:
    place: Placement = Placement()
    lift: float = 0.25                       # m of mast travel
    carry: float = 2.0                       # m reversed with the pallet
    timeout: float = 30.0                    # s


@dataclass
class PickResult:
    """World-0 metrics (lengths in mm unless named) and the cost per exchange step."""

    seconds: float
    phases_done: int                         # of 4
    stop_error_mm: float                     # chassis x at the lift start minus the fork-in position
    lift_mm: float                           # pallet rise
    carried_m: float                         # pallet travel while carried
    slip_on_forks_mm: float                  # pallet motion relative to the carriage while carried
    set_down_mm: float                       # pallet height at the end minus at the start
    peak_bar: float                          # lift cylinder pressure, gauge
    max_drive_slip: float                    # worst |κ| of the drive wheels
    min_rear_load_n: float                   # smallest rear (steer) wheel load: the counterweight keeps it > 0
    worlds: int
    ms_per_step: float
    ms_per_step_without: float


class Operator:
    """Host-side phase machine; one ``act`` per control period sets the vehicle's and the lift's inputs."""

    def __init__(self, sim: Any, task: Pick, car: Any, lift: Any) -> None:
        self.sim, self.task, self.car, self.lift = sim, task, car, lift
        m = sim.model
        self.chassis = int(car.chassis.numpy()[0])
        self.pallet = int(np.flatnonzero([str(x).endswith(PALLET) for x in m.body_label])[0])
        self.forks = int(np.flatnonzero([str(x).endswith(FORKS) for x in m.body_label])[0])
        joint = int(np.flatnonzero([str(x).endswith("mast_lift") for x in m.joint_label])[0])
        self.q_index = int(m.joint_q_start.numpy()[joint])
        self.phase, self.settled, self.marks = 0, 0, {}

    def _drive(self, target: float, x: float, v: float) -> bool:
        error = target - x
        v_ref = float(np.sign(error) * min(V_MAX, np.sqrt(2.0 * DECEL * abs(error))))
        arrived = abs(error) < 0.02 and abs(v) < 0.03
        self.car.set_input("throttle", 0.0 if arrived else float(np.clip(GAIN * (v_ref - v), -1.0, 1.0)))
        self.car.set_input("brake", 1.0 if arrived else 0.0)
        self.settled = self.settled + 1 if arrived else 0
        return self.settled >= 5

    def act(self) -> bool:
        """Set this period's inputs; True when every phase is done."""
        q = self.sim.state.body_q.numpy()
        x, v = float(q[self.chassis, 0]), float(self.sim.state.body_qd.numpy()[self.chassis, 0])
        mast = float(self.sim.state.joint_q.numpy()[self.q_index])
        p = self.task.place
        if self.phase == 0 and self._drive(p.insert_x, x, v):
            self._mark("inserted", q, x)
        elif self.phase == 1:
            up = mast >= self.task.lift
            self.lift.set_opening("pump", 0.0 if up else 1.0)
            if up:
                self._mark("lifted", q, x)
        elif self.phase == 2 and self._drive(self.marks["lifted"]["x"] - self.task.carry, x, v):
            self._mark("carried", q, x)
        elif self.phase == 3:
            down = mast <= 0.005
            self.lift.set_opening("lower", 0.0 if down else 1.0)
            if down:
                self._mark("set_down", q, x)
        return self.phase >= 4

    def _mark(self, name: str, q: np.ndarray, x: float) -> None:
        self.marks[name] = {"x": x, "pallet": q[self.pallet, :3].astype(np.float64).copy(),
                            "forks": q[self.forks, :3].astype(np.float64).copy(), "t": float(self.sim.clock.t)}
        self.phase += 1
        self.settled = 0
        log.info("forklift: %s at t=%.2f s", name, self.marks[name]["t"])


def run(task: Pick | None = None, *, worlds: int = 1, device: str = "cpu", timing: bool = True) -> PickResult:
    task = task or Pick()
    car, lift = vehicle(), lift_network()
    scene = TestScene(build=build(task.place), worlds=worlds, control_dt=CONTROL_DT, dt=1.25e-3, solver_kwargs=SOLVER)
    sim = build_simulation(scene, [car, lift], device)
    op = Operator(sim, task, car, lift)
    start = sim.state.body_q.numpy()[op.pallet, :3].astype(np.float64)
    peak, slip, rear = 0.0, 0.0, float("inf")
    for _ in range(int(task.timeout / CONTROL_DT)):
        if op.act():
            break
        sim.step_period()
        r_lift, r_car = lift.read(), car.read()
        peak = max(peak, float(r_lift["pressure"][0].max()) - 1.0e5)
        slip = max(slip, float(np.abs(r_car["slip"][0, :2, 0]).max()))
        rear = min(rear, float(r_car["force"][0, 2:, 2].min()))
    marks = op.marks
    end = sim.state.body_q.numpy()[op.pallet, :3].astype(np.float64)
    nan = float("nan")
    got = lambda a, b, k: float(marks[a]["pallet"][k] - marks[b]["pallet"][k]) if a in marks and b in marks else nan  # noqa: E731
    slid = nan
    if "carried" in marks:
        rel = lambda m: m["pallet"][0] - m["forks"][0]  # noqa: E731
        slid = 1e3 * float(rel(marks["carried"]) - rel(marks["lifted"]))
    cost = (_cost(task, scene, device, True), _cost(task, scene, device, False)) if timing else (nan, nan)
    return PickResult(seconds=float(sim.clock.t), phases_done=len(marks),
                      stop_error_mm=1e3 * (marks["inserted"]["x"] - task.place.insert_x) if "inserted" in marks else nan,
                      lift_mm=1e3 * float(marks["lifted"]["pallet"][2] - start[2]) if "lifted" in marks else nan,
                      carried_m=-got("carried", "lifted", 0), slip_on_forks_mm=slid,
                      set_down_mm=1e3 * float(end[2] - start[2]), peak_bar=peak / 1e5, max_drive_slip=slip,
                      min_rear_load_n=rear,
                      worlds=worlds, ms_per_step=cost[0], ms_per_step_without=cost[1])


def _cost(task: Pick, scene: TestScene, device: str, with_vehicle: bool, periods: int = 25) -> float:
    """ms per exchange step while driving at 0.5 throttle, with or without the vehicle plug-in (best of two, warm)."""
    car, lift = vehicle(), lift_network()
    sim = build_simulation(scene, [car, lift] if with_vehicle else [lift], device)
    if with_vehicle:
        car.set_input("throttle", 0.5)
    sim.step_period()
    best = float("inf")
    for _ in range(2):
        wp.synchronize_device(device)
        steps, t0 = sim.steps, time.perf_counter()
        for _ in range(periods):
            sim.step_period()
        wp.synchronize_device(device)
        best = min(best, 1e3 * (time.perf_counter() - t0) / max(sim.steps - steps, 1))
    return best


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    print(json.dumps(asdict(run(worlds=args.worlds, device=args.device)), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
