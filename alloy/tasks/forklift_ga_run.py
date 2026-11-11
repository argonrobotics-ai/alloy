# SPDX-License-Identifier: Apache-2.0
"""Run one of Gym-Anything's forklift tasks on Alloy's physics with the scripted operator, and score it with GA's
ported criteria (SCR-WP8-12 F5).

A ``Task`` is a scene (built in its rig frame, mapped to GA's facility frame by a ``Frame``), the operator's ``Plan``,
the loads and trailer the criteria watch, and the criteria function. ``run`` steps the simulation one control period
at a time: the operator reads an ``Observation``, its ``Controls`` go to the truck, and a ``Recorder`` samples the
episode GA's evaluator reads (``forklift_ga_types.Episode``). After the operator's last phase the world runs on for
``settle`` seconds (GA judges the final state, some criteria over its last 2 s), never past GA's time limit (where GA's
episode ends: a red-restraint run waits there). The verdict is GA's rule over the criteria the port judges
(``Verdict.summary()`` lists the rest): none failed, no fault.

Out of scope, as the SCR-WP8-12 ruling states: the LPG engine and torque converter (an electric traction motor), the
solid tyres' K·d^1.5 (a linear radial spring), the wheels' width and curbs (point wheels), and the humanoid operator
(a scripted one; GA's operator-seat and head-clearance criteria are not judged).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from alloy._vendor import newton
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.vehicle import laws_grip as lg
from alloy.tasks import forklift_truck as ft
from alloy.tasks.forklift_control import REST_SPEED, Controls, TruckSpec
from alloy.tasks.forklift_ga_contacts import ContactWatch
from alloy.tasks.forklift_ga_truck import Extras, Frame, TruckIO
from alloy.tasks.forklift_ga_types import BOX_SIGNS, Episode, LoadFloorTrack, Verdict
from alloy.tasks.forklift_operator import Operator
from alloy.tasks.forklift_plans import Plan
from alloy.tasks.forklift_site import SITE, ForkliftSite

log = logging.getLogger("alloy.tasks.forklift_ga")

CAPACITY = {"nconmax": 1024, "njmax": 6144}


@dataclass(frozen=True)
class Box:
    """A watched load: its body label, the box's half extents and its centre in the body frame, total mass."""

    body: str
    half: tuple[float, float, float]
    centre: tuple[float, float, float]
    mass: float


@dataclass(frozen=True)
class Task:
    """One GA task as Alloy runs it."""

    name: str                                                      # GA's task name, e.g. "start-and-park"
    build: Callable[[newton.ModelBuilder], None]
    frame: Frame
    plan: Plan
    score: Callable[[Episode, ForkliftSite], Verdict]
    extras: Extras = Extras()
    surfaces: tuple[str, ...] = ()                                 # drivable box labels (the vehicle's)
    resting: tuple[tuple[str, str], ...] = ()
    ground_plane: bool = True                                      # a ground plane at rig z = 0 (else boxes only)
    companions: Callable[[], Sequence[Any]] | None = None          # e.g. the trailer's support vehicles
    loads: dict[str, Box] = field(default_factory=dict)            # episode name -> box
    trailer: str | None = None                                     # target trailer's body label (origin: rear sill
    trailer_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)   # centre on the floor top, at this body offset)
    watch: dict[str, str] = field(default_factory=dict)            # shape label prefix -> GA structure class
    floors: tuple[str, ...] = ("ground",)                          # structure classes a load lands on
    rename: dict[str, str] = field(default_factory=dict)           # shape label -> GA's geom label
    tamper: Callable[[Controls], Controls] | None = None           # a deliberate operator error (negative tests)
    settle: float = 2.5                                            # s run on after the last phase
    time_limit: float | None = None                                # s; default GA's for the task
    waits_at: str | None = None                                    # phase where the plan's job ends in a wait


WAIT_DWELL = 0.5             # s at rest in the plan's final wait before it counts as finished (that phase's dwell)


@dataclass
class WaitWatch:
    """When the plan of a task whose job ends in a wait has finished (``Task.waits_at``; GA's load-trailer with the
    restraint red: park short of the leveler and wait). The operator stays in that phase to GA's time limit, so
    ``Operator.done`` never comes; the plan has finished once the truck has been at rest in the phase for
    ``WAIT_DWELL`` (AB-gym-5's time_limit rule judges whether the plan finished before the limit). The finish is
    withdrawn when the truck moves again or leaves the phase (AUDIT-BENCH): only a wait that lasts counts."""

    phase: str | None
    since: float | None = None

    def update(self, phase: str, speed: float, t: float) -> float | None:
        """The time the plan finished in its wait: the start of the current stretch at rest in the phase plus
        ``WAIT_DWELL``, once that is reached. None before, and again from the period the truck moves or leaves."""
        if self.phase is None or phase != self.phase or abs(speed) >= REST_SPEED:
            self.since = None
            return None
        if self.since is None:
            self.since = t
        return self.since + WAIT_DWELL if t - self.since >= WAIT_DWELL - 1.0e-9 else None


def _body(model: Any, label: str) -> int:
    labels = [str(x).rsplit("/", 1)[-1] for x in model.body_label]
    return labels.index(label)


class Recorder:
    """Samples one world's episode, once per control period."""

    def __init__(self, io: TruckIO, task: Task, watch: ContactWatch) -> None:
        self.io, self.task, self.watch = io, task, watch
        m = io.sim.model
        self.load_body = {n: _body(m, b.body) for n, b in task.loads.items()}
        self.trailer = _body(m, task.trailer) if task.trailer else None
        self.rows: list[dict[str, Any]] = []

    def sample(self, t: float) -> None:
        io = self.io
        row = {"t": t, **io.sample()}
        q = io._q
        for name, box in self.task.loads.items():
            b = q[self.load_body[name]].astype(np.float64)
            local = np.asarray(box.centre) + BOX_SIGNS * np.asarray(box.half)
            row[f"load:{name}"] = io.frame.point(Rotation.from_quat(b[3:]).apply(local) + b[:3])
            row[f"vel:{name}"] = io._qd[self.load_body[name], :3].astype(np.float64)
        if self.trailer is not None:
            b = q[self.trailer].astype(np.float64)
            row["trailer_pos"] = io.frame.point(Rotation.from_quat(b[3:]).apply(self.task.trailer_origin) + b[:3])
            row["trailer_quat_xyzw"] = io.frame.quat(b[3:])
        row["vel:truck"] = io._qd[io.chassis, :3].astype(np.float64)
        row["contacts"] = self.watch.read()
        row["touch"] = self.watch.touch
        row["vel_body"] = {b: io._qd[b, :3].astype(np.float64) for b in self.watch.structure_bodies}
        self.rows.append(row)

    def episode(self, fault: str | None = None) -> Episode:
        rows = self.rows
        t = np.array([r["t"] for r in rows])
        stack = lambda k: np.stack([r[k] for r in rows])  # noqa: E731
        signals = {k: stack(k).astype(np.float64) for k in self.io.signals}
        loads = {n: stack(f"load:{n}") for n in self.task.loads}
        floor = {n: self._floor(n, t) for n in self.task.loads}
        trailer = (stack("trailer_pos"), stack("trailer_quat_xyzw")) if self.trailer is not None else (None, None)
        return Episode(t=t, truck_pos=stack("truck_pos"), truck_quat_xyzw=stack("truck_quat_xyzw"),
                       speed=stack("speed"), wheel_fz=stack("wheel_fz"), fork_tips=stack("fork_tips"),
                       blade_pitch=stack("blade_pitch"), loads=loads,
                       load_masses={n: b.mass for n, b in self.task.loads.items()}, floor=floor,
                       contacts=self.watch.episodes(t, rows, self.io), signals=signals, trailer_pos=trailer[0],
                       trailer_quat_xyzw=trailer[1], beam_contacts=self.watch.beams(rows[-1]), fault=fault)

    def _floor(self, name: str, t: np.ndarray) -> LoadFloorTrack:
        """The load touching a floor-class structure per sample, its plan position and its speed at that sample."""
        touching = np.array([any(s in self.task.floors for s in r["contacts"].get(f"load:{name}", {}))
                             for r in self.rows])
        corners = np.stack([r[f"load:{name}"] for r in self.rows])
        speed = np.array([float(np.linalg.norm(r[f"vel:{name}"])) for r in self.rows])
        return LoadFloorTrack(touching=touching, xy=corners.mean(axis=1)[:, :2], onset_speed=speed)


def simulate(task: Task, device: str = "cpu", spec: TruckSpec = TruckSpec()) -> tuple[Episode, Verdict]:
    """Build the task, run the scripted operator to its end (or GA's time limit) plus ``settle``, score it."""
    car = ft.vehicle(surfaces=task.surfaces, resting=task.resting, ground_plane=task.ground_plane)
    hyd = ft.hydraulics()
    scene = TestScene(build=task.build, ground=task.ground_plane, worlds=1, dt=lg.STEP, control_dt=lg.PERIOD,
                      companions=task.companions, solver_kwargs=CAPACITY)
    sim = build_simulation(scene, [car, hyd], device, worlds=1)
    io = TruckIO(sim, car, hyd, task.frame, task.extras)
    watch = ContactWatch(sim, task.watch, {name: box.body for name, box in task.loads.items()}, task.rename)
    op, rec = Operator(task.plan, spec), Recorder(io, task, watch)
    limit = task.time_limit or SITE.limits.time_limit_s[task.name]
    t, end, fault, k, done_t = 0.0, None, None, 0, None
    waiting = WaitWatch(task.waits_at)
    io.refresh()
    while t < min(limit, end if end is not None else limit) - 1.0e-9:      # GA's episode ends at its time limit
        ctrl = op.step(io.observe(t))
        io.apply(task.tamper(ctrl) if task.tamper is not None else ctrl)
        sim.step_period()
        k += 1
        t = round(k * lg.PERIOD, 9)                                  # periods counted, not summed (GA's limit is exact)
        io.refresh()
        if not np.isfinite(io._q).all():
            fault = "non-finite state"
            break
        rec.sample(t)
        if end is None and op.done:
            end, done_t = min(t + task.settle, limit), t
            log.info("%s: operator done at t=%.2f s", task.name, t)
        elif end is None and task.waits_at is not None:              # its job ends in a wait: run on to the limit
            done_t = waiting.update(op.phase, io.speed(), t)          # speed and t both after the step
    log.info("%s: last phase %s at t=%.2f s, plan finished at %s", task.name, op.phase, t, done_t)
    ep = replace(rec.episode(fault), plan_done_t=done_t, plan_known=True)
    return ep, task.score(ep, SITE)


def truck_at(x: float, y: float, yaw: float, pose: ft.Pose = ft.Pose(), load: ft.Load | None = None,
             z: float = 0.0) -> Any:
    """The truck's transform at rest on a level floor at height ``z`` with its front axle at (x, y) heading ``yaw``
    (rig frame)."""
    import warp as wp

    from alloy.tasks import forklift_mast as fm  # noqa: PLC0415

    place = wp.transform(wp.vec3(x, y, z), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw))
    return wp.transform_multiply(place, fm.rest_at(pose, load, lg.TRUCK["radial_stiffness"][0]))


def arc(x: float, y: float, heading: float, turn: float, radius: float, n: int = 6,
        direction: int = 1) -> list[tuple[float, float, float, int]]:
    """Waypoints along a circular arc from (x, y) with the truck at ``heading``, its heading changing by ``turn`` (rad,
    + = counter-clockwise) at ``radius`` (m), driven forward or in reverse (``direction`` -1: the truck moves against
    its heading). The first point is excluded. The Hermite joins between close arc points stay on the arc (a 90°
    corner given as two waypoints bends tighter than the truck can steer)."""
    back = 0.0 if direction > 0 else math.pi
    travel = heading + back
    side = 1.0 if turn > 0 else -1.0
    cx, cy = x - side * radius * math.sin(travel), y + side * radius * math.cos(travel)
    out = []
    for k in range(1, n + 1):
        h = travel + turn * k / n
        out.append((cx + side * radius * math.sin(h), cy - side * radius * math.cos(h), h - back, direction))
    return out


def yaw_of(quat_xyzw: np.ndarray) -> float:
    return float(Rotation.from_quat(quat_xyzw).as_euler("ZYX")[0])


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


__all__ = ["Box", "CAPACITY", "Recorder", "Task", "arc", "simulate", "truck_at", "wrap", "yaw_of"]
