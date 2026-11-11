# SPDX-License-Identifier: Apache-2.0
"""Scripted heads for the drilling task ports (``drilling_common``).

- ``Driller``: per world, runs its ``Plan``'s operations in order. Each starts with the spindle placed over its mark
  (its tip ``GAP`` above the face along the operation's axis, already spinning) and ``FeedHead``'s ramp re-aimed
  there: hold = start - direction x rate x t and travel = rate x t + GAP + depth, so the head feeds ``depth`` past the
  face from where it stands. An operation ends after its ramp plus ``settle``, or when the bit jams on a bar it
  cannot cut (an operator stops; the drill's ``jammed`` flag latches, so only a world's first jam ends one early).
  With ``park`` the spindle then goes back to where its first operation started, stopped.
- ``Inserter``: places each pin over the hole nearest its mark in the top part (the mark itself without one), tip
  ``lift`` above the face, and pushes it with its ``PushHead``; then torques the anchors' nuts.

Placing a body sets its free joint's pose and velocity between control periods and re-runs forward kinematics.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.npgeom import quat_rotate
from alloy.tasks.drilling_common import (CONTROL_DT, GAP, MARK_RADIUS, RPM, STEP, Check, Plan, Verdict,
                                         mark_point, nearest, report)

log = logging.getLogger("alloy.tasks.drilling")


def quat_from_z(direction: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) turning a body's +z onto -``direction`` (a tool whose axis points along the feed)."""
    up = -np.asarray(direction, np.float64) / np.linalg.norm(direction)
    axis = np.cross([0.0, 0.0, 1.0], up)
    s, c = float(np.linalg.norm(axis)), float(up[2])
    if s < 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0]) if c > 0.0 else np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * math.atan2(s, c)
    return np.array([*(axis / s * math.sin(half)), math.cos(half)])


class Poser:
    """Sets bodies' free-joint poses and velocities between control periods."""

    def __init__(self, sim: Any) -> None:
        self.sim = sim
        m = sim.model
        self.child = m.joint_child.numpy()
        self.q_start, self.qd_start = m.joint_q_start.numpy(), m.joint_qd_start.numpy()
        self.q = self.qd = None

    def begin(self) -> None:
        self.q, self.qd = self.sim.state.joint_q.numpy(), self.sim.state.joint_qd.numpy()

    def place(self, body: int, pos: Any, quat: Any, omega: Any = (0.0, 0.0, 0.0)) -> None:
        j = int(np.flatnonzero(self.child == body)[0])
        q0, qd0 = int(self.q_start[j]), int(self.qd_start[j])
        self.q[q0:q0 + 7] = (*pos, *quat)
        self.qd[qd0:qd0 + 6] = (0.0, 0.0, 0.0, *omega)

    def commit(self) -> None:
        s = self.sim.state
        s.joint_q.assign(self.q)
        s.joint_qd.assign(self.qd)
        newton.eval_fk(self.sim.model, s.joint_q, s.joint_qd, s)


@dataclass
class _Inputs:
    """Host copies of the head's and the drill's per-world inputs."""

    hold: np.ndarray
    rate: np.ndarray
    spin: np.ndarray
    travel: np.ndarray
    direction: np.ndarray
    diameter: np.ndarray
    energy: np.ndarray
    blows: np.ndarray


class Driller:
    """Runs each world's drilling operations (see the module docstring)."""

    def __init__(self, sim: Any, plans: Sequence[Plan], top: float) -> None:
        self.sim, self.top = sim, top
        self.head, self.drill = sim.plugins["feed_head"], sim.plugins["drill"]
        W = sim.model.world_count
        self.plans = [plans[min(w, len(plans) - 1)] for w in range(W)]
        self.bodies = self.head.head.numpy()
        self.tip = abs(float(self.drill.config.tip[2]))
        self.poser = Poser(sim)
        self.k = np.full(W, -1)
        self.t_end = np.zeros(W)
        self.jam0 = np.zeros(W, np.int32)
        self.stops = np.ones(W, bool)                                  # the current operation stops at a jam
        self.done = np.zeros(W, bool)
        self.home = np.zeros((W, 3))
        z = np.zeros(W, np.float32)
        self.inp = _Inputs(np.zeros((W, 3), np.float32), z.copy(), z.copy(), z.copy(),
                           np.tile(np.float32([0.0, 0.0, -1.0]), (W, 1)), z + float(self.drill.config.diameter),
                           z.copy(), z.copy())

    def start(self) -> None:
        self._advance(np.arange(len(self.plans)), np.zeros(len(self.plans)))

    def update(self) -> bool:
        """After a period: start each world's next operation when its current one is over. True when all are done."""
        t = self.head.ctx.time.numpy().astype(np.float64)
        jammed = self.drill.jammed.numpy()
        due = ~self.done & ((t >= self.t_end) | ((jammed > self.jam0) & self.stops))
        if due.any():
            self._advance(np.flatnonzero(due), t)
        return bool(self.done.all())

    def _advance(self, worlds: np.ndarray, t: np.ndarray) -> None:
        self.poser.begin()
        jammed = self.drill.jammed.numpy()
        for w in worlds:
            self.k[w] += 1
            ops = self.plans[w].ops
            if self.k[w] < len(ops):
                self._begin(int(w), ops[self.k[w]], float(t[w]))
                self.jam0[w] = jammed[w]
            else:
                self._finish(int(w))
                self.done[w] = True
        self.poser.commit()
        self._push()

    def _begin(self, w: int, op: Any, t: float) -> None:
        d = op.direction
        aim = mark_point((op.mark[0] + op.offset[0], op.mark[1] + op.offset[1]), self.top)
        com = aim - d * (GAP + self.tip)
        if self.k[w] == 0:
            self.home[w] = com
        self.poser.place(int(self.bodies[w]), com, quat_from_z(d), op.spin * -d)
        i = self.inp
        i.hold[w] = com - d * op.rate * t
        i.rate[w], i.spin[w], i.travel[w], i.direction[w] = op.rate, op.spin, op.rate * t + GAP + op.depth, d
        i.diameter[w], (i.energy[w], i.blows[w]) = op.diameter, op.hammer
        self.t_end[w] = t + op.seconds
        self.stops[w] = op.stop_on_jam

    def _finish(self, w: int) -> None:
        if not self.plans[w].park or not self.plans[w].ops:
            return                                                    # left in the work, spinning (or never used)
        i = self.inp
        self.poser.place(int(self.bodies[w]), self.home[w], quat_from_z(i.direction[w]))
        i.hold[w], i.rate[w], i.spin[w], i.travel[w] = self.home[w], 0.0, 0.0, 0.0

    def _push(self) -> None:
        h, i = self.head, self.inp
        h.hold.assign(i.hold)
        h.set_input("feed_rate", i.rate)
        h.set_input("spin", i.spin)
        h.set_input("travel", i.travel)
        h.set_input("direction", i.direction)
        self.drill.set_diameter(i.diameter)
        self.drill.set_hammer(i.energy, i.blows)


class Inserter:
    """Pushes each world's pins into the holes at their marks, then torques the anchors' nuts."""

    def __init__(self, sim: Any, plans: Sequence[Plan], top: float) -> None:
        self.sim, self.top = sim, top
        self.drill = sim.plugins["drill"]
        W = sim.model.world_count
        self.plans = [plans[min(w, len(plans) - 1)] for w in range(W)]
        self.pins = [p.body for p in self.drill.config.pins]
        self.length = {p.body: float(np.linalg.norm(np.subtract(p.tip, p.top))) for p in self.drill.config.pins}

    @property
    def active(self) -> bool:
        return any(p.inserts for p in self.plans)

    def start(self) -> None:
        W = len(self.plans)
        poser = Poser(self.sim)
        poser.begin()
        body_q = self.sim.state.body_q.numpy()
        force = {pin: np.zeros(W, np.float32) for pin in self.pins}
        for w, plan in enumerate(self.plans):
            for ins in plan.inserts:
                body = int(self.sim.plugins[f"push_{ins.pin}"].body.numpy()[w])
                aim = self._aim(w, ins.mark, body_q)
                poser.place(body, aim + np.array([0.0, 0.0, ins.lift + self.length[ins.pin]]), (0.0, 0.0, 0.0, 1.0))
                force[ins.pin][w] = ins.force
        poser.commit()
        for pin, f in force.items():
            self.sim.plugins[f"push_{pin}"].set_input("force", f)

    def torque(self) -> None:
        if not any(ins.torque for p in self.plans for ins in p.inserts):
            return                                                    # no anchors' nuts to turn
        T = np.zeros((len(self.plans), len(self.pins)), np.float32)
        for w, plan in enumerate(self.plans):
            for ins in plan.inserts:
                T[w, self.pins.index(ins.pin)] = ins.torque
        self.drill.set_input("nut_torque", T)

    def _aim(self, w: int, mark: tuple[float, float], body_q: np.ndarray) -> np.ndarray:
        """The entry of the top part's hole nearest the mark (a hand feels for the hole), else the mark."""
        p = mark_point(mark, self.top)
        hit = nearest(self.drill, w, 0, body_q, p)
        if hit is None or hit[1] > MARK_RADIUS:
            return p
        entry = hit[0].in_world(self.drill.face(w, 0, body_q))[0]
        return np.array([entry[0], entry[1], self.top])


@dataclass
class Episode:
    """A finished run: the simulation, the bodies' start poses, the spindle's start of its first operation per world
    (its "top") and how long it ran."""

    sim: Any
    start_q: np.ndarray
    home: np.ndarray
    seconds: float

    @property
    def drill(self) -> Any:
        return self.sim.plugins["drill"]

    @property
    def body_q(self) -> np.ndarray:
        return self.sim.state.body_q.numpy()


def run_episode(sim: Any, plans: Sequence[Plan], top: float, *, push_seconds: float = 1.5,
                torque_seconds: float = 0.2, limit: float = 600.0) -> Episode:
    """Drill every world's plan, then push its pins in and torque its nuts."""
    start_q = sim.state.body_q.numpy().copy()
    driller = Driller(sim, plans, top)
    driller.start()
    periods, cap = 0, int(limit / CONTROL_DT)
    while periods < cap:
        sim.step_period()
        periods += 1
        if driller.update():
            break
    else:
        log.warning("drilling: stopped at the %.0f s limit with worlds %s unfinished", limit,
                    np.flatnonzero(~driller.done).tolist())
    inserter = Inserter(sim, plans, top)
    if inserter.active:
        inserter.start()
        for _ in range(int(round(push_seconds / CONTROL_DT))):
            sim.step_period()
        inserter.torque()
        for _ in range(int(round(torque_seconds / CONTROL_DT))):
            sim.step_period()
        periods += int(round((push_seconds + torque_seconds) / CONTROL_DT))
    return Episode(sim, start_q, driller.home.copy(), periods * CONTROL_DT)


def simulate(task: Any, worlds: int, device: str) -> Episode:
    """Build ``task``'s scene (``task.build``, ``task.drill``, ``task.head``, ``task.plans``, ``task.top``, optional
    ``task.dt``) with a ``FeedHead`` on the spindle and a ``PushHead`` per pin, and run its plans."""
    from alloy.plugins.drilling import Drill, FeedHead  # noqa: PLC0415
    from alloy.plugins.harness.scene import TestScene, build_simulation  # noqa: PLC0415
    from alloy.plugins.holes.pins import PushHead  # noqa: PLC0415

    drill = Drill(**task.drill)
    hands = [PushHead(label=f"push_{p.body}", body=p.body, force=0.0, d_push=task.d_push) for p in drill.config.pins]
    scene = TestScene(build=task.build, worlds=worlds, ground=False, gravity=(0.0, 0.0, 0.0), control_dt=CONTROL_DT,
                      dt=getattr(task, "dt", STEP))
    sim = build_simulation(scene, [FeedHead(head=drill.config.tool, **task.head), *hands, drill], device)
    return run_episode(sim, task.plans(worlds), task.top, push_seconds=task.push_seconds,
                       limit=task.limit)


def run_task(task: Any, worlds: int, device: str) -> tuple[list[Verdict], Episode]:
    """Run ``task`` on ``worlds`` worlds; each world's ``Verdict`` and the episode."""
    ep = simulate(task, worlds, device)
    return [task.score(ep, w) for w in range(worlds)], ep


def cli(task: Any, doc: str, argv: list[str] | None = None) -> int:
    """A task module's ``main``: run it and print each world's verdict as JSON."""
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=len(task.variants))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    verdicts, ep = run_task(task, args.worlds, args.device)
    print(json.dumps({"seconds": ep.seconds, "worlds": report(verdicts)}, indent=1, default=str))
    return 0


def spindle_checks(ep: Episode, w: int, top: float) -> list[Check]:
    """The source's ``spindle_stopped_quill_up``: the spindle stopped (< 1 rpm), the quill within 2 mm of its top (the
    start of the first operation, along the feed) and the bit clear of the work (its tip above the top face)."""
    drill = ep.drill
    b = int(ep.sim.plugins["feed_head"].head.numpy()[w])
    q, qd = ep.body_q[b].astype(np.float64), ep.sim.state.body_qd.numpy()[b].astype(np.float64)
    up = quat_rotate(q[3:], np.array([0.0, 0.0, 1.0]))
    rpm = abs(float(qd[3:6] @ up)) / RPM
    down = float((q[:3] - ep.home[w]) @ -up)
    tip = q[:3] + quat_rotate(q[3:], np.asarray(drill.config.tip, np.float64))
    return [Check("spindle switched off and stopped", rpm < 1.0, round(rpm, 2), "< 1 rpm", "constraint"),
            Check("quill within 2 mm of the top", down < 0.002, round(-down * 1e3, 2), "> -2 mm", "constraint"),
            Check("bit clear of the work", bool(tip[2] > top), round(float(tip[2] - top) * 1e3, 2),
                  "tip above the face", "constraint")]


def pin_layers(ep: Episode, w: int, pin: str) -> list[int]:
    """The layers whose holes the pin's shank is in (the drill's ``pin_hole``), top first."""
    drill = ep.drill
    p = [q.body for q in drill.config.pins].index(pin)
    L = len(drill.config.layers)
    member = drill.read()["pin_hole"][w]
    return [i for i in range(L) if int(member[p * L + i]) > 0]


def push_of(ep: Episode, w: int, pin: str) -> float:
    """The push the pin's hand applies in world ``w`` [N] (its per-world input)."""
    return float(ep.sim.plugins[f"push_{pin}"].force.numpy()[w])


__all__ = ["Driller", "Episode", "Inserter", "Poser", "cli", "pin_layers", "push_of", "quat_from_z",
           "run_episode", "run_task", "simulate", "spindle_checks"]
