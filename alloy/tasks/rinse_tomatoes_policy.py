# SPDX-License-Identifier: Apache-2.0
"""The scripted policy of task rinse_tomatoes: carry the colander under the faucet, aim the stream at each dusty
tomato in turn, drain the colander, and set it in the clean bowl.

A privileged hand (``rinse_hand.PoseHand``, the colander's weight carried) holds the colander level under the faucet.
Each control period, in each world, it moves the colander (at most ``V_AIM``) so that the dusty tomato nearest the
stream sits under the faucet's axis: the 16 mm stream lands on it at about 2.9 m/s (14 kPa against dust that gives
way at 4 Pa dry), and the next one follows once its load is under ``clean_to``. Tomatoes roll as the colander moves;
the aim reads where they are each period. Then a flush over the middle, the faucet off, a level drain and a tilted
one, and the carry: high over the drainboard and the table (the foot ring clears the bowl's rim), down into the bowl.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.tasks.rinse_dunk import turn_x
from alloy.tasks.rinse_hand import PoseHand, mjw_solver, overflow_watch, period_overflow
from alloy.tasks.rinse_plates_scene import slerp
from alloy.tasks.rinse_tomatoes_scene import FAUCET_AT, RIM_Z, PrepSink

WORK_DROP = 0.42                      # m, the faucet's outlet over the colander's inner floor while rinsing
V_AIM = 0.10                          # m/s, the fastest the hand moves the colander while aiming
AIM_REACH = 0.09                      # m, the farthest the colander's centre leaves the faucet's axis
TILT = math.radians(20.0)             # the tilted drain


@dataclass
class ColanderRunner:
    """Steps ``sim`` while ``hand`` holds the colander (its body 0); ``watch`` [W, n] bodies (the tomatoes, the
    colander and the bowl) the aim and the verdict read; ``overflowed`` [W] a world whose contacts overflowed."""

    sim: Any
    hand: PoseHand
    flow: Any
    control_dt: float
    watch: np.ndarray
    t: float = 0.0
    faucet_on: bool = False

    def __post_init__(self) -> None:
        q = self.sim.state.body_q.numpy()[self.hand.held.numpy()[:, 0]]
        self.pos, self.quat = q[:, :3].astype(np.float64), q[0, 3:].astype(np.float64)
        self.grip = np.zeros((len(q), 1), np.float32)
        self.solver, self.watch_overflow = mjw_solver(self.sim), overflow_watch(self.sim)
        self.overflowed = np.zeros(len(q), bool)

    def faucet(self, on: bool) -> None:
        self.flow.set_input("regulator", 1.0 if on else 0.0)
        self.faucet_on = on

    def bodies(self, ids: np.ndarray) -> np.ndarray:
        """``[W, n, 3]`` positions of the bodies ``ids`` [W, n]."""
        return self.sim.state.body_q.numpy()[ids][..., :3].astype(np.float64)

    def step(self, pos: np.ndarray, quat: np.ndarray, vel: np.ndarray) -> None:
        """One period toward ``pos`` / ``vel`` [W, 3] and ``quat`` (x, y, z, w)."""
        self.pos, self.quat = np.array(pos, np.float64), np.array(quat, np.float64)
        W = len(self.pos)
        for name, value in (("target_pos", self.pos[:, None]), ("target_quat", np.broadcast_to(self.quat, (W, 1, 4))),
                            ("target_vel", np.asarray(vel)[:, None]), ("grip", self.grip)):
            self.hand.set_input(name, np.ascontiguousarray(value, np.float32))
        self.sim.step_period()
        self.t += self.control_dt
        over = period_overflow(self.watch_overflow, self.solver)
        if over is not None:
            self.overflowed |= over

    def go(self, pos: np.ndarray, quat: np.ndarray | None, seconds: float) -> None:
        """A smooth-step move of every world's colander from where it is to ``pos`` ([3] or [W, 3])."""
        p0, q0 = self.pos.copy(), self.quat.copy()
        q1 = q0 if quat is None else np.asarray(quat, np.float64)
        d = np.broadcast_to(np.asarray(pos, np.float64), p0.shape) - p0
        n = max(int(round(seconds / self.control_dt)), 1)
        for i in range(1, n + 1):
            s = i / n
            self.step(p0 + s * s * (3.0 - 2.0 * s) * d, slerp(q0, q1, s * s * (3.0 - 2.0 * s)),
                      6.0 * s * (1.0 - s) / seconds * d)

    def hold(self, seconds: float) -> None:
        for _ in range(max(int(round(seconds / self.control_dt)), 1)):
            self.step(self.pos, self.quat, np.zeros_like(self.pos))


def to_faucet(run: ColanderRunner, st: PrepSink, pace: float = 1.0) -> None:
    """Take the colander off the drainboard, over the sink's front wall, and down under the faucet."""
    run.grip[:] = 1.0
    run.hold(0.3)
    c = st.colander
    clear = RIM_Z + c.foot + c.bottom + 0.04
    p = run.pos[0]
    run.go(np.array([p[0], p[1], clear]), None, 1.5 * pace)
    run.go(np.array([FAUCET_AT[0], FAUCET_AT[1], clear]), None, 2.5 * pace)
    run.go(np.array([FAUCET_AT[0], FAUCET_AT[1], FAUCET_AT[2] - WORK_DROP]), None, 1.5 * pace)


def aim(run: ColanderRunner, tomatoes: np.ndarray, clean_to: float, seconds: float) -> float:
    """Faucet on: per world, bring the dusty tomato nearest the stream under it until every tomato's load is under
    ``clean_to`` (kg/m²) or ``seconds`` pass; returns the seconds sprayed."""
    fa = np.asarray(FAUCET_AT[:2])
    run.faucet(True)
    t0, step_max = run.t, V_AIM * run.control_dt
    for _ in range(int(round(seconds / run.control_dt))):
        soil = run.flow.lumped.read()["lump_soil"]                       # [W, 15] kg/m²
        if not (soil > clean_to).any():
            break
        tp = run.bodies(tomatoes)[..., :2]
        target = run.pos.copy()
        for w in range(len(target)):
            dirty = np.flatnonzero(soil[w] > clean_to)
            if dirty.size:
                k = dirty[np.argmin(np.linalg.norm(tp[w, dirty] - fa, axis=1))]
                target[w, :2] = run.pos[w, :2] + fa - tp[w, k]
            else:
                target[w, :2] = fa
        off = target[:, :2] - fa
        target[:, :2] = fa + off * np.minimum(1.0, AIM_REACH / np.maximum(np.linalg.norm(off, axis=1), 1e-9))[:, None]
        d = target - run.pos
        d *= np.minimum(1.0, step_max / np.maximum(np.linalg.norm(d, axis=1), 1e-12))[:, None]
        run.step(run.pos + d, run.quat, d / run.control_dt)
    return run.t - t0


def flush_and_drain(run: ColanderRunner, flush_s: float, level_s: float, tilt_s: float, pace: float = 1.0) -> None:
    """Rinse over the middle for ``flush_s``, faucet off; drain level, then tilted ``TILT``, then level again."""
    run.go(np.array([FAUCET_AT[0], FAUCET_AT[1], run.pos[0, 2]]), None, 1.0 * pace)
    run.hold(flush_s)
    run.faucet(False)
    run.hold(level_s)
    q = run.quat.copy()
    run.go(run.pos, turn_x(q, TILT), 1.0 * pace)
    run.hold(tilt_s)
    run.go(run.pos, q, 1.0 * pace)


def to_bowl(run: ColanderRunner, st: PrepSink, pace: float = 1.0) -> None:
    """Carry the colander high over the drainboard and table, lower it onto the bowl's floor and let go."""
    c, b0 = st.colander, st.bowl_start()
    lift = c.foot + c.bottom
    high = b0[2] + st.bowl.depth + lift + 0.05
    run.go(np.array([FAUCET_AT[0], FAUCET_AT[1], high]), None, 2.0 * pace)
    run.go(np.array([b0[0], b0[1], high]), None, 4.0 * pace)
    run.go(np.array([b0[0], b0[1], b0[2] + lift + 0.002]), None, 2.5 * pace)
    run.grip[:] = 0.0
    run.hold(2.0)


__all__ = ["AIM_REACH", "ColanderRunner", "V_AIM", "WORK_DROP", "aim", "flush_and_drain", "to_bowl", "to_faucet"]
