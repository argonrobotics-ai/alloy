# SPDX-License-Identifier: Apache-2.0
"""The scripted hand's dunk moves for the rinse tasks that carry vessels through the sink (rinse_glassware, rinse_pan):
lift, carry over a bowl, dunk mouth up until the inside floods, swirl, lift out, pour over the bowl, and finally turn
mouth down and set the vessel down. Moves are ``rinse_plates_scene.move`` segments (smooth-step, feed-forward
velocity); every control period the runner hands each world's ``PanSample`` to that vessel's tracker.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.tasks.rinse_hand import PoseHand, mjw_solver, overflow_watch, period_overflow
from alloy.tasks.rinse_tracker import PanTracker
from alloy.tasks.rinse_plates_scene import Pose, Step, move
from alloy.tasks.rinse_sense import VesselSense
from alloy.tasks.rinse_sink import BOWL, BOWLS
from alloy.tasks.rinse_wash_station import WashStation


CARRY = 0.20                          # m: a carried vessel's origin over the rims (a glass's foot then clears a 150 mm
                                      # mixing glass standing on the board by 37 mm; at 0.15 two glasses carried past
                                      # the last one clipped its rim and tipped it into the wash bowl)


def turn_x(q: np.ndarray, angle: float) -> np.ndarray:
    """``q`` (x, y, z, w) turned by ``angle`` about world x."""
    s, c = math.sin(0.5 * angle), math.cos(0.5 * angle)
    x, y, z, w = q
    return np.array([c * x + s * w, c * y - s * z, c * z + s * y, c * w - s * x])


@dataclass
class DunkRunner:
    """Steps ``sim`` while ``hand`` follows moves of one held body at a time; samples every tracked vessel each
    period. ``outlets`` returns whether any outlet runs (the faucet, a valve), per world [W] or one for all.
    ``overflowed`` [W]: a world whose contacts overflowed in any substep (a simulation fault; every substep with the
    sim's ``OverflowWatch``, else each period's last)."""

    sim: Any
    hand: PoseHand
    station: WashStation
    senses: list[VesselSense]
    trackers: list[list[PanTracker]]          # [vessel][world]
    control_dt: float
    outlets: Callable[[], bool | np.ndarray] = lambda: False
    t: float = 0.0
    peak_speed: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        q = self.sim.state.body_q.numpy()[self.hand.held.numpy()]
        self.pos, self.quat = q[..., :3].copy(), q[..., 3:].copy()
        self.vel = np.zeros_like(self.pos)
        self.grip = np.zeros(self.pos.shape[:2], np.float32)
        self.peak_speed = np.zeros(self.pos.shape[:2])          # [W, N] fastest while not held, every period
        self.solver, self.watch = mjw_solver(self.sim), overflow_watch(self.sim)
        self.overflowed = np.zeros(self.pos.shape[0], bool)

    def pose(self, i: int) -> Pose:
        return Pose(self.pos[0, i].astype(np.float64), self.quat[0, i].astype(np.float64))

    def step(self, moves: dict[int, Step]) -> None:
        self.vel[:] = 0.0
        for i, s in moves.items():
            self.pos[:, i], self.quat[:, i], self.vel[:, i] = s.pose.pos, s.pose.quat, s.vel
        for name, value in (("target_pos", self.pos), ("target_quat", self.quat), ("target_vel", self.vel),
                            ("grip", self.grip)):
            self.hand.set_input(name, value.astype(np.float32))
        self.sim.step_period()
        self.t += self.control_dt
        v = np.linalg.norm(self.sim.state.body_qd.numpy()[self.hand.held.numpy(), :3], axis=2)
        self.peak_speed = np.maximum(self.peak_speed, np.where(self.grip <= 0.0, v, 0.0))
        over = period_overflow(self.watch, self.solver)
        if over is not None:
            self.overflowed |= over
        self.sample()

    def sample(self) -> None:
        bq, net = self.sim.state.body_q.numpy(), self.senses[0].flow.supply.read()
        out = np.broadcast_to(np.asarray(self.outlets(), bool), (len(self.pos),))
        for sense, worlds in zip(self.senses, self.trackers, strict=True):
            for w, tr in enumerate(worlds):
                tr.sample(sense.sample(w, self.t, bool(out[w]), bq, net))

    def go(self, i: int, target: Pose, seconds: float) -> None:
        for s in move(self.pose(i), target, seconds, self.control_dt):
            self.step({i: s})

    def hold(self, seconds: float) -> None:
        for _ in range(max(int(round(seconds / self.control_dt)), 1)):
            self.step({})

    def bowl_top(self, bowl: str, rise: float) -> np.ndarray:
        """A point ``rise`` above the rim over ``bowl``'s centre."""
        c = self.station.bowl_centre(BOWLS.index(bowl))
        return np.array([c[0], c[1], self.station.sink.rim_z + rise])

    def dunk(self, i: int, bowl: str, depth: float, soak: float, pace: float = 1.0, pour: bool = True) -> None:
        """Carry body ``i`` mouth up over ``bowl``, lower its origin to ``depth`` under the surface (the inside floods
        over the rim), hold ``soak`` s with a slow swirl, lift it out, and (``pour``) pour it back over the bowl."""
        top = self.bowl_top(bowl, CARRY)
        up = self.pose(i).quat
        self.go(i, Pose(np.array([self.pos[0, i, 0], self.pos[0, i, 1], top[2]]), up), 1.0 * pace)
        self.go(i, Pose(top, up), 1.5 * pace)
        level = self.station.floor_z + 0.254
        self.go(i, Pose(np.array([top[0], top[1], level - depth]), up), 1.5 * pace)
        for k in range(max(int(soak / 2.0), 1)):
            a = 0.02 * (1 if k % 2 == 0 else -1)
            self.go(i, Pose(np.array([top[0] + a, top[1], level - depth]), up), min(2.0, soak))
        if not pour:
            return
        self.go(i, Pose(top, up), 1.5 * pace)
        self.go(i, Pose(top, turn_x(up, math.radians(-115.0))), 2.0 * pace)       # pour toward the back, slowly
        self.hold(1.0 * pace)
        self.go(i, Pose(top, up), 1.0 * pace)

    def place_inverted(self, i: int, xy: np.ndarray, rest_z: float, pace: float = 1.0) -> None:
        """Carry body ``i`` over ``xy`` 0.15 m above ``rest_z``, turn it mouth down there, lower its origin to
        ``rest_z`` (+2 mm) and let go."""
        p = self.pose(i)
        high = max(float(p.pos[2]), rest_z + 0.15)
        self.go(i, Pose(np.array([p.pos[0], p.pos[1], high]), p.quat), 1.0 * pace)
        self.go(i, Pose(np.array([xy[0], xy[1], high]), p.quat), 2.0 * pace)
        down = turn_x(p.quat, math.pi)
        self.go(i, Pose(np.array([xy[0], xy[1], high]), down), 1.5 * pace)
        self.go(i, Pose(np.array([xy[0], xy[1], rest_z + 0.002]), down), 1.5 * pace)
        self.release(i)

    def release(self, i: int, settle: float = 0.5) -> None:
        self.grip[:, i] = 0.0
        self.hold(settle)

    def inside_bowl(self, bowl: str) -> tuple[float, float]:
        """Half extents a vessel's origin may roam over ``bowl`` (its inner box less a 0.15 m margin)."""
        return 0.5 * BOWL[0] - 0.15, 0.5 * BOWL[1] - 0.15


__all__ = ["CARRY", "DunkRunner", "turn_x"]
