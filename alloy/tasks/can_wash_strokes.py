# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can, robot stage: a whole revolution of the can on the turntable in the UR10e's strokes (the lathe's
turns; ``can_wash_turn``).

One grip cannot turn the can a whole revolution (the UR's rim grasp reached 23 of 24 azimuths of an orbit), so a
revolution is ``ceil(2 pi / stroke)`` strokes: grip, turn ``stroke``, let go, lift, swing back in the air, come down
and grip again. The can is round and its table turns with it, so every stroke is the same motion in the world: its
joint paths are planned once and replayed. The FR3's nozzle stands at one azimuth per kind of band (stage 1's poses
there): for the inside bands its tip is over the can on the FR3's side, at ``SECTORS["inside"]``, aiming across at
the far wall (180° on); for the outside bands it aims at the wall at ``SECTORS["outside"]`` from outside. The grip
arc is centred at ``ARC_CENTRE``, a right angle off the line to the FR3 on the side away from the UR, clear of every
aim and tip, so the hand never comes between the wand and the can (its fingers are shapes: jets on them would leave
the books).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.tasks.can_wash_table import TurnLog
from alloy.tasks.can_wash_turn import APPROACH, OPEN, PINCH_N, RELEASE_N, Lathe, eased
from alloy.tasks.rinse_flush_policy import CONTROL_DT

STROKE = math.radians(60.0)          # one stroke's turn (at 90° the wrist lifting off at the arc's end met the valve)
SECTORS = {"inside": math.radians(218.0), "outside": math.radians(278.0)}   # the FR3's nozzle azimuths (world; 30°
                                     # either side of its line, 248°: both reached in every band, reach notes)
ARC_CENTRE = math.radians(125.0)     # the grip arc's centre (95-155°), away from the UR (at 0°) and the FR3's tip;
                                     # at 120° the UR's IK misses 113 poses of the stroke, at 158° its pads slip 18 mm
SWING_S = 1.5                        # s, the swing back in the air
OPEN_S, CLOSE_S, HOLD_S = 0.4, 0.6, 0.3   # s: the jaws opening, closing, and the hold before a turn


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def grip_azimuth(lathe: Lathe, phi: float) -> float:
    """The world azimuth (about the can's axis) of the grip at turn ``phi``."""
    p = lathe.grip_pose(phi)[:2] - lathe.centre()[:2]
    return math.atan2(p[1], p[0])


def start_phi(lathe: Lathe, stroke: float = STROKE, centre: float = ARC_CENTRE) -> float:
    """The turn at which a stroke's grip starts so that its arc is centred at ``centre``."""
    return wrap(centre - 0.5 * stroke - grip_azimuth(lathe, 0.0))


def arc_clearance(lathe: Lathe, azimuth: float, stroke: float = STROKE) -> float:
    """The least angle between the stroke's grip arc and ``azimuth`` [rad]."""
    a0 = grip_azimuth(lathe, start_phi(lathe, stroke))
    return min(abs(wrap(a0 + s - azimuth)) for s in np.linspace(0.0, stroke, 91))


def nozzle_azimuths() -> dict[str, float]:
    """Where the FR3's jets and nozzle stand about the can's axis: the inside tip, the inside aim (across), the
    outside aim (the tip outside it)."""
    return {"inside tip": SECTORS["inside"], "inside aim": wrap(SECTORS["inside"] + math.pi),
            "outside aim": SECTORS["outside"]}


class RevolutionError(RuntimeError):
    """The strokes did not turn the can the whole way (``revolve_to``)."""


@dataclass
class StrokePlan:
    """The joint paths of one stroke (UR, [J] per period): down onto the rim, the turn, up off it, the swing back."""

    down: list[np.ndarray]
    turn: list[np.ndarray]
    up: list[np.ndarray]
    swing: list[np.ndarray]
    misses: int = 0

    @staticmethod
    def make(lathe: Lathe, tracker: Any, omega: float, stroke: float = STROKE, seed: Any = None) -> StrokePlan:
        """A stroke of ``stroke`` (at most ``STROKE``) from the full stroke's start, so a short closing stroke starts
        where every swing ends. ``seed`` ([1, 1, J]) keeps the arm on a known IK branch (a re-centred stroke on the
        full stroke's); without it the start is the widest-margin IK solution."""
        p0 = start_phi(lathe)
        before = tracker.plan.misses
        down_poses = [lathe.grip_pose(p0, APPROACH * (1 - k / 50)) for k in range(51)]
        down = tracker.track(down_poses, tracker.start(down_poses) if seed is None else seed)
        turn = tracker.track([lathe.grip_pose(p0 + a) for a in eased(stroke, omega, CONTROL_DT)], down[-1][None, None])
        up = tracker.track([lathe.grip_pose(p0 + stroke, APPROACH * k / 25) for k in range(1, 26)],
                           turn[-1][None, None])
        back = eased(-stroke, stroke / SWING_S, CONTROL_DT)
        swing = tracker.track([lathe.grip_pose(p0 + stroke + a, APPROACH) for a in back], up[-1][None, None])
        return StrokePlan(down, turn, up, swing, tracker.plan.misses - before)


@dataclass
class Revolution:
    """The books of a revolution in strokes: each stroke's ``TurnLog`` and the turn it made [rad]."""

    strokes: list[TurnLog] = field(default_factory=list)

    @property
    def turned(self) -> float:
        return float(sum(s.turned for s in self.strokes))

    def worst(self, name: str) -> float:
        return max(float(getattr(s, name)) for s in self.strokes)


def revolve(run: Any, plan: StrokePlan, n: int, before_turn: Any = None, after_turn: Any = None) -> Revolution:
    """``n`` strokes of the can held by ``run`` (``can_wash_turn.LatheRun``): grip, turn, let go, lift, swing back.
    ``before_turn(i)`` / ``after_turn(i)`` run around each turn (the valve's opening and shutting)."""
    rev = Revolution()
    for i in range(n):
        for q in plan.down:
            run.step(q, OPEN, RELEASE_N)
        for _ in range(int(round((CLOSE_S + HOLD_S) / CONTROL_DT))):
            run.step(plan.down[-1], 0.0, PINCH_N)
        if before_turn is not None:
            before_turn(i)
        rev.strokes.append(run.turn(plan.turn, settle=10))
        if after_turn is not None:
            after_turn(i)
        for _ in range(int(round(OPEN_S / CONTROL_DT))):
            run.step(plan.turn[-1], OPEN, RELEASE_N)
        for q in [*plan.up, *plan.swing]:
            run.step(q, OPEN, RELEASE_N)
    return rev


def ease_to(run: Any, q: np.ndarray) -> None:
    """The UR from its last command to ``q`` in the air, jaws open (SOFT-3's minimum-jerk ``smooth``)."""
    from alloy.tasks.paint_arm import smooth  # noqa: PLC0415

    if run.last_q is None:
        return
    for qk in smooth(np.asarray(run.last_q, np.float64), np.asarray(q, np.float64), CONTROL_DT, rate=0.8):
        run.step(qk, OPEN, RELEASE_N)


def revolve_to(run: Any, lathe: Lathe, tracker: Any, omega: float, total: float = 2.0 * math.pi,
               before_turn: Any = None, after_turn: Any = None, tol: float = math.radians(0.5),
               full: StrokePlan | None = None, recentre: bool = False) -> Revolution:
    """Strokes until the can has turned ``total`` (measured): full strokes (``full``, planned when not given), then
    one planned for what is left (so the band is sprayed once round, without the dose a whole extra stroke would
    add). With ``recentre`` each stroke is planned about the can's measured axis (the UR eased onto it in the air).
    Raises ``RevolutionError`` when three revolutions' worth of strokes have not turned it ``total``."""
    full, rev = full or StrokePlan.make(lathe, tracker, omega), Revolution()
    cap = 3 * math.ceil(total / STROKE)
    while rev.turned < total - tol:
        if len(rev.strokes) >= cap:
            raise RevolutionError(f"the can turned {math.degrees(rev.turned):.2f} of {math.degrees(total):.2f} deg in "
                                  f"{cap} strokes: the pinch is not turning it (pads slip, or the can is held)")
        left = total - rev.turned
        if recentre:                                   # each stroke about the can's measured axis (``LatheRun.offset``)
            at = dataclasses.replace(lathe, shift=tuple(float(v) for v in run.offset()))
            plan = StrokePlan.make(at, tracker, omega, stroke=min(left, STROKE), seed=full.down[0][None, None])
            ease_to(run, plan.down[0])
        else:
            plan = full if left >= STROKE else StrokePlan.make(lathe, tracker, omega, stroke=left)
        k = len(rev.strokes)
        before = None if before_turn is None else (lambda i, k=k: before_turn(k))
        after = None if after_turn is None else (lambda i, k=k: after_turn(k))
        rev.strokes += revolve(run, plan, 1, before, after).strokes
    return rev


__all__ = ["ARC_CENTRE", "SECTORS", "STROKE", "Revolution", "RevolutionError", "StrokePlan", "arc_clearance",
           "grip_azimuth", "nozzle_azimuths", "revolve", "revolve_to", "start_phi", "wrap"]
