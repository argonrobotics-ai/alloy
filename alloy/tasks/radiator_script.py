# SPDX-License-Identifier: Apache-2.0
"""behind-radiator's scripted brush: dips in the cut pail, lanes along the gap from each end of the radiator, and
sweeps down the middle from above it. Until a robot holds it, the brush is a kinematic body put at each exchange's
pose (:func:`alloy.tasks.radiator_task.place`).

The reach (``radiator_brush``): the hand stays 35 mm outside the radiator's box, so along the gap the tips reach to
26 mm from the middle; from above, the TCP stays between the radiator's top (+35 mm) and the stool, so the brush
turns from along the gap (``alpha`` 0) to head down (-90) as its tips go down the middle, to z 0.343 m. The bottom of
the middle below that is the pocket the task does not score.

The moves:
- **load**: over the pail with the head down, the tips to 10 mm off its floor (20 mm into the paint) for ``DIP_S``,
  then up and still for ``DRAIN_S`` while the load past the held share drips back (``DRIP`` 0.5 1/s). 24 s leaves
  e^-12 of the excess, 8 nL: were it all to land in one 4 mm cell it would dry under the 1 µm that marks a protected
  cell. The stores cast each exchange's drip where the brush is, so a 12 s drain's 3 µL fell on the baseboard's
  ledge as each lane came to the wall (0.03-0.05 µL a lane), in the same cells lane after lane (31 dips). Gym-Anything
  gathers a brush's drips into a pendant drop released at the Tate volume (``PT/drips.py:25, :37-53``) and never
  releases the sub-drop rest (``PT/task.py:652-656``). A dip is good for (held - retained) of the 20 mm's wetted
  pores, 2.8 mL; a lane lays about 1.5 mL (0.29 m in at 99 µm over the 52 mm the plug-in meters: whole 4 mm cells)
  and a sweep 1.6 mL (96 µm over 56 mm), so the brush goes back to the pail before every lane and every sweep
  (``LANES_PER_DIP``). The store fills to the planned 20 mm's capacity at every dip (its dip is a point test: the
  depth is never read), while the pail's level falls from 29.8 to 25.5 mm over the run (drips go to the pool's own
  tank): Gym-Anything's last dips would wet 15.7 mm, 3.8 mL (-32 %), 1.9 mL usable, still over a sweep's 1.6 mL;
- **lanes** from an end: in along the gap 2 mm off the wall, onto the wall, a stroke in to the reach and one back
  out to the band's end (the lay-off), off the wall, ``PITCH`` up to the next lane; lanes high enough for the hand
  to pass over the radiator's top cross the middle;
- **sweeps** down the middle: in along the gap just over the radiator's top, onto the wall, down the middle while
  the head turns down, back up, off; at each of ``MIDDLE_X``.
Between them the brush goes through the room in front of the radiator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.tasks import radiator_brush as rb
from alloy.tasks import radiator_scene as rs

SUB = 0.025                     # s, one exchange: a brushing stroke moves 3.75 mm (under a 4 mm cell)
SPEED = 0.15                    # m/s brushing
TRAVEL = 0.30                   # m/s in the air
LIFT = 0.002                    # m off the wall between lanes (the handle keeps 2 mm from the radiator's back)
PITCH = 0.045                   # m between lanes (the brush is 50.8 mm wide)
DIP_S, DRAIN_S = 2.0, 24.0      # s
ABOVE = 0.30                    # m: the tips over the pail's floor while draining
DIP = 0.010                     # m: the tips over the pail's floor in a dip
STAGE_Y, STAGE_X = -0.35, 0.40  # m: the room-side point in front of each end
X_OUT = 0.5 * rs.RAD_LEN        # m: a lane's outer end (the band's end)
X_IN = 0.5 * rs.RAD_LEN + rb.HAND_CLEAR - float(rb.TIP[0]) + 5e-4    # m from the middle: the reach along the gap
TCP_TOP = rs.RAD_HEIGHT + rb.HAND_CLEAR + 1.6e-3                      # m: the TCP's height on the sweeps
Z_LOW = rs.BAND[2] + 0.5 * rb.WIDTH + 1e-3                            # m: the lowest lane's centre (1 mm up)
Z_HIGH = rs.BAND[3] - 0.5 * rb.WIDTH - 1e-3                           # the highest (1 mm under the apron)
MIDDLE_X = (-0.015, 0.0, 0.015)    # m: the sweeps
LANES_PER_DIP = 1                  # a dip lays one lane (1.4 of 2.8 mL usable; module docstring)
SWEEP_SEGMENTS = 24


@dataclass(frozen=True)
class Move:
    """A straight move of the bristle tips' centre to ``tip`` [m, world] while the brush turns to ``alpha`` [deg]
    (the shorter way), at ``speed`` [m/s], then ``hold`` s still."""

    tip: tuple[float, float, float]
    alpha: float
    speed: float = SPEED
    hold: float = 0.0


def _turn(a: float, b: float) -> float:
    return (b - a + 180.0) % 360.0 - 180.0


def poses(start: tuple[np.ndarray, float], moves: list[Move], dt: float = SUB) -> tuple[np.ndarray, np.ndarray]:
    """The tips' centre [N, 3] and the attitude [N] at the start of each exchange. A turn counts as the arc the
    TCP sweeps (|TIP| x angle) for the move's duration."""
    tip, alpha = np.asarray(start[0], np.float64), float(start[1])
    tips, alphas = [], []
    reach = float(np.linalg.norm(rb.TIP))
    for m in moves:
        to, turn = np.asarray(m.tip, np.float64), _turn(alpha, m.alpha)
        span = max(float(np.linalg.norm(to - tip)), reach * math.radians(abs(turn)))
        n = max(1, int(math.ceil(span / (m.speed * dt) - 1e-9)))
        for k in range(n):
            tips.append(tip + (to - tip) * (k / n))
            alphas.append(alpha + turn * (k / n))
        tip, alpha = to, alpha + turn
        for _ in range(int(round(m.hold / dt))):
            tips.append(tip.copy())
            alphas.append(alpha)
    return np.asarray(tips), np.asarray(alphas)


def pail_tip(z: float) -> tuple[float, float, float]:
    x, y, z0 = rb.PAIL_POS
    return (x, y, z0 + rb.PAIL_WALL + z)


def load() -> list[Move]:
    """Over the pail head down, a dip, then still while the excess drips back."""
    return [Move(pail_tip(ABOVE), -90.0, TRAVEL), Move(pail_tip(DIP), -90.0, 0.1, hold=DIP_S),
            Move(pail_tip(ABOVE), -90.0, 0.1, hold=DRAIN_S)]


def _wall(x: float, z: float, lift: bool = False) -> tuple[float, float, float]:
    return tuple(float(v) for v in rb.on_wall(x, z, LIFT if lift else 0.0))


def lanes(side: int, rows: list[float]) -> list[Move]:
    """Lanes from the radiator's left (``side`` -1) or right (+1) end at heights ``rows`` [m], from and back to the
    room-side point in front of that end."""
    a = 0.0 if side < 0 else 180.0
    out = [Move((side * STAGE_X, STAGE_Y, rows[0]), a, TRAVEL), Move(_wall(side * STAGE_X, rows[0], True), a, TRAVEL)]
    for z in rows:
        inner = -0.03 if z >= TCP_TOP else X_IN                  # over the radiator's top the hand may cross
        out += [Move(_wall(side * X_OUT, z, True), a), Move(_wall(side * X_OUT, z), a),
                Move(_wall(side * inner, z), a), Move(_wall(side * X_OUT, z), a), Move(_wall(side * X_OUT, z, True), a)]
    return out + [Move(_wall(side * STAGE_X, rows[-1], True), a, TRAVEL),
                  Move((side * STAGE_X, STAGE_Y, rows[-1]), a, TRAVEL)]


def _sweep(x: float, down: bool) -> list[Move]:
    """Down (or up) the middle at ``x`` with the TCP at ``TCP_TOP``: sin(alpha) = (z_tip - TCP_TOP)/TIP_x."""
    a0, a1 = (0.0, -90.0) if down else (-90.0, 0.0)
    angles = np.linspace(a0, a1, SWEEP_SEGMENTS + 1)[1:]
    return [Move(_wall(x, TCP_TOP + float(rb.TIP[0]) * math.sin(math.radians(a))), float(a)) for a in angles]


def sweeps(xs: tuple[float, ...] = MIDDLE_X) -> list[Move]:
    out = [Move((-STAGE_X, STAGE_Y, TCP_TOP), 0.0, TRAVEL), Move(_wall(-STAGE_X, TCP_TOP, True), 0.0, TRAVEL)]
    for x in xs:
        out += [Move(_wall(x, TCP_TOP, True), 0.0), Move(_wall(x, TCP_TOP), 0.0), *_sweep(x, True),
                *_sweep(x, False), Move(_wall(x, TCP_TOP, True), 0.0)]
    return out + [Move(_wall(-STAGE_X, TCP_TOP, True), 0.0, TRAVEL), Move((-STAGE_X, STAGE_Y, TCP_TOP), 0.0, TRAVEL)]


def rows(pitch: float = PITCH) -> list[float]:
    n = int(math.ceil((Z_HIGH - Z_LOW) / pitch - 1e-9))
    return [float(z) for z in np.linspace(Z_LOW, Z_HIGH, n + 1)]


def side(s: int, zs: list[float], lanes_per_dip: int = LANES_PER_DIP) -> list[Move]:
    """The lanes at heights ``zs`` from end ``s`` (-1 left, +1 right), a dip before each ``lanes_per_dip``."""
    out: list[Move] = []
    for k in range(0, len(zs), lanes_per_dip):
        out += load() + lanes(s, zs[k:k + lanes_per_dip])
    return out


def script(lanes_per_dip: int = LANES_PER_DIP, pitch: float = PITCH) -> tuple[tuple[np.ndarray, float], list[Move]]:
    """The whole job from over the pail: each side's lanes (:func:`side`); a dip before each sweep down the middle;
    then back over the pail."""
    start = (np.asarray(pail_tip(ABOVE)), -90.0)
    zs = rows(pitch)
    moves = side(-1, zs, lanes_per_dip) + side(1, zs, lanes_per_dip)
    for x in MIDDLE_X:
        moves += load() + sweeps((x,))
    return start, moves + [Move(pail_tip(ABOVE), -90.0, TRAVEL)]


__all__ = ["DRAIN_S", "DIP_S", "LANES_PER_DIP", "MIDDLE_X", "Move", "PITCH", "SPEED", "SUB", "TCP_TOP", "X_IN",
           "lanes", "load",
           "pail_tip", "poses", "rows", "script", "side", "sweeps"]
