# SPDX-License-Identifier: Apache-2.0
"""Unit checks on a recording, before any fit: degrees given as radians and milliseconds given as seconds.

The fit works in radians and seconds. A recording in degrees is the classic mistake: the numbers are 57x too large,
the simulated robot is commanded far outside its joint ranges, and a fit would return nonsense with a straight
face. ``check`` refuses such a file and names the option to pass; ``convert`` applies a stated unit.
"""

from __future__ import annotations

import math

import numpy as np

from alloy.sysid.errors import SysidError
from alloy.sysid.recording import Episode, Recording

ANGLE_UNITS = ("rad", "deg")
TIME_UNITS = ("s", "ms")
RANGE_SLACK = 0.10          # rad (or m) a measured position may sit outside the joint's range (offset, soft stop)
MAX_DT = 0.5                # s: a slower sample period is a time-unit mistake or not a motion recording
MIN_DT = 1e-5


def convert(rec: Recording, angle: str = "rad", time: str = "s", hinge: np.ndarray | None = None) -> Recording:
    """``rec`` in radians and seconds, given the units its file is in. ``hinge`` (J,) marks the joints that are
    angles (slides stay in metres); None means all. Example: ``convert(rec, angle="deg")``."""
    if angle not in ANGLE_UNITS or time not in TIME_UNITS:
        raise SysidError(f"units angle={angle!r}, time={time!r} are not known",
                         f"angle is one of {ANGLE_UNITS}, time one of {TIME_UNITS}", field="units")
    if angle == "rad" and time == "s":
        return rec
    k = np.ones(len(rec.joints))
    if angle == "deg":
        k = np.where(np.ones(len(rec.joints), bool) if hinge is None else hinge, math.pi / 180.0, 1.0)
    s = 1e-3 if time == "ms" else 1.0

    def scaled(e: Episode) -> Episode:
        return Episode(e.t * s, e.command * k, e.position * k,
                       None if e.velocity is None else e.velocity * k / s, e.torque)

    return Recording(list(rec.joints), [scaled(e) for e in rec.episodes], rec.source,
                     [*rec.notes, f"converted from angle={angle}, time={time}"])


def check(rec: Recording, lower: np.ndarray, upper: np.ndarray, hinge: np.ndarray) -> None:
    """Refuse a recording whose numbers cannot be radians and seconds for this robot.

    ``lower`` / ``upper`` (J,): the joints' ranges from the robot file (``-inf`` / ``inf`` when a joint has none);
    ``hinge`` (J,) bool. Example: ``check(rec, lo, hi, hinge)`` raises ``SysidError`` for a file in degrees."""
    _check_time(rec)
    pos = np.concatenate([e.position for e in rec.episodes])
    cmd = np.concatenate([e.command for e in rec.episodes])
    for j, name in enumerate(rec.joints):
        _check_joint(name, pos[:, j], cmd[:, j], float(lower[j]), float(upper[j]), bool(hinge[j]))


def _check_time(rec: Recording) -> None:
    dt = rec.dt
    if dt > MAX_DT:
        guess = "the time column looks like milliseconds: pass --time-unit ms (Python: time_unit='ms')" \
            if dt >= 1.0 else "record at 20 Hz or faster"
        raise SysidError(f"the median sample period is {dt:.4g} s, too slow for a joint recording in seconds", guess,
                         field="t")
    if dt < MIN_DT:
        raise SysidError(f"the median sample period is {dt:.3g} s, shorter than any joint encoder's",
                         "give time in seconds (a column in microseconds or nanoseconds must be converted first)",
                         field="t")


def _check_joint(name: str, pos: np.ndarray, cmd: np.ndarray, lo: float, hi: float, hinge: bool) -> None:
    peak = float(max(np.max(np.abs(pos)), np.max(np.abs(cmd))))
    ranged = math.isfinite(lo) and math.isfinite(hi) and lo < hi
    outside = ranged and (float(pos.min()) < lo - RANGE_SLACK or float(pos.max()) > hi + RANGE_SLACK)
    if not outside and not (hinge and not ranged and peak > 4.0 * math.pi):
        return
    where = f"range [{lo:.3g}, {hi:.3g}] rad" if ranged else "any plausible angle in radians"
    span = f"[{float(pos.min()):.4g}, {float(pos.max()):.4g}]"
    if hinge and _fits_as_degrees(pos, lo, hi, ranged):
        raise SysidError(f"joint {name!r}: measured positions span {span}, outside its {where}; as degrees they fit",
                         "the recording looks like degrees: pass --angle-unit deg (Python: angle_unit='deg'), or "
                         "convert it to radians", field="angle_unit")
    raise SysidError(f"joint {name!r}: measured positions span {span}, outside its {where}",
                     "check the units (radians for hinges, metres for slides), the joint order, and that the "
                     "recording's zero angles are the robot file's (an offset between the two must be removed first)",
                     field="data")


def _fits_as_degrees(pos: np.ndarray, lo: float, hi: float, ranged: bool) -> bool:
    rad = pos * math.pi / 180.0
    if not ranged:
        return float(np.max(np.abs(rad))) <= 2.0 * math.pi
    return float(rad.min()) >= lo - RANGE_SLACK and float(rad.max()) <= hi + RANGE_SLACK


__all__ = ["ANGLE_UNITS", "TIME_UNITS", "check", "convert"]
