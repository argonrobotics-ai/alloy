# SPDX-License-Identifier: Apache-2.0
"""A safe excitation trajectory to play on the real robot: small, slow motions of each joint inside its limits.

``trajectory`` returns commanded joint targets over time. Each joint gets, around a centre pose well inside its
range: a chirp (a sine whose frequency rises slowly, here 0.2 to 2 Hz) that starts and ends at rest, then two quick
small moves out and back with holds between. ``mode="sequential"`` (the default) moves one joint at a time while
the others hold; ``mode="together"`` moves all at once, each on its own frequency band, for a shorter recording.
The first second holds the centre pose: bring the robot there before playing the file.

Alloy never talks to hardware: you play the file with your own controller, at the file's rate, and log time, the
commanded targets and the measured joint positions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alloy.sysid.errors import SysidError

HOLD = 1.0                 # s at rest before and after
MARGIN = 0.05              # rad (m for a slide: scaled by SLIDE) kept from each joint limit
SLIDE = 0.1                # a slide's amplitude and margin, as a share of a hinge's numbers (m per rad)
F0, F1 = 0.2, 2.0          # Hz, chirp band
STEP_TIME = 0.15           # s, the quick moves (minimum-jerk)


@dataclass
class Excitation:
    """Commanded targets ``command`` (T, J) at times ``t`` (T,) for ``joints``; ``centre`` and ``amplitude`` (J,)."""

    joints: list[str]
    t: np.ndarray
    command: np.ndarray
    centre: np.ndarray
    amplitude: np.ndarray

    def save(self, path: str | Path) -> Path:
        """Write a CSV with columns ``t, cmd.<joint>...``. Example: ``exc.save("excite.csv")``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as fh:
            out = csv.writer(fh)
            out.writerow(["t"] + [f"cmd.{j}" for j in self.joints])
            for k in range(len(self.t)):
                out.writerow([f"{self.t[k]:.6f}"] + [f"{v:.8f}" for v in self.command[k]])
        return path


def _smooth(x: np.ndarray) -> np.ndarray:
    """Minimum-jerk blend 0 -> 1 for x in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (10.0 - 15.0 * x + 6.0 * x * x)


def _chirp(t: np.ndarray, seconds: float, f0: float, f1: float) -> np.ndarray:
    """Unit chirp over ``seconds`` with a soft start and end (zero position and velocity at both)."""
    phase = 2.0 * np.pi * (f0 * t + 0.5 * (f1 - f0) * t * t / seconds)
    window = _smooth(t / (0.15 * seconds)) * _smooth((seconds - t) / (0.15 * seconds))
    return np.where((t >= 0.0) & (t <= seconds), window * np.sin(phase), 0.0)


def _steps(t: np.ndarray, hold: float) -> np.ndarray:
    """Out (+0.5), back, out the other way (-0.5), back: quick moves with ``hold`` s between, from t = 0."""
    out = np.zeros_like(t)
    for k, level in enumerate((0.5, 0.0, -0.5, 0.0)):
        prev = (0.0, 0.5, 0.0, -0.5)[k]
        out = np.where(t >= k * hold, prev + (level - prev) * _smooth((t - k * hold) / STEP_TIME), out)
    return out


def _segment(t: np.ndarray, seconds: float, f0: float, f1: float) -> np.ndarray:
    """One joint's unit motion from t = 0: a chirp for 70 % of ``seconds``, then the quick moves."""
    chirp_s = 0.7 * seconds
    hold = 0.3 * seconds / 4.0
    return _chirp(t, chirp_s, f0, f1) + np.where(t > chirp_s, _steps(t - chirp_s, hold), 0.0)


def plan(lower: np.ndarray, upper: np.ndarray, hinge: np.ndarray, amplitude: float) -> tuple[np.ndarray, np.ndarray]:
    """``(centre, amplitude)`` per joint: the pose nearest zero that leaves room, and a motion that fits the range."""
    scale = np.where(hinge, 1.0, SLIDE)
    margin, want = MARGIN * scale, float(amplitude) * scale
    room = np.where(np.isfinite(upper - lower), (upper - lower) / 2.0 - margin, np.inf)
    if np.any(room <= 0.0):
        raise SysidError("a joint's range is narrower than the safety margin", "leave that joint out (--joints)",
                         field="joints")
    amp = np.minimum(want, 0.8 * room)
    lo = np.where(np.isfinite(lower), lower + margin + amp, -np.inf)
    hi = np.where(np.isfinite(upper), upper - margin - amp, np.inf)
    return np.clip(0.0, lo, hi), amp


def trajectory(joints: list[str], lower: Any, upper: Any, hinge: Any, *, rate: float = 100.0,
               seconds: float = 8.0, amplitude: float = 0.15, mode: str = "sequential") -> Excitation:
    """The excitation for ``joints`` with ranges ``lower`` / ``upper`` (J,). ``rate`` Hz, ``seconds`` per joint,
    ``amplitude`` rad (a slide: 0.1 m per rad). Example: ``trajectory(["elbow"], [-2], [2], [True])``."""
    if mode not in ("sequential", "together"):
        raise SysidError(f"mode={mode!r} is not known", "use 'sequential' or 'together'", field="mode")
    if not 20.0 <= rate <= 1000.0 or seconds < 2.0 or amplitude <= 0.0:
        raise SysidError("excite needs rate in 20..1000 Hz, seconds >= 2 per joint and amplitude > 0",
                         "for example rate=100, seconds=8, amplitude=0.15", field="rate")
    lower, upper, hinge = np.asarray(lower, float), np.asarray(upper, float), np.asarray(hinge, bool)
    centre, amp = plan(lower, upper, hinge, amplitude)
    count = len(joints)
    moving = seconds * (1 if mode == "together" else count)
    t = np.arange(int(round((2.0 * HOLD + moving) * rate)) + 1) / rate
    command = np.tile(centre, (len(t), 1))
    for j in range(count):
        if mode == "sequential":
            unit = _segment(t - HOLD - j * seconds, seconds, F0, F1)
            unit = np.where((t >= HOLD + j * seconds) & (t <= HOLD + (j + 1) * seconds), unit, 0.0)
        else:                                  # each joint on its own slice of the band, so they stay apart
            shift = 0.6 + 0.8 * (j / max(count - 1, 1))
            unit = _segment(t - HOLD, seconds, F0 * shift, F1 * shift)
            unit = np.where((t >= HOLD) & (t <= HOLD + seconds), unit, 0.0)
        command[:, j] += amp[j] * unit
    return Excitation(list(joints), t, command, centre, amp)


def load(path: str | Path) -> Excitation:
    """Read a file written by ``Excitation.save``. Example: ``load("excite.csv").command.shape``."""
    with Path(path).open(newline="") as fh:
        rows = list(csv.reader(fh))
    joints = [h[4:] for h in rows[0][1:]]
    table = np.asarray([[float(v) for v in r] for r in rows[1:] if r], np.float64)
    zero = np.zeros(len(joints))
    return Excitation(joints, table[:, 0], table[:, 1:], table[0, 1:], zero)


__all__ = ["Excitation", "load", "plan", "trajectory"]
