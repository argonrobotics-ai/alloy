# SPDX-License-Identifier: Apache-2.0
"""One time grid in integer base ticks (SPEC D4, §4.1).

Every step size in a ``Simulation`` run is ``control_dt / 2**j`` with ``0 <= j <= TICK_EXP``; the base tick is
``control_dt / 2**TICK_EXP``. Time is an int64 tick counter and ``t = ticks * tick`` in float64, so it is exact across
dt changes (never ``step * dt``, never accumulated float seconds).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from alloy.core.errors import ClockError

TICK_EXP: Final[int] = 10
INT64_MAX: Final[int] = 2**63 - 1


def _check_control_dt(control_dt: float) -> None:
    if not (isinstance(control_dt, (int, float)) and math.isfinite(control_dt) and control_dt > 0.0):
        raise ClockError(f"control_dt must be a positive finite number of seconds, got {control_dt!r}")


def grid_dt(control_dt: float, j: int) -> float:
    """``control_dt / 2**j`` for ``0 <= j <= TICK_EXP`` (exact: division by a power of two)."""
    _check_control_dt(control_dt)
    if not (isinstance(j, int) and 0 <= j <= TICK_EXP):
        raise ClockError(f"grid index j must be an int in [0, {TICK_EXP}], got {j!r}")
    return control_dt / float(2**j)


def grid_index(dt: float, control_dt: float) -> int:
    """The j with ``control_dt / 2**j == dt`` exactly; ``ClockError`` if ``dt`` is off the grid."""
    _check_control_dt(control_dt)
    for j in range(TICK_EXP + 1):
        if control_dt / float(2**j) == dt:
            return j
    raise ClockError(f"dt={dt!r} is not on the grid control_dt/2**j (control_dt={control_dt!r}, j<={TICK_EXP})")


def snap_to_grid(dt: float, control_dt: float) -> float:
    """Largest ``control_dt / 2**j <= dt`` with ``j <= TICK_EXP``.

    ``dt >= control_dt`` gives ``control_dt``. ``ClockError`` if ``dt`` is below the base tick (no grid value fits;
    callers such as the assembler turn this into ``PrecisionSpecError`` with their numbers) or not a positive number.
    """
    _check_control_dt(control_dt)
    if not (isinstance(dt, (int, float)) and dt > 0.0) or math.isnan(dt):
        raise ClockError(f"dt must be a positive number, got {dt!r}")
    for j in range(TICK_EXP + 1):
        value = control_dt / float(2**j)
        if value <= dt:
            return value
    raise ClockError(f"dt={dt!r} is below the base tick control_dt/2**{TICK_EXP} = {control_dt / 2**TICK_EXP!r}")


@dataclass
class SimClock:
    """Tick clock of one ``Simulation``: ``dt = control_dt / 2**j``, ``t = ticks * tick``."""

    control_dt: float
    j: int
    ticks: int = 0

    def __post_init__(self) -> None:
        grid_dt(self.control_dt, self.j)  # validates control_dt and j
        if not (isinstance(self.ticks, int) and 0 <= self.ticks <= INT64_MAX):
            raise ClockError(f"ticks must be an int in [0, 2**63), got {self.ticks!r}")

    @property
    def tick(self) -> float:
        """Base tick ``control_dt / 2**TICK_EXP`` [s]."""
        return self.control_dt / float(2**TICK_EXP)

    @property
    def dt(self) -> float:
        """Current step ``control_dt / 2**j`` [s]."""
        return self.control_dt / float(2**self.j)

    @property
    def ticks_per_step(self) -> int:
        """Base ticks per step at the current dt, ``2**(TICK_EXP - j)``."""
        return 2 ** (TICK_EXP - self.j)

    @property
    def steps_per_period(self) -> int:
        """Steps of the current dt in one control period, ``2**j``."""
        return 2**self.j

    @property
    def period_ticks(self) -> int:
        """Base ticks per control period, ``2**TICK_EXP``."""
        return 2**TICK_EXP

    def set_dt(self, dt: float) -> None:
        """Switch to ``dt``; it must be exactly on the grid, else ``ClockError`` (a ``ValueError``)."""
        self.j = grid_index(dt, self.control_dt)

    def advance(self, n: int = 1) -> None:
        """Advance ``n`` steps of the current dt: ``ticks += n * 2**(TICK_EXP - j)``."""
        if not isinstance(n, int) or n < 0:
            raise ClockError(f"advance needs a non-negative int, got {n!r}")
        ticks = self.ticks + n * self.ticks_per_step
        if ticks > INT64_MAX:
            raise ClockError(f"tick counter overflows int64 ({ticks})")
        self.ticks = ticks

    @property
    def t(self) -> float:
        """Simulated time ``ticks * tick`` in float64 (one rounding; exact across dt changes)."""
        return float(self.ticks) * self.tick

    @property
    def period(self) -> int:
        """Completed control periods (floor of ticks / period_ticks)."""
        return self.ticks >> TICK_EXP

    @property
    def at_period_boundary(self) -> bool:
        """True when ``ticks`` is a whole number of control periods (dt changes happen only here)."""
        return self.ticks % (2**TICK_EXP) == 0
