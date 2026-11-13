# SPDX-License-Identifier: Apache-2.0
"""Park health-latched worlds in a coupled scene's Scheduler (sample-sweep F2, WP11; opt-in ``ALLOY_PARK_LATCHED=1``).

Health freezes an INVALID world by restoring it from the period start after every exchange (``simulation.py``, Latch
source), but the Scheduler still steps it and its shared exchange clock still reads it. A world that froze on a start
overlap (a sheet started inside a box wall, a soft pad flush on the ground) pushes the same vertices out on every
exchange, the contact band and VBD's stable step read that speed, the clock halves the exchange step, the same depth is
undone faster at the finer step, and within three periods the clock is at control_dt / 2^j_max with VBD at its top
rungs: every world of the scene then steps at minutes per control step (known gap coupling-frozen-world-sets-clock).

Gap #10b's parked worlds (``alloy.coupling.parked``) are what a frozen world needs: a parked world holds no clock bound,
asks for no re-run and is held. With the option on, ``park_latched`` parks each latched world that is not parked yet,
at the end of the period it latched in. It reads the health record's flags at once, on CUDA too: ``LatchLog.poll``
reads a period-old copy (so a period never waits on the device), and the period after a latch would still run at the
frozen world's clock, over 37 min of wall time at 1,024 worlds on an L4. The Scheduler's own report has already waited
for the device in that period (``SchedulerMixin.report`` reads the stable steps), so the read adds one small copy. A
reset unparks the world (``ParkedWorldsMixin.reset_worlds``) and unlatches it (``HealthMonitor.reset_worlds``) together.

``prepare`` (at build) keeps per-world contact-band rows, and each park here passes ``follow=True``: a parked world
leaves the band bound at once (``alloy.coupling.parked.band_over``) instead of holding the next period at its need,
and the substep rungs it raised drop at once. Other parks (``clock_worlds=``'s driver) keep the old clock rule.

Left alone: scenes run with ``clock_worlds=`` (its driver parks and unparks residents itself) and steppers without
``park_worlds`` (no shared exchange clock to protect).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from alloy.precision.types import HealthFlags

log = logging.getLogger("alloy.integration")

ENV = "ALLOY_PARK_LATCHED"


def enabled() -> bool:
    """True when ``ALLOY_PARK_LATCHED`` is set to anything but empty or 0 (read at each call)."""
    return os.environ.get(ENV, "0").strip() not in ("", "0")


def prepare(sim: Any) -> None:
    """At build, with the option on: ask the stepper for per-world contact-band rows, so the world parked at the first
    latch leaves the band bound too (``parked.band_over``; without rows the next clock still reads its band need)."""
    keep = getattr(sim.stepper, "keep_world_rows", None)
    if enabled() and callable(keep) and callable(getattr(sim.stepper, "park_worlds", None)):
        keep(True)


def latched_now(record: Any) -> np.ndarray:
    """bool [W]: the worlds whose health record has LATCHED set now (a host read of ``record``, synchronous)."""
    return (np.asarray(record.numpy()["flags"]).astype(np.int64) & int(HealthFlags.LATCHED)) != 0


def park_latched(sim: Any) -> list[int]:
    """At a period's end, after ``LatchLog.poll``: park every latched world not parked yet, so it leaves the shared
    clock from the next period on; returns the worlds parked."""
    log_ = getattr(sim, "latch_log", None)
    park = getattr(sim.stepper, "park_worlds", None)
    if not enabled() or log_ is None or park is None or getattr(sim, "world_clock", None) is not None:
        return []
    fresh = latched_now(log_.record) & ~np.asarray(sim.stepper.parked, dtype=bool)
    worlds = np.flatnonzero(fresh).tolist()
    if not worlds:
        return []
    park(np.asarray(worlds, dtype=np.int64), sim.state, follow=True)
    log.warning("health-latched world(s) %s parked: they leave the shared exchange clock until reset (%s)", worlds, ENV)
    return worlds
