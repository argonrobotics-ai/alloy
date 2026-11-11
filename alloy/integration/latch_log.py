# SPDX-License-Identifier: Apache-2.0
"""One warning per world when the health monitor latches it (SCR-JOB2-4).

``HealthMonitor.latch`` restores an INVALID world from the last good snapshot every step and keeps it frozen until it
is reset; until now it logged nothing. In JOB2's pour scene a PEN_FAIL latched the world within a second, and the robot
looked like an arm that would not move (commands still wrote ``ctrl``; the restore put the arm, and the snapshot's
``mjw_data.ctrl``, back every step).

``LatchLog.poll`` runs once per control period, after the period (and any rollback) is final. It warns once per newly
latched world, naming its flags and first bad step, and keeps the set in ``latched``. With WP5's first-failure snapshot in the record (9e19d854) each warning line also says what first made
one of its worlds INVALID (``FirstFailure.text``: the check, sim time, deepest contact and its shapes or particle,
speed and body), since a latched world reads clean afterwards. On CUDA it reads a pinned host
copy of the health record that the previous poll enqueued, after an event that is a period old by then, so a period
never waits on the device. The warning then lags the latch by at most one period. On CPU the copy is synchronous and
read at once.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from alloy.precision.types import FAIL_FIELDS, FirstFailure, HealthFlags

log = logging.getLogger("alloy.integration")

__all__ = ["LatchLog"]

SHOWN_WORLDS = 8          # world ids listed per warning line; the count is always given


def _first_failure(rec: np.ndarray) -> FirstFailure | None:
    """WP5's snapshot from a host copy of the record; None for records without the ``fail_*`` fields (fakes)."""
    names = rec.dtype.names or ()
    if not all(f"fail_{n}" in names for n in FAIL_FIELDS):
        return None
    return FirstFailure(rec["first_bad_step"].astype(np.int64), *(rec[f"fail_{n}"] for n in FAIL_FIELDS))


def _flag_names(bits: int) -> str:
    names = [f.name for f in HealthFlags if bits & int(f) and f is not HealthFlags.LATCHED]
    return "|".join(names) or "none"


class LatchLog:
    """Per-world latch warnings from a health record array (``HealthMonitor.record``, or a fake's ``health``)."""

    def __init__(self, record: wp.array) -> None:
        self.record = record
        dev = wp.get_device(record.device)
        self._cuda = bool(dev.is_cuda)
        self._host = wp.empty(record.shape, dtype=record.dtype, device="cpu", pinned=self._cuda)
        self._event = wp.Event(dev) if self._cuda else None
        self._pending = False
        self.latched: set[int] = set()

    @classmethod
    def for_health(cls, health: Any) -> LatchLog | None:
        """None without a health monitor or a per-world record array with ``flags``."""
        if health is None:
            return None
        record = getattr(health, "record", None)
        if not isinstance(record, wp.array):
            record = getattr(health, "health", None)
        if not isinstance(record, wp.array) or "flags" not in getattr(record.dtype, "vars", {}):
            return None
        return cls(record)

    def poll(self) -> list[int]:
        """Warn about worlds latched since the last poll; returns them (sorted)."""
        if self._cuda:
            fresh = self._read_pending()
            self._enqueue()
        else:
            self._enqueue()
            fresh = self._host.numpy()
        return [] if fresh is None else self._warn(fresh)

    def reset(self, world_mask: wp.array | None = None) -> None:
        """Forget the reset worlds (all for None), so a later latch warns again."""
        if world_mask is None:
            self.latched.clear()
            return
        mask = np.asarray(world_mask.numpy() if isinstance(world_mask, wp.array) else world_mask)
        self.latched -= {int(w) for w in np.flatnonzero(mask)}

    def _enqueue(self) -> None:
        wp.copy(self._host, self.record)
        if self._event is not None:
            wp.record_event(self._event)
        self._pending = True

    def _read_pending(self) -> np.ndarray | None:
        if not self._pending:
            return None
        wp.synchronize_event(self._event)          # recorded a period ago: done unless the host ran a period ahead
        return self._host.numpy().copy()

    def _warn(self, rec: np.ndarray) -> list[int]:
        flags = rec["flags"].astype(np.int64)
        on = np.flatnonzero(flags & int(HealthFlags.LATCHED))
        new = sorted(int(w) for w in on if int(w) not in self.latched)
        if not new:
            return []
        self.latched.update(new)
        groups: dict[str, list[int]] = {}
        for w in new:
            groups.setdefault(_flag_names(int(flags[w])), []).append(w)
        first = _first_failure(rec)
        for names, worlds in groups.items():
            steps = sorted({int(rec["first_bad_step"][w]) for w in worlds})
            shown = ", ".join(str(w) for w in worlds[:SHOWN_WORLDS]) + (" ..." if len(worlds) > SHOWN_WORLDS else "")
            known = first is not None and first.world(worlds[0]) is not None
            why = f"; first failure of world {worlds[0]}: {first.text(worlds[0])}" if known else ""
            log.warning("health latched %d world(s) [%s]: %s from step %s; each is put back to its last good state "
                        "every step and stays frozen until reset (commands to it no longer move anything)%s",
                        len(worlds), shown, names, steps[0] if len(steps) == 1 else f"{steps[0]}..{steps[-1]}", why)
        return new
