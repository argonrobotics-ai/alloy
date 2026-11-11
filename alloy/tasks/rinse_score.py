# SPDX-License-Identifier: Apache-2.0
"""Verdict rules shared by the rinse_* proof tasks, ported from Gym-Anything's surface-rinsing task set.

Sources (Gym-Anything's RoboWorld ``surface_rinsing`` environment; file:line):
- ``clip``, ``reduction`` (log reduction to a limit), ``excess``, the 0.9 fail scale: ``task_set/scoring.py:12-26``;
- the score: 1 if every gate passes, else 0.9 x progress; None (unverifiable) on a fault, an open ledger (water to
  1e-9 relative, species to 1e-9 relative or 1e-12 kg absolute) or a non-finite value: ``task_set/scoring.py:126-139``;
- common gates: ``water_off_and_hands_free`` (no outlet over 1e-4 gpm, nothing held), ``intact_and_not_dropped``,
  ``no_simulation_fault``: ``task.py:677-692``; floor water limit 50 mL: ``params.py:251-254`` (CRITERIA).

A gate is (passed, detail). ``passed`` is True or False only when the gate is read from the simulation's state;
None means this port does not judge Gym-Anything's criterion (the detail says why): never a literal True, never the
script's own flag. ``partly`` joins a criterion's parts. Success counts the judged gates only, and ``judged`` says how
many of Gym-Anything's criteria that is (audit AB-gym-3, 2026-11-09). A score or progress term that multiplies by a
gate counts the judged parts only: a None gate contributes 1.0, a False gate 0.0. A gate that is vacuous in
Gym-Anything itself is (None, "vacuous in Gym-Anything too: <why>").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

FAIL_SCALE = 0.9
OUTLET_OFF_GPM = 1.0e-4               # an outlet below this counts as off
FLOOR_WATER_ML = 50.0
LEDGER_REL = 1.0e-9
SPECIES_ABS_KG = 1.0e-12


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def reduction(start: float, value: float, limit: float) -> float:
    """Log reduction of a quantity that was above ``limit`` at reset (0 if it was not)."""
    if start <= limit:
        return 0.0
    return clip(math.log(start / max(value, 1e-30)) / math.log(start / limit))


def excess(value: float, limit: float) -> float:
    return clip((value - limit) / limit)


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return not isinstance(value, (int, float)) or math.isfinite(value)


def partly(*parts: bool | None) -> bool | None:
    """A criterion from its parts: False if a judged part fails, None if a part is not judged, else True."""
    if any(p is not None and not p for p in parts):       # numpy bools are not ``False``
        return False
    return None if any(p is None for p in parts) else True


HANDS_NOT_JUDGED = ("not judged, the port's hand has no finger pads; Gym-Anything: no pad touching a movable "
                    "body")


def hands_gate(outlet_off: bool, gripping: bool | None, **detail: Any) -> tuple[bool | None, dict]:
    """``water_off_and_hands_free``: the outlet part is read from the simulation; the hands part is only the script's
    own grip flag, so it can fail the gate (still gripping) but never pass it (``gripping`` None: no hand at all)."""
    hands = False if gripping else None
    out = {"outlet_off": bool(outlet_off), "hands": "still gripping" if gripping else HANDS_NOT_JUDGED}
    return partly(bool(outlet_off), hands), {**out, **detail}


@dataclass
class Verdict:
    """One world's verdict: each gate (passed, detail; passed None = not judged), progress in [0, 1], the score and
    success. ``extra`` names the port's own gates that are not Gym-Anything criteria (they count for success, not in
    ``judged``)."""

    criteria: dict[str, tuple[bool | None, Any]] = field(default_factory=dict)
    progress: float | None = None
    score: float | None = None
    terms: dict[str, float] = field(default_factory=dict)
    extra: tuple[str, ...] = ()

    @property
    def success(self) -> bool:
        """Every judged gate passes (at least one is judged)."""
        judged = [ok for ok, _ in self.criteria.values() if ok is not None]
        return bool(judged) and all(judged)

    def failed(self) -> list[str]:
        return [k for k, (ok, _) in self.criteria.items() if ok is not None and not ok]

    def not_judged(self) -> list[str]:
        """Gym-Anything criteria this port does not judge."""
        return [k for k, (ok, _) in self.criteria.items() if ok is None and k not in self.extra]

    @property
    def judged(self) -> tuple[int, int]:
        """(judged, all) over Gym-Anything's criteria in this verdict."""
        ga = [ok for k, (ok, _) in self.criteria.items() if k not in self.extra]
        return sum(ok is not None for ok in ga), len(ga)

    def summary(self) -> str:
        j, n = self.judged
        return f"{j} of {n} Gym-Anything criteria judged" + (
            f"; not judged: {', '.join(self.not_judged())}" if j < n else "")


def ledger_closed(water_rel: float, species_rel: float = 0.0, species_kg: float = 0.0) -> bool:
    return water_rel < LEDGER_REL and (species_rel < LEDGER_REL or abs(species_kg) < SPECIES_ABS_KG)


def finish(v: Verdict, value: float, verifiable: bool) -> Verdict:
    """Fill ``v``'s progress and score from the task's progress ``value`` (``task_set/scoring.py:126-139``)."""
    if not verifiable:
        v.progress, v.score = None, None
        return v
    v.progress = clip(value)
    v.score = 1.0 if v.success else FAIL_SCALE * v.progress
    return v


__all__ = ["FAIL_SCALE", "FLOOR_WATER_ML", "HANDS_NOT_JUDGED", "OUTLET_OFF_GPM", "Verdict", "clip", "excess",
           "finish", "finite", "hands_gate", "ledger_closed", "partly", "reduction"]
