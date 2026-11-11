# SPDX-License-Identifier: Apache-2.0
"""What a concrete-finishing port judges of Gym-Anything's task, said in its verdict (audit AB-gym-7, 2026-11-09).

The rule: each Gym-Anything criterion is judged from the simulation's state or listed as not judged (``passed`` None:
never a literal True, never a default the port cannot move); success counts the judged criteria only and the verdict
says "k of n"; a scene that differs from Gym-Anything's says so in the verdict and in the module docstring.

A verdict (``verdict_dict``) is one world's JSON:
- ``criteria``: per Gym-Anything criterion ``value``, ``threshold``, ``unit``, ``passed`` (True, False or None) and
  ``measured`` (True, False or "partly"), with a ``note`` when the port does not judge it as Gym-Anything does. A
  criterion judged in part (``partly``) is False when its judged part fails and None otherwise: a pass of the part
  alone is not counted (``judged_part`` keeps it).
- ``alloy_gates``: Alloy's own gates (closed books, a ride rule). They are not Gym-Anything criteria, they are
  stricter, and they gate ``success`` too.
- ``success``: every judged criterion passes (at least one is judged) and every gate of Alloy's holds. ``score`` is
  the metrics' own (1 on their success, else 0.95 x progress); where theirs is 1 and ``success`` is False it is None
  (``score_before_judging`` keeps it): a failed verdict never reads score 1.
- ``judged`` [judged, all], ``not_judged`` (names), ``scene_differs`` and ``summary`` (all of it in a sentence).

The ports' own lists (what each does not judge and how its scene differs) are in ``concrete_ports``. Gym-Anything
file:line references are to ``benchmarks/roboworld/environments/concrete_finishing`` (read-only in [internal note]).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Port:
    """What one port judges of its Gym-Anything task: criteria it cannot judge (name -> why, and what Gym-Anything
    judges), criteria it judges in part (name -> which part), criteria judged by another measure (name -> which), and
    how its scene differs. Each ``scene_differs`` entry starts with a short head before its colon."""

    task: str
    not_judged: Mapping[str, str] = field(default_factory=dict)
    partly: Mapping[str, str] = field(default_factory=dict)
    notes: Mapping[str, str] = field(default_factory=dict)
    scene_differs: tuple[str, ...] = ()


def partly(*parts: bool | None) -> bool | None:
    """A criterion from its parts: False if a judged part fails, None if a part is not judged, else True."""
    if any(p is False for p in parts):
        return False
    return None if any(p is None for p in parts) else True


def criterion(c: Any, port: Port) -> dict[str, Any]:
    """One criterion (``concrete_metrics.Criterion``) as JSON, by the port's lists."""
    d: dict[str, Any] = {"value": c.value, "threshold": c.threshold, "unit": c.unit, "passed": bool(c.passed),
                         "measured": True}
    if c.name in port.not_judged:
        d.update(passed=None, measured=False, note=port.not_judged[c.name])
    elif c.name in port.partly:
        d.update(passed=partly(bool(c.passed), None), measured="partly", judged_part=bool(c.passed),
                 note=port.partly[c.name])
    elif c.name in port.notes:
        d["note"] = port.notes[c.name]
    return d


def alloy_gates(note: str, **passed: bool) -> dict[str, dict[str, Any]]:
    """Alloy's own gates for ``verdict_dict``: name -> passed, each with the same ``note``."""
    return {k: {"passed": bool(ok), "note": note} for k, ok in passed.items()}


def judged_pass(d: Mapping[str, Any]) -> bool:
    """Every judged Gym-Anything criterion of verdict ``d`` passes, and at least one is judged."""
    judged = [c["passed"] for c in d["criteria"].values() if c["passed"] is not None]
    return bool(judged) and all(judged)


def _heads(differs: Sequence[str]) -> str:
    return ", ".join(x.split(":", 1)[0] for x in differs)


def summary(d: Mapping[str, Any]) -> str:
    """The verdict in a sentence: "k of n Gym-Anything criteria judged", what is not, Alloy's gates, the scene."""
    crit = d["criteria"]
    j, n = d["judged"]
    out = f"{j} of {n} Gym-Anything criteria judged"
    if d["not_judged"]:
        out += "; not judged: " + ", ".join(k + (" (judged in part)" if crit[k].get("measured") == "partly" else "")
                                            for k in d["not_judged"])
    other = [k for k, c in crit.items() if c["passed"] is not None and "note" in c]
    if other:
        out += "; judged by another measure than Gym-Anything's: " + ", ".join(other)
    if d.get("alloy_gates"):
        out += "; Alloy's own gates, not Gym-Anything criteria: " + ", ".join(d["alloy_gates"])
    if d["scene_differs"]:
        out += "; the scene differs from Gym-Anything's in: " + _heads(d["scene_differs"])
    return out


def counted(d: dict[str, Any], port: Port, differs: Sequence[str] = ()) -> dict[str, Any]:
    """Fill verdict ``d``'s ``judged``, ``not_judged``, ``scene_differs`` and ``summary`` from its criteria as they
    stand (a task that re-judges a criterion after ``verdict_dict``, as pad-trowel's forms proxy, calls it again).
    ``differs``: this run's own differences (a window, a coarser grid), after the port's."""
    crit = d["criteria"]
    d["not_judged"] = [k for k, c in crit.items() if c["passed"] is None]
    d["judged"] = [len(crit) - len(d["not_judged"]), len(crit)]
    d["scene_differs"] = [*port.scene_differs, *differs]
    d["summary"] = summary(d)
    return d


def verdict_dict(v: Any, port: Port, *, gates: Mapping[str, dict[str, Any]] | None = None,
                 differs: Sequence[str] = ()) -> dict[str, Any]:
    """One world's verdict ``v`` (its ``criteria`` and ``score``) as JSON by the rule above. ``gates``: Alloy's own
    (``alloy_gates``); ``differs``: this run's own scene differences."""
    d: dict[str, Any] = {"task": port.task, "success": False, "score": v.score,
                         "criteria": {c.name: criterion(c, port) for c in v.criteria}}
    if gates:
        d["alloy_gates"] = dict(gates)
    d["success"] = judged_pass(d) and all(g["passed"] for g in (gates or {}).values())
    if not d["success"] and v.score is not None and v.score >= 1.0:   # the metrics' own pass (defaults in) is not ours
        d["score_before_judging"], d["score"] = v.score, None
    return counted(d, port, differs)


def grid_differs(cell: float, theirs: float, reads: str) -> tuple[str, ...]:
    """A run on cells coarser than Gym-Anything's says so; on theirs it says nothing."""
    if cell <= theirs * (1.0 + 1e-9):
        return ()
    return (f"grid: {cell * 1e3:g} mm cells, Gym-Anything's are {theirs * 1e3:g} mm: {reads} are read at "
            f"{cell * 1e3:g} mm, where thin smears, spikes and narrow lows average away (lenient)",)


__all__ = ["Port", "alloy_gates", "counted", "criterion", "grid_differs", "judged_pass", "partly", "summary",
           "verdict_dict"]
