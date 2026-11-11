# SPDX-License-Identifier: Apache-2.0
"""GA's four welding tasks on Alloy, scripted, judged by GA's own evaluator (SCR-WP8-14; ``weld_ga_criteria``).

Each task is a start state and a sequence of torch passes (``weld_ga.Pass``) at GA's operating point; ``run_task`` runs
it on ``GridThermal``, reads GA's sections (``weld_ga_measures``) on the start state and the final weld, and judges
them:
- fillet-2f: the tacked start (GA's fixture tacks, ``weld_ga.TACK_PASSES``), then GA's aim_tacks pass along the joint.
- set-up-and-weld: the same start with the source preset to 6.4 m/min and 20 V; the operator sets 12.7 m/min and
  29 V (GA's tolerances ±0.3 m/min, ±0.5 V) and welds the joint as fillet-2f. The scripted operator sets them exactly,
  so the settings check passes by construction (``settings``).
- half-fillet: the tacked start, then aim_tacks from x = 1 mm to the middle (75 mm); GA's half-fillet judgement.
- tack-and-weld: the loose start; the operator tacks as GA's fixture does, then welds aim_tacks; fillet-2f's checks on
  the final weld, and the assembly (bonded, square, placed, released), which holds by construction: the bracket is a
  fixed part of the grid (heat v3's limit; GA's FR3 holds it until it is tacked).

Score as GA: 1 on success, else 0.9 x progress (net of the start state's own sound metal).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alloy.tasks import weld_ga as wg
from alloy.tasks import weld_ga_criteria as gc
from alloy.tasks.weld_ga_measures import measure


@dataclass(frozen=True)
class Task:
    name: str
    passes: tuple[wg.Pass, ...]
    judge: str                                         # "fillet", "half" or "tack_and_weld"
    start: int = len(wg.TACK_PASSES)                   # passes that make the start state (read after them)
    by_construction: dict[str, str] = field(default_factory=dict)


_AIMED = wg.Pass(aim_zones=wg.AIM_TACKS)               # GA's aim_tacks: the root, 1 mm out over the tacks' starts
TASKS = {t.name: t for t in (
    Task("fillet-2f", (*wg.TACK_PASSES, _AIMED), "fillet"),
    Task("set-up-and-weld", (*wg.TACK_PASSES, _AIMED), "fillet",
         by_construction={"settings": "the scripted operator sets 12.7 m/min and 29.0 V exactly"}),
    Task("half-fillet", (*wg.TACK_PASSES, wg.Pass(end=0.075, aim_zones=wg.AIM_TACKS)), "half"),
    Task("tack-and-weld", (*wg.TACK_PASSES, _AIMED), "tack_and_weld", start=0,
         by_construction={"assembly": "the bracket is a fixed part of the grid: bonded, square, placed, released"}),
)}


def judge(task: Task, rows: list[dict], start_rows: list[dict] | None) -> dict:
    """GA's judgement of the final rows for ``task`` (start_rows: the start state's sections)."""
    if task.judge == "half":
        return gc.judge_half_fillet(rows, start_rows)
    if task.judge == "tack_and_weld":
        return gc.judge_tack_and_weld(rows)
    return gc.judge_fillet(rows, start_rows)


def run_task(name: str, device: str = "cuda:0", config: dict | None = None, log=print) -> dict:
    """Run task ``name``; returns GA's judgement {"success", "score", "progress", "reward", "checks"} plus "task",
    "reads" (the plug-in's books), "by_construction" and "rows" (the final sections)."""
    task = TASKS[name]
    start: dict = {}

    def after(i: int, grid) -> None:
        if task.start and i == task.start - 1:
            start["rows"] = measure(grid)

    grid = wg.run(task.passes, device=device, config=config, log=log, after_pass=after)
    rows = measure(grid)
    out = judge(task, rows, start.get("rows"))
    reads = {k: float(v[0]) for k, v in grid.read().items() if np.ndim(v) == 1}
    return {"task": name, **out, "reads": reads, "by_construction": dict(task.by_construction), "rows": rows,
            "start_rows": start.get("rows")}


__all__ = ["TASKS", "Task", "judge", "run_task"]
