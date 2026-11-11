# SPDX-License-Identifier: Apache-2.0
"""drilling_press_drill: Gym-Anything's press-drill, scored by its v1 checks. A 9 mm through hole in a 50 x 150 x 6 mm
steel bar held in the press vise, at its punched mark.

Scene: the bar (static: the vise holds it) over open space, the press's spindle
(``FeedHead``) with a 9 mm, 118 deg HSS drill at 1000 rpm (the source's press setting) and 0.15 mm/rev, the bar cut
by the ``steel_200bhn`` row. The ``Driller`` feeds the quill 14 mm past the face (the point clears the back at 8.7 mm;
the cut trails the quill by its lag thrust / kp and the bit's press thrust / k, about 5 mm at 2.4 kN), then retracts
the quill and stops the spindle.

Checks (the source's ``_press_drill`` and ``spindle_stopped_quill_up``): through hole at the mark at 9.0 mm; entry
within 0.5 mm of the mark; axis within 1 deg of the face normal; diameter 9.00-9.25 mm; spindle stopped (< 1 rpm);
quill within 2 mm of the top; bit clear of the work. Unscored: the bar moved less than 2 mm and turned less than
2 deg (the bar is static).

Worlds (``VARIANTS``, the last repeats): 0 as asked; 1 0.3 mm off the mark (passes); 2 0.8 mm off (fails the entry);
3 tilted 2 deg (fails the axis); 4 an 8.5 mm bit (not through at 9 mm); 5 fed 5 mm only (not through); 6 left
spinning in the hole (fails the spindle, quill and bit checks); 7 a 9.5 mm bit (fails the diameter).

Run: ``python -m alloy.tasks.drilling_press_drill [--worlds W] [--device cuda:0]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alloy.tasks.drilling_common import (RPM, STATIC, Op, Plan, VariantTask, Verdict, add_part, add_spindle, bit_tip,
                                         density, mark_point, nearest, through_checks)
from alloy.tasks.drilling_script import Episode, cli, run_task, spindle_checks

BAR_HALF = (0.075, 0.025, 0.003)        # m: 150 x 50 x 6 mm
TOP = 0.1                               # m: the bar's top face height (on the vise jaws)
MARK = (0.0, 0.0)                       # m: the punched mark
BIT = 0.009
DEPTH = 0.014
OP = Op(MARK, BIT, DEPTH, feed=0.15e-3, spin=1000.0 * RPM, settle=0.3)
VARIANTS: tuple[tuple[str, Plan], ...] = (
    ("as asked", Plan((OP,))),
    ("0.3 mm off the mark", Plan((Op(MARK, BIT, DEPTH, OP.feed, OP.spin, offset=(0.0003, 0.0), settle=0.3),))),
    ("0.8 mm off the mark", Plan((Op(MARK, BIT, DEPTH, OP.feed, OP.spin, offset=(0.0, 0.0008), settle=0.3),))),
    ("tilted 2 deg", Plan((Op(MARK, BIT, DEPTH, OP.feed, OP.spin, tilt=2.0, settle=0.3),))),
    ("an 8.5 mm bit", Plan((Op(MARK, 0.0085, DEPTH, OP.feed, OP.spin, settle=0.3),))),
    ("fed 5 mm", Plan((Op(MARK, BIT, 0.005, OP.feed, OP.spin, settle=0.3),))),
    ("left spinning in the hole", Plan((OP,), park=False)),
    ("a 9.5 mm bit", Plan((Op(MARK, 0.0095, DEPTH, OP.feed, OP.spin, settle=0.3),))),
)


@dataclass(frozen=True)
class PressDrill(VariantTask):
    """The task: scene, scripts per world (``VARIANTS``) and the checks."""

    variants: tuple[tuple[str, Plan], ...] = VARIANTS
    drill: dict[str, Any] = field(default_factory=lambda: {
        "tool": "spindle", "workpiece": "bar", "tip": bit_tip(0.016), "diameter": BIT, "max_holes": 2,
        "layers": ({"workpiece": "bar", "work": "steel_200bhn", "density": density("steel_carbon")},)})
    head: dict[str, Any] = field(default_factory=dict)
    top: float = TOP
    push_seconds: float = 0.0
    d_push: float = 400.0
    limit: float = 30.0

    def build(self, b: Any) -> None:
        add_part(b, "bar", (0.0, 0.0, TOP - BAR_HALF[2]), BAR_HALF)
        add_spindle(b)

    def score(self, ep: Episode, w: int) -> Verdict:
        hit = nearest(ep.drill, w, 0, ep.body_q, mark_point(MARK, TOP))
        out, _ = through_checks("hole A", hit, BIT, 0.0005, 0.00925)
        unscored = (f"bar moved less than 2 mm: {STATIC}", f"bar turned less than 2 deg while drilling: {STATIC}")
        return Verdict(out + spindle_checks(ep, w, TOP), unscored, detail={"variant": self.variant(w)})


def run(task: PressDrill | None = None, *, worlds: int = len(VARIANTS), device: str = "cpu"
        ) -> tuple[list[Verdict], Episode]:
    return run_task(task or PressDrill(), worlds, device)


def main(argv: list[str] | None = None) -> int:
    return cli(PressDrill(), __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
