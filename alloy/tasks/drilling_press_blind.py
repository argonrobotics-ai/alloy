# SPDX-License-Identifier: Apache-2.0
"""drilling_press_blind: Gym-Anything's press-blind, scored by its v1 checks. A 6.8 mm tap-drill hole 15.0-15.5 mm deep
at full diameter in a 50 x 50 x 25 mm aluminium block held in the press vise.

Scene: the block (static: the vise holds it) cut by the ``aluminium`` row (the 6061 card's density), the press's
spindle with a 6.8 mm, 118 deg drill at 2400 rpm (the source's setting), 0.1 mm/rev. The ``Driller`` feeds the quill
to its stop 17.44 mm past the face (15.4 mm at full diameter plus the 2.04 mm point) and dwells 2 s there while the
cut closes on the stop (it trails the quill by thrust / kp + thrust / k, and slows as the thrust falls: it ends
0.18 mm short, at 15.22 mm), then retracts and stops.

Checks (the source's ``_press_blind`` and ``spindle_stopped_quill_up``): full-diameter depth 15.0-15.5 mm; not through;
entry within 0.5 mm of the mark; axis within 1 deg; spindle stopped (< 1 rpm); quill within 2 mm of the top; bit
clear of the work.

Worlds (``VARIANTS``, the last repeats): 0 as asked; 1 fed 1 mm deeper (fails the depth); 2 tilted 2 deg (fails the
axis); 3 fed through the block (fails the depth and not-through); 4 0.8 mm off the mark (fails the entry); 5 left
spinning in the hole (fails the spindle, quill and bit checks).

Run: ``python -m alloy.tasks.drilling_press_blind [--worlds W] [--device cuda:0]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from alloy.tasks.drilling_common import (RPM, Check, Op, Plan, VariantTask, Verdict, add_part, add_spindle, bit_tip,
                                         density, mark_point, nearest)
from alloy.tasks.drilling_script import Episode, cli, run_task, spindle_checks

BLOCK_HALF = (0.025, 0.025, 0.0125)     # m: 50 x 50 x 25 mm
TOP = 0.1
MARK = (0.0, 0.0)
BIT = 0.0068
POINT = 0.5 * BIT / math.tan(math.radians(59.0))
DEPTH = 0.0154 + POINT                  # m: the stop, 15.4 mm at full diameter (the cut ends 0.18 mm short)
FEED, SPIN, DWELL = 0.1e-3, 2400.0 * RPM, 2.0
VARIANTS: tuple[tuple[str, Plan], ...] = (
    ("as asked", Plan((Op(MARK, BIT, DEPTH, FEED, SPIN, settle=DWELL),))),
    ("fed 1 mm deeper", Plan((Op(MARK, BIT, DEPTH + 0.001, FEED, SPIN, settle=DWELL),))),
    ("tilted 2 deg", Plan((Op(MARK, BIT, DEPTH, FEED, SPIN, tilt=2.0, settle=DWELL),))),
    ("fed through", Plan((Op(MARK, BIT, 0.032, FEED, SPIN),))),
    ("0.8 mm off the mark", Plan((Op(MARK, BIT, DEPTH, FEED, SPIN, offset=(0.0008, 0.0), settle=DWELL),))),
    ("left spinning in the hole", Plan((Op(MARK, BIT, DEPTH, FEED, SPIN, settle=DWELL),), park=False)),
)


@dataclass(frozen=True)
class PressBlind(VariantTask):
    """The task: scene, scripts per world (``VARIANTS``) and the checks."""

    variants: tuple[tuple[str, Plan], ...] = VARIANTS
    drill: dict[str, Any] = field(default_factory=lambda: {
        "tool": "spindle", "workpiece": "block", "tip": bit_tip(0.034), "diameter": BIT, "max_holes": 2,
        "layers": ({"workpiece": "block", "work": "aluminium", "density": density("aluminium_6061_t6")},)})
    head: dict[str, Any] = field(default_factory=dict)
    top: float = TOP
    push_seconds: float = 0.0
    d_push: float = 400.0
    limit: float = 30.0

    def build(self, b: Any) -> None:
        add_part(b, "block", (0.0, 0.0, TOP - BLOCK_HALF[2]), BLOCK_HALF)
        add_spindle(b)

    def score(self, ep: Episode, w: int) -> Verdict:
        name = self.variant(w)
        hit = nearest(ep.drill, w, 0, ep.body_q, mark_point(MARK, TOP))
        if hit is None:
            out = [Check("full-diameter depth 15.0-15.5 mm", False, None, "15.0-15.5 mm")]
            return Verdict(out + spindle_checks(ep, w, TOP), detail={"variant": name})
        h, e = hit
        fd = h.full_depth(BIT)
        out = [Check("full-diameter depth 15.0-15.5 mm", 0.0150 <= fd <= 0.0155, round(fd * 1e3, 3), "15.0-15.5 mm"),
               Check("not through", not h.through, h.through),
               Check("entry within 0.5 mm of the mark", e <= 0.0005, round(e * 1e3, 3), "<= 0.5 mm"),
               Check("axis within 1 deg", h.tilt_deg <= 1.0, round(h.tilt_deg, 3), "<= 1 deg")]
        return Verdict(out + spindle_checks(ep, w, TOP), detail={"variant": name})


def run(task: PressBlind | None = None, *, worlds: int = len(VARIANTS), device: str = "cpu"
        ) -> tuple[list[Verdict], Episode]:
    return run_task(task or PressBlind(), worlds, device)


def main(argv: list[str] | None = None) -> int:
    return cli(PressBlind(), __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
