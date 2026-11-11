# SPDX-License-Identifier: Apache-2.0
"""drilling_timber_bolt: Gym-Anything's timber-bolt, scored by its v1 checks. A 14.3 mm (9/16 in) hole square through
two 38 mm Douglas-fir boards on an 18 mm plywood backer, then a 1/2 x 4 in carriage bolt pushed through by hand.

Scene: the boards (static, 140 x 300 mm; the bottom one 2.5 mm off the top one, as the source sets it) and the backer.
The drill's table has no fir or plywood row, so both are cut as the ``pine_southern`` softwood row (an assumption),
with the fir and marine plywood cards' densities. A hand drill held by its scripted head: a 14.3 mm, 118 deg bit (the
source's brad point is not modelled) at 1500 rpm and 0.5 mm/rev to 85 mm (the point clears the bottom board at 80.3
mm). The bolt (12.7 mm shank, 101.6 mm under its 31 mm head) is a listed pin; a hand (``PushHead``) pushes it at 40 N
into the hole it finds nearest the mark.

Checks (the source's ``_timber_bolt``): through hole in the top board; entry within 2 mm of the mark; square to the
faces (axis within 1.19 deg: atan of the 1/16 in clearance over 3 in); through hole in the bottom board; the carriage
bolt through both boards (as in the source, the bolt's shank in a hole of each board: a bolt in the bottom board's
blind hole passes it); pushed in with less than 50 N (the hand's push in that world). Unscored: "no charred wall
(< 200 C)" (the drill has no heat model), and whether the bolt stays in the boards once the hand lets go (no gravity).

Worlds (``VARIANTS``, the last repeats): 0 as asked; 1 tilted 1.5 deg (fails square); 2 3 mm off the mark (fails the
entry; the bolt still passes); 3 fed 30 mm and the bolt pushed at 60 N (neither board through; the bolt stops in
the top board's blind hole; fails the push).

Run: ``python -m alloy.tasks.drilling_timber_bolt [--worlds W] [--device cuda:0]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from alloy.tasks.drilling_common import (NO_GRAVITY, RPM, Check, Insert, Op, Plan, VariantTask, Verdict, add_part,
                                         add_pin, add_spindle, bit_tip, density, mark_point, nearest)
from alloy.tasks.drilling_script import Episode, cli, pin_layers, push_of, run_task

BOARD_HALF = (0.15, 0.07, 0.019)        # m: 300 x 140 x 38 mm
BACKER_HALF = (0.2, 0.1, 0.009)
SHIFT = 0.0025                          # m: the bottom board's offset
TOP = 0.2
MARK = (0.0, 0.0)
BIT = 0.0143
AXIS_TOL = math.degrees(math.atan((9.0 / 16.0 - 0.5) * 0.0254 / 0.0762))      # 1.19 deg
PUSH = 40.0
BOLT = {"body": "carriage_bolt", "tip": (0.0, 0.0, -0.1016), "radius": 0.00635, "head": 0.0155, "stiffness": 5.0e4}


def _plan(push: float = PUSH, **kw: Any) -> Plan:
    op = Op(MARK, BIT, kw.pop("depth", 0.085), 0.5e-3, 1500.0 * RPM, settle=0.3, **kw)
    return Plan((op,), inserts=(Insert("carriage_bolt", MARK, push),))


VARIANTS: tuple[tuple[str, Plan], ...] = (
    ("as asked", _plan()),
    ("tilted 1.5 deg", _plan(tilt=1.5)),
    ("3 mm off the mark", _plan(offset=(0.003, 0.0))),
    ("fed 30 mm, pushed at 60 N", _plan(depth=0.030, push=60.0)),
)


@dataclass(frozen=True)
class TimberBolt(VariantTask):
    """The task: scene, scripts per world (``VARIANTS``) and the checks."""

    variants: tuple[tuple[str, Plan], ...] = VARIANTS
    drill: dict[str, Any] = field(default_factory=lambda: {
        "tool": "spindle", "workpiece": "board1", "tip": bit_tip(0.09), "diameter": BIT, "max_holes": 2,
        "layers": tuple({"workpiece": name, "work": "pine_southern", "density": density(card)} for name, card in (
            ("board1", "wood_douglas_fir"), ("board0", "wood_douglas_fir"), ("backer", "plywood_marine"))),
        "pins": (BOLT,)})
    head: dict[str, Any] = field(default_factory=dict)
    top: float = TOP
    push_seconds: float = 1.5
    d_push: float = 400.0
    limit: float = 60.0

    def build(self, b: Any) -> None:
        t = 2.0 * BOARD_HALF[2]
        add_part(b, "board1", (0.0, 0.0, TOP - 0.5 * t), BOARD_HALF)
        add_part(b, "board0", (SHIFT, 0.0, TOP - 1.5 * t), BOARD_HALF)
        add_part(b, "backer", (0.0, 0.0, TOP - 2.0 * t - BACKER_HALF[2]), BACKER_HALF)
        add_spindle(b)
        add_pin(b, "carriage_bolt", 0)

    def score(self, ep: Episode, w: int) -> Verdict:
        out: list[Check] = []
        hit = nearest(ep.drill, w, 0, ep.body_q, mark_point(MARK, TOP))
        if hit is None or not hit[0].through:
            out.append(Check("through hole in the top board", False))
        else:
            h, e = hit
            out += [Check("through hole in the top board", True),
                    Check("entry within 2 mm of the mark", e <= 0.002, round(e * 1e3, 3), "<= 2 mm"),
                    Check(f"square to the faces (axis within {AXIS_TOL:.2f} deg of the face normal)",
                          h.tilt_deg <= AXIS_TOL, round(h.tilt_deg, 3), f"<= {AXIS_TOL:.2f} deg")]
        b0 = [h for h in ep.drill.holes(w, 1) if h.through]
        out.append(Check("through hole in the bottom board", bool(b0), len(b0)))
        inside, push = pin_layers(ep, w, "carriage_bolt"), push_of(ep, w, "carriage_bolt")
        out += [Check("carriage bolt through both boards", {0, 1} <= set(inside), inside),
                Check("bolt pushed in with less than 50 N", push < 50.0, push, "< 50 N", "constraint")]
        unscored = ("no charred wall (< 200 C): no heat model",
                    f"the bolt still through both boards after the hand lets go: {NO_GRAVITY}")
        return Verdict(out, unscored, detail={"variant": self.variant(w)})


def run(task: TimberBolt | None = None, *, worlds: int = len(VARIANTS), device: str = "cpu"
        ) -> tuple[list[Verdict], Episode]:
    return run_task(task or TimberBolt(), worlds, device)


def main(argv: list[str] | None = None) -> int:
    return cli(TimberBolt(), __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
