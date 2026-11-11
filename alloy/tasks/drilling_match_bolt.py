# SPDX-License-Identifier: Apache-2.0
"""drilling_match_bolt: Gym-Anything's match-bolt, scored by its v1 checks. Two 9 mm holes 60 mm apart through a
50 x 50 x 5 mm steel angle clamped on a 50 x 6 mm flat bar, drilled through both at once, then two M8 bolts pushed
through by hand.

Scene: the angle's drilled leg and the flat bar (static: clamped in the vise; the angle's upright leg is not drilled
and not modelled), both cut by the ``steel_200bhn`` row. The press's spindle: a 9 mm drill
at 1000 rpm and 0.2 mm/rev to 21 mm (the point clears the bar's back at 13.7 mm; the cut trails the quill by about
6 mm at its 3 kN thrust) at the marks y = -30 and +30 mm. The
bolts (8 mm shank, 30 mm under a 13 mm head, 0.5 mm chamfer) are listed pins; hands (``PushHead``) push each at 40 N
into the hole they find nearest its mark.

Checks (the source's ``_match_bolt``): per angle hole, through at 9.0 mm, entry within 0.5 mm of its mark, axis within
1 deg, diameter 9.00-9.25 mm; two through holes in the flat bar; hole spacing within 0.5 mm of 60 mm; each bolt
through both parts, pushed in with less than 50 N (its hand's push in that world). Unscored: neither part spun (< 2
deg; the parts are static), and whether the bolts stay in both parts once the hands let go (no gravity).

Worlds (``VARIANTS``, the last repeats): 0 as asked; 1 hole 1 at y = +30.8 mm (fails its entry and the spacing); 2
hole 0 tilted 2 deg (fails its axis); 3 a 7.5 mm bit (fails the holes; the bolts stop on the angle); 4 a 9.5 mm bit
fed 16 mm, the bolts pushed at 60 N (fails the diameters, the bar's holes, the bolts and the pushes).

Run: ``python -m alloy.tasks.drilling_match_bolt [--worlds W] [--device cuda:0]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.tasks.drilling_common import (NO_GRAVITY, RPM, STATIC, Check, Insert, Op, Plan, VariantTask, Verdict,
                                         add_part, add_pin, add_spindle, bit_tip, density, mark_point, nearest,
                                         through_checks)
from alloy.tasks.drilling_script import Episode, cli, pin_layers, push_of, run_task

ANGLE_HALF = (0.025, 0.05, 0.0025)      # m: the angle's drilled leg, 50 x 100 x 5 mm
BAR_HALF = (0.025, 0.075, 0.003)        # m: 50 x 150 x 6 mm
TOP = 0.1
MARKS = ((0.0, -0.03), (0.0, 0.03))
BIT = 0.009
PUSH = 40.0
BOLTS = ("m8_bolt0", "m8_bolt1")


def _bolt(name: str) -> dict[str, Any]:
    return {"body": name, "tip": (0.0, 0.0, -0.03), "radius": 0.004, "head": 0.0065, "chamfer": 0.0005,
            "stiffness": 5.0e4}


def _plan(bit: float = BIT, at: dict[int, dict[str, Any]] | None = None, marks=MARKS, depth: float = 0.021,
          push: float = PUSH) -> Plan:
    ops = tuple(Op(m, bit, depth, 0.2e-3, 1000.0 * RPM, settle=0.3, **(at or {}).get(k, {}))
                for k, m in enumerate(marks))
    return Plan(ops, inserts=tuple(Insert(b, m, push) for b, m in zip(BOLTS, MARKS, strict=True)))


VARIANTS: tuple[tuple[str, Plan], ...] = (
    ("as asked", _plan()),
    ("hole 1 at +30.8 mm", _plan(marks=(MARKS[0], (0.0, 0.0308)))),
    ("hole 0 tilted 2 deg", _plan(at={0: {"tilt": 2.0}})),
    ("a 7.5 mm bit", _plan(bit=0.0075)),
    ("a 9.5 mm bit fed 16 mm, pushed at 60 N", _plan(bit=0.0095, depth=0.016, push=60.0)),
)


@dataclass(frozen=True)
class MatchBolt(VariantTask):
    """The task: scene, scripts per world (``VARIANTS``) and the checks."""

    variants: tuple[tuple[str, Plan], ...] = VARIANTS
    drill: dict[str, Any] = field(default_factory=lambda: {
        "tool": "spindle", "workpiece": "angle_bracket", "tip": bit_tip(0.024), "diameter": BIT, "max_holes": 4,
        "layers": tuple({"workpiece": name, "work": "steel_200bhn", "density": density("steel_carbon")}
                        for name in ("angle_bracket", "flat_bar")),
        "pins": tuple(_bolt(b) for b in BOLTS)})
    head: dict[str, Any] = field(default_factory=dict)
    top: float = TOP
    push_seconds: float = 1.0
    d_push: float = 400.0
    limit: float = 60.0

    def build(self, b: Any) -> None:
        add_part(b, "angle_bracket", (0.0, 0.0, TOP - ANGLE_HALF[2]), ANGLE_HALF)
        add_part(b, "flat_bar", (0.0, 0.0, TOP - 2.0 * ANGLE_HALF[2] - BAR_HALF[2]), BAR_HALF)
        add_spindle(b)
        for k, name in enumerate(BOLTS):
            add_pin(b, name, k)

    def score(self, ep: Episode, w: int) -> Verdict:
        out: list[Check] = []
        holes = []
        for k, m in enumerate(MARKS):
            checks, h = through_checks(f"angle hole {k}", nearest(ep.drill, w, 0, ep.body_q, mark_point(m, TOP)),
                                       BIT, 0.0005, 0.00925)
            out += checks
            holes.append(h)
        bar = [h for h in ep.drill.holes(w, 1) if h.through]
        out.append(Check("two through holes in the flat bar", len(bar) >= 2, len(bar), ">= 2"))
        if all(holes):
            face = ep.drill.face(w, 0, ep.body_q)
            sp = float(np.linalg.norm(holes[0].in_world(face)[0] - holes[1].in_world(face)[0]))
            out.append(Check("hole spacing within 0.5 mm of 60 mm", abs(sp - 0.060) <= 0.0005, round(sp * 1e3, 3),
                             "60 +- 0.5 mm"))
        for k, name in enumerate(BOLTS):
            inside, push = pin_layers(ep, w, name), push_of(ep, w, name)
            out += [Check(f"bolt {k} passes through both parts", {0, 1} <= set(inside), inside),
                    Check(f"bolt {k} pushed in with less than 50 N", push < 50.0, push, "< 50 N", "constraint")]
        unscored = (f"angle_bracket not spun (< 2 deg): {STATIC}", f"flat_bar not spun (< 2 deg): {STATIC}",
                    f"the bolts still through both parts after the hands let go: {NO_GRAVITY}")
        return Verdict(out, unscored, detail={"variant": self.variant(w)})


def run(task: MatchBolt | None = None, *, worlds: int = len(VARIANTS), device: str = "cpu"
        ) -> tuple[list[Verdict], Episode]:
    return run_task(task or MatchBolt(), worlds, device)


def main(argv: list[str] | None = None) -> int:
    return cli(MatchBolt(), __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
