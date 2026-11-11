# SPDX-License-Identifier: Apache-2.0
"""drilling_panel_holes: Gym-Anything's panel-holes, scored by its v1 checks. Four 6 mm through holes in a 1.5 mm steel
cover strapped on an 18 mm plywood backer, each started with a 3 mm pilot.

Scene: the cover (static, ``steel_200bhn`` row, the carbon-steel card's density) on the backer (static; plywood has
no row in the drill's table, so it is cut as the ``pine_southern`` softwood row, an assumption, with the marine
plywood card's density). A hand drill held square by its scripted head (``FeedHead``): the 3 mm pilot at 3000 rpm and
0.06 mm/rev to 3.5 mm, then the 6 mm drill at 1500 rpm and 0.1 mm/rev to 5.5 mm (the points clear the cover's back
at 2.4 and 3.3 mm; the cut trails the head by thrust / kp + thrust / k; both run into the backer).

Checks (the source's ``_panel_holes``, per mark): through hole at the mark at 6.0 mm; entry within 1.0 mm of the
mark; axis within 1 deg of the face normal. Unscored: roundness (d + 2 wander) / d at most 1.10 (the hole is cut along
the bit's axis, which does not walk: the wander reads 0 in every world).

Worlds (``VARIANTS``, the last repeats): 0 as asked; 1 the 6 mm drill without pilots (passes: no check asks for the
pilot); 2 hole 2 tilted 2 deg (fails its axis); 3 hole 1 1.5 mm off its mark (fails its entry); 4 hole 3 not drilled
(its nearest hole is another mark's: fails its entry) and hole 0's 6 mm drill fed 2 mm (through at 3 mm only).

Run: ``python -m alloy.tasks.drilling_panel_holes [--worlds W] [--device cuda:0]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alloy.tasks.drilling_common import (RPM, Check, Op, Plan, VariantTask, Verdict, add_part, add_spindle, bit_tip,
                                         density, mark_point, nearest, through_checks)
from alloy.tasks.drilling_script import Episode, cli, run_task

COVER_HALF = (0.1, 0.075, 0.00075)      # m: a 200 x 150 x 1.5 mm cover
BACKER_HALF = (0.12, 0.09, 0.009)       # m: 18 mm plywood
TOP = 0.1
MARKS = ((-0.06, -0.04), (0.06, -0.04), (0.06, 0.04), (-0.06, 0.04))
PILOT, BIT = 0.003, 0.006


def _pilot(mark: tuple[float, float], **kw: Any) -> Op:
    return Op(mark, PILOT, 0.0035, 0.06e-3, 3000.0 * RPM, settle=0.2, **kw)


def _drill(mark: tuple[float, float], depth: float = 0.0055, **kw: Any) -> Op:
    return Op(mark, BIT, depth, 0.1e-3, 1500.0 * RPM, settle=0.2, **kw)


def _ops(skip: int = -1, pilots: bool = True, at: dict[int, dict[str, Any]] | None = None,
         short: int = -1) -> tuple[Op, ...]:
    """Pilot and drill at each mark (``at[k]``: mark k's tilt or offset); ``skip`` leaves a mark undrilled, ``short``
    feeds a mark's 6 mm drill 2 mm only."""
    out: list[Op] = []
    for k, m in enumerate(MARKS):
        if k == skip:
            continue
        kw = (at or {}).get(k, {})
        out += ([_pilot(m, **kw)] if pilots else []) + [_drill(m, 0.002 if k == short else 0.0055, **kw)]
    return tuple(out)


VARIANTS: tuple[tuple[str, Plan], ...] = (
    ("as asked", Plan(_ops())),
    ("no pilots", Plan(_ops(pilots=False))),
    ("hole 2 tilted 2 deg", Plan(_ops(at={2: {"tilt": 2.0}}))),
    ("hole 1 1.5 mm off", Plan(_ops(at={1: {"offset": (0.0, 0.0015)}}))),
    ("hole 3 not drilled, hole 0 short", Plan(_ops(skip=3, short=0))),
)


@dataclass(frozen=True)
class PanelHoles(VariantTask):
    """The task: scene, scripts per world (``VARIANTS``) and the checks."""

    variants: tuple[tuple[str, Plan], ...] = VARIANTS
    drill: dict[str, Any] = field(default_factory=lambda: {
        "tool": "spindle", "workpiece": "cover_panel", "tip": bit_tip(0.008), "diameter": BIT, "max_holes": 6,
        "layers": ({"workpiece": "cover_panel", "work": "steel_200bhn", "density": density("steel_carbon")},
                   {"workpiece": "backer", "work": "pine_southern", "density": density("plywood_marine")})})
    head: dict[str, Any] = field(default_factory=dict)
    top: float = TOP
    push_seconds: float = 0.0
    d_push: float = 400.0
    limit: float = 60.0

    def build(self, b: Any) -> None:
        add_part(b, "cover_panel", (0.0, 0.0, TOP - COVER_HALF[2]), COVER_HALF)
        add_part(b, "backer", (0.0, 0.0, TOP - 2.0 * COVER_HALF[2] - BACKER_HALF[2]), BACKER_HALF)
        add_spindle(b)

    def score(self, ep: Episode, w: int) -> Verdict:
        out: list[Check] = []
        for k, m in enumerate(MARKS):
            hit = nearest(ep.drill, w, 0, ep.body_q, mark_point(m, TOP))
            checks, h = through_checks(f"hole {k}", hit, BIT, 0.001)
            out += checks
        unscored = tuple(f"hole {k}: roundness at most 1.10: the bit does not walk (no wander)" for k in range(4))
        return Verdict(out, unscored, detail={"variant": self.variant(w)})


def run(task: PanelHoles | None = None, *, worlds: int = len(VARIANTS), device: str = "cpu"
        ) -> tuple[list[Verdict], Episode]:
    return run_task(task or PanelHoles(), worlds, device)


def main(argv: list[str] | None = None) -> int:
    return cli(PanelHoles(), __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
