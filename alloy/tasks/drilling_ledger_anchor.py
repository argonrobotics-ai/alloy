# SPDX-License-Identifier: Apache-2.0
"""drilling_ledger_anchor: Gym-Anything's ledger-anchor, scored by its v1 checks. A 38 x 184 mm ledger fastened to a
200 mm concrete wall with four 1/2 in wedge anchors through it: each hole bored through the ledger and 108-125 mm
into the wall with a rotary hammer, the anchor driven until its washer bears on the ledger and its nut torqued to
68 N m.

Scene (gravity off, so the wall lies face up): the ledger (static, a Douglas-fir board cut by the ``pine_southern``
softwood row, an assumption: the table has no fir row) on the wall (static, the ``concrete_medium_hard`` hammer row,
the C25/30 card's density) with #4 bars at 300 mm each way: vertical bars under 25 mm of cover at x = -450, -150, 150,
450 mm, horizontal ones under 38.1 mm at y = -150, 150 mm (the far mat is past the 125 mm limit and left out). The
hammer drill: the source's DeWalt DCH273 (2.1 J per blow, EPTA 05/2009; 4600 blows/min; 1100 rpm) with a 12.7 mm
carbide bit, fed at the rate its blows cut the concrete, P / (e_v A) (the work row's e_v), 156.8 mm past the
ledger's face (115 mm at full diameter in the wall); a jam on a bar stops the hole. The anchors (Hilti KB-TZ2 1/2
in: 12.6 mm shank, 133 mm from washer to tip, a 24 mm washer, expansion from 93 mm of embed, h_nom 95 mm less the
source's 2 mm, gripping holes to 13.46 mm, the largest 1/2 in carbide bit: ANSI B212.15-1994's 0.520-0.530 in,
as reproduced in Relton's technical sheet, https://www.gamurdock.com/img/documents/Relton-technical.pdf) are listed
pins pushed at 60 N, then torqued.

Checks (the source's ``_ledger_anchor``, per mark k): ledger hole through within 3 mm of the mark; concrete axis
within 3 deg of the wall normal; depth 108-125 mm (full diameter); reinforcement damage at most 0.5 mm; anchor driven
to at least 93 mm with its washer at most 1 mm off the ledger; anchor straight (within 3 deg); nut torqued to T_inst
(61.2-115.6 N m: 0.9-1.7 x 68); expansion engaged. Then: no hole away from the marks. Unscored: the anchor did not
spin (no nut is turned), the ledger fastened where it was set and still level (it is static), the bit not chipped on
steel (no chipping model), and the anchors and the ledger once the hands and brackets let go (no gravity). The bar
damage is the world's (the drill books it per world, not per hole).

Worlds (``VARIANTS``, the last repeats): 0 as asked; 1 hole 0 drilled onto the bar at x = -450 mm (jams; its anchor
stops short); 2 hole 0 tilted 4 deg (fails its axis; its anchor follows it); 3 a 14.3 mm bit (holes too wide: no
expansion grips); 4 nut 0 torqued to 50 N m (fails its torque); 5 hole 0 18 mm off its mark toward that bar, pushed on
after the jam and 8 s past its ramp (the bit's lip cuts the bar: fails its mark, depth, bar damage and anchor). The
bars are the wall's, the same in every world: mark 0 lies 30 mm from one (17.3 mm clear of the bit), and world 1
drills on it, 30 mm off the mark (it fails its mark, and the hole is away from every mark).
``marks`` drills and scores the first n marks (the CPU test runs one).

Run: ``python -m alloy.tasks.drilling_ledger_anchor [--worlds W] [--device cuda:0]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.plugins.holes.work import PERCUSSIVE, energy_per_volume
from alloy.plugins.npgeom import quat_rotate
from alloy.tasks.drilling_common import (CONTROL_DT, MARK_RADIUS, NO_GRAVITY, RPM, STATIC, Check, Insert, Op, Plan,
                                         VariantTask, Verdict, add_part, add_pin, add_spindle, angle_deg, bit_tip,
                                         density, in_plane, mark_point, nearest)
from alloy.tasks.drilling_script import Episode, cli, run_task

WALL_HALF = (0.6, 0.3, 0.1)              # m: 1.2 x 0.6 m of a 200 mm wall
LEDGER_HALF = (0.5, 0.092, 0.019)        # m: 1 m of 38 x 184 mm
TOP = 0.3                                # m: the ledger's face
WALL = TOP - 2.0 * LEDGER_HALF[2]        # m: the wall's face
MARKS = ((-0.42, 0.0), (-0.125, 0.0), (0.125, 0.0), (0.375, 0.0))
BAR_R = 0.00635
BARS = tuple({"point": (x, 0.0, -(0.025 + BAR_R)), "direction": (0.0, 1.0, 0.0), "radius": BAR_R}
             for x in (-0.45, -0.15, 0.15, 0.45)) + tuple(
    {"point": (0.0, y, -(0.0381 + BAR_R)), "direction": (1.0, 0.0, 0.0), "radius": BAR_R} for y in (-0.15, 0.15))
BIT = 0.0127
POINT = 0.5 * BIT / math.tan(math.radians(59.0))
DEPTH = 2.0 * LEDGER_HALF[2] + 0.115 + POINT          # m: tip past the ledger face for 115 mm at full diameter
BLOW = (2.1, 4600.0 / 60.0)                            # J, 1/s: DeWalt DCH273
SPIN = 1100.0 * RPM
T_INST, H_LO, WASHER = 68.0, 0.093, 0.001
ANCHORS = tuple(f"anchor{k}" for k in range(4))


def hammer_rate(d: float, blow: tuple[float, float] = BLOW) -> float:
    """The rate [m/s] the blows cut the wall at bit diameter ``d``: E f / (e_v pi d^2 / 4)."""
    ev = energy_per_volume(PERCUSSIVE["concrete_medium_hard"], d)
    return blow[0] * blow[1] / (ev * math.pi * d * d / 4.0)


def _anchor(name: str) -> dict[str, Any]:
    return {"body": name, "tip": (0.0, 0.0, -0.133), "radius": 0.0063, "head": 0.012, "stiffness": 5.0e4,
            "anchor": H_LO, "anchor_hole": 0.01346}


def _plan(bit: float = BIT, at: dict[int, dict[str, Any]] | None = None, torque0: float = T_INST,
          marks: tuple[tuple[float, float], ...] = MARKS) -> Plan:
    feed = hammer_rate(bit) * 2.0 * math.pi / SPIN
    ops = tuple(Op(m, bit, DEPTH, feed, SPIN, **{"hammer": BLOW, "settle": 3.0, **(at or {}).get(k, {})})
                for k, m in enumerate(marks))
    marks = tuple((m[0] + op.offset[0], m[1] + op.offset[1]) for m, op in zip(marks, ops, strict=True))
    ins = tuple(Insert(a, m, 60.0, torque0 if k == 0 else T_INST) for k, (a, m) in enumerate(zip(ANCHORS, marks)))
    return Plan(ops, inserts=ins)


VARIANTS: tuple[tuple[str, Plan], ...] = (
    ("as asked", _plan()),
    ("hole 0 drilled onto a bar", _plan(marks=((-0.45, 0.0), *MARKS[1:]))),
    ("hole 0 tilted 4 deg", _plan(at={0: {"tilt": 4.0}})),
    ("a 14.3 mm bit", _plan(bit=0.0143)),
    ("nut 0 at 50 N m", _plan(torque0=50.0)),
    ("hole 0 onto a bar's side, pushed on", _plan(at={0: {"offset": (-0.018, 0.0), "stop_on_jam": False,
                                                          "settle": 8.0}})),
)


@dataclass(frozen=True)
class LedgerAnchor(VariantTask):
    """The task: scene, scripts per world (``VARIANTS``) and the checks; ``marks`` limits the marks drilled."""

    marks: int = 4
    variants: tuple[tuple[str, Plan], ...] = VARIANTS
    drill: dict[str, Any] = field(default_factory=lambda: {
        "tool": "spindle", "workpiece": "ledger", "tip": bit_tip(0.16), "diameter": BIT, "max_holes": 6,
        "capture": 0.9, "radial_stiffness": 5.0e4, "layers": (
            {"workpiece": "ledger", "work": "pine_southern", "density": density("wood_douglas_fir")},
            {"workpiece": "wall", "work": "concrete_medium_hard", "density": density("concrete_c25_30"),
             "bars": BARS}),
        "pins": tuple(_anchor(a) for a in ANCHORS)})
    head: dict[str, Any] = field(default_factory=lambda: {"kp": 2.0e4, "kd": 1.0e3})
    top: float = TOP
    push_seconds: float = 1.5
    d_push: float = 400.0
    limit: float = 300.0
    dt: float = CONTROL_DT / 16.0

    def build(self, b: Any) -> None:
        add_part(b, "wall", (0.0, 0.0, WALL - WALL_HALF[2]), WALL_HALF)
        add_part(b, "ledger", (0.0, 0.0, TOP - LEDGER_HALF[2]), LEDGER_HALF)
        add_spindle(b)
        for k, name in enumerate(ANCHORS):
            add_pin(b, name, k)

    def plans(self, worlds: int) -> list[Plan]:
        n = self.marks
        return [Plan(p.ops[:n], p.park, p.inserts[:n]) for p in super().plans(worlds)]

    def score(self, ep: Episode, w: int) -> Verdict:
        out: list[Check] = []
        r = ep.drill.read()
        for k in range(self.marks):
            out += self._hole(ep, w, k, r)
        out.append(self._strays(ep, w))
        unscored = tuple(f"hole {k}: anchor did not spin (<= 45 deg): no nut is turned" for k in range(self.marks))
        unscored += (f"ledger fastened where it was set (moved at most 10 mm): {STATIC}",
                     f"ledger still level within 3 mm over its length: {STATIC}",
                     "bit not chipped on steel: no chipping model",
                     f"the anchors and the ledger once the hands and brackets let go: {NO_GRAVITY}")
        return Verdict(out, unscored, detail={"variant": self.variant(w)})

    def _hole(self, ep: Episode, w: int, k: int, r: dict[str, np.ndarray]) -> list[Check]:
        mark = mark_point(MARKS[k], TOP)
        led = nearest(ep.drill, w, 0, ep.body_q, mark)
        through = led is not None and led[1] <= MARK_RADIUS and led[0].through
        e = led[1] if led is not None else math.inf
        out = [Check(f"hole {k}: through the ledger within 3 mm of the mark", through and e <= 0.003,
                     round(e * 1e3, 2), "<= 3 mm")]
        wall = nearest(ep.drill, w, 1, ep.body_q, mark)
        if wall is None or wall[1] > MARK_RADIUS:
            return out + [Check(f"hole {k}: hole in the concrete at the mark", False)]
        h = wall[0]
        fd = h.full_depth()
        cut = float(r["bar_cut"][w])
        out += [Check(f"hole {k}: axis within 3 deg of the wall normal", h.tilt_deg <= 3.0, round(h.tilt_deg, 2),
                      "<= 3 deg"),
                Check(f"hole {k}: depth 108-125 mm", 0.108 <= fd <= 0.125, round(fd * 1e3, 1)),
                Check(f"hole {k}: reinforcement damage at most 0.5 mm", cut <= 0.0005, round(cut * 1e3, 3),
                      "<= 0.5 mm")]
        return out + self._anchor_checks(ep, w, k, r)

    def _anchor_checks(self, ep: Episode, w: int, k: int, r: dict[str, np.ndarray]) -> list[Check]:
        b = int(ep.sim.plugins[f"push_{ANCHORS[k]}"].body.numpy()[w])
        q = ep.body_q[b].astype(np.float64)
        embed = float(r["pin_embed"][w, k])
        gap = float(q[2] - TOP)                                          # the washer's bearing face over the ledger
        axis = quat_rotate(q[3:], np.array([0.0, 0.0, -1.0]))
        tilt = angle_deg(axis, (0.0, 0.0, 1.0))
        torque = float(r["nut_torque"][w, k])
        engaged = bool(r["pin_anchored"][w, k])
        return [Check(f"hole {k}: anchor driven to h_nom (at least 93 mm) with its washer on the ledger",
                      embed >= H_LO and gap <= WASHER, round(embed * 1e3, 2), "at least 93 mm, washer <= 1 mm off"),
                Check(f"hole {k}: anchor straight (within 3 deg)", tilt <= 3.0, round(tilt, 2), "<= 3 deg"),
                Check(f"hole {k}: nut torqued to T_inst (61-116 N m)", 0.9 * T_INST <= torque <= 1.7 * T_INST,
                      round(torque, 2), "61-116 N m"),
                Check(f"hole {k}: expansion engaged (cone drawn into a gripping clip)", engaged, engaged)]

    def _strays(self, ep: Episode, w: int) -> Check:
        """Full-diameter holes in either part farther than the search radius from every mark."""
        bad = 0
        for layer in (0, 1):
            face = ep.drill.face(w, layer, ep.body_q)
            for h in ep.drill.holes(w, layer):
                p = h.in_world(face)[0]
                far = all(in_plane(face, p, mark_point(m, TOP)) > MARK_RADIUS for m in MARKS)
                bad += int(far and h.full_depth() > 0.0)
        return Check("no hole drilled away from the marks", bad == 0, bad, "0", "constraint")


def run(task: LedgerAnchor | None = None, *, worlds: int = len(VARIANTS), device: str = "cpu"
        ) -> tuple[list[Verdict], Episode]:
    return run_task(task or LedgerAnchor(), worlds, device)


def main(argv: list[str] | None = None) -> int:
    return cli(LedgerAnchor(), __doc__, argv)


if __name__ == "__main__":
    raise SystemExit(main())
