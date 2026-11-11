# SPDX-License-Identifier: Apache-2.0
"""The pin laws of the drill plug-in (drilling v2, D-5): a hole is geometry a bolt can pass.

Two 6 mm plates (``plate_a`` on ``plate_b``) under an M8-like bolt: an 8 mm shank 30 mm long under a 13 mm head,
on a 0.5 kg holder (no solver collision: the drill owns the pin contact). ``PushHead`` pushes it down. Each world's
hole table is seeded before the run (no drilling here; the drilling laws cover that); plate_b's hole is offset along
x. A through stage's point ends past the plate; a blind or counterbored stage has its full diameter to the depth
given. ``u*`` is where the tip should rest below the face that stops it (on a stage's point cone, (0.5 d - r) /
tan 59 deg below the full-diameter depth), plus F / k for the one contact spring that holds it (2e5 N/m).

``pin_through``: a square-ended bolt (no chamfer) at 150 N, 60 mm/s.

| world | plate_a / plate_b | the bolt |
|---|---|---|
| 0 | 9 mm / 9 mm, aligned | passes; the head seats on plate_a |
| 1 | none | stops on plate_a's face |
| 2 | 7.9 mm (0.05 mm narrower than the shank) / 9 mm | stops on plate_a's face |
| 3 | 9 mm / 9 mm, 0.7 mm off (past the 0.5 mm clearance) | stops on plate_b's face |
| 4 | 9 mm / 9 mm, 0.3 mm off (within the clearance) | passes |
| 5 | 9 mm blind, 4 mm full depth / none | stops on its point cone, u* 4.30 mm |
| 6 | 9 mm counterbore 3 mm deep over a 5 mm through pilot / 9 mm | stops on the shoulder, u* 3.30 mm |
| 7 | 5 mm (a pilot) / 9 mm | stops on plate_a's face |
| 8 | 14 mm counterbore 4 mm deep over a 9 mm through hole / 9 mm | the 13 mm head lands on its floor, u* 4.30 mm |

``pin_chamfer``: the same bolt with a 0.5 mm 45 deg chamfer on its end, at 150 N, 60 mm/s. The chamfer guides an
offset bolt in while its end overlaps the edge by no more than the chamfer, as far as plate_a's clearance lets it
move: it passes only where one axis fits both holes (offset up to both clearances: 1.0 mm here with 9 mm holes,
0.7 mm with plate_a 8.4 mm).

| world | plate_a / plate_b | the bolt |
|---|---|---|
| 0 | 9 mm / 9 mm, aligned | passes |
| 1 | 9 mm / 9 mm, 0.7 mm off | is guided in and passes |
| 2 | 9 mm / 9 mm, 0.95 mm off | is guided in and passes |
| 3 | 9 mm / 9 mm, 1.05 mm off (its flat end past the edge) | stops on plate_b's face |
| 4 | 7.2 mm / 9 mm (the end fits, the shank does not) | stops on its chamfer, u* 0.10 mm |
| 5, 6 | 8.4 mm / 9 mm, 0.75 and 0.9 mm off (past both clearances; within b's + the chamfer) | stops on its chamfer |

The checks: seated heads within 0.2 mm of plate_a's face less F / k with the tip through both plates; stopped tips
within 0.2 mm of u* + F / (a k) below the face that stops them, where a is 1 for a flat end and sin 45 deg for a
centred chamfer (its 45 deg spring, along the axis) and sin^2 45 deg = 0.5 for an off-centre one (worlds 5 and 6:
u* 0.45 and 0.30 mm, the shank against plate_a's wall); a head resting in a counterbore likewise below plate_a's
face. In every world the bolt ends within 4 deg of upright (plate_a's 8.4 mm hole lets the 8 mm shank tilt 3.8 deg
over the 6 mm plate). A wedged bolt holds upright only to about 150 N (the known gap): plate_a's walls resist its
tilt by two springs 6 mm apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.npgeom import quat_rotate

PLATE = 0.006                       # m: each plate's thickness
HALF = 0.03                         # m: plate half extent
SHANK_R, HEAD_R, SHANK = 0.004, 0.0065, 0.03
CHAMFER = 0.0005                    # m: the chamfered bolt end (pin_chamfer)
STIFFNESS = 2.0e5                   # N/m: the pins' contact springs (Pin's default)
HOLDER_HALF, HOLDER_MASS = 0.025, 0.5
GAP = 0.002                         # m: tip above plate_a at the start
STEP = 0.02 / 128                   # s: the pin and holder rotate fast against the wall springs
TOL = 2.0e-4                        # m
TILT = 4.0                          # deg: the most a bolt may end tilted
TAN = math.tan(math.radians(59.0))
THROUGH = PLATE + 0.01              # m: a through stage's bottom (its point out past the plate)
LAYERS = ({"workpiece": "plate_a", "work": "steel_200bhn", "density": 7850.0},
          {"workpiece": "plate_b", "work": "steel_200bhn", "density": 7850.0})


def through(d: float) -> tuple[float, float]:
    return (d, THROUGH)


def blind(d: float, depth: float) -> tuple[float, float]:
    """A stage of diameter d at full diameter to ``depth`` (its point cone below)."""
    return (d, depth + 0.5 * d / TAN)


def wedged(off: float) -> float:
    """u* of the chamfered end on plate_b's 9 mm mouth with the shank moved over to plate_a's 8.4 mm wall: the
    chamfer binds where its radius meets the edge, c - (off - 0.2 mm + r - 4.5 mm)."""
    return CHAMFER - (off - 0.0002 + SHANK_R - 0.0045)


def cone_rest(depth: float) -> float:
    """u* of the square bolt end on a 9 mm stage's point cone below ``depth``."""
    return depth + (0.0045 - SHANK_R) / TAN


@dataclass(frozen=True)
class Case:
    a: tuple[tuple[float, float], ...]       # plate_a's stages (diameter, bottom) at the origin
    b: tuple[tuple[float, float], ...]       # plate_b's, at x = off
    off: float
    want: str                                # "pass", the plate whose face stops the tip ("a", "b"), or "head"
    stop: float = 0.0                        # its u* below that face (the head's below plate_a's)
    axial: float = 1.0                       # the holding spring's stiffness along the axis, per k


@dataclass(frozen=True)
class PinLaw:
    name: str
    cases: tuple[Case, ...]
    force: float                             # N
    d_push: float                            # N s/m (speed = force / d_push)
    chamfer: float                           # m
    stiffness: float = STIFFNESS             # N/m: the pin's contact springs

    def pins(self) -> tuple[dict, ...]:
        return ({"body": "bolt", "top": (0.0, 0.0, 0.0), "tip": (0.0, 0.0, -SHANK), "radius": SHANK_R,
                 "head": HEAD_R, "chamfer": self.chamfer, "stiffness": self.stiffness},)


NINE = (through(0.009),)
STRONG = PinLaw("pin_through", (
    Case(NINE, NINE, 0.0, "pass"),
    Case((), (), 0.0, "a", 0.0),
    Case((through(0.0079),), NINE, 0.0, "a", 0.0),
    Case(NINE, NINE, 0.0007, "b", 0.0),
    Case(NINE, NINE, 0.0003, "pass"),
    Case((blind(0.009, 0.004),), (), 0.0, "a", cone_rest(0.004)),
    Case((blind(0.009, 0.003), through(0.005)), NINE, 0.0, "a", cone_rest(0.003)),
    Case((through(0.005),), NINE, 0.0, "a", 0.0),
    Case((blind(0.014, 0.004), through(0.009)), NINE, 0.0, "head", 0.004 + (0.007 - HEAD_R) / TAN),
), force=150.0, d_push=2500.0, chamfer=0.0)
GUIDE = PinLaw("pin_chamfer", (
    Case(NINE, NINE, 0.0, "pass"),
    Case(NINE, NINE, 0.0007, "pass"),
    Case(NINE, NINE, 0.00095, "pass"),
    Case(NINE, NINE, 0.00105, "b", 0.0),
    Case((through(0.0072),), NINE, 0.0, "a", 0.0036 - (SHANK_R - CHAMFER), math.sqrt(0.5)),
    Case((through(0.0084),), NINE, 0.00075, "b", wedged(0.00075), 0.5),
    Case((through(0.0084),), NINE, 0.0009, "b", wedged(0.0009), 0.5),
), force=150.0, d_push=2500.0, chamfer=CHAMFER)


def _build(c: Any):
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.5 * PLATE), hx=HALF, hy=HALF, hz=0.5 * PLATE, label="plate_b")
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 1.5 * PLATE), hx=HALF, hy=HALF, hz=0.5 * PLATE, label="plate_a")
        bolt = b.add_body(xform=body_xform(0.0, 0.0, 2.0 * PLATE + GAP + SHANK), label="bolt")
        b.add_shape_box(bolt, xform=body_xform(0.0, 0.0, HOLDER_HALF), hx=HOLDER_HALF, hy=HOLDER_HALF,
                        hz=HOLDER_HALF, cfg=newton.ModelBuilder.ShapeConfig(
                            density=HOLDER_MASS / (2.0 * HOLDER_HALF) ** 3, has_shape_collision=False))
        spindle = b.add_body(xform=body_xform(0.2, 0.0, 0.2), label=c.tool)    # the drill, idle beside the plates
        b.add_shape_box(spindle, hx=0.03, hy=0.03, hz=0.03,
                        cfg=newton.ModelBuilder.ShapeConfig(density=3.0 / 0.06**3, has_shape_collision=False))
    return build


def _seeder(cases: tuple[Case, ...]):
    def seed(sim: Any) -> None:
        """Each world's holes per its case (plate_a's at the origin, plate_b's offset along x)."""
        drill = sim.plugins["drill"]
        H, S, W = int(drill.config.max_holes), int(drill.config.max_stages), sim.model.world_count
        L = len(drill.config.layers)                                    # plate_a, then plate_b if it is drilled
        n, entry = np.zeros((W, L), np.int32), np.zeros((W, L * H, 2), np.float32)
        axis, sn = np.zeros((W, L * H, 3), np.float32), np.zeros((W, L * H), np.int32)
        sd, sb = np.zeros((W, L * H * S), np.float32), np.zeros((W, L * H * S), np.float64)
        for w in range(W):
            case = cases[min(w, len(cases) - 1)]
            for layer, stages, x in ((0, case.a, 0.0), (1, case.b, case.off)):
                if not stages or layer >= L:
                    continue
                slot = layer * H
                n[w, layer], entry[w, slot], axis[w, slot], sn[w, slot] = 1, (x, 0.0), (0.0, 0.0, -1.0), len(stages)
                for k, (d, bottom) in enumerate(stages):
                    sd[w, slot * S + k], sb[w, slot * S + k] = d, bottom
        for name, value in (("hole_n", n), ("hole_entry", entry), ("hole_axis", axis), ("stage_n", sn),
                            ("stage_d", sd), ("stage_b", sb)):
            getattr(drill, name).assign(value)
    return seed


def pin_scene(c: Any, law: PinLaw = GUIDE) -> TestScene:
    def companions():
        from alloy.plugins.holes.pins import PushHead  # noqa: PLC0415

        return [PushHead(body="bolt", force=law.force, d_push=law.d_push)]
    return TestScene(build=_build(c), init=_seeder(law.cases), worlds=len(law.cases), periods=50, dt=STEP,
                     ground=False, gravity=(0.0, 0.0, 0.0), companions=companions)


def _error(case: Case, head: float, tip: float, tilt: float, law: PinLaw) -> float:
    """This world's error in units of its bound (<= 1 passes): the rest and the tilt [deg]."""
    sink = law.force / law.stiffness
    upright = tilt / TILT
    if case.want == "pass":
        return max(abs(head - (2.0 * PLATE - sink)) / TOL, 0.0 if tip < 0.0 else math.inf, upright)
    if case.want == "head":
        return max(abs(2.0 * PLATE - head - case.stop - sink) / TOL, upright)
    depth = (2.0 * PLATE if case.want == "a" else PLATE) - tip
    return max(abs(depth - case.stop - sink / case.axial) / TOL, upright)


def checker(law: PinLaw):
    def check(run: Any) -> LawResult:
        q = run.trace.body_q[-1].astype(np.float64)
        idx = run.bodies("bolt")
        errs, rows = [], []
        for w, case in enumerate(law.cases):
            head = (q[idx[w], :3] + quat_rotate(q[idx[w], 3:], np.zeros(3)))[2]
            tip = (q[idx[w], :3] + quat_rotate(q[idx[w], 3:], np.array([0.0, 0.0, -SHANK])))[2]
            axis = quat_rotate(q[idx[w], 3:], np.array([0.0, 0.0, -1.0]))
            tilt = math.degrees(math.acos(min(max(-float(axis[2]), -1.0), 1.0)))
            errs.append(_error(case, head, tip, tilt, law))
            rows.append(f"w{w} {case.want}: head {head * 1e3:.3f} tip {tip * 1e3:.3f} mm, tilt {tilt:.2f} deg")
        return LawResult.within(max(errs), 1.0, "; ".join(rows) + f" (within {TOL * 1e3:.1f} mm of the rest "
                                f"+ F/(a k), F/k = {law.force / law.stiffness * 1e3:.2f} mm; tilt within {TILT:g} deg)",
                                "only listed pins see holes: a shank passes holes it fits, else the narrowing or the "
                                "face stops it")
    return check


def pin_laws(c: Any) -> list[Law]:
    return [Law(STRONG.name, checker(STRONG), pin_scene(c, STRONG),
                "D-5: a square bolt end passes drilled holes only where it fits, stopped at any push otherwise",
                config={"layers": LAYERS, "pins": STRONG.pins()}),
            Law(GUIDE.name, checker(GUIDE), pin_scene(c, GUIDE),
                "D-5: a chamfered bolt end is guided into an offset hole within its chamfer",
                config={"layers": LAYERS, "pins": GUIDE.pins()})]


__all__ = ["GUIDE", "STRONG", "Case", "PinLaw", "blind", "checker", "pin_laws", "pin_scene", "through"]
