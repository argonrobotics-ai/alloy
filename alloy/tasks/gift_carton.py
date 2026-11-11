# SPDX-License-Identifier: Apache-2.0
"""A regular slotted carton (RSC) with its four top flaps on crease hinges, as MJCF (the gift-bags open-flaps task).

The reference task set's supply carton: RSC 14 x 10 x 8 in (356 x 254 x 203 mm inside), C-flute board, body and four
top flaps rigid, the flaps on creased hinges; empty it weighs 0.33 kg. Here:

- **Body.** One free rigid body: bottom and four walls of the board's caliper (4 mm), 0.67 kg/m² (0.33 kg over the
  carton's 0.49 m² of board).
- **Flaps.** RSC flaps are half the width deep (127 mm). Minor flaps hinge on the end walls (x), major flaps on the
  side walls (y); closed, the minors rest on the side walls' tops, under the majors, which meet at the middle. Each
  hinge sits on its wall's top, mid-board, so a flap clears its own wall from closed to folded down outside.
- **Crease.** Each hinge is a Jenkins element, as SC4's creased blank (alloy/clips/fold_box_blank.py): a slider
  hinge whose dry friction is the crease's plastic moment (its angle is the plastic set), in series with an elastic
  hinge spring. Released after a fold past the elastic range, a flap springs back by that range and keeps the rest.
  Values per metre of crease: plastic moment 0.8 N·m/m (ESTIMATE: the reference's C-flute score moment), elastic range
  45 deg (creased board folded to 90 deg springs back 44-46 deg, Nagasawa 2019; the library's paper_board fold memory),
  so k = 1.02 N·m/rad per m. A major flap's weight moment (0.02 N·m) is small against its 0.28 N·m.
- **Start.** The tape is cut and one major (+y) sits sprung 12 deg open (its set; the reference ages its carton closed,
  0.6 deg); the other major lies closed and the minors closed under both.

Flap angles read from closed: 0 closed, 90 upright, 180 folded out level; past that toward the outside wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

FLAPS = ("major_p", "major_m", "minor_p", "minor_m")     # +y / -y side walls, +x / -x end walls


@dataclass(frozen=True)
class CartonSpec:
    length: float = 0.356               # inside x [m] (14 in)
    width: float = 0.254                # inside y [m] (10 in)
    height: float = 0.203               # inside z [m] (8 in)
    caliper: float = 0.004              # C-flute board [m]
    areal: float = 0.67                 # kg/m² (0.33 kg empty)
    crease_moment: float = 0.8          # N·m per m of crease (plastic)
    elastic_deg: float = 45.0           # elastic range of a crease [deg]
    crease_damping: float = 0.002       # N·m·s/rad per m
    sprung_deg: float = 12.0            # the +y major's set at the start (the tape cut; the others closed)
    held_deg: float = 0.0               # the +y major held this far below its set (taped shut: sprung_deg), by a
                                        # torque actuator "major_p_hold" (hold_torque) standing in for the tape
    gap: float = 2.0e-4                 # under each closed flap (wall tops, minors) [m]
    reserve: int = 12                   # inert geoms: the engine budgets 8 contacts per shape, and a packed carton
                                        # (box on box, up to 8 points a pair) needs more
    slider_armature: float = 1.0e-3     # kg·m² on each plastic slider: MuJoCo's soft dry friction lets a DOF drift at
                                        # about R·load, R ~ 1e-4 / inertia; massless sliders drifted 6 deg/s

    @property
    def k(self) -> float:
        return self.crease_moment / math.radians(self.elastic_deg)

    @property
    def flap(self) -> float:
        return 0.5 * self.width


def _f(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def _hinges(s: CartonSpec) -> dict[str, dict]:
    """Per flap: hinge point (carton frame, bottom centre at the origin), opening axis, inward direction, crease
    length and the closed flap's height."""
    t, L, W, H, g = s.caliper, s.length, s.width, s.height, s.gap
    zmin, zmaj = H + 0.5 * t + g, H + 1.5 * t + 2 * g      # minors on the wall tops, majors on the minors
    return {               # widths reach half a caliper over the neighbouring walls (minors rest on the side walls)
        "major_p": {"p": (0.0, 0.5 * W + 0.5 * t, zmaj), "axis": (-1, 0, 0), "in": (0, -1, 0), "len": L, "w": L + t},
        "major_m": {"p": (0.0, -0.5 * W - 0.5 * t, zmaj), "axis": (1, 0, 0), "in": (0, 1, 0), "len": L, "w": L + t},
        "minor_p": {"p": (0.5 * L + 0.5 * t, 0.0, zmin), "axis": (0, 1, 0), "in": (-1, 0, 0), "len": W, "w": W + t},
        "minor_m": {"p": (-0.5 * L - 0.5 * t, 0.0, zmin), "axis": (0, -1, 0), "in": (1, 0, 0), "len": W, "w": W + t},
    }


def _quat_axis(axis, angle: float) -> str:
    a = np.asarray(axis, float)
    return _f([math.cos(0.5 * angle), *(math.sin(0.5 * angle) * a)])


def _flap(name: str, h: dict, s: CartonSpec, start: float, rho: float, held: float = 0.0) -> list[str]:
    """Slider (plastic set, dry friction) carrying the flap on an elastic hinge; geometry closed along ``in``. ``held``
    [rad]: the flap starts this far below its set, the elastic hinge loaded (its rest still at the set)."""
    k, mp, c = s.k * h["len"], s.crease_moment * h["len"], s.crease_damping * h["len"]
    q_held = f' quat="{_quat_axis(h["axis"], -held)}"' if held else ""
    d = np.asarray(h["in"], float)
    centre = d * 0.5 * s.flap
    half = np.abs(d) * 0.5 * s.flap + (1 - np.abs(d)) * 0.5 * h["w"]
    half[2] = 0.5 * s.caliper
    fr = 'solreffriction="0.0025 1" solimpfriction="0.99 0.9999 0.001 0.5 2"'
    return [
        f'<body name="{name}_slider" pos="{_f(h["p"])}" quat="{_quat_axis(h["axis"], start)}">',
        '  <inertial pos="0 0 0" mass="1e-4" diaginertia="1e-9 1e-9 1e-9"/>',
        f'  <joint name="{name}_set" type="hinge" axis="{_f(h["axis"])}" range="-0.2 4.6" frictionloss="{mp:.5g}" '
        f'damping="{c:.5g}" armature="{s.slider_armature:g}" {fr}/>',
        f'  <body name="{name}"{q_held}>',
        f'    <joint name="{name}_el" type="hinge" axis="{_f(h["axis"])}" range="-1.4 1.4" stiffness="{k:.5g}" '
        f'springref="{held:.6g}" damping="{c:.5g}" armature="1e-6"/>',
        f'    <geom name="{name}_board" type="box" pos="{_f(centre)}" size="{_f(half)}" density="{rho:.5g}"/>',
        "  </body>",
        "</body>",
    ]


def carton_mjcf(s: CartonSpec = CartonSpec(), pos=(0.0, 0.0, 0.0), name: str = "carton") -> str:
    """The carton on a free joint, its bottom centre at ``pos``."""
    t, L, W, H = s.caliper, s.length, s.width, s.height
    rho = s.areal / t
    walls = [("bottom", (0, 0, 0.5 * t), (0.5 * L + t, 0.5 * W + t, 0.5 * t)),
             ("wall_yp", (0, 0.5 * W + 0.5 * t, 0.5 * H + 0.5 * t), (0.5 * L + t, 0.5 * t, 0.5 * H - 0.5 * t)),
             ("wall_ym", (0, -0.5 * W - 0.5 * t, 0.5 * H + 0.5 * t), (0.5 * L + t, 0.5 * t, 0.5 * H - 0.5 * t)),
             ("wall_xp", (0.5 * L + 0.5 * t, 0, 0.5 * H + 0.5 * t), (0.5 * t, 0.5 * W, 0.5 * H - 0.5 * t)),
             ("wall_xm", (-0.5 * L - 0.5 * t, 0, 0.5 * H + 0.5 * t), (0.5 * t, 0.5 * W, 0.5 * H - 0.5 * t))]
    lines = [f'<mujoco model="{name}"><compiler angle="radian"/><worldbody>',
             f'<body name="{name}" pos="{_f(pos)}">', '<freejoint name="free"/>']
    lines += [f'<geom name="{g}" type="box" pos="{_f(p)}" size="{_f(h)}" density="{rho:.5g}"/>'
              for g, p, h in walls]
    lines += [f'<geom name="reserve_{k}" type="sphere" size="0.001" pos="0 0 {0.5 * H:.4g}" mass="0" contype="0" '
              f'conaffinity="0" group="5"/>' for k in range(s.reserve)]
    hs = _hinges(s)
    for f in FLAPS:
        start = math.radians(s.sprung_deg) if f == "major_p" else 0.0
        lines += _flap(f, hs[f], s, start, rho, math.radians(s.held_deg) if f == "major_p" else 0.0)
    lines += ["</body>", "</worldbody>", "<contact>"]          # majors meet only the minors and each other: the
    lines += [f'<exclude body1="{name}" body2="{f}"/>' for f in FLAPS[:2]]    # contact budget is 8 per shape
    lines += ["</contact>"]
    if s.held_deg:
        lines += ['<actuator><motor name="major_p_hold" joint="major_p_el" gear="1"/></actuator>']
    lines += ["</mujoco>"]
    return "\n".join(lines)


def hold_torque(s: CartonSpec) -> float:
    """The torque [N·m] on ``major_p_hold`` that holds a taped major shut: its loaded elastic hinge's, reversed. A
    tape's bonds act on the film only (alloy.plugins.bonds), so this stands in for their pull on the flap."""
    return -s.k * _hinges(s)["major_p"]["len"] * math.radians(s.held_deg)


def flap_angles(s: CartonSpec, carton_quat_wxyz: np.ndarray, flap_quats_wxyz: dict[str, np.ndarray]) -> dict:
    """Each flap's opening angle from closed [deg], from the flap's and the carton's orientations (one world)."""
    Rc = _rot(carton_quat_wxyz)
    out = {}
    for f, h in _hinges(s).items():
        R = Rc.T @ _rot(flap_quats_wxyz[f])                    # flap frame in the carton frame
        d0, ax = np.asarray(h["in"], float), np.asarray(h["axis"], float)
        d = R @ d0
        a = math.degrees(math.atan2(float(np.dot(np.cross(d0, d), ax)), float(np.dot(d0, d))))
        out[f] = a + 360.0 if a < -90.0 else a
    return out


def _rot(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
