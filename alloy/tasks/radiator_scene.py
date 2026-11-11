# SPDX-License-Identifier: Apache-2.0
"""behind-radiator's corner of the room: the wall under the window, the cast-iron radiator in front of it, the trim,
and the band to paint (Gym-Anything's painting wall preset, task behind-radiator; ``PT`` below is its
``benchmarks/roboworld/environments/painting/``).

Frame: x along the wall from the radiator's centre (Gym-Anything's window centre, x 0.55 m in its room), y toward the
wall with the wall's face at y = 0 (the room at y < 0), z up from the floor. Gym-Anything's room maps by
``x - 0.55``, ``y - 1.675`` (``PT/layout.py:14-16``).

Sources, all Gym-Anything's (``PT/params.py:225-232``, ``PT/layout.py:58-73, 134``):
- the radiator: 4-column cast iron, 0.64 m long, 0.229 m deep, 0.635 m tall, its back 64 mm (2.5 in) off the wall,
  10 sections at the 63.5 mm pitch (each 0.76 of the pitch wide), hubs 25 mm deep at z 0.11 and 0.585 m, two feet;
  its supply pipe (r 16.7 mm) 30 mm past its left end, up to z 0.11 m;
- the trim: a 3-1/4 x 1/2 in baseboard (top at z 0.083 m); under the window (sill 0.813 m), a 22 mm stool 75 mm
  deep plus 30 mm into the opening, and an 83 mm apron 17.5 mm proud, 1.028 m long (the opening and both casings);
- the band to paint: the wall between the baseboard's top and the apron's underside over the radiator's length
  (``BEHIND_RAD``, 0.400 m²); the pocket the task's own brush cannot reach (the bottom of the middle: Gym-Anything's
  x 0.525-0.580 m below z 0.350 m, 145 cm² in their cells) is not scored (``PT/task.py:57-86``).

The wall chart (4 mm cells, surface flow's paint wall scale) spans the band plus 100 mm of wall past each radiator
end, from the baseboard's top to 1 mm under the apron (whole cells). All its edges are closed, as Gym-Anything's:
it masks the wall below the baseboard's top (``PT/system.py:150``) and its film's faces into masked cells
(``PT/film.py:413-415``), so creeping film stops at the edge (``PT/task.py:29-31``). Two more charts catch drips
(``PROTECTED``): the radiator's top and the baseboard's ledge; film leaving their open edges is on the radiator or
the trim (spilled: protected's sinks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

GA_X, GA_Y = 0.55, 1.675                 # Gym-Anything's window centre and wall face in its room [m]

# the radiator (PT/params.py:231)
RAD_LEN, RAD_DEPTH, RAD_HEIGHT, RAD_GAP, RAD_SECTIONS = 0.64, 0.229, 0.635, 0.064, 10
PIPE_R, PIPE_X, PIPE_TOP = 0.0167, -0.5 * 0.64 - 0.03, 0.11      # PT/layout.py:75
# the trim (PT/params.py:225-228)
BASE_H, BASE_T = 0.083, 0.0127
SILL, STOOL_T, STOOL_D, APRON, CASING, CASING_T, WIN_W = 0.813, 0.022, 0.075, 0.083, 0.057, 0.0175, 0.914
APRON_BOTTOM = SILL - STOOL_T - APRON                             # 0.708 m

# the scored band and its unreachable pocket (PT/layout.py:134, PT/task.py:86), in this frame
BAND = (-0.5 * RAD_LEN, 0.5 * RAD_LEN, BASE_H, APRON_BOTTOM)     # x0, x1, z0, z1
POCKET = (0.525 - GA_X, 0.580 - GA_X, 0.350)                     # x0, x1, z1 (from the band's bottom)
ON_EDGE = 1e-9                     # m: a centre this close to a bound is on it (0.580 - 0.55 rounds under 0.030)

CELL = 0.004                       # m
MARGIN = 0.10                      # m of wall charted past each radiator end
WALL_T = 0.02                      # m, the wall box's thickness (its top face is the chart)
TOP_T = 0.004                      # m, the radiator-top and baseboard-ledge chart boxes

#: chart indices: the wall is chart 0 (the brush's); the others are protected and only catch drips
WALL, RAD_TOP, LEDGE = 0, 1, 2
PROTECTED = (RAD_TOP, LEDGE)


@dataclass(frozen=True)
class Box:
    """A static box: its name, centre, quaternion (x, y, z, w) and half extents, and what it is (``kind``:
    "wall", "radiator", "trim" or "chart" for a protected chart's thin box)."""

    name: str
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    half: tuple[float, float, float]
    kind: str

    def harness(self) -> tuple:
        return (self.name, self.pos, self.quat, self.half)


_UP = (0.0, 0.0, 0.0, 1.0)
_FACE_ROOM = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))           # local +z onto -y, local y up


def wall_cells() -> tuple[int, int]:
    """The wall chart's cells (nx along the wall, ny up)."""
    return int(round((RAD_LEN + 2.0 * MARGIN) / CELL)), int(math.floor((APRON_BOTTOM - BASE_H) / CELL + 1e-9))


def wall_size() -> tuple[float, float]:
    nx, ny = wall_cells()
    return nx * CELL, ny * CELL


def radiator_boxes() -> list[Box]:
    """The radiator's sections, hubs and feet and its supply pipe (as a square post of the pipe's width)."""
    y0, y1 = -RAD_GAP - RAD_DEPTH, -RAD_GAP
    yc, pitch = 0.5 * (y0 + y1), RAD_LEN / RAD_SECTIONS
    out = []
    for k in range(RAD_SECTIONS):
        x = -0.5 * RAD_LEN + (k + 0.5) * pitch
        out.append(Box(f"rad_sec_{k}", (x, yc, 0.06 + 0.5 * (RAD_HEIGHT - 0.06)), _UP,
                       (0.38 * pitch, 0.5 * RAD_DEPTH, 0.5 * (RAD_HEIGHT - 0.06)), "radiator"))
    for k, z in enumerate((0.11, RAD_HEIGHT - 0.05)):
        out.append(Box(f"rad_hub_{k}", (0.0, yc, z), _UP, (0.5 * RAD_LEN, 0.025, 0.02), "radiator"))
    for k, x in enumerate((-0.5 * RAD_LEN + 0.02, 0.5 * RAD_LEN - 0.02)):
        out.append(Box(f"rad_foot_{k}", (x, yc, 0.03), _UP, (0.018, 0.5 * RAD_DEPTH, 0.03), "radiator"))
    out.append(Box("rad_pipe", (PIPE_X, yc, 0.5 * PIPE_TOP), _UP, (PIPE_R, PIPE_R, 0.5 * PIPE_TOP), "radiator"))
    return out


def trim_boxes() -> list[Box]:
    """The baseboard along the charted wall, the apron and the stool."""
    w, _ = wall_size()
    ys = -0.5 * STOOL_D + 0.03                       # the stool: 75 mm into the room, 30 mm into the opening
    return [Box("baseboard", (0.0, -0.5 * BASE_T, 0.5 * BASE_H), _UP, (0.5 * w + 0.1, 0.5 * BASE_T, 0.5 * BASE_H),
                "trim"),
            Box("apron", (0.0, -0.5 * CASING_T, APRON_BOTTOM + 0.5 * APRON), _UP,
                (0.5 * WIN_W + CASING, 0.5 * CASING_T, 0.5 * APRON), "trim"),
            Box("stool", (0.0, ys, SILL - 0.5 * STOOL_T), _UP, (0.5 * WIN_W + CASING + 0.02, 0.5 * STOOL_D + 0.03,
                                                               0.5 * STOOL_T), "trim")]  # fmt: skip


def chart_boxes() -> list[Box]:
    """The charts' boxes, in chart order: the wall (top face toward the room, local y up), the radiator's top and
    the baseboard's ledge (top faces up)."""
    w, h = wall_size()
    return [Box("wall", (0.0, 0.5 * WALL_T, BASE_H + 0.5 * h), _FACE_ROOM, (0.5 * w, 0.5 * h, 0.5 * WALL_T), "wall"),
            Box("radiator_top", (0.0, -RAD_GAP - 0.5 * RAD_DEPTH, RAD_HEIGHT - 0.5 * TOP_T), _UP,
                (0.5 * RAD_LEN, 0.5 * RAD_DEPTH, 0.5 * TOP_T), "chart"),
            Box("baseboard_ledge", (0.0, -0.5 * BASE_T, BASE_H - 0.5 * TOP_T), _UP,
                (0.5 * w, 0.5 * BASE_T, 0.5 * TOP_T), "chart")]  # fmt: skip


def obstacles() -> list[Box]:
    """What the brush must clear: the radiator and the trim."""
    return radiator_boxes() + trim_boxes()


def charts():  # noqa: ANN201 - tuple of surface_flow PlaneChart
    """The plug-in's charts, in chart order (``WALL``, ``RAD_TOP``, ``LEDGE``): the wall's edges are closed
    (Gym-Anything's, the module's note); the protected charts' edges are open (paint runs off them)."""
    from alloy.plugins.surface_flow import PlaneChart  # noqa: PLC0415

    w, _ = wall_size()
    top = (int(round(RAD_LEN / CELL)), int(math.floor(RAD_DEPTH / CELL + 1e-9)))
    ledge = (int(round(w / CELL)), 3)
    return (PlaneChart("wall", cells=wall_cells(), edges=("wall",) * 4),
            PlaneChart("radiator_top", cells=top, edges=("open",) * 4,
                       size=(top[0] * CELL, top[1] * CELL)),
            PlaneChart("baseboard_ledge", cells=ledge, edges=("open",) * 4))  # fmt: skip


def wall_centres(local: np.ndarray) -> np.ndarray:
    """The wall chart's cell centres [nx, ny, 2] as (x, z) in this frame, from the plug-in's ``cell_centres``
    (box-local, centred: local x along the wall, local y up)."""
    _, h = wall_size()
    out = np.array(local[..., :2], dtype=np.float64)
    out[..., 1] += BASE_H + 0.5 * h
    return out


def band_mask(xz: np.ndarray) -> np.ndarray:
    """Cells whose centres lie in the band (Gym-Anything's ``_rect_cells``: inclusive on every side)."""
    x0, x1, z0, z1 = BAND
    x, z, e = xz[..., 0], xz[..., 1], ON_EDGE
    return (x >= x0 - e) & (x <= x1 + e) & (z >= z0 - e) & (z <= z1 + e)


def pocket_mask(xz: np.ndarray) -> np.ndarray:
    """The band's cells the task does not score: the pocket at the bottom of the middle, inclusive (its right
    bound falls on a column of centres: 14 columns, 56 mm, as Gym-Anything's 0.525-0.580 m)."""
    x0, x1, z1 = POCKET
    x, z, e = xz[..., 0], xz[..., 1], ON_EDGE
    return band_mask(xz) & (x >= x0 - e) & (x <= x1 + e) & (z <= z1 + e)


def edge_mask(xz: np.ndarray, run_edge: float) -> np.ndarray:
    """The band's bottom rows, centres within ``run_edge`` of the baseboard's top: not read for runs. In 4 mm cells
    from the baseboard's top that is the bottom row (4 mm); Gym-Anything's 2 mm cells give up to three rows (about
    5 mm), so runs are read about 1 mm lower here (stricter)."""
    return band_mask(xz) & (xz[..., 1] <= BAND[2] + run_edge)


__all__ = ["BAND", "BASE_H", "Box", "CELL", "LEDGE", "POCKET", "PROTECTED", "RAD_GAP", "RAD_HEIGHT", "RAD_LEN",
           "RAD_TOP", "WALL", "band_mask", "chart_boxes", "charts", "edge_mask", "obstacles", "pocket_mask",
           "radiator_boxes", "trim_boxes", "wall_cells", "wall_centres", "wall_size"]
