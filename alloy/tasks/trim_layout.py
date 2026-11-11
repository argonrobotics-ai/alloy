# SPDX-License-Identifier: Apache-2.0
"""cut-in-trim's wall: Gym-Anything's window casing and outlet plate on two cropped wall charts, the bands to paint,
the protected trim, and the cut pail.

World frame as :mod:`alloy.tasks.paint_scene`: the wall face is the plane y = ``face_y``, the room at y < face_y, x
along the wall (Gym-Anything's room x), z up from the floor. Gym-Anything's geometry (``PT/params.py:223-231``,
``PT/layout.py:43-57, 131-133``):
- a 36 x 48 in window centred at x 0.55 m, sill 0.813 m, head 2.032 m, with 2-1/4 in colonial casing (57 mm wide,
  11/16 in = 17.5 mm proud), a stool (22 mm thick, 20 mm past the casings) and an 83 mm apron;
- a duplex outlet plate 70 x 114 mm, its centre at x -0.90 m, z 0.356 m;
- the bands (``CASING_BAND``, ``OUTLET_BAND``): 40 mm of wall left of the left casing, from the sill to the top of
  the casing head; 40 mm round the plate.

Gym-Anything paints one wall chart at 2 mm cells. Here each job has a crop of it (``Crop``): the casing band with
40 mm of wall round it and 40 mm of the trim's footprint, and the outlet ring with 40 mm round it. Every band and
footprint edge falls on a cell face. The trim and the plate stand proud of the wall, so the film on the wall cannot
reach their faces: their footprints on the chart are a region of their own (``set_regions``), sealed from the wall,
and paint a brush lays there is paint on the trim (the brush overran the line into the trim's side or face).
Gym-Anything charts those faces separately; the footprint is the same test of the line.

The wall's first coat (seed 0: naval over its P-shade primer, ``PT/layout.py:170-181``) is Gym-Anything's: 43 µm dry,
a little patchy, ``43·(1 + 0.15 f)`` with ``f`` a smooth random field over 0.1-0.3 m scaled to [-1, 1]
(``PT/system.py:383-392``, its seed 7000 + the episode's), on Gym-Anything's whole 2 mm wall chart; each crop reads
it at its cell centres. Where it is thinnest the finish coat must bring 25.7 µm dry to hide (19.2 µm on 43 µm).

The cut pail: Gym-Anything's 1 qt pail on the floor (0.6 L in it), 0.375 m out from the wall, as a square ``Tray``
of the same area as the paint's surface in the tapered pail (``PT/params.py:189-190``, ``PT/layout.py:178``).

Not measured here (a robot port must add them): Gym-Anything's ``protected`` also reads every trim face and the paint
dripped into sinks anywhere in the room (``PT/task.py:224-250``); these crops see only the trim's footprint next to
the two bands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# Gym-Anything's room parts [m] (PT/params.py:225-231)
WINDOW = {"x": 0.55, "w": 0.914, "sill": 0.813, "head": 2.032, "casing": 0.057, "casing_t": 0.0175,
          "stool_t": 0.022, "apron": 0.083, "stool_over": 0.02}  # fmt: skip
OUTLET = {"x": -0.90, "z": 0.356, "w": 0.070, "h": 0.114, "t": 0.005}
PAIL = {"r_top": 0.089, "r_bottom": 0.080, "height": 0.140, "fill": 0.6e-3, "x": -0.02, "out": 0.375}
BAND = 0.04                     # m: the band to cut in (Gym-Anything's CASING_BAND / OUTLET_BAND)
ROOM = {"x": 3.66, "z": 2.438}  # m: the work wall's width and height (PT/params.py:223)
GA_CELL = 0.002                 # m: Gym-Anything's wall chart cell
PATCHY = 0.15                   # the first coat's relative variation (PT/system.py:385)
JOBS = ("casing", "outlet")


@dataclass(frozen=True)
class Rect:
    """A rectangle on the wall, world x0 < x1 and z0 < z1 [m]."""

    x0: float
    x1: float
    z0: float
    z1: float

    def mask(self, xz: np.ndarray) -> np.ndarray:
        x, z = xz[..., 0], xz[..., 1]
        return (x >= self.x0) & (x <= self.x1) & (z >= self.z0) & (z <= self.z1)

    def grown(self, d: float) -> Rect:
        return Rect(self.x0 - d, self.x1 + d, self.z0 - d, self.z1 + d)


@dataclass(frozen=True)
class Crop:
    """One cropped wall chart: ``rect`` on the wall in ``cell`` cells, for the job ``kind`` ("casing" or "outlet")."""

    kind: str
    rect: Rect
    cell: float

    @property
    def name(self) -> str:
        return f"wall_{self.kind}"

    @property
    def cells(self) -> tuple[int, int]:
        r = self.rect
        return int(round((r.x1 - r.x0) / self.cell)), int(round((r.z1 - r.z0) / self.cell))

    @property
    def centre(self) -> tuple[float, float]:
        return 0.5 * (self.rect.x0 + self.rect.x1), 0.5 * (self.rect.z0 + self.rect.z1)

    def world_xz(self, local: np.ndarray) -> np.ndarray:
        """World (x, z) [nx, ny, 2] of the chart's cell centres from the plug-in's ``cell_centres`` (box-local, centred:
        local x along the wall, local y up)."""
        cx, cz = self.centre
        out = np.array(local[..., :2], dtype=np.float64)
        out[..., 0] += cx
        out[..., 1] += cz
        return out

    def box(self, face_y: float) -> tuple:
        """The crop as a ``paint_scene.chart_scene`` box: its top face is the wall face, facing -y, local y up."""
        s = math.sqrt(0.5)
        nx, ny = self.cells
        cx, cz = self.centre
        half = (0.5 * nx * self.cell, 0.5 * ny * self.cell, 0.01)
        return (self.name, (cx, face_y + 0.01, cz), (s, 0.0, 0.0, s), half)


@dataclass(frozen=True)
class TrimLayout:
    """The cut-in-trim wall. ``casing_top`` [m] cuts the casing band short (light tests); None: up to the top of the
    casing head, as Gym-Anything. ``margin``: wall round the bands on each crop. ``jobs``: the bands in the run (both
    for the task; one for light tests)."""

    cell: float = 0.002                     # m (Gym-Anything's wall cell)
    margin: float = 0.04                    # m
    face_y: float = 0.0
    casing_top: float | None = None
    paint: str = "latex_eggshell"
    colour: str = "naval"                   # SW 6244, Gym-Anything's seed-0 wall paint (a deep colour)
    substrate: str = "primer_grey"          # its P-shade primer (seed 0)
    first_coat_dft: float = 43e-6           # m: the wall's first coat of the colour, dry (seed 0: one 1.7 mil coat)
    jobs: tuple[str, ...] = JOBS
    seed: int = 0                           # Gym-Anything's episode seed (the first coat's field)

    def __post_init__(self) -> None:
        if not self.jobs or len(set(self.jobs)) != len(self.jobs) or not set(self.jobs) <= set(JOBS):
            raise ValueError(f"TrimLayout.jobs {self.jobs}: one or both of {JOBS}, once each")
        for crop in self.crops():
            for v in (crop.rect.x1 - crop.rect.x0, crop.rect.z1 - crop.rect.z0):
                if abs(v / self.cell - round(v / self.cell)) > 1e-6:
                    raise ValueError(f"TrimLayout: crop {crop.name} ({v:.4f} m) is not a whole number of "
                                     f"{self.cell} m cells; give a cell that divides 1 mm steps")
        if self.casing_top is not None and not WINDOW["sill"] + 0.1 <= self.casing_top <= self.casing_band().z1:
            raise ValueError(f"TrimLayout.casing_top {self.casing_top} m must lie 0.1 m above the sill "
                             f"({WINDOW['sill']} m) or more, up to the casing head's top")

    # --- the trim and the bands (world x, z) ---------------------------------------------------------------------

    @staticmethod
    def casing_x0() -> float:
        """The left casing's outer edge [m]: the line to cut to."""
        return WINDOW["x"] - 0.5 * WINDOW["w"] - WINDOW["casing"]

    def casing_band(self) -> Rect:
        """Gym-Anything's CASING_BAND (PT/layout.py:131), cut at ``casing_top``."""
        x1 = self.casing_x0()
        top = WINDOW["head"] + WINDOW["casing"]
        return Rect(x1 - BAND, x1, WINDOW["sill"], top if self.casing_top is None else self.casing_top)

    @staticmethod
    def plate() -> Rect:
        o = OUTLET
        return Rect(o["x"] - 0.5 * o["w"], o["x"] + 0.5 * o["w"], o["z"] - 0.5 * o["h"], o["z"] + 0.5 * o["h"])

    def outlet_band(self) -> Rect:
        """Gym-Anything's OUTLET_BAND (PT/layout.py:132): the plate grown by 40 mm (the plate itself is not wall)."""
        return self.plate().grown(BAND)

    def trim(self) -> list[Rect]:
        """The window trim's footprints on the wall: casing legs, head, stool and apron (Gym-Anything masks the same
        rectangle out of its wall chart, PT/system.py:149), and the stool's horn past the casing."""
        w = WINDOW
        x0, x1 = self.casing_x0(), w["x"] + 0.5 * w["w"] + w["casing"]
        low = w["sill"] - w["stool_t"] - w["apron"]
        horn = x0 - w["stool_over"]
        return [Rect(x0, x1, low, w["head"] + w["casing"]), Rect(horn, x1 + w["stool_over"], w["sill"] - w["stool_t"],
                                                               w["sill"])]

    # --- the crops --------------------------------------------------------------------------------------------------

    def crops(self) -> tuple[Crop, ...]:
        """The jobs' crops, in ``jobs`` order (the film's chart order)."""
        rects = {"casing": self.casing_band().grown(self.margin), "outlet": self.outlet_band().grown(self.margin)}
        return tuple(Crop(kind, rects[kind], self.cell) for kind in self.jobs)

    def masks(self, crop: Crop, xz: np.ndarray) -> dict[str, np.ndarray]:
        """``target`` (the band) and ``protected`` (the trim's or the plate's footprint) on ``crop``, over its cell
        centres ``xz``."""
        if crop.kind == "casing":
            prot = np.zeros(xz.shape[:2], bool)
            for r in self.trim():
                prot |= r.mask(xz)
            return {"target": self.casing_band().mask(xz) & ~prot, "protected": prot}
        plate = self.plate().mask(xz)
        return {"target": self.outlet_band().mask(xz) & ~plate, "protected": plate}

    def regions(self, crop: Crop, xz: np.ndarray) -> np.ndarray:
        """Region ids for ``set_regions``: 0 wall, 1 trim or plate (sealed from the wall's film)."""
        return self.masks(crop, xz)["protected"].astype(np.int32)

    def first_coat(self, xz: np.ndarray) -> np.ndarray:
        """The wall's dry first coat [m] at wall points ``xz`` [..., 2]: Gym-Anything's patchy coat, read from its
        2 mm wall chart (cells from x = -1.83 m, z = 0)."""
        f = first_coat_field(self.seed)
        i = np.clip(np.floor((xz[..., 0] + 0.5 * ROOM["x"]) / GA_CELL + 1e-6).astype(int), 0, f.shape[0] - 1)
        j = np.clip(np.floor(xz[..., 1] / GA_CELL + 1e-6).astype(int), 0, f.shape[1] - 1)
        return self.first_coat_dft * (1.0 + PATCHY * f[i, j])

    def pail(self):  # noqa: ANN201 - surface_flow.stores.Tray
        """The cut pail as a ``Tray``: the paint's surface area at its fill level, as a square on the floor."""
        from alloy.plugins.surface_flow.stores import Tray  # noqa: PLC0415

        return Tray(center=self.pail_floor(), half=(pail_half(),) * 2, volume=PAIL["fill"])

    def pail_floor(self) -> tuple[float, float, float]:
        return PAIL["x"], self.face_y - PAIL["out"], 0.0


@lru_cache(maxsize=4)
def first_coat_field(seed: int) -> np.ndarray:
    """Gym-Anything's first-coat field ``f`` [nu, nv] in [-1, 1] over its whole wall chart (PT/system.py:386-391): a
    normal field on a 1/60 grid, cubic-zoomed to the cells, over its largest magnitude."""
    from scipy.ndimage import zoom  # noqa: PLC0415

    nu, nv = int(round(ROOM["x"] / GA_CELL)), int(round(ROOM["z"] / GA_CELL))
    g = np.random.default_rng(7000 + int(seed)).normal(0.0, 1.0, (max(2, nu // 60), max(2, nv // 60)))
    f = zoom(g, (nu / g.shape[0], nv / g.shape[1]), order=3)[:nu, :nv]
    return f / max(float(np.abs(f).max()), 1e-9)


def pail_half() -> float:
    """Half the side of the square with the area of the paint's surface in the tapered pail at its fill."""
    p = PAIL
    r, slope = p["r_bottom"], (p["r_top"] - p["r_bottom"]) / p["height"]
    level = 0.0
    for _ in range(50):                     # the level holding the fill in the frustum (fixed point)
        level = p["fill"] / (math.pi * (r * r + r * slope * level + (slope * level) ** 2 / 3.0))
    top = r + slope * level
    return 0.5 * math.sqrt(math.pi) * top


__all__ = ["BAND", "Crop", "GA_CELL", "JOBS", "OUTLET", "PAIL", "PATCHY", "ROOM", "Rect", "TrimLayout", "WINDOW",
           "first_coat_field", "pail_half"]
