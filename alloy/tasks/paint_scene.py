# SPDX-License-Identifier: Apache-2.0
"""Scenes of the painting proof tasks: the work wall and its chart, the strip, the prepainted bands, the paint tray.

The wall is a static box whose top face (local +z) faces the room (-y), local x along the wall and local y up, so the
surface-flow ``PlaneChart`` on it has i along the wall and j up. Gym-Anything's roll-strip, scaled to a fixed FR3's
reach (docs/notes/soft-3-painting.md):

- the chart is the wall between the baseboard's top (its bottom edge) and the ceiling (its top edge);
- the strip: 0.6 m wide, 1.0 m tall, between two tape lines; above it a cut-in band under the ceiling, below it one
  above the baseboard, both brushed 60 s before (wet);
- left of the strip: the strip rolled before, 120 s old (the wet edge the new strip must join);
- the wall had one coat of the new colour over grey primer (43 µm dry: one 4-mil coat), so a second coat must bring
  the dry film to hide (62 µm for naval over grey primer);
- the ceiling and the baseboard are protected and are not on the wall chart, as in Gym-Anything. Paint reaches the
  baseboard over the chart's open bottom edge (booked as spilled); the ceiling corner is a closed edge. The tools'
  transfers to either come with the robot;
- the prepainted bands carry the yield stress the paint has rebuilt at their age
  (:func:`alloy.tasks.paint_materials.aged_yield`); a new coat starts fresh;
- the tray: a Wooster R404 deep-well tray, 14.5 x 11 x 3.5 in.

Coordinates in metres; ``WallLayout`` fields are in the chart frame (x along the wall from its centre, y up from the
chart's bottom edge).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TRAY_SIZE = (0.368, 0.279, 0.089)        # Wooster R404, 14.5 x 11 x 3.5 in


@dataclass(frozen=True)
class Band:
    """A rectangle of the chart, x0 < x1 along the wall, y0 < y1 up [m]."""

    x0: float
    x1: float
    y0: float
    y1: float

    def mask(self, centres: np.ndarray) -> np.ndarray:
        x, y = centres[..., 0], centres[..., 1]
        return (x >= self.x0) & (x < self.x1) & (y >= self.y0) & (y < self.y1)


@dataclass(frozen=True)
class WallLayout:
    width: float = 1.0                      # chart (wall face) width [m]
    height: float = 1.152                   # chart height [m]: the baseboard's top to the ceiling
    bottom: float = 0.18                    # world z of the chart's bottom edge (the baseboard's top) [m]
    face_y: float = 0.0                     # world y of the wall face (the room is at y < face_y)
    cell: float = 0.004                     # m (4 mm: surface-flow's paint wall scale)
    strip: Band = Band(-0.3, 0.3, 0.076, 1.076)     # the cut-in bands fill the chart above and below it (3 in)
    paint: str = "latex_eggshell"
    colour: str = "naval"
    substrate: str = "primer_grey"
    first_coat_dft: float = 43e-6           # m, the coat already dry on the wall
    cut_in_wet: float = 110e-6              # m wet
    cut_in_age: float = 60.0                # s
    left_wet: float = 105e-6                # m wet
    left_age: float = 120.0                 # s
    tray_pos: tuple[float, float, float] = (0.45, -0.45, 0.0)
    extra: dict = field(default_factory=dict)

    #: chart edges (i-, i+, j-, j+): paint leaves only over the bottom edge, onto the baseboard
    edges: tuple[str, str, str, str] = ("wall", "wall", "open", "wall")

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            n = getattr(self, name) / self.cell
            if abs(n - round(n)) > 1e-6:
                raise ValueError(f"WallLayout.{name} {getattr(self, name)} m is not a whole number of "
                                 f"{self.cell} m cells")
        s = self.strip
        if not (-0.5 * self.width <= s.x0 < s.x1 <= 0.5 * self.width and 0.0 < s.y0 < s.y1 < self.height):
            raise ValueError(f"WallLayout.strip {s} must lie inside the {self.width} x {self.height} m chart with "
                             "room for the cut-in bands above and below it")

    @property
    def cells(self) -> tuple[int, int]:
        return int(round(self.width / self.cell)), int(round(self.height / self.cell))

    def bands(self) -> dict[str, Band]:
        """The strip and the prepainted bands, in the chart frame."""
        s, left = self.strip, -0.5 * self.width
        return {"strip": s,
                "cut_in_top": Band(s.x0, s.x1 + 0.3, s.y1, self.height),
                "cut_in_bottom": Band(s.x0, s.x1 + 0.3, 0.0, s.y0),
                "left_strip": Band(left, s.x0, 0.0, self.height)}  # fmt: skip

    def masks(self, centres: np.ndarray) -> dict[str, np.ndarray]:
        return {name: band.mask(centres) for name, band in self.bands().items()}

    def chart_centres(self, local: np.ndarray) -> np.ndarray:
        """Chart-frame cell centres [nx, ny, 2] from the plug-in's ``cell_centres`` (box-local, centred)."""
        out = np.array(local[..., :2], dtype=np.float64)
        out[..., 1] += 0.5 * self.height
        return out

    def prepaint(self, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The wet paint already on the wall [nx, ny] (the cut-in bands and the strip rolled before) and its age [s]
        (-1 where there is none)."""
        m = self.masks(centres)
        h = np.zeros(centres.shape[:2], dtype=np.float32)
        age = np.full(centres.shape[:2], -1.0)
        cut = m["cut_in_top"] | m["cut_in_bottom"]
        h[cut], age[cut] = self.cut_in_wet, self.cut_in_age
        h[m["left_strip"]], age[m["left_strip"]] = self.left_wet, self.left_age
        return h, age


def add_wall(scene, layout: WallLayout, name: str = "wall") -> str:  # noqa: ANN001 - alloy.Scene
    """The static wall box: its top face is the chart, facing -y, local y up."""
    z = layout.bottom + 0.5 * layout.height
    scene.add_box(name, size=(layout.width, layout.height, 0.02), pos=(0.0, layout.face_y + 0.01, z),
                  rpy=(np.pi / 2, 0.0, 0.0), static=True)  # fmt: skip
    return name


def add_tray(scene, layout: WallLayout, name: str = "tray") -> str:  # noqa: ANN001 - alloy.Scene
    """The paint tray's outer box, static on the floor (its well is a supply pool once the stores land)."""
    x, y, z = layout.tray_pos
    scene.add_box(name, size=TRAY_SIZE, pos=(x, y, z + 0.5 * TRAY_SIZE[2]), static=True)
    return name


def chart_scene(boxes, init, control_dt: float):  # noqa: ANN001, ANN201 - a harness TestScene
    """A light scene for runs without the robot (the plug-in harness's ``TestScene``): static boxes ``boxes`` =
    [(label, (x, y, z), (qx, qy, qz, qw), (hx, hy, hz))], a free dummy body far away (a scene needs one dynamic body),
    XPBD, one world, ``init(sim)`` after the build. Step it with ``build_simulation`` and ``step_period``."""
    import warp as wp  # noqa: PLC0415

    from alloy.plugins.harness.scene import TestScene, body_xform  # noqa: PLC0415

    def build(b) -> None:  # noqa: ANN001 - newton.ModelBuilder
        for label, pos, quat, half in boxes:
            b.add_shape_box(-1, xform=wp.transform(wp.vec3(*pos), wp.quat(*quat)), hx=half[0], hy=half[1],
                            hz=half[2], label=label)  # fmt: skip
        dummy = b.add_body(xform=body_xform(2.0, -2.0, 1.0), label="dummy")
        b.add_shape_sphere(dummy, radius=0.01)

    return TestScene(build=build, init=init, worlds=1, periods=1, control_dt=control_dt, dt=control_dt,
                     solver="xpbd", ground=False)  # fmt: skip


def wall_box(layout: WallLayout) -> tuple:
    """The wall as a ``chart_scene`` box: its top face faces -y, local y up."""
    s = float(np.sin(np.pi / 4))
    return ("wall", (0.0, layout.face_y + 0.01, layout.bottom + 0.5 * layout.height), (s, 0.0, 0.0, s),
            (0.5 * layout.width, 0.5 * layout.height, 0.01))


__all__ = ["Band", "TRAY_SIZE", "WallLayout", "add_tray", "add_wall", "chart_scene", "wall_box"]
