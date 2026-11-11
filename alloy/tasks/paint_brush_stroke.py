# SPDX-License-Identifier: Apache-2.0
"""brush-stroke: with a round brush, paint the pencilled S-stroke in red acrylic, swelling from 2 mm to 6 mm and back
to 2 mm, inside the outline (Gym-Anything's painting-tasks-v1 brush-stroke; checks in :mod:`alloy.tasks.paint_metrics`).

The paper is a plane chart at 1 mm cells, cropped to the template (0.24 x 0.10 m), on a board lying flat. Two
brushes:
- the real one, :func:`run_brush`: the round brush on the plug-in's stores (:mod:`alloy.tasks.paint_brush`), on a
  kinematic script until a robot holds it;
- the stand-in, :func:`run_standin`: as it passes each station of the S at ``SPEED``, it lays ``coat`` (wet) on the
  cells within ``width_factor`` times the outline's half width of that station (``stroke_rates``), to test the checks.

The film then settles, drying and soaking into the paper (the process stage, :mod:`alloy.tasks.paint_process`), and
:func:`evaluate` judges it on the dry film the stage reports.

What the verdict judges (audit AB-gym-6; ``SCENE`` and each check's note say the same in the result). Of
Gym-Anything's three brush-stroke criteria one is judged: ``within_line``, inside the crop only. ``coverage`` is
judged in part (the film's thickness; the wanted paint's share cannot fail with one paint) and ``untouched`` is not
judged (no marks are tracked). The scene is smaller and easier than theirs: a 0.24 x 0.10 m crop of their 0.381 x
0.279 m sheet, an acrylic of 0.65 solids by volume (theirs 0.45, so the same wet coat leaves 1.44x the dry
film against the same 5 µm gate), a 10 s settle (theirs until the film sets, up to 1800 s), a fixed FR3, one start.

Example::

    from alloy.tasks.paint_brush_stroke import run_brush
    print(run_brush()["succeeded"])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.tasks import paint_metrics as pm
from alloy.tasks import paint_process as pp
from alloy.tasks.paint_scene import chart_scene

PAINT = "acrylic_fluid"
PAPER = (0.24, 0.10)            # m, the chart: the template's bounding box with margin
CELL = 0.001                    # m (Gym-Anything's paper cells)
BOARD_Z = 0.75                  # m, the paper's surface height
SPEED = 0.05                    # m/s along the stroke
CONTROL_DT = 0.02               # s
SETTLE = 10.0                   # s
IN_CROP = ("judged inside the port's 0.24 x 0.10 m crop of the paper only (Gym-Anything's sheet: 0.381 x 0.279 m): "
           "paint beyond the crop is not seen")
SCENE = ("smaller and easier than Gym-Anything's: a 0.24 x 0.10 m crop of the paper around the template (theirs: "
         "the 0.381 x 0.279 m sheet); an acrylic of 0.65 solids by volume (theirs: 0.45, so the same wet "
         "coat leaves 1.44x the dry film against the same 5 µm gate); one paint (theirs: two on the palette); a "
         "settle of 10 s by default (theirs: until the film sets, up to 1800 s); a fixed FR3 on a pedestal or a "
         "scripted brush (theirs: a mobile FR3); one start, the seed changes nothing")


def cells() -> tuple[int, int]:
    return int(round(PAPER[0] / CELL)), int(round(PAPER[1] / CELL))


def stroke_rates(centres_xy: np.ndarray, coat: float, width_factor: float = 1.0, n: int = 400) -> list[np.ndarray]:
    """Per control period, the source [m/s]: each cell within ``width_factor`` x the half width of its nearest
    station gets ``coat`` in the period the brush passes that station."""
    pts, half = pm.s_curve(n)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    c = centres_xy.reshape(-1, 2)
    d2 = ((c[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    k = np.argmin(d2, axis=1)
    laid = np.sqrt(d2[np.arange(len(c)), k]) <= width_factor * half[k]
    period = np.floor(arc[k] / (SPEED * CONTROL_DT)).astype(int)
    shape = centres_xy.shape[:2]
    out = []
    for p in range(int(period[laid].max()) + 1 if laid.any() else 0):
        out.append(np.where(laid & (period == p), coat / CONTROL_DT, 0.0).reshape(shape).astype(np.float32))
    return out


def evaluate(centres_xy: np.ndarray, h_end: np.ndarray, read: dict, set_volume: float = 0.0,
             set_solids: float = 0.0) -> dict:
    """The brush-stroke verdict: the dry film is the process stage's ``dft`` (without the stage, ``h·φ0``, and the
    solids check fails; ``dry_film`` says which). One paint on the paper, so the pigment share is 1 by construction
    (a second colour is a second film instance) and ``coverage`` is judged on the thickness alone, which is not a
    pass of Gym-Anything's criterion; ``untouched`` is not judged (no marks are tracked)."""
    core, outside = pm.template_masks(centres_xy)
    dry, source = pp.dry_film(read, h_end, PAINT)
    coverage, line = pm.stroke(dry, core, outside, CELL * CELL)
    v = pm.verdict([coverage, line.differs(IN_CROP), pm.untouched(None, core, CELL * CELL),
                    pm.ledger(read, set_volume=set_volume), pm.solids_ledger(read, set_solids=set_solids)],
                   scene=SCENE)  # fmt: skip
    return {**v, "dry_film": source, "dry_core": pm.dry_summary(dry, core)}


def run_standin(coat: float = 40e-6, width_factor: float = 1.0, device: str = "cpu") -> dict:
    """The stand-in brush-stroke on a paper-only scene."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415

    flow = SurfaceFlow(charts=(PlaneChart("paper", cells=cells()),), inertia=False, max_depth=4e-4,
                       **pp.film_settings(PAINT, substrate="paper"))  # fmt: skip
    carried = pp.source_species(PAINT)
    box: dict = {}

    def init(sim) -> None:  # noqa: ANN001
        c = np.asarray(flow.cell_centres(0))[..., :2].astype(np.float64)
        box.update(centres=c, rates=stroke_rates(c, coat, width_factor))

    paper = ("paper", (0.0, 0.0, BOARD_Z - 0.005), (0.0, 0.0, 0.0, 1.0), (0.5 * PAPER[0], 0.5 * PAPER[1], 0.005))
    sim = build_simulation(chart_scene([paper], init, CONTROL_DT), [flow], device)
    for r in box["rates"]:
        flow.set_source(r[None], chart=0, species=carried)
        sim.step_period()
    flow.set_source(0.0)
    for _ in range(int(round(SETTLE / CONTROL_DT))):
        sim.step_period()
    return evaluate(box["centres"], flow.depth(0)[0], flow.read())


WELL_AT = (0.0, 0.10, BOARD_Z - 0.01)       # m: the palette's red well beside the paper, its floor 1 cm down


@dataclass(frozen=True)
class BrushStep:
    """One control period of the brush's script: the tip's pose, and whether it is painting the S."""

    pose: object                    # alloy.tasks.paint_brush.BrushPose
    stroke: bool = False


def brush_script(loaded: bool = False) -> list[BrushStep]:
    """The brush's script, one pose a control period: dip the tip in the palette's well and hold it over the well while
    the load past the held share drips back (not when ``loaded``), lower it onto the S's start, paint the S
    (:func:`alloy.tasks.paint_brush.stroke_pose`: the press for the template's width), lift."""
    from alloy.tasks import paint_brush as pb  # noqa: PLC0415

    air = ([] if loaded else pb.load_poses(WELL_AT)) + pb.approach_poses(WELL_AT, BOARD_Z)
    paint = [pb.stroke_pose(point, tangent, width, 0.0, BOARD_Z) for point, tangent, width in pb.stroke_points()]
    return ([BrushStep(p) for p in air] + [BrushStep(p, True) for p in paint] +
            [BrushStep(p) for p in pb.lift_poses(BOARD_Z)])  # fmt: skip


class KinematicBrush:
    """:func:`run_brush`'s default driver: the brush a kinematic body put at each pose of :func:`brush_script`. A
    driver gives the scene, the body the tip rides (``tool``), its ``steps``, how it applies one, and a report."""

    tool = "brush"

    def __init__(self, loaded: bool = False) -> None:
        self.steps = brush_script(loaded)
        self.body = -1

    @staticmethod
    def scene(init):  # noqa: ANN001, ANN205
        from alloy.tasks import paint_brush as pb  # noqa: PLC0415

        return pb.brush_scene(init, BOARD_Z, PAPER, (WELL_AT[0], WELL_AT[1], WELL_AT[2] + 0.05))

    def bind(self, sim) -> None:  # noqa: ANN001
        self.body = list(sim.model.body_label).index(self.tool)

    def apply(self, sim, step: BrushStep | None) -> None:  # noqa: ANN001
        """Put the brush at ``step``'s pose (``None``: leave it be)."""
        from alloy.tasks import paint_brush as pb  # noqa: PLC0415

        if step is not None:
            pb.place(sim, self.body, step.pose)

    @staticmethod
    def report() -> dict:
        return {}


def run_brush(device: str = "cpu", settle: float = SETTLE, loaded: bool = False, driver=None) -> dict:  # noqa: ANN001
    """brush-stroke with the real round brush (the stores, :mod:`alloy.tasks.paint_brush`): ``driver`` (default
    :class:`KinematicBrush`) moves the tip through :func:`brush_script`; the film then settles ``settle`` s and
    :func:`evaluate` judges it. ``loaded``: start with the store at its held share over the well, skipping the dip and
    the hold. Besides the verdict: the load left and the range of the film the store metered while painting."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415
    from alloy.tasks import paint_brush as pb  # noqa: PLC0415

    driver = driver or KinematicBrush(loaded)
    flow = SurfaceFlow(charts=(PlaneChart("paper", cells=cells()),), inertia=False, max_depth=4e-4,
                       tools=(pb.brush(PAINT, initial=pb.HELD if loaded else 0.0, body=driver.tool),),
                       trays=(pb.palette_well(WELL_AT),), tool_speed=pb.SPEED,
                       **pp.film_settings(PAINT, substrate="paper"))  # fmt: skip
    box: dict = {}

    def init(sim) -> None:  # noqa: ANN001
        box["centres"] = np.asarray(flow.cell_centres(0))[..., :2].astype(np.float64)

    sim = build_simulation(driver.scene(init), [flow], device)
    driver.bind(sim)
    h_m = []
    for step in driver.steps:
        driver.apply(sim, step)
        sim.step_period()
        if step.stroke:
            h_m.append(float(np.asarray(flow.read()["tool_hm"]).ravel()[0]))
    for _ in range(int(round(settle / pb.DT))):
        driver.apply(sim, None)
        sim.step_period()
    read = flow.read()
    v = evaluate(box["centres"], flow.depth(0)[0], read)
    h_m = np.array(h_m)
    painting = h_m[h_m > 0.0]
    return {**v, "load_left": float(read["tool_load"][0].sum()), **driver.report(),
            "metered_um": (float(painting.min()) * 1e6, float(painting.max()) * 1e6) if painting.size else (0.0, 0.0)}


__all__ = ["BOARD_Z", "CELL", "PAINT", "PAPER", "SPEED", "WELL_AT", "BrushStep", "KinematicBrush", "brush_script",
           "cells", "evaluate", "run_brush", "run_standin", "stroke_rates"]
