# SPDX-License-Identifier: Apache-2.0
"""swatch-card: paint the pencilled 50 x 50 mm square on the watercolour paper to full hide with the blue acrylic, using
the round brush (Gym-Anything's painting-tasks-v1 swatch-card, ``PT/layout.py:115-116``).

The paper is a plane chart at Gym-Anything's 1 mm, an 81 mm crop centred on the square, on the drawing board. Its cell
centres sit on whole millimetres, as on Gym-Anything's paper (``PT/task.py:99-102``), so the edge cells straddle the
line: the square is the 51 x 51 cells within 25 mm (Gym-Anything's float rounding drops its y = -25 mm row: 51 x 50),
and "outside" starts at the 27 mm centres. The brush is SOFT-3's round #8 on the surface-flow stores
(:mod:`alloy.tasks.paint_brush`): a tip whose pressed band widens with the press, loaded from the palette's blue well
0.5 m away, the red well beside it (Gym-Anything's palette and wells 0 and 1, ``PT/layout.py:205-214``,
``PT/tools.py:218-219``), kinematic until a robot holds it. Each well's paint carries its pigment (WP15's tray
species): the film carries the paint's solids and, of them, the blue and the red. The script fills the
square with parallel strokes along x, ``BAND`` wide and overlapping, their outer edges half a cell past the line (the
edge cells' far side), reloading every ``per_load`` strokes; each period it places the brush at SOFT-3's
``stroke_pose`` for the band at the store's load. It lands straight down one period's travel short of each stroke's
start (SOFT-3's metering tip lays only while it slides, under its whole contact) and lifts straight up. The film
then settles
``SETTLE`` s, drying and soaking into the paper (the process stage), and :func:`evaluate` judges the dry film.

Checks (Gym-Anything's ``PT/task.py:378-395`` with the painting-tasks-v1 contract, ``PT/scoring.py:29-40``):
- ``coverage``: >= 95 % of the square at >= 15 µm dry (``stencil_hide_dft``, ``PT/params.py:239``), with the blue's
  share of the dry film >= 0.95 (``share``; None: one paint on the paper, share 1);
- ``within_line``: paint (> 3 µm dry) more than 1 mm outside the square, plus the brush's drips over the paper outside
  the line as Gym-Anything's sinks count them (their volume spread 100 µm thick), on at most 0.5 cm²;
- ``untouched``: not judged (None): no robot here, and no marks on the wet film are tracked;
- ``books`` (the film's, the brush's and the wells' ledgers), ``solids_ledger`` and ``pigment_books`` (each pigment
  through the film, the brush's bins and the wells), to 1e-12: the port's own checks, not Gym-Anything criteria.

Of Gym-Anything's three criteria two are judged, each by another measure (audit AB-gym-6; the verdict's notes and
``SCENE`` say the same): ``coverage`` on the total dry film with the blue's share, where theirs asks 15 µm of the
blue itself (looser by up to 5 %), and ``within_line`` inside the 81 mm crop only.

Not as Gym-Anything (disclosed):
- SOFT-3's fluid acrylic is 0.65 solids by volume, Gym-Anything's 0.45 (``PT/params.py:52``): the same wet coat
  leaves 1.44x the dry film against the same 15 µm gate;
- SOFT-3's brush meters (WP15's ``Store(meter=C)``, Gym-Anything's law) and lays under its whole contact, past its
  centre: the strokes end :func:`end_inset` short of the band's ends (``paint_brush.end_inset(BAND)``), and ``BAND``
  stays under the tip's widest band (``paint_brush.BAND_MAX``);
- the settle is ``SETTLE`` s; Gym-Anything settles until every wet cell has set (up to 30 min).

Gym-Anything's honest swatch-card episode passes and its ``corner`` episode fails coverage (its report's variant
table, ``[internal note]:630-640``); :class:`Plan` has both, and an overrun.

Example::

    python -m alloy.tasks.swatch_card --out swatch.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass

import numpy as np

from alloy.tasks import paint_metrics as pm
from alloy.tasks import paint_process as pp
from alloy.tasks import trim_sash as ts

PAINT = "acrylic_fluid"         # Gym-Anything's ultramarine artist acrylic (the colour does not enter the checks)
CELL = 0.001                    # m (Gym-Anything's paper cell)
HALF = 25                       # cells: the square's half side, 25 mm
MARGIN = 1                      # cells: outside starts beyond 1 mm past the line (Gym-Anything's overrun)
CROP = (0.081, 0.081)           # m, the paper chart: 81 cells, centres on whole millimetres
BOARD_Z = 0.75                  # m, the paper's surface (as brush-stroke)
PALETTE = (-0.50, -0.08)        # m: the palette's centre from the square's, as Gym-Anything's (PT/layout.py:211-213)
WELLS = {"blue": (-0.07, 0.035), "red": (0.0, 0.035)}    # its wells 0 and 1 in the palette (PT/tools.py:218)
SPECIES = ("solids", "blue", "red")                      # the film's: dry solids, and of them each pigment
FILL_DFT = 15e-6                # m dry: a filled, near-opaque coat (Gym-Anything's stencil_hide_dft)
SINK_FILM = pm.SINK_FILM        # m: a drip counts as its volume spread this thick
BAND = 0.006                    # m: each stroke's band
OVERLAP = 0.001                 # m between neighbouring bands
EDGE = (HALF + 0.5) * CELL      # m: the bands' outer edges and the strokes' ends: the edge cells' far side
SETTLE = 10.0                   # s, as brush-stroke
TOTAL_FILM = ("judged by another measure: the total dry film >= 15 µm with the blue's share >= 0.95; Gym-Anything "
              "asks 15 µm of the blue itself with that share, so the port is looser by up to 5 %")
IN_CROP = ("judged inside the port's 81 mm crop of the paper only (Gym-Anything's sheet: 0.381 x 0.279 m), with the "
           "brush's drips past the line added: paint beyond the crop is not seen")
SCENE = ("smaller and easier than Gym-Anything's: an 81 mm crop of the paper round the square (theirs: the 0.381 x "
         "0.279 m sheet); an acrylic of 0.65 solids by volume (theirs: 0.45, so the same wet coat leaves "
         "1.44x the dry film against the same 15 µm gate); a settle of 10 s by default (theirs: until the film sets, "
         "up to 1800 s); a free-moving brush on a script (theirs: a mobile FR3 holds it); the palette at one place, "
         "one start")
LIFT = 0.03                     # m over the paper between strokes
TRAVEL = 0.3                    # m/s off the paper
LEDGER_REL = 1e-12              # the books' drift relative to the paint picked up (WP15)


@dataclass(frozen=True)
class Plan:
    """``per_load`` strokes a load; ``corner`` [m]: the last stroke stops this short of the square's end (Gym-Anything's
    corner episode); ``overrun`` [m]: the outer strokes sit this far out past the line; ``well``: the well the brush
    loads from ("red" is the wrong paint)."""

    per_load: int = 3
    corner: float = 0.0
    overrun: float = 0.0
    well: str = "blue"


def well_at(colour: str) -> tuple[float, float, float]:
    """The floor's centre of the palette's ``colour`` well [m], 1 cm under the paper."""
    dx, dy = WELLS[colour]
    return PALETTE[0] + dx, PALETTE[1] + dy, BOARD_Z - 0.01


def wells():  # noqa: ANN201 - list[surface_flow.stores.Tray]
    """The palette's wells, blue then red, each with SOFT-3's well size and fill, its paint carrying its pigment."""
    from alloy.plugins.surface_flow.stores import Tray  # noqa: PLC0415
    from alloy.tasks import paint_brush as pb  # noqa: PLC0415
    from alloy.tasks.paint_materials import paint  # noqa: PLC0415

    phi = paint(PAINT).volume_solids
    return [Tray(center=well_at(c), half=pb.WELL, volume=pb.WELL_PAINT, species={"solids": phi, c: phi})
            for c in WELLS]


def cells() -> tuple[int, int]:
    return int(round(CROP[0] / CELL)), int(round(CROP[1] / CELL))


def masks(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The square (``PT/task.py:189``) and the paper more than 1 mm outside it, by each cell's index from the square's
    centre (cell centres ``xy`` [nx, ny, 2] on whole millimetres)."""
    ix, iy = np.abs(np.rint(xy[..., 0] / CELL)), np.abs(np.rint(xy[..., 1] / CELL))
    return (ix <= HALF) & (iy <= HALF), (ix > HALF + MARGIN) | (iy > HALF + MARGIN)


def max_band() -> float:
    """The widest band SOFT-3's tip lays [m]."""
    from alloy.tasks import paint_brush as pb  # noqa: PLC0415

    return float(pb.BAND_MAX)


def end_inset() -> float:
    """How far [m] a stroke ends short of the band's ends: SOFT-3's metering tip lays under its whole contact, past its
    centre's ends."""
    from alloy.tasks import paint_brush as pb  # noqa: PLC0415

    return float(pb.end_inset(BAND))


def strokes(plan: Plan) -> list[tuple[np.ndarray, np.ndarray]]:
    """(start, end) of each stroke's centre line, paper frame [m]: rows along x, back and forth, from the square's
    -y side; the bands' outer edges and the strokes' ends at ``EDGE`` (``plan.overrun`` further out)."""
    edge = EDGE - 0.5 * BAND + plan.overrun
    end = EDGE - end_inset()
    n = int(math.ceil((2.0 * edge) / (BAND - OVERLAP) - 1e-9)) + 1
    out = []
    for k, y in enumerate(np.linspace(-edge, edge, n)):
        a, b = np.array([-end, y]), np.array([end, y])
        if k % 2:
            a, b = b, a
        if k == n - 1 and plan.corner > 0.0:
            b = b - plan.corner * (b - a) / np.linalg.norm(b - a)
        out.append((a, b))
    return out


def stroke_points(a: np.ndarray, b: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray, float]]:
    """(point, tangent, width) along a stroke every ``SPEED·DT`` of SOFT-3's brush, ``b`` included."""
    from alloy.tasks import paint_brush as pb  # noqa: PLC0415

    d = b - a
    n = max(1, int(math.ceil(float(np.linalg.norm(d)) / (pb.SPEED * pb.DT) - 1e-9)))
    for k in range(1, n + 1):
        yield a + d * k / n, d, BAND


@dataclass
class Drips:
    """The brush's drips [m³] by where it was: over the well, over the paper inside the line's 1 mm margin, over the
    paper outside it (lands as paint past the line), elsewhere (the table)."""

    into_well: float = 0.0
    on_square: float = 0.0
    past_line: float = 0.0
    elsewhere: float = 0.0

    def add(self, volume: float, xy: np.ndarray, well_half: float) -> None:
        reach = (HALF + MARGIN + 0.5) * CELL
        if any(abs(xy[0] - w[0]) <= well_half and abs(xy[1] - w[1]) <= well_half for w in map(well_at, WELLS)):
            self.into_well += volume
        elif max(abs(xy[0]), abs(xy[1])) <= reach:
            self.on_square += volume
        elif max(abs(xy[0]), abs(xy[1])) <= 0.5 * CROP[0]:
            self.past_line += volume
        else:
            self.elsewhere += volume


def pigment_books(read: dict, wells0: np.ndarray, world: int = 0, rel: float = LEDGER_REL) -> pm.Check:
    """Each species closes through the film (held + taken off - laid = 0), the brush's bins and the wells (``wells0``
    [K], their start), relative to what the brush picked up. The bins' ledger is their start species: 0 because the
    brush starts empty (``run`` refuses a loaded one)."""
    picked = np.asarray(read["tool_species_picked"])[world].sum(axis=0)[:len(wells0)]
    gaps = [np.asarray(read["species_ledger"])[world], np.asarray(read["store_species_ledger"])[world],
            np.asarray(read["tray_species_ledger"])[world] - wells0]  # fmt: skip
    drift = float(max(np.abs(g[:len(wells0)]).max() for g in gaps) / max(float(picked.max()), 1e-15))
    return pm.Check("pigment_books", drift, rel, drift <= rel)


def evaluate(xy: np.ndarray, read: dict, well0: float, past_line_drips: float = 0.0, share: np.ndarray | None = None,
             world: int = 0, wells0: np.ndarray | None = None) -> dict:
    """The swatch-card verdict of one world from the film's ``read()`` after the settle (cell centres ``xy``; the
    wells' start volume ``well0`` [m³] and species ``wells0`` [K] (None: no pigment books); drips over the paper past
    the line [m³]; the blue's ``share`` of the dry film [nx, ny], None for one paint)."""
    square, outside = masks(xy)
    dry = pm.chart_field(read, "dft", xy.shape[:2], world)
    if dry is None:
        raise RuntimeError("swatch-card needs the process stage's dry film (read()['dft'])")
    coverage, line = pm.stroke(dry, square, outside, CELL * CELL, share=share, dft_min=FILL_DFT)
    over = line.value + max(past_line_drips, 0.0) / SINK_FILM            # drips past the line, GA's sink rule
    checks = [coverage if share is None else coverage.differs(TOTAL_FILM),
              pm.Check("within_line", over, line.limit, over <= line.limit, IN_CROP),
              pm.untouched(None, square, CELL * CELL),
              ts.brush_books(read, well0, world, rel=LEDGER_REL),
              pm.solids_ledger(read, world=world, rel=LEDGER_REL)]  # fmt: skip
    if wells0 is not None:
        checks.append(pigment_books(read, np.asarray(wells0, np.float64), world))
    return {**pm.verdict(checks, scene=SCENE), "dry_square": pm.dry_summary(dry, square)}


def glide(a, b, speed: float, quat):  # noqa: ANN001, ANN201 - list[paint_brush.BrushPose]
    """SOFT-3's brush poses from ``a`` to ``b`` at ``speed``, one a period, at orientation ``quat``."""
    from alloy.tasks import paint_brush as pb  # noqa: PLC0415

    return [pb.BrushPose(p.pos, p.quat, p.vel) for p in ts.line(a, b, speed, quat)]


def run(plan: Plan | None = None, settle: float = SETTLE, device: str = "cpu") -> dict:
    """Load, fill the square stroke by stroke, settle ``settle`` s, judge (plus the drips, the load left, the wall
    time)."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415
    from alloy.tasks import paint_brush as pb  # noqa: PLC0415

    plan = plan or Plan()
    trays, at_well = wells(), well_at(plan.well)
    tool = pb.brush(PAINT)
    if tool.store.initial != 0.0:   # pigment_books takes the bins' ledger as 0: the brush must start empty (WP15)
        raise ValueError(f"swatch-card: the brush must start empty of paint, got initial={tool.store.initial}")
    settings = {**pp.film_settings(PAINT, substrate="paper"), "species": SPECIES}
    flow = SurfaceFlow(charts=(PlaneChart("paper", cells=cells()),), inertia=False, max_depth=4e-4,
                       tools=(tool,), trays=tuple(trays), tool_speed=0.1, **settings)  # fmt: skip
    box: dict = {}

    def init(sim) -> None:  # noqa: ANN001
        box["xy"] = np.asarray(flow.cell_centres(0))[..., :2].astype(np.float64)

    t0 = time.perf_counter()
    above = (at_well[0], at_well[1], at_well[2] + 0.05)
    sim = build_simulation(pb.brush_scene(init, BOARD_Z, CROP, above), [flow], device)
    body = list(sim.model.body_label).index("brush")
    drips, seen = Drips(), [0.0]

    def put(pose) -> None:  # noqa: ANN001 - paint_brush.BrushPose
        pb.place(sim, body, pose)
        sim.step_period()
        now = float(flow.tool_dripped.numpy()[0].sum())
        drips.add(now - seen[0], pose.pos, trays[0].half[0])
        seen[0] = now

    def load() -> float:
        return float(flow.tool_load.numpy()[0].sum())

    at = np.asarray(above, np.float64)
    for k, (a, b) in enumerate(strokes(plan)):
        if k % plan.per_load == 0:
            for pose in [*glide(at, above, TRAVEL, pb.stroke_pose(a, b - a, BAND, load(), BOARD_Z).quat),
                         *pb.load_poses(at_well)]:  # fmt: skip
                put(pose)
            at = np.array([at_well[0], at_well[1], at_well[2] + 0.03])
        a0 = a - (b - a) / np.linalg.norm(b - a) * pb.SPEED * pb.DT    # a period short: the tip meters as it slides
        last = pb.stroke_pose(a0, b - a, BAND, load(), BOARD_Z)       # SOFT-3's stable entry point
        over = last.pos + (0.0, 0.0, LIFT)
        for pose in [*glide(at, over, TRAVEL, last.quat), *glide(over, last.pos, 0.02, last.quat)]:
            put(pose)
        for point, tangent, width in stroke_points(a0, b):
            last = pb.stroke_pose(point, tangent, width, load(), BOARD_Z)
            put(last)
        lift = glide(last.pos, last.pos + (0.0, 0.0, LIFT), 0.05, last.quat)     # straight up
        for pose in lift:
            put(pose)
        at = lift[-1].pos
    for _ in range(int(round(settle / pb.DT))):
        sim.step_period()
    read = flow.read()
    drips.elsewhere += float(read["tool_dripped"][0].sum()) - seen[0]
    mass = np.asarray(read["species"])[0].reshape(*cells(), -1)              # [nx, ny, K] per area
    solids, blue = mass[..., SPECIES.index("solids")], mass[..., SPECIES.index("blue")]
    share = np.divide(blue, solids, out=np.zeros_like(solids), where=solids > 0.0)
    wells0 = np.array([sum(dict(t.species).get(s, 0.0) * t.volume for t in trays) for s in SPECIES])
    return {**evaluate(box["xy"], read, len(trays) * pb.WELL_PAINT, past_line_drips=drips.past_line, share=share,
                       wells0=wells0),
            "drips_mm3": {k: round(v * 1e9, 4) for k, v in asdict(drips).items()},
            "load_left_mL": round(float(read["tool_load"][0].sum()) * 1e6, 5),
            "wall_s": round(time.perf_counter() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--per-load", type=int, default=3)
    p.add_argument("--corner", type=float, default=0.0)
    p.add_argument("--overrun", type=float, default=0.0)
    p.add_argument("--well", default="blue", choices=tuple(WELLS))
    p.add_argument("--settle", type=float, default=SETTLE)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    v = run(Plan(a.per_load, a.corner, a.overrun, a.well), a.settle, a.device)
    text = json.dumps(v, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    print(text)
    return 0 if v["succeeded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["BAND", "CROP", "Drips", "EDGE", "FILL_DFT", "HALF", "LEDGER_REL", "PAINT", "Plan", "cells",
           "end_inset", "evaluate", "masks", "max_band", "pigment_books", "run", "strokes", "well_at", "wells"]
