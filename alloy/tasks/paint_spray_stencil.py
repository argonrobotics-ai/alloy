# SPDX-License-Identifier: Apache-2.0
"""spray-stencil: spray a bold "3" through a stencil onto a crate side in a spray box, an even 25-50 µm dry coat with
sharp edges, keeping the overspray in the box (Gym-Anything's painting-tasks-v1 spray-stencil; checks in
:mod:`alloy.tasks.paint_metrics`).

The crate side is a plywood panel lying flat in an open spray box. Its top face is a plane chart at 1 mm. A 0.3 mm
PET stencil lies on it with the "3" cut out (:func:`stencil_holes`, Gym-Anything's). The sheet seals the panel, as
Gym-Anything's does: its cells are a region of their own (``set_regions``), so the faces between them and the open
crate are walls. Wet paint does not creep under the sheet or climb its edge.

The task's can runs on the supply (:mod:`alloy.tasks.paint_spray_supply`). The stand-in here, :class:`Can`, lays the
same cone in closed form and stays as its cross-check (docs/notes/soft-3-painting.md). It rasters rows at a
standoff. Each control period it lays its cone on the cells it can see, the apertures and the panel beyond the sheet,
through ``set_source``. The cone is the supply's spray pattern: a
Gaussian with σ = half the cone's half angle, cut off at the cone's edge. What lands on the sheet is caught there.
The 30 % mist falls as Gym-Anything's dry dust, drifting σ = 0.25 x its fall. Whatever is outside the box when it
falls to the rim (cone rays or mist) leaves the box, so from below the rim nothing does (:func:`period_rates`). The
dust lands as dry solids on the crate's open cells (:func:`period_dust`). The film carries the paint's solids, dries
and soaks into the sealed plywood (:mod:`alloy.tasks.paint_process`), and its dry film is the solids it holds.

What the verdict judges (audit AB-gym-6; ``SCENE`` and each check's note say the same in the result). Of
Gym-Anything's five spray-stencil criteria four are judged: ``coat``, ``sharp_edges``, ``no_runs`` and ``overspray``
(from the port's own books). ``untouched`` is not judged (no marks are tracked). The crate, stencil and box are their
sizes, but the scene is easier than theirs: one fixed crate pose (theirs moves with the seed), the valve is an input
(theirs: a button a fingertip presses), a 10 s settle (theirs until the film sets, up to 1800 s), a fixed FR3.

Example::

    from alloy.tasks.paint_spray_stencil import run_standin
    print(run_standin()["succeeded"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from alloy.tasks import paint_metrics as pm
from alloy.tasks import paint_process as pp
from alloy.tasks.paint_materials import paint
from alloy.tasks.paint_scene import chart_scene

PAINT = "enamel_aerosol"
CELL = 1e-3                     # m (Gym-Anything's spray charts)
CONTROL_DT = 0.05               # s
BOX = (0.60, 0.45, 0.45)        # m: the spray box's floor and wall height (Gym-Anything's, 4 mm cardboard)
CRATE_Z = 0.016                 # m: the crate side's face above the box floor (12 mm plywood on 4 mm cardboard)


@dataclass(frozen=True)
class SprayLayout:
    crate: tuple[float, float] = (0.45, 0.30)       # m, the crate side (Gym-Anything's)
    stencil: tuple[float, float] = (0.40, 0.30)     # m, a 16 x 12 in blank of 12 mil PET, centred on the crate
    thick: float = 0.3e-3                           # m (the sheet's body; its seal is a film region)
    board_z: float = 0.75                           # m, world z of the crate's face (in the scene)

    @property
    def cells(self) -> tuple[int, int]:
        return int(round(self.crate[0] / CELL)), int(round(self.crate[1] / CELL))

    def centres(self) -> np.ndarray:
        """Cell centres [nx, ny, 2] in the crate frame (centred)."""
        nx, ny = self.cells
        x = (np.arange(nx) + 0.5) * CELL - 0.5 * self.crate[0]
        y = (np.arange(ny) + 0.5) * CELL - 0.5 * self.crate[1]
        return np.stack(np.meshgrid(x, y, indexing="ij"), -1)

    def sheet(self, c: np.ndarray) -> np.ndarray:
        """Cells under the sheet's material (not its apertures)."""
        on = (np.abs(c[..., 0]) <= 0.5 * self.stencil[0]) & (np.abs(c[..., 1]) <= 0.5 * self.stencil[1])
        return on & ~stencil_holes(c)


def stencil_holes(c: np.ndarray) -> np.ndarray:
    """Gym-Anything's bold "3": two bowls of a stroke 22 mm wide (rings 30-52 mm about (0, ±36 mm), open on the left),
    in the stencil's frame [m]."""
    x, y = c[..., 0], c[..., 1]
    out = np.zeros(x.shape, dtype=bool)
    for cy, (a0, a1) in ((0.036, (-105.0, 125.0)), (-0.036, (-125.0, 105.0))):
        r = np.hypot(x, y - cy)
        ang = np.degrees(np.arctan2(y - cy, x))
        out |= (r > 0.030) & (r < 0.052) & (ang > a0) & (ang < a1)
    return out & (np.abs(x) < 0.13) & (np.abs(y) < 0.09)


@dataclass(frozen=True)
class Can:
    """The stand-in aerosol can: Gym-Anything's estimates, against the can's data sheet (10-16 in, 25-50 µm a coat).
    2 g/s of the can's contents, 60 % of it flashing in flight, lands 0.73 mL/s of wet enamel. 340 g at 0.70 transfer
    is 25 mL dry, 0.67 m² at 38 µm; the data sheet gives 0.74-1.11 m² with 15 % loss."""

    wet_rate: float = 0.8e-3 / 1100.0       # m³/s of landed-state enamel at a full valve
    efficiency: float = 0.70                # landed share; the rest is mist (estimate: data sheet 0.85, GA ~0.6)
    half_angle: float = math.radians(12.0)
    standoff: float = 0.30                  # m above the crate face (10-16 in)
    speed: float = 0.19                     # m/s along a row: 67 µm wet a coat, two coats 34 µm dry
    rows: tuple[float, ...] = (-0.12, -0.08, -0.04, 0.0, 0.04, 0.08, 0.12)    # m, y of each row
    span: float = 0.12                      # m: each row runs x from -span to +span
    coats: int = 2
    pause: float = 10.0                     # s between coats
    substeps: int = 4                       # nozzle positions per control period

    @property
    def sigma(self) -> float:
        """σ of the cone's Gaussian on the crate [m]."""
        return self.standoff * math.tan(0.5 * self.half_angle)

    @property
    def above_rim(self) -> float:
        """How far the nozzle is above the box's rim [m] (<= 0: inside the box)."""
        return self.standoff + CRATE_Z - BOX[2]


DRIFT = 0.25                    # mist's lateral drift, σ per metre of fall (Gym-Anything's estimate)
EDGES = ("open",) * 4           # the crate's edges are free: paint that runs over them leaves (Gym-Anything's runoff)
BOOKS_REL = 1e-5                # the film's float32 sources against the books' float64 sums
OWN_BOOKS = ("judged from the port's own books: the solids that left the box (cone rays and mist outside it at the "
             "rim) plus runoff, per solids emitted; Gym-Anything sums its sinks and charts outside the box, runoff, "
             "paint on robot skins and airborne paint")
SCENE = ("the crate, stencil and spray box are Gym-Anything's sizes; easier than theirs in that the crate has one "
         "fixed pose (theirs: moved up to 2 cm and 3° by the seed), the can's valve is an input (theirs: a button a "
         "fingertip presses), the verdict follows a settle of 10 s by default (theirs: until the film sets, up to "
         "1800 s) and a fixed FR3 on a pedestal or a scripted can sprays (theirs: a mobile FR3); one start")


@lru_cache(maxsize=8)
def _unit_cone(cut: float, n_r: int = 48, n_t: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature of a Gaussian of σ 1 cut off at radius ``cut``: points [K, 2] and weights [K] summing to 1."""
    r = (np.arange(n_r) + 0.5) * cut / n_r
    t = (np.arange(n_t) + 0.5) * 2.0 * math.pi / n_t
    rr, tt = np.meshgrid(r, t, indexing="ij")
    w = (rr * np.exp(-0.5 * rr * rr)).ravel()
    return np.stack([(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()], -1), w / w.sum()


def _cone_inside(mx: float, my: float, s: float, cut: float, hx: float, hy: float) -> float:
    """The share of a cone's flux (Gaussian σ ``s``, cut off at ``cut`` σ, centred (mx, my)) inside |x| <= hx,
    |y| <= hy."""
    u, w = _unit_cone(round(cut, 6))
    x, y = mx + s * u[:, 0], my + s * u[:, 1]
    return float(w[(np.abs(x) <= hx) & (np.abs(y) <= hy)].sum())


def _gauss_rect(mx: float, my: float, s: float, hx: float, hy: float) -> float:
    """The share of an isotropic Gaussian (centre (mx, my), σ s) inside |x| <= hx, |y| <= hy."""
    def share(m: float, h: float) -> float:
        return 0.5 * (math.erf((h - m) / (math.sqrt(2.0) * s)) - math.erf((-h - m) / (math.sqrt(2.0) * s)))
    return share(mx, hx) * share(my, hy)


def schedule(can: Can) -> list[list[tuple[float, float]] | None]:
    """Per control period, the nozzle's aim points (crate frame), or None while the valve is shut."""
    out: list = []
    step = can.speed * CONTROL_DT
    for coat in range(can.coats):
        if coat:
            out += [None] * int(round(can.pause / CONTROL_DT))
        for k, y in enumerate(can.rows):
            sgn = 1.0 if (k + coat) % 2 == 0 else -1.0                  # back and forth
            n = int(math.ceil(2.0 * can.span / step - 1e-9))
            for p in range(n):
                x0 = -can.span + p * step
                xs = x0 + (np.arange(can.substeps) + 0.5) * step / can.substeps
                out.append([(float(sgn * x), float(y)) for x in xs])
    return out


@dataclass
class Books:
    """The stand-in can's solids [m³]: emitted = on_crate + on_sheet + in_box + outside."""

    emitted: float = 0.0
    on_crate: float = 0.0       # landed on the crate's open cells (the film's supply)
    on_sheet: float = 0.0       # caught by the stencil
    in_box: float = 0.0         # spray past the crate and mist, landed on the box's floor or walls (or as dust)
    outside: float = 0.0        # left the box over its rim
    dust: float = 0.0           # of in_box: mist dust laid on the crate's open cells

    def closure(self) -> float:
        """Relative imbalance of the books."""
        return abs(self.emitted - self.on_crate - self.on_sheet - self.in_box - self.outside) / max(self.emitted, 1e-30)


def period_rates(can: Can, aims: list[tuple[float, float]] | None, c: np.ndarray, seen: np.ndarray,
                 books: Books) -> np.ndarray:
    """The source [m/s] on the crate's cells ``c`` this period, booking the solids.

    The cone lands on the cells the can sees (``seen``); the sheet catches the rest over it; beyond the crate it lands
    in the box. From above the rim, the cone's rays that pass the rim outside the box, and the mist that has drifted
    outside the box by the time it falls to the rim (σ = ``DRIFT`` x the fall to the rim), leave it. From below the
    rim nothing leaves: the box's walls catch both."""
    rate = np.zeros(c.shape[:2])
    if aims is None:
        return rate
    phi, dt = paint(PAINT).volume_solids, CONTROL_DT
    s, q = can.sigma, can.wet_rate / len(aims)                         # m³/s per aim point
    cut = math.tan(can.half_angle) / math.tan(0.5 * can.half_angle)   # the cone's edge in σ (the supply's cut-off)
    norm = 2.0 * math.pi * s * s * (1.0 - math.exp(-0.5 * cut * cut))
    hx, hy, up = 0.5 * BOX[0], 0.5 * BOX[1], can.above_rim
    for mx, my in aims:
        r2 = (c[..., 0] - mx) ** 2 + (c[..., 1] - my) ** 2
        g = np.where(r2 <= (cut * s) ** 2, np.exp(-r2 / (2.0 * s * s)) / norm, 0.0)
        land, mist = q * can.efficiency * dt * phi, q * (1.0 - can.efficiency) * dt * phi
        rate += q * can.efficiency * g * seen
        crate = float((g * seen).sum()) * CELL * CELL
        sheet = float((g * ~seen).sum()) * CELL * CELL
        cone_out = 0.0 if up <= 0.0 else 1.0 - _cone_inside(mx, my, up * math.tan(0.5 * can.half_angle), cut, hx, hy)
        mist_out = rim_escape((mx, my), up)
        books.emitted += land + mist
        books.on_crate += land * crate
        books.on_sheet += land * sheet
        books.outside += land * cone_out + mist * mist_out
        books.in_box += land * (1.0 - crate - sheet - cone_out) + mist * (1.0 - mist_out)
    return rate.astype(np.float32)


def period_dust(can: Can, aims: list[tuple[float, float]] | None, c: np.ndarray, seen: np.ndarray,
                books: Books) -> np.ndarray:
    """The mist's dry dust [m of solids] laid on the crate's open cells this period: released at the aim points, the
    share that stays in the box (:func:`rim_escape`) falls the standoff drifting σ = ``DRIFT`` x the fall
    (Gym-Anything's dry fall). It is part of the books' ``in_box``; what lands on the sheet moves to ``on_sheet``."""
    dust = np.zeros(c.shape[:2])
    if aims is None:
        return dust
    mist = can.wet_rate / len(aims) * (1.0 - can.efficiency) * CONTROL_DT * paint(PAINT).volume_solids
    for mx, my in aims:
        inside = mist * (1.0 - rim_escape((mx, my), can.above_rim))
        dust += drift_dust(inside, (mx, my), c, seen, can.standoff)
        caught = float(drift_dust(inside, (mx, my), c, ~seen, can.standoff).sum()) * CELL * CELL
        books.on_sheet += caught
        books.in_box -= caught
    books.dust += float(dust.sum()) * CELL * CELL
    return dust


def rim_escape(aim: tuple[float, float], up: float) -> float:
    """The share of mist released over ``aim`` [m, crate frame] ``up`` m above the box's rim that has drifted
    outside the box (σ = ``DRIFT`` x the fall to the rim) when it falls to the rim; none from below the rim."""
    if up <= 0.0:
        return 0.0
    return 1.0 - _gauss_rect(aim[0], aim[1], DRIFT * up, 0.5 * BOX[0], 0.5 * BOX[1])


def drift_dust(mist: float, aim: tuple[float, float], c: np.ndarray, seen: np.ndarray, fall: float) -> np.ndarray:
    """``mist`` m³ of solids released over ``aim``, fallen ``fall`` drifting σ = ``DRIFT`` x the fall: the dust [m] on
    the cells ``c`` it lands on (``seen``: the sheet catches the rest over it)."""
    s = DRIFT * fall
    g = np.exp(-((c[..., 0] - aim[0]) ** 2 + (c[..., 1] - aim[1]) ** 2) / (2.0 * s * s)) / (2.0 * math.pi * s * s)
    return mist * g * seen


def _books_drift(read: dict, books: Books, ran: float = 0.0) -> float:
    """How far the film's solids stray from the books (relative): those its sources brought against ``on_crate``,
    and those it holds against ``on_crate + dust - ran`` (``ran``: what ran off its edges). Infinite (the check fails)
    when the film carries no solids."""
    if "species_supplied" not in read:                 # no solids in the film: the books cannot be checked
        return float("inf")
    supplied = float(np.asarray(read["species_supplied"])[0, :, 0].sum())
    held = float(np.asarray(read["species_mass"])[0, 0])
    laid = books.on_crate + books.dust - ran
    return max(abs(supplied - books.on_crate) / max(books.on_crate, 1e-30), abs(held - laid) / max(laid, 1e-30))


def runoff(read: dict) -> float:
    """The solids [m³] that ran off the crate's open edges with the film (Gym-Anything's runoff, a leak path)."""
    out = read.get("species_out")
    return 0.0 if out is None else float(np.asarray(out)[0, :, 0].sum())


def evaluate(layout: SprayLayout, h_end: np.ndarray, h_declared: np.ndarray, read: dict, books: Books,
             marred: np.ndarray | None = None, books_rel: float = BOOKS_REL) -> dict:
    """The spray-stencil verdict: the coat and its edges on the settled film, the wet film at declaration for runs,
    the overspray from the books plus the runoff, crate cells marred by a robot link (``marred``; None: no marks
    were tracked, and ``untouched`` is not judged), the film's ledger, and the books against the film
    (:func:`_books_drift`, within ``books_rel``). ``dry_film`` says where the dry film came from."""
    c = layout.centres()
    target = stencil_holes(c)
    dry, source = pp.dry_film(read, h_end, PAINT)
    ran = runoff(read)
    drift = _books_drift(read, books, ran)
    v = pm.verdict([*pm.spray_coat(dry, target, CELL),
                    pm.no_runs(float(h_declared[target].max()), 0.0, settled=False),
                    pm.overspray(books.outside + ran, books.emitted).differs(OWN_BOOKS),
                    pm.untouched(marred, target, CELL * CELL),
                    pm.ledger(read),
                    pm.Check("books", drift, books_rel, drift <= books_rel)], scene=SCENE)  # fmt: skip
    return {**v, "dry_film": source, "dry_target": pm.dry_summary(dry, target), "runoff": ran}


def run_standin(layout: SprayLayout | None = None, can: Can | None = None, settle: float = 10.0,
                device: str = "cpu") -> dict:
    """The stand-in spray-stencil on a crate-only scene (the plug-in harness: a static panel and a free dummy body)."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415

    layout, can = layout or SprayLayout(), can or Can()
    c = layout.centres()
    sheet = layout.sheet(c)
    flow = SurfaceFlow(charts=(PlaneChart("crate", cells=layout.cells, edges=EDGES),), inertia=False, max_depth=3e-4,
                       **pp.film_settings(PAINT, substrate="plywood_sealed"))  # fmt: skip
    carried = pp.source_species(PAINT)

    def init(sim) -> None:  # noqa: ANN001
        flow.set_regions(sheet.astype(np.int32), chart=0)                 # the seal: no film crosses the sheet's edge

    hx, hy = 0.5 * layout.crate[0], 0.5 * layout.crate[1]
    panel = ("crate", (0.0, 0.0, layout.board_z - 0.006), (0.0, 0.0, 0.0, 1.0), (hx, hy, 0.006))
    sim = build_simulation(chart_scene([panel], init, CONTROL_DT), [flow], device)
    books = Books()
    for aims in schedule(can):
        flow.set_source(period_rates(can, aims, c, ~sheet, books)[None], chart=0, species=carried)
        if aims is not None:
            pp.lay_solids(flow, period_dust(can, aims, c, ~sheet, books)[None])
        sim.step_period()
    flow.set_source(0.0)
    declared = flow.depth(0)[0].copy()
    for _ in range(int(round(settle / CONTROL_DT))):
        sim.step_period()
    return evaluate(layout, flow.depth(0)[0], declared, flow.read(), books)


__all__ = ["BOX", "Books", "CELL", "Can", "DRIFT", "EDGES", "PAINT", "SprayLayout", "drift_dust", "evaluate",
           "period_dust", "period_rates", "rim_escape", "run_standin", "runoff", "schedule", "stencil_holes"]
