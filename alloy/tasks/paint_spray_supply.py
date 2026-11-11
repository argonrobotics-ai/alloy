# SPDX-License-Identifier: Apache-2.0
"""spray-stencil on the surface-flow supply: the aerosol can is a spray nozzle on a kinematic body, fed by the can on
the fluid network. The stencil is a mesh the spray cannot pass. The box's walls and floor catch what misses the crate.

The supply lays each spray ray's footprint on the crate cells its nozzle sees, and books the share the stencil hides
as caught by it (``supply_caught_by``). Rays that miss the crate land on the box. The (1 - efficiency) share is mist
(``supply_mist``). The checks, the seal and the film are the stand-in's (:mod:`alloy.tasks.paint_spray_stencil`): the
same crate, sheet, "3", can motion and verdict. Only the paint's way to the crate changes.

- **The can on the network.** A gas node at the propellant's pressure expels the enamel through the valve
  (``expel``). The propellant is liquefied, so it boils to hold its vapour pressure as the liquid leaves: a linear
  edge from a reservoir at that pressure stands in for the boiling. The valve's area gives the stand-in's wet rate.
  The input ``valve`` (0 or 1) opens and shuts it.
- **The nozzle.** A spray of ``RAYS`` rays over the stand-in's cone (σ = a quarter of the full angle, the stand-in's
  Gaussian), carrying the enamel's solids at φ0. It moves with the can's body, placed at each aim point of the
  stand-in's schedule (one aim a sub-period).
- **The stencil.** The sheet's top face as a static mesh, 0.3 mm over the crate, its edges on the cells' edges. A cell
  is hidden when the segment from the nozzle to its centre crosses the sheet.
- **The mist.** Each sub-period's new mist falls from the nozzle as Gym-Anything's dry dust (σ = 0.25 x the fall,
  :func:`~alloy.tasks.paint_spray_stencil.drift_dust`) onto the crate's open cells. With the nozzle above the box's
  rim, the share outside the rim when it falls to it leaves the box.

Example::

    from alloy.tasks.paint_spray_supply import run_supply
    print(run_supply()["succeeded"])
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from alloy.tasks import paint_process as pp
from alloy.tasks.paint_materials import film_material, paint
from alloy.tasks.paint_spray_stencil import (BOX, CELL, CONTROL_DT, CRATE_Z, EDGES, PAINT, Books, Can, SprayLayout,
                                             drift_dust, evaluate, rim_escape, schedule)

RAYS = 128                  # spray rays: 10 mm apart at 0.3 m, footprints 1.5 spacings across
PRESSURE = 4.0e5            # Pa abs: the propellant's vapour pressure at 20 °C (ESTIMATE: aerosol paint, ~3 bar gauge)
P_ATM = 101325.0            # Pa
CAPACITY = 1.24e-4          # m³ of enamel in a 340 g can (40 % of the contents at 1100 kg/m³; the rest flashes)
HEADSPACE = 1.0e-4          # m³ of propellant gas over the liquid
BOILING = 1.0e-2            # m³/s: the boiling's conductance (29 Pa below the vapour pressure at a full valve)
BOOKS_REL = 1e-7            # supplied solids match to 1e-15; the dust laid through the species input rounds (7e-9)
WALL = 0.004                # m: the box's cardboard


def valve_area(can: Can) -> float:
    """The valve's Cd·A [m²] that expels the can's wet rate at its vapour pressure."""
    return can.wet_rate / math.sqrt(2.0 * (PRESSURE - P_ATM) / film_material(PAINT).density)


def can_network(can: Can) -> dict:
    """The can as network data: the propellant's gas node, the valve (``expel``) and the boiling (a linear edge from
    the vapour pressure)."""
    return {"nodes": [{"name": "can", "medium": "gas", "volume": HEADSPACE, "pressure": PRESSURE}],
            "edges": [{"name": "valve", "kind": "expel", "a": "can", "cd_area": valve_area(can),
                       "density": film_material(PAINT).density, "dp_lam": 1.0, "capacity": CAPACITY},
                      {"name": "boiling", "kind": "linear", "a": "can", "reservoir": PRESSURE,
                       "conductance": BOILING}]}  # fmt: skip


def can_nozzle(can: Can, rays: int = RAYS, body: str = "can"):  # noqa: ANN201 - surface_flow.Nozzle
    """The can's spray nozzle on the body ``body`` (labelled), along its -z (down when the can is upright)."""
    from alloy.plugins.surface_flow.supply import Nozzle  # noqa: PLC0415

    m = film_material(PAINT)
    return Nozzle("can", edge="valve", body=body, axis=(0.0, 0.0, -1.0), pattern="spray", angle=2.0 * can.half_angle,
                  rays=rays, efficiency=can.efficiency, exit_area=valve_area(can), density=m.density,
                  viscosity=m.viscosity / m.density, surface_tension=m.surface_tension,
                  species=pp.source_species(PAINT))  # fmt: skip


def sheet_quads(layout: SprayLayout) -> list[tuple[float, float, float, float]]:
    """The sheet as rectangles (x0, x1, y0, y1) in the crate frame, on the cells' edges: each column's runs of sheet
    cells, merged over neighbouring columns with the same runs."""
    sheet = layout.sheet(layout.centres())
    x0, y0 = -0.5 * layout.crate[0], -0.5 * layout.crate[1]
    out: list = []
    runs_prev, start = None, 0
    for i in range(sheet.shape[0] + 1):
        runs = None
        if i < sheet.shape[0]:
            d = np.diff(np.concatenate([[0], sheet[i].astype(np.int8), [0]]))
            runs = tuple(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))
        if runs == runs_prev:
            continue
        for a, b in runs_prev or ():
            out.append((x0 + start * CELL, x0 + i * CELL, y0 + a * CELL, y0 + b * CELL))
        runs_prev, start = runs, i
    return out


def stencil_mesh(layout: SprayLayout):  # noqa: ANN201 - newton.Mesh
    """The sheet's top face, ``layout.thick`` over the crate's face (mesh frame: the face's centre)."""
    from alloy._vendor import newton  # noqa: PLC0415

    verts, tris = [], []
    for xa, xb, ya, yb in sheet_quads(layout):
        k = len(verts)
        verts += [(xa, ya, layout.thick), (xb, ya, layout.thick), (xb, yb, layout.thick), (xa, yb, layout.thick)]
        tris += [k, k + 1, k + 2, k, k + 2, k + 3]
    return newton.Mesh(np.asarray(verts, np.float32), np.asarray(tris, np.int32), compute_inertia=False,
                       is_solid=False)  # fmt: skip


def box_shapes(layout: SprayLayout) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """The spray box's floor and walls as (label, centre, half extents): 4 mm cardboard, ``BOX`` inside, the crate on
    the floor."""
    hx, hy, floor = 0.5 * BOX[0], 0.5 * BOX[1], layout.board_z - CRATE_Z
    zc, hz = floor + 0.5 * BOX[2], 0.5 * BOX[2]
    return [("box_floor", (0.0, 0.0, floor + 0.5 * WALL), (hx + WALL, hy + WALL, 0.5 * WALL)),
            ("box_wall_x0", (-hx - 0.5 * WALL, 0.0, zc), (0.5 * WALL, hy + WALL, hz)),
            ("box_wall_x1", (hx + 0.5 * WALL, 0.0, zc), (0.5 * WALL, hy + WALL, hz)),
            ("box_wall_y0", (0.0, -hy - 0.5 * WALL, zc), (hx, 0.5 * WALL, hz)),
            ("box_wall_y1", (0.0, hy + 0.5 * WALL, zc), (hx, 0.5 * WALL, hz))]  # fmt: skip


def spray_scene(layout: SprayLayout, can: Can, init, dt: float):  # noqa: ANN001, ANN201 - a harness TestScene
    """The crate (the chart's box), the stencil mesh, the box and the kinematic can (no robot), XPBD, one world,
    control period ``dt``."""
    from alloy._vendor import newton  # noqa: PLC0415

    from alloy.plugins.harness.scene import TestScene, body_xform  # noqa: PLC0415

    hx, hy = 0.5 * layout.crate[0], 0.5 * layout.crate[1]

    def build(b) -> None:  # noqa: ANN001 - newton.ModelBuilder
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, layout.board_z - 0.006), hx=hx, hy=hy, hz=0.006, label="crate")
        b.add_shape_mesh(-1, xform=body_xform(0.0, 0.0, layout.board_z), mesh=stencil_mesh(layout), label="stencil")
        for label, pos, half in box_shapes(layout):
            b.add_shape_box(-1, xform=body_xform(*pos), hx=half[0], hy=half[1], hz=half[2], label=label)
        body = b.add_body(xform=body_xform(0.0, 0.0, layout.board_z + can.standoff), is_kinematic=True, label="can")
        b.add_shape_cylinder(body, radius=0.033, half_height=0.08, xform=body_xform(0.0, 0.0, 0.09),
                             cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False))  # fmt: skip
        dummy = b.add_body(xform=body_xform(2.0, -2.0, 1.0), label="dummy")
        b.add_shape_sphere(dummy, radius=0.01)

    return TestScene(build=build, init=init, worlds=1, periods=1, control_dt=dt, dt=dt, solver="xpbd", ground=False)


def place_can(sim, body: int, pos: tuple[float, float, float]) -> None:  # noqa: ANN001 - alloy Simulation
    """Put the kinematic can at ``pos``, upright and still, for the next period."""
    q = sim.state.body_q.numpy().copy()
    q[body] = (*pos, 0.0, 0.0, 0.0, 1.0)
    sim.state.body_q.assign(q)
    qd = sim.state.body_qd.numpy().copy()
    qd[body] = 0.0
    sim.state.body_qd.assign(qd)


class KinematicCan:
    """:func:`run_supply`'s default driver: the can a kinematic body, put at each aim point of the stand-in's schedule
    (one a sub-period; ``None`` while the valve is shut and the can waits). A driver gives the scene, the body the
    nozzle rides (``can``), its ``steps``, the valve's state and the nozzle's place for each step, and a report."""

    can = "can"

    def __init__(self, layout: SprayLayout, can: Can) -> None:
        self.layout, self.spec, self.last = layout, can, (0.0, 0.0)
        self.steps = [aim for aims in schedule(can) for aim in (aims or [None] * can.substeps)]
        self.body = -1

    def scene(self, init, dt: float):  # noqa: ANN001, ANN201
        return spray_scene(self.layout, self.spec, init, dt)

    def bind(self, sim) -> None:  # noqa: ANN001
        self.body = list(sim.model.body_label).index(self.can)

    @staticmethod
    def valve(step) -> bool:  # noqa: ANN001
        return step is not None

    def apply(self, sim, step) -> None:  # noqa: ANN001
        if step is not None:
            place_can(sim, self.body, (step[0], step[1], self.layout.board_z + self.spec.standoff))
            self.last = step

    def nozzle_xy(self, sim) -> tuple[float, float]:  # noqa: ANN001, ARG002
        return self.last

    @staticmethod
    def report() -> dict:
        return {}


@dataclass
class Mist:
    """The mist's solids [m³] as the task lays it: ``inside`` the box (``dust`` on the crate's open cells and
    ``sheet`` on the stencil among it), ``outside`` over the rim."""

    dust: float = 0.0
    sheet: float = 0.0
    inside: float = 0.0
    outside: float = 0.0

    def fall(self, solids: float, aim: tuple[float, float], c: np.ndarray, seen: np.ndarray, can: Can) -> np.ndarray:
        """``solids`` of mist released over ``aim``: the share outside the rim leaves (:func:`rim_escape`); the rest
        falls in the box. Returns the dust [m] it lays on the open cells, booked."""
        out = solids * rim_escape(aim, can.above_rim)
        dust = drift_dust(solids - out, aim, c, seen, can.standoff)
        self.dust += float(dust.sum()) * CELL * CELL
        self.sheet += float(drift_dust(solids - out, aim, c, ~seen, can.standoff).sum()) * CELL * CELL
        self.inside += solids - out
        self.outside += out
        return dust


INSIDE = ("crate", "stencil")    # shapes in the box besides its own (box_*): the crate's edges, the sheet


def supply_books(sr: dict, occluders: tuple[str, ...], mist: Mist) -> Books:
    """The can's solids from the supply's ledgers (``supply.read()``, world 0) and the laid mist: emitted (all the can
    sent), on the crate (the solids that landed on its face), on the sheet (the stencil's catch), in the box (the
    box's catch, the crate's edges' and the mist inside), outside (rays that hit nothing or a shape outside the box,
    and the mist over the rim)."""
    phi = paint(PAINT).volume_solids
    by = dict(zip(occluders, np.asarray(sr["caught_by"])[0].tolist()))
    box = sum(v for k, v in by.items() if k.startswith("box_") or k == "crate")
    other = sum(v for k, v in by.items() if k not in INSIDE and not k.startswith("box_"))
    return Books(emitted=float(sr["sent"][0]) * phi, on_crate=float(sr["species_sent"][0, 0]),
                 on_sheet=by.get("stencil", 0.0) * phi + mist.sheet, in_box=box * phi + mist.inside - mist.sheet,
                 outside=(float(sr["overspray"][0] + sr["lost"][0]) + other) * phi + mist.outside, dust=mist.dust)


def run_supply(layout: SprayLayout | None = None, can: Can | None = None, settle: float = 10.0, device: str = "cpu",
               rays: int = RAYS, driver=None) -> dict:  # noqa: ANN001
    """spray-stencil with the supply's can: the stand-in's schedule, one aim point a sub-period (``can.substeps`` a
    control period), then ``settle`` s before the verdict. Besides the verdict: the books, their closure (the supply's
    ledgers against the task's split), the solids of the rays that hit nothing (left the box) and the
    most dry film on any cell under the sheet [µm]. ``driver`` moves the can (default :class:`KinematicCan`)."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415

    layout, can = layout or SprayLayout(), can or Can()
    driver = driver or KinematicCan(layout, can)
    c = layout.centres()
    sheet = layout.sheet(c)
    flow = SurfaceFlow(charts=(PlaneChart("crate", cells=layout.cells, edges=EDGES),), inertia=False, max_depth=3e-4,
                       nozzles=(can_nozzle(can, rays, driver.can),), network=can_network(can),
                       **pp.film_settings(PAINT, substrate="plywood_sealed"))  # fmt: skip

    def init(sim) -> None:  # noqa: ANN001
        flow.set_regions(sheet.astype(np.int32), chart=0)                 # the seal: no film crosses the sheet's edge

    sub = CONTROL_DT / can.substeps
    sim = build_simulation(driver.scene(init, sub), [flow], device)
    driver.bind(sim)
    phi, mist, booked = paint(PAINT).volume_solids, Mist(), 0.0

    def period(step) -> None:  # noqa: ANN001
        """One sub-period: the valve, the can moved, the step, and the new mist laid where the nozzle is."""
        nonlocal booked
        flow.set_input("valve", np.full(1, 1.0 if step is not None and driver.valve(step) else 0.0, np.float32))
        driver.apply(sim, step)
        sim.step_period()
        now = float(flow.supply.read()["mist"][0])
        if now > booked:
            pp.lay_solids(flow, mist.fall((now - booked) * phi, driver.nozzle_xy(sim), c, ~sheet, can)[None])
        booked = now

    for step in driver.steps:
        period(step)
    period(None)                    # the valve shuts an exchange late: the network answers the next exchange
    declared = flow.depth(0)[0].copy()
    for _ in range(int(round(settle / sub))):
        period(None)
    sr, read = flow.supply.read(), flow.read()
    books = supply_books(sr, flow.supply.occluders, mist)
    v = evaluate(layout, flow.depth(0)[0], declared, read, books, books_rel=BOOKS_REL)
    dry = pp.dry_film(read, flow.depth(0)[0], PAINT)[0]
    return {**v, "books": asdict(books), "books_closure": books.closure(),
            "rays_outside": float(sr["overspray"][0] + sr["lost"][0]) * phi,
            "under_sheet_um": float(dry[sheet].max()) * 1e6, **driver.report()}


__all__ = ["INSIDE", "RAYS", "KinematicCan", "Mist", "box_shapes", "can_network", "can_nozzle", "run_supply",
           "sheet_quads", "spray_scene", "stencil_mesh", "supply_books"]
