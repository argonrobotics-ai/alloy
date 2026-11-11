# SPDX-License-Identifier: Apache-2.0
"""clear-coat: clear-coat a photo print lying in a spray box evenly, 25-50 µm dry from light overlapping passes at
25-40 cm, without runs, keeping the overspray in the box (Gym-Anything's painting-tasks-v1 clear-coat). The verdict,
the books, the score and their Gym-Anything sources are :mod:`alloy.tasks.paint_clear_coat_checks`, which also says
what the verdict judges: four of Gym-Anything's five criteria (``untouched`` is not judged), on a scene with a
free-moving can, a 10 s settle and one start (its ``SCENE``).

The scene is Gym-Anything's spray preset with the box centred at the origin:
- An open spray box, 0.60 x 0.45 x 0.45 m of 4 mm cardboard, on a folding table whose top is at 0.74 m.
- A 10 x 8 in photo print on the box floor. One plane chart at 1 mm (Gym-Anything's spray chart) covers it and 30 mm
  of the floor round it (``BoxLayout.bed``); the print's cells are a film region of their own (no film crosses its
  edge). DC5's spray lands each ray over a footprint on the chart the ray hits, so a print charted alone misses the
  share of the rays landing just past its edge: one coat read 11 % thin along its edges and 21 % at its corners.

The can is DC5's aerosol supply:
- A gas node (the can's headspace) drives the lacquer out through an ``expel`` valve. The network input ``valve``
  opens and shuts it.
- A ``spray`` :class:`~alloy.plugins.surface_flow.Nozzle` sits on the can, a kinematic body that follows the passes
  (:func:`schedule`). The spray is Gaussian over the 12° half cone (σ 6°). ``efficiency`` of it lands where its rays
  hit, the rest is mist.
- The network carries the lacquer as it lands. 60 % of the can's liquid mass (propellant and solvent) flashes in
  flight (Gym-Anything's estimate), so the valve passes 0.8 g/s of landed lacquer for the can's 2.0 g/s. The flashed
  mass is booked here (:func:`can_books`), not in the film.

DC5's ledger books the rays that miss the print as overspray and the rest of the can's output as mist:
- The mist dries in the air and falls on the first surface below as dust, drifting σ = 0.25 x its fall (Gym-Anything's
  aerosol; SOFT-3's rule). Every 0.5 s the mist DC5 booked since falls, in equal shares, from where the nozzle was
  (:func:`mist_dust`, integrated over each cell) and lands on the chart as dry solids
  (:func:`~alloy.tasks.paint_process.lay_solids`). The rest lands on the box, or leaves it from above the rim. Its
  books close against DC5's mist x φ0 (:func:`~alloy.tasks.paint_clear_coat_checks.dust_books`).
- :func:`escape_share` decides what leaves the box, from the nozzle's own ray table: a ray that crosses the rim's
  plane outside the opening, or leaves through it, and mist drifting out of the opening on its fall to the rim. From
  below the rim and over the box, none of it leaves.

The film dries (WP9's process stage, :data:`DRYING`): the spray lands the solids as the film's species ``solids`` at
``φ0`` by volume, only the solvent leaves, and the dry film is the solids per area (``read()["dft"]``). The values are
Gym-Anything's lacquer (their enamel without pigment): solids pack at 0.64, the wet edge passes at 0.45, and the
solvent evaporates at their k_m 0.030 m/s (5 x water's, at the 50 % RH reference).

Example::

    from alloy.tasks.paint_clear_coat import run
    print(run()["worlds"][0]["succeeded"])
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np

from alloy.tasks import paint_metrics as pm
from alloy.tasks.paint_clear_coat_checks import (CELL, COAT_FRAC, DENSITY, EVEN, FLASH, MAX_DEPTH, PHI0, DustBooks,
                                                 can_books, coat_checks, dust_books, escape_share, evaluate,
                                                 mist_dust, mist_escape, spray_progress)
from alloy.tasks.paint_materials import film_material
from alloy.tasks.paint_process import lay_solids
from alloy.tasks.paint_spray_stencil import Can
from alloy.tasks.paint_supply import bodies

P_ATM = 101325.0                # Pa (alloy.plugins.fluid's; importing it loads warp)
CONTROL_DT = 0.02               # s: the can moves 5.2 mm a period at 0.26 m/s
EDGE = "valve"                  # the can's expel edge (and its opening input)
CAN = "can"                     # the can's body label
DUST_EVERY = 25                 # periods: the mist released in 0.5 s lands at once (Gym-Anything's lives up to 3 s)
DRYING = {"phi_0": PHI0, "phi_m": 0.64, "phi_we": 0.45, "k_m": 0.030, "rh": 0.5}   # Gym-Anything's LACQUER


def lacquer():  # noqa: ANN201 - FilmMaterial
    """Gym-Anything's LACQUER: the aerosol enamel without pigment, at density 1000 kg/m³. Viscosity and surface
    tension are the enamel card's (``paint_enamel_aerosol``)."""
    return dataclasses.replace(film_material("enamel_aerosol"), name="lacquer_aerosol", density=DENSITY)


@dataclass(frozen=True)
class BoxLayout:
    """The spray box and the print, box frame: the box's centre at the origin, z up."""

    print_size: tuple[float, float] = (0.254, 0.2032)   # m, a 10 x 8 in print (Gym-Anything's)
    box: tuple[float, float, float] = (0.60, 0.45, 0.45)
    wall: float = 0.004                                 # m, cardboard
    table_z: float = 0.74                               # m, the folding table's top
    print_thick: float = 0.25e-3                        # m, photo paper
    bed: float = 0.03                                   # m of the box floor charted round the print (see run)

    @property
    def chart_size(self) -> tuple[float, float]:
        return self.print_size[0] + 2.0 * self.bed, self.print_size[1] + 2.0 * self.bed

    @property
    def cells(self) -> tuple[int, int]:
        """The chart's cells: the print and its bed."""
        return int(round(self.chart_size[0] / CELL)), int(round(self.chart_size[1] / CELL))

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        """The chart's cell edges along x [nx + 1] and y [ny + 1], box frame."""
        (nx, ny), (sx, sy) = self.cells, self.chart_size
        return np.linspace(-0.5 * sx, 0.5 * sx, nx + 1), np.linspace(-0.5 * sy, 0.5 * sy, ny + 1)

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        """The chart's cell centres along x [nx] and y [ny], box frame (the chart is centred on the box)."""
        (nx, ny), (sx, sy) = self.cells, self.chart_size
        return (np.arange(nx) + 0.5) * sx / nx - 0.5 * sx, (np.arange(ny) + 0.5) * sy / ny - 0.5 * sy

    @property
    def cell_area(self) -> float:
        return self.chart_size[0] / self.cells[0] * self.chart_size[1] / self.cells[1]

    def print_mask(self) -> np.ndarray:
        """The chart's cells [nx, ny] whose centres lie on the print."""
        x, y = self.axes()
        return (np.abs(x)[:, None] <= 0.5 * self.print_size[0]) & (np.abs(y)[None, :] <= 0.5 * self.print_size[1])

    @property
    def print_z(self) -> float:
        """The print's face [m]."""
        return self.table_z + self.wall + self.print_thick

    @property
    def rim_z(self) -> float:
        return self.table_z + self.box[2]

    def shapes(self) -> list[tuple]:
        """Static boxes (label, centre, quat xyzw, half extents): the table, the box's floor and walls, and the bed
        (the print and the floor round it, one chart)."""
        (bx, by, bz), t, z0, q = self.box, self.wall, self.table_z, (0.0, 0.0, 0.0, 1.0)
        hx, hy, hz = 0.5 * bx, 0.5 * by, 0.5 * bz
        px, py = 0.5 * self.chart_size[0], 0.5 * self.chart_size[1]
        return [("table", (-0.25, -0.03, z0 - 0.01), q, (0.61, 0.305, 0.01)),
                ("box_floor", (0.0, 0.0, z0 + 0.5 * t), q, (hx, hy, 0.5 * t)),
                ("box_x0", (-hx + 0.5 * t, 0.0, z0 + hz), q, (0.5 * t, hy, hz)),
                ("box_x1", (hx - 0.5 * t, 0.0, z0 + hz), q, (0.5 * t, hy, hz)),
                ("box_y0", (0.0, -hy + 0.5 * t, z0 + hz), q, (hx, 0.5 * t, hz)),
                ("box_y1", (0.0, hy - 0.5 * t, z0 + hz), q, (hx, 0.5 * t, hz)),
                ("bed", (0.0, 0.0, self.print_z - 0.5 * self.print_thick), q, (px, py, 0.5 * self.print_thick))]


@dataclass(frozen=True)
class AerosolCan:
    """Gym-Anything's 12 oz aerosol (``AEROSOL``) on DC5's expel valve. The valve is sized so a full can passes
    ``mdot`` of contents, of which ``1 - FLASH`` lands: 0.8 mL/s of lacquer at ``exit_speed``. The headspace gas
    expands as the can empties (DC5's expel model, no propellant boiling), so the rate falls a few % over a clear
    coat."""

    mdot: float = 2.0e-3                    # kg/s of the can's contents (Gym-Anything's)
    exit_speed: float = 25.0                # m/s (Gym-Anything's U0)
    pressure: float = 4.0e5                 # Pa absolute, ~43 psig (DC5's aerosol test; an estimate)
    fill: float = 0.34                      # kg of contents (Gym-Anything's)
    headspace: float = 2.2e-4               # m³: a 66 x 196 mm can (0.67 L) less 0.45 L of liquid (estimate)
    half_angle: float = Can.half_angle      # 12°, SOFT-3's stand-in can and Gym-Anything's
    efficiency: float = Can.efficiency      # 0.70 lands, SOFT-3's estimate (data sheet 0.85, Gym-Anything ~0.6)
    rays: int = 128                         # converged with DC5's overlapping footprints (44596dc64) and the bed:
                                            # crop max/min 1.013 at 64 rays, 1.007 at 128, 1.005 at 512

    @property
    def rate(self) -> float:
        """Landed lacquer [m³/s] at a full valve, full can."""
        return self.mdot * (1.0 - FLASH) / DENSITY

    @property
    def cd_area(self) -> float:
        return self.rate / math.sqrt(2.0 * (self.pressure - P_ATM) / DENSITY)

    @property
    def exit_area(self) -> float:
        return self.rate / self.exit_speed

    @property
    def capacity(self) -> float:
        """The can's lacquer as it lands [m³]."""
        return self.fill * (1.0 - FLASH) / DENSITY

    def network(self) -> dict:
        return {"nodes": [{"name": CAN, "volume": self.headspace, "pressure": self.pressure}],
                "edges": [{"name": EDGE, "kind": "expel", "a": CAN, "cd_area": self.cd_area, "density": DENSITY,
                           "dp_lam": 1.0, "capacity": self.capacity}]}

    def ray_table(self) -> tuple[np.ndarray, np.ndarray]:
        """The nozzle's ray directions [R, 3] (box frame: the can is never turned) and flow fractions [R]."""
        from alloy.plugins.surface_flow.supply_spec import ray_table  # noqa: PLC0415

        dirs, wt, _ = ray_table(self.nozzle())
        return dirs, wt

    def nozzle(self):  # noqa: ANN201 - Nozzle
        from alloy.plugins.surface_flow.supply_spec import Nozzle  # noqa: PLC0415

        return Nozzle(CAN, edge=EDGE, body=CAN, axis=(0.0, 0.0, -1.0), pattern="spray", exit_area=self.exit_area,
                      angle=2.0 * self.half_angle, rays=self.rays, efficiency=self.efficiency, density=DENSITY,
                      species={"solids": PHI0})


@dataclass(frozen=True)
class Passes:
    """Light overlapping passes: rows along x, back and forth, ``pitch`` apart, ``margin`` past the print on every
    side, at ``speed``. Each row starts with the valve shut for one period (the can moves to it). The second coat's rows
    sit half a pitch over. Wet per coat ≈ efficiency · rate / (speed · pitch): 72 µm at the defaults, 14.4 µm dry;
    the mist's dust adds 9-11 µm over the two coats (0.40-0.25 m), so the print reads 30-39 µm. At 0.20 m/s the dust
    took 21-48 % of the print over 50 µm at 0.25-0.30 m."""

    standoff: float = 0.30                  # m, nozzle above the print (Gym-Anything's 25-40 cm)
    speed: float = 0.26                     # m/s, our choice (Gym-Anything gives no pass speed, layout.py:121-123)
    pitch: float = 0.03                     # m between rows
    margin: float = 0.06                    # m past the print's edges (about 2 σ of the cone at 0.30 m)
    coats: int = 2
    pause: float = 2.0                      # s, valve shut between coats


def schedule(lay: BoxLayout, ps: Passes) -> list[tuple[float, float, bool]]:
    """Per control period: the nozzle's (x, y) in the box frame and whether the valve is open."""
    hx, hy = 0.5 * lay.print_size[0] + ps.margin, 0.5 * lay.print_size[1] + ps.margin
    step = ps.speed * CONTROL_DT
    n = int(math.ceil(2.0 * hx / step - 1e-9))
    out: list[tuple[float, float, bool]] = []
    for coat in range(ps.coats):
        if coat:
            out += [(out[-1][0], out[-1][1], False)] * int(round(ps.pause / CONTROL_DT))
        ys = np.arange(-hy + (0.5 * ps.pitch if coat % 2 else 0.0), hy + 1e-9, ps.pitch)
        for k, y in enumerate(ys):
            xs = -hx + (np.arange(n) + 0.5) * step
            xs = xs if k % 2 == 0 else xs[::-1]
            out.append((float(xs[0]), float(y), False))
            out += [(float(x), float(y), True) for x in xs]
    return out


def build_scene(lay: BoxLayout, init, worlds: int = 1):  # noqa: ANN001, ANN201 - a harness TestScene
    """The static box and print, the kinematic can (a 66 x 196 mm cylinder lying behind its nozzle) and a resting
    dummy sphere on the table (a scene needs a dynamic body)."""
    import warp as wp  # noqa: PLC0415

    from alloy.plugins.harness.scene import TestScene, body_xform  # noqa: PLC0415

    def build(b) -> None:  # noqa: ANN001 - newton.ModelBuilder
        for label, pos, quat, half in lay.shapes():
            b.add_shape_box(-1, xform=wp.transform(wp.vec3(*pos), wp.quat(*quat)), hx=half[0], hy=half[1],
                            hz=half[2], label=label)  # fmt: skip
        can = b.add_body(xform=body_xform(0.0, 0.0, lay.rim_z + 0.2), is_kinematic=True, label=CAN)
        s = math.sqrt(0.5)
        b.add_shape_cylinder(can, xform=wp.transform(wp.vec3(-0.104, 0.0, 0.035), wp.quat(0.0, s, 0.0, s)),
                             radius=0.033, half_height=0.098)
        dummy = b.add_body(xform=body_xform(-0.75, -0.03, lay.table_z + 0.01), label="dummy")
        b.add_shape_sphere(dummy, radius=0.01)

    return TestScene(build=build, init=init, worlds=worlds, periods=1, control_dt=CONTROL_DT, dt=CONTROL_DT,
                     solver="xpbd", ground=False)  # fmt: skip


def _place(sim, idx: np.ndarray, x: float, y: float, z: np.ndarray) -> None:  # noqa: ANN001 - Simulation
    """Put the kinematic can's nozzle at (x, y, z[w]) in every world, upright (it holds the pose while stepped)."""
    q = sim.state.body_q.numpy().copy()
    q[idx, 0], q[idx, 1], q[idx, 2], q[idx, 3:] = x, y, z, (0.0, 0.0, 0.0, 1.0)
    sim.state.body_q.assign(q)


def run(lay: BoxLayout | None = None, ps: Passes | None = None, can: AerosolCan | None = None,
        standoffs=None, settle: float = 10.0, device: str = "cpu", worlds: int = 1,
        graph: bool = False) -> dict:  # noqa: ANN001
    """The scripted clear coat: the can follows :func:`schedule` at ``standoffs`` [W] m above the print (default
    ``ps.standoff`` in every world), the valve shuts, the can parks over a corner of the box, the film settles for
    ``settle`` s, then each world's verdict (:func:`evaluate`)."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run
    from alloy.plugins.surface_flow import PlaneChart, SurfaceFlow  # noqa: PLC0415
    from alloy.plugins.surface_flow.process_specs import Drying  # noqa: PLC0415

    lay, ps, can = lay or BoxLayout(), ps or Passes(), can or AerosolCan()
    off = np.full(worlds, ps.standoff) if standoffs is None else np.asarray(standoffs, dtype=np.float64)
    z = lay.print_z + off
    flow = SurfaceFlow(charts=(PlaneChart("bed", cells=lay.cells),), material=lacquer(), inertia=False,
                       max_depth=MAX_DEPTH, nozzles=(can.nozzle(),), network=can.network(), species=("solids",),
                       process={"drying": Drying(**DRYING)})  # fmt: skip
    shut, rays, mask, (xe, ye) = np.zeros(worlds, np.float32), can.ray_table(), lay.print_mask(), lay.edges()

    def init(sim) -> None:  # noqa: ANN001
        flow.set_input(EDGE, shut)
        flow.set_regions(mask.astype(np.int32), chart=0)          # the print's edge: no film crosses it

    sim = build_simulation(build_scene(lay, init, worlds), [flow], device, graph=graph)
    idx, outside, dust, valve = bodies(sim.model, CAN), np.zeros(worlds), DustBooks(), False
    window, last, mist0 = [], [], flow.supply.read()["mist"].copy()
    areas = flow.tables.area_np[: mask.size].reshape(mask.shape)

    def lay_dust() -> None:
        """The mist sent since the last call (DC5's mist ledger, exact) falls as dust in equal shares from where the
        nozzle was (mist booked after the valve shut: from where it last was): the share that stays in the box
        (:func:`mist_escape`) lands, and every share is booked."""
        nonlocal mist0
        mist = flow.supply.read()["mist"].copy()
        aims = np.asarray(window or last).reshape(-1, 2)
        if len(aims) and (mist > mist0).any():
            solids = np.outer((mist - mist0) * PHI0, np.full(len(aims), 1.0 / len(aims)))      # [W, P] m³
            keep = 1.0 - np.array([[mist_escape(lay, (a[0], a[1], zw)) for a in aims] for zw in z])
            d = mist_dust(xe, ye, areas, aims, off, solids * keep)
            lay_solids(flow, d)
            dust_books(lay, aims, z, off, solids, (d * areas).sum(axis=(1, 2)), dust)   # the film's cell areas
        mist0 = mist
        window.clear()

    for k, (x, y, on) in enumerate(schedule(lay, ps)):
        _place(sim, idx, x, y, z)
        if on != valve:
            flow.set_input(EDGE, np.full(worlds, float(on), np.float32))
            valve = on
        share = np.array([escape_share(lay, can.efficiency, rays, (x, y, zw)) for zw in z]) if on else np.zeros(worlds)
        before = flow.read()["supply_sent"].copy() if share.any() else None
        sim.step_period()
        if before is not None:
            outside += (flow.read()["supply_sent"] - before) * share
        if on:
            window.append((x, y))
            last[:] = [(x, y)]
        if (k + 1) % DUST_EVERY == 0:
            lay_dust()
    flow.set_input(EDGE, shut)
    sim.step_period()                                       # the last open period's lacquer lands
    lay_dust()
    declared = flow.depth(0).copy()
    _place(sim, idx, 0.5 * lay.box[0] - 0.05, 0.5 * lay.box[1] - 0.05, z)
    for _ in range(int(round(settle / CONTROL_DT))):
        sim.step_period()
    read = flow.read()
    rows = [evaluate(declared[w], read, float(outside[w]), dust, mask, lay.cell_area, w)
            for w in range(worlds)]
    return {"worlds": rows, "succeeded": all(r["succeeded"] for r in rows), "standoffs": off.tolist(),
            "passes": dataclasses.asdict(ps)}


#: a print crop for CPU tests: 40 x 30 mm in a 15 mm bed, 4,200 cells (its print reads max/min 1.03 at 0.30 m). The
#: full layout keeps 30 mm: on a 60 x 50 mm crop at 0.40 m a 30 or 50 mm bed lays the same print (31.05 / 31.03 µm)
SMALL = BoxLayout(print_size=(0.04, 0.03), bed=0.015)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m alloy.tasks.paint_clear_coat [--small] [--worlds W] [--device D] [--coats N] --out rows.json``.
    With W > 1 the standoffs spread over Gym-Anything's 25-40 cm."""
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--coats", type=int, default=2)
    ap.add_argument("--settle", type=float, default=10.0)
    ap.add_argument("--graph", action="store_true", help="replay each control period as a captured graph")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    offs = np.linspace(0.25, 0.40, a.worlds) if a.worlds > 1 else None
    t0 = time.perf_counter()
    res = run(SMALL if a.small else None, Passes(coats=a.coats), standoffs=offs, settle=a.settle, device=a.device,
              worlds=a.worlds, graph=a.graph)
    res["wall_s"], res["device"], res["small"] = round(time.perf_counter() - t0, 1), a.device, a.small
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    ok = sum(r["succeeded"] for r in res["worlds"])
    print(f"clear-coat: {ok}/{a.worlds} worlds pass, wall {res['wall_s']} s -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["AerosolCan", "BoxLayout", "CELL", "COAT_FRAC", "DENSITY", "DustBooks", "EVEN", "FLASH", "MAX_DEPTH", "PHI0",
           "Passes", "SMALL", "build_scene", "can_books", "coat_checks", "dust_books", "escape_share", "evaluate",
           "lacquer", "main", "mist_dust", "mist_escape", "run", "schedule", "spray_progress"]
