# SPDX-License-Identifier: Apache-2.0
"""flush-floor: hose the soiled can-wash bay into its floor drain, dock the hose valve and keep the dunnage shelf dry
(Gym-Anything's surface-rinsing-tasks-v1 flush-floor; gates, settle and score in :mod:`alloy.tasks.rinse_flush_checks`,
the shared rules in :mod:`alloy.tasks.rinse_score`).

The bay is Gym-Anything's, in the bay frame (their x - 5.2 m):
- Quarry tile, 2.0 x 1.8 m, contact angle 60°, a plane chart at 10 mm cells, open on its -x and +y edges.
- It slopes 1/96 to a 6 in floor drain at its centre (radius 76 mm), a cone round the drain as theirs (they tilt the
  cells' normals, ``water/system.py:144-153``; here the film's ``base`` height rises from the drain's rim). The
  collision floor stays flat.
- The drain takes the film off the cells under it (:class:`~alloy.plugins.surface_flow.supply_spec.Drain`, away).
- The dunnage shelf (three 20 mm boards on four 25 mm posts, west of the bay, ``layout.py:282-287``): charts on
  the boards' tops measure the water landing there; the boards' other faces and the posts count what they catch.
- Their 32 gal garbage can stands dry at (6.75, 1.55) (``layout.py:514-516``): a static 245 mm cylinder (theirs
  tapers 224-263 mm) 692 mm tall; no film runs under it. It has no chart: water striking it is booked caught (away),
  not run down to the floor.

The soil is Gym-Anything's start soil:
- Dust on 85 % of the cells of x 0.3-1.3, y 0.35-1.45 m, 0.5 g/m² x 4 x U(0.2, 1): 40-200 µg/cm².
- None lies on the drain's opening. Gym-Anything soils those cells too (about 1.6 % of its dirty set), and they wet
  before its drain empties them (``engine.py:1786,1817``). Here the drain takes the film off its cells inside the film
  step, so soil there would never wet.
- Under a 0.6 mm wash film on the soiled cells at the start (``layout.py:519-521``, ``water/system.py:413-427``), so
  the soil soaks from the start. Their film is soapy (0.75 g/L); the soap is not modelled here (it moves no soil in
  their law either).
- Dry 4 Pa, wet 0.5 Pa, soaking in 2 s (their DUST).
- Removed by jet impact and film shear at their rates, ``k_i`` 2.0 and ``k_e`` 0.05 (their SOIL), through the
  surface-flow process stage's soil law (WP9).
- Removed soil rides in the film as the species ``soil`` and leaves down the drain or off the open edges.

The hose is Gym-Anything's reel and valve on DC5's supply:
- 60 psi mains, a hose loss and the valve's orifices, sized so the valve delivers their 3.3 gpm at 24 m/s. Their
  seeds scale the mains by U(40, 80)/60 (``layout.py:434``); every world here runs seed 0's 60 psi.
- The valve opening is the network input ``valve``.
- Their nine jets: a centre jet and a ring of eight tilted 6° off the axis (``water/network.py:98-104``). Here a
  ``cone`` nozzle of nine rays filling that 6° half-angle (DC5's ``angle`` is the full angle: 12°).
- The valve is a kinematic body the scripted policy moves; its nozzle is 0.104 m down its axis, docked at their
  holster's centre (``scene.py:281``).

Food scraps (Alloy's addition; Gym-Anything has no debris bodies):
- 30 x 30 x 12 mm boxes of 11 g on the floor, listed debris of the film (its drag) that the jets push where they hit.
- The gate ``scraps_on_drain`` asks every scrap's centre within the drain's radius. ``floor_water`` (at most 50 mL off
  the bay) is Alloy's too: Gym-Anything does not gate it for flush-floor. ``success_ga`` leaves both out.
- MuJoCo steps the rigid bodies at 5 ms; one 20 ms XPBD step tunnels a scrap through the floor.

The policy is :mod:`alloy.tasks.rinse_flush_policy`: spirals over the soil, then each scrap chased onto the drain, then
the valve docked. The floor settles hands-off, as Gym-Anything's, before each world's verdict.

Example::

    from alloy.tasks.rinse_flush_floor import run
    print(run()["worlds"][0]["success"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.plugins.fluid import P_ATM
from alloy.tasks.paint_supply import Watched
from alloy.tasks.rinse_flush_checks import (FLUSHED, PARK_M, RHO, SETTLE_MIN, SETTLE_STEP, SHELF_G, Columns,
                                            End, dirty_cells, gauge, quiet, quiet_books, shelf_grams, shelf_slots,
                                            verdict)
from alloy.tasks.rinse_flush_policy import (CONTROL_DT, NOZZLE, Flush, bodies, chase, parked, q_starts, spiral, sweep,
                                           transit)
from alloy.tasks.rinse_sink import GPM
from alloy.tasks.rinse_ware import PSI

RIGID_DT = 0.005                # s: MuJoCo's step, so the scraps rest on the floor (one 20 ms XPBD step tunnels them)
RIGID_GAP = 0.005               # m: contact gap of every shape (Newton's default 0.1 m)
NCONMAX = 64                    # contacts per world: 8 a scrap on the floor, room to spare; overflow faults the run
EDGE = "valve"                  # the valve's orifice edge (and its opening input)
VALVE = "hose_valve"            # the valve's body label
SOIL = "soil"                   # the film's species for suspended soil
SHELF_CELL = 0.02               # m: the shelf's measuring charts
POST = (0.015, 0.0127, 1.70)    # m: the shelf's posts, inset from the boards' corners, radius, height


@dataclass(frozen=True)
class Bay:
    """Gym-Anything's can-wash bay in the bay frame (x from their 5.2 m, z up, floor at 0)."""

    size: tuple[float, float] = (2.0, 1.8)
    cell: float = 0.010
    drain: tuple[float, float] = (1.0, 0.9)
    drain_r: float = 0.076
    slope: float = 1.0 / 96.0
    region: tuple[tuple[float, float], tuple[float, float]] = ((0.3, 1.3), (0.35, 1.45))   # the soiled area
    cover: float = 0.85                 # share of the region's cells soiled
    load: float = 0.5e-3                # kg/m², DUST's load; a cell gets load x 4 x U(0.2, 1)
    holster: tuple[float, float, float] = (1.08, 0.10, 1.055)   # the docked valve body's centre, head down
    shelf: tuple[tuple[float, float], tuple[float, float]] = ((-0.85, -0.15), (0.05, 0.55))
    levels: tuple[float, ...] = (0.25, 0.80, 1.35)
    debris: tuple[tuple[float, float], ...] = ((0.55, 0.75), (1.15, 1.20), (0.80, 1.35))   # scraps' starts, m
    scrap: tuple[float, float, float] = (0.030, 0.030, 0.012)   # m, a food scrap's box (full edges)
    scrap_density: float = 1050.0       # kg/m³ (vegetable matter; ~11 g a scrap)
    contact_angle: float | None = 60.0  # deg, quarry tile (Gym-Anything's)
    can: tuple[float, float, float, float] | None = (1.55, 1.55, 0.245, 0.692)   # x, y, radius, height (m); None: none
    wash_h: float = 0.6e-3              # m: the wash film over the soiled area at the start

    @property
    def cells(self) -> tuple[int, int]:
        return int(round(self.size[0] / self.cell)), int(round(self.size[1] / self.cell))

    def centres(self) -> np.ndarray:
        """Cell centres [nx, ny, 2] in the bay frame."""
        nx, ny = self.cells
        x, y = (np.arange(nx) + 0.5) * self.cell, (np.arange(ny) + 0.5) * self.cell
        return np.stack(np.meshgrid(x, y, indexing="ij"), -1)

    def drain_local(self) -> tuple[float, float, float]:
        """The drain's centre in the floor chart's frame (the chart's centre is the floor's)."""
        return self.drain[0] - 0.5 * self.size[0], self.drain[1] - 0.5 * self.size[1], 0.0

    def base(self) -> np.ndarray:
        """The floor's height above its chart [nx, ny] (m): 1/96 rising from the drain's rim."""
        c = self.centres()
        r = np.hypot(c[..., 0] - self.drain[0], c[..., 1] - self.drain[1])
        return self.slope * np.maximum(r - self.drain_r, 0.0)

    def under_can(self) -> np.ndarray:
        """The floor's cells [nx, ny] under the standing can."""
        if self.can is None:
            return np.zeros(self.cells, dtype=bool)
        c = self.centres()
        return np.hypot(c[..., 0] - self.can[0], c[..., 1] - self.can[1]) <= self.can[2]

    def wash(self, soil: np.ndarray) -> np.ndarray:
        """The wash film (m) on the soiled cells of ``soil`` ([nx, ny] or [W, nx, ny], kg/m²) at the start."""
        return np.where(soil > 0.0, self.wash_h, 0.0)

    def soil(self, rng: np.random.Generator) -> np.ndarray:
        """One world's start soil [nx, ny] (kg/m²), none on the drain's opening (its cells never hold film)."""
        c = self.centres()
        (x0, x1), (y0, y1) = self.region
        inside = (c[..., 0] >= x0) & (c[..., 0] <= x1) & (c[..., 1] >= y0) & (c[..., 1] <= y1)
        inside &= (np.hypot(c[..., 0] - self.drain[0], c[..., 1] - self.drain[1]) > self.drain_r) & ~self.under_can()
        on = inside & (rng.random(inside.shape) < self.cover)
        return np.where(on, self.load * 4.0 * rng.uniform(0.2, 1.0, inside.shape), 0.0)

    def shapes(self) -> list[tuple]:
        """Static boxes (label, centre, quat xyzw, half extents): the floor and the shelf's boards (posts:
        :meth:`posts`)."""
        q, (sx, sy) = (0.0, 0.0, 0.0, 1.0), self.size
        out = [("bay_floor", (0.5 * sx, 0.5 * sy, -0.01), q, (0.5 * sx, 0.5 * sy, 0.01))]
        (a0, a1), (b0, b1) = self.shelf
        for k, z in enumerate(self.levels):
            out.append((f"shelf_{k}", (0.5 * (a0 + a1), 0.5 * (b0 + b1), z - 0.01), q,
                        (0.5 * (a1 - a0), 0.5 * (b1 - b0), 0.01)))
        return out

    def posts(self) -> list[tuple[str, float, float]]:
        """The shelf's posts (label, x, y): one inset ``POST[0]`` from each corner of the boards."""
        (a0, a1), (b0, b1) = self.shelf
        i = POST[0]
        return [(f"shelf_post_{k}", x, y) for k, (x, y) in enumerate((x, y) for x in (a0 + i, a1 - i)
                                                                         for y in (b0 + i, b1 - i))]

    def shelf_cells(self) -> tuple[int, int]:
        (a0, a1), (b0, b1) = self.shelf
        return int(round((a1 - a0) / SHELF_CELL)), int(round((b1 - b0) / SHELF_CELL))

    def charts(self) -> tuple:
        """The floor's chart (index 0: tile, open on -x and +y) and the shelf's measuring charts."""
        from alloy.plugins.surface_flow import PlaneChart  # noqa: PLC0415

        floor = PlaneChart("bay_floor", cells=self.cells, edges=("open", "wall", "wall", "open"),
                           contact_angle=self.contact_angle)
        return (floor, *(PlaneChart(f"shelf_{k}", cells=self.shelf_cells()) for k in range(len(self.levels))))

    def drains(self) -> tuple:
        """The floor drain on chart 0, free, to nowhere (the building drain)."""
        from alloy.plugins.surface_flow.supply_spec import Drain  # noqa: PLC0415

        return (Drain("floor_drain", chart=0, center=self.drain_local(), radius=self.drain_r),)

    def debris_spec(self) -> tuple[dict, ...]:
        """The scraps as the film's listed debris (``SurfaceFlow(drops={"debris": ...})``)."""
        return tuple({"body": f"scrap_{k}", "radius": 0.5 * min(self.scrap[:2])} for k in range(len(self.debris)))

    def build(self, b) -> None:  # noqa: ANN001 - newton.ModelBuilder
        """Add the static floor, shelf boards and posts and can and the scraps (``scrap_k``) to a world template."""
        import warp as wp  # noqa: PLC0415

        from alloy.plugins.harness.scene import body_xform  # noqa: PLC0415

        for label, pos, quat, half in self.shapes():
            b.add_shape_box(-1, xform=wp.transform(wp.vec3(*pos), wp.quat(*quat)), hx=half[0], hy=half[1],
                            hz=half[2], label=label)  # fmt: skip
        for label, x, y in self.posts():
            b.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(x, y, 0.5 * POST[2]), wp.quat_identity()),
                                 radius=POST[1], half_height=0.5 * POST[2], label=label)
        if self.can is not None:
            x, y, r, h = self.can
            b.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(x, y, 0.5 * h), wp.quat_identity()), radius=r,
                                 half_height=0.5 * h, label="garbage_can")
        cfg = b.ShapeConfig(density=self.scrap_density)
        for k, (x, y) in enumerate(self.debris):
            scrap = b.add_body(xform=body_xform(x, y, 0.5 * self.scrap[2] + 1e-4), label=f"scrap_{k}")
            b.add_shape_box(scrap, hx=0.5 * self.scrap[0], hy=0.5 * self.scrap[1], hz=0.5 * self.scrap[2], cfg=cfg)


#: Gym-Anything's quarry-tile bay with its soil, cropped round the drain for CPU tests (0.6 x 0.6 m, 3,600 cells)
CROP = Bay(size=(0.6, 0.6), drain=(0.3, 0.3), region=((0.12, 0.48), (0.12, 0.48)), holster=(0.55, 0.05, 1.055),
           debris=((0.14, 0.32), (0.42, 0.44)), can=None)


@dataclass(frozen=True)
class Hose:
    """Gym-Anything's open reel (REEL) and EB-0107 valve: 60 psi mains, then the hose and the valve's nine orifices
    in series, sized so the valve delivers ``flow`` at ``speed``. The hose's Blasius loss is an orifice here, so
    the flow scales with the square root of the head, not its 4/7 power."""

    mains: float = 60.0 * PSI                # Pa gauge
    flow: float = 3.3 * GPM                  # m³/s delivered (their 3.3 gpm)
    speed: float = 24.0                      # m/s at the orifices (theirs)
    cone_deg: float = 6.0                    # their ring jets' tilt off the axis: the cone's half-angle
    jets: int = 9

    @property
    def angle(self) -> float:
        """The nozzle's full cone angle (DC5's ``Nozzle.angle``), twice their tilt."""
        return math.radians(2.0 * self.cone_deg)

    @property
    def exit_area(self) -> float:
        return self.flow / self.speed

    @property
    def hose_cd_area(self) -> float:
        """The hose as an orifice taking the head the jets do not."""
        return self.flow / math.sqrt(2.0 * (self.mains - 0.5 * RHO * self.speed ** 2) / RHO)

    def network(self) -> dict:
        return {"nodes": [{"name": "line", "medium": "liquid", "volume": 1e-4}],
                "edges": [{"name": "hose", "kind": "orifice", "a": "line", "reservoir": P_ATM + self.mains,
                           "cd_area": self.hose_cd_area, "density": RHO, "dp_lam": 1.0},
                          {"name": EDGE, "kind": "orifice", "a": "line", "reservoir": P_ATM, "cd_area": self.exit_area,
                           "density": RHO, "dp_lam": 1.0}]}

    def nozzle(self):  # noqa: ANN201 - Nozzle
        from alloy.plugins.surface_flow.supply_spec import Nozzle  # noqa: PLC0415

        return Nozzle(VALVE, edge=EDGE, body=VALVE, pos=(0.0, 0.0, -NOZZLE), axis=(0.0, 0.0, -1.0), pattern="cone",
                      exit_area=self.exit_area, angle=self.angle, rays=self.jets, density=RHO)

    def nozzles(self) -> tuple:
        return (self.nozzle(),)

    def add_valve(self, b, pose: tuple[float, ...]) -> int:  # noqa: ANN001 - newton.ModelBuilder
        """The valve: a kinematic body ``VALVE`` (a 54 x 200 mm cylinder, its nozzle ``NOZZLE`` down its axis) at
        ``pose`` (x, y, z, qx, qy, qz, qw). Drive it with ``rinse_flush_policy.place``."""
        from alloy.plugins.harness.scene import body_xform  # noqa: PLC0415

        valve = b.add_body(xform=body_xform(*pose[:3], pose[3:]), is_kinematic=True, label=VALVE)
        b.add_shape_cylinder(valve, radius=0.027, half_height=0.10)
        return valve


def make_flow(bay: Bay, hose: Hose):  # noqa: ANN201 - SurfaceFlow
    """The bay's charts, drain and scraps, water with the soil species, GA's dust soil law, and the hose."""
    from alloy.plugins.surface_flow import SurfaceFlow  # noqa: PLC0415
    from alloy.plugins.surface_flow.process_specs import Soil  # noqa: PLC0415

    soil = Soil(initial=0.0, charts=(0,), sigma_dry=4.0, sigma_wet=0.5, t_soak=2.0, k_e=0.05, k_i=2.0, k_wipe=0.0)
    return SurfaceFlow(charts=bay.charts(), material="water", species=(SOIL,), process={"soil": soil},
                       drops={"debris": bay.debris_spec()}, nozzles=hose.nozzles(), network=hose.network(),
                       drains=bay.drains(), max_depth=0.01)


def build_scene(bay: Bay, hose: Hose, init, worlds: int = 1):  # noqa: ANN001, ANN201 - a harness TestScene
    """The bay (:meth:`Bay.build`), the hose valve docked (:meth:`Hose.add_valve`) and a resting dummy sphere on the
    top shelf (a scene needs a dynamic body), stepped by MuJoCo at ``RIGID_DT``."""
    from alloy.plugins.harness.scene import TestScene, body_xform  # noqa: PLC0415

    def build(b) -> None:  # noqa: ANN001 - newton.ModelBuilder
        b.rigid_gap = RIGID_GAP                       # Newton's 0.1 m default makes contacts of every pair within 0.2 m
        bay.build(b)
        hose.add_valve(b, parked(bay))
        (a0, a1), (b0, b1) = bay.shelf
        dummy = b.add_body(xform=body_xform(0.5 * (a0 + a1), 0.5 * (b0 + b1), bay.levels[-1] + 0.01), label="dummy")
        b.add_shape_sphere(dummy, radius=0.01)

    return TestScene(build=build, init=init, worlds=worlds, periods=1, control_dt=CONTROL_DT, dt=RIGID_DT,
                     solver="mujoco", ground=False,
                     solver_kwargs={"nconmax": NCONMAX, "njmax": 4 * NCONMAX})  # fmt: skip


def scrap_speed(sim, scraps: list[np.ndarray], worlds: int) -> np.ndarray:  # noqa: ANN001 - a simulation
    """The fastest scrap's speed per world [W] (m/s); 0 without scraps."""
    if not scraps:
        return np.zeros(worlds)
    qd = sim.state.body_qd.numpy()
    return np.stack([np.linalg.norm(qd[i, :3], axis=1) for i in scraps], 1).max(axis=1)


def reading(flow) -> dict:  # noqa: ANN001 - SurfaceFlow
    """The flow's read, with its drop stage's species books as ``drops_<key>`` [W, K] once the stage carries the
    film's species (WP9): the soil they put back on landing (``LANDED``) leaves the soil's chain, and their own
    identity is checked (``rinse_flush_checks.drops_soil_gap``)."""
    read = flow.read()
    drops = getattr(flow, "drops", None)
    for key, arr in (drops.read() if drops is not None else {}).items():
        if key.startswith("species_") or key == "lump_species_dropped":
            read[f"drops_{key}"] = np.asarray(arr)
    return read


def settle(sim, flow, cap: float, scraps: list[np.ndarray], cols: Columns, judge) -> list[dict]:  # noqa: ANN001
    """Gym-Anything's settle (:mod:`alloy.tasks.rinse_flush_checks`): hands-off for ``SETTLE_MIN``, then
    ``SETTLE_STEP`` at a time until a world is quiet, at most ``cap`` s; each world is judged then by
    ``judge(world, read, seconds)``."""
    worlds = int(sim.model.world_count)
    rows: list[dict | None] = [None] * worlds
    t, prev = 0.0, None
    while any(r is None for r in rows):
        step = min(SETTLE_MIN if prev is None else SETTLE_STEP, cap - t)
        for _ in range(int(round(step / CONTROL_DT))):
            sim.step_period()
        t += step
        read, last = reading(flow), t >= cap - 1e-9
        now = quiet_books(read)
        done = np.full(worlds, last)
        if prev is not None:
            q_off = np.abs(np.asarray(read["supply_discharge"])[:, cols.outlet]) / GPM
            done |= quiet(prev, now, q_off, scrap_speed(sim, scraps, worlds))
        for w in np.flatnonzero(done):
            if rows[w] is None:
                rows[w] = judge(int(w), read, round(t, 3))
        prev = now
    return rows


def run(bay: Bay | None = None, fl: Flush | None = None, hose: Hose | None = None, worlds: int = 1, seed: int = 0,
        device: str = "cpu", graph: bool = False) -> dict:
    """The scripted flush in ``worlds`` worlds (each its own start soil under the wash film, seeded): the spirals with
    the valve open, the scraps chased onto the drain, the valve docked (jets off between, :func:`transit`), then
    Gym-Anything's settle, at most ``fl.settle`` s, and each world's :func:`verdict`."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run

    bay, fl, hose = bay or Bay(), fl or Flush(), hose or Hose()
    rng = np.random.default_rng(seed)
    soil0 = np.stack([bay.soil(rng) for _ in range(worlds)])
    wash, flow = bay.wash(soil0), make_flow(bay, hose)
    shut = np.zeros(worlds, np.float32)

    def init(sim) -> None:  # noqa: ANN001
        flow.set_input(EDGE, shut)
        flow.process.set_soil(soil0, chart=0)
        base = np.zeros(flow.read()["h"].shape, np.float32)
        base[:, : soil0[0].size] = bay.base().ravel()
        flow.set_input("base", base)
        flow.set_regions(bay.under_can().astype(np.int32), chart=0)       # no film under the standing can
        flow.set_depth(wash, chart=0)

    from alloy.tasks.rinse_hand import OverflowWatch  # noqa: PLC0415 - every substep's overflow (WP8, DC5)

    sim = Watched(build_simulation(build_scene(bay, hose, init, worlds), [flow, OverflowWatch()], device, graph=graph))
    n0 = soil0[0].size
    set_volume = (wash.reshape(worlds, n0) * flow.tables.area_np[:n0]).sum(axis=1)
    cols = Columns(flow.supply.net.edge_index(EDGE), shelf_slots(sim.model, flow.supply.occluder_shapes))
    starts, valve = q_starts(sim.model, bodies(sim.model, VALVE)), bodies(sim.model, VALVE)
    scraps = [bodies(sim.model, f"scrap_{k}") for k in range(len(bay.debris))]
    pose = np.broadcast_to(np.asarray(parked(bay)), (worlds, 7))
    for k in range(fl.passes):
        pose = sweep(sim, flow, bay, fl, starts, EDGE, pose, shift=0.5 * fl.pitch * (k % 2))
    took, pose = chase(sim, flow, bay, fl, starts, scraps, EDGE, pose)
    transit(sim, flow, starts, pose, parked(bay), EDGE)
    pre = flow.read()
    pre_gauge = [gauge(bay, pre, w) for w in range(worlds)]
    pre_shelf = [shelf_grams(pre, n0, cols, w) for w in range(worlds)]

    def judge(w: int, read: dict, t: float) -> dict:
        q = sim.state.body_q.numpy()
        at = np.stack([q[i[w], :2] for i in scraps]) if scraps else np.zeros((0, 2))
        end = End(q[valve[w], :3], at, pre_gauge[w], float(set_volume[w]), pre_shelf[w],
                  int(sim.watch.worlds(worlds)[w]), t)
        return verdict(bay, soil0[w], read, end, w, cols)

    rows = settle(sim, flow, fl.settle, scraps, cols, judge)
    return {"worlds": rows, "success": all(r["success"] for r in rows),
            "success_ga": all(r["success_ga"] for r in rows),
            "judged": [r["judged"] for r in rows], "not_judged": [r["not_judged"] for r in rows],
            "summary": [r["summary"] for r in rows],
            "flush_s": round(fl.passes * len(spiral(bay, fl)) * CONTROL_DT, 2), "chase_s": took,
            "settle_cap_s": fl.settle, "seed": seed}


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m alloy.tasks.rinse_flush_floor [--crop] [--worlds W] [--device D] [--settle S] --out f.json``."""
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--settle", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--graph", action="store_true", help="replay each control period as a captured graph")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    res = run(CROP if a.crop else None, Flush(settle=a.settle), worlds=a.worlds, seed=a.seed, device=a.device,
              graph=a.graph)
    res.update(wall_s=round(time.perf_counter() - t0, 1), device=a.device, crop=a.crop)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    ok, ga = (sum(r[k] for r in res["worlds"]) for k in ("success", "success_ga"))
    print(f"flush-floor: {ok}/{a.worlds} worlds pass ({ga} on Gym-Anything's gates), wall {res['wall_s']} s -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["Bay", "CROP", "End", "FLUSHED", "Hose", "POST", "RIGID_DT", "SOIL", "VALVE", "build_scene",
           "dirty_cells", "gauge", "main", "make_flow", "run", "scrap_speed", "settle", "verdict"]
