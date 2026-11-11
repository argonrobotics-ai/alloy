# SPDX-License-Identifier: Apache-2.0
"""Scene of task rinse_tomatoes (Gym-Anything's rinse-tomatoes): the prep sink, a colander of 15 dusty cherry
tomatoes on its drainboard, and the work table with the cutting board and a clean steel bowl.

Sources (``surface_rinsing`` in Gym-Anything's RoboWorld; file:line):
- prep sink: one 24 x 24 x 12 in bowl, rim 0.914 m, its 35 mm basket drain OPEN; the drainboard (24 in) before it;
  the gooseneck faucet's outlet over the bowl, pointing down, a 16 mm laminar stream of 2.2 gpm: ``layout.py:60-69,
  141-147, 346-348, 497-499``, ``params.py:45, 64-70``;
- work table (top 0.876 m) with the HDPE cutting board (0.610 x 0.457 x 0.0127 m, yaw 90°) and the clean steel bowl
  (``ware.py:244-256``: rim R 0.150, base r 0.080, H 0.110, 0.8 mm wall, 0.40 kg): ``layout.py:68, 506-508``;
- the colander (``ware.py:259-303``, ``params.py:189``): outer wall r = 0.095 + 0.045 s^0.8 over 100 mm on a 25 mm
  foot ring (16 boxes at r 70 mm), 0.8 mm wall, 1.2 mm floor, 0.45 kg, two loop handles; 4 mm holes on a 9 mm pitch
  in the floor (r < 89 mm) and in rings up the lower 60 % of the wall, Q = Cd a sqrt(2 g depth) per submerged hole
  (Cd 0.6, ``params.py:162``; ``water/pools.py:230-250``); on the drainboard at (6.769, 3.4952), foot at z 0.9175
  (``layout.py:500-502``);
- 15 cherry tomatoes (``ware.py:306-320``, ``params.py:190, 203, 219``): 28.6 x 27 mm ellipsoids of 11.3 g,
  colliding as their mean sphere (r 13.9 mm) and floating as the sphere of their volume (r 14.0 mm: 977 kg/m³, they
  float), friction 0.40 on steel, rolling 0.5 mm, torsion 1 mm; one layer in the colander, rings of 1 + 6 + 8 at
  radii 0, 32, 64 mm (``scene.py:466-477``).

Here each tomato's film is one lump (WP9's lumped films, WP15's ruling for small bodies): its water, soil and dust
erosion are one number per tomato, so a tomato's gauge is its mean load (theirs: the worst 1 cm bin of a 2 mm
chart). The colander and the bowl carry revolution charts (4 mm), the board a plane chart (5 mm). Frame: their x - X0,
y - Y0 (the sink bowl's front left corner at the origin), z up from the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from alloy._vendor import newton

from alloy.plugins.fluid import P_ATM, Edge, Node, tank_overflow
from alloy.plugins.surface_flow.charts import PlaneChart
from alloy.plugins.surface_flow.supply import Drain, Nozzle, Pool
from alloy.plugins.surface_flow.supply_float import Float
from alloy.tasks.rinse_sink import DRAIN_CD, DRAIN_D, FAUCET
from alloy.tasks.rinse_vessels import Vessel, xform
from alloy.tasks.rinse_wash_station import POUR_EXIT, POUR_MAX, RIGID_GAP

X0, Y0 = 6.476, 3.80                  # their sink bowl's x0 and front y (layout.py:60-69)
RIM_Z = 0.914
SINK = (0.6096, 0.6096, 0.3048)       # the bowl's inside: 24 x 24 x 12 in
SINK_WALL = 0.01                      # m, the static walls' thickness (outside the bowl)
TABLE = ((6.438 - X0, 7.2 - X0), (1.95 - Y0, 3.1692 - Y0), 0.876)    # x span, y span, top z (layout.py:68)
BOARD = ((6.819 - X0, 2.31 - Y0), (0.457, 0.610, 0.0127))           # centre, size (yawed 90°: 0.610 along y)
BOWL_AT = (6.819 - X0, 2.8692 - Y0)
COLANDER_AT = (6.769 - X0, 3.4952 - Y0, 0.9175)                    # the foot ring's underside
FAUCET_AT = (6.8008 - X0, 4.1048 - Y0, 1.274)
FAUCET_EXIT = math.pi * 0.008 ** 2    # m², the 16 mm stream (0.69 m/s at 2.2 gpm)
TOMATO_R = 0.5 * (0.0143 + 0.0135)    # m, the collider (their mean sphere)
TOMATO_MASS = 0.0113                  # kg
TOMATO_VOL_R = (0.0143 ** 2 * 0.0135) ** (1.0 / 3.0)   # m, the sphere of their ellipsoid's volume (977 kg/m³: floats)
TOMATOES = 15
STEEL = dict(contact_angle=70.0, h_break=6.0e-5)       # params.py:105
SKIN_HOLD = 1.0e-4                                       # m, the film a tomato's skin clings (h_break, params.py:108)
BOARD_ANGLE = 85.0                    # deg: HDPE's ~95° (their polymer) held under the height-field film's 90°
LINE = 1.0e-5                         # m³, the faucet's line (its regulator feeds it; the 16 mm outlet drains it)


@dataclass(frozen=True)
class Cup(Vessel):
    """Their ``_cup``: an outer wall ``outer_r(z)`` from the base (z 0) to the rim (``height``), the inside ``wall``
    in from it, a floor ``bottom`` thick; the body origin on the inner floor."""

    height: float = 0.1
    r_base: float = 0.095
    r_rim: float = 0.140
    wall: float = 0.0008
    bottom: float = 0.0012
    contact_angle: float = STEEL["contact_angle"]
    ds: float = 4.0e-3
    n_theta: int = 64
    bands: int = 4
    shell: float | None = 0.002       # their slab colliders are 2 mm thick (ware.py:252, 274)

    def outer_r(self, z: np.ndarray | float) -> np.ndarray | float:
        raise NotImplementedError

    def radius(self, z: np.ndarray | float) -> np.ndarray | float:
        zz = np.clip(np.asarray(z, np.float64), 0.0, self.depth) + self.bottom
        return self.outer_r(zz) - self.wall


@dataclass(frozen=True)
class SteelBowl(Cup):
    """The clean 300 mm bowl: r = rb + (R - rb) sin(πs/2), z = H (1 - cos(πs/2))^0.9 (``ware.py:244-256``)."""

    height: float = 0.110
    r_base: float = 0.080
    r_rim: float = 0.150
    depth: float = 0.110 - 0.0012
    mass: float = 0.40

    def outer_r(self, z: np.ndarray | float) -> np.ndarray | float:
        u = 1.0 - np.clip(np.asarray(z, np.float64) / self.height, 0.0, 1.0) ** (1.0 / 0.9)   # cos(πs/2)
        return self.r_base + (self.r_rim - self.r_base) * np.sqrt(np.maximum(1.0 - u * u, 0.0))


@dataclass(frozen=True)
class Colander(Cup):
    """The 5 qt colander (``ware.py:259-303``): the bowl r = rb + (R - rb) s^0.8 over ``height`` on a ``foot`` ring."""

    height: float = 0.100
    depth: float = 0.100 - 0.0012
    mass: float = 0.45
    foot: float = 0.025
    pitch: float = 0.009
    hole_d: float = 0.004
    hole_cd: float = 0.6

    def outer_r(self, z: np.ndarray | float) -> np.ndarray | float:
        s = np.clip(np.asarray(z, np.float64) / self.height, 0.0, 1.0)
        return self.r_base + (self.r_rim - self.r_base) * s ** 0.8

    def perforated(self) -> float:
        """The radius about the inner floor's centre that takes in the floor and the wall up to the top hole ring (60 %
        of the bowl), not above: their perforated cells, whose film runs into the holes (``system.py:631-643``)."""
        z = 0.6 * self.height - self.bottom
        lo, hi = math.hypot(float(self.radius(z)) + self.wall, z), math.hypot(float(self.radius(z + 0.01)), z + 0.01)
        return 0.5 * (lo + hi)

    def holes(self) -> tuple[tuple[float, float, float], ...]:
        """Rings of holes (z above the inner floor, Cd·area, r): the floor's grid (r < rb - 6 mm) as one ring at its
        holes' RMS radius, then the wall's rings every ``pitch`` from 12 mm up to 60 % of the bowl."""
        a = self.hole_cd * math.pi * (0.5 * self.hole_d) ** 2
        g = np.arange(-self.r_base, self.r_base, self.pitch)
        x, y = np.meshgrid(g, g)
        r2 = (x * x + y * y)[x * x + y * y < (self.r_base - 0.006) ** 2]
        rings = [(0.0, a * r2.size, float(np.sqrt(r2.mean())))]
        for z in np.arange(0.012, 0.6 * self.height, self.pitch):              # their z above the bowl's base
            r = float(self.outer_r(z)) - 0.5 * self.wall
            rings.append((float(z - self.bottom), a * int(2.0 * math.pi * r / self.pitch), r))
        return tuple(rings)

    def extra_volume(self) -> float:
        return 1.0                    # the foot ring and handles carry no mass (theirs: mass 0); their shapes collide

    def build_extra(self, b: newton.ModelBuilder, body: int, label: str, density: float) -> None:
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=self.mu)
        z0 = -self.bottom - self.foot                                          # the ring's underside
        for k in range(16):
            a = 2.0 * math.pi * (k + 0.5) / 16
            b.add_shape_box(body, xform=xform((0.070 * math.cos(a), 0.070 * math.sin(a), z0 + 0.5 * self.foot),
                                              (0.0, 0.0, math.sin(0.5 * a), math.cos(0.5 * a))),
                            hx=0.0015, hy=0.0140, hz=0.5 * self.foot, cfg=cfg, label=f"{label}_foot{k}")
        for sgn in (1.0, -1.0):
            b.add_shape_box(body, xform=xform((sgn * (self.r_rim + 0.018), 0.0, self.depth - 0.004)), hx=0.022,
                            hy=0.030, hz=0.003, cfg=cfg, label=f"{label}_handle{'p' if sgn > 0 else 'n'}")


def tomato_starts(floor: np.ndarray) -> np.ndarray:
    """``[15, 3]`` tomato centres: rings 1 + 6 + 8 at 0, 32, 64 mm, resting 1 mm over the inner ``floor``."""
    pts = [(0.0, 0.0)]
    for n, r in ((6, 0.032), (8, 0.064)):
        pts += [(r * math.cos(2.0 * math.pi * k / n), r * math.sin(2.0 * math.pi * k / n)) for k in range(n)]
    return np.array([(floor[0] + x, floor[1] + y, floor[2] + TOMATO_R + 0.001) for x, y in pts])


@dataclass(frozen=True)
class PrepSink:
    """The scene. ``ds`` scales every chart's cells (theirs: 4 mm vessels, 5 mm board)."""

    colander: Colander = field(default_factory=Colander)
    bowl: SteelBowl = field(default_factory=SteelBowl)
    board_cell: float = 0.005

    def colander_start(self) -> np.ndarray:
        """The colander's origin (inner floor centre) on the drainboard."""
        c = self.colander
        return np.array([COLANDER_AT[0], COLANDER_AT[1], COLANDER_AT[2] + c.foot + c.bottom])

    def bowl_start(self) -> np.ndarray:
        return np.array([BOWL_AT[0], BOWL_AT[1], TABLE[2] + self.bowl.bottom])

    def sink_floor(self) -> np.ndarray:
        return np.array([0.5 * SINK[0], 0.5 * SINK[1], RIM_Z - SINK[2]])

    def nodes(self) -> tuple[Node, ...]:
        area, c, w = SINK[0] * SINK[1], self.colander, self.bowl
        return (Node("sink", medium="tank", volume=0.0, area=area, capacity=area * SINK[2], species=(0.0,)),
                Node("colander_cavity", medium="tank", volume=0.0, area=math.pi * float(c.radius(c.depth)) ** 2,
                     capacity=c.cavity_volume(), species=(0.0,)),
                Node("bowl_cavity", medium="tank", volume=0.0, area=math.pi * float(w.radius(w.depth)) ** 2,
                     capacity=w.cavity_volume(), species=(0.0,)),
                Node("line", medium="liquid", volume=LINE, species=(0.0,)))

    def edges(self) -> tuple[Edge, ...]:
        drain = DRAIN_CD * math.pi * DRAIN_D ** 2 / 4.0
        return (Edge("drain", "orifice", "sink", cd_area=drain, density=998.0, dp_lam=1.0),
                tank_overflow(self.nodes()[0], name="sink_overflow"),
                Edge("regulator", "source", "line", flow=FAUCET, species=(0.0,)),
                Edge("faucet", "orifice", "line", reservoir=P_ATM, cd_area=FAUCET_EXIT, dp_lam=1.0),
                Edge("colander_pour", "source", "colander_cavity", flow=-POUR_MAX))

    def pools(self) -> tuple[Pool, ...]:
        half = (0.5 * SINK[0], 0.5 * SINK[1])
        c = self.colander
        return (Pool("sink", floor=tuple(self.sink_floor()), half=half),
                Pool("colander_cavity", body="colander", cavity=c.cavity(), flood=True, spill="colander_pour",
                     holes=c.holes(), holes_to=None),
                Pool("bowl_cavity", body="bowl", cavity=self.bowl.cavity(), flood=True))

    def network(self) -> dict:
        """The plumbing (the drain starts open, the faucet's regulator closed: 2.2 gpm when open, out of its 16 mm
        outlet) and the tomatoes as floats."""
        floats = [Float(f"tomato{k}", sphere=TOMATO_VOL_R) for k in range(TOMATOES)]
        return {"nodes": list(self.nodes()), "edges": list(self.edges()), "pools": list(self.pools()),
                "species": ("soil",), "floats": floats}

    def closed(self) -> tuple[str, ...]:
        return ("regulator",)

    def drains(self) -> tuple[Drain, ...]:
        """The colander's perforated film (floor and lower wall, both faces) runs into its cavity, whose holes drip."""
        return (Drain("perforated", chart=0, to="colander_cavity", radius=self.colander.perforated()),)

    def nozzles(self) -> tuple[Nozzle, ...]:
        return (Nozzle("faucet", edge="faucet", pos=FAUCET_AT, axis=(0.0, 0.0, -1.0), exit_area=FAUCET_EXIT),
                Nozzle("colander_lip", edge="colander_pour", rim_of="colander_cavity", body="colander",
                       exit_area=POUR_EXIT))

    def charts(self) -> tuple:
        nx, ny = (max(int(round(s / self.board_cell)), 2) for s in BOARD[1][:2])
        return (self.colander.chart("colander"), self.bowl.chart("bowl"),
                PlaneChart(shape="board", cells=(nx, ny), edges=("open",) * 4, contact_angle=BOARD_ANGLE))

    def build(self, b: newton.ModelBuilder) -> None:
        """Static sink, drainboard, table and board; the bowl, the colander and the tomatoes free."""
        b.rigid_gap = RIGID_GAP
        self.build_fixed(b)
        self.bowl.build(b, "bowl", xform(self.bowl_start()))
        self.colander.build(b, "colander", xform(self.colander_start()))
        rho = TOMATO_MASS / (4.0 / 3.0 * math.pi * TOMATO_R ** 3)
        cfg = newton.ModelBuilder.ShapeConfig(density=rho, mu=0.40, mu_rolling=5.0e-4, mu_torsional=1.0e-3)
        for k, p in enumerate(tomato_starts(self.colander_start())):
            body = b.add_body(xform=xform(p), label=f"tomato{k}")
            b.add_shape_sphere(body, radius=TOMATO_R, cfg=cfg, label=f"tomato{k}_flesh")

    def build_fixed(self, b: newton.ModelBuilder) -> None:
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.30)
        c, (hx, hy, hz), t = self.sink_floor(), (0.5 * SINK[0], 0.5 * SINK[1], 0.5 * SINK[2]), SINK_WALL
        b.add_shape_box(-1, xform=xform(c - [0.0, 0.0, 0.5 * t]), hx=hx + t, hy=hy + t, hz=0.5 * t, cfg=cfg,
                        label="sink_floor")
        for sx in (-1.0, 1.0):
            b.add_shape_box(-1, xform=xform(c + [sx * (hx + 0.5 * t), 0.0, hz]), hx=0.5 * t, hy=hy + t, hz=hz,
                            cfg=cfg, label=f"sink_wall_x{int(sx)}")
        for sy in (-1.0, 1.0):
            b.add_shape_box(-1, xform=xform(c + [0.0, sy * (hy + 0.5 * t), hz]), hx=hx, hy=0.5 * t, hz=hz, cfg=cfg,
                            label=f"sink_wall_y{int(sy)}")
        top = COLANDER_AT[2] - 5.0e-4                          # the drainboard (their 6 mm pitch left out)
        b.add_shape_box(-1, xform=xform((hx, -hy - t, top - 0.01)), hx=hx + t, hy=hy, hz=0.01, cfg=cfg,
                        label="drainboard")
        (x0, x1), (y0, y1), zt = TABLE
        b.add_shape_box(-1, xform=xform((0.5 * (x0 + x1), 0.5 * (y0 + y1), zt - 0.02)), hx=0.5 * (x1 - x0),
                        hy=0.5 * (y1 - y0), hz=0.02, cfg=cfg, label="table")
        (bx, by), (sx_, sy_, sz) = BOARD
        b.add_shape_box(-1, xform=xform((bx, by, zt + 0.5 * sz)), hx=0.5 * sx_, hy=0.5 * sy_, hz=0.5 * sz,
                        cfg=newton.ModelBuilder.ShapeConfig(mu=0.40), label="board")


__all__ = ["BOARD", "Colander", "Cup", "FAUCET_AT", "PrepSink", "RIM_Z", "SINK", "SKIN_HOLD", "STEEL", "SteelBowl",
           "TABLE", "TOMATOES", "TOMATO_MASS", "TOMATO_R", "TOMATO_VOL_R", "tomato_starts"]
