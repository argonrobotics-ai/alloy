# SPDX-License-Identifier: Apache-2.0
"""The three-bowl pot sink as flooding pools, with vessels whose insides are cavity pools: the shared station of the
rinse_* tasks that dunk ware (rinse_pan, rinse_glassware).

Each bowl is a network tank (``rinse_sink.PotSink``: fills, basket drains, rim overflows, the faucet) and a flooding
``Pool`` over its inner box; static boxes are its floor and walls, and the ledge deck behind the bowls tops at the
rim. Each vessel ``Item`` brings its chart (``rinse_vessels``), a cavity tank ``<name>_cavity`` (``Pool(cavity=...)``,
flooding, spilling over its rim's lowest point through the source edge ``<name>_pour`` and the nozzle
``<name>_lip``, exchanging over its rim with the bowls named in ``weir``), and may start full of a bowl's water (taken
from that bowl). Species: soap (wash 0.75 kg/m³), chlorine (sanitizer 0.1 kg/m³), soil.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from alloy._vendor import newton

from alloy.plugins.fluid import Edge, Node, tank_overflow
from alloy.plugins.surface_flow.supply import Nozzle, Pool
from alloy.tasks.rinse_sink import BOWL, BOWLS, DRAIN_CD, DRAIN_D, FAUCET, WALL, PotSink, drain_edge, overflow_edge
from alloy.tasks.rinse_vessels import Vessel, xform

SPECIES = ("soap", "chlorine", "soil")
SOAP = 0.75                           # kg/m³ in the wash (0.75 g/L)
CHLORINE = 0.10                       # kg/m³ in the sanitizer (100 ppm)
CONC = {"wash": (SOAP, 0.0, 0.0), "rinse": (0.0, 0.0, 0.0), "sanitize": (0.0, CHLORINE, 0.0)}
TEMPS = {"wash": 43.0, "rinse": 43.0, "sanitize": 24.0}      # °C (layout.py:433-443)
Y_BACK = 5.4 - 4.676                  # m, the back wall: their L.Y with the bowls' front (their 4.676) at y = 0
POUR_MAX = 5.0e-3                     # m³/s: a pour over a rim at full opening (a cap; the weir law sets it)
POUR_EXIT = 2.0e-3                    # m², a pour's exit area: speed Q/A, about sqrt(2 g H) at 1 L/s (1e-4 m² sent
                                      # 5 L/s at 50 m/s and its 250 N reaction threw a pan out of the hand)
CONTACTS = {"nconmax": 768, "njmax": 3072}   # MuJoCo Warp capacity per world (default 128): six glasses at rest
                                              # make 240 contacts; the runners treat an overflow as a fault
RIGID_GAP = 5.0e-3                    # m, each shape's contact detection gap (Newton's builder default, 0.1 m, made
                                      # every pair of boxes within 0.2 m a contact: 4096 in one six-glass world)


@dataclass(frozen=True)
class Item:
    """A vessel ``name`` at ``pos`` / ``quat`` (x, y, z, w); ``fill_from``: starts full of that bowl's water."""

    name: str
    vessel: Vessel
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    fill_from: str | None = None

    @property
    def cavity(self) -> str:
        return f"{self.name}_cavity"


@dataclass(frozen=True)
class WashStation:
    """The sink and its vessels (world frame; the sink's apron at y = 0). Subclasses list ``items()``."""

    sink: PotSink = field(default_factory=PotSink)
    weir: str | tuple[str, ...] = "wash"   # DC5's several-pool weir (11-08) takes ("wash", "rinse", "sanitize")

    def items(self) -> tuple[Item, ...]:
        return ()

    @property
    def floor_z(self) -> float:
        return self.sink.rim_z - BOWL[2]

    def bowl_centre(self, k: int) -> np.ndarray:
        return np.array([self.sink.bowl_x(k), 0.5 * BOWL[1], self.floor_z])

    def nodes(self) -> tuple[Node, ...]:
        s, its = self.sink, self.items()
        taken = {b: sum(i.vessel.cavity_volume() for i in its if i.fill_from == b) for b in BOWLS}
        out = [Node(b, medium="tank", volume=s.area * d - taken[b], area=s.area, capacity=s.capacity,
                    species=CONC[b]) for b, d in zip(BOWLS, s.fill, strict=True)]
        for i in its:
            v = i.vessel.cavity_volume()
            out.append(Node(i.cavity, medium="tank", volume=v if i.fill_from else 0.0,
                            area=math.pi * float(i.vessel.radius(i.vessel.depth)) ** 2, capacity=v,
                            species=CONC[i.fill_from] if i.fill_from else (0.0, 0.0, 0.0)))
        return tuple(out)

    def edges(self) -> tuple[Edge, ...]:
        drain = DRAIN_CD * math.pi * DRAIN_D ** 2 / 4.0
        out = [Edge(drain_edge(b), "orifice", b, cd_area=drain, density=998.0, dp_lam=1.0) for b in BOWLS]
        out += [tank_overflow(n, name=overflow_edge(n.name)) for n in self.nodes()[:len(BOWLS)]]
        out.append(Edge("faucet", "source", "wash", flow=FAUCET, species=(0.0, 0.0, 0.0)))
        out += [Edge(f"{i.name}_pour", "source", i.cavity, flow=-POUR_MAX) for i in self.items()]
        return tuple(out)

    def pools(self) -> tuple[Pool, ...]:
        half = (0.5 * BOWL[0] - WALL, 0.5 * BOWL[1])
        bowls = tuple(Pool(b, floor=tuple(self.bowl_centre(k)), half=half, flood=True) for k, b in enumerate(BOWLS))
        return (*bowls, *(Pool(i.cavity, body=i.name, cavity=i.vessel.cavity(), flood=True, spill=f"{i.name}_pour",
                               weir=self.weir) for i in self.items()))

    def network(self) -> dict:
        """The plumbing; drains, the faucet and the pours start closed (inputs named after their edges)."""
        return {"nodes": list(self.nodes()), "edges": list(self.edges()), "pools": list(self.pools()),
                "species": SPECIES}

    def closed(self) -> tuple[str, ...]:
        return (*(drain_edge(b) for b in BOWLS), "faucet")

    def nozzles(self) -> tuple[Nozzle, ...]:
        return tuple(Nozzle(f"{i.name}_lip", edge=f"{i.name}_pour", rim_of=i.cavity, body=i.name, exit_area=POUR_EXIT)
                     for i in self.items())

    def charts(self) -> tuple:
        return tuple(i.vessel.chart(i.name) for i in self.items())

    def build_sink(self, b: newton.ModelBuilder) -> None:
        """Static bowls (floors and walls) and the ledge deck behind them (its top at the rim). Sets the builder's
        contact gap (``RIGID_GAP``) for every shape added after it."""
        b.rigid_gap = RIGID_GAP
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.30)
        hx, hy, hz = 0.5 * BOWL[0], 0.5 * BOWL[1], 0.5 * BOWL[2]
        for k in range(len(BOWLS)):
            c = self.bowl_centre(k)
            b.add_shape_box(-1, xform=xform(c - [0.0, 0.0, 0.005]), hx=hx, hy=hy, hz=0.005, cfg=cfg,
                            label=f"bowl{k}_floor")
            for sx in (-1.0, 1.0):
                b.add_shape_box(-1, xform=xform(c + [sx * (hx - 0.5 * WALL), 0.0, hz]), hx=0.5 * WALL, hy=hy, hz=hz,
                                cfg=cfg, label=f"bowl{k}_wall_x{int(sx)}")
            for sy in (-1.0, 1.0):
                b.add_shape_box(-1, xform=xform(c + [0.0, sy * (hy + 0.5 * WALL), hz]), hx=hx, hy=0.5 * WALL, hz=hz,
                                cfg=cfg, label=f"bowl{k}_wall_y{int(sy)}")
        ledge = np.array([self.sink.x0 + 1.5 * BOWL[0], 0.5 * (BOWL[1] + Y_BACK), self.sink.rim_z - 0.01])
        b.add_shape_box(-1, xform=xform(ledge), hx=1.5 * BOWL[0], hy=0.5 * (Y_BACK - BOWL[1]), hz=0.01, cfg=cfg,
                        label="ledge")

    def build_items(self, b: newton.ModelBuilder) -> None:
        for i in self.items():
            i.vessel.build(b, i.name, xform(i.pos, i.quat))

    def build(self, b: newton.ModelBuilder) -> None:
        self.build_sink(b)
        self.build_items(b)


__all__ = ["CHLORINE", "CONC", "CONTACTS", "Item", "POUR_EXIT", "POUR_MAX", "RIGID_GAP",
           "SOAP", "SPECIES", "TEMPS", "WashStation", "Y_BACK"]
