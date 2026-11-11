# SPDX-License-Identifier: Apache-2.0
"""Ware and the pre-rinse spray valve for the rinse_* proof tasks: a 9 in china plate with its face as a surface-flow
chart, the peg rack it stands in, and the 9-jet pre-rinse spray valve with its plumbing.

Sources (Gym-Anything's RoboWorld ``surface_rinsing`` environment; file:line):
- plate: 9 in vitrified china, R 114.3 mm, a flat well of radius 88.9 mm, 12 mm deep, body 4.5 mm, 22 mm tall,
  0.47 kg, glaze (water contact angle 25°): ``params.py:187``, ``params.py:104``, ``ware.py:203-226``;
- peg rack: 0.5 m square, rows of posts (r 4 mm, 66 mm tall) 35 mm apart, plates standing face to -y leaning back
  6-12°: ``ware.py:384-412``, ``scene.py:433-451``;
- spray valve (T&S B-0107 class): 1.15 gpm at 60 psi through nine 0.60 mm (effective) orifices, one central and eight
  on a 6 mm ring, in a 7° cone, 28 m/s, 0.55 kg: ``params.py:49-57``; ``water/network.py:93-103``;
- pre-rinse sink: 20 x 20 x 5 in, 35 mm drain, open: ``layout.py:50-53, 344-345``.

Model choices (proof v1): the plate collides as a 6 mm disc (its well and foot are not resolved for contact); its
face (well floor and rim) is a ``RevolutionChart`` on the plate body whose outflow drains to the pre-rinse sink's
tank. The chart leaves the centre uncharted inside ``r_min`` (``ds·n_theta/4π``; a cap chart is phase 2): rays there
land on the innermost ring, and the inner edge is open too (film that would cross the centre drains, booked;
closed, a held vertical plate pooled 9 mm of water against it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import warp as wp

from alloy._vendor import newton

from alloy.plugins.fluid import P_ATM
from alloy.plugins.surface_flow import RevolutionChart
from alloy.plugins.surface_flow.supply import Drain, Nozzle

INCH = 0.0254
PSI = 6894.757


@dataclass(frozen=True)
class Plate:
    """A 9 in china plate; the body frame's z is the face normal, its origin the disc's centre."""

    radius: float = 0.1143            # m
    well_radius: float = 0.0889       # m, the flat well
    well_depth: float = 0.012         # m, rim top above the well floor
    thickness: float = 0.006          # m, the collision disc (body 4.5 mm; the foot ring is not resolved)
    mass: float = 0.47                # kg
    contact_angle: float = 25.0       # deg, water on glaze
    ds: float = 2.0e-3                # m, chart cells along the profile
    n_theta: int = 64

    @property
    def r_min(self) -> float:
        """Innermost charted radius: the off-axis rule r >= ds·n_theta/(4π), plus 2 %."""
        return 1.02 * self.ds * self.n_theta / (4.0 * math.pi)

    def profile(self) -> tuple[tuple[float, float], ...]:
        """The face from the well's centre out over the rim (material below: normal +z on the well floor)."""
        z0 = 0.5 * self.thickness
        return ((self.r_min, z0), (self.well_radius, z0), (self.radius, z0 + self.well_depth))

    def chart(self, body: str) -> RevolutionChart:
        return RevolutionChart(profile=self.profile(), n_theta=self.n_theta, ds=self.ds, body=body,
                               ends=("open", "open"), contact_angle=self.contact_angle, shape=f"{body}_disc")

    def build(self, b: newton.ModelBuilder, label: str, xform: wp.transform) -> int:
        """A free plate body at ``xform`` (a disc collision shape labelled ``<label>_disc``)."""
        body = b.add_body(xform=xform, label=label)
        hz = 0.5 * self.thickness
        rho = self.mass / (math.pi * self.radius ** 2 * self.thickness)
        cfg = newton.ModelBuilder.ShapeConfig(density=rho, mu=0.30)
        b.add_shape_cylinder(body, radius=self.radius, half_height=hz, cfg=cfg, label=f"{label}_disc")
        return body


@dataclass(frozen=True)
class SprayValve:
    """The pre-rinse spray valve and its plumbing: mains -> squeeze valve -> line -> nine jets; the pre-rinse sink is
    a tank whose open drain empties it. The valve body's z axis is the spray direction; the jets leave its tip at
    the body origin."""

    supply_gauge: float = 60.0 * PSI  # Pa above the atmosphere (static)
    jet_d: float = 0.60e-3            # m, effective orifice diameter
    ring: float = 6.0e-3              # m, ring of eight orifices
    cone: float = math.radians(7.0)   # full cone of the ring jets
    valve_cd_area: float = 1.66e-5    # m²: Cv 0.98 drops 1.38 psi at 1.15 gpm
    mass: float = 0.55                # kg
    sink_center: tuple[float, float, float] = (0.0, 0.0, 0.0)   # the sink floor's centre (world)
    sink_half: float = 10.0 * INCH
    sink_depth: float = 5.0 * INCH
    drain_d: float = 0.035

    @property
    def jet_area(self) -> float:
        return math.pi * self.jet_d ** 2 / 4.0

    def jets(self) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """(position, axis) of the nine jets in the valve frame."""
        out = [((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))]
        s, c = math.sin(0.5 * self.cone), math.cos(0.5 * self.cone)
        for k in range(8):
            phi = 2.0 * math.pi * k / 8.0
            out.append(((self.ring * math.cos(phi), self.ring * math.sin(phi), 0.0),
                        (s * math.cos(phi), s * math.sin(phi), c)))
        return out

    def nozzles(self, body: str) -> tuple[Nozzle, ...]:
        return tuple(Nozzle(f"jet{k}", edge=f"jet{k}", pos=p, axis=a, body=body, pattern="jet",
                            exit_area=self.jet_area, reach=1.0) for k, (p, a) in enumerate(self.jets()))

    def network(self) -> dict:
        area = (2.0 * self.sink_half) ** 2
        nodes = [{"name": "line", "medium": "liquid", "volume": 1.0e-5},
                 {"name": "sink", "medium": "tank", "volume": 0.0, "area": area, "capacity": area * self.sink_depth}]
        edges = [{"name": "valve", "kind": "orifice", "a": "line", "reservoir": P_ATM + self.supply_gauge,
                  "cd_area": self.valve_cd_area, "dp_lam": 1.0}]
        edges += [{"name": f"jet{k}", "kind": "orifice", "a": "line", "reservoir": P_ATM, "cd_area": self.jet_area,
                   "dp_lam": 1.0} for k in range(9)]
        edges.append({"name": "sink_drain", "kind": "orifice", "a": "sink", "reservoir": P_ATM,
                      "cd_area": 0.6 * math.pi * self.drain_d ** 2 / 4.0, "dp_lam": 1.0})
        pools = [{"tank": "sink", "floor": self.sink_center, "half": (self.sink_half, self.sink_half)}]
        return {"nodes": nodes, "edges": edges, "pools": pools}

    def drains(self, charts: int) -> tuple[Drain, ...]:
        """Each chart's open edges (a plate's rim) drain into the sink."""
        return tuple(Drain(f"rim{k}", chart=k, to="sink", open_edges=True) for k in range(charts))

    def build(self, b: newton.ModelBuilder, label: str, xform: wp.transform) -> int:
        """The valve body: a 40 mm x 100 mm cylinder behind its tip (so the jets do not start inside it)."""
        body = b.add_body(xform=xform, label=label)
        r, hz = 0.02, 0.05
        rho = self.mass / (math.pi * r * r * 2.0 * hz)
        b.add_shape_cylinder(body, xform=wp.transform(wp.vec3(0.0, 0.0, -hz - 0.005), wp.quat_identity()),
                             radius=r, half_height=hz, cfg=newton.ModelBuilder.ShapeConfig(density=rho),
                             label=f"{label}_body")
        return body


__all__ = ["Plate", "SprayValve"]
