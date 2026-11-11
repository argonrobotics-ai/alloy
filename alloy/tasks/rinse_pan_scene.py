# SPDX-License-Identifier: Apache-2.0
"""Scene of task rinse_pan (Gym-Anything's scrub-rinse-pan): the pot sink (``rinse_wash_station``), a 10 in aluminium
fry pan in the wash bowl whose inside is a cavity pool, and the drying shelf over the clean drainboard.

Sources (``surface_rinsing`` in Gym-Anything's RoboWorld; file:line):
- pan: ``rinse_vessels.Pan`` (``params.py:193``, ``ware.py:416-437``); water chart in regions inner / rim / outer / foot
  (``ware.py:141-147, 428``); aluminium: contact angle 75° (``params.py:106``), friction 0.30 against steel
  (``ware.py:429``);
- bowls filled 10 in: wash 43 °C with 0.75 g/L surfactant, rinse 43 °C, sanitizer 24 °C with 100 ppm chlorine,
  drains closed: ``layout.py:433-443``;
- the pan starts in the wash bowl, BOWL_Z0 + 0.5 mm, 55 mm behind the bowl's centre, handle toward the apron (their
  yaw -90°), fully under and full of wash water: ``layout.py:464-472``; the sponge on the back ledge behind the rinse
  bowl: ``layout.py:461-463``; the shelf over the clean drainboard at z 1.50-1.65: ``task.py:570-576``;
- the sponge: a 111 x 66 x 40 mm block whose scour face (its body origin, as theirs) is a ceiling ``Applicator`` on the
  pan's chart (WP15's implicit pad on revolution charts; WP9's Archard abrasion reads its pressure and slide; their
  scour face abrades 4x the foam's k_abr, ``params.py:207-218``). Its rigid collider sits 5 mm above the face (the
  pad, about 2 mm deep at 15 N, is the contact with the pan; a collider at the face grazed the bowl floor) and does
  not collide with the pan's shapes. Pores: WP9's sponge store (G5, 11-12); none yet;
- the pre-rinse spray valve (``rinse_ware.SprayValve``, 60 psi, nine 0.6 mm jets) parked over the rinse bowl, for the
  egg on the band where the floor meets the wall, which a flat pad cannot reach. The supply warns that "revolution
  chart 0 has no shape: its whole body is exempt from occluding jets". That is intended: the pan is sprayed from
  inside, and its own wall boxes would block the jets that graze the wall band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from alloy._vendor import newton

from alloy.plugins.fluid import P_ATM, Edge, Node
from alloy.plugins.surface_flow.applicators import Applicator
from alloy.plugins.surface_flow.supply import Nozzle
from alloy.tasks.rinse_sink import BOWL
from alloy.tasks.rinse_vessels import Pan, xform
from alloy.tasks.rinse_ware import SprayValve
from alloy.tasks.rinse_wash_station import CHLORINE, SOAP, SPECIES, TEMPS, Y_BACK, Item, WashStation

CAVITY = "pan_cavity"
SPONGE = (0.0555, 0.033, 0.020)       # half extents [m] (111 x 66 x 40 mm)
SPONGE_MASS = 0.08                    # kg, wet
SPONGE_LIFT = 0.005                   # m, its collider's bottom above the scour face


@dataclass(frozen=True)
class Station(WashStation):
    """The sink with the pan in the wash bowl, and the shelf."""

    pan: Pan = field(default_factory=Pan)
    valve: SprayValve = field(default_factory=SprayValve)
    shelf_z: float = 1.52                # the shelf's top (theirs: 1.50-1.52 m)

    def pan_start(self) -> np.ndarray:
        """The pan's origin (inner floor centre) resting in the wash bowl, 55 mm behind its centre so the handle
        (toward the apron, -y) clears the front wall (their (2.8382, 5.036) in a bowl centred at y 4.981)."""
        return self.bowl_centre(0) + np.array([0.0, 0.055, self.pan.bottom + 5.0e-4])

    def pan_quat(self) -> tuple[float, float, float, float]:
        """Handle (body +x) toward the apron (-y): a turn of -90° about z."""
        return (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))

    def items(self) -> tuple[Item, ...]:
        return (Item("pan", self.pan, tuple(self.pan_start()), self.pan_quat(), fill_from="wash"),)

    def sponge_start(self) -> np.ndarray:
        """The sponge body's origin (its pad's face, ``SPONGE_LIFT`` under the collider) with the scour face (the
        collider's bottom) 0.5 mm over the ledge behind the rinse bowl, face down, where theirs starts
        (``layout.py:459-463``: 0.5·(BOWL_Y1 + Y - 0.02), RIM + 0.5 mm; no jitter)."""
        return np.array([self.sink.bowl_x(1), 0.5 * (BOWL[1] + Y_BACK - 0.02), self.sink.rim_z - SPONGE_LIFT + 5.0e-4])

    @staticmethod
    def scour_face(pose: np.ndarray) -> np.ndarray:
        """The scour face's centre (their sponge origin, task.py:582) of a sponge at body ``pose`` (x, y, z, qx, qy,
        qz, qw): the collider's bottom, ``SPONGE_LIFT`` up the body's z from the pad's face (the pad hangs below it
        so it can press the film; at rest the face is on the deck as theirs)."""
        x, y, z, w = (float(v) for v in pose[3:7])
        up = np.array([2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)])
        return np.asarray(pose[:3], np.float64) + SPONGE_LIFT * up

    def valve_park(self) -> np.ndarray:
        """The valve's tip 0.45 m over the rinse bowl's centre, spraying down."""
        return self.bowl_centre(1) + np.array([0.0, 0.0, BOWL[2] + 0.45])

    def tools(self) -> tuple[Applicator, ...]:
        return (Applicator(body="sponge", shape="rect", size=SPONGE[:2], center=(0.0, 0.0, 0.0),
                           normal=(0.0, 0.0, -1.0), chart=0, implicit=True, friction=0.4),)

    def nodes(self) -> tuple[Node, ...]:
        return (*super().nodes(), Node("line", medium="liquid", volume=1.0e-5))

    def edges(self) -> tuple[Edge, ...]:
        v = self.valve
        out = [Edge("valve", "orifice", "line", reservoir=P_ATM + v.supply_gauge, cd_area=v.valve_cd_area, dp_lam=1.0)]
        out += [Edge(f"jet{k}", "orifice", "line", reservoir=P_ATM, cd_area=v.jet_area, dp_lam=1.0) for k in range(9)]
        return (*super().edges(), *out)

    def nozzles(self) -> tuple[Nozzle, ...]:
        return (*super().nozzles(), *self.valve.nozzles("valve"))

    def closed(self) -> tuple[str, ...]:
        return (*super().closed(), "valve")

    def build(self, b: newton.ModelBuilder) -> None:
        """The sink, the wall shelf (theirs, ``layout.py:170-172``: from the clean drainboard's start to 0.30 m past
        its end, 0.30 m deep from the back wall, 20 mm thick, its top at ``shelf_z``), the pan, the sponge on the
        ledge (not colliding with the pan: its pad is that contact), the valve parked over the rinse bowl."""
        self.build_sink(b)
        x0 = self.sink.x0 + 3 * BOWL[0]
        x1 = x0 + 24 * 0.0254 + 0.30
        shelf = np.array([0.5 * (x0 + x1), Y_BACK - 0.15, self.shelf_z - 0.01])
        b.add_shape_box(-1, xform=xform(shelf), hx=0.5 * (x1 - x0), hy=0.15, hz=0.01,
                        cfg=newton.ModelBuilder.ShapeConfig(mu=0.30), label="shelf")
        self.build_items(b)
        pan = b.body_label.index("pan")
        pan_shapes = [k for k, sb in enumerate(b.shape_body) if sb == pan]
        body = b.add_body(xform=xform(self.sponge_start()), label="sponge")
        hx, hy, hz = SPONGE[0], SPONGE[1], SPONGE[2] - 0.5 * SPONGE_LIFT
        rho = SPONGE_MASS / (8.0 * hx * hy * hz)
        s = b.add_shape_box(body, xform=xform((0.0, 0.0, SPONGE_LIFT + hz)), hx=hx, hy=hy, hz=hz,
                            cfg=newton.ModelBuilder.ShapeConfig(density=float(rho), mu=0.6), label="sponge_block")
        for k in pan_shapes:
            b.add_shape_collision_filter_pair(s, k)
        self.valve.build(b, "valve", xform(self.valve_park(), (1.0, 0.0, 0.0, 0.0)))


__all__ = ["CAVITY", "CHLORINE", "Pan", "SOAP", "SPECIES", "SPONGE", "SPONGE_LIFT", "Station", "TEMPS", "Y_BACK"]
