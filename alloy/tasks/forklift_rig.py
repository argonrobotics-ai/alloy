# SPDX-License-Identifier: Apache-2.0
"""The forklift for task_forklift_pick: a 2.9 t counterbalance truck on four virtual wheels, a mast lift on a hydraulic
cylinder, and a 725 kg loaded EUR pallet.

The truck:
- the chassis body (main frame plus a rear counterweight) rides on the ``vehicle`` plug-in's wheels: front drive
  wheels of 0.3 m radius on a 0.9 m track, rear steer wheels of 0.23 m on a 0.8 m track, 1.4 m wheelbase. The rear
  wheels steer opposite to the input, as on a forklift. Travel motor 60 N·m at 20:1 (top speed about 6 m/s).
- the fork carriage (a plate and two 1 m tines, 150 kg) slides on a prismatic joint ``mast_lift`` at the chassis
  front. A single-acting 60 mm hydraulic cylinder from the ``fluid_network`` plug-in lifts it: a pump of
  0.85 L/s (opening = command), a lowering orifice to the tank, a relief valve at 200 bar, and 2 L of hose volume.

The pallet: a 1.2 x 0.8 m deck on three stringers (fork pockets 100 mm high between them) carrying a 700 kg load.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from alloy._vendor import newton
import warp as wp

from alloy.plugins.fluid import P_ATM, Edge, Node
from alloy.plugins.fluid.actuators import Cylinder
from alloy.plugins.harness.scene import body_xform
from alloy.plugins.library.fluid_network import FluidNetwork
from alloy.plugins.vehicle import Vehicle

TRUCK, FORKS, PALLET = "forklift", "forks", "pallet"
FRONT_R, REAR_R = 0.3, 0.23
MAST_X = 0.75                     # m: carriage plate, chassis frame
TINE_X = (0.025, 1.025)           # tine span ahead of the carriage origin [m]
FORK_LOW = 0.03                   # m: tine bottom above the ground at the lowest mast position
BORE = 0.06                       # m
OIL = 850.0                       # kg/m³


@dataclass(frozen=True)
class Placement:
    """Where the truck starts and where the pallet sits (x along the aisle; both at y = 0)."""

    truck_x: float = -2.4
    pallet_x: float = 0.0

    @property
    def insert_x(self) -> float:
        """Chassis x with the carriage plate at the pallet's near edge (forks fully in)."""
        return self.pallet_x - 0.6 - (MAST_X + TINE_X[0])


def vehicle() -> Vehicle:
    return Vehicle(chassis=TRUCK, positions=((0.5, 0.45, 0.0), (0.5, -0.45, 0.0), (-0.9, 0.4, REAR_R - FRONT_R),
                                             (-0.9, -0.4, REAR_R - FRONT_R)),
                   radius=(FRONT_R, FRONT_R, REAR_R, REAR_R), steer=(0.0, 0.0, -1.0, -1.0),
                   driven=(True, True, False, False), inertia=(2.0, 2.0, 0.8, 0.8), radial_stiffness=(1.0e6,),
                   radial_damping=(1.5e4,), brake_torque=(3000.0,), mu=0.8, rolling_resistance=0.02,
                   motor_torque=60.0, motor_free_speed=400.0, motor_inertia=0.05, gear=20.0)


def lift_network() -> FluidNetwork:
    area = math.pi * BORE**2 / 4.0
    return FluidNetwork(
        nodes=(Node("lift", medium="liquid", volume=2.0e-3),),
        edges=(Edge("pump", "source", "lift", flow=0.3 * area),                               # 0.3 m/s at full command
               Edge("lower", "orifice", "lift", reservoir=P_ATM, cd_area=1.5e-5, density=OIL),
               Edge("relief", "relief", "lift", conductance=1.0e-9, p_set=200.0e5)),
        actuators=(Cylinder("lift", joint="mast_lift", a="lift", bore=BORE, stroke=1.5, dead_a=2.0e-3),),
        openings=(("pump", 0.0), ("lower", 0.0), ("relief", 1.0)))


def build(place: Placement):
    """The truck (articulated: free chassis + prismatic carriage) and the loaded pallet, for a ``TestScene``."""
    def scene(b: newton.ModelBuilder) -> None:
        z0 = FRONT_R
        chassis = b.add_link(xform=body_xform(place.truck_x, 0.0, z0), label=TRUCK)
        b.add_shape_box(chassis, xform=body_xform(-0.05, 0.0, 0.35), hx=0.65, hy=0.5, hz=0.35,
                        cfg=newton.ModelBuilder.ShapeConfig(density=1500.0))
        b.add_shape_box(chassis, xform=body_xform(-0.85, 0.0, 0.45), hx=0.15, hy=0.5, hz=0.35,
                        cfg=newton.ModelBuilder.ShapeConfig(density=6000.0))           # counterweight
        free = b.add_joint_free(chassis)
        carriage = b.add_link(xform=body_xform(place.truck_x + MAST_X, 0.0, FORK_LOW), label=FORKS)
        steel = newton.ModelBuilder.ShapeConfig(density=7800.0 * 0.15, mu=0.4)
        b.add_shape_box(carriage, xform=body_xform(0.0, 0.0, 0.3), hx=0.025, hy=0.5, hz=0.27, cfg=steel)
        for y in (-0.175, 0.175):
            b.add_shape_box(carriage, xform=body_xform(sum(TINE_X) / 2.0, y, 0.02), hx=(TINE_X[1] - TINE_X[0]) / 2.0,
                            hy=0.06, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.4))
        lift = b.add_joint_prismatic(chassis, carriage, parent_xform=body_xform(MAST_X, 0.0, FORK_LOW - z0),
                                     child_xform=wp.transform_identity(), axis=(0.0, 0.0, 1.0), limit_lower=0.0,
                                     limit_upper=1.5, limit_ke=1.0e7, limit_kd=2.0e5, label="mast_lift")
        b.add_articulation([free, lift], label=TRUCK)
        _pallet(b, place.pallet_x)
    return scene


def _pallet(b: newton.ModelBuilder, x: float) -> None:
    pallet = b.add_body(xform=body_xform(x, 0.0, 0.0), label=PALLET)
    wood = newton.ModelBuilder.ShapeConfig(density=600.0, mu=0.5)
    b.add_shape_box(pallet, xform=body_xform(0.0, 0.0, 0.115), hx=0.6, hy=0.4, hz=0.015, cfg=wood)
    for y in (-0.35, 0.0, 0.35):
        b.add_shape_box(pallet, xform=body_xform(0.0, y, 0.05), hx=0.6, hy=0.05, hz=0.05, cfg=wood)
    b.add_shape_box(pallet, xform=body_xform(0.0, 0.0, 0.43), hx=0.5, hy=0.35, hz=0.3,
                    cfg=newton.ModelBuilder.ShapeConfig(density=700.0 / 0.42, mu=0.5))     # the 700 kg load


__all__ = ["FORKS", "PALLET", "TRUCK", "Placement", "build", "lift_network", "vehicle"]
