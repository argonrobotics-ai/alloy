# SPDX-License-Identifier: Apache-2.0
"""Gym-Anything's forklift truck for the forklift tasks (SCR-WP8-12 F2): Hyster H2.5A numbers (``F/`` =
gym-anything-robotics benchmarks/roboworld/environments/forklift/; F/params.py), on the vehicle plug-in (transient
tyres, oscillating steer axle) with a three-stage full-free-lift mast on the fluid core.

The truck frame: x forward from the front axle, z up from the ground. Bodies and joints, as GA's (F/truck.py:246-259):
- ``forklift`` (chassis, free) and ``axle`` (the steer axle on its pin, ±3.5°: plug-in laws_axle);
- ``mast_outer`` on revolute ``mast_tilt`` (y axis at F/params.py:83's pivot, 6° forward and back, :84-85);
- ``mast_int`` on prismatic ``mast_stage`` (1.69 m, :86) and ``mast_inner`` on ``mast_inner_slide``, tied to it 1:1 by a
  MuJoCo joint equality, as GA's chain (F/truck.py:319): the inner stage rises twice the intermediate's travel;
- ``carriage`` on prismatic ``mast_free`` (free lift 0.96 m, :88) and ``sideshift`` on prismatic ``sideshift_slide``
  (±0.10 m, :89), carrying the two 1 m forks.
Masses (F/params.py:126-131): outer 230, intermediate 160, inner 130, carriage 120, side shift and backrest 70, forks
2 x 42 kg; the axle 150 kg (INFERRED); the chassis takes the rest of the 3,990 kg service weight (:91), its centre of
mass placed so that the whole truck's is 0.935 m behind the front axle and 0.68 m up (:113-114) with the mast vertical
and lowered.

Hydraulics (``hydraulics``): one pump (1.25 L/s: 32.7 cm³/rev at 2,500 rpm, η_v 0.92, :423-424) feeds the lift line,
one fluid node shared (``Node(shared=True)``, a tee) by the free-lift cylinder (73 mm, 2:1 chain: area A/2 on the
carriage, :434) and the two main lift cylinders (50 mm, :433) on the intermediate stage; the lower-pressure stage
moves first, so free lift precedes staging. Two
63 mm tilt cylinders on a 0.75 m lever (:435-436) are a vane of D = 2·A·r on the tilt joint (the lever's cos error is
under 0.6 % over ±6°); a 40 mm cylinder works the side shift (:437). Relief 18.5 MPa (:428).

``build(pose, load, locked=True)`` welds the mast at ``pose`` (fixed joints) for statics laws; ``at`` places the truck
(``forklift_mast.rest_at``: at rest on its tyres, so a law starts without a drop). Closed forms: ``forklift_mast``.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton._src.solvers.mujoco import add_equality_joint

from alloy.plugins.fluid import P_ATM, Edge, Node
from alloy.plugins.fluid.actuators import Cylinder, Vane
from alloy.plugins.harness.scene import body_xform
from alloy.plugins.library.fluid_network import FluidNetwork
from alloy.plugins.vehicle import Vehicle
from alloy.plugins.vehicle import laws_axle as ax
from alloy.plugins.vehicle import laws_grip as lg

TRUCK, AXLE = ax.TRUCK, ax.AXLE
MASS = 3990.0                                          # F/params.py:91
CG = np.array([-0.935, 0.0, 0.68])                     # F/params.py:113-114
PIVOT = np.array([0.17, 0.0, 0.33])                    # F/params.py:83
TILT_MAX = math.radians(6.0)                           # :84-85
STAGE, FREE, SHIFT = 1.69, 0.96, 0.10                  # :86, :88, :89
FORK_FACE = 0.545                                      # m ahead of the front axle (:102, LOAD_DISTANCE)
FORK_LENGTH, FORK_Y, FORK_BOTTOM = 1.0, 0.25, 0.009    # m; the blades reach 9 mm above the floor (:87)
FORK_THICK, FORK_WIDE = 0.04, 0.10
OIL = 850.0
PUMP_FLOW = 32.7e-6 * 2500.0 / 60.0 * 0.92             # m³/s
BORE_FREE, BORE_MAIN, BORE_TILT, BORE_SHIFT = 0.073, 0.050, 0.063, 0.040
TILT_ARM, RELIEF = 0.75, 18.5e6
AREA_FREE = 0.5 * math.pi * BORE_FREE**2 / 4.0          # 2:1 chain: per metre of carriage travel
AREA_MAIN = 2.0 * math.pi * BORE_MAIN**2 / 4.0          # two cylinders
TILT_D = 2.0 * math.pi * BORE_TILT**2 / 4.0 * TILT_ARM  # m³/rad


@dataclasses.dataclass(frozen=True)
class Part:
    body: str
    mass: float
    centre: tuple[float, float, float]                  # its centre of mass, body frame (origin at the pivot)
    half: tuple[float, float, float]                    # one box per part (two rails: half its mass each side)
    rails: float = 0.0                                  # y of a pair of rails (0: one box at y = 0)
    collide: bool = False


# rails clear each other in y; plates and blades sit ahead of the uprights
PARTS = (Part("mast_outer", 230.0, (0.05, 0.0, 0.77), (0.04, 0.025, 1.07), rails=0.40),
         Part("mast_int", 160.0, (0.10, 0.0, 0.77), (0.04, 0.025, 1.07), rails=0.34),
         Part("mast_inner", 130.0, (0.15, 0.0, 0.77), (0.04, 0.025, 1.07), rails=0.28),
         Part("carriage", 120.0, (0.26, 0.0, 0.02), (0.02, 0.45, 0.30)),
         Part("sideshift", 70.0, (0.30, 0.0, 0.0), (0.02, 0.45, 0.20)))
FORK_MASS = 42.0
STACK = ("mast_outer", "mast_int", "mast_inner", "carriage", "sideshift")


@dataclasses.dataclass(frozen=True)
class Pose:
    """The mast's joint coordinates: tilt [rad, + forward], stage travel (each moving stage), free lift, side shift."""

    tilt: float = 0.0
    stage: float = 0.0
    free: float = 0.0
    shift: float = 0.0


@dataclasses.dataclass(frozen=True)
class Load:
    """A box load on the forks: mass, full size, its centre ``centre`` ahead of the fork face."""

    mass: float
    size: tuple[float, float, float] = (1.0, 0.8, 1.0)
    centre: float = 0.5


def _fork_local(k: int) -> np.ndarray:
    """Fork ``k``'s blade centre, side-shift body frame (origin at the pivot at the build pose)."""
    return np.array([FORK_FACE + 0.5 * FORK_LENGTH - PIVOT[0], (-1, 1)[k] * FORK_Y,
                     FORK_BOTTOM + 0.5 * FORK_THICK - PIVOT[2]])


def load_local(load: Load) -> np.ndarray:
    return np.array([FORK_FACE + load.centre - PIVOT[0], 0.0, FORK_BOTTOM + FORK_THICK + 0.5 * load.size[2] - PIVOT[2]])


def body_offset(body: str, pose: Pose) -> np.ndarray:
    """A mast body's origin in the outer upright's frame (stacked prismatic travel)."""
    lift = {"mast_outer": 0.0, "mast_int": pose.stage, "mast_inner": 2.0 * pose.stage,
            "carriage": 2.0 * pose.stage + pose.free, "sideshift": 2.0 * pose.stage + pose.free}[body]
    return np.array([0.0, pose.shift if body == "sideshift" else 0.0, lift])


def _tilt(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def to_truck(local: np.ndarray, body: str, pose: Pose) -> np.ndarray:
    """A point in a mast body's frame -> the truck frame."""
    return PIVOT + _tilt(pose.tilt) @ (body_offset(body, pose) + np.asarray(local, np.float64))


def mast_masses(pose: Pose, load: Load | None = None) -> list[tuple[float, np.ndarray]]:
    """Every mast part's (mass, centre of mass) in the truck frame, forks and load included."""
    out = [(p.mass, to_truck(np.array(p.centre), p.body, pose)) for p in PARTS]
    out += [(FORK_MASS, to_truck(_fork_local(k), "sideshift", pose)) for k in range(2)]
    if load is not None:
        out.append((load.mass, to_truck(load_local(load), "sideshift", pose)))
    return out


def chassis_centre() -> np.ndarray:
    """The chassis's centre of mass that puts the whole truck's at GA's with the mast vertical and lowered."""
    parts = mast_masses(Pose()) + [(ax.AXLE_MASS, ax.PIN)]
    rest = MASS - sum(m for m, _ in parts)
    return (MASS * CG - sum(m * c for m, c in parts)) / rest


def chassis_mass() -> float:
    return MASS - sum(m for m, _ in mast_masses(Pose())) - ax.AXLE_MASS


# --- the model ------------------------------------------------------------------------------------------------------

def _box(b: newton.ModelBuilder, body: int, centre, half, mass: float, collide: bool = True) -> None:
    cfg = newton.ModelBuilder.ShapeConfig(density=float(mass / (8.0 * half[0] * half[1] * half[2])),
                                          has_shape_collision=collide)
    b.add_shape_box(body, xform=body_xform(*(float(v) for v in centre)), hx=half[0], hy=half[1], hz=half[2], cfg=cfg)


def _part_shapes(b: newton.ModelBuilder, body: int, part: Part) -> None:
    if part.rails == 0.0:
        _box(b, body, part.centre, part.half, part.mass, part.collide)
        return
    for y in (-part.rails, part.rails):
        _box(b, body, (part.centre[0], y, part.centre[2]), part.half, 0.5 * part.mass, part.collide)


# The inner stage's chain: MuJoCo's default equality (time constant 20 ms) stretches 3.5 mm under the free-lift
# cylinder's reaction and lets the main stage rise early. At two task steps (MuJoCo's refsafe floor) it stretches
# ~0.05 mm; at a larger step it softens with the step.
CHAIN_SOLREF = (2.0 * lg.STEP, 1.0)
# The mast's end stops (MuJoCo joint limits, implicit): at the relief pressure they give 0.2 mm in all (1e8: 1.8 mm).
STOP_KE, STOP_KD = 1.0e9, 3.0e6

_SPECS = (("mast_outer", "mast_tilt", (0.0, 1.0, 0.0), "tilt", -TILT_MAX, TILT_MAX),
          ("mast_int", "mast_stage", (0.0, 0.0, 1.0), "stage", 0.0, STAGE),
          ("mast_inner", "mast_inner_slide", (0.0, 0.0, 1.0), "stage", 0.0, STAGE),
          ("carriage", "mast_free", (0.0, 0.0, 1.0), "free", 0.0, FREE),
          ("sideshift", "sideshift_slide", (0.0, 1.0, 0.0), "shift", -SHIFT, SHIFT))


def _mast_joint(b: newton.ModelBuilder, parent: int, body: int, spec: tuple, pose: Pose, locked: bool,
                previous: str) -> int:
    """The joint from ``parent`` to the mast body ``body`` (``previous`` names the parent): a revolute or prismatic
    joint at the pose's value inside its limits, or (``locked``) a fixed joint at that pose."""
    name, label, axis, field, lo, hi = spec
    value = float(getattr(pose, field))
    if locked:
        rel = (wp.transform(wp.vec3(*PIVOT), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), value))
               if name == "mast_outer" else body_xform(*(body_offset(name, pose) - body_offset(previous, pose))))
        return b.add_joint_fixed(parent, body, parent_xform=rel, label=label)
    add = b.add_joint_revolute if name == "mast_outer" else b.add_joint_prismatic
    pxf = body_xform(*PIVOT) if name == "mast_outer" else wp.transform_identity()
    joint = add(parent, body, parent_xform=pxf, axis=axis, limit_lower=lo, limit_upper=hi, limit_ke=STOP_KE,
                limit_kd=STOP_KD, label=label)
    b.joint_q[b.joint_q_start[joint]] = value
    return joint


def build(pose: Pose = Pose(), load: Load | None = None, locked: bool = False, at: wp.transform | None = None):
    """The truck at ``at`` (default the origin: front axle at ground level), the mast at ``pose`` (``locked``: welded
    there by fixed joints, for statics laws; else free in its ranges for the hydraulics), with ``load`` on the forks."""
    place = at if at is not None else wp.transform_identity()

    def scene(b: newton.ModelBuilder) -> None:
        chassis = b.add_link(xform=place, label=TRUCK)
        _box(b, chassis, chassis_centre(), (0.9, 0.5, 0.35), chassis_mass())
        joints = [b.add_joint_free(chassis)]
        axle = b.add_link(xform=wp.transform_multiply(place, body_xform(*ax.PIN)), label=AXLE)
        _box(b, axle, (0.0, 0.0, 0.0), (0.075, 0.6, 0.06), ax.AXLE_MASS)
        joints.append(b.add_joint_revolute(chassis, axle, parent_xform=body_xform(*ax.PIN), axis=(1.0, 0.0, 0.0),
                                           limit_lower=-ax.STOP, limit_upper=ax.STOP, limit_ke=1.0e7, limit_kd=1.0e5,
                                           label="axle_pin"))
        outer_xf = wp.transform_multiply(place, wp.transform(wp.vec3(*PIVOT), wp.quat_from_axis_angle(
            wp.vec3(0.0, 1.0, 0.0), pose.tilt)))
        parent, previous, index = chassis, TRUCK, {}
        for spec, part in zip(_SPECS, PARTS, strict=True):
            body = b.add_link(xform=wp.transform_multiply(outer_xf, body_xform(*body_offset(spec[0], pose))),
                              label=spec[0])
            _part_shapes(b, body, part)
            joint = _mast_joint(b, parent, body, spec, pose, locked, previous)
            joints.append(joint)
            index[spec[0]] = (body, joint)
            parent, previous = body, spec[0]
        _forks_and_load(b, index["sideshift"][0], load)
        b.add_articulation(joints, label=TRUCK)
        if not locked:
            add_equality_joint(b, index["mast_inner"][1], index["mast_int"][1], polycoef=(0.0, 1.0),
                               solref=CHAIN_SOLREF, label="mast_chain")
    return scene


def _forks_and_load(b: newton.ModelBuilder, body: int, load: Load | None) -> None:
    for k in range(2):
        _box(b, body, _fork_local(k), (0.5 * FORK_LENGTH, 0.5 * FORK_WIDE, 0.5 * FORK_THICK), FORK_MASS)
    if load is not None:
        _box(b, body, load_local(load), tuple(0.5 * v for v in load.size), load.mass)


def vehicle(**over: Any) -> Vehicle:
    """GA's wheels: front drive and parking brakes (3,000 N·m each, F/params.py:374), rear on the steer axle."""
    return Vehicle(**{**lg.TRUCK, **over})


NODES = ("lift", "tilt_a", "tilt_b", "shift_a", "shift_b")                 # read() order
BULK = 0.8e9                                           # Pa, mineral oil with some entrained air


def hydraulics(line: float = 1.0e-3, bulk: float = BULK) -> FluidNetwork:
    """The lift, tilt and side-shift circuits (all valves closed, relief set). The lift line is one node shared by the
    free-lift and main cylinders (a tee): the one needing less pressure moves first. Open ``lift_up`` / ``lift_down``,
    ``tilt_fwd`` + ``tilt_b_drain`` (forward) or ``tilt_back`` + ``tilt_a_drain``, ``shift_left`` + ``shift_b_drain``
    or ``shift_right`` + ``shift_a_drain`` with ``set_opening``. ``line``: the lift line's oil (hoses, valve block)
    [m³], half in each lift cylinder's dead volume (every node's volume is its chambers')."""
    nodes = tuple(Node(n, medium="liquid", bulk_modulus=bulk, shared=n == "lift") for n in NODES)
    drain = {"reservoir": P_ATM, "cd_area": 2.0e-5, "density": OIL}
    edges = (Edge("lift_up", "source", "lift", flow=PUMP_FLOW),
             Edge("lift_down", "orifice", "lift", **drain),
             Edge("relief", "relief", "lift", conductance=1.0e-9, p_set=RELIEF),
             Edge("tilt_fwd", "source", "tilt_a", flow=PUMP_FLOW),
             Edge("tilt_back", "source", "tilt_b", flow=PUMP_FLOW),
             Edge("tilt_a_drain", "orifice", "tilt_a", **drain), Edge("tilt_b_drain", "orifice", "tilt_b", **drain),
             Edge("shift_left", "source", "shift_a", flow=PUMP_FLOW),
             Edge("shift_right", "source", "shift_b", flow=PUMP_FLOW),
             Edge("shift_a_drain", "orifice", "shift_a", **drain), Edge("shift_b_drain", "orifice", "shift_b", **drain),
             *(Edge(f"relief_{n}", "relief", n, conductance=1.0e-9, p_set=RELIEF)
               for n in ("tilt_a", "tilt_b", "shift_a", "shift_b")))
    acts = (Cylinder("free", joint="mast_free", a="lift", bore=math.sqrt(4.0 * AREA_FREE / math.pi), stroke=FREE,
                     dead_a=1.0e-4 + 0.5 * line),
            Cylinder("main", joint="mast_stage", a="lift", bore=math.sqrt(4.0 * AREA_MAIN / math.pi), stroke=STAGE,
                     dead_a=1.0e-4 + 0.5 * line),
            Vane("tilt", joint="mast_tilt", a="tilt_a", b="tilt_b", displacement=TILT_D, travel=2.0 * TILT_MAX,
                 x0=TILT_MAX, dead_a=1.0e-4, dead_b=1.0e-4),
            Cylinder("shift", joint="sideshift_slide", a="shift_a", b="shift_b", bore=BORE_SHIFT, stroke=2.0 * SHIFT,
                     x0=SHIFT, dead_a=1.0e-5, dead_b=1.0e-5))
    openings = tuple((e.name, 1.0 if e.name.startswith("relief") else 0.0) for e in edges)
    return FluidNetwork(nodes=nodes, edges=edges, actuators=acts, openings=openings)


__all__ = ["AREA_FREE", "AREA_MAIN", "BULK", "CG", "FREE", "Load", "MASS", "NODES", "PARTS", "PIVOT", "PUMP_FLOW",
           "Pose", "STAGE", "TILT_D", "TILT_MAX", "build", "chassis_centre", "chassis_mass", "hydraulics",
           "mast_masses", "to_truck", "vehicle"]
