# SPDX-License-Identifier: Apache-2.0
"""Fluid actuators on the lumped-fluid core: cylinders on prismatic joints, rotary vanes on revolute joints (engine
item 10 part 3).

An actuator owns the volumes of its chambers: every exchange it reads its joint's coordinate from the body poses (any
solver), writes the chamber volumes into the network and applies the pressure force along the joint axis, on the child
at the joint anchor and back on the parent (explicit, from the pressures at the step start). A node declared
``shared`` (a common line, e.g. one pump teed to a mast's free-lift and main cylinders) is the chamber of several
actuator sides: its volume is the sum of theirs, and each force reads the one pressure. Joining two chambers by a wide
edge instead does not work: each force reads its own node's response, which a wide edge holds near the step-start
pressure, so the oil column's stiffness lags a step and its ringing grows.

    Cylinder (prismatic joint, axis from the A end toward the rod):
        x   = x0 + q                       piston travel from the A end, q the joint coordinate
        V_A = dead_a + A_A·x               V_B = dead_b + A_B·(stroke - x)        (A_B = A_A - π·rod²/4)
        F   = (P_A - P_atm)·A_A - (P_B - P_atm)·A_B - coulomb·tanh(ẋ/v_stick) - viscous·ẋ
            = P_A·A_A - P_B·A_B - P_atm·(A_A - A_B) - friction  (the rod side's annulus outside sees the atmosphere)
        single-acting (b = None): the rod side vents to the atmosphere, F = (P_A - P_atm)·A_A - friction

    Vane (revolute joint, displacement D [m³/rad], angle θ about the joint axis from the A end):
        V_A = dead_a + D·θ                 V_B = dead_b + D·(travel - θ)
        τ   = (P_A - P_B)·D - friction     about the axis, on the child and back on the parent

Friction (coulomb·tanh(ẋ/v_stick) + viscous·ẋ) is explicit, so it is capped at m·|ẋ|/dt (m: the child's and parent's
reduced mass, a lower bound on the joint's inertia): it can stop the slide within a step, never reverse it. Below the
breakaway force the piston creeps at about F_net·max(v_stick/coulomb, dt/m).

Stiffness: the chamber is a spring of κ = A²·dP/dV (gas n·P/V, liquid K/V) on the moving mass m, so the explicit force
needs dt < 2/sqrt(κ/m) (``Cylinder.check_dt``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import warp as wp

from alloy.plugins.fluid.kernels import after_volume_change, node_kappa
from alloy.plugins.fluid.network import P_ATM, Network
from alloy.plugins.funcs import add_wrench, com_world, point_velocity, torque_wrench, wrench_at

CYLINDER = wp.constant(0)
VANE = wp.constant(1)


@wp.struct
class ActuatorSpec:
    kind: wp.int32
    na: wp.int32                 # node of the A (cap) side
    nb: wp.int32                 # node of the B (rod) side, -1: atmosphere
    area_a: wp.float32           # m²
    area_b: wp.float32           # m²
    stroke: wp.float32           # m
    x0: wp.float32               # m, piston travel from the A end at joint q = 0
    dead_a: wp.float32           # m³
    dead_b: wp.float32           # m³
    coulomb: wp.float32          # N
    viscous: wp.float32          # N·s/m
    v_stick: wp.float32          # m/s, tanh width of the Coulomb term


@dataclass(frozen=True)
class Cylinder:
    """A pneumatic or hydraulic cylinder on prismatic joint ``joint`` (its axis points from the A end to the rod).

    Example (double acting, 20 mm bore, 8 mm rod, 100 mm stroke, starting retracted)::

        Cylinder("lift", joint="arm_slide", a="lift_a", b="lift_b", bore=0.02, rod=0.008, stroke=0.1)
    """

    name: str
    joint: str                   # prismatic joint label
    a: str                       # node of the cap side (pressure there extends the rod)
    b: str | None = None         # node of the rod side; None: single-acting, the rod side vents to the atmosphere
    bore: float = 0.02           # m
    rod: float = 0.0             # m, rod diameter (A_B = A_A - π·rod²/4)
    stroke: float = 0.1          # m
    x0: float = 0.0              # m, piston travel from the A end at joint q = 0
    dead_a: float = 1.0e-6       # m³ at x = 0
    dead_b: float = 1.0e-6       # m³ at x = stroke
    coulomb: float = 0.0         # N, seal friction
    viscous: float = 0.0         # N·s/m
    v_stick: float = 1.0e-3      # m/s

    def __post_init__(self) -> None:
        if min(self.bore, self.stroke, self.dead_a, self.dead_b, self.v_stick) <= 0:
            raise ValueError(f"cylinder {self.name!r}: bore, stroke, dead volumes and v_stick must be > 0")
        if not 0.0 <= self.rod < self.bore or min(self.coulomb, self.viscous) < 0:
            raise ValueError(f"cylinder {self.name!r}: 0 <= rod < bore and friction >= 0 are required")

    @property
    def area_a(self) -> float:
        return math.pi * self.bore ** 2 / 4.0

    @property
    def area_b(self) -> float:
        return self.area_a - math.pi * self.rod ** 2 / 4.0

    def volumes(self, x: float) -> tuple[float, float]:
        """(V_A, V_B) at piston travel x [m³]."""
        return self.dead_a + self.area_a * x, self.dead_b + self.area_b * (self.stroke - x)

    def force(self, p_a: float, p_b: float = P_ATM, atmosphere: float = P_ATM) -> float:
        """The static rod force at chamber pressures (absolute) [N], without friction."""
        return (p_a - atmosphere) * self.area_a - (p_b - atmosphere) * self.area_b

    def check_dt(self, net: Network, x: float, mass: float, pressures: dict[str, float] | None = None) -> float:
        """The largest stable exchange step [s] for the moving ``mass`` at travel x: 2/sqrt(κ/m), κ = Σ A²·dP/dV."""
        k = 0.0
        for node, area, vol in ((self.a, self.area_a, self.volumes(x)[0]), (self.b, self.area_b, self.volumes(x)[1])):
            if node is None:
                continue
            nd = net.nodes[net.node_index(node)]
            p = (pressures or {}).get(node, nd.pressure)
            k += area ** 2 * (nd.bulk_modulus if nd.medium == "liquid" else nd.polytropic * p) / vol
        return 2.0 / math.sqrt(k / mass) if k > 0 else math.inf

    def spec(self, net: Network) -> ActuatorSpec:
        s = ActuatorSpec()
        s.kind, s.na = CYLINDER, net.node_index(self.a)
        s.nb = -1 if self.b is None else net.node_index(self.b)
        for name in ("area_a", "area_b", "stroke", "x0", "dead_a", "dead_b", "coulomb", "viscous", "v_stick"):
            setattr(s, name, float(getattr(self, name)))
        return s


@dataclass(frozen=True)
class Vane:
    """A rotary vane actuator on revolute joint ``joint``: pressure in A turns the child about the joint axis.

    Example (double vane, 2 cm³ per radian, 270° of travel, starting at the A end)::

        Vane("wrist", joint="wrist_hinge", a="wrist_a", b="wrist_b", displacement=2e-6, travel=math.radians(270))
    """

    name: str
    joint: str                   # revolute joint label
    a: str                       # node of the A side (pressure there turns the child +about the axis)
    b: str | None = None         # node of the B side; None: B vents to the atmosphere
    displacement: float = 1.0e-6  # m³/rad, D
    travel: float = math.pi      # rad
    x0: float = 0.0              # rad, angle from the A end at joint q = 0
    dead_a: float = 1.0e-7       # m³
    dead_b: float = 1.0e-7       # m³
    coulomb: float = 0.0         # N·m
    viscous: float = 0.0         # N·m·s/rad
    v_stick: float = 1.0e-2      # rad/s

    def __post_init__(self) -> None:
        if min(self.displacement, self.travel, self.dead_a, self.dead_b, self.v_stick) <= 0:
            raise ValueError(f"vane {self.name!r}: displacement, travel, dead volumes and v_stick must be > 0")
        if min(self.coulomb, self.viscous) < 0:
            raise ValueError(f"vane {self.name!r}: friction must be >= 0")

    @property
    def area_a(self) -> float:                            # m³/rad: the cylinder formulas with x an angle
        return self.displacement

    @property
    def area_b(self) -> float:
        return self.displacement

    @property
    def stroke(self) -> float:
        return self.travel

    def volumes(self, x: float) -> tuple[float, float]:
        return self.dead_a + self.displacement * x, self.dead_b + self.displacement * (self.travel - x)

    def force(self, p_a: float, p_b: float = P_ATM, atmosphere: float = P_ATM) -> float:
        """The static torque [N·m], without friction."""
        return (p_a - p_b) * self.displacement

    check_dt = Cylinder.check_dt                          # mass -> moment of inertia about the axis

    def spec(self, net: Network) -> ActuatorSpec:
        s = Cylinder.spec(self, net)
        s.kind = VANE
        return s


Actuator = Cylinder | Vane


def validate(actuators: tuple[Actuator, ...], net: Network) -> None:
    """Every actuator chamber is a node of ``net`` owned by one actuator side only, unless the node is ``shared``."""
    if net.mixed and actuators:                           # their force pressure would be the pocket's alone
        raise ValueError("cylinders and vanes drive gas or liquid chambers; a mixed (liquid + gas pocket) chamber "
                         "is a syringe: use the syringe plug-in (alloy.plugins.library.syringe)")
    owned: dict[str, str] = {}
    names = [a.name for a in actuators]
    nodes = [n.name for n in net.nodes]
    if len(set(names)) != len(names):
        raise ValueError(f"actuator names must be unique, got {names}")
    for act in actuators:
        for node in (act.a, act.b):
            if node is None:
                continue
            if node not in nodes:
                raise ValueError(f"actuator {act.name!r}: unknown node {node!r} (nodes: {nodes})")
            if net.nodes[nodes.index(node)].medium == "tank":
                raise ValueError(f"actuator {act.name!r}: {node!r} is an open tank, not a chamber")
            if node in owned and not net.nodes[nodes.index(node)].shared:
                raise ValueError(f"node {node!r} is a chamber of both {owned[node]!r} and {act.name!r} (declare it "
                                 "Node(shared=True) for a common line)")
            if act.a == act.b:
                raise ValueError(f"actuator {act.name!r}: its two chambers are the same node {node!r}")
            owned[node] = act.name


def spec_array(actuators: tuple[Actuator, ...], net: Network, device: object) -> wp.array:
    specs = [a.spec(net) for a in actuators] or [Cylinder("_none", "_", net.nodes[0].name).spec(net)]
    return wp.array(specs, dtype=ActuatorSpec, device=device)


@wp.func
def _inv_inertia_about(body_q: wp.array[wp.transform], body_inv_inertia: wp.array[wp.mat33], b: wp.int32,
                       ax: wp.vec3) -> float:
    a = wp.quat_rotate_inv(wp.transform_get_rotation(body_q[b]), ax)
    return wp.dot(a, body_inv_inertia[b] * a)


@wp.func
def _relative_angle(xp: wp.transform, xc: wp.transform, axis_local: wp.vec3) -> float:
    """Rotation of the child joint frame about the axis (parent joint frame), in (-π, π]."""
    qr = wp.quat_inverse(wp.transform_get_rotation(xp)) * wp.transform_get_rotation(xc)
    sgn = 1.0
    if qr[3] < 0.0:
        sgn = -1.0
    return 2.0 * wp.atan2(sgn * wp.dot(wp.vec3(qr[0], qr[1], qr[2]), wp.normalize(axis_local)), sgn * qr[3])


@wp.func
def _chamber_travel(body_q: wp.array[wp.transform], joint_parent: wp.array[wp.int32], joint_child: wp.array[wp.int32],
                    joint_X_p: wp.array[wp.transform], joint_X_c: wp.array[wp.transform], joint_axis: wp.array[wp.vec3],
                    joint_qd_start: wp.array[wp.int32], j: wp.int32, s: ActuatorSpec) -> float:
    """Piston travel (cylinder) or vane angle from the A end at the joint's current coordinate."""
    par = joint_parent[j]
    xp = joint_X_p[j]
    if par >= 0:
        xp = wp.transform_multiply(body_q[par], xp)
    xc = wp.transform_multiply(body_q[joint_child[j]], joint_X_c[j])
    axis = joint_axis[joint_qd_start[j]]
    if s.kind == CYLINDER:
        ax = wp.normalize(wp.transform_vector(xp, axis))
        return s.x0 + wp.dot(wp.transform_get_translation(xc) - wp.transform_get_translation(xp), ax)
    return s.x0 + _relative_angle(xp, xc, axis)


@wp.func
def _chamber_a(s: ActuatorSpec, x: float) -> wp.float64:
    return wp.max(wp.float64(s.dead_a) + wp.float64(s.area_a) * wp.float64(x), wp.float64(0.05 * s.dead_a))


@wp.func
def _chamber_b(s: ActuatorSpec, x: float) -> wp.float64:
    return wp.max(wp.float64(s.dead_b) + wp.float64(s.area_b) * wp.float64(s.stroke - x),
                  wp.float64(0.05 * s.dead_b))


@wp.func
def _write_volumes(w: wp.int32, n_act: wp.int32, spec: wp.array[ActuatorSpec], joint: wp.array2d[wp.int32],
                   body_q: wp.array[wp.transform], joint_parent: wp.array[wp.int32],
                   joint_child: wp.array[wp.int32], joint_X_p: wp.array[wp.transform],
                   joint_X_c: wp.array[wp.transform], joint_axis: wp.array[wp.vec3],
                   joint_qd_start: wp.array[wp.int32], volume: wp.array2d[wp.float64]):
    """Every chamber node's new volume: the sum of the chambers on it (one, unless the node is shared)."""
    for k in range(n_act):
        s = spec[k]
        volume[w, s.na] = wp.float64(0.0)
        if s.nb >= 0:
            volume[w, s.nb] = wp.float64(0.0)
    for k in range(n_act):
        s = spec[k]
        x = _chamber_travel(body_q, joint_parent, joint_child, joint_X_p, joint_X_c, joint_axis, joint_qd_start,
                            joint[w, k], s)
        volume[w, s.na] = volume[w, s.na] + _chamber_a(s, x)
        if s.nb >= 0:
            volume[w, s.nb] = volume[w, s.nb] + _chamber_b(s, x)


@wp.func
def _force_pressure(w: wp.int32, i: wp.int32, dt: wp.float64, medium: wp.array[wp.int32],
                    bulk: wp.array[wp.float64], p_ref: wp.array[wp.float64], poly: wp.array[wp.float64],
                    ea: wp.array[wp.int32], eb: wp.array[wp.int32], g: wp.array2d[wp.float64],
                    pressure: wp.array2d[wp.float64], volume_old: wp.array2d[wp.float64],
                    volume: wp.array2d[wp.float64]) -> wp.float64:
    """The pressure the force uses at node i's new volume: the node's implicit (backward Euler) response to its own
    volume change, P + (P' - P)·κ/(κ + dt·Σg), P' after the exact change, Σg its edges' conductances of the last step.
    A closed chamber gets P' (no lag: a liquid chamber is a stiff spring); one held by a wide valve stays near P (the
    flow refills it within the step)."""
    v_new = volume[w, i]
    p = pressure[w, i]
    p1 = after_volume_change(medium[i], p, volume_old[w, i], v_new, bulk[i], p_ref[i], poly[i])
    gsum = wp.float64(0.0)
    for e in range(ea.shape[0]):
        if ea[e] == i or eb[e] == i:
            gsum = gsum + g[w, e]
    k = node_kappa(medium[i], v_new, bulk[i], poly[i])
    return p + (p1 - p) * k / (k + dt * gsum)


@wp.kernel
def actuate(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
            body_inv_mass: wp.array[wp.float32], body_inv_inertia: wp.array[wp.mat33], dt: wp.float32,
            joint_parent: wp.array[wp.int32], joint_child: wp.array[wp.int32], joint_X_p: wp.array[wp.transform],
            joint_X_c: wp.array[wp.transform], joint_axis: wp.array[wp.vec3], joint_qd_start: wp.array[wp.int32],
            n_act: wp.int32, joint: wp.array2d[wp.int32], spec: wp.array[ActuatorSpec], atm: wp.float64,
            medium: wp.array[wp.int32], bulk: wp.array[wp.float64], p_ref: wp.array[wp.float64],
            poly: wp.array[wp.float64], ea: wp.array[wp.int32], eb: wp.array[wp.int32], g: wp.array2d[wp.float64],
            pressure: wp.array2d[wp.float64], volume_old: wp.array2d[wp.float64],
            volume: wp.array2d[wp.float64], travel: wp.array2d[wp.float32],
            speed: wp.array2d[wp.float32], force: wp.array2d[wp.float32], piston: wp.array3d[wp.float32],
            body_f: wp.array[wp.spatial_vector]):
    """One thread per world: every actuator's travel -> chamber volumes, pressures -> force (cylinder) or torque
    (vane) on the joint's bodies."""
    w = wp.tid()
    _write_volumes(w, n_act, spec, joint, body_q, joint_parent, joint_child, joint_X_p, joint_X_c, joint_axis,
                   joint_qd_start, volume)
    for k in range(n_act):
        s = spec[k]
        j = joint[w, k]
        par, ch = joint_parent[j], joint_child[j]
        xp = joint_X_p[j]
        if par >= 0:
            xp = wp.transform_multiply(body_q[par], xp)
        xc = wp.transform_multiply(body_q[ch], joint_X_c[j])
        ax = wp.normalize(wp.transform_vector(xp, joint_axis[joint_qd_start[j]]))
        anchor = wp.transform_get_translation(xc)
        com_c = com_world(body_q, body_com, ch)
        com_p = wp.vec3(0.0)
        if par >= 0:
            com_p = com_world(body_q, body_com, par)
        x, xd, inv = float(0.0), float(0.0), float(0.0)
        if s.kind == CYLINDER:
            x = s.x0 + wp.dot(anchor - wp.transform_get_translation(xp), ax)
            v = point_velocity(body_qd, com_c, ch, anchor)
            inv = body_inv_mass[ch]
            if par >= 0:
                v = v - point_velocity(body_qd, com_p, par, anchor)
                inv = inv + body_inv_mass[par]
            xd = wp.dot(v, ax)
        else:
            x = s.x0 + _relative_angle(xp, xc, joint_axis[joint_qd_start[j]])
            qc = body_qd[ch]
            om = wp.vec3(qc[3], qc[4], qc[5])
            inv = _inv_inertia_about(body_q, body_inv_inertia, ch, ax)
            if par >= 0:
                qp = body_qd[par]
                om = om - wp.vec3(qp[3], qp[4], qp[5])
                inv = inv + _inv_inertia_about(body_q, body_inv_inertia, par, ax)
            xd = wp.dot(om, ax)
        dt64 = wp.float64(dt)
        dpa = _force_pressure(w, s.na, dt64, medium, bulk, p_ref, poly, ea, eb, g, pressure, volume_old, volume) - atm
        dpb = wp.float64(0.0)
        if s.nb >= 0:
            dpb = _force_pressure(w, s.nb, dt64, medium, bulk, p_ref, poly, ea, eb, g, pressure, volume_old,
                                  volume) - atm
        f = float(dpa * wp.float64(s.area_a) - dpb * wp.float64(s.area_b))
        fr = s.coulomb * wp.tanh(xd / s.v_stick) + s.viscous * xd
        if inv > 0.0:                                    # explicit friction: stop the slide at most, never reverse it
            cap = wp.abs(xd) / (inv * dt)
            fr = wp.clamp(fr, -cap, cap)
        f = f - fr
        wc = torque_wrench(f * ax)                       # vane: a pure torque about the axis
        if s.kind == CYLINDER:
            wc = wrench_at(f * ax, anchor, com_c)
        add_wrench(body_f, ch, wc)
        if par >= 0:
            wpar = torque_wrench(-f * ax)
            if s.kind == CYLINDER:
                wpar = wrench_at(-f * ax, anchor, com_p)
            add_wrench(body_f, par, wpar)
        travel[w, k] = x
        speed[w, k] = xd
        force[w, k] = f
        piston[w, k, 0] = float(dpa)                       # the gauge pressures the force acted with (A, B)
        piston[w, k, 1] = float(dpb)


__all__ = ["Actuator", "ActuatorSpec", "Cylinder", "Vane", "actuate", "spec_array", "validate"]
