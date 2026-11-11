# SPDX-License-Identifier: Apache-2.0
"""Mass properties of latch parts and the CON-45 regularisation of sprung members (pure numpy, float64).

A member's physical mass is spread over its boxes by volume (or placed at ``com``). Its spring mode must satisfy
``omega·h <= OMEGA_H_MAX`` at the spec's ``design_dt`` ([internal reference]: regularise with
body inertia or mass, never armature, because MuJoCo's ``body_invweight0`` ignores armature and the contact
impedance would then see a far lighter part):

- hinge: the moment of inertia about the hinge axis is raised to ``I_min = k·(h/omega_h)²`` by adding an isotropic
  term at the centre of mass (no added mass, so no added gravity torque about the hinge);
- slide: the mass is raised to ``m_min = k·(h/omega_h)²`` (the added weight is reported; ``design.check`` flags it
  when it exceeds 10 % of the preload).

``Member.regularise = "armature"`` puts the same floor in the joint armature instead (DC3: four 2 g C13 pads at
6.84 N preload would add 51 g to a 22 g plug). That is safe for the stop and the spring mode. Contacts on the member
still see its light body where MuJoCo's metric ignores armature: measured on MuJoCo 3.16, a slide pad under a welded
root keeps dof/body invweight0 1/m with any armature; under a free root the slide's dof_invweight0 counts it.
Give the touched faces precision pairs (card ``precision_features``; ``alloy.testing.latch.touched_cfg``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.latch.spec import OMEGA_H_MAX, Box, LatchSpec, Member


def quat_matrix(q: tuple[float, float, float, float]) -> np.ndarray:
    """Rotation matrix of a unit quaternion (x, y, z, w)."""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@dataclass(frozen=True)
class MassProps:
    mass: float
    com: np.ndarray            # [3], part frame
    inertia: np.ndarray        # [3, 3] about the com, part frame
    added_inertia: float = 0.0  # isotropic term added by the regularisation (kg·m²)
    added_mass: float = 0.0     # kg added by the regularisation (slides)

    def principal_ok(self) -> bool:
        """Positive definite and the triangle inequality holds (MuJoCo's inertia check)."""
        ev = np.linalg.eigvalsh(0.5 * (self.inertia + self.inertia.T))
        return bool(ev.min() > 0.0 and ev[0] + ev[1] >= ev[2] * (1.0 - 1e-9))


def box_props(boxes: tuple[Box, ...], mass: float, com: tuple[float, float, float] | None = None) -> MassProps:
    """Mass properties of ``mass`` spread over the boxes by volume (``visual`` boxes carry none)."""
    solid = [b for b in boxes if b.role != "visual"] or list(boxes)
    vol = sum(b.volume for b in solid)
    c = np.zeros(3)
    for b in solid:
        c += (b.volume / vol) * np.asarray(b.pos, np.float64)
    inertia = np.zeros((3, 3))
    for b in solid:
        m = mass * b.volume / vol
        hx, hy, hz = (2.0 * h for h in b.half)
        local = np.diag([m * (hy * hy + hz * hz), m * (hx * hx + hz * hz), m * (hx * hx + hy * hy)]) / 12.0
        R = quat_matrix(b.quat)
        d = np.asarray(b.pos, np.float64) - c
        inertia += R @ local @ R.T + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    if com is not None:                      # explicit com (measured): keep the centroidal shape inertia there
        c = np.asarray(com, np.float64)
    floor = 1e-12 * max(mass, 1e-9)
    inertia = 0.5 * (inertia + inertia.T) + floor * np.eye(3)
    return MassProps(mass, c, inertia)


def mode_floor(rate: float, dt: float, omega_h: float = OMEGA_H_MAX) -> float:
    """Smallest inertia (hinge, kg·m²) or mass (slide, kg) with sqrt(rate/x)·dt <= omega_h."""
    return rate * (dt / omega_h) ** 2


def member_floor(m: Member, dt: float) -> float:
    """Regularisation floor of a member: the omega·h bound and, with ``snap_step``, preload·dt²/snap_step."""
    floor = mode_floor(m.spring.rate, dt)
    if m.snap_step is not None:
        floor = max(floor, m.spring.preload * dt * dt / m.snap_step)
    return floor


def snap_per_step(m: Member, dt: float, inertia: float) -> float:
    """Travel in one step from rest under the preload alone (rad or m): preload·dt²/I."""
    return m.spring.preload * dt * dt / inertia


def axis_inertia(m: Member, props: MassProps) -> float:
    """Generalised inertia of the member's own body along its joint: I about the hinge axis (through the member
    origin) or the mass (slide)."""
    if m.kind == "slide":
        return float(props.mass)
    a = np.asarray(m.axis, np.float64)
    r = props.com - np.dot(props.com, a) * a                  # com offset from the axis (axis through the origin)
    return float(a @ props.inertia @ a) + props.mass * float(np.dot(r, r))


def member_props(m: Member, dt: float) -> MassProps:
    """Regularised mass properties of a member (member frame: origin on the joint axis); physical when the member
    is regularised by armature (``member_armature``)."""
    p = box_props(m.shapes, m.mass, m.com)
    if m.regularise == "armature":
        return p
    floor = member_floor(m, dt)
    q = axis_inertia(m, p)
    if m.kind == "hinge":
        add = max(0.0, floor - q)
        return MassProps(p.mass, p.com, p.inertia + add * np.eye(3), added_inertia=add)
    add_m = max(0.0, floor - q)
    if add_m == 0.0:
        return p
    scale = (p.mass + add_m) / p.mass
    return MassProps(p.mass + add_m, p.com, p.inertia * scale, added_mass=add_m)


def member_armature(m: Member, dt: float) -> float:
    """Joint armature (kg·m² or kg) lifting the member's mode to omega·h = OMEGA_H_MAX; 0 unless
    ``regularise == "armature"``."""
    if m.regularise != "armature":
        return 0.0
    return max(0.0, member_floor(m, dt) - axis_inertia(m, box_props(m.shapes, m.mass, m.com)))


def mode_omega(m: Member, props: MassProps, armature: float = 0.0) -> float:
    """omega = sqrt(rate / (I_axis or mass + armature)) of the regularised member (rad/s)."""
    return math.sqrt(m.spring.rate / (axis_inertia(m, props) + armature))


D_MIN, D_MAX = 1.0e-4, 0.9999           # MuJoCo solimp d in (0, 1)


def stop_solimp(m: Member, k_acc: float, inv_weight: float) -> tuple[float, ...] | None:
    """Flat ``solimplimit`` for ``m.stop_stiffness``. With a direct solref (stiffness k) and flat d, MuJoCo's static
    limit stiffness is ``k / ((1 - d)·A)``, A = diagApprox = the DOF's compiled ``dof_invweight0``; so
    ``d = 1 - k / (K·A)`` (CON-47). ``inv_weight`` should bound the compiled A from above (``mjcf.stop_inv_weights``:
    no armature, root free), so the stop is at least ``K``, never softer."""
    if m.stop_stiffness is None:
        return None
    d = min(D_MAX, max(D_MIN, 1.0 - abs(k_acc) / (m.stop_stiffness * inv_weight)))
    return (d, d, 0.001, 0.5, 2.0)


def root_props(spec: LatchSpec) -> MassProps:
    return box_props(spec.root_shapes, spec.root_mass, spec.root_com)


def stop_solref(spec: LatchSpec, zeta: float = 1.0) -> tuple[float, float]:
    """Direct (negative) MuJoCo solreflimit for every stop: stiffness (w/dt)², damping 2·zeta·w/dt with
    w = ``stop_omega_h`` at ``design_dt`` (CON-47 fix: a stiff one-dof limit has no normal to flip)."""
    w = spec.stop_omega_h / spec.design_dt
    return (-(w * w), -2.0 * zeta * w)


def damping_for(m: Member, dt: float, zeta: float = 0.5) -> float:
    """Joint damping giving damping ratio ``zeta`` on the regularised spring mode (N·m·s/rad or N·s/m); a light
    regularised member otherwise rings for many periods after a press or a click."""
    p = member_props(m, dt)
    return 2.0 * zeta * m.spring.rate / mode_omega(m, p, member_armature(m, dt))
