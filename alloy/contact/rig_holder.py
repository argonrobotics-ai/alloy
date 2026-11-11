# SPDX-License-Identifier: Apache-2.0
"""R1 compliant holder on device: a spring-damper between the stage's kinematic target and the peg (body forces).

Per world, the compliance centre (on the peg axis, ``l_g`` above the tip) is pulled toward the stage target by
``k_x`` (x, y) and ``k_z`` (z), and the peg's orientation toward its initial (stage) orientation by ``k_theta``; the
dampers act on the velocity relative to the target, which moves down at the stage speed. The wrench goes in at the
COM (force; torque plus r × F). The springs are stage-frame (small-angle holder, as the card's k_x / k_theta are
measured). The springs are preloaded so that the built pose is the hanging equilibrium (the protocol's tilt and
offset are the hanging peg's): the axial spring carries the weight and the reference orientation is turned by the
weight's moment about the compliance centre / k_theta (0.014° at 2°). The insertion force is the tared holder force
``m g - F_z`` (compression positive). The stage stops for good (per world, latched on device, no atomics) when the
insertion force reaches the protocol's ``f_stop``.

Damping: zeta 0.3 per mode, scaled down where the explicit (body-force) damping would exceed ``c·dt/I = 1`` for the
light peg's COM rotation and its spin about the axis (semi-implicit Euler stability; an isotropic rotational damper
blows up the spin mode); the stiffness terms must keep ``k·dt²/I < 1``.
Recording box-averages the insertion force over each recorded column (decimation, like the rig's 1 kHz cell).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.contact.rig_r1 import R1Holder, R1Protocol
from alloy.core.errors import LawError
from alloy.core.log import get_logger

log = get_logger("contact.rig")

ZETA = 0.3
B_MAX = 1.0          # explicit damping c·dt/I (or m) per mode
A_MAX = 1.0          # explicit stiffness k·dt²/I (or m) per mode
T_NEVER = 1.0e30


@wp.struct
class HolderParams:
    k_lat: float
    k_ax: float
    k_rot: float
    c_lat: float
    c_ax: float
    c_rot: float
    c_spin: float        # damping about the peg axis (its inertia is ~12x smaller than the transverse one)
    speed: float
    t_settle: float
    stroke: float
    f_stop: float
    weight: float
    top_z: float
    centre: wp.vec3      # compliance centre, peg frame
    tip: wp.vec3         # tip centre, peg frame


@wp.struct
class HolderWorlds:
    bodies: wp.array(dtype=wp.int32)
    p_ref: wp.array(dtype=wp.vec3)       # compliance-centre target at stage travel 0 (weight preload included)
    q_ref: wp.array(dtype=wp.quat)
    t_stop: wp.array(dtype=wp.float32)   # stage stop time (latched), T_NEVER while running
    acc_f: wp.array(dtype=wp.float64)    # insertion-force sum since the last recorded column
    acc_n: wp.array(dtype=wp.int32)


@wp.func
def _travel(p: HolderParams, t: float, t_stop: float) -> float:
    return wp.clamp(p.speed * (wp.min(t, t_stop) - p.t_settle), 0.0, p.stroke)


@wp.func
def _moving(p: HolderParams, t: float, t_stop: float) -> float:
    s = p.speed * (t - p.t_settle)
    if t < t_stop and s > 0.0 and s < p.stroke:
        return 1.0
    return 0.0


@wp.func
def _rotvec(q: wp.quat) -> wp.vec3:
    """Rotation vector (axis × angle) of ``q``, shortest way round."""
    v = wp.vec3(q[0], q[1], q[2])
    s = wp.length(v)
    sign = 1.0
    if q[3] < 0.0:
        sign = -1.0
    if s < 1.0e-9:
        return 2.0 * sign * v
    return v * (sign * 2.0 * wp.atan2(s, wp.abs(q[3])) / s)


@wp.func
def _holder(p: HolderParams, X: wp.transform, qd: wp.spatial_vector, com: wp.vec3, p_ref: wp.vec3,
            q_ref: wp.quat, t: float, t_stop: float) -> wp.spatial_vector:
    """(force at the compliance centre, holder moment about it), world frame."""
    pos_c = wp.transform_point(X, p.centre)
    r = pos_c - wp.transform_point(X, com)
    w = wp.spatial_bottom(qd)
    v_c = wp.spatial_top(qd) + wp.cross(w, r)          # Newton: (linear COM, angular), world frame
    e = p_ref - wp.vec3(0.0, 0.0, _travel(p, t, t_stop)) - pos_c
    de = wp.vec3(0.0, 0.0, -p.speed * _moving(p, t, t_stop)) - v_c
    f = wp.vec3(p.k_lat * e[0] + p.c_lat * de[0], p.k_lat * e[1] + p.c_lat * de[1], p.k_ax * e[2] + p.c_ax * de[2])
    q = wp.transform_get_rotation(X)
    axis = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))
    w_spin = wp.dot(w, axis) * axis
    damp = p.c_rot * (w - w_spin) + p.c_spin * w_spin
    return wp.spatial_vector(f, p.k_rot * _rotvec(q_ref * wp.quat_inverse(q)) - damp)


@wp.func
def _at_com(p: HolderParams, X: wp.transform, com: wp.vec3, h: wp.spatial_vector) -> wp.spatial_vector:
    """The holder wrench moved to the COM: torque = moment + r × force."""
    f = wp.spatial_top(h)
    r = wp.transform_point(X, p.centre) - wp.transform_point(X, com)
    return wp.spatial_vector(f, wp.spatial_bottom(h) + wp.cross(r, f))


@wp.kernel
def _drive(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
           body_com: wp.array(dtype=wp.vec3), p: HolderParams, hw: HolderWorlds, t: float,
           body_f: wp.array(dtype=wp.spatial_vector)):
    w = wp.tid()
    b = hw.bodies[w]
    h = _holder(p, body_q[b], body_qd[b], body_com[b], hw.p_ref[w], hw.q_ref[w], t, hw.t_stop[w])
    body_f[b] = _at_com(p, body_q[b], body_com[b], h)
    if p.weight - wp.spatial_top(h)[2] >= p.f_stop and t < hw.t_stop[w]:
        hw.t_stop[w] = t


@wp.kernel
def _record(body_q: wp.array(dtype=wp.transform), body_qd: wp.array(dtype=wp.spatial_vector),
            body_com: wp.array(dtype=wp.vec3), p: HolderParams, hw: HolderWorlds, t: float, col: int,
            depth: wp.array2d(dtype=wp.float32), force: wp.array2d(dtype=wp.float32),
            tilt: wp.array2d(dtype=wp.float32), fx: wp.array2d(dtype=wp.float32),
            my: wp.array2d(dtype=wp.float32)):
    w = wp.tid()
    b = hw.bodies[w]
    X = body_q[b]
    h = _holder(p, X, body_qd[b], body_com[b], hw.p_ref[w], hw.q_ref[w], t, hw.t_stop[w])
    hw.acc_f[w] = hw.acc_f[w] + wp.float64(p.weight - wp.spatial_top(h)[2])
    hw.acc_n[w] = hw.acc_n[w] + 1
    if col < 0:
        return
    force[w, col] = wp.float32(hw.acc_f[w] / wp.float64(hw.acc_n[w]))
    hw.acc_f[w] = wp.float64(0.0)
    hw.acc_n[w] = 0
    depth[w, col] = p.top_z - wp.transform_point(X, p.tip)[2]
    axis = wp.quat_rotate(wp.transform_get_rotation(X), wp.vec3(0.0, 0.0, 1.0))
    tilt[w, col] = wp.acos(wp.clamp(axis[2], -1.0, 1.0))
    fx[w, col] = wp.spatial_top(h)[0]           # holder lateral force and moment (lean plane), diagnostics
    my[w, col] = wp.spatial_bottom(h)[1]


@dataclass(frozen=True)
class HolderSetup:
    """What the drive needs besides the model: card holder, protocol, run speed and dt, peg length, hole top."""

    holder: R1Holder
    proto: R1Protocol
    speed: float
    dt: float
    peg_len: float
    top_z: float
    gravity: float = 9.81


@dataclass(frozen=True)
class PegInertia:
    mass: float
    i_cm: float          # transverse, about the COM
    i_axial: float       # about the peg axis
    arm: float           # compliance centre to COM, along the axis


@dataclass(frozen=True)
class Damping:
    c_lat: float
    c_ax: float
    c_rot: float
    c_spin: float
    margins: dict


def holder_damping(holder: R1Holder, peg: PegInertia, dt: float) -> Damping:
    """Dampers at zeta ``ZETA``, scaled into the explicit-integration limits; ``LawError`` if a spring is too stiff
    for ``dt`` with explicit body forces."""
    m, arm2 = peg.mass, peg.arm * peg.arm
    c_lat, c_ax = 2 * ZETA * math.sqrt(holder.k_x * m), 2 * ZETA * math.sqrt(holder.k_z * m)
    c_rot = 2 * ZETA * math.sqrt(holder.k_theta * (peg.i_cm + m * arm2))
    scale_rot = min(1.0, B_MAX / ((c_rot + c_lat * arm2) * dt / peg.i_cm))
    c_rot, c_lat = c_rot * scale_rot, c_lat * scale_rot
    c_ax *= min(1.0, B_MAX / (c_ax * dt / m))
    c_spin = min(2 * ZETA * math.sqrt(holder.k_theta * peg.i_axial), B_MAX * peg.i_axial / dt)
    margins = {"a_ax": holder.k_z * dt * dt / m, "a_rot": (holder.k_theta + holder.k_x * arm2) * dt * dt / peg.i_cm,
               "a_spin": holder.k_theta * dt * dt / peg.i_axial, "b_ax": c_ax * dt / m,
               "b_rot": (c_rot + c_lat * arm2) * dt / peg.i_cm, "b_spin": c_spin * dt / peg.i_axial}
    if max(margins["a_ax"], margins["a_rot"], margins["a_spin"]) >= A_MAX:
        raise LawError(f"R1 holder too stiff for dt={dt:g} s with explicit body forces: {margins}")
    return Damping(c_lat, c_ax, c_rot, c_spin, margins)


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` [3] (or [N, 3]) by quaternions ``q`` [N, 4] (x, y, z, w)."""
    u, w = q[:, :3], q[:, 3:4]
    t = 2.0 * np.cross(u, np.broadcast_to(v, u.shape))
    return v + w * t + np.cross(u, t)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton products of [N, 4] (x, y, z, w) quaternions."""
    va, wa, vb, wb = a[:, :3], a[:, 3:4], b[:, :3], b[:, 3:4]
    return np.concatenate([wa * vb + wb * va + np.cross(va, vb), wa * wb - np.sum(va * vb, 1, keepdims=True)], 1)


def _quat_exp(rv: np.ndarray) -> np.ndarray:
    """Quaternions of rotation vectors [N, 3]."""
    ang = np.linalg.norm(rv, axis=1, keepdims=True)
    k = np.where(ang > 0.0, np.sin(0.5 * ang) / np.where(ang > 0.0, ang, 1.0), 0.5)
    return np.concatenate([rv * k, np.cos(0.5 * ang)], 1)


def hanging_reference(q0: np.ndarray, com: np.ndarray, centre: np.ndarray, weight: float, k_theta: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Compliance-centre positions [N, 3] and reference orientations [N, 4] that make the poses ``q0`` [N, 7] the
    hanging equilibrium: the holder carries the weight at the centre and cancels its moment about the COM."""
    pos_c = q0[:, :3] + _rotate(q0[:, 3:7], centre)
    r = pos_c - (q0[:, :3] + _rotate(q0[:, 3:7], com))
    force = np.array([0.0, 0.0, weight])
    q_ref = _quat_mul(_quat_exp(-np.cross(r, force) / k_theta), q0[:, 3:7])
    return pos_c, q_ref


class HolderDrive:
    """Per-world holder state on device; ``apply`` before each solver step, ``record`` after it."""

    def __init__(self, model: Any, bodies: np.ndarray, setup: HolderSetup) -> None:
        dev = model.device
        bodies = np.asarray(bodies, np.int32)
        mass = model.body_mass.numpy()[bodies]
        if not np.allclose(mass, mass[0]):
            raise LawError("R1 holder: every world's peg must have the same mass")
        inertia = model.body_inertia.numpy()[bodies[0]]
        com_z = float(model.body_com.numpy()[bodies[0]][2])
        hd, m = setup.holder, float(mass[0])
        centre = np.array([0.0, 0.0, -0.5 * setup.peg_len + hd.l_g])
        peg = PegInertia(m, float(inertia[0, 0]), float(inertia[2, 2]), centre[2] - com_z)
        self.damping = holder_damping(hd, peg, setup.dt)
        self.params = self._params(setup, m, centre)
        q0 = model.body_q.numpy()[bodies].astype(np.float64)
        com = model.body_com.numpy()[bodies[0]].astype(np.float64)
        pos_c, q_ref = hanging_reference(q0, com, centre, m * setup.gravity, hd.k_theta)
        p_ref = pos_c + np.array([0.0, 0.0, m * setup.gravity / hd.k_z])
        hw = HolderWorlds()
        hw.bodies = wp.array(bodies, dtype=wp.int32, device=dev)
        hw.p_ref = wp.array(p_ref.astype(np.float32), dtype=wp.vec3, device=dev)
        hw.q_ref = wp.array(q_ref.astype(np.float32), dtype=wp.quat, device=dev)
        hw.t_stop = wp.full(len(bodies), T_NEVER, dtype=wp.float32, device=dev)
        hw.acc_f = wp.zeros(len(bodies), dtype=wp.float64, device=dev)
        hw.acc_n = wp.zeros(len(bodies), dtype=wp.int32, device=dev)
        self.worlds, self.n, self.device, self.model = hw, len(bodies), dev, model
        log.info("R1 holder: %d worlds, weight %.4f N, damping %s, margins %s", self.n, self.params.weight,
                 (round(self.damping.c_lat, 4), round(self.damping.c_ax, 4), round(self.damping.c_rot, 6),
                  round(self.damping.c_spin, 8)),
                 {k: round(v, 3) for k, v in self.damping.margins.items()})

    def _params(self, s: HolderSetup, mass: float, centre: np.ndarray) -> HolderParams:
        p, hd, d = HolderParams(), s.holder, self.damping
        p.k_lat, p.k_ax, p.k_rot = hd.k_x, hd.k_z, hd.k_theta
        p.c_lat, p.c_ax, p.c_rot, p.c_spin = d.c_lat, d.c_ax, d.c_rot, d.c_spin
        p.speed, p.t_settle, p.stroke, p.f_stop = s.speed, s.proto.t_settle, s.proto.stroke, s.proto.f_stop
        p.weight, p.top_z = mass * s.gravity, s.top_z
        p.centre = wp.vec3(*centre.tolist())
        p.tip = wp.vec3(0.0, 0.0, -0.5 * s.peg_len)
        return p

    def apply(self, state: Any, t: float) -> None:
        """Write the holder wrench into ``state.body_f`` (call after ``clear_forces``)."""
        wp.launch(_drive, dim=self.n, inputs=[state.body_q, state.body_qd, self.model.body_com, self.params,
                                              self.worlds, float(t)], outputs=[state.body_f], device=self.device)

    def record(self, state: Any, t: float, col: int, log_: Any) -> None:
        """Accumulate the insertion force; at a recorded column write ``depth``, ``force`` (mean), ``tilt`` and the
        holder's lateral force ``fx`` and moment ``my`` in the lean plane."""
        outs = [log_.channel(name) for name in ("depth", "force", "tilt", "fx", "my")]
        wp.launch(_record, dim=self.n, inputs=[state.body_q, state.body_qd, self.model.body_com, self.params,
                                               self.worlds, float(t), int(col)], outputs=outs, device=self.device)

    def stopped(self) -> np.ndarray:
        """Host readback after the run: True where the stage latched on the force limit."""
        return self.worlds.t_stop.numpy() < 0.5 * T_NEVER


__all__ = ["HolderParams", "HolderWorlds", "HolderSetup", "HolderDrive", "PegInertia", "Damping", "holder_damping",
           "hanging_reference"]
