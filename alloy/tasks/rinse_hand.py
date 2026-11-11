# SPDX-License-Identifier: Apache-2.0
"""A privileged scripted hand for the rinse_* proofs: holds free bodies at target poses.

``PoseHand`` acts like a robot's grasp under Cartesian impedance control: while it grips a body it pulls the body's
centre of mass toward a target (PD on position and velocity, plus the body's weight) and turns the body toward a
target orientation (PD on the rotation vector and the spin), with forces and torques capped. Released, it applies
nothing and the body is free (it rests in its rack, or falls). Targets are inputs the task's policy sets between
control periods: ``target_pos`` (body origin, world) [W, N] vec3, ``target_quat`` (x, y, z, w) [W, N], ``target_vel``
(feed-forward COM velocity) [W, N] vec3 and ``grip`` [W, N] in {0, 1}. Forces from the scene (jets, contacts) act
on the held body as on a held object, so the tracking error shows the load. No robot: like ``sand_plate``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin, Forces, SensorPlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import add_wrench


@wp.struct
class Gains:
    kp: wp.float32            # N/m
    kd: wp.float32            # N·s/m
    kr: wp.float32            # N·m/rad
    dr: wp.float32            # N·m·s/rad
    f_max: wp.float32         # N
    t_max: wp.float32         # N·m


@wp.func
def clamp_norm(v: wp.vec3, cap: float) -> wp.vec3:
    n = wp.length(v)
    if n > cap:
        return v * (cap / n)
    return v


@wp.func
def rotation_error(q_target: wp.quat, q: wp.quat) -> wp.vec3:
    """Rotation vector (world) turning ``q`` into ``q_target``."""
    e = q_target * wp.quat_inverse(q)
    if e[3] < 0.0:
        e = wp.quat(-e[0], -e[1], -e[2], -e[3])
    v = wp.vec3(e[0], e[1], e[2])
    s = wp.length(v)
    if s < 1.0e-9:
        return 2.0 * v
    return v * (2.0 * wp.atan2(s, e[3]) / s)


@wp.kernel
def hand_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                body_mass: wp.array[wp.float32], held: wp.array2d[wp.int32], target_pos: wp.array2d[wp.vec3],
                target_quat: wp.array2d[wp.quat], target_vel: wp.array2d[wp.vec3], grip: wp.array2d[wp.float32],
                gravity: wp.vec3, g: Gains, error: wp.array2d[wp.float32], body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    for k in range(held.shape[1]):
        b = held[w, k]
        X = body_q[b]
        q = wp.transform_get_rotation(X)
        com = wp.transform_point(X, body_com[b])
        com_t = target_pos[w, k] + wp.quat_rotate(target_quat[w, k], body_com[b])
        error[w, k] = wp.length(com_t - com)
        if grip[w, k] <= 0.0:
            continue
        qd = body_qd[b]
        v = wp.vec3(qd[0], qd[1], qd[2])
        om = wp.vec3(qd[3], qd[4], qd[5])
        f = g.kp * (com_t - com) + g.kd * (target_vel[w, k] - v) - body_mass[b] * gravity
        tau = g.kr * rotation_error(target_quat[w, k], q) - g.dr * om
        f = clamp_norm(f, g.f_max)
        tau = clamp_norm(tau, g.t_max)
        add_wrench(body_f, b, wp.spatial_vector(f[0], f[1], f[2], tau[0], tau[1], tau[2]))


class PoseHand(ForcePlugin):
    """Holds the bodies ``bodies`` (labels) at target poses while gripping them (see the module docstring)."""

    name = "pose_hand"
    summary = "a privileged scripted hand: PD pose tracking with weight compensation on gripped bodies"

    @dataclass
    class Config:
        bodies: tuple[str, ...] = field(default_factory=tuple)
        kp: float = 2000.0            # N/m
        kd: float = 60.0              # N·s/m
        kr: float = 2.0               # N·m/rad
        dr: float = 0.06              # N·m·s/rad
        f_max: float = 80.0           # N
        t_max: float = 8.0            # N·m

        def __post_init__(self) -> None:
            self.bodies = tuple(str(b) for b in self.bodies)
            if not self.bodies:
                raise ValueError("PoseHand needs at least one body")
            if min(self.kp, self.kd, self.kr, self.dr, self.f_max, self.t_max) < 0.0:
                raise ValueError("gains and caps must be >= 0")

    def setup(self, ctx: Any) -> None:
        c, W, N = self.config, ctx.world_count, len(self.config.bodies)
        cols = [ctx.bodies(b).numpy() for b in c.bodies]
        self.held = ctx.array(np.stack(cols, axis=1).astype(np.int32), wp.int32)
        q0 = ctx.model.body_q.numpy()[self.held.numpy()]                          # [W, N, 7]
        self.target_pos = self.add_input("target_pos", ctx.array(q0[..., :3].astype(np.float32), wp.vec3))
        self.target_quat = self.add_input("target_quat", ctx.array(q0[..., 3:].astype(np.float32), wp.quat))
        self.target_vel = self.add_input("target_vel", ctx.zeros((W, N), wp.vec3))
        self.grip = self.add_input("grip", ctx.zeros((W, N), wp.float32))
        self.error = self.add_state("error", ctx.zeros((W, N), wp.float32))
        self.gains = Gains()
        for name in ("kp", "kd", "kr", "dr", "f_max", "t_max"):
            setattr(self.gains, name, float(getattr(c, name)))
        self.gravity = wp.vec3(*(float(v) for v in ctx.gravity))

    def check_dt(self, dt: float) -> None:
        """Explicit PD on each held body's own mass and inertia: refuse an unstable exchange step."""
        m = self.ctx.model
        mass, inertia = m.body_mass.numpy(), m.body_inertia.numpy()
        c = self.config
        for k, name in enumerate(c.bodies):
            b = int(self.held.numpy()[0, k])
            i_min = float(np.min(np.linalg.eigvalsh(inertia[b])))
            lin = (c.kp * dt * dt / mass[b], c.kd * dt / mass[b])
            rot = (c.kr * dt * dt / i_min, c.dr * dt / i_min)
            if max(lin[0], rot[0]) > 0.5 or max(lin[1], rot[1]) > 1.0:
                raise PluginError(f"{self.label}: gains unstable on {name!r} at dt {dt:.3g} s (mass {mass[b]:.3g} kg, "
                                  f"inertia {i_min:.3g} kg·m²): lower the gains or the exchange step")

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        m = self.ctx.model
        wp.launch(hand_kernel, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, m.body_com, m.body_mass, self.held, self.target_pos,
                          self.target_quat, self.target_vel, self.grip, self.gravity, self.gains],
                  outputs=[self.error, forces.body_f], device=self.ctx.device)


def mjw_solver(sim: Any) -> Any | None:
    """The MuJoCo Warp solver stepping ``sim`` (None on a route without one, e.g. MuJoCo C)."""
    return next((s for s in sim.stepper.solvers() if getattr(s, "mjw_data", None) is not None), None)


@wp.kernel
def or_overflow_kernel(bits: wp.array[wp.int32], mask: wp.int32, acc: wp.array[wp.int32]):
    w = wp.tid()
    acc[w] = acc[w] | (bits[w] & mask)


def capacity_mask() -> int:
    """MuJoCo Warp's overflow bits other than the iteration limits (an int32 mask): dropped contacts or rows."""
    from alloy._vendor.mujoco_warp import OverflowType  # noqa: PLC0415

    return ~int(OverflowType.ITERATIONS | OverflowType.LS_ITERATIONS)


class OverflowWatch(SensorPlugin):
    """``[W]`` the OR of MuJoCo Warp's capacity overflow bits after every solver step since the last ``take``. Newton
    zeroes the word before each MuJoCo step (and the MuJoCo C route at each call), so one read per control period
    sees only its last substep (DC5's review of JOB3's rinse-sponge: 2 of 107 overflowing periods missed at 20
    substeps). The plug-in host attaches it at install (``attach(stepper)``: the MuJoCo Warp solver among the
    stepper's ``solvers()``), before any GPU period graph captures its launch; ``overflow_watch(sim)`` attaches it
    again. Without the word (no MuJoCo solver) it reads None. It sees one MuJoCo step per stepper step: a solver that
    substeps inside one call would still lose the inner steps' bits."""

    name = "overflow_watch"
    summary = "accumulates MuJoCo Warp's capacity overflow bits over every substep"

    @dataclass
    class Config:
        pass

    def setup(self, ctx: Any) -> None:
        self.acc = self.add_state("overflow", ctx.zeros(ctx.world_count, wp.int32))
        self.bits: wp.array | None = None

    def attach(self, target: Any | None) -> OverflowWatch:
        """Watch the MuJoCo Warp solver ``target``, or the one among a stepper's ``solvers()``; none: unattached."""
        found = target.solvers() if callable(getattr(target, "solvers", None)) else [target]
        solver = next((s for s in found if getattr(s, "mjw_data", None) is not None), None)
        bits = getattr(getattr(solver, "mjw_data", None), "overflow", None)
        self.bits = bits if isinstance(bits, wp.array) else None
        self.mask = wp.int32(capacity_mask()) if self.bits is not None else wp.int32(0)
        return self

    def measure(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        if self.bits is not None:
            wp.launch(or_overflow_kernel, dim=self.ctx.world_count, inputs=[self.bits, self.mask],
                      outputs=[self.acc], device=self.ctx.device)

    def take(self) -> np.ndarray | None:
        """``[W]`` True where any step since the last take dropped contacts or rows; clears the record."""
        if self.bits is None:
            return None
        out = self.acc.numpy() != 0
        self.acc.zero_()
        return out


def overflow_watch(sim: Any) -> OverflowWatch | None:
    """``sim``'s ``OverflowWatch`` plug-in attached to its MuJoCo Warp solver; None when it has none. Call before the
    first step (a GPU period graph captures the watch's launch only if it is attached by then)."""
    host = getattr(getattr(sim, "stepper", None), "_alloy_plugin_host", None)
    watch = next((p for p in getattr(host, "plugins", {}).values() if isinstance(p, OverflowWatch)), None)
    return None if watch is None else watch.attach(mjw_solver(sim))


def period_overflow(watch: OverflowWatch | None, solver: Any | None) -> np.ndarray | None:
    """``[W]`` contacts or rows dropped in the control period just stepped: in any substep with an attached watch,
    else in its last substep only (``contact_overflow``). None without the bits."""
    if watch is not None and watch.bits is not None:
        return watch.take()
    return contact_overflow(solver)


def contact_overflow(solver: Any | None) -> np.ndarray | None:
    """``[W]`` True where MuJoCo Warp ran out of contact or constraint capacity in the last step (its ``overflow``
    bits other than the iteration limits): contacts were dropped and the step is not to be trusted. None without the
    bits. One step only: ``OverflowWatch`` keeps every substep's."""
    bits = getattr(getattr(solver, "mjw_data", None), "overflow", None)
    if bits is None:
        return None
    return (bits.numpy().astype(np.int64) & capacity_mask()) != 0


__all__ = ["OverflowWatch", "PoseHand", "capacity_mask", "contact_overflow", "mjw_solver", "overflow_watch",
           "period_overflow"]
