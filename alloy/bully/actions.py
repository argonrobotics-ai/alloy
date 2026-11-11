# SPDX-License-Identifier: Apache-2.0
"""Legal action traces for the bully: knot splines, legalisation, and scattering into a Newton ``Control``.

A trace is ``[W, T, A]`` (worlds, action ticks, actuators) in SI. The B1 search works on normalised knots
``[W, K, A]`` in [0, 1] of each actuator's range; ``knot_basis`` maps them to ticks with a uniform Catmull-Rom spline
(interpolating, C1, local). Every trace that reaches a simulation passes ``legalize`` first: clip to the box, then
rate-limit by ``max_delta_per_tick`` starting from the initial command; the number of modified entries per world is
returned and logged (bench §4.7: "commands are clipped in code and the number of clips is logged").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.core.errors import BullyError

log = logging.getLogger("alloy.bully")


def tick_count(horizon_s: float, rate_hz: float) -> int:
    """Number of action ticks in ``horizon_s`` at ``rate_hz`` (at least 1)."""
    if horizon_s <= 0.0 or rate_hz <= 0.0:
        raise BullyError(f"horizon_s and rate_hz must be positive, got {horizon_s}, {rate_hz}")
    return max(1, int(round(horizon_s * rate_hz)))


def _catmull_rom_weights(s: float) -> tuple[float, float, float, float]:
    """Weights of p[i-1], p[i], p[i+1], p[i+2] at local parameter s in [0, 1] (uniform Catmull-Rom)."""
    s2, s3 = s * s, s * s * s
    return (
        0.5 * (-s3 + 2.0 * s2 - s),
        0.5 * (3.0 * s3 - 5.0 * s2 + 2.0),
        0.5 * (-3.0 * s3 + 4.0 * s2 + s),
        0.5 * (s3 - s2),
    )


def knot_basis(knots: int, ticks: int) -> np.ndarray:
    """Basis ``M`` [ticks, knots] with ``trace = M @ knot_values``; knots spread evenly over [0, ticks - 1].

    End segments clamp the missing neighbour to the end knot, so a constant knot vector gives a constant trace
    (each row sums to 1) and every knot value is reproduced exactly at its tick.
    """
    if knots < 2:
        raise BullyError(f"need at least 2 knots, got {knots}")
    basis = np.zeros((ticks, knots), dtype=np.float64)
    span = max(ticks - 1, 1) / (knots - 1)
    for j in range(ticks):
        u = j / span
        i = min(int(np.floor(u)), knots - 2)
        w = _catmull_rom_weights(u - i)
        for offset, weight in zip((-1, 0, 1, 2), w, strict=True):
            basis[j, min(max(i + offset, 0), knots - 1)] += weight
    return basis


def knots_to_trace(knots_u: np.ndarray, low: np.ndarray, high: np.ndarray, ticks: int) -> np.ndarray:
    """Normalised knots [W, K, A] (or [K, A]) -> SI trace [W, T, A] (or [T, A]); not yet legalised."""
    squeeze = knots_u.ndim == 2
    k = knots_u[None] if squeeze else knots_u
    u = np.einsum("tk,wka->wta", knot_basis(k.shape[1], ticks), k)
    trace = low + u * (high - low)
    return trace[0] if squeeze else trace


def trace_to_knots(trace: np.ndarray, low: np.ndarray, high: np.ndarray, knots: int) -> np.ndarray:
    """Least-squares normalised knots [K, A] for one SI trace [T, A] (seeds B1 from a B0 trace)."""
    span = np.where(high > low, high - low, 1.0)
    u = (trace - low) / span
    basis = knot_basis(knots, trace.shape[0])
    sol, *_ = np.linalg.lstsq(basis, u, rcond=None)
    return np.clip(sol, 0.0, 1.0)


@dataclass(frozen=True)
class Legalized:
    """A legal trace and how many entries per world were changed to make it legal."""

    trace: np.ndarray  # [W, T, A] float32, SI
    clips: np.ndarray  # [W] int64


def legalize(
    trace: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    max_delta: np.ndarray,
    initial: np.ndarray | None = None,
) -> Legalized:
    """Clip a trace [W, T, A] to [low, high] and rate-limit it by ``max_delta`` per tick from ``initial``.

    With ``initial=None`` the first tick is free inside the box. The result stays inside the box because each
    step moves from a point in the box toward a point in the box.
    """
    raw = np.asarray(trace, dtype=np.float64)
    if raw.ndim == 2:
        raw = raw[None]
    if np.any(np.asarray(max_delta) <= 0.0):
        raise BullyError("max_delta_per_tick must be positive")
    boxed = np.clip(raw, low, high)
    tol = 1e-6 * np.maximum(np.abs(np.asarray(high) - np.asarray(low)), 1.0)  # float32 re-legalisation is no clip
    changed = np.abs(boxed - raw) > tol
    out = np.empty_like(boxed)
    prev = boxed[:, 0] if initial is None else np.broadcast_to(np.clip(initial, low, high), boxed[:, 0].shape)
    for t in range(boxed.shape[1]):
        out[:, t] = prev + np.clip(boxed[:, t] - prev, -max_delta, max_delta)
        prev = out[:, t]
    changed |= np.abs(out - boxed) > tol
    clips = changed.reshape(changed.shape[0], -1).sum(axis=1).astype(np.int64)
    if clips.any():
        log.debug("legalize: %d of %d worlds clipped (max %d entries)", int((clips > 0).sum()), len(clips),
                  int(clips.max()))
    return Legalized(out.astype(np.float32), clips)


ActionKind = Literal["joint_target_q", "joint_target_qd", "joint_f", "body_f", "particle_f", "body_pose", "passive"]
# control array written per kind, and the Model array that gives a joint's first coordinate (q) or dof (qd, f)
_KIND_START: dict[str, str] = {
    "joint_target_q": "joint_q_start",
    "joint_target_qd": "joint_qd_start",
    "joint_f": "joint_qd_start",
}


MUJOCO_CTRL = "mujoco.ctrl"  # control.mujoco.ctrl, world-major [W * nu] (SolverMuJoCo, ctrl_direct actuators)


def global_action_index(model: newton.Model, kind: str, template_index: tuple[int, ...], length: int = 0
                        ) -> np.ndarray:
    """[W, A] indices into the control array for actuator ``a`` of world ``w``.

    ``template_index`` lists coordinates (``joint_target_q``) or dofs (``joint_target_qd``, ``joint_f``) counted from
    the first joint of a world; all worlds of a pack share topology, so each world adds its own first-joint offset.
    """
    if kind == MUJOCO_CTRL:  # ``length`` = len(control.mujoco.ctrl) = W * nu
        nu = int(length) // max(model.world_count, 1)
        return (np.arange(model.world_count)[:, None] * nu + np.asarray(template_index)[None, :]).astype(np.int32)
    if kind not in _KIND_START:
        raise BullyError(f"unknown action kind {kind!r}; expected one of {sorted([*_KIND_START, MUJOCO_CTRL])}")
    joint_world = model.joint_world.numpy()
    starts = getattr(model, _KIND_START[kind]).numpy()
    bases = np.empty(model.world_count, dtype=np.int64)
    for w in range(model.world_count):
        joints = np.nonzero(joint_world == w)[0]
        if joints.size == 0:
            raise BullyError(f"world {w} has no joints; cannot bind actions of kind {kind}")
        bases[w] = starts[joints[0]]
    return (bases[:, None] + np.asarray(template_index, dtype=np.int64)[None, :]).astype(np.int32)


@wp.kernel
def _scatter_tick_kernel(
    trace: wp.array3d[wp.float32],
    tick: int,
    index: wp.array2d[wp.int32],
    out: wp.array[float],
):
    w, a = wp.tid()
    out[index[w, a]] = trace[w, tick, a]


class BoundActions:
    """A legal trace [W, T, A] on the device, written into one ``Control`` array per action tick."""

    def __init__(self, model: newton.Model, control: newton.Control, kind: str, template_index: tuple[int, ...],
                 trace: np.ndarray) -> None:
        target = control
        for part in kind.split("."):  # "joint_target_q", or a namespaced array such as "mujoco.ctrl"
            target = getattr(target, part, None)
        if target is None:
            raise BullyError(f"control.{kind} is not allocated for this model")
        idx = global_action_index(model, kind, template_index, target.shape[0])
        if trace.shape[0] != idx.shape[0] or trace.shape[2] != idx.shape[1]:
            raise BullyError(f"trace {trace.shape} does not match {idx.shape[0]} worlds x {idx.shape[1]} actuators")
        self.ticks = int(trace.shape[1])
        self._out = target
        self._index = wp.array(idx, dtype=wp.int32, device=model.device)
        self._trace = wp.array(np.ascontiguousarray(trace, dtype=np.float32), dtype=wp.float32, device=model.device)

    def apply(self, tick: int) -> None:
        if not 0 <= tick < self.ticks:
            raise BullyError(f"tick {tick} outside [0, {self.ticks})")
        wp.launch(_scatter_tick_kernel, dim=self._index.shape, inputs=[self._trace, tick, self._index, self._out],
                  device=self._out.device)


@wp.kernel
def _scatter_wrench_kernel(
    trace: wp.array3d[wp.float32],
    tick: int,
    body: wp.array2d[wp.int32],
    comp: wp.array2d[wp.int32],
    out: wp.array[wp.spatial_vector],
):
    # one thread per world, actuators in order: several actuators may address components of the same body, and a
    # read-modify-write per (world, actuator) thread would race on the GPU
    w = wp.tid()
    for a in range(body.shape[1]):
        b = body[w, a]
        f = out[b]
        f[comp[w, a]] = trace[w, tick, a]
        out[b] = f


@wp.kernel
def _scatter_particle_force_kernel(
    trace: wp.array3d[wp.float32],
    tick: int,
    particle: wp.array2d[wp.int32],
    comp: wp.array2d[wp.int32],
    out: wp.array[wp.vec3],
):
    # one thread per world, actuators in order (see _scatter_wrench_kernel)
    w = wp.tid()
    for a in range(particle.shape[1]):
        p = particle[w, a]
        f = out[p]
        f[comp[w, a]] = trace[w, tick, a]
        out[p] = f


@dataclass(frozen=True)
class _ForceKind:
    """How a force action kind addresses the model: item world map, components per item, buffer and kernel."""

    world_attr: str  # Model item -> world array
    count_attr: str  # Model item count
    width: int  # components per item: 6 for a body wrench (force xyz, torque xyz), 3 for a particle force
    dtype: type
    state_attr: str  # State array written before every substep
    kernel: wp.Kernel


_FORCE_KINDS: dict[str, _ForceKind] = {
    "body_f": _ForceKind("body_world", "body_count", 6, wp.spatial_vector, "body_f", _scatter_wrench_kernel),
    "particle_f": _ForceKind("particle_world", "particle_count", 3, wp.vec3, "particle_f",
                             _scatter_particle_force_kernel),
}


class HookedForceActions:
    """Actions as applied forces: actuator ``a`` sets component ``index[a] % width`` of template item
    ``index[a] // width`` (world frame; body wrenches at the COM). The command is held in a buffer and copied into
    the State array before every substep through the simulation's ``pre_substep`` hooks (``DirectSim``)."""

    def __init__(self, model: newton.Model, pre_substep: list, kind: str, template_index: tuple[int, ...],
                 trace: np.ndarray) -> None:
        from alloy.bully.roles import template_to_global

        self._kind = _FORCE_KINDS[kind]
        k, dev = self._kind, model.device
        idx = np.asarray(template_index, dtype=np.int64)
        items = template_to_global(getattr(model, k.world_attr).numpy(), model.world_count,
                                   tuple(int(i) for i in idx // k.width))
        self.ticks = int(trace.shape[1])
        self._item = wp.array(items.astype(np.int32), dtype=wp.int32, device=dev)
        self._comp = wp.array(np.broadcast_to(idx % k.width, items.shape).astype(np.int32), dtype=wp.int32,
                              device=dev)
        self._trace = wp.array(np.ascontiguousarray(trace, dtype=np.float32), dtype=wp.float32, device=dev)
        self._cmd = wp.zeros(getattr(model, k.count_attr), dtype=k.dtype, device=dev)
        self._hooks = pre_substep
        pre_substep.append(self._write)

    def _write(self, state: newton.State) -> None:
        wp.copy(getattr(state, self._kind.state_attr), self._cmd)

    def apply(self, tick: int) -> None:
        if not 0 <= tick < self.ticks:
            raise BullyError(f"tick {tick} outside [0, {self.ticks})")
        wp.launch(self._kind.kernel, dim=self._item.shape[0], inputs=[self._trace, tick, self._item, self._comp,
                                                                      self._cmd], device=self._cmd.device)

    def detach(self) -> None:
        if self._write in self._hooks:
            self._hooks.remove(self._write)


@wp.kernel
def _pose_start_kernel(
    trace: wp.array3d[wp.float32],
    tick: int,
    body: wp.array2d[wp.int32],
    comp: wp.array2d[wp.int32],
    body_q: wp.array[wp.transform],
    start: wp.array2d[wp.float32],
    cmd: wp.array2d[wp.float32],
):
    w = wp.tid()
    for a in range(body.shape[1]):
        p = wp.transform_get_translation(body_q[body[w, a]])
        start[w, a] = p[comp[w, a]]
        cmd[w, a] = trace[w, tick, a]


@wp.kernel
def _pose_write_kernel(
    body: wp.array2d[wp.int32],
    comp: wp.array2d[wp.int32],
    start: wp.array2d[wp.float32],
    cmd: wp.array2d[wp.float32],
    frac: float,
    inv_period: float,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    # one thread per world, actuators in order (see _scatter_wrench_kernel)
    w = wp.tid()
    for a in range(body.shape[1]):
        b, c = body[w, a], comp[w, a]
        x = body_q[b]
        p = wp.transform_get_translation(x)
        p[c] = start[w, a] + frac * (cmd[w, a] - start[w, a])
        body_q[b] = wp.transform(p, wp.transform_get_rotation(x))
        v = body_qd[b]
        v[c] = (cmd[w, a] - start[w, a]) * inv_period
        v[3] = 0.0
        v[4] = 0.0
        v[5] = 0.0
        body_qd[b] = v


POSE_FRAME_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0))  # a blade: face normal, edge, cutting side


def frame_from_poses(poses: np.ndarray, axes: tuple = POSE_FRAME_AXES) -> tuple[np.ndarray, np.ndarray]:
    """Per-world action frames from body poses [W, 7] (x, y, z, qx, qy, qz, qw): (origin [W, 3], axes [W, 3, 3]) with
    axes[w, k] = the pose's rotation applied to body-frame direction ``axes[k]``. The default reads a blade built by
    ``alloy.testing.topology.blade_pose``: body x is the face normal, y the edge, -z the cutting side."""
    p = np.asarray(poses, dtype=np.float64)
    u, w = p[:, None, 3:6], p[:, None, 6:7]
    v = np.broadcast_to(np.asarray(axes, dtype=np.float64), (p.shape[0], 3, 3))
    t = 2.0 * np.cross(u, v)
    return p[:, :3].copy(), v + w * t + np.cross(u, t)


def framed_pose_trace(sim: object, index: np.ndarray, trace: np.ndarray) -> np.ndarray:
    """``body_pose`` commands in a per-world action frame -> world positions [W, T, 3].

    The target's sim may define ``bully_action_frame()`` -> (origin [W, 3], axes [W, 3, 3]) or None. A command c is
    then the world position origin[w] + c @ axes[w] (axes[w, k] = world direction of command axis k), so the legal
    box and rate limit hold in that frame. Coordinator ruling 2026-11-06T14:17Z: ``cut_block_free(_copies)`` command
    the blade in its own frame with the face-normal axis held to +-1 mm. A frame addresses one body's three
    components in order. Without the hook (or a None frame) the commands are world coordinates."""
    fn = getattr(sim, "bully_action_frame", None)
    frame = fn() if callable(fn) else None
    if frame is None:
        return trace
    origin, axes = (np.asarray(a, dtype=np.float64) for a in frame)
    w = trace.shape[0]
    if [int(i) % 3 for i in index] != [0, 1, 2] or len({int(i) // 3 for i in index}) != 1:
        raise BullyError(f"a body_pose action frame needs one body's components (0, 1, 2) in order, got {index}")
    if origin.shape != (w, 3) or axes.shape != (w, 3, 3):
        raise BullyError(f"bully_action_frame: origin {origin.shape} / axes {axes.shape}, expected ({w}, 3) / ({w}, 3, 3)")
    return origin[:, None, :] + np.einsum("wtk,wkj->wtj", np.asarray(trace, dtype=np.float64), axes)


class HookedPoseActions:
    """``body_pose`` (SB-7 cutting target, WP10): actions as kinematic body positions. Actuator ``a`` sets world
    coordinate ``index[a] % 3`` of template body ``index[a] // 3`` (or, when the sim defines a per-world action frame,
    frame coordinate ``index[a] % 3``: ``framed_pose_trace``). Each tick the body moves on a straight line from
    where it is to the command over the period's substeps: before substep k of n (``DirectSim.pre_substep``) its
    position is start + k/n (command - start), its linear velocity (command - start) / control_dt and its angular
    velocity 0; the orientation is kept. For bodies the solver does not integrate (colliders: a VBD blade)."""

    WIDTH = 3

    def __init__(self, model: newton.Model, sim: object, template_index: tuple[int, ...], trace: np.ndarray) -> None:
        from alloy.bully.roles import template_to_global

        dev = model.device
        idx = np.asarray(template_index, dtype=np.int64)
        items = template_to_global(model.body_world.numpy(), model.world_count, tuple(int(i) for i in idx // self.WIDTH))
        if trace.shape[0] != items.shape[0] or trace.shape[2] != items.shape[1]:
            raise BullyError(f"trace {trace.shape} does not match {items.shape[0]} worlds x {items.shape[1]} actuators")
        trace = framed_pose_trace(sim, idx, trace)
        self.ticks = int(trace.shape[1])
        self._sim, self._substeps = sim, int(sim.substeps)  # type: ignore[attr-defined]
        self._inv_period = 1.0 / float(sim.control_dt)  # type: ignore[attr-defined]
        self._body = wp.array(items.astype(np.int32), dtype=wp.int32, device=dev)
        self._comp = wp.array(np.broadcast_to(idx % self.WIDTH, items.shape).astype(np.int32), dtype=wp.int32,
                              device=dev)
        self._trace = wp.array(np.ascontiguousarray(trace, dtype=np.float32), dtype=wp.float32, device=dev)
        self._start = wp.zeros(items.shape, dtype=wp.float32, device=dev)
        self._cmd = wp.zeros(items.shape, dtype=wp.float32, device=dev)
        self._k = self._substeps
        self._hooks = sim.pre_substep  # type: ignore[attr-defined]
        self._hooks.append(self._write)

    def apply(self, tick: int) -> None:
        if not 0 <= tick < self.ticks:
            raise BullyError(f"tick {tick} outside [0, {self.ticks})")
        wp.launch(_pose_start_kernel, dim=self._body.shape[0], device=self._cmd.device,
                  inputs=[self._trace, tick, self._body, self._comp, self._sim.state.body_q, self._start, self._cmd])
        self._k = 0

    def _write(self, state: newton.State) -> None:
        self._k = min(self._k + 1, self._substeps)  # past the period's last substep the body holds the command
        wp.launch(_pose_write_kernel, dim=self._body.shape[0], device=self._cmd.device,
                  inputs=[self._body, self._comp, self._start, self._cmd, self._k / self._substeps,
                          self._inv_period, state.body_q, state.body_qd])

    def detach(self) -> None:
        if self._write in self._hooks:
            self._hooks.remove(self._write)


BODY_POSE = "body_pose"
PASSIVE = "passive"  # no command channel: the scene runs untouched (``alloy.bully.audit`` on a scene without actuators)


class PassiveActions:
    """A [W, T, 0] trace: nothing to write; ``apply`` only checks the tick."""

    def __init__(self, trace: np.ndarray) -> None:
        self.ticks = int(trace.shape[1])

    def apply(self, tick: int) -> None:
        if not 0 <= tick < self.ticks:
            raise BullyError(f"tick {tick} outside [0, {self.ticks})")


def bind_actions(sim: object, model: newton.Model, kind: str, template_index: tuple[int, ...],
                 trace: np.ndarray) -> BoundActions | HookedForceActions | HookedPoseActions | PassiveActions:
    """Bind a legal trace to a simulation: control arrays for joint kinds, a pre-substep force for ``body_f`` and
    ``particle_f``, pre-substep kinematic positions for ``body_pose`` (both need ``DirectSim.pre_substep``), nothing
    for ``passive``."""
    if kind == PASSIVE:
        return PassiveActions(trace)
    if kind not in _FORCE_KINDS and kind != BODY_POSE:
        return BoundActions(model, sim.control, kind, template_index, trace)  # type: ignore[attr-defined]
    hooks = getattr(sim, "pre_substep", None)
    if hooks is None:
        raise BullyError(f"{type(sim).__name__} has no pre_substep hooks; {kind} actions need DirectSim")
    if kind == BODY_POSE:
        return HookedPoseActions(model, sim, template_index, trace)
    return HookedForceActions(model, hooks, kind, template_index, trace)
