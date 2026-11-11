# SPDX-License-Identifier: Apache-2.0
"""A privileged scripted carriage for the concrete proof tasks: it drives free tool bodies (floats, trowels, darby,
jointer) through stroke programs, like Gym-Anything's phantom-carriage probes (``probes/common.py:1-4``, forces through
``world.ext_wrench``) and alloy's ``sand_head`` gantry. No robot: the proofs are the paste physics and the verdict.

Each tool follows keyframes on the device clock (``ctx.time``). A keyframe holds the tool body's origin target (x, y,
z), its yaw about +z and pitch about its own +y (the leading edge up), a normal force and a mode for the segment it
starts:
- ``FORCE``: x, y and the attitude follow the program; the tool presses on the work (along -z) with ``force`` in
  total, as a finisher leaning on a float (its weight included when gravity is -z);
- ``HEIGHT``: z follows the program too (a screed or strike-off at a set height), with the vertical push capped at
  ``force_cap``.
Between keyframes the targets are linear in time and the velocity target is the segment's slope; after the last
keyframe the tool holds it. Every world runs the same programs, or each its own (``world_programs``: a policy that
aims at each world's seeded start, as a mason aims at the low spot in view). The carriage carries the tool's weight
along the scene's gravity, so a work face laid flat with gravity along it (a wall's joint, gravity -y) is pressed with
``force`` exactly. All gains act on the tool's own mass and inertia; ``check_dt`` refuses unstable ones. The applied
vertical load (weight plus push, N, down positive) of every tool is published per world as the state ``load`` (the
process layer's pad pressure until the film publishes its own).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import add_wrench

FORCE, HEIGHT = 0, 1
STABLE = 1.0
G = 9.81


@wp.struct
class Key:
    t: wp.float32          # [s] on the world's clock
    pos: wp.vec3           # body origin target [m], world frame
    yaw: wp.float32        # [rad] about +z
    pitch: wp.float32      # [rad] about the body's +y
    force: wp.float32      # [N] total normal force on the work (FORCE segments)
    mode: wp.int32         # FORCE or HEIGHT for the segment this keyframe starts
    release: wp.float32    # share of a store tool's free load let go each exchange of the segment


@wp.struct
class Gains:
    kp: wp.float32         # [N/m] in-plane (and vertical in HEIGHT segments)
    kd: wp.float32         # [N·s/m]
    kd_z: wp.float32       # [N·s/m] vertical damping in FORCE segments
    k_rot: wp.float32      # [N·m/rad]
    d_rot: wp.float32      # [N·m·s/rad]


@dataclass(frozen=True)
class Stroke:
    """One keyframe: ``t`` [s], body origin ``pos`` (x, y, z) [m], ``yaw`` and ``pitch`` [rad], the total normal
    ``force`` [N] and the ``mode`` (FORCE or HEIGHT) of the segment it starts."""

    t: float
    pos: tuple[float, float, float]
    yaw: float = 0.0
    pitch: float = 0.0
    force: float = 0.0
    mode: int = FORCE
    release: float = 0.0      # 0-1: a store tool lets go of this share of its free load each exchange of the
                              # segment this key starts (surface flow's tool_release; with a supply it falls)


@dataclass(frozen=True)
class ToolProgram:
    """A tool body (its label) and its keyframes in time order."""

    body: str
    keys: tuple[Stroke, ...]

    def __post_init__(self) -> None:
        if not self.keys:
            raise PluginError(f"tool {self.body!r}: a program needs at least one keyframe")
        times = [k.t for k in self.keys]
        if any(b <= a for a, b in zip(times, times[1:], strict=False)):
            raise PluginError(f"tool {self.body!r}: keyframe times must increase, got {times}")
        for k in self.keys:
            if k.mode not in (FORCE, HEIGHT) or not (math.isfinite(k.force) and k.force >= 0.0):
                raise PluginError(f"tool {self.body!r}: mode must be FORCE or HEIGHT and force finite >= 0 ({k})")
            if not 0.0 <= k.release <= 1.0:
                raise PluginError(f"tool {self.body!r}: release must be in [0, 1] ({k})")

    @property
    def duration(self) -> float:
        return self.keys[-1].t


@wp.func
def _segment(keys: wp.array2d[Key], n: wp.array[wp.int32], r: int, t: float):
    """(index of the segment's start key, its fraction) for program row ``r`` at time ``t``; past the end: the last
    key."""
    last = n[r] - 1
    i = int(0)
    for j in range(last):
        if keys[r, j + 1].t <= t:
            i = j + 1
    if i >= last:
        return last, float(0.0)
    a, b = keys[r, i], keys[r, i + 1]
    return i, wp.clamp((t - float(a.t)) / float(b.t - a.t), 0.0, 1.0)


@wp.func
def _attitude(yaw: float, pitch: float):
    return wp.mul(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
                  wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), pitch))


@wp.kernel
def _carriage_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                     body_com: wp.array[wp.vec3], tool: wp.array2d[wp.int32], mass: wp.array[wp.float32],
                     keys: wp.array2d[Key], n: wp.array[wp.int32], n_tools: int, time: wp.array[wp.float64],
                     gain: wp.array[Gains], force_cap: float, grav: wp.vec3,
                     body_f: wp.array[wp.spatial_vector], load: wp.array2d[wp.float32],
                     release: wp.array2d[wp.float32]):
    w, k = wp.tid()
    r = w * n_tools + k                                         # world w's program of tool k
    b = tool[w, k]
    t = float(time[w])
    i, s = _segment(keys, n, r, t)
    a = keys[r, i]
    last = n[r] - 1
    target = a.pos
    vel = wp.vec3(0.0, 0.0, 0.0)
    yaw, pitch, force = float(a.yaw), float(a.pitch), float(a.force)
    if i < last:
        c = keys[r, i + 1]
        span = float(c.t - a.t)
        target = a.pos + s * (c.pos - a.pos)
        vel = (c.pos - a.pos) / span
        yaw = yaw + s * float(c.yaw - a.yaw)
        pitch = pitch + s * float(c.pitch - a.pitch)
        force = force + s * float(c.force - a.force)
    origin = wp.transform_get_translation(body_q[b])
    qd = body_qd[b]
    v = wp.vec3(qd[0], qd[1], qd[2])
    omega = wp.vec3(qd[3], qd[4], qd[5])
    m = float(mass[k])
    gains = gain[k]
    fx = gains.kp * (target[0] - origin[0]) + gains.kd * (vel[0] - v[0]) - m * grav[0]   # the weight carried
    fy = gains.kp * (target[1] - origin[1]) + gains.kd * (vel[1] - v[1]) - m * grav[1]   # in plane (0 when -z)
    wz = -m * grav[2]                                           # the lift that carries the weight's z part
    fz = wz - force - gains.kd_z * v[2]                         # FORCE: weight + push = force on the work
    if a.mode == 1:                                             # HEIGHT: hold z, push at most force_cap
        fz = wz + gains.kp * (target[2] - origin[2]) + gains.kd * (vel[2] - v[2])
        fz = wp.clamp(fz, wz - force_cap, wz + force_cap)
    q_err = wp.mul(_attitude(yaw, pitch), wp.quat_inverse(wp.transform_get_rotation(body_q[b])))
    sign = wp.where(q_err[3] < 0.0, -1.0, 1.0)
    rot = 2.0 * sign * wp.vec3(q_err[0], q_err[1], q_err[2])
    torque = gains.k_rot * rot - gains.d_rot * omega
    add_wrench(body_f, b, wp.spatial_vector(fx, fy, fz, torque[0], torque[1], torque[2]))
    load[w, k] = wp.float32(wz - fz)                          # down positive: weight plus the carriage's push
    release[w, k] = a.release


@wp.kernel
def _release_kernel(release: wp.array2d[wp.float32], index: wp.array[wp.int32],
                    tool_release: wp.array2d[wp.float64]):
    """The carriage's release shares into the film's per-tool input (its tool ``index[k]``; -1: not a film tool)."""
    w, k = wp.tid()
    i = index[k]
    if i >= 0:
        tool_release[w, i] = wp.float64(release[w, k])


class ToolCarriage(ForcePlugin):
    """Scripted carriage: every tool body follows its keyframed strokes (FORCE or HEIGHT segments)."""

    name = "concrete_carriage"
    summary = "scripted tool carriage: keyframed strokes, pressing force or held height, level attitude PD"

    def __init__(self, config: Any = None, *, flow: Any = None, label: str | None = None, **params: Any) -> None:
        """``flow``: a SurfaceFlow whose store tools the keyframes' ``release`` drives (its ``tool_release`` input,
        matched by body); without one, ``release`` is only published. A runtime link, as ``ConcretePaste``'s
        ``flow=``: not part of the config, so a saved config carries no flow."""
        super().__init__(config, label=label, **params)
        self.flow = flow

    @dataclass
    class Config:
        programs: tuple[Any, ...] = ()                     # ToolProgram (or dicts): the tools, in order
        world_programs: tuple[Any, ...] = ()               # optional: per world, its programs of the same tools
        omega: float = 400.0                               # rad/s in-plane servo on each tool's own mass
        zeta: float = 0.7                                  # (20 N of friction on 0.12 kg leaves 1 mm)
        omega_z: float = 40.0                              # rad/s: vertical damping of FORCE segments (2 ζ m ω_z)
        omega_rot: float = 150.0                           # rad/s attitude servo on each tool's smallest inertia
        zeta_rot: float = 0.7
        force_cap: float = 200.0                           # N (HEIGHT segments)

        def __post_init__(self) -> None:
            self.programs = tuple(_program(p) for p in self.programs)
            if not self.programs:
                raise ValueError("a carriage needs at least one tool program")
            self.world_programs = tuple(tuple(_program(p) for p in ps) for ps in self.world_programs)
            for w, ps in enumerate(self.world_programs):
                if [p.body for p in ps] != [p.body for p in self.programs]:
                    raise ValueError(f"world_programs[{w}] must drive the tools {[p.body for p in self.programs]} "
                                     f"in that order, got {[p.body for p in ps]}")
            for name in ("omega", "zeta", "omega_z", "omega_rot", "zeta_rot", "force_cap"):
                v = getattr(self, name)
                if not (math.isfinite(v) and v >= 0.0):
                    raise ValueError(f"{name} must be finite and >= 0, got {v}")

    def setup(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        W, n_tools = ctx.world_count, len(c.programs)
        bodies = np.stack([ctx.bodies(p.body).numpy() for p in c.programs], axis=1).astype(np.int32)   # [W, T]
        self.tool = ctx.array(bodies, wp.int32)
        b0 = bodies[0]
        self.mass_np = model.body_mass.numpy()[b0].astype(np.float32)
        inert = model.body_inertia.numpy()[b0].reshape(n_tools, 3, 3).astype(np.float64)
        self.inertia_np = np.array([np.linalg.eigvalsh(i).min() for i in inert])
        self.mass = ctx.array(self.mass_np, wp.float32)
        sets = self._sets(W)
        K = max(len(p.keys) for ps in sets for p in ps)
        table = np.zeros((W * n_tools, K), dtype=Key.numpy_dtype())        # row w * T + k: world w, tool k
        for w, ps in enumerate(sets):
            for k, p in enumerate(ps):
                for j in range(K):                         # padding repeats the last key (never read: n[row])
                    s = p.keys[min(j, len(p.keys) - 1)]
                    table[w * n_tools + k, j] = (s.t, s.pos, s.yaw, s.pitch, s.force, s.mode, s.release)
        self.keys = wp.array(table, dtype=Key, device=ctx.device)
        self._table, self._tools = table, n_tools               # host copy for set_heights (pad-trowel's lifts)
        self.n = ctx.array(np.array([len(p.keys) for ps in sets for p in ps], np.int32), wp.int32)
        gains = np.zeros(n_tools, dtype=Gains.numpy_dtype())
        for k in range(n_tools):                           # each tool's own mass and inertia set its gains
            m, i = float(self.mass_np[k]), float(self.inertia_np[k])
            gains[k] = (m * c.omega ** 2, 2.0 * c.zeta * m * c.omega, 2.0 * c.zeta * m * c.omega_z,
                        i * c.omega_rot ** 2, 2.0 * c.zeta_rot * i * c.omega_rot)
        self.gain_np = gains
        self.gain = wp.array(gains, dtype=Gains, device=ctx.device)
        self.load = self.add_state("load", ctx.zeros((W, n_tools), wp.float32))
        self.release = self.add_state("release", ctx.zeros((W, n_tools), wp.float32))
        self._release_index = None                          # the film's tool of each program, on the first apply
        self.grav = wp.vec3(*(float(v) for v in ctx.gravity))

    def set_heights(self, tool: int, world: int, heights: Mapping[int, float]) -> None:
        """Set the z [m] of keys ``index -> z`` of tool ``tool`` (its index in ``programs``) in world ``world``, in
        place: the same device array, so a captured graph reads them on its next launch. Times, xy, yaw and the rest
        of every key stay as they are (pad-trowel's lifts: a set-down's height read at the lift, WP15 2026-11-08)."""
        row = world * self._tools + tool
        before = self._table[row].copy()
        for j, z in heights.items():
            if not 0 <= j < len(before):
                raise IndexError(f"{self.label}: key {j} of tool {tool} (it has {len(before)})")
            self._table["pos"][row, j, 2] = z
        after = self._table[row]
        same = [n for n in before.dtype.names if n != "pos"]
        if any(not np.array_equal(before[n], after[n]) for n in same) or not np.array_equal(before["pos"][:, :2],
                                                                                          after["pos"][:, :2]):
            raise PluginError(f"{self.label}: set_heights changed more than z")
        self.keys.assign(self._table)

    def _sets(self, worlds: int) -> tuple[tuple[ToolProgram, ...], ...]:
        """Each world's programs: its own from ``world_programs``, else the shared ``programs``."""
        c = self.config
        if not c.world_programs:
            return (c.programs,) * worlds
        if len(c.world_programs) != worlds:
            raise PluginError(f"{self.label}: world_programs has {len(c.world_programs)} sets for {worlds} worlds")
        return c.world_programs

    @property
    def duration(self) -> float:
        """Seconds until the last tool's last keyframe (in any world)."""
        c = self.config
        return max(p.duration for ps in (c.world_programs or (c.programs,)) for p in ps)

    def check_dt(self, dt: float) -> None:
        """Explicit gains on each tool's own mass and smallest inertia: sqrt(k/m)·dt and c·dt/m below 1."""
        c = self.config
        checks = {"omega": c.omega * dt, "kd": 2.0 * c.zeta * c.omega * dt, "kd_z": 2.0 * c.zeta * c.omega_z * dt,
                  "omega_rot": c.omega_rot * dt, "d_rot": 2.0 * c.zeta_rot * c.omega_rot * dt}
        bad = {k: round(v, 3) for k, v in checks.items() if v >= STABLE}
        if bad:
            raise PluginError(f"{self.label}: gains unstable at dt={dt:.3g} s: {bad} (each < {STABLE}); lower omega, "
                              f"omega_rot or the damping ratios")

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        W, T = self.ctx.world_count, len(self.config.programs)
        wp.launch(_carriage_kernel, dim=(W, T),
                  inputs=[state.body_q, state.body_qd, self.ctx.model.body_com, self.tool, self.mass, self.keys,
                          self.n, T, self.ctx.time, self.gain, float(self.config.force_cap), self.grav],
                  outputs=[forces.body_f, self.load, self.release], device=self.ctx.device)
        target = getattr(self.flow, "tool_release", None)
        if target is None:
            return
        if self._release_index is None:
            bodies = [t.body for t in self.flow.config.tools]
            idx = [bodies.index(p.body) if p.body in bodies else -1 for p in self.config.programs]
            self._release_index = self.ctx.array(np.asarray(idx, np.int32), wp.int32)
        wp.launch(_release_kernel, dim=(W, T), inputs=[self.release, self._release_index], outputs=[target],
                  device=self.ctx.device)


def add_tool_body(b: newton.ModelBuilder, label: str, half: tuple[float, float, float], mass: float,
                  pos: tuple[float, float, float], collide: bool = True) -> None:
    """A free tool body: a box of half extents ``half`` [m] and ``mass`` [kg] at ``pos`` (its sole is the box's
    underside). ``collide=False``: no rigid contacts, so the tool meets the work only through the film's pad (the
    same physics on the CPU's MuJoCo C and the GPU's MuJoCo Warp, whose box-box contacts differ)."""
    body = b.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()), label=label)
    cfg = newton.ModelBuilder.ShapeConfig(density=mass / (8.0 * half[0] * half[1] * half[2]))
    cfg.has_shape_collision = collide
    b.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)


def lane(keys: list[Stroke], t: float, p0: tuple[float, float], p1: tuple[float, float], z: float, yaw: float,
         force: float, speed: float, air: float, mode: int = FORCE, lift: float = 0.03) -> float:
    """Append one working lane to ``keys``: travel in the air to p0 (``air`` m/s, ``lift`` above ``z``), set down,
    work to p1 at ``speed`` (pressing ``force`` in FORCE mode, or held at ``z`` in HEIGHT mode), lift. Returns the
    end time."""
    up = z + lift
    t += math.dist(keys[-1].pos[:2], p0) / air + 0.3
    keys.append(Stroke(t, (*p0, up), yaw=yaw, mode=HEIGHT))
    t += 0.4
    keys.append(Stroke(t, (*p0, z), yaw=yaw, force=force, mode=mode))
    t += math.dist(p0, p1) / speed
    keys.append(Stroke(t, (*p1, z), yaw=yaw, force=force, mode=HEIGHT))
    t += 0.3
    keys.append(Stroke(t, (*p1, up), yaw=yaw, mode=HEIGHT))
    return t


def _program(p: Any) -> ToolProgram:
    if isinstance(p, ToolProgram):
        return p
    if isinstance(p, dict):
        return ToolProgram(body=p["body"], keys=tuple(k if isinstance(k, Stroke) else Stroke(**k) for k in p["keys"]))
    raise PluginError(f"carriage: expected a ToolProgram or a dict, got {p!r}")


__all__ = ["FORCE", "HEIGHT", "Stroke", "ToolCarriage", "ToolProgram", "add_tool_body", "lane"]
