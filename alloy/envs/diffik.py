# SPDX-License-Identifier: Apache-2.0
"""Task-space (end-effector) actions for arm environments: the action is a hand motion, the env finds joint targets.

Each step the action's (dx, dy, dz) (and optionally a small rotation vector) become joint-target changes by damped
least squares on the arm's Jacobian, for every world at once, on the env's array backend. Joint limits, target kinds
(joint position targets, or MuJoCo ``<general>`` controls) and resets are those of ``alloy.envs.ActionWriter``. The
changes add to the last targets, first pulled back until every arm joint is within ``window`` rad of its measured
position (the arm's lead shrinks as one vector, so the hand keeps its direction), so a hand blocked by a table or a
part does not wind its targets up. The default 0.03 rad is set for the shipped Franka
Panda's gains (choose it again for another arm), and the servos then set the hand's top speed: steer it closed loop.

Example::

    env = alloy.envs.diffik.panda_reach_ee(num_envs=4, device="cpu")       # a ready environment
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(np.array([[1.0, 0.0, 0.0]] * 4))   # hand +2 cm in x

Your own arm::

    spec = EEActionSpec(body="arm/hand", joints=["arm/joint1", "arm/joint2", "arm/joint3"], extra=["arm/gripper"])
    Task(actions=spec, ...)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.envs.actions import UNBOUNDED, WINDOW_STEPS, ActionSpec, ActionWriter, resolve_kind
from alloy.envs.errors import EnvError
from alloy.envs.layout import Layout

log = logging.getLogger("alloy.envs")
EE_COLUMNS = ("ee_dx", "ee_dy", "ee_dz", "ee_rx", "ee_ry", "ee_rz")



# --- batched kinematics (every world at once, on the env's array backend) -------------------------------------

PRISMATIC, REVOLUTE = 0, 1     # newton.JointType


def cross(xp: Any, a: Any, b: Any) -> Any:
    """Cross product over the last axis, any leading axes. Example: ``cross(xp, a, b)``."""
    return xp.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1], a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


def rotate(xp: Any, q: Any, v: Any) -> Any:
    """Rotate (..., 3) vectors by (..., 4) unit quaternions (x, y, z, w). Example: ``rotate(xp, q, v)``."""
    u, w = q[..., 0:3], q[..., 3:4]
    t = 2.0 * cross(xp, u, v)
    return v + w * t + cross(xp, u, t)


def quat_mul(xp: Any, a: Any, b: Any) -> Any:
    """Hamilton product of (..., 4) quaternions (x, y, z, w): ``a`` after ``b``. Example: ``quat_mul(xp, a, b)``."""
    av, aw, bv, bw = a[..., 0:3], a[..., 3:4], b[..., 0:3], b[..., 3:4]
    vec = aw * bv + bw * av + cross(xp, av, bv)
    dot = av[..., 0:1] * bv[..., 0:1] + av[..., 1:2] * bv[..., 1:2] + av[..., 2:3] * bv[..., 2:3]
    return xp.cat([vec, aw * bw - dot], -1)


def solve(xp: Any, a: Any, b: Any) -> Any:
    """Solve batched ``a @ x = b`` with a (W, m, m) and b (W, m, k). Example: ``solve(xp, A, B)``."""
    if xp.name == "torch":
        import torch  # noqa: PLC0415

        return torch.linalg.solve(a, b)
    return np.linalg.solve(a, b)


def damped_least_squares(xp: Any, jac: Any, dx: Any, damping: float, eye: Any) -> Any:
    """Joint change ``J^T (J J^T + damping^2 I)^-1 dx`` for J (W, m, n) and dx (W, m); ``eye`` is (m, m).

    Example: ``dq = damped_least_squares(xp, J[:, :3], dx, 0.05, xp.asarray(np.eye(3)))``."""
    jt = jac.swapaxes(-1, -2)
    y = solve(xp, jac @ jt + (damping * damping) * eye, dx[..., None])
    return (jt @ y)[..., 0]


@dataclass
class ArmKinematics:
    """Index arrays of the arm's joints, built once: where each joint's frame hangs and which joints move the body."""

    xp: Any
    body_rows: Any           # (W,) rows of the end body in body_q
    parent_rows: Any         # (W, n) rows of each joint's parent body (0 where the parent is the world)
    parent_world: Any        # (W, n, 1) 1.0 where the parent is the world (identity frame)
    frame_pos: Any           # (W, n, 3) joint frame in the parent body (joint_X_p)
    frame_quat: Any          # (W, n, 4)
    axis: Any                # (W, n, 3) joint axis in that frame
    hinge: Any               # (W, n, 1) 1.0 for a hinge, 0.0 for a slide
    on_chain: Any            # (W, n, 1) 1.0 when the joint moves the body

    @classmethod
    def build(cls, layout: Layout, engine: Any, xp: Any, body: str, joints: Sequence[str]) -> ArmKinematics:
        """Read the model's tree for ``joints`` (layout names) and ``body``. Example: see the module docstring."""
        model = engine.model
        rows_qd = np.stack([layout.joint(j).qd[:, 0] for j in joints], axis=1)             # (W, n)
        jidx = np.searchsorted(model.joint_qd_start.numpy(), rows_qd, side="right") - 1
        jtype = model.joint_type.numpy()[jidx]
        bad = [j for j, t in zip(joints, jtype[0]) if int(t) not in (PRISMATIC, REVOLUTE)]
        if bad:
            raise EnvError(f"joints {bad} are not hinge or slide joints", "task-space actions drive hinge and slide "
                           "joints only: list the arm's revolute / prismatic joints in EEActionSpec.joints",
                           field="joints")
        body_rows = layout.body(body)
        parent = model.joint_parent.numpy()[jidx]
        x_p = model.joint_X_p.numpy()[jidx]                                                 # (W, n, 7)
        axis = model.joint_axis.numpy()[rows_qd]
        chain = _chain_joints(model, body_rows)
        on_chain = (jidx[:, :, None] == chain[:, None, :]).any(-1)

        def put(a: np.ndarray) -> Any:
            return xp.asarray(a)
        return cls(xp, xp.asarray(body_rows, "int64"), xp.asarray(np.maximum(parent, 0), "int64"),
                   put((parent < 0)[..., None].astype(np.float32)), put(x_p[..., 0:3]), put(x_p[..., 3:7]),
                   put(axis), put((jtype == REVOLUTE)[..., None].astype(np.float32)),
                   put(on_chain[..., None].astype(np.float32)))

    def end_pose(self, body_q: Any) -> tuple[Any, Any]:
        """(position (W, 3), quaternion xyzw (W, 4)) of the end body. Example: ``pos, quat = arm.end_pose(bq)``."""
        q = body_q[self.body_rows]
        return q[:, 0:3], q[:, 3:7]

    def jacobian(self, body_q: Any) -> Any:
        """(W, 6, n) Jacobian of the end body's origin: linear rows then angular rows, world frame.

        Example: ``arm.jacobian(xp.view(engine.state.body_q))``."""
        xp = self.xp
        p_end = body_q[self.body_rows][:, None, :3]                                         # (W, 1, 3)
        pq = body_q[self.parent_rows]                                                       # (W, n, 7)
        identity = xp.asarray(np.array([0, 0, 0, 1], np.float32))
        pquat = pq[..., 3:7] * (1.0 - self.parent_world) + identity * self.parent_world
        ppos = pq[..., 0:3] * (1.0 - self.parent_world)
        anchor = ppos + rotate(xp, pquat, self.frame_pos)
        axis = rotate(xp, quat_mul(xp, pquat, self.frame_quat), self.axis)
        lin_hinge = cross(xp, axis, p_end - anchor)
        lin = self.hinge * lin_hinge + (1.0 - self.hinge) * axis
        ang = self.hinge * axis
        col = xp.cat([lin, ang], -1) * self.on_chain                                        # (W, n, 6)
        return col.swapaxes(1, 2)


def _chain_joints(model: Any, body_rows: np.ndarray) -> np.ndarray:
    """(W, depth) model joint indices from the body up to the root (-1 padded)."""
    child_to_joint = np.full(int(model.body_count), -1, np.int64)
    child_to_joint[model.joint_child.numpy()] = np.arange(int(model.joint_count))
    parents = model.joint_parent.numpy()
    cur, out = body_rows.astype(np.int64), []
    for _ in range(int(model.body_count)):
        j = np.where(cur >= 0, child_to_joint[np.maximum(cur, 0)], -1)
        out.append(j)
        cur = np.where(j >= 0, parents[np.maximum(j, 0)], -1)
        if np.all(cur < 0):
            break
    return np.stack(out, axis=1)


# --- the action spec and its writer ---------------------------------------------------------------------------

@dataclass
class EEActionSpec:
    """Actions as end-effector deltas in [-1, 1]: ``delta`` metres (``rot_delta`` radians) per step at action 1.

    ``joints`` are the arm's joints (or their actuators), ``body`` the end body whose origin moves. ``extra`` joints
    or actuators (a gripper) follow after the end-effector columns in plain delta mode.
    Example: ``EEActionSpec(body="panda/hand", joints=[f"panda/actuator{k}" for k in range(1, 8)])``."""

    body: str
    joints: Sequence[str]
    orientation: bool = False                  # True: 3 more columns (rotation vector) and the hand's turn is held
    delta: float = 0.02                        # metres per step at action 1
    rot_delta: float = 0.1                     # radians per step at action 1
    damping: float = 0.01                      # damped least squares: larger is steadier, undershoots
    extra: Sequence[str] = field(default_factory=tuple)
    extra_delta: float | Sequence[float] = 0.02
    window: float = 0.03                       # rad: arm targets kept near the measured joints (Panda gains; inf: none)
    frame: str = "world"
    kind: str = "auto"                         # as ActionSpec.kind, for the arm and extra joints
    mode: str = "delta"

    def validate(self) -> None:
        """Raise ``EnvError`` on a bad setting. Example: ``EEActionSpec("hand", []).validate()`` raises."""
        if not self.body or not len(self.joints):
            raise EnvError("EEActionSpec needs a body and at least one joint", "pass body='robot/hand' and the arm's "
                           "joints, e.g. joints=['robot/joint1', ...]", field="joints")
        if self.frame != "world":
            raise EnvError(f"frame is {self.frame!r}", "only frame='world' is supported", field="frame")
        for name in ("delta", "rot_delta", "damping", "window"):
            if not float(getattr(self, name)) > 0:
                raise EnvError(f"{name} is {getattr(self, name)}", f"pass a positive number for {name}", field=name)
        n = len(self.extra)
        deltas = np.atleast_1d(np.asarray(self.extra_delta, np.float64))
        if n and deltas.size not in (1, n):
            raise EnvError(f"extra_delta has {deltas.size} values for {n} extra joints", f"pass one number or {n}",
                           field="extra_delta")
        if self.kind == "effort" or self.kind == "velocity":
            raise EnvError(f"kind={self.kind!r}", "task-space actions write joint position targets: use 'auto', "
                           "'position' or 'ctrl'", field="kind")

    def make_writer(self, layout: Layout, engine: Any, xp: Any) -> EEWriter:
        """The writer ``VectorEnv`` uses. Example: ``spec.make_writer(env.layout, env.sim.engine, env.xp)``."""
        return EEWriter(self, layout, engine, xp)


def _target_limits(layout: Layout, kind: str, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Each target column's (lower, upper) range from the robot file, a joint without limits (a continuous hinge)
    open both ways: inverse kinematics sets its target, so it needs no range to scale into."""
    lows, highs = [], []
    for name in names:
        if kind == "ctrl":
            a = layout.actuator(name)
            lows.append(np.array([a.lower], np.float64))
            highs.append(np.array([a.upper], np.float64))
            continue
        r = layout.joint(name)
        width = (r.target_q if kind == "position" else r.qd).shape[1]
        same = width == len(r.lower)
        lows.append(np.asarray(r.lower, np.float64) if same else np.full(width, -np.inf))
        highs.append(np.asarray(r.upper, np.float64) if same else np.full(width, np.inf))
    lo, hi = np.concatenate(lows), np.concatenate(highs)
    open_ = ~(np.isfinite(lo) & np.isfinite(hi)) | (np.abs(lo) >= UNBOUNDED) | (np.abs(hi) >= UNBOUNDED)
    return np.where(open_, -np.inf, lo), np.where(open_, np.inf, hi)


class EEWriter:
    """Turns (W, 3 or 6 + extras) end-effector actions into arm joint targets; same surface as ``ActionWriter``."""

    def __init__(self, spec: EEActionSpec, layout: Layout, engine: Any, xp: Any) -> None:
        spec.validate()
        self.spec, self.layout, self.engine, self.xp = spec, layout, engine, xp
        arm = list(spec.joints)
        self.n_arm, self.n_extra = len(arm), len(spec.extra)
        self.n_ee = 6 if spec.orientation else 3
        names = [*arm, *spec.extra]
        kind = resolve_kind(ActionSpec(names, kind=spec.kind, mode="absolute"), layout)
        lo, hi = _target_limits(layout, kind, names)
        inner = ActionSpec(names, kind=spec.kind, mode="absolute", low=lo, high=hi)
        self.inner = ActionWriter(inner, layout, engine, xp)
        self.arm = ArmKinematics.build(layout, engine, xp, spec.body, [_joint_name(layout, j) for j in arm])
        self._scale = xp.asarray(np.array([spec.delta] * 3 + [spec.rot_delta] * 3, np.float32)[:self.n_ee])
        self._extra_delta = xp.asarray(np.broadcast_to(np.asarray(spec.extra_delta, np.float32),
                                                       (self.n_extra,)).copy())
        self._eye = xp.asarray(np.eye(self.n_ee, dtype=np.float32))
        extra_window = WINDOW_STEPS * np.broadcast_to(np.asarray(spec.extra_delta, np.float32), (self.n_extra,))
        window = np.concatenate([np.full(self.n_arm, spec.window, np.float32), extra_window.astype(np.float32)])
        self._window = xp.asarray(np.where(self.inner.measured_np, window, np.inf).astype(np.float32))
        self.low_np = np.full(self.size, -1.0, np.float32)
        self.high_np = np.ones(self.size, np.float32)
        self.low, self.high = xp.asarray(self.low_np), xp.asarray(self.high_np)

    # --- the surface the environment uses --------------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Action columns per world: 3 or 6, plus the extras. Example: ``writer.size``."""
        return self.n_ee + self.n_extra

    @property
    def kind(self) -> str:
        """The control kind written ("position" or "ctrl"). Example: ``writer.kind``."""
        return self.inner.kind

    @property
    def joints(self) -> list[str]:
        """Arm joints then extras, as named. Example: ``writer.joints``."""
        return self.inner.joints

    @property
    def _world(self) -> Any:
        return self.inner._world

    def column_names(self) -> list[str]:
        """"ee_dx", "ee_dy", "ee_dz" (+ "ee_rx", "ee_ry", "ee_rz"), then the extra joints' names."""
        return [*EE_COLUMNS[:self.n_ee], *self.inner.column_names()[self.n_arm:]]

    def _check(self, actions: Any) -> Any:
        xp, w = self.xp, self.layout.worlds
        shape = tuple(getattr(actions, "shape", np.shape(actions)))
        if shape != (w, self.size):
            raise EnvError(f"actions have shape {shape}", f"pass an array of shape ({w}, {self.size}): one row per "
                           f"world, columns {self.column_names()}", field="actions")
        return xp.asarray(actions, "float32")

    def apply(self, actions: Any) -> Any:
        """Write joint targets for (W, A) actions; returns the targets written (W, arm + extra joints).

        Example: ``targets = writer.apply(np.zeros((env.num_envs, writer.size)))``."""
        xp = self.xp
        a = xp.clip(self._check(actions), -1.0, 1.0)
        current = self._held(xp.view(self.inner.target)[self.inner.rows])                     # (W, arm + extra)
        dq = damped_least_squares(xp, self._rows(self.jacobian()), a[:, :self.n_ee] * self._scale,
                                  self.spec.damping, self._eye)
        parts = [current[:, :self.n_arm] + dq]
        if self.n_extra:
            parts.append(current[:, self.n_arm:] + self._extra_delta * a[:, self.n_ee:])
        return self.inner.apply(xp.cat(parts, -1))

    def reset(self, mask: Any) -> None:
        """Put the build-time targets back for the worlds in the bool (W,) ``mask``. Example: ``writer.reset(done)``."""
        self.inner.reset(mask)

    # --- kinematics ------------------------------------------------------------------------------------------------

    def jacobian(self) -> Any:
        """(W, 6, n_arm) Jacobian of the end body's origin at the live state. Example: ``writer.jacobian()``."""
        return self.arm.jacobian(self.xp.view(self.engine.state.body_q))

    def pose(self) -> tuple[Any, Any]:
        """(position (W, 3), quaternion xyzw (W, 4)) of the end body now. Example: ``pos, quat = writer.pose()``."""
        return self.arm.end_pose(self.xp.view(self.engine.state.body_q))

    def _rows(self, jac: Any) -> Any:
        return jac[:, :self.n_ee]

    def _held(self, current: Any) -> Any:
        """Last targets pulled toward the measured joints: the arm's lead (target - joint) shrunk as one vector until
        every arm joint is within ``window``, so the hand keeps the direction it was sent (a clip per joint bends its
        path); extra joints clipped per column. Exact where nothing binds. Example: internal."""
        xp, n = self.xp, self.n_arm
        held = self.inner.held(current, self._window)
        q = xp.view(self.engine.state.joint_q)[self.inner.q_rows][:, :n]
        lead, window = current[:, :n] - q, self._window[:n]
        over = xp.clip(xp.amax(abs(lead) / window, axis=-1)[:, None], 1.0, np.inf)   # (W, 1): > 1 where held
        arm = q + lead / over
        arm = xp.where((window < np.inf) & (over > 1.0) & (arm == arm), arm, current[:, :n])
        return xp.cat([arm, held[:, n:]], axis=-1)


def _joint_name(layout: Layout, name: str) -> str:
    """The layout joint behind a joint name or an actuator name. Example: internal."""
    if name in layout.joints:
        return name
    joint = layout.actuator(name).joint
    if joint is None or joint not in layout.joints:
        raise EnvError(f"{name!r} does not drive a joint", "list joints (or actuators of joints) in "
                       "EEActionSpec.joints", field="joints")
    return joint


def _writer_of(source: Any) -> EEWriter:
    writer = getattr(source, "writer", source)
    if not isinstance(writer, EEWriter):
        raise EnvError("this environment's actions are not task-space actions", "build it with "
                       "Task(actions=EEActionSpec(...)), or use panda_reach_ee()", field="actions")
    return writer


def jacobian(env_or_writer: Any) -> Any:
    """(W, 6, n) Jacobian of the end body's origin over the arm joints now: linear rows then angular rows, world frame.

    Example: ``J = jacobian(panda_reach_ee(2, "cpu"))``."""
    return _writer_of(env_or_writer).jacobian()


def ee_pose(env_or_writer: Any) -> tuple[Any, Any]:
    """(position (W, 3), quaternion xyzw (W, 4)) of the end body now. Example: ``pos, quat = ee_pose(env)``."""
    return _writer_of(env_or_writer).pose()


def panda_reach_ee(num_envs: int = 1, device: str | None = None, seed: int = 0, max_steps: int = 100,
                   **env_options: Any) -> Any:
    """PandaReach with end-effector actions: dx, dy, dz of the hand (2 cm per step at action 1). Torch arrays when
    PyTorch is installed; ``max_steps`` is the episode length.

    Example: ``env = panda_reach_ee(num_envs=64, device="cuda:0")``."""
    import dataclasses  # noqa: PLC0415

    from alloy.envs.tasks import panda_reach_task, panda_scene  # noqa: PLC0415
    from alloy.envs.vector import VectorEnv  # noqa: PLC0415

    spec = EEActionSpec(body="panda/hand", joints=[f"panda/actuator{k}" for k in range(1, 8)])
    task = dataclasses.replace(panda_reach_task(max_steps), actions=spec, name="PandaReachEE")
    scene = panda_scene(num_envs, seed, props=False)
    return VectorEnv(scene, task, worlds=num_envs, device=device, seed=seed, **env_options)


__all__ = ["ArmKinematics", "EEActionSpec", "EEWriter", "damped_least_squares", "ee_pose", "jacobian", "panda_reach_ee"]
