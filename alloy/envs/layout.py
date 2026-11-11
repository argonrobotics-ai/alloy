# SPDX-License-Identifier: Apache-2.0
"""Where each named thing lives in the engine's batched arrays, for every world (built once per environment).

Bodies, joints and particle objects keep their ``alloy.Sim`` names (``sim.names()``): objects by the name given to
the scene, robot parts as "<robot>/<body or joint>". Rows are indices into the flat engine arrays, world-major::

    lay = Layout.from_sim(sim)
    lay.bodies["cube"]            # (worlds,) rows into state.body_q / body_qd
    lay.joints["gripper/lift"].q  # (worlds, coords) rows into state.joint_q
    lay.robots["gripper"]         # the robot's joint names
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.envs.errors import EnvError

UNBOUNDED = 1e9      # |limit| at or above this means the joint has no limit
FREE_JOINT = 4       # newton.JointType.FREE


@dataclass(frozen=True)
class JointRows:
    """Rows of one joint in every world: coordinates (``q``), dofs (``qd``) and the position-target slots."""

    q: np.ndarray            # (W, coords) into state.joint_q
    qd: np.ndarray           # (W, dofs) into state.joint_qd, control.joint_target_qd, control.joint_f
    target_q: np.ndarray     # (W, coords) into control.joint_target_q
    lower: np.ndarray        # (dofs,) joint limits of world 0 (metres or radians)
    upper: np.ndarray
    actuated: bool           # has a position or velocity target (an MJCF <position>/<velocity> actuator)

    @property
    def width(self) -> int:
        """Coordinates per world (1 for a hinge or slide). Example: ``rows.width``."""
        return int(self.q.shape[1])


@dataclass(frozen=True)
class ActuatorRows:
    """A MuJoCo-native actuator (an MJCF ``<general>`` / ``<motor>`` read from ``control.mujoco.ctrl``) per world."""

    rows: np.ndarray         # (W,) into control.mujoco.ctrl
    joint: str | None        # the joint it drives ("robot/joint"), if any
    lower: float             # ctrlrange
    upper: float
    position: bool = False   # its control is the joint's position target (an affine position servo, gear 1)


@dataclass
class Layout:
    """Rows of every named body, joint and particle object, plus the world of each engine row."""

    worlds: int
    bodies: dict[str, np.ndarray] = field(default_factory=dict)
    joints: dict[str, JointRows] = field(default_factory=dict)
    particles: dict[str, np.ndarray] = field(default_factory=dict)
    robots: dict[str, list[str]] = field(default_factory=dict)
    objects: list[str] = field(default_factory=list)
    world_of: dict[str, np.ndarray] = field(default_factory=dict)   # engine array -> world id per row (-1 global)
    free: dict[str, np.ndarray] = field(default_factory=dict)       # free object -> (W,) first joint_q row (7 coords)
    free_frame: dict[str, np.ndarray] = field(default_factory=dict)  # free object -> its joint's parent rotation (xyzw)
    actuators: dict[str, ActuatorRows] = field(default_factory=dict)  # "robot/actuator" -> ctrl rows (MuJoCo-native)

    @classmethod
    def from_sim(cls, sim: Any) -> Layout:
        """The layout of a built ``alloy.Sim``. Example: ``Layout.from_sim(alloy.build(scene, worlds=8))``."""
        idx, model = sim.index, sim.engine.model
        lay = cls(int(sim.worlds))
        lay.world_of = _world_maps(model)                  # first: _add_actuators adds the "ctrl" rows' worlds
        for name, b in idx.bodies.items():
            lay.bodies[name] = idx.per_world(b, "body")
        for name, r in idx.particles.items():
            lay.particles[name] = idx.per_world(r.start, "particle")[:, None] + np.arange(len(r))[None, :]
        if model.joint_count:
            lay._add_joints(idx, model)
            lay._add_free(idx, model)
            lay._add_actuators(idx, model)
        lay.objects = [n for n in idx.bodies if "/" not in n and not any(b.startswith(n + "/") for b in idx.bodies)]
        return lay

    def _add_joints(self, idx: Any, model: Any) -> None:
        q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        tq = getattr(model, "joint_target_q_start", None)
        tq_start = tq.numpy() if tq is not None else q_start
        lower, upper = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        mode = getattr(model, "joint_target_mode", None)
        mode = mode.numpy() if mode is not None else model.joint_target_ke.numpy()
        for name, j in idx.joints.items():
            rows = idx.per_world(j, "joint")
            nq, nd = int(q_start[j + 1] - q_start[j]), int(qd_start[j + 1] - qd_start[j])
            d0 = int(qd_start[j])
            self.joints[name] = JointRows(
                q=q_start[rows][:, None] + np.arange(nq)[None, :], qd=qd_start[rows][:, None] + np.arange(nd)[None, :],
                target_q=tq_start[rows][:, None] + np.arange(nq)[None, :],
                lower=lower[d0:d0 + nd].astype(np.float64), upper=upper[d0:d0 + nd].astype(np.float64),
                actuated=bool(nd and np.any(mode[d0:d0 + nd] != 0)))
            if "/" in name:
                self.robots.setdefault(name.split("/", 1)[0], []).append(name)

    def _add_free(self, idx: Any, model: Any) -> None:
        """Free bodies: scene objects, and free bodies inside a robot file (e.g. "robot/box")."""
        q_start, x_p = model.joint_q_start.numpy(), model.joint_X_p.numpy()
        free = dict(getattr(idx, "free_joint", {}))
        jtype, child = model.joint_type.numpy(), model.joint_child.numpy()
        body_name = {b: n for n, b in idx.bodies.items() if "/" in n}
        for name, j in idx.joints.items():
            if "/" in name and int(jtype[j]) == FREE_JOINT and int(child[j]) in body_name:
                free.setdefault(body_name[int(child[j])], j)
        for name, j in free.items():
            self.free[name] = q_start[idx.per_world(j, "joint")].astype(np.int64)
            self.free_frame[name] = x_p[j, 3:7].astype(np.float64)

    def _add_actuators(self, idx: Any, model: Any) -> None:
        """Actuators that read ``control.mujoco.ctrl`` directly (Newton's CtrlSource.CTRL_DIRECT)."""
        mj = getattr(model, "mujoco", None)
        src = getattr(mj, "ctrl_source", None)
        if src is None or not len(src):
            return
        src, world = src.numpy(), mj.actuator_world.numpy()
        labels, dof = [str(x) for x in mj.actuator_label], mj.actuator_trnid.numpy()[:, 0]
        lo_hi, qd_start = mj.actuator_ctrlrange.numpy(), model.joint_qd_start.numpy()
        per_world = [np.flatnonzero(world == w) for w in range(self.worlds)]
        if len({len(p) for p in per_world}) != 1:
            return
        rows = np.stack(per_world)                                     # (W, actuators per world)
        names = {j: n for n, j in idx.joints.items() if "/" in n}
        servo = _position_servos(mj, len(src))
        for k, a in enumerate(rows[0]):
            if int(src[a]) != 1:                                       # JOINT_TARGET: driven by joint targets
                continue
            j = int(np.searchsorted(qd_start, dof[a], side="right") - 1) if dof[a] >= 0 else -1
            joint = names.get(j)
            robot = joint.split("/", 1)[0] if joint else next(iter(self.robots), "robot")
            self.actuators[f"{robot}/{labels[a] or f'actuator{k}'}"] = ActuatorRows(
                rows=rows[:, k].astype(np.int64), joint=joint, lower=float(lo_hi[a, 0]), upper=float(lo_hi[a, 1]),
                position=bool(servo[a]))
        self.world_of["ctrl"] = world.astype(np.int32)

    def actuator(self, name: str) -> ActuatorRows:
        """A MuJoCo-native actuator by "robot/actuator" or by the joint it drives. Example: ``lay.actuator("a/a1")``."""
        if name in self.actuators:
            return self.actuators[name]
        for a in self.actuators.values():
            if a.joint is not None and name in (a.joint, a.joint.split("/", 1)[1]):
                return a
        raise self.unknown(name, "actuator")

    # --- lookups ---------------------------------------------------------------------------------------------------

    def kind(self, name: str) -> str:
        """"body", "joint", "particles" or "robot" for a name. Example: ``lay.kind("cube") == "body"``."""
        if name in self.robots:
            return "robot"
        for kind, table in (("body", self.bodies), ("joint", self.joints), ("particles", self.particles)):
            if name in table:
                return kind
        raise self.unknown(name)

    def body(self, name: str) -> np.ndarray:
        """(W,) rows of a body. Example: ``lay.body("cube")``."""
        if name in self.bodies:
            return self.bodies[name]
        raise self.unknown(name, "body")

    def joint(self, name: str) -> JointRows:
        """Rows of a joint. Example: ``lay.joint("gripper/lift").q``."""
        if name in self.joints:
            return self.joints[name]
        raise self.unknown(name, "joint")

    def particle(self, name: str) -> np.ndarray:
        """(W, n) rows of a cloth / liquid / soft object. Example: ``lay.particle("water")``."""
        if name in self.particles:
            return self.particles[name]
        raise self.unknown(name, "particle object")

    def actuated_joints(self) -> list[str]:
        """Full names ("robot/joint") of every actuated joint, in scene order. Example: ``lay.actuated_joints()``."""
        return [n for n, r in self.joints.items() if "/" in n and r.actuated]

    def unknown(self, name: str, kind: str = "body, joint, robot or particle object") -> EnvError:
        """The error for a name that is not in the scene, listing the names that are."""
        listing = {"bodies": list(self.bodies), "joints": [j for j in self.joints if "/" in j],
                   "robots": list(self.robots), "particles": list(self.particles), "actuators": list(self.actuators)}
        text = "; ".join(f"{k}: {', '.join(v[:12]) or '-'}" for k, v in listing.items())
        return EnvError(f"no {kind} named {name!r}", f"use a name from the scene ({text})", field=name)


def _world_maps(model: Any) -> dict[str, np.ndarray]:
    """World id of every row of body_q, joint_q, joint_qd and particle_q (-1: shared by all worlds)."""
    out = {"body": _ids(model, "body_world", model.body_count),
           "particle": _ids(model, "particle_world", model.particle_count)}
    if not model.joint_count:
        out["q"], out["qd"], out["joint"] = (np.zeros(0, np.int32) for _ in range(3))
        return out
    jw = _ids(model, "joint_world", model.joint_count)
    out["joint"] = jw
    for key, start in (("q", model.joint_q_start.numpy()), ("qd", model.joint_qd_start.numpy())):
        out[key] = np.repeat(jw, np.diff(start)).astype(np.int32)
    return out


def _ids(model: Any, attr: str, count: int) -> np.ndarray:
    arr = getattr(model, attr, None)
    if arr is None or not count:
        return np.zeros(count, np.int32)
    return arr.numpy().astype(np.int32)


__all__ = ["ActuatorRows", "JointRows", "Layout", "UNBOUNDED"]


def _position_servos(mj: Any, count: int) -> np.ndarray:
    """(actuators,) True where the control is a joint position target: an affine bias with bias[1] = -gain[0] > 0 and
    gear 1 (MJCF ``<position>``, and the position-like ``<general>`` of MuJoCo Menagerie files), not a ``<motor>``."""
    arrays = [getattr(mj, k, None) for k in ("actuator_biastype", "actuator_gainprm", "actuator_biasprm",
                                             "actuator_gear")]
    if any(a is None or len(a) != count for a in arrays):
        return np.zeros(count, bool)
    bias_type, gain, bias, gear = (a.numpy() for a in arrays)
    kp = gain[:, 0]
    return ((bias_type == 1) & (kp > 0) & np.isclose(bias[:, 1], -kp, rtol=1e-6) & np.isclose(bias[:, 0], 0.0)
            & np.isclose(gear[:, 0], 1.0))
