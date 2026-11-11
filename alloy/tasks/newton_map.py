# SPDX-License-Identifier: Apache-2.0
"""Bind a name-keyed trace to a Newton model: joint/body lookup by name and MuJoCo -> Newton coordinates.

Newton labels are ``/``-joined paths ending in the MJCF name, so a joint or body is found by its last path segment
(world 0 or global entities only; a duplicate raises). Conversions follow
newton@ac7f7830:newton/_src/solvers/mujoco/kernels.py:845-922 (``convert_mj_coords_to_warp_kernel``): FREE
``q = X_p⁻¹·X_world·X_c`` with xyzw quaternions, ``qd`` = child-COM linear and angular velocity in the parent frame;
hinge/slide ``q = qpos − dof_ref``. No MuJoCo model is needed: the trace's columns carry the names.
"""

from __future__ import annotations

from dataclasses import dataclass

from alloy._vendor import newton
import numpy as np

from alloy.core.errors import TraceError


def _by_name(labels: list[str], world: np.ndarray, kind: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, label in enumerate(labels):
        if int(world[i]) > 0:
            continue
        name = label.rsplit("/", 1)[-1]
        if name in out:
            raise TraceError(f"{kind} name {name!r} is not unique in the Newton model")
        out[name] = i
    return out


def joint_index(model: newton.Model) -> dict[str, int]:
    return _by_name(model.joint_label, model.joint_world.numpy(), "joint")


def body_index(model: newton.Model) -> dict[str, int]:
    return _by_name(model.body_label, model.body_world.numpy(), "body")


def runs(columns: tuple[str, ...]) -> list[tuple[str, int, int]]:
    """Consecutive runs ``(name, start, width)`` of equal names in an element-column list."""
    out: list[tuple[str, int, int]] = []
    for i, name in enumerate(columns):
        if out and out[-1][0] == name:
            out[-1] = (name, out[-1][1], out[-1][2] + 1)
        else:
            out.append((name, i, 1))
    return out


def lookup(index: dict[str, int], names: tuple[str, ...], kind: str) -> np.ndarray:
    missing = [n for n in names if n not in index]
    if missing:
        raise TraceError(f"{kind}s missing from the Newton model: {missing[:5]}")
    return np.array([index[n] for n in names], dtype=np.int64)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])  # fmt: skip


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]])


@dataclass(frozen=True)
class NewtonJointData:
    """Host copies of the Newton model arrays the conversion needs."""

    jtype: np.ndarray
    q_start: np.ndarray
    qd_start: np.ndarray
    x_p: np.ndarray
    x_c: np.ndarray
    child: np.ndarray
    body_com: np.ndarray
    dof_ref: np.ndarray | None

    @classmethod
    def of(cls, model: newton.Model) -> NewtonJointData:
        ref = getattr(getattr(model, "mujoco", None), "dof_ref", None)
        return cls(model.joint_type.numpy(), model.joint_q_start.numpy(), model.joint_qd_start.numpy(),
                   model.joint_X_p.numpy().astype(np.float64), model.joint_X_c.numpy().astype(np.float64),
                   model.joint_child.numpy(), model.body_com.numpy().astype(np.float64),
                   None if ref is None else ref.numpy())  # fmt: skip


def free_to_newton(nj: NewtonJointData, j: int, qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MuJoCo free-joint qpos[7]/qvel[6] -> Newton joint_q[7]/joint_qd[6] for Newton joint ``j``."""
    p_w, q_w = qpos[:3], wxyz_to_xyzw(qpos[3:7])
    p_p, q_p = nj.x_p[j, :3], nj.x_p[j, 3:]
    p_c, q_c = nj.x_c[j, :3], nj.x_c[j, 3:]
    q_pi = _conj(q_p)
    p = rotate(q_pi, p_w - p_p) + rotate(quat_mul(q_pi, q_w), p_c)
    q = quat_mul(quat_mul(q_pi, q_w), q_c)
    w_world = rotate(q_w, qvel[3:6])
    v_com = qvel[:3] + np.cross(w_world, rotate(q_w, nj.body_com[nj.child[j]]))
    return np.concatenate([p, q]), np.concatenate([rotate(q_pi, v_com), rotate(q_pi, w_world)])


def initial_to_newton(model: newton.Model, columns: dict[str, tuple[str, ...]], qpos: np.ndarray,
                      qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Name-keyed MuJoCo ``qpos``/``qvel`` -> Newton ``joint_q``/``joint_qd`` (float64; untouched joints keep
    the model's defaults)."""
    nj, index = NewtonJointData.of(model), joint_index(model)
    joint_q = model.joint_q.numpy().astype(np.float64)
    joint_qd = np.zeros(model.joint_dof_count)
    dof_runs = {name: (start, width) for name, start, width in runs(columns["qvel"])}
    for name, qa, nq in runs(columns["qpos"]):
        j = int(lookup(index, (name,), "joint")[0])
        da, nd = dof_runs[name]
        wq, wd = int(nj.q_start[j]), int(nj.qd_start[j])
        if int(nj.jtype[j]) == int(newton.JointType.FREE):
            joint_q[wq : wq + 7], joint_qd[wd : wd + 6] = free_to_newton(nj, j, qpos[qa : qa + 7], qvel[da : da + 6])
        elif nq == 1 and nd == 1:
            joint_q[wq] = qpos[qa] - (0.0 if nj.dof_ref is None else float(nj.dof_ref[wd]))
            joint_qd[wd] = qvel[da]
        else:
            raise TraceError(f"joint {name!r}: only free, hinge and slide joints are converted")
    return joint_q, joint_qd
