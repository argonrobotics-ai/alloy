# SPDX-License-Identifier: Apache-2.0
"""Read-only helpers on a recorded ``mujoco.MjModel`` (offline conversion side): names, state layout, task objects."""

from __future__ import annotations

import mujoco
import numpy as np

from alloy.core.errors import TraceError

_WIDTH = {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6), int(mujoco.mjtJoint.mjJNT_BALL): (4, 3)}


def joint_names(m: mujoco.MjModel) -> list[str]:
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, k) or f"joint{k}" for k in range(m.njnt)]


def actuator_names(m: mujoco.MjModel) -> tuple[str, ...]:
    return tuple(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator{i}" for i in range(m.nu))


def joint_width(m: mujoco.MjModel, k: int) -> tuple[int, int]:
    """(qpos width, dof width) of joint ``k``."""
    return _WIDTH.get(int(m.jnt_type[k]), (1, 1))


def element_columns(m: mujoco.MjModel, which: int) -> tuple[str, ...]:
    """Joint name of every qpos (``which`` = 0) or dof (``which`` = 1) element, in MuJoCo order."""
    out: list[str] = []
    for k, name in enumerate(joint_names(m)):
        out += [name] * joint_width(m, k)[which]
    return tuple(out)


def split_state0(m: mujoco.MjModel, state0: np.ndarray) -> dict[str, np.ndarray]:
    """Split an ``mjSTATE_INTEGRATION`` vector into its named parts (MuJoCo 3.16 order)."""
    sizes = [("time", 1), ("qpos", m.nq), ("qvel", m.nv), ("act", m.na), ("qacc_warmstart", m.nv),
             ("ctrl", m.nu), ("qfrc_applied", m.nv), ("xfrc_applied", 6 * m.nbody), ("eq_active", m.neq),
             ("mocap_pos", 3 * m.nmocap), ("mocap_quat", 4 * m.nmocap), ("userdata", m.nuserdata)]  # fmt: skip
    if sum(n for _, n in sizes) != len(state0):
        raise TraceError(f"state0 length {len(state0)} does not match the mjSTATE_INTEGRATION layout")
    out, off = {}, 0
    for name, n in sizes:
        out[name] = state0[off : off + n]
        off += n
    return out


def free_bodies(m: mujoco.MjModel) -> list[tuple[int, int, str]]:
    """``(body, qposadr, name)`` of every body with a free joint (the task objects)."""
    out = []
    for k in range(m.njnt):
        if int(m.jnt_type[k]) == int(mujoco.mjtJoint.mjJNT_FREE):
            b = int(m.jnt_bodyid[k])
            out.append((b, int(m.jnt_qposadr[k]), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"))
    return out


def body_radius(m: mujoco.MjModel, b: int) -> float:
    """Largest distance from body ``b``'s origin to any point of its collision geoms' bounding spheres."""
    geoms = range(int(m.body_geomadr[b]), int(m.body_geomadr[b] + m.body_geomnum[b]))
    in_pair = set(m.pair_geom1.tolist()) | set(m.pair_geom2.tolist())
    radii = [np.linalg.norm(m.geom_pos[g]) + m.geom_rbound[g] for g in geoms
             if m.geom_contype[g] or m.geom_conaffinity[g] or g in in_pair]  # fmt: skip
    return float(max(radii, default=0.0))


def recorded_multiccd(m: mujoco.MjModel) -> bool:
    """MuJoCo 3.16 runs multi-CCD unless ``mjDSBL_MULTICCD`` is set (Newton's default is off)."""
    return not int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
