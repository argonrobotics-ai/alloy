# SPDX-License-Identifier: Apache-2.0
"""Offline checks that a converted MJCF equals the recorded ``.mjb`` model (CPU MuJoCo, offline only).

Two checks, both run by the converter and never inside an Alloy simulation loop (README ground rule):

1. :func:`compare_models`: every converted field of the reloaded MJCF against the original compiled model; mesh
   geoms are compared as body-frame vertex positions, which is what collision sees, because MuJoCo may pick a
   different principal frame for a symmetric mesh when it recentres it again.
2. :func:`replay_cpu`: open-loop replay of the recorded inputs on a model, sampled like ``ref_qpos``. On the
   original model this reproduces the recording (the agent check of 2026-11-01 saw ~1e-16 over 300 steps); on the
   converted model it measures what the conversion changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mujoco
import numpy as np

from alloy.tasks.mjb_convert import RecordedTrace
from alloy.tasks.mjcf_tree import hull_vertex_count

log = logging.getLogger("alloy.tasks")

_FIELDS = (
    "body_parentid", "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia",
    "body_gravcomp", "body_mocapid", "body_weldid", "body_rootid", "body_subtreemass", "body_invweight0",
    "jnt_type", "jnt_bodyid", "jnt_qposadr", "jnt_dofadr", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited",
    "jnt_stiffness", "jnt_margin", "jnt_solref", "jnt_solimp", "jnt_actfrcrange", "jnt_actfrclimited",
    "jnt_actgravcomp", "qpos0", "qpos_spring", "dof_bodyid", "dof_parentid", "dof_armature", "dof_damping",
    "dof_frictionloss", "dof_solref", "dof_solimp", "dof_invweight0", "dof_M0",
    "geom_type", "geom_bodyid", "geom_contype", "geom_conaffinity", "geom_condim", "geom_priority",
    "geom_friction", "geom_solmix", "geom_solref", "geom_solimp", "geom_margin", "geom_gap", "geom_rbound",
    "geom_group", "pair_dim", "pair_geom1", "pair_geom2", "pair_signature", "pair_solref", "pair_solreffriction",
    "pair_solimp", "pair_margin", "pair_gap", "pair_friction", "exclude_signature",
    "eq_type", "eq_obj1id", "eq_obj2id", "eq_active0", "eq_solref", "eq_solimp", "eq_data",
    "actuator_trntype", "actuator_trnid", "actuator_dyntype", "actuator_gaintype", "actuator_biastype",
    "actuator_dynprm", "actuator_gainprm", "actuator_biasprm", "actuator_gear", "actuator_ctrllimited",
    "actuator_ctrlrange", "actuator_forcelimited", "actuator_forcerange", "actuator_acc0", "actuator_length0",
    "mesh_vertnum", "mesh_facenum",
)  # fmt: skip
_OPT_FIELDS = ("timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance", "integrator", "cone",
               "solver", "iterations", "ls_iterations", "noslip_iterations", "disableflags", "enableflags",
               "gravity", "o_margin", "ccd_iterations", "ccd_tolerance")  # fmt: skip


@dataclass(frozen=True)
class FieldDiff:
    """Max absolute difference of one field (``inf`` if the shapes differ)."""

    field: str
    max_abs: float
    ref_scale: float


def _diff(name: str, a: np.ndarray, b: np.ndarray) -> FieldDiff:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return FieldDiff(name, float("inf"), 0.0)
    if a.size == 0:
        return FieldDiff(name, 0.0, 0.0)
    return FieldDiff(name, float(np.max(np.abs(a - b))), float(np.max(np.abs(a))))


def _geom_body_frame_vertices(m: mujoco.MjModel, k: int) -> np.ndarray:
    mesh = int(m.geom_dataid[k])
    va, vn = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.geom_quat[k])
    return m.mesh_vert[va : va + vn].astype(np.float64) @ rot.reshape(3, 3).T + m.geom_pos[k]


def _geom_pose_diffs(ref: mujoco.MjModel, new: mujoco.MjModel) -> list[FieldDiff]:
    is_mesh = ref.geom_type == int(mujoco.mjtGeom.mjGEOM_MESH)
    prim = ~is_mesh
    out = [
        _diff("geom_pos[primitive]", ref.geom_pos[prim], new.geom_pos[prim]),
        _diff("geom_quat[primitive]", ref.geom_quat[prim], new.geom_quat[prim]),
        _diff("geom_size[primitive]", ref.geom_size[prim], new.geom_size[prim]),
    ]
    worst = 0.0
    for k in np.nonzero(is_mesh)[0]:
        d = _diff("v", _geom_body_frame_vertices(ref, int(k)), _geom_body_frame_vertices(new, int(k)))
        worst = max(worst, d.max_abs)
    out.append(FieldDiff("geom_mesh_vertices[body frame]", worst, 1.0))
    hull_ref = [hull_vertex_count(ref, i) for i in range(ref.nmesh)]
    hull_new = [hull_vertex_count(new, i) for i in range(new.nmesh)]
    out.append(_diff("mesh_hull_vertex_count", np.array(hull_ref), np.array(hull_new)))
    return out


def compare_models(ref: mujoco.MjModel, new: mujoco.MjModel) -> list[FieldDiff]:
    """Field-by-field differences between the original and the converted model (sizes must match)."""
    for size in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "nmesh", "npair", "nexclude", "neq"):
        if getattr(ref, size) != getattr(new, size):
            return [FieldDiff(size, float("inf"), float(getattr(ref, size)))]
    out = [_diff(f, getattr(ref, f), getattr(new, f)) for f in _FIELDS]
    out += [_diff(f"opt.{f}", getattr(ref.opt, f), getattr(new.opt, f)) for f in _OPT_FIELDS]
    return out + _geom_pose_diffs(ref, new)


def apply_inputs(m: mujoco.MjModel, d: mujoco.MjData, rec: RecordedTrace, i: int) -> None:
    """Write row ``i`` of every recorded input into ``d`` (and the gripper force range into ``m``)."""
    d.ctrl[:] = rec.inputs["ctrl"][i]
    d.qfrc_applied[:] = rec.inputs["qfrc"][i]
    d.xfrc_applied[:] = rec.inputs["xfrc"][i].reshape(-1, 6)
    m.jnt_actfrcrange[:] = rec.inputs["afr"][i].reshape(-1, 2)
    d.eq_active[:] = rec.inputs["eq"][i]


def replay_cpu(m: mujoco.MjModel, rec: RecordedTrace, steps: int) -> np.ndarray:
    """Open-loop replay from ``state0``; returns ``qpos`` after every ``ref_every`` steps, [steps // ref_every, nq].

    ``m.jnt_actfrcrange`` is overwritten by the trace (pass a copy if the model is reused)."""
    d = mujoco.MjData(m)
    mujoco.mj_setState(m, d, rec.state0, mujoco.mjtState.mjSTATE_INTEGRATION)
    samples = []
    for i in range(steps):
        apply_inputs(m, d, rec, i)
        mujoco.mj_step(m, d)
        if (i + 1) % rec.ref_every == 0:
            samples.append(d.qpos.copy())
    return np.array(samples).reshape(-1, m.nq)
