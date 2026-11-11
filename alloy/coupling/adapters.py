# SPDX-License-Identifier: Apache-2.0
"""State adapters for Newton solvers (SPEC §2.2 "solver internals", §3.6): arrays outside ``State`` that the next
step reads, so ``WorldSnapshot``, rollback and save/load restore bitwise.

Sources (newton@ac7f7830, mujoco_warp@a9daa05f; the cvel / cdof_dot item at mujoco_warp@86e2fae0):
- SolverMuJoCo: ``mjw_data.qacc_warmstart`` (read by the next solve, mujoco_warp/_src/solver.py:1688-1701, written at
  forward.py:472), ``qacc``, ``act``, ``act_dot``, ``time``, ``history``, ``eq_active``, mocap; ``cvel`` and
  ``cdof_dot``, which the weld / connect rows read in constraint.make_constraint through support.jac_dot_dof
  (support.py:611-650) before smooth.com_vel rewrites them in fwd_velocity (forward.py:1546); with sleeping on,
  every non-empty ``(nworld, ...)`` array, because sleeping trees keep their cached kinematics
  (mujoco_warp/_src/sleep.py; newton solver_mujoco.py:5201-5265). All are world-major. When Newton does not push
  ``joint_q``/``joint_qd`` into MuJoCo every step (``update_data_interval != 1``: alloy's ``SyncedMuJoCo`` pushes
  only changed worlds), ``qpos``/``qvel`` are MuJoCo's own state, and ``SyncedMuJoCo._last_q``/``_last_qd`` (per
  coordinate / dof, mapped to worlds) decide the push; without them a restored or kept world was re-pushed through
  the float32 coordinate round trip and left its reference trajectory at rounding level (SCR-SC4-7). The coordinate
  carry's residual and its exchange save (``alloy.scene.coordinate_carry``, per carried coordinate, mapped to worlds;
  SCR-SC10-3) are solver state too: a restore without them replays from the rolled-back attempt's residual.
- SolverVBD: ``body_q_prev`` (rigid AVBD velocity history, vbd/solver_vbd.py:1316, 5095-5115), joint warm starts
  (:1389-1428), the coupling snapshots ``_coupling_*_q_prev_snapshot`` (:1123, 1317, used by coupled restarts
  :1581-1622), ``particle_q_prev`` and the DAT buffers (:1118, 1196-1198, 3118-3122), contact history when enabled
  (:1942-1949, whole-array: slot-indexed).
- SolverImplicitMPM: ``_last_step_data`` warm starts (solver_implicit_mpm.py:748-772, 1681); grid fields have no
  world axis (whole-array).
- SolverXPBD: ``_contact_impulse`` (solver_xpbd.py:1251, force export only).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import warp as wp

from alloy.core.registry import register_state_adapter
from alloy.core.worldops import item_world_for

log = logging.getLogger("alloy.coupling")

MUJOCO_CORE = ("qacc_warmstart", "qacc", "act", "act_dot", "time", "history", "eq_active", "mocap_pos",
               "mocap_quat", "ctrl", "qfrc_applied", "xfrc_applied", "userdata",
               # MuJoCo Warp's weld / connect rows build their J-dot v aref shift in make_constraint (fwd_position,
               # support.jac_dot_dof) from cvel and cdof_dot, which smooth.com_vel rewrites only later, in
               # fwd_velocity: the previous step's values are state (dc_show's checkpoint resume, WP3; cdof_dot: WP4)
               "cvel", "cdof_dot")
MUJOCO_OWN_COORDINATES = ("qpos", "qvel")
SYNCED_LAST = (("_last_q", "_coord_world"), ("_last_qd", "_dof_world"))     # alloy.scene.mujoco_sync.SyncedMuJoCo
VBD_ARRAYS = (
    "body_q_prev", "_coupling_body_q_prev_snapshot", "_coupling_particle_q_prev_snapshot", "particle_q_prev",
    "joint_lambda_lin", "joint_lambda_ang", "joint_C0_lin", "joint_C0_ang", "joint_penalty_k", "joint_angle_prev",
    "joint_drive_lambda", "joint_limit_lambda", "joint_sigma_prev", "joint_kappa_prev", "joint_dkappa_prev",
    "joint_dahl_eps_max", "joint_dahl_tau", "particle_displacements", "pos_prev_collision_detection",
    "truncation_ts", "body_q_prev_collision_detection", "body_truncation_ts", "_prev_contact_lambda",
    "_prev_contact_penalty_k", "_prev_contact_normal", "_prev_contact_frame",
    "_accel_v_cur", "_accel_a_last", "_accel_hist", "_accel_h_cur",   # SolverVBDAlloy's adaptive start (vbd/accel.py)
    "_bend_theta_s", "_bend_rest_angle",                 # hysteretic cloth hinges (vbd/hysteresis.py, gap #4)
    "_stick_shape", "_stick_slip",                       # stick friction anchors (vbd/stick.py, SCR-WP11-STICK)
    "_plastic_tet_fp_inv", "_plastic_tet_ebar", "_plastic_tet_poses",   # J2 tets (plastic/tets.py, SCR-WP7-15)
    "_plastic_rod_eps_p", "_plastic_rod_gam_p", "_plastic_rod_ebar", "_rest_rel",   # J2 rods (rod/plastic.py)
    "_rigid_pose_rebaseline_mask", "_contact_history_reset_mask",   # per world (+ a global slot): VBD_WORLD
)
XPBD_ARRAYS = ("_contact_impulse",)
VBD_BODY = ("body_q_prev", "_coupling_body_q_prev_snapshot", "body_q_prev_collision_detection", "body_truncation_ts")
VBD_PARTICLE = ("_coupling_particle_q_prev_snapshot", "particle_q_prev", "particle_displacements",
                "pos_prev_collision_detection", "truncation_ts", "_accel_v_cur", "_accel_a_last", "_accel_hist",
                "_accel_h_cur", "_stick_shape", "_stick_slip")
VBD_JOINT = ("joint_lambda_lin", "joint_lambda_ang", "joint_C0_lin", "joint_C0_ang",
             "joint_sigma_prev", "joint_kappa_prev", "joint_dkappa_prev", "joint_dahl_eps_max", "joint_dahl_tau",
             "_plastic_rod_eps_p", "_plastic_rod_gam_p", "_plastic_rod_ebar", "_rest_rel")   # rows: joints
VBD_DOF = ("joint_drive_lambda", "joint_limit_lambda", "joint_angle_prev")   # angle_prev: per DOF, as Newton's
#                                                                          _init_joint_rest_angle sizes it
VBD_SLOT = ("joint_penalty_k",)        # per joint-constraint slot: joint j owns joint_constraint_dim[j] slots
VBD_TET = ("_plastic_tet_fp_inv", "_plastic_tet_ebar", "_plastic_tet_poses")   # per tet (J2 tets)
VBD_EDGE = ("_bend_theta_s", "_bend_rest_angle")      # per hinge edge (hysteretic cloth hinges)
# per world plus a last slot for entities without one (solver_vbd.py: history_mask_size = world_count + 1). The pose
# rebaseline flag is set for every world at construction and on reset, and cleared by the first rigid step: a world
# restored to a period start that came straight after a reset, or moved into a pack that has not stepped yet (#10b
# P1's fine pack), must take its own flag, or its first step rebaselines body_q_prev differently (WP4 2026-11-09)
VBD_WORLD = ("_rigid_pose_rebaseline_mask", "_contact_history_reset_mask")


def _present(obj: Any, names: Iterable[str], prefix: str) -> dict[str, wp.array]:
    """Non-empty Warp arrays among ``names`` on ``obj``, keyed ``prefix.name``."""
    out: dict[str, wp.array] = {}
    for name in names:
        arr = getattr(obj, name, None)
        if isinstance(arr, wp.array) and arr.size > 0:
            out[f"{prefix}.{name}"] = arr
    return out


def _keeps_own_coordinates(solver: Any) -> bool:
    """MuJoCo Warp's qpos/qvel are state when Newton does not push them every step (``update_data_interval != 1``,
    e.g. alloy.scene.mujoco_sync.SyncedMuJoCo, which pushes only worlds whose Newton coordinates changed)."""
    return int(getattr(solver, "update_data_interval", 1)) != 1


def _synced_last(solver: Any) -> dict[str, tuple[wp.array, wp.array]]:
    """SyncedMuJoCo's last-written Newton coordinates and their item->world maps (SCR-SC4-7): restored with the world,
    so a restored or kept world is not re-pushed through the float32 coordinate round trip."""
    out = {}
    for name, world in SYNCED_LAST:
        arr, item_world = getattr(solver, name, None), getattr(solver, world, None)
        if isinstance(arr, wp.array) and arr.size > 0 and isinstance(item_world, wp.array) \
                and item_world.shape[0] == arr.shape[0]:
            out[f"sync.{name.lstrip('_')}"] = (arr, item_world)
    return out


def mujoco_state_arrays(solver: Any) -> Mapping[str, wp.array]:
    data = getattr(solver, "mjw_data", None)
    if data is None:                      # CPU-MuJoCo backend keeps its state in mj_data (not snapshot-able here)
        log.warning("SolverMuJoCo without mjw_data (use_mujoco_cpu): no state adapter arrays")
        return {}
    if not getattr(solver, "enable_sleeping", False):
        out = _present(data, MUJOCO_CORE, "mjw")
        if _keeps_own_coordinates(solver):
            out.update(_present(data, MUJOCO_OWN_COORDINATES, "mjw"))
    else:
        nworld = int(getattr(data, "nworld", 0))
        names = [n for n, v in vars(data).items() if isinstance(v, wp.array) and v.ndim >= 1 and v.shape[0] == nworld]
        out = _present(data, sorted(names), "mjw")
    out.update({key: arr for key, (arr, _) in _synced_last(solver).items()})
    from alloy.scene import coordinate_carry  # noqa: PLC0415  (WP3's carry residual: solver state, SCR-SC10-3)

    out.update(coordinate_carry.state_arrays(solver))
    return out


def mujoco_world_maps(solver: Any) -> Mapping[str, wp.array]:
    """Item->world maps for the per-coordinate arrays of ``mujoco_state_arrays`` (the (nworld, ...) ones need none)."""
    from alloy.scene import coordinate_carry  # noqa: PLC0415

    return {**{key: item_world for key, (_, item_world) in _synced_last(solver).items()},
            **coordinate_carry.world_maps(solver)}


def vbd_state_arrays(solver: Any) -> Mapping[str, wp.array]:
    return _present(solver, VBD_ARRAYS, "vbd")


def vbd_world_maps(solver: Any) -> Mapping[str, wp.array]:
    """Item->world maps (entry-view ids) for VBD's per-body/particle/joint/dof arrays; contact history stays whole."""
    from alloy._vendor import newton

    freq = newton.Model.AttributeFrequency
    view = solver.model
    out: dict[str, wp.array] = {}
    for names, frequency, count in ((VBD_BODY, freq.BODY, view.body_count),
                                    (VBD_PARTICLE, freq.PARTICLE, view.particle_count),
                                    (VBD_JOINT, freq.JOINT, view.joint_count),
                                    (VBD_DOF, freq.JOINT_DOF, view.joint_dof_count)):
        present = [k for k, a in _present(solver, names, "vbd").items() if a.shape[0] == count]
        if not present:
            continue
        world = item_world_for(view, frequency)
        if world is None:
            continue
        for key in present:
            out[key] = world
    out.update(_slot_maps(solver, view))
    worlds = int(view.world_count)
    flags = [k for k, a in _present(solver, VBD_WORLD, "vbd").items() if a.shape[0] == worlds + 1]
    if flags:
        world = wp.array(np.append(np.arange(worlds), -1).astype(np.int32), dtype=wp.int32, device=view.device)
        out.update({key: world for key in flags})
    plastic = getattr(solver, "plastic", None)              # J2 tets: each tet's world (plastic/tets.py)
    if plastic is not None and hasattr(plastic, "tet_world"):
        out.update({key: plastic.tet_world() for key in _present(solver, VBD_TET, "vbd")})
    hyst = getattr(solver, "hysteresis", None)              # hysteretic hinges: each edge's world (vbd/hysteresis.py)
    if hyst is not None:
        out.update({key: hyst.edge_world for key in _present(solver, VBD_EDGE, "vbd")})
    return out


def _slot_maps(solver: Any, view: Any) -> dict[str, wp.array]:
    """Item->world maps for VBD's per-constraint-slot arrays (``VBD_SLOT``): slots run in joint order, joint j owning
    ``joint_constraint_dim[j]`` of them (SolverVBD's ``joint_constraint_start``)."""
    from alloy._vendor import newton

    present = _present(solver, VBD_SLOT, "vbd")
    dim = getattr(solver, "joint_constraint_dim", None)
    joint = newton.Model.AttributeFrequency.JOINT
    joint_world = item_world_for(view, joint) if present and dim is not None else None
    if joint_world is None or dim.shape[0] != joint_world.shape[0]:
        return {}
    slots = np.repeat(joint_world.numpy(), dim.numpy())
    world = wp.array(slots.astype(np.int32), dtype=wp.int32, device=joint_world.device)
    return {key: world for key, arr in present.items() if arr.shape[0] == len(slots)}


def _field_arrays(obj: Any, prefix: str, depth: int = 0) -> dict[str, wp.array]:
    """Arrays inside a warp.fem field / tuple / plain object (``dof_values`` for fields), two levels deep."""
    if isinstance(obj, wp.array):
        return {prefix: obj} if obj.size > 0 else {}
    if depth > 2 or obj is None:
        return {}
    if isinstance(obj, (tuple, list)):
        out: dict[str, wp.array] = {}
        for i, item in enumerate(obj):
            out.update(_field_arrays(item, f"{prefix}.{i}", depth + 1))
        return out
    values = getattr(obj, "dof_values", None)
    if isinstance(values, wp.array):
        return {f"{prefix}.dof_values": values} if values.size > 0 else {}
    out = {}
    for name, value in vars(obj).items() if hasattr(obj, "__dict__") else ():
        if not name.startswith("__") and name != "gimp_warmstart_state":
            out.update(_field_arrays(value, f"{prefix}.{name}", depth + 1))
    return out


def mpm_state_arrays(solver: Any) -> Mapping[str, wp.array]:
    data = getattr(solver, "_last_step_data", None)
    return _field_arrays(data, "mpm.last_step") if data is not None else {}


def xpbd_state_arrays(solver: Any) -> Mapping[str, wp.array]:
    return _present(solver, XPBD_ARRAYS, "xpbd")


def register_builtin_adapters() -> None:
    """Register the adapters above for Newton's solver classes (idempotent)."""
    from alloy._vendor.newton.solvers import SolverImplicitMPM, SolverMuJoCo, SolverVBD, SolverXPBD

    register_state_adapter(SolverMuJoCo, mujoco_state_arrays, world_map=mujoco_world_maps)
    register_state_adapter(SolverVBD, vbd_state_arrays, world_map=vbd_world_maps)
    register_state_adapter(SolverImplicitMPM, mpm_state_arrays)
    register_state_adapter(SolverXPBD, xpbd_state_arrays)
