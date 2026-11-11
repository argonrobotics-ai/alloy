# SPDX-License-Identifier: Apache-2.0
"""Host-side topology analysis for the bench monitors: body roles and template -> global ids.

All worlds of a pack share topology (SPEC §1.2), so a template index ``i`` (the ``i``-th body of a world, in
ascending global order) names the same part in every world. Computed once at construction from Model arrays.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from alloy._vendor import newton

from alloy.bully.kernels import world_csr
from alloy.core.errors import BullyError

log = logging.getLogger("alloy.bully")

KIND_FREE, KIND_ROBOT, KIND_NON_TARGET = 1, 2, 4
_FREE, _DISTANCE = int(newton.JointType.FREE), int(newton.JointType.DISTANCE)


@dataclass(frozen=True)
class BodyRoles:
    """Per-body arrays (global body index) used by the monitor kernels."""

    kind: np.ndarray  # int32 bit set: KIND_FREE | KIND_ROBOT | KIND_NON_TARGET
    radius: np.ndarray  # float32, largest shape bounding radius [m]
    thin: np.ndarray  # float32, part thickness [m] (inf if unknown)
    world_start: np.ndarray  # int32 [W+1] CSR of passive (non-robot) bodies: the energy ledger's subsystem
    world_index: np.ndarray  # int32


def template_to_global(item_world: np.ndarray, world_count: int, template: Sequence[int]) -> np.ndarray:
    """Global ids [W, len(template)] of template items; raises IndexError when a world has too few items."""
    start, index = world_csr(item_world, world_count)
    out = np.empty((world_count, len(template)), dtype=np.int64)
    for w in range(world_count):
        items = index[start[w]: start[w + 1]]
        for k, i in enumerate(template):
            if not 0 <= i < items.size:
                raise BullyError(f"template index {i} outside world {w} ({items.size} items)")
            out[w, k] = items[i]
    return out


def _robot_bodies_from_joints(model: newton.Model) -> np.ndarray:
    """Bodies on the non-free joints of articulations that have a driven dof (``joint_target_ke`` or ``_kd`` > 0).
    A passive jointed object (door, pendulum) has no drive and stays in the passive subsystem. Targets whose drives
    are solver-native (MuJoCo actuators) name their robot bodies explicitly (``BullyTarget.robot_bodies``)."""
    jt, child = model.joint_type.numpy(), model.joint_child.numpy()
    art = model.joint_articulation.numpy() if model.joint_articulation is not None else np.full(jt.size, -1)
    ke, kd = model.joint_target_ke.numpy(), model.joint_target_kd.numpy()
    qd0 = model.joint_qd_start.numpy()
    qd0 = qd0 if qd0.size == jt.size + 1 else np.append(qd0[: jt.size], model.joint_dof_count)
    driven = {int(art[j]) for j in range(jt.size)
              if art[j] >= 0 and np.any((ke[qd0[j]:qd0[j + 1]] > 0) | (kd[qd0[j]:qd0[j + 1]] > 0))}
    robot = np.zeros(model.body_count, dtype=bool)
    for j in range(jt.size):
        if int(art[j]) in driven and int(jt[j]) not in (_FREE, _DISTANCE) and child[j] >= 0:
            robot[child[j]] = True
    return robot


def free_bodies(model: newton.Model) -> np.ndarray:
    """Bodies with no joint, or a FREE / DISTANCE joint (they move only through contact and gravity)."""
    free = np.ones(model.body_count, dtype=bool)
    if model.joint_count:
        jt, child = model.joint_type.numpy(), model.joint_child.numpy()
        for j in range(jt.size):
            if child[j] >= 0 and int(jt[j]) not in (_FREE, _DISTANCE):
                free[child[j]] = False
    mass = model.body_mass.numpy()
    return free & (mass > 0.0)


def body_roles(
    model: newton.Model,
    non_target: Sequence[int] = (),
    robot: Sequence[int] | None = None,
    thin_parts: Mapping[int, float] | None = None,
) -> BodyRoles:
    """Roles from the joint graph; ``robot`` (template ids) overrides the inferred robot set."""
    world = model.body_world.numpy() if model.body_count else np.zeros(0, np.int32)
    w_count = model.world_count
    kind = np.zeros(model.body_count, dtype=np.int32)
    if robot is None:
        robot_mask = _robot_bodies_from_joints(model)
    else:
        robot_mask = np.zeros(model.body_count, dtype=bool)
        robot_mask[template_to_global(world, w_count, robot).ravel()] = True
    kind[free_bodies(model) & ~robot_mask] |= KIND_FREE
    kind[robot_mask] |= KIND_ROBOT
    if non_target:
        kind[template_to_global(world, w_count, non_target).ravel()] |= KIND_NON_TARGET
    thin = np.full(model.body_count, np.inf, dtype=np.float32)
    for i, t in (thin_parts or {}).items():
        thin[template_to_global(world, w_count, (i,)).ravel()] = t
    passive_world = np.where(robot_mask, -1, world)
    start, index = world_csr(passive_world, w_count)
    return BodyRoles(kind, _body_radius(model), thin, start, index)


def _body_radius(model: newton.Model) -> np.ndarray:
    """Largest distance from the body COM to a point of its collision shapes: shape offset + bounding radius."""
    radius = np.zeros(model.body_count, dtype=np.float32)
    if model.shape_count and model.body_count:
        sb, sr = model.shape_body.numpy(), model.shape_collision_radius.numpy()
        offset = model.shape_transform.numpy()[:, :3]
        com = model.body_com.numpy()
        for s in np.nonzero(sb >= 0)[0]:
            b = sb[s]
            radius[b] = max(radius[b], float(np.linalg.norm(offset[s] - com[b]) + sr[s]))
    return radius
