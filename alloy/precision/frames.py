# SPDX-License-Identifier: Apache-2.0
"""Frames: worlds at the origin (F1), origin checks and world rebasing (F4) (SPEC §3.7, §4.1; research §4.2).

Newton bakes ``replicate(spacing=...)`` and ``add_world(xform=...)`` offsets into ``body_q``, free-joint
``joint_q`` and static ``shape_transform`` and keeps no offset on the Model
(newton@ac7f7830:newton/_src/sim/builder.py:3576, 3686-3687, 3884-3992), so spacing is detected from geometry: a
world whose bodies and static shapes are all world 0's moved by one common nonzero vector is a spaced copy.
Rebasing subtracts a float64 shift from a world's positions between steps and adds it to ``world_origin``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import PrecisionError
from alloy.precision.types import FramePolicy
from alloy.precision.world_groups import WorldGroups, equal_size_blocks, group_by_world, world_reduce

log = logging.getLogger("alloy.precision")

SPACING_MIN_M = 1e-5  # a common shift below this is rounding, not spacing
SPACING_RTOL = 1e-6  # residual allowed around the common shift, relative to its size (plus SPACING_ABS_M)
SPACING_ABS_M = 1e-6
SINGLE_ITEM_SPACING_M = 0.1  # one-item worlds: a shift this large is spacing (randomization stays below)
REBASE_GRID_M = 2.0**-7  # shifts on this grid keep the float32 subtraction exact for |p| < 2**17 m
JOINT_FREE = wp.constant(int(newton.JointType.FREE))
JOINT_DISTANCE = wp.constant(int(newton.JointType.DISTANCE))


@dataclass(frozen=True)
class WorldPoints:
    """Reference points of all worlds (bodies, per-world static shapes, particles), grouped by world. World ``w``'s
    points are ``points[groups.order[groups.begin[w]:groups.end[w]]]``: its bodies, then static shapes, then
    particles, each in index order."""

    points: np.ndarray  # [n, 3] float64, all sources concatenated
    groups: WorldGroups

    @property
    def world_count(self) -> int:
        return self.groups.world_count

    def of(self, w: int) -> np.ndarray:
        return self.points[self.groups.order[self.groups.begin[w]:self.groups.end[w]]]


def world_points(model: newton.Model) -> WorldPoints:
    """Positions per world from the initial Model (bodies from ``body_q``, static shapes, particles)."""
    W = int(model.world_count)
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    if model.body_count:
        parts.append((model.body_q.numpy()[:, :3], model.body_world.numpy()))
    if model.shape_count:
        static = model.shape_body.numpy() < 0
        parts.append((model.shape_transform.numpy()[static, :3], model.shape_world.numpy()[static]))
    if model.particle_count:
        parts.append((model.particle_q.numpy(), model.particle_world.numpy()))
    if not parts:
        return WorldPoints(np.zeros((0, 3)), group_by_world(np.zeros(0, np.int64), W))
    points = np.concatenate([q for q, _ in parts]).astype(np.float64)
    return WorldPoints(points, group_by_world(np.concatenate([w for _, w in parts]), W))


def spaced_worlds(pts: WorldPoints) -> list[tuple[int, np.ndarray]]:
    """Worlds that are world 0 moved by one common vector (spacing), with that vector, in world order."""
    ref = pts.of(0) if pts.world_count else np.zeros((0, 3))
    out: list[tuple[int, np.ndarray]] = []
    for ids, idx in equal_size_blocks(pts.groups, len(ref)):
        d = pts.points[idx] - ref[None]  # [k, n0, 3]
        shift = d.mean(axis=1)
        size = np.linalg.norm(shift, axis=1)
        if len(ref) == 1:
            ok = size > SINGLE_ITEM_SPACING_M
        else:
            ok = np.abs(d - shift[:, None, :]).max(axis=(1, 2)) <= SPACING_ABS_M + SPACING_RTOL * size
        ok &= (size > SPACING_MIN_M) & (ids != 0)
        out.extend((int(ids[i]), shift[i]) for i in np.flatnonzero(ok))
    return out


def world_centroids(points: np.ndarray, groups: WorldGroups) -> tuple[np.ndarray, np.ndarray]:
    """(world ids with at least one point, their centroids [k, 3] float64)."""
    has = np.flatnonzero(groups.counts > 0)
    sums = world_reduce(groups, np.asarray(points, dtype=np.float64), np.add, 0.0)
    return has, sums[has] / groups.counts[has][:, None]


def check_worlds_at_origin(model: newton.Model, cfg: FramePolicy) -> None:
    """Raise ``PrecisionError`` if worlds are spaced apart or a world sits beyond ``r_warn`` (F1)."""
    if not cfg.worlds_at_origin:
        return
    pts = world_points(model)
    spaced = spaced_worlds(pts)
    if spaced:
        w, shift = spaced[0]
        raise PrecisionError(
            f"{len(spaced)} of {pts.world_count} worlds are spaced copies (world {w} = world 0 + "
            f"{np.round(shift, 6).tolist()} m); replicate with spacing=(0, 0, 0) and keep offsets in "
            "BuiltScene.world_origin (FramePolicy.worlds_at_origin)")
    ids, centre = world_centroids(pts.points, pts.groups)
    r = np.linalg.norm(centre, axis=1)
    far = np.flatnonzero(r > cfg.r_warn_m)
    if len(far):
        raise PrecisionError(f"world {int(ids[far[0]])} is centred {float(r[far[0]]):.3f} m from the origin (> r_warn "
                             f"{cfg.r_warn_m} m); {len(far)} world(s) affected (F1/F2)")
    log.debug("worlds at origin: %d worlds checked", pts.world_count)


# ----- rebasing (F4) -----

@wp.kernel
def _scatter_shift(world_ids: wp.array[wp.int32], shift_m: wp.array[wp.vec3d], shift_w: wp.array[wp.vec3d],
                   active: wp.array[wp.int32], world_origin: wp.array[wp.vec3d]):
    k = wp.tid()
    w = world_ids[k]
    shift_w[w] = shift_m[k]
    active[w] = 1
    world_origin[w] = world_origin[w] + shift_m[k]


@wp.func
def _shifted(p: wp.vec3, s: wp.vec3d) -> wp.vec3:
    return wp.vec3(wp.vec3d(p) - s)  # one rounding, in float64


@wp.kernel
def _shift_transforms(xf: wp.array[wp.transform], item_world: wp.array[wp.int32], item_body: wp.array[wp.int32],
                      only_static: wp.int32, shift_w: wp.array[wp.vec3d], active: wp.array[wp.int32]):
    i = wp.tid()
    w = item_world[i]
    if w < 0 or active[w] == 0:
        return
    if only_static != 0 and item_body[i] >= 0:
        return
    t = xf[i]
    xf[i] = wp.transform(_shifted(wp.transform_get_translation(t), shift_w[w]), wp.transform_get_rotation(t))


@wp.kernel
def _shift_points(q: wp.array[wp.vec3], item_world: wp.array[wp.int32], shift_w: wp.array[wp.vec3d],
                  active: wp.array[wp.int32]):
    i = wp.tid()
    w = item_world[i]
    if w >= 0 and active[w] != 0:
        q[i] = _shifted(q[i], shift_w[w])


@wp.kernel
def _shift_root_joints(joint_type: wp.array[wp.int32], joint_parent: wp.array[wp.int32],
                       joint_world: wp.array[wp.int32], joint_q_start: wp.array[wp.int32],
                       joint_q: wp.array[wp.float32], joint_X_p: wp.array[wp.transform],
                       shift_w: wp.array[wp.vec3d], active: wp.array[wp.int32]):
    j = wp.tid()
    w = joint_world[j]
    if w < 0 or active[w] == 0 or joint_parent[j] >= 0:
        return
    jt = joint_type[j]
    if jt == JOINT_FREE or jt == JOINT_DISTANCE:
        s = joint_q_start[j]
        p = _shifted(wp.vec3(joint_q[s], joint_q[s + 1], joint_q[s + 2]), shift_w[w])
        joint_q[s] = p[0]
        joint_q[s + 1] = p[1]
        joint_q[s + 2] = p[2]
    else:  # fixed-base root: its anchor in the world frame moves
        t = joint_X_p[j]
        joint_X_p[j] = wp.transform(_shifted(wp.transform_get_translation(t), shift_w[w]),
                                    wp.transform_get_rotation(t))


def _check_rebase_allowed(model: newton.Model) -> None:
    if model.shape_count and bool((model.shape_world.numpy() < 0).any()):
        raise PrecisionError("rebase_worlds needs per-world statics: the model has shared (world -1) shapes, "
                             "which would move relative to the rebased worlds (research precision §4.2 F4)")


def rebase_worlds(model: newton.Model, state: newton.State, world_ids: wp.array[wp.int32],
                  shift_m: wp.array[wp.vec3d], world_origin: wp.array[wp.vec3d], *,
                  solvers: Sequence[newton.solvers.SolverBase] = ()) -> None:
    """Between steps: positions of ``world_ids[k]`` -= ``shift_m[k]``; ``world_origin[w]`` += shift (float64).

    Moves bodies, particles, free/distance root ``joint_q``, fixed-base root anchors (``joint_X_p``) and per-world
    static shapes; velocities are unchanged. Solvers in ``solvers`` are notified of the shape and joint changes.
    SolverMuJoCo reads ``joint_q`` from ``state_in`` each step only with ``update_data_interval == 1`` (default).
    """
    _check_rebase_allowed(model)
    W, dev = int(model.world_count), model.device
    shift_w = wp.zeros(W, dtype=wp.vec3d, device=dev)
    active = wp.zeros(W, dtype=wp.int32, device=dev)
    wp.launch(_scatter_shift, dim=world_ids.shape[0], inputs=[world_ids, shift_m, shift_w, active, world_origin],
              device=dev)
    if model.body_count:
        wp.launch(_shift_transforms, dim=model.body_count,
                  inputs=[state.body_q, model.body_world, model.body_world, 0, shift_w, active], device=dev)
    if model.shape_count:
        wp.launch(_shift_transforms, dim=model.shape_count,
                  inputs=[model.shape_transform, model.shape_world, model.shape_body, 1, shift_w, active], device=dev)
    if model.particle_count:
        wp.launch(_shift_points, dim=model.particle_count, inputs=[state.particle_q, model.particle_world,
                                                                    shift_w, active], device=dev)
    if model.joint_count:
        wp.launch(_shift_root_joints, dim=model.joint_count,
                  inputs=[model.joint_type, model.joint_parent, model.joint_world, model.joint_q_start,
                          state.joint_q, model.joint_X_p, shift_w, active], device=dev)
    flags = newton.ModelFlags.SHAPE_PROPERTIES | newton.ModelFlags.JOINT_PROPERTIES
    for solver in solvers:
        solver.notify_model_changed(flags)
    log.info("rebased %d world(s)", world_ids.shape[0])


def plan_rebase(model: newton.Model, state: newton.State, cfg: FramePolicy
                ) -> tuple[np.ndarray, np.ndarray] | None:
    """For ``rebase="auto"``: worlds whose dynamic-body centroid is beyond ``rebase_radius_m``, and shifts that
    bring the centroid back to the origin, snapped to ``REBASE_GRID_M`` (exact float32 subtraction). Host sync."""
    if cfg.rebase != "auto" or not model.body_count:
        return None
    ids, centre = world_centroids(state.body_q.numpy()[:, :3],
                                  group_by_world(model.body_world.numpy(), int(model.world_count)))
    far = np.linalg.norm(centre, axis=1) > cfg.rebase_radius_m
    ids, shifts = ids[far], np.round(centre[far] / REBASE_GRID_M) * REBASE_GRID_M
    if not len(ids):
        return None
    return ids.astype(np.int32), shifts.astype(np.float64)
