# SPDX-License-Identifier: Apache-2.0
"""Per-world reductions and item->world maps (SPEC §3.1).

Reductions use ``atomic_max``/``atomic_min`` and integer ``atomic_add`` only, which are order-independent and allowed
under the determinism rules (§3.7). Items whose world is -1 (global entities) or outside ``out`` are skipped. NaN
values are skipped by ``per_world_max``/``per_world_min`` (report them through the NONFINITE health flags); a world
with no valid item keeps the identity (-inf for max, +inf for min). Every function zero-fills/initialises ``out``
itself in the same launch sequence, so it is capture-safe and allocation-free.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy._vendor import newton


@wp.kernel
def _fill_f32(out: wp.array(dtype=wp.float32), value: wp.float32):
    out[wp.tid()] = value


@wp.kernel
def _fill_i32(out: wp.array(dtype=wp.int32), value: wp.int32):
    out[wp.tid()] = value


@wp.kernel
def _max_kernel(values: wp.array(dtype=wp.float32), item_world: wp.array(dtype=wp.int32),
                out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    w = item_world[i]
    v = values[i]
    if w >= 0 and w < out.shape[0] and not wp.isnan(v):
        wp.atomic_max(out, w, v)


@wp.kernel
def _min_kernel(values: wp.array(dtype=wp.float32), item_world: wp.array(dtype=wp.int32),
                out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    w = item_world[i]
    v = values[i]
    if w >= 0 and w < out.shape[0] and not wp.isnan(v):
        wp.atomic_min(out, w, v)


@wp.kernel
def _count_kernel(mask: wp.array(dtype=wp.int32), item_world: wp.array(dtype=wp.int32),
                  out: wp.array(dtype=wp.int32)):
    i = wp.tid()
    w = item_world[i]
    if w >= 0 and w < out.shape[0] and mask[i] != 0:
        wp.atomic_add(out, w, 1)


def _check_lengths(values: wp.array, item_world: wp.array) -> None:
    if values.shape[0] != item_world.shape[0]:
        raise ValueError(f"values ({values.shape[0]}) and item_world ({item_world.shape[0]}) differ in length")


def per_world_max(values: wp.array, item_world: wp.array, out: wp.array) -> None:
    """``out[w] = max(values[i] for i with item_world[i] == w)``; float32, NaN skipped, empty world -> -inf."""
    _check_lengths(values, item_world)
    wp.launch(_fill_f32, dim=out.shape[0], inputs=[out, wp.float32(-np.inf)], device=out.device)
    if values.shape[0]:
        wp.launch(_max_kernel, dim=values.shape[0], inputs=[values, item_world, out], device=out.device)


def per_world_min(values: wp.array, item_world: wp.array, out: wp.array) -> None:
    """``out[w] = min(values[i] for i with item_world[i] == w)``; float32, NaN skipped, empty world -> +inf."""
    _check_lengths(values, item_world)
    wp.launch(_fill_f32, dim=out.shape[0], inputs=[out, wp.float32(np.inf)], device=out.device)
    if values.shape[0]:
        wp.launch(_min_kernel, dim=values.shape[0], inputs=[values, item_world, out], device=out.device)


def per_world_count(mask: wp.array, item_world: wp.array, out: wp.array) -> None:
    """``out[w]`` = number of items of world w with ``mask[i] != 0`` (int32)."""
    _check_lengths(mask, item_world)
    wp.launch(_fill_i32, dim=out.shape[0], inputs=[out, 0], device=out.device)
    if mask.shape[0]:
        wp.launch(_count_kernel, dim=mask.shape[0], inputs=[mask, item_world, out], device=out.device)


# --- item -> world maps (host-built once, at construction) ------------------------------------------------------


def _expand_starts(starts: np.ndarray, total: int, owner_world: np.ndarray) -> np.ndarray:
    """World of every coordinate when ``starts[k]`` is the first coordinate of owner k."""
    starts = np.asarray(starts, dtype=np.int64)[: owner_world.shape[0]]
    bounds = np.append(starts, total)
    counts = np.diff(bounds)
    if np.any(counts < 0):
        raise ValueError("start offsets are not monotone")
    return np.repeat(owner_world.astype(np.int32), counts)


def _int_array(values: np.ndarray, device: wp.context.Device | str) -> wp.array:
    return wp.array(np.ascontiguousarray(values, dtype=np.int32), dtype=wp.int32, device=device)


def joint_coord_world(model: newton.Model) -> wp.array:
    """World of every joint coordinate (``joint_q`` entry), [joint_coord_count] int32."""
    worlds = model.joint_world.numpy() if model.joint_count else np.zeros(0, np.int32)
    starts = model.joint_q_start.numpy() if model.joint_count else np.zeros(0, np.int32)
    return _int_array(_expand_starts(starts, model.joint_coord_count, worlds), model.device)


def joint_dof_world(model: newton.Model) -> wp.array:
    """World of every joint DOF (``joint_qd`` entry), [joint_dof_count] int32."""
    worlds = model.joint_world.numpy() if model.joint_count else np.zeros(0, np.int32)
    starts = model.joint_qd_start.numpy() if model.joint_count else np.zeros(0, np.int32)
    return _int_array(_expand_starts(starts, model.joint_dof_count, worlds), model.device)


def world_iota(world_count: int, device: wp.context.Device | str) -> wp.array:
    """``[0, 1, ..., W-1]`` int32: the item->world map of a world-major array."""
    return _int_array(np.arange(world_count, dtype=np.int32), device)


def item_world_for(model: newton.Model, frequency: object) -> wp.array | None:
    """Item->world map for a ``Model.AttributeFrequency`` (BODY, PARTICLE, SHAPE, JOINT, JOINT_COORD, JOINT_DOF,
    ARTICULATION, WORLD); None for frequencies without a per-world map."""
    freq = newton.Model.AttributeFrequency
    direct = {freq.BODY: "body_world", freq.PARTICLE: "particle_world", freq.SHAPE: "shape_world",
              freq.JOINT: "joint_world", freq.ARTICULATION: "articulation_world"}
    if frequency in direct:
        return getattr(model, direct[frequency], None)
    if frequency == freq.JOINT_COORD:
        return joint_coord_world(model)
    if frequency == freq.JOINT_DOF:
        return joint_dof_world(model)
    if frequency == freq.WORLD:
        return world_iota(model.world_count, model.device)
    return None
