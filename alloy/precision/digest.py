# SPDX-License-Identifier: Apache-2.0
"""Per-world state digests and the run digest (SPEC §3.7; precision research §4.6).

One thread per world hashes the raw bits of ``body_q, body_qd, joint_q, joint_qd, particle_q, particle_qd`` and then
every ``extra`` array in name order, with two independent splitmix64 lanes (128 bits per world). Items of a world are
visited in index order; global items (world -1) are hashed into every world first. An ``extra`` array whose first axis
is the world count is hashed per row; any other ``extra`` array is hashed whole into every world (a warm start the
next step reads must change every digest it can affect). Each array is framed by a tag (its position and word count)
and the world's own item count, so moving bits between arrays changes the digest while a world's digest stays
independent of how many other worlds the model holds (batch invariance, R4).

The run digest is SHA-256 over the world digests, ``PrecisionConfig.digest()`` and the environment tuple.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import weakref
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from alloy.core.ids import canonical_json
from alloy.precision.world_groups import group_by_world

log = logging.getLogger("alloy.precision")

STATE_ARRAYS = ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd")
# SCR-WP11-F32 (main fb7d80a8a): VBD keeps each particle position's sub-ulp remainder in State.particle_q_lo. It is
# state (two runs equal in particle_q and different here diverge at the next step), so it is in the digest.
LATE_PARTICLE_ARRAY = "particle_q_lo"
_LATE_TAG = 0x5F10
_LANE_SEEDS = (0x243F6A8885A308D3, 0x13198A2E03707344)  # pi digits: fixed, arbitrary, distinct


@wp.func
def splitmix64(x: wp.uint64) -> wp.uint64:
    z = x + wp.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> wp.uint64(30))) * wp.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> wp.uint64(27))) * wp.uint64(0x94D049BB133111EB)
    return z ^ (z >> wp.uint64(31))


@wp.func
def _mix(lane_a: wp.uint64, lane_b: wp.uint64, word: wp.uint64) -> wp.vec2ul:
    a = splitmix64(lane_a ^ word)
    b = splitmix64(lane_b + word * wp.uint64(0xD6E8FEB86659FD93))
    return wp.vec2ul(a, b)


@wp.kernel
def _hash_words32(words: wp.array2d[wp.uint32], begin: wp.array[wp.int32], end: wp.array[wp.int32],
                  items: wp.array[wp.int32], tag: wp.uint64, lanes: wp.array2d[wp.uint64]):
    w = wp.tid()
    ab = _mix(lanes[w, 0], lanes[w, 1], tag)
    ab = _mix(ab[0], ab[1], wp.uint64(end[w] - begin[w]))  # this world's item count, not the array's (R4)
    for k in range(begin[w], end[w]):
        it = items[k]
        for j in range(words.shape[1]):
            ab = _mix(ab[0], ab[1], wp.uint64(words[it, j]))
    lanes[w, 0] = ab[0]
    lanes[w, 1] = ab[1]


@wp.kernel
def _hash_words64(words: wp.array2d[wp.uint64], begin: wp.array[wp.int32], end: wp.array[wp.int32],
                  items: wp.array[wp.int32], tag: wp.uint64, lanes: wp.array2d[wp.uint64]):
    w = wp.tid()
    ab = _mix(lanes[w, 0], lanes[w, 1], tag)
    ab = _mix(ab[0], ab[1], wp.uint64(end[w] - begin[w]))  # this world's item count, not the array's (R4)
    for k in range(begin[w], end[w]):
        it = items[k]
        for j in range(words.shape[1]):
            ab = _mix(ab[0], ab[1], words[it, j])
    lanes[w, 0] = ab[0]
    lanes[w, 1] = ab[1]


@dataclass(frozen=True)
class WorldItems:
    """Per-world item lists on the device: world ``w`` hashes ``items[begin[w]:end[w]]`` in that order."""

    begin: wp.array
    end: wp.array
    items: wp.array
    count: int  # total items in the source array


def world_items(item_world: np.ndarray, world_count: int, device: wp.context.Devicelike,
                include_global: bool = True) -> WorldItems:
    """CSR lists from a per-item world id; global items (world < 0) go first into every world, or nowhere."""
    item_world = np.asarray(item_world, dtype=np.int64)
    glob = np.flatnonzero(item_world < 0) if include_global else np.zeros(0, dtype=np.int64)
    groups = group_by_world(item_world, world_count)
    n_glob = len(glob)
    sizes = groups.counts + n_glob
    end = np.cumsum(sizes)
    begin = end - sizes
    flat = np.empty(int(end[-1]) if world_count else 0, dtype=np.int64)
    if n_glob:
        flat[begin[:, None] + np.arange(n_glob)[None, :]] = glob[None, :]
    # sorted item k of world w lands after the w+1 global prefixes so far
    flat[np.arange(len(groups.order)) + (groups.world_of_sorted() + 1) * n_glob] = groups.order
    return WorldItems(wp.array(begin.astype(np.int32), device=device), wp.array(end.astype(np.int32), device=device),
                      wp.array(flat.astype(np.int32), device=device), len(item_world))


def _coord_world(starts: np.ndarray, joint_world: np.ndarray) -> np.ndarray:
    """World id per joint coordinate from ``joint_q_start``/``joint_qd_start``."""
    counts = np.diff(np.asarray(starts, dtype=np.int64))
    return np.repeat(np.asarray(joint_world, dtype=np.int64)[: len(counts)], counts)


class _ModelIndex:
    """Item lists per state array for one Model (built once on the host, cached per Model)."""

    def __init__(self, model) -> None:
        self.world_count = int(model.world_count)
        dev = model.device
        jw = _np(model.joint_world, model.joint_count)
        self.lists: dict[str, WorldItems] = {
            "body": world_items(_np(model.body_world, model.body_count), self.world_count, dev),
            "particle": world_items(_np(model.particle_world, model.particle_count), self.world_count, dev),
            "joint_q": world_items(_coord_world(_np(model.joint_q_start, model.joint_count + 1), jw),
                                   self.world_count, dev),
            "joint_qd": world_items(_coord_world(_np(model.joint_qd_start, model.joint_count + 1), jw),
                                    self.world_count, dev),
        }
        self.per_row = world_items(np.arange(self.world_count), self.world_count, dev)
        self._whole: dict[int, WorldItems] = {}
        self._mapped: dict[tuple, WorldItems] = {}

    def mapped(self, name: str, item_world: wp.array, device) -> WorldItems:
        """Item lists from an item->world map (cached per array identity)."""
        key = (name, item_world.ptr, item_world.shape[0])
        if key not in self._mapped:
            self._mapped[key] = world_items(item_world.numpy(), self.world_count, device)
        return self._mapped[key]

    def whole(self, n: int, device) -> WorldItems:
        """Every world hashes all ``n`` rows."""
        if n not in self._whole:
            zeros = np.zeros(self.world_count, dtype=np.int32)
            ends = np.full(self.world_count, n, dtype=np.int32)
            self._whole[n] = WorldItems(wp.array(zeros, device=device), wp.array(ends, device=device),
                                        wp.array(np.arange(n, dtype=np.int32), device=device), n)
        return self._whole[n]


def _np(arr: wp.array | None, n: int) -> np.ndarray:
    return arr.numpy() if arr is not None and n > 0 else np.zeros(0, dtype=np.int32)


_INDEX: "weakref.WeakKeyDictionary[object, _ModelIndex]" = weakref.WeakKeyDictionary()


def _index_for(model) -> _ModelIndex:
    idx = _INDEX.get(model)
    if idx is None or idx.world_count != int(model.world_count):
        idx = _ModelIndex(model)
        _INDEX[model] = idx
    return idx


def as_words(arr: wp.array) -> wp.array:
    """A zero-copy 2-d word view ``[rows, words]`` of ``arr`` (uint32 for 4-byte scalars, uint64 for 8-byte)."""
    if not arr.is_contiguous:
        arr = arr.contiguous()
    scalar = getattr(arr.dtype, "_wp_scalar_type_", arr.dtype)  # vec/mat/transform -> component type
    word = None if wp.types.type_is_struct(arr.dtype) else {4: wp.uint32, 8: wp.uint64}.get(
        wp.types.type_size_in_bytes(scalar))
    if word is None:  # 1- and 2-byte scalars (bool, int8, float16): widen on the host
        host = arr.numpy().astype(np.uint32).reshape(arr.shape[0] if arr.ndim else 1, -1)
        return wp.array(host, dtype=wp.uint32, device=arr.device)
    view = arr.view(word)
    rows = view.shape[0] if view.ndim else 1
    return view.reshape((rows, -1)) if view.ndim != 2 else view


def _hash_array(lanes: wp.array, arr: wp.array, items: WorldItems, tag: int) -> None:
    words = as_words(arr)
    if words.shape[0] < items.count:
        raise ValueError(f"array has {words.shape[0]} rows, item lists expect {items.count}")
    full_tag = (tag << 32) ^ words.shape[1]
    kernel = _hash_words32 if words.dtype == wp.uint32 else _hash_words64
    wp.launch(kernel, dim=lanes.shape[0], inputs=[words, items.begin, items.end, items.items,
                                                  wp.uint64(full_tag & 0xFFFFFFFFFFFFFFFF), lanes],
              device=lanes.device)


def world_digests(model, state, extra: Mapping[str, wp.array] = {}, *,
                  extra_world: Mapping[str, wp.array] | None = None, whole_extras: bool = True) -> np.ndarray:
    """``[W, 2]`` uint64 digest per world of the state arrays plus ``extra`` (sorted by name). An ``extra`` array
    with an item->world map in ``extra_world`` (as for ``WorldSnapshot``) is hashed per item world; one with neither a
    map nor a world-major first axis is hashed whole into every world, unless ``whole_extras=False`` (batch-invariance
    comparisons across world counts, R4, where such arrays depend on the other worlds by construction). Host sync."""
    idx = _index_for(model)
    W = idx.world_count
    seeds = np.tile(np.array(_LANE_SEEDS, dtype=np.uint64), (W, 1))
    lanes = wp.array(seeds, dtype=wp.uint64, device=model.device)
    groups = ("body", "body", "joint_q", "joint_qd", "particle", "particle")
    for tag, (name, group) in enumerate(zip(STATE_ARRAYS, groups)):
        arr = getattr(state, name, None)
        if arr is not None and idx.lists[group].count > 0:
            _hash_array(lanes, arr, idx.lists[group], tag)
    for k, name in enumerate(sorted(extra)):
        arr = extra[name]
        if arr is None or arr.size == 0:
            continue
        rows = arr.shape[0] if arr.ndim else 1
        mapped = (extra_world or {}).get(name)
        if mapped is not None:
            items = idx.mapped(name, mapped, model.device)
        elif arr.ndim and rows == W:
            items = idx.per_row
        elif whole_extras:
            items = idx.whole(rows, model.device)
        else:
            continue
        _hash_array(lanes, arr, items, len(STATE_ARRAYS) + k)
    lo = getattr(state, LATE_PARTICLE_ARRAY, None)
    if lo is not None and idx.lists["particle"].count > 0:   # after the extras, own tag: older digests keep theirs
        _hash_array(lanes, lo, idx.lists["particle"], _LATE_TAG)
    return lanes.numpy()


def run_digest(world_digests: np.ndarray, cfg, env: Mapping[str, str]) -> str:
    """SHA-256 hex over the world digests (little-endian), ``cfg.digest()`` and the canonical env tuple."""
    h = hashlib.sha256(b"alloy.run_digest/1\0")
    h.update(np.ascontiguousarray(world_digests, dtype="<u8").tobytes())
    h.update(cfg.digest().encode())
    h.update(canonical_json(dict(env)))
    return h.hexdigest()


def env_tuple(device: wp.context.Devicelike = None, world_count: int = 0) -> dict[str, str]:
    """GPU, sm, driver, CUDA, Warp, Newton commit, Alloy commit and world count (SPEC §6.5 golden key)."""
    from alloy._vendor import newton

    from alloy.core.newton_compat import NEWTON_PIN

    dev = wp.get_device(device)
    env = {"device": dev.name if dev.is_cuda else "cpu", "arch": str(getattr(dev, "arch", "") or "cpu"),
           "warp": wp.config.version, "newton": f"{newton.__version__}@{NEWTON_PIN[:8]}",
           "alloy": _alloy_commit(), "world_count": str(world_count)}
    if dev.is_cuda:
        env["driver"] = ".".join(map(str, wp.get_cuda_driver_version() or ()))
        env["cuda"] = ".".join(map(str, wp.get_cuda_toolkit_version() or ()))
    return env


def _alloy_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=root, capture_output=True, text=True,
                             timeout=10, check=False)
    except OSError as exc:
        log.debug("alloy commit unknown: %s", exc)
        return "unknown"
    return out.stdout.strip() or "unknown"
