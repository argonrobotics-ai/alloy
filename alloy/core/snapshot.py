# SPDX-License-Identifier: Apache-2.0
"""Per-world snapshot and restore of simulation state (SPEC §3.1), used by latching, rollback and save/load.

A ``WorldSnapshot`` preallocates a bitwise copy of every ``State`` array (bodies, joints, particles, custom
attributes) plus ``extra`` arrays (state adapters, the stepper's ``state_arrays()``, plug-in ``alloy_state_arrays``).
Copies are raw 32-bit words (bytes for odd-sized items), so NaN payloads and -0.0 survive exactly.

Per-world axis of each array:
- ``State`` arrays: from ``Model.get_attribute_frequency`` (BODY, PARTICLE, SHAPE, JOINT, JOINT_COORD, JOINT_DOF,
  ARTICULATION, WORLD), else by name prefix (``body_``, ``particle_``, ``joint_q``/``joint_qd``, ``shape_``).
- ``extra`` arrays: an item->world map from ``extra_world[name]`` if given, else world-major if
  ``shape[0] == world_count`` (MJWarp ``(nworld, ...)`` arrays), else none.
An array with no per-world axis is snapshotted whole; ``save``/``restore`` with a ``world_mask`` then raise
``SnapshotError``. ``world_mask`` is an int32 array of length ``world_count`` (nonzero = selected; a longer mask,
like Newton's ``world_count + 1`` reset masks, is accepted and its tail ignored). Global entities (world -1) are
copied only when ``world_mask`` is None. ``save``/``restore`` launch kernels only (capture-safe, no allocation).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton import Model, State

from alloy.core.errors import SnapshotError
from alloy.core.worldops import item_world_for, joint_coord_world, joint_dof_world, world_iota

log = logging.getLogger("alloy.core")


@wp.kernel
def _copy_words(src: wp.array(dtype=wp.uint32), dst: wp.array(dtype=wp.uint32), item_world: wp.array(dtype=wp.int32),
                words: wp.int32, mask: wp.array(dtype=wp.int32), use_mask: wp.int32):
    tid = wp.tid()
    if use_mask != 0:
        w = item_world[tid // words]
        if w < 0 or w >= mask.shape[0]:
            return
        if mask[w] == 0:
            return
    dst[tid] = src[tid]


@wp.kernel
def _copy_bytes(src: wp.array(dtype=wp.uint8), dst: wp.array(dtype=wp.uint8), item_world: wp.array(dtype=wp.int32),
                words: wp.int32, mask: wp.array(dtype=wp.int32), use_mask: wp.int32):
    tid = wp.tid()
    if use_mask != 0:
        w = item_world[tid // words]
        if w < 0 or w >= mask.shape[0]:
            return
        if mask[w] == 0:
            return
    dst[tid] = src[tid]


FUSED = 16                         # word slots per fused launch (SCR-WP0-RIGIDDT-2: one launch, not one per slot)
_vec_ptr = wp.types.vector(length=FUSED, dtype=wp.uint64)
_vec_off = wp.types.vector(length=FUSED + 1, dtype=wp.int32)


@wp.struct
class _Table:
    """Up to FUSED word slots by value: source / destination / item->world pointers (0 = whole array), words per
    item, and each slot's first word in the launch (unused slots start at the launch size)."""
    src: _vec_ptr
    dst: _vec_ptr
    world: _vec_ptr
    words: _vec_off
    start: _vec_off


@wp.kernel
def _copy_slots(t: _Table, mask: wp.array(dtype=wp.int32), use_mask: wp.int32):
    """`_copy_words` for up to FUSED slots in one launch. The slot lookup is an unrolled scan with constant indices:
    a runtime index into the by-value table would copy the whole table into each thread's local memory (1 KB per
    thread: 1.5x slower than the per-slot launches on an L4 at 64 worlds)."""
    tid = wp.tid()
    src, dst, world, words, base, end = t.src[0], t.dst[0], t.world[0], t.words[0], int(0), t.start[1]
    for j in range(1, wp.static(FUSED)):
        if tid >= t.start[j]:
            src, dst, world, words, base, end = t.src[j], t.dst[j], t.world[j], t.words[j], t.start[j], t.start[j + 1]
    i = tid - base
    n = end - base
    if use_mask != 0:
        iw = wp.array(ptr=world, shape=(n // words,), dtype=wp.int32)
        w = iw[i // words]
        if w < 0 or w >= mask.shape[0]:
            return
        if mask[w] == 0:
            return
    out = wp.array(ptr=dst, shape=(n,), dtype=wp.uint32)
    inp = wp.array(ptr=src, shape=(n,), dtype=wp.uint32)
    out[i] = inp[i]


@wp.kernel
def _any_selected(mask: wp.array(dtype=wp.int32), n: wp.int32, out: wp.array(dtype=wp.int32)):
    w = wp.tid()
    if w < n and mask[w] != 0:
        out[0] = 1                         # every writer writes 1


@dataclass
class _Slot:
    name: str                       # "state:body_q", "state:sph:rest_density", "extra:<key>"
    n_items: int
    words: int                      # words (or bytes) per item
    byte_mode: bool
    item_world: wp.array | None     # None -> whole array, no per-world axis
    store: wp.array                 # flat u32 / u8 copy
    extra: wp.array | None = None   # the fixed source array for extra slots


def _nbytes(arr: wp.array) -> int:
    return int(arr.size) * wp.types.type_size_in_bytes(arr.dtype)


def _flat_view(arr: wp.array, byte_mode: bool) -> wp.array:
    """Reinterpret a contiguous array as flat uint32 words (or bytes); shares memory, allocates nothing."""
    dtype = wp.uint8 if byte_mode else wp.uint32
    n = _nbytes(arr) // (1 if byte_mode else 4)
    return wp.array(ptr=arr.ptr, dtype=dtype, shape=(n,), device=arr.device, copy=False)


def _state_arrays(state: State) -> Iterator[tuple[str, str, wp.array]]:
    """(slot name, attribute path, array) for every array on ``state``, including one level of namespaces."""
    for name, value in sorted(vars(state).items()):
        if isinstance(value, wp.array):
            yield f"state:{name}", name, value
        elif isinstance(value, newton.Model.AttributeNamespace):
            for sub, arr in sorted(vars(value).items()):
                if isinstance(arr, wp.array):
                    yield f"state:{name}:{sub}", f"{name}:{sub}", arr


def _resolve(state: State, path: str) -> wp.array:
    obj: object = state
    for part in path.split(":"):
        obj = getattr(obj, part)
    return obj  # type: ignore[return-value]


def _state_item_world(model: Model, path: str, arr: wp.array) -> wp.array | None:
    try:
        freq = model.get_attribute_frequency(path)
    except (KeyError, AttributeError):
        freq = None
    if freq is not None:
        mapped = item_world_for(model, freq)
        if mapped is not None and mapped.shape[0] == arr.shape[0]:
            return mapped
    return _prefix_item_world(model, path, arr)


def _prefix_item_world(model: Model, path: str, arr: wp.array) -> wp.array | None:
    n = arr.shape[0]
    if path.startswith("body_") and n == model.body_count:
        return model.body_world
    if path.startswith("particle_") and n == model.particle_count:
        return model.particle_world
    if path.startswith("shape_") and n == model.shape_count:
        return model.shape_world
    if path == "joint_q" or (path.startswith("joint_") and n == model.joint_coord_count != model.joint_dof_count):
        return joint_coord_world(model)
    if path.startswith("joint_") and n == model.joint_dof_count:
        return joint_dof_world(model)
    if n == model.world_count:
        return world_iota(model.world_count, model.device)
    return None


class WorldSnapshot:
    """Per-world copy of ``State`` arrays plus ``extra`` arrays (see module docstring). Preallocated; capture-safe."""

    def __init__(self, model: Model, state: State, extra: Mapping[str, wp.array] | None = None, *,
                 extra_world: Mapping[str, wp.array] | None = None) -> None:
        self.model = model
        self.world_count = int(model.world_count)
        self.device = model.device
        self._dummy_mask = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._any = wp.zeros(1, dtype=wp.int32, device=self.device)        # restore_selected's graph condition
        self._views: dict[tuple[int, bool], wp.array] = {}
        self._tables: dict[tuple, tuple[_Table, int]] = {}
        self.slots: list[_Slot] = []
        for slot_name, path, arr in _state_arrays(state):
            self._add(slot_name, arr, _state_item_world(model, path, arr), extra=None)
        for key, arr in sorted((extra or {}).items()):
            mapped = (extra_world or {}).get(key)
            if mapped is None and arr.ndim >= 1 and arr.shape[0] == self.world_count:
                mapped = world_iota(self.world_count, self.device)
            self._add(f"extra:{key}", arr, mapped, extra=arr)
        whole = self.whole_arrays
        if whole:
            log.info("WorldSnapshot: arrays without a per-world axis (per-world restore will raise): %s", whole)

    # --- construction --------------------------------------------------------------------------------------------

    def _add(self, name: str, arr: wp.array, item_world: wp.array | None, extra: wp.array | None) -> None:
        if arr is None or arr.size == 0:
            return
        if not arr.is_contiguous:
            raise SnapshotError(f"{name}: non-contiguous arrays cannot be snapshotted")
        n_items = int(arr.shape[0]) if item_world is not None else 1
        item_bytes = _nbytes(arr) // n_items
        byte_mode = item_bytes % 4 != 0
        words = item_bytes if byte_mode else item_bytes // 4
        store = wp.zeros(words * n_items, dtype=wp.uint8 if byte_mode else wp.uint32, device=self.device)
        self.slots.append(_Slot(name, n_items, words, byte_mode, item_world, store, extra))

    @property
    def whole_arrays(self) -> tuple[str, ...]:
        """Slot names snapshotted whole (no per-world axis)."""
        return tuple(s.name for s in self.slots if s.item_world is None)

    @property
    def nbytes(self) -> int:
        """Device bytes held by the snapshot."""
        return sum(int(s.store.size) * (1 if s.byte_mode else 4) for s in self.slots)

    # --- host export (R5 save -> load -> continue, SCR-WP5-4) ---------------------------------------------------

    def to_host(self) -> dict[str, np.ndarray]:
        """Slot name -> copy of the stored words (uint32, or uint8 for odd-sized items)."""
        return {slot.name: slot.store.numpy().copy() for slot in self.slots}

    def from_host(self, data: Mapping[str, np.ndarray]) -> None:
        """Load words saved by ``to_host`` (possibly from another process) into this snapshot. Raises
        ``SnapshotError`` if the slot names, dtypes or sizes differ."""
        names = {slot.name for slot in self.slots}
        if set(data) != names:
            raise SnapshotError(f"slot mismatch: missing {sorted(names - set(data))}, "
                                f"extra {sorted(set(data) - names)}")
        for slot in self.slots:
            words = np.asarray(data[slot.name])
            expected = np.uint8 if slot.byte_mode else np.uint32
            if words.dtype != expected or words.size != int(slot.store.size):
                raise SnapshotError(f"{slot.name}: expected {slot.store.size} {np.dtype(expected).name} words, got "
                                    f"{words.size} {words.dtype.name}")
            slot.store.assign(wp.array(np.ascontiguousarray(words.ravel()), dtype=slot.store.dtype, device="cpu"))

    # --- save / restore ------------------------------------------------------------------------------------------

    def save(self, state: State, world_mask: wp.array | None = None) -> None:
        """Copy ``state`` (and the extra arrays) into the snapshot, for the masked worlds only if a mask is given."""
        self._check_mask(world_mask)
        self._copy(state, world_mask, into_store=True)

    def restore(self, state: State, world_mask: wp.array | None = None) -> None:
        """Copy the snapshot back into ``state`` (and the extra arrays), for the masked worlds only if given."""
        self._check_mask(world_mask)
        self._copy(state, world_mask, into_store=False)

    def _copy(self, state: State, world_mask: wp.array | None, into_store: bool) -> None:
        """Byte slots one launch each; word slots FUSED per launch (bit for bit the per-slot `_copy_words`)."""
        words = []
        for slot in self.slots:
            arr = self._source(slot, state)
            src, dst = (arr, slot.store) if into_store else (slot.store, arr)
            if slot.byte_mode:
                self._launch(slot, src, dst, world_mask)
            else:
                words.append((slot, src, dst))
        mask = world_mask if world_mask is not None else self._dummy_mask
        use_mask = 0 if world_mask is None else 1
        for k in range(0, len(words), FUSED):
            table, size = self._table(words[k:k + FUSED])
            wp.launch(_copy_slots, dim=size, inputs=[table, mask, use_mask], device=self.device)

    def _table(self, group: list[tuple[_Slot, wp.array, wp.array]]) -> tuple[_Table, int]:
        """The by-value pointer table for one group, cached by its pointers (the State objects ping-pong)."""
        key = tuple((src.ptr, dst.ptr) for _, src, dst in group)
        hit = self._tables.get(key)
        if hit is not None:
            return hit
        src, dst, world = [0] * FUSED, [0] * FUSED, [0] * FUSED
        words, start = [1] * (FUSED + 1), [0] * (FUSED + 1)
        for j, (slot, s, d) in enumerate(group):
            src[j], dst[j] = s.ptr, d.ptr
            world[j] = slot.item_world.ptr if slot.item_world is not None else 0
            words[j] = slot.words
            start[j + 1] = start[j] + int(d.size)
        for j in range(len(group) + 1, FUSED + 1):
            start[j] = start[len(group)]
        table = _Table()
        table.src, table.dst, table.world = _vec_ptr(*src), _vec_ptr(*dst), _vec_ptr(*world)
        table.words, table.start = _vec_off(*words), _vec_off(*start)
        self._tables[key] = (table, start[len(group)])
        return self._tables[key]

    def restore_selected(self, state: State, world_mask: wp.array) -> None:
        """``restore(state, world_mask)`` that does nothing, and launches no copy, when no world is selected (a healthy
        latch every step: SCR-WP0-RIGIDDT-2).
        - Under a CUDA graph capture: a conditional node on "any world selected".
        - On the CPU, eager: the mask is read on the host.
        - Eager on CUDA, and while a CPU graph records (its mask is not computed yet, and APIC has no conditional
          nodes): the copy runs every step, as ``restore`` does (an empty mask copies nothing)."""
        self._check_mask(world_mask)
        dev = wp.get_device(self.device)
        if dev.is_cpu and not dev.is_capturing:
            if world_mask.numpy()[:self.world_count].any():
                self._copy(state, world_mask, into_store=False)
            return
        if not (dev.is_cuda and dev.is_capturing and wp.is_conditional_graph_supported()):
            self._copy(state, world_mask, into_store=False)
            return
        self._any.zero_()
        wp.launch(_any_selected, dim=self.world_count, inputs=[world_mask, self.world_count, self._any],
                  device=self.device)
        wp.capture_if(self._any, on_true=lambda: self._copy(state, world_mask, into_store=False))

    def _check_mask(self, world_mask: wp.array | None) -> None:
        if world_mask is None:
            return
        if world_mask.dtype != wp.int32 or world_mask.shape[0] < self.world_count:
            raise SnapshotError(f"world_mask must be int32 of length >= {self.world_count}, got "
                                f"{wp.types.type_repr(world_mask.dtype)}[{world_mask.shape[0]}]")
        if world_mask.device != self.device:
            raise SnapshotError(f"world_mask device {world_mask.device} != snapshot device {self.device}")
        whole = self.whole_arrays
        if whole:
            raise SnapshotError(f"per-world save/restore impossible: no per-world axis for {whole}")

    def _source(self, slot: _Slot, state: State) -> wp.array:
        arr = slot.extra if slot.extra is not None else _resolve(state, slot.name.removeprefix("state:"))
        key = (arr.ptr, slot.byte_mode)
        view = self._views.get(key)
        if view is None:
            if _nbytes(arr) != int(slot.store.size) * (1 if slot.byte_mode else 4):
                raise SnapshotError(f"{slot.name}: array size changed since the snapshot was built")
            view = _flat_view(arr, slot.byte_mode)
            self._views[key] = view
        return view

    def _launch(self, slot: _Slot, src: wp.array, dst: wp.array, world_mask: wp.array | None) -> None:
        kernel = _copy_bytes if slot.byte_mode else _copy_words
        item_world = slot.item_world if slot.item_world is not None else self._dummy_mask
        mask = world_mask if world_mask is not None else self._dummy_mask
        use_mask = 0 if world_mask is None else 1
        wp.launch(kernel, dim=int(dst.size), inputs=[src, dst, item_world, slot.words, mask, use_mask],
                  device=self.device)


def mask_from_worlds(worlds: list[int] | np.ndarray, world_count: int, device: wp.context.Device | str) -> wp.array:
    """Host helper: int32 mask of length ``world_count`` with 1 at ``worlds``."""
    mask = np.zeros(world_count, dtype=np.int32)
    mask[np.asarray(worlds, dtype=np.int64)] = 1
    return wp.array(mask, dtype=wp.int32, device=device)
