# SPDX-License-Identifier: Apache-2.0
"""The check's contact probe in world chunks, for a many-world scene whose probe does not fit the free device memory
at once (``alloy.check.contacts.ContactProbe``).

``plan_chunks(model, budget)`` estimates the probe's buffers (``probe_bytes``: Newton's own bytes per rigid contact
slot and per triangle pair) against the device's free memory and returns None when all worlds fit, else world ranges
that each fit. ``collide_chunks`` then builds one pipeline per range in turn (Newton's explicit pairs of those worlds
only, buffers sized for them), collides it and keeps a host copy of its contacts, so only one chunk's buffers are on
the device at a time. The copies come back as one ``HostContacts`` buffer, read like Newton's ``Contacts``.

Example::

    >>> plan_chunks(built.model, budget, free=2e9)                     # doctest: +SKIP
    [(0, 512), (512, 1024)]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger("alloy.check")

#: Newton's base rigid-contact buffers take 96 bytes per slot (collide.py, _RIGID_CONTACT_BASE_BYTES_PER_SLOT); the
#: fallback layer adds stiffness, damping and friction (3 float32)
RIGID_SLOT_BYTES = 96 + 12
#: Newton's narrow phase with the contact reducer: 52 bytes per triangle pair plus 68 per hashtable entry, the table
#: 0.25 entries per pair rounded up to a power of two (collide.py, _triangle_pair_allocation_bytes): at most 86
TRIANGLE_PAIR_BYTES = 52 + 34
USABLE = 0.8                 # share of the free device memory one probe pipeline may take
RIGID_FIELDS = ("rigid_contact_shape0", "rigid_contact_shape1", "rigid_contact_point0", "rigid_contact_point1",
                "rigid_contact_normal", "rigid_contact_margin0", "rigid_contact_margin1")


def probe_bytes(budget: Any, worlds: int) -> int:
    """Device bytes of a probe with per-world ``budget`` (``ContactBudget``) over ``worlds`` worlds.

    Example::

        >>> from alloy.scene.steppers import ContactBudget
        >>> probe_bytes(ContactBudget(triangle_pairs=2000, rigid_contacts=1000), 1024)
        286720000
    """
    rigid, tri = int(budget.rigid_contacts or 0), int(budget.triangle_pairs or 0)
    return (rigid * RIGID_SLOT_BYTES + tri * TRIANGLE_PAIR_BYTES) * int(worlds)


def full(need: dict[str, int]) -> bool:
    """True when the rigid contact buffer is full: Newton stops counting rigid contacts at the buffer's size (a
    box-on-box scene asked 192 and read 128 for 128 slots, no overflow flagged), so a full buffer may have lost some.

    Example::

        >>> full({"rigid_contacts": 128, "rigid_slots": 128}), full({"rigid_contacts": 127, "rigid_slots": 128})
        (True, False)
    """
    return need["rigid_contacts"] >= need["rigid_slots"] > 0


def device_free(device: Any) -> float | None:
    """Free bytes of a CUDA device once Warp's mempool has handed back what it holds unused; None on the CPU."""
    from alloy.check.memory import free_bytes  # noqa: PLC0415

    return free_bytes(str(device))


def plan_chunks(model: Any, budget: Any, free: float | None = None) -> list[tuple[int, int]] | None:
    """World ranges ``[lo, hi)`` whose probe each fits ``USABLE`` of the free device memory (``free``, else the
    device's own); None when every world fits at once, the budget is Newton's automatic size, or the device is a CPU.

    Example::

        >>> budget = ContactBudget(triangle_pairs=2000, rigid_contacts=1000)   # doctest: +SKIP
        >>> plan_chunks(built.model, budget, free=2e8)                  # doctest: +SKIP
        [(0, 512), (512, 1024)]
    """
    if budget is None:
        return None
    worlds = max(int(model.world_count or 0), 1)
    free = device_free(model.device) if free is None else free
    need = probe_bytes(budget, worlds)
    if free is None or need <= USABLE * free:
        return None
    per_chunk = max(int(USABLE * free // max(probe_bytes(budget, 1), 1)), 1)
    n = math.ceil(worlds / per_chunk)
    size = math.ceil(worlds / n)
    ranges = [(lo, min(lo + size, worlds)) for lo in range(0, worlds, size)]
    log.info("check probe: %.0f MB for %d worlds does not fit %.0f MB free; %d chunks of up to %d worlds",
             need / 1e6, worlds, free / 1e6, len(ranges), size)
    return ranges


def halved(ranges: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Every range split in two (a one-world range kept); None when every range is one world already.

    Example::

        >>> halved([(0, 5), (5, 6)])
        [(0, 3), (3, 5), (5, 6)]
    """
    if all(hi - lo <= 1 for lo, hi in ranges):
        return None
    out = []
    for lo, hi in ranges:
        mid = lo + -(-(hi - lo) // 2)
        out += [(lo, mid), (mid, hi)] if hi - lo > 1 else [(lo, hi)]
    return out


def chunk_pairs(model: Any, lo: int, hi: int, first: bool) -> np.ndarray:
    """Newton's shape pairs that may collide (``model.shape_contact_pairs``'s rule) with a shape in worlds
    ``[lo, hi)``, plus pairs of two global shapes in the ``first`` chunk only (each pair is collided once)."""
    sw = model.shape_world.numpy()
    mask = ((sw >= lo) & (sw < hi)) | (sw < 0)
    pairs = _pairs_for_mask(model, mask)
    if not first and pairs.shape[0]:
        pairs = pairs[(sw[pairs[:, 0]] >= 0) | (sw[pairs[:, 1]] >= 0)]
    return pairs


def _pairs_for_mask(model: Any, mask: np.ndarray) -> np.ndarray:
    """The pairs of shapes both in ``mask``, without building the model's full pair list on the device."""
    try:
        from alloy._vendor.newton._src.sim.shape_contact_pairs import _shape_contact_pairs_for_mask  # noqa: PLC0415
    except ImportError:                                  # a Newton without it: the model's own list, filtered
        pairs = model.shape_contact_pairs.numpy().reshape(-1, 2)
        return pairs[mask[pairs[:, 0]] & mask[pairs[:, 1]]]
    return np.asarray(_shape_contact_pairs_for_mask(model, mask), dtype=np.int32).reshape(-1, 2)


def _share(per_world: int | None, worlds: int, total: int) -> int | None:
    """Per-world value for a pipeline over all ``total`` worlds of the model holding ``per_world`` x ``worlds``."""
    return None if per_world is None else max(math.ceil(int(per_world) * worlds / total), 1)


class ChunkLayer:
    """The probe's collision pipeline for worlds ``[lo, hi)``: explicit pairs of those worlds, ``budget`` per world.

    Example::

        >>> layer = ChunkLayer(built.model, budget, (0, 256), first=True)  # doctest: +SKIP
        >>> layer.collide(state); int(layer.overflow.numpy()[0])          # doctest: +SKIP
        0
    """

    def __init__(self, model: Any, budget: Any, worlds: tuple[int, int], first: bool) -> None:
        import warp as wp  # noqa: PLC0415

        from alloy.coupling.proxy_contacts import ProxyContactPipeline  # noqa: PLC0415

        lo, hi = worlds
        total = max(int(model.world_count or 0), 1)
        pairs = chunk_pairs(model, lo, hi, first)
        self.pair_count = int(pairs.shape[0])
        self.factory = ProxyContactPipeline(
            None, _share(budget.rigid_contacts, hi - lo, total),
            triangle_pairs_per_world=_share(budget.triangle_pairs, hi - lo, total), label="check probe",
            pass_through=False, publish_capacity=False, broad_phase="explicit",
            shape_pairs_filtered=wp.array(pairs, dtype=wp.vec2i, device=model.device))
        self.pipeline = self.factory(model)
        self.contacts = self.pipeline.contacts()
        self.overflow = self.factory.overflow

    def collide(self, state: Any) -> None:
        self.factory.reset_overflow()
        self.pipeline.collide(state, self.contacts)
        self.factory.update_overflow()


class _Host:
    """A host array read like a Warp array (``.numpy()``)."""

    def __init__(self, a: np.ndarray) -> None:
        self.a = a

    def numpy(self) -> np.ndarray:
        return self.a


@dataclass
class HostContacts:
    """Rigid contacts of several chunks, copied to the host, read like Newton's ``Contacts`` (``.numpy()`` fields,
    ``rigid_contact_count``, ``rigid_contact_max``; no soft contacts: chunks are for rigid-only scenes)."""

    fields: dict[str, np.ndarray]
    rigid_contact_max: int
    soft_contact_max: int = 0

    def __getattr__(self, name: str) -> Any:
        if name in RIGID_FIELDS:
            return _Host(self.fields[name])
        if name == "rigid_contact_count":
            return _Host(np.array([len(self.fields[RIGID_FIELDS[0]])], np.int32))
        if name == "soft_contact_count":
            return _Host(np.zeros(1, np.int32))
        raise AttributeError(name)

    @classmethod
    def merge(cls, parts: list[HostContacts]) -> HostContacts:
        """One buffer holding every part's contacts, in order."""
        fields = {k: np.concatenate([p.fields[k] for p in parts]) if parts else np.zeros(0) for k in RIGID_FIELDS}
        return cls(fields, sum(p.rigid_contact_max for p in parts))


def collide_chunks(model: Any, budget: Any, ranges: list[tuple[int, int]], state: Any,
                   need: Any) -> tuple[HostContacts, dict[str, int], bool]:
    """Collide ``state`` one world range at a time: (the merged contacts, the need, whether any chunk lost contacts).
    ``need(layer.pipeline, layer.contacts)`` reads one chunk's counts (``alloy.check.contacts.layer_need``). The need
    is summed over the chunks; after a loss it is the worst lossy chunk's, scaled to every world, so a budget grown
    from it (per world, as the probe sizes) also holds that chunk."""
    total_worlds = max(int(model.world_count or 0), 1)
    parts, total, worst, lost = [], {}, {}, False
    for i, (lo, hi) in enumerate(ranges):
        layer = ChunkLayer(model, budget, (lo, hi), first=i == 0)
        if layer.pair_count == 0:
            continue
        layer.collide(state)
        n = need(layer.pipeline, layer.contacts)
        _add(total, n)
        if int(layer.overflow.numpy()[0]) > 0 or full(n):
            lost = True
            _worst(worst, n, total_worlds / (hi - lo))
        c = layer.contacts
        count = min(int(c.rigid_contact_count.numpy()[0]), int(c.rigid_contact_max))
        parts.append(HostContacts({k: getattr(c, k).numpy()[:count].copy() for k in RIGID_FIELDS},
                                  int(c.rigid_contact_max)))
        del layer, c                                        # free this chunk's buffers before the next
    log.info("check probe: collided %d worlds in %d chunks", total_worlds, len(ranges))
    return HostContacts.merge(parts), worst if lost else total, lost


def _add(total: dict[str, int], n: dict[str, int]) -> None:
    for k, v in n.items():
        total[k] = max(total.get(k, 0), v) if k == "reducer" else total.get(k, 0) + v


def _worst(worst: dict[str, int], n: dict[str, int], scale: float) -> None:
    """Keep the largest count and size of each buffer over the lossy chunks, scaled to every world."""
    for k, v in n.items():
        worst[k] = max(worst.get(k, 0), v if k == "reducer" else math.ceil(v * scale))


__all__ = ["ChunkLayer", "HostContacts", "chunk_pairs", "collide_chunks", "full", "halved", "plan_chunks",
           "probe_bytes"]
