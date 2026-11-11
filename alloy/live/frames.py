# SPDX-License-Identifier: Apache-2.0
"""The picked worlds' 3D frames, from the engine's frame export (``alloy.export.FrameExporter``, SCR-WP13-EXPORT).

The training thread calls :meth:`PickedFrames.capture` (the exporter's only GPU work: small gathers and one async
copy, refused while a CUDA graph is being captured). The link's sender thread calls the packer, which polls the
exporter (no CUDA call) and packs the frames for the hub. One exporter serves one set of picked worlds; a new pick
set gets a new exporter (picks are sticky, so this is rare), and each world's first frame from it carries its
topology (names, shapes, statics, mesh faces), which the hub caches for the viewer.

Wire form: ``{"frames": [...]}`` where every numpy array inside a frame (poses, mesh vertices, particles, rod lines,
topology faces and indices) is replaced by ``{"$a": key}`` and packed with :func:`alloy.live.wire.pack_arrays`;
:func:`inflate` reverses it (arrays to lists, for JSON viewers).

Example::

    frames = PickedFrames(env.sim)
    frames.capture([3, 17])                          # training thread, after env.step
    link.offer(Kind.STATE, {"step": k}, frames.packer())
"""
from __future__ import annotations

import dataclasses
import logging
import weakref
from typing import Any

import numpy as np

from alloy.live.wire import pack_arrays

log = logging.getLogger(__name__)


RETIRE_AFTER = 8          # captures a replaced exporter is kept before the training thread frees it (~0.7 s at 12 Hz)


class PickedFrames:
    """Owns the exporter in the training thread. Packers hold only a weak reference, and a replaced exporter is kept
    for ``RETIRE_AFTER`` captures, so its pinned and device buffers are always freed here, never on the sender thread
    (a free there could land inside the engine's CUDA graph capture; WP13 review)."""

    def __init__(self, sim: Any, max_points: int = 20000, max_worlds: int = 8) -> None:
        self.sim, self.max_points, self.max_worlds = sim, max_points, max_worlds
        self._exp: Any = None
        self._worlds: tuple[int, ...] = ()
        self._retired: list[tuple[int, Any]] = []
        self._captures = 0

    @property
    def stats(self) -> dict[str, int]:
        e = self._exp
        return {"refused": e.refused, "dropped": e.dropped} if e is not None else {"refused": 0, "dropped": 0}

    def reset(self) -> None:
        """A hub reconnect: every picked world's next frame carries its topology again (no CUDA call)."""
        if self._exp is not None:
            self._exp.resend_topology()

    def _retire(self) -> None:
        if self._exp is not None:
            self._retired.append((self._captures, self._exp))
        self._exp, self._worlds = None, ()

    def capture(self, worlds: list[int]) -> bool:
        """Training thread: (re)build the exporter for these worlds if they changed, then capture."""
        from alloy.export import FrameExporter  # noqa: PLC0415

        self._captures += 1
        self._retired = [(n, e) for n, e in self._retired if self._captures - n < RETIRE_AFTER]   # frees here
        key = tuple(sorted(worlds))[: self.max_worlds]
        if not key:
            return False
        if key != self._worlds:
            self._retire()
            self._exp = FrameExporter(self.sim, worlds=list(key), max_points=self.max_points)
            self._worlds = key
            log.info("live: frame export for worlds %s", list(key))
        return bool(self._exp.capture())

    def packer(self) -> Any:
        """A callable for the sender thread: ``(header, payload)`` of the frames that landed, or None. The frames are
        from an earlier capture (about one capture interval old): each carries its own ``t``, ``period`` and
        ``topology_version``."""
        if self._exp is None:
            return None
        ref = weakref.ref(self._exp)

        def pack() -> tuple[dict[str, Any], bytes] | None:
            exp = ref()
            if exp is None:
                return None
            frames = exp.poll()
            if not frames:
                return None
            arrays: dict[str, np.ndarray] = {}
            meta = [_flatten(_as_dict(f), f"f{i}", arrays) for i, f in enumerate(frames)]
            packed, blob = pack_arrays(arrays)
            return {"frames": meta, **packed}, blob
        return pack


def _as_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _as_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    return obj


def _flatten(obj: Any, key: str, arrays: dict[str, np.ndarray]) -> Any:
    """Replace numpy arrays by {"$a": key} references (stored in ``arrays``); keep everything else JSON-able."""
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.float64:
            obj = obj.astype(np.float32)
        arrays[key] = obj
        return {"$a": key}
    if isinstance(obj, dict):
        return {str(k): _flatten(v, f"{key}.{k}", arrays) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_flatten(v, f"{key}.{i}", arrays) for i, v in enumerate(obj)]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def inflate(obj: Any, arrays: dict[str, np.ndarray], digits: int = 5) -> Any:
    """Reverse of the wire form: {"$a": key} -> nested lists (floats rounded to ``digits``)."""
    if isinstance(obj, dict):
        if set(obj) == {"$a"}:
            a = arrays[obj["$a"]]
            return np.round(a.astype(np.float64), digits).tolist() if a.dtype.kind == "f" else a.tolist()
        return {k: inflate(v, arrays, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [inflate(v, arrays, digits) for v in obj]
    return obj
