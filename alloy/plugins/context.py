# SPDX-License-Identifier: Apache-2.0
"""What a plug-in sees at setup and while stepping: the model, label lookups and a device clock.

Bodies, shapes and joints are found by their Newton label (``ModelBuilder.add_body(label=...)``). A scene holds the
same template in every world, so a lookup returns one global index per world: an ``int32`` array of length
``world_count`` that kernels index with the world id.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.errors import PluginError

log = logging.getLogger("alloy.plugins")


def _matches(labels: Sequence[str], label: str) -> np.ndarray:
    """Indices whose label is ``label`` or ends with ``/label`` (imported files prefix their own names)."""
    exact = [i for i, x in enumerate(labels) if x == label]
    if exact:
        return np.asarray(exact, np.int64)
    return np.asarray([i for i, x in enumerate(labels) if str(x).endswith(f"/{label}")], np.int64)


@dataclass
class PluginContext:
    """Model access for plug-ins. Built by ``PluginHost``; read it in ``setup``.

    Example::

        def setup(self, ctx):
            self.pen = ctx.bodies("pen")                       # wp.array int32 [W]: the pen of each world
            self.ink = self.add_state("ink", ctx.zeros((ctx.world_count, 64, 64)))
    """

    model: newton.Model
    control_dt: float
    contacts: Any = None                     # newton.Contacts of the current step, or None (solver-native contacts)
    contact_force: Any = None                # wp.array[spatial_vector]: per contact of ``contacts``, the force on
                                             # shape0 in world frame; set when a plug-in has ``needs_contacts``
    names: Any = None                        # the Scene's names (alloy.api.names.NameIndex) under alloy.build, so
                                             # bodies("arm/finger") / joints() take Scene names too; None: raw labels
    time: wp.array = field(init=False)       # [W] float64 time [s] since the world's last reset, advanced every step

    def __post_init__(self) -> None:
        self.time = wp.zeros(int(self.model.world_count), dtype=wp.float64, device=self.model.device)

    @property
    def world_count(self) -> int:
        """Number of parallel worlds in the model."""
        return int(self.model.world_count)

    @property
    def device(self) -> Any:
        """The Warp device every array must live on."""
        return self.model.device

    def zeros(self, shape: int | tuple[int, ...], dtype: Any = wp.float32) -> wp.array:
        """A zero array on the model's device. Example: ``ctx.zeros((ctx.world_count, 3), wp.float64)``."""
        return wp.zeros(shape, dtype=dtype, device=self.device)

    def full(self, shape: int | tuple[int, ...], value: Any, dtype: Any = wp.float32) -> wp.array:
        """An array filled with ``value``. Example: ``ctx.full(ctx.world_count, 100.0)`` (a full reservoir)."""
        return wp.full(shape, value, dtype=dtype, device=self.device)

    @property
    def gravity(self) -> np.ndarray:
        """World 0's gravity vector [m/s²], host ``(3,)`` float64 (``model.gravity`` holds one row per world)."""
        return np.asarray(self.model.gravity.numpy(), np.float64).reshape(-1, 3)[0]

    def array(self, values: Any, dtype: Any = wp.float32) -> wp.array:
        """A device copy of host ``values``. Example: ``ctx.array([0.0, 0.0, 1.0], wp.vec3)``."""
        return wp.array(np.asarray(values), dtype=dtype, device=self.device)

    def bodies(self, label: str) -> wp.array:
        """Global body index of ``label`` in every world, ``int32 [W]``. Example: ``ctx.bodies("magnet_a")``."""
        return self._per_world("body", label)

    def shapes(self, label: str) -> wp.array:
        """Global shape index of ``label`` in every world (static shapes too), ``int32 [W]``."""
        return self._per_world("shape", label)

    def joints(self, label: str) -> wp.array:
        """Global joint index of ``label`` in every world, ``int32 [W]``."""
        return self._per_world("joint", label)

    def labels(self, kind: str = "body", world: int = 0) -> list[str]:
        """Labels of every ``kind`` ("body", "shape", "joint") in ``world``; use it to find names."""
        names = list(getattr(self.model, f"{kind}_label"))
        where = getattr(self.model, f"{kind}_world").numpy()
        return [names[i] for i in range(len(names)) if int(where[i]) == world]

    def has(self, label: str, kind: str = "body") -> bool:
        """True if some ``kind`` item carries ``label``."""
        return _matches(list(getattr(self.model, f"{kind}_label")), label).size > 0

    def _per_world(self, kind: str, label: str) -> wp.array:
        scene = self._scene_name(kind, label)
        if scene is not None:
            return wp.array(scene.astype(np.int32), dtype=wp.int32, device=self.device)
        return self._raw_per_world(kind, label)

    def _scene_name(self, kind: str, label: str) -> np.ndarray | None:
        """Per-world indices of a Scene name ("<robot>/<part>", an object name), or None when ``label`` is not one.
        A label that is a Scene name and also a raw label of other items raises (never a silent pick)."""
        table = {"body": "bodies", "joint": "joints"}.get(kind)
        names = getattr(self.names, table, None) if table and self.names is not None else None
        if not names or label not in names:
            return None
        out = np.asarray(self.names.per_world(names[label], kind), np.int64)
        raw = _matches(list(getattr(self.model, f"{kind}_label")), label)
        if raw.size:
            try:
                same = np.array_equal(self._raw_per_world(kind, label).numpy(), out)
            except PluginError:
                same = False
            if not same:
                lab = list(getattr(self.model, f"{kind}_label"))
                raise PluginError(f"{kind} {label!r} is the Scene name of {kind} {int(out[0])} "
                                  f"({lab[int(out[0])]!r}) and also the raw label of {[lab[int(i)] for i in raw[:4]]}:"
                                  f" rename the scene object, or pass the full raw label of the one you mean")
        return out

    def _raw_per_world(self, kind: str, label: str) -> wp.array:
        idx = _matches(list(getattr(self.model, f"{kind}_label")), label)
        where = getattr(self.model, f"{kind}_world").numpy()
        out = np.full(self.world_count, -1, np.int64)
        for i in idx:
            w = int(where[i])
            if w < 0:
                continue                      # global items (ground plane) belong to no world
            if out[w] >= 0:
                raise PluginError(f"{kind} label {label!r} appears twice in world {w}: give each {kind} a unique "
                                  f"label in the world template, or use its Scene name{self._scene_hint(kind)}")
            out[w] = i
        if (out < 0).any():
            raise PluginError(f"no {kind} labelled {label!r} in world {int(np.argmax(out < 0))}; {kind} labels in "
                              f"world 0: {self.labels(kind)[:30]}{self._scene_hint(kind)}. Pass one of them, or "
                              f"label= when you add it to the builder")
        return wp.array(out.astype(np.int32), dtype=wp.int32, device=self.device)

    def _scene_hint(self, kind: str) -> str:
        table = {"body": "bodies", "joint": "joints"}.get(kind)
        names = getattr(self.names, table, None) if table and self.names is not None else None
        return f"; Scene names: {sorted(names)[:30]}" if names else ""


__all__ = ["PluginContext"]
