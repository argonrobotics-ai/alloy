# SPDX-License-Identifier: Apache-2.0
"""Per-world placement of particles for the Newton viewers.

A Newton viewer spreads the worlds of a model over a grid by shifting each world's rigid shapes, and hides the shapes
of worlds that are not shown. Particles (cloth, soft-body and film vertices, liquid, sand) get neither: every world's
particles would be drawn where the simulation has them, on top of each other and away from their own world's shapes.
``ParticlePlacer`` gives the viewer a state whose ``particle_q`` is a shifted copy: each shown world's particles moved
by that world's offset, and the particles of hidden worlds moved far below the scene. The simulation's own state is
never written.

This module imports Warp: import it only when a model with particles is drawn.

Example::

    placer = ParticlePlacer(model)
    backend.log_state(placer.state(backend, sim.engine.state, shown=[0, 3]))
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import warp as wp

HIDDEN_DROP_M = 1.0e4          # particles of a hidden world are drawn this far below the scene


@wp.kernel
def place_particles(q: wp.array(dtype=wp.vec3), world: wp.array(dtype=wp.int32), shift: wp.array(dtype=wp.vec3),
                    out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    w = world[i]
    if w < 0:
        out[i] = q[i]
    else:
        out[i] = q[i] + shift[w]


class StateView:
    """A Newton state with another ``particle_q``; every other attribute reads through to the real state."""

    def __init__(self, state: Any, particle_q: Any) -> None:
        self._state = state
        self.particle_q = particle_q

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state, name)


def world_shifts(world_count: int, offsets: np.ndarray | None, shown: Sequence[int], up_axis: int) -> np.ndarray:
    """(world_count, 3) float32 shifts: a shown world's viewer offset, a hidden world's drop below the scene."""
    shift = np.zeros((world_count, 3), np.float32)
    if offsets is not None:
        rows = min(world_count, len(offsets))
        shift[:rows] = np.asarray(offsets, np.float32).reshape(-1, 3)[:rows]
    hidden = np.ones(world_count, bool)
    hidden[[w for w in shown if 0 <= w < world_count]] = False
    shift[hidden] = 0.0
    shift[hidden, up_axis] = -HIDDEN_DROP_M
    return shift


def has_loose_particles(model: Any) -> bool:
    """True when some particles belong to no triangle or tetrahedron (liquid, sand): only spheres can show them."""
    count = int(model.particle_count)
    if count == 0:
        return False
    used = np.zeros(count, bool)
    for indices in (getattr(model, "tri_indices", None), getattr(model, "tet_indices", None)):
        if indices is not None and len(indices):
            used[np.asarray(indices.numpy()).reshape(-1)] = True
    return not bool(used.all())


class ParticlePlacer:
    """Makes the shifted particle state for one model; reuses one device buffer across frames."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.count = int(model.particle_count)
        self.worlds = int(model.world_count)
        self.up_axis = int(model.up_axis)
        self._out = wp.empty(self.count, dtype=wp.vec3, device=model.device)
        self._shift: Any = None
        self._key: tuple[Any, ...] | None = None
        self._offsets: Any = None

    def needed(self) -> bool:
        """False for a model whose particles need no placement (one world, or particles of no world)."""
        return self.count > 0 and self.worlds > 1 and self.model.particle_world is not None

    def state(self, backend: Any, state: Any, shown: Sequence[int]) -> Any:
        """``state`` for drawing: a view with placed particles, or ``state`` itself when nothing needs moving."""
        if not self.needed():
            return state
        shift = self._shifts(getattr(backend, "world_offsets", None), shown)
        wp.launch(place_particles, dim=self.count, inputs=[state.particle_q, self.model.particle_world, shift],
                  outputs=[self._out], device=self.model.device)
        return StateView(state, self._out)

    def _shifts(self, offsets: Any, shown: Sequence[int]) -> Any:
        key = (id(offsets), tuple(shown))
        if key != self._key:
            host = None if offsets is None else np.array(offsets.numpy(), copy=True)
            table = world_shifts(self.worlds, host, shown, self.up_axis)
            self._shift = wp.array(table, dtype=wp.vec3, device=self.model.device)
            self._key = key
            self._offsets = offsets      # held so its id is not reused by a later offsets array
        return self._shift


__all__ = ["HIDDEN_DROP_M", "ParticlePlacer", "StateView", "has_loose_particles", "world_shifts"]
