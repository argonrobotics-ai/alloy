# SPDX-License-Identifier: Apache-2.0
"""The edger's lip mark: Gym-Anything's ``compact`` on the cells an edger's lip runs down (``coupling.py:1076-1080``).

Their edger marks, on every working step, the cells under its sole where the sole lies within half the lip's depth
of its lowest point: the 1.5 mm lip that cuts the concrete from the form. pad-edge's ``lip_passed`` (T:303) and
pad-finish's read that mark within 2.5 mm of the forms. Here, per exchange, one thread per cell: a paste cell is
marked while the edger works it, as their sole work (``coupling.py:641-642, 1003-1006``) and ``concrete_paste``'s:
the tool covers the cell, its underside there (``flow.tf.ceil``) is within ``BEAR_BAND`` of the paste's surface, and
it slides (its low-passed advance over ``MIN_ADVANCE``); and there the underside lies within half the lip's depth of
the sole's lowest point, the lower of the lip's two bottom corners at the tool's pose (pitched or level). A layer
listed after ``surface_flow`` and ``concrete_paste`` that reads their state and never writes it (WP15's layer
rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.tasks.concrete_paste import BEAR_BAND, MIN_ADVANCE


@wp.kernel(enable_backward=False)
def _lip_kernel(paste: wp.array2d[wp.int32], h: wp.array2d[wp.float64], base: wp.array2d[wp.float32],
                ceil: wp.array2d[wp.float32], cover: wp.array2d[wp.float32], advance: wp.array2d[wp.vec3],
                body_q: wp.array[wp.transform], tool: wp.array2d[wp.int32], k: int, edge: wp.vec3, depth: float,
                lip: wp.array2d[wp.float32]):
    w, c = wp.tid()
    if paste[w, c] == 0 or cover[w, c] <= 0.0:
        return
    if float(ceil[w, c]) > float(base[w, c]) + float(h[w, c]) + BEAR_BAND:
        return                                          # the sole is above the paste here: no contact
    if wp.length(advance[w, k]) <= MIN_ADVANCE:
        return                                          # not sliding: no work (concrete_paste's rule)
    X = body_q[tool[w, k]]
    a = wp.transform_point(X, wp.vec3(edge[0], edge[1], edge[2] - depth))       # the lip's two bottom corners
    b = wp.transform_point(X, wp.vec3(-edge[0], edge[1], edge[2] - depth))
    if ceil[w, c] < wp.min(a[2], b[2]) + 0.5 * depth:                           # within lip/2 of the lowest point
        lip[w, c] = 1.0


class EdgerLip(ForcePlugin):
    """Marks the lip's cells (state ``lip`` [W, C], 0 or 1) under the carriage's ``edger`` program."""

    name = "edger_lip"
    summary = "the edger's lip mark (Gym-Anything compact) on the paste cells its lip cuts"

    @dataclass
    class Config:
        body: str = "edger"                 # the carriage program of the edger
        half: tuple[float, float, float] = (0.076, 0.0396, 0.001)   # m: sole half extents (the lip's face at +y),
                                                                    # its flat this far below the body origin
        lip: float = 0.0127                 # m: the lip's depth below the flat

    def __init__(self, config: Any = None, *, flow: Any = None, carriage: Any = None, paste: Any = None,
                 label: str | None = None, **params: Any) -> None:
        super().__init__(config, label=label, **params)
        if flow is None or carriage is None or paste is None:
            raise PluginError("edger_lip: pass flow= (SurfaceFlow), carriage= (ToolCarriage), paste= (ConcretePaste)")
        self.flow, self.carriage, self.layer = flow, carriage, paste

    def setup(self, ctx: Any) -> None:
        names = [p.body for p in self.carriage.config.programs]
        if self.config.body not in names:
            raise PluginError(f"{self.label}: no carriage program {self.config.body!r} (has {names})")
        self.k = names.index(self.config.body)
        self.lip = self.add_state("lip", ctx.zeros((ctx.world_count, self.flow.tables.cell_count), wp.float32))

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        ctx, c = self.ctx, self.config
        wp.launch(_lip_kernel, dim=self.lip.shape,
                  inputs=[self.layer.paste, self.flow.h, self.flow.base, self.flow.tf.ceil, self.flow.tf.cover,
                          self.layer.advance, state_out.body_q, self.carriage.tool, self.k,
                          wp.vec3(c.half[0], c.half[1], -c.half[2]), float(c.lip)], outputs=[self.lip],
                  device=ctx.device)

    def marks(self) -> np.ndarray:
        """Host: the mark [W, C]."""
        return self.lip.numpy()
