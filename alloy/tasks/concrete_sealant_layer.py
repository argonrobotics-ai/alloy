# SPDX-License-Identifier: Apache-2.0
"""tool-sealant's companion layer: wall contact and the work age (Gym-Anything ``concrete_finishing/coupling.py:
1070-1075, 1085``), read off the concrete process layer (``concrete_paste``, material "sealant") and the film.

Listed after ``concrete_paste`` in the plug-in list; each exchange, per world:

- **Wall contact.** When the working tool's pad pressure exceeds twice the median surface yield ``tau_s`` over the
  cells it works this exchange (``bear``: touching and advancing), every such cell on a wall's edge (within 1.5 mm
  of it, the start port's masks) that holds over 0.1 mm of sealant and is younger than the skin is wetted against
  that wall (``contact_l`` / ``contact_r`` = 1, for good: after the skin no new contact is made).
- **Work age.** ``seal_age`` = the cell's age when last worked.

It writes only its own states; ``concrete_paste`` owns skin_torn and texture (WP6). Tack is not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError

MAX_BEAR = 2048                         # bearing cells per world for the median (a nose covers ~300 1 mm cells)
WET_T = 1.0e-4                          # m of sealant an edge cell needs to be wetted (coupling.py:1074)
WET_FACTOR = 2.0                        # pressure over the median tau_s that wets the walls (coupling.py:1071)


@wp.kernel
def _gather_kernel(bear: wp.array2d[wp.int32], tau_s: wp.array2d[wp.float32], count: wp.array[wp.int32],
                   buf: wp.array2d[wp.float32]):
    w, c = wp.tid()
    if bear[w, c] == 0:
        return
    k = wp.atomic_add(count, w, 1)
    if k < MAX_BEAR:
        buf[w, k] = tau_s[w, c]


@wp.kernel
def _median_kernel(count: wp.array[wp.int32], buf: wp.array2d[wp.float32], median: wp.array[wp.float32]):
    """Per world: the median of the gathered values (numpy's: the mean of the two middle ones), by insertion sort;
    -1 when nothing bears."""
    w = wp.tid()
    m = wp.min(count[w], MAX_BEAR)
    if m == 0:
        median[w] = -1.0
        return
    for i in range(1, m):
        v = buf[w, i]
        j = i - 1
        while j >= 0 and buf[w, j] > v:
            buf[w, j + 1] = buf[w, j]
            j = j - 1
        buf[w, j + 1] = v
    median[w] = 0.5 * (buf[w, (m - 1) / 2] + buf[w, m / 2])


@wp.kernel
def _contact_kernel(bear: wp.array2d[wp.int32], age: wp.array2d[wp.float64], h: wp.array2d[wp.float64],
                    pressure: wp.array[wp.float32], median: wp.array[wp.float32], edge_l: wp.array2d[wp.int32],
                    edge_r: wp.array2d[wp.int32], skin: wp.float64, contact_l: wp.array2d[wp.int32],
                    contact_r: wp.array2d[wp.int32], seal_age: wp.array2d[wp.float64]):
    w, c = wp.tid()
    if bear[w, c] == 0:
        return
    seal_age[w, c] = age[w, c]
    if median[w] < 0.0 or pressure[w] <= WET_FACTOR * median[w]:
        return
    if h[w, c] > wp.float64(WET_T) and age[w, c] < skin:
        if edge_l[w, c] != 0:
            contact_l[w, c] = 1
        if edge_r[w, c] != 0:
            contact_r[w, c] = 1


class SealantContact(ForcePlugin):
    """Wall contact and the work age for a sealant joint worked under ``concrete_paste`` (``paste=``) on ``flow=``."""

    name = "sealant_contact"
    summary = "sealant joint: wall contact made by tooling before the skin, and the age at last working"

    @dataclass
    class Config:
        skin: float = 7200.0                # s: the sealant's skin age (concrete_materials.Sealant.skin_time)

        def __post_init__(self) -> None:
            if not (np.isfinite(self.skin) and self.skin > 0.0):
                raise ValueError(f"skin must be finite and > 0, got {self.skin}")

    def __init__(self, config: Any = None, *, paste: Any = None, flow: Any = None, label: str | None = None,
                 **params: Any) -> None:
        super().__init__(config, label=label, **params)
        if paste is None or flow is None:
            raise PluginError("sealant_contact: pass paste= (the concrete_paste layer it reads) and flow=")
        self.paste, self.flow = paste, flow

    def setup(self, ctx: Any) -> None:
        W, C = ctx.world_count, self.flow.tables.cell_count
        self.contact_l = self.add_state("contact_l", ctx.zeros((W, C), wp.int32))
        self.contact_r = self.add_state("contact_r", ctx.zeros((W, C), wp.int32))
        self.seal_age = self.add_state("seal_age", ctx.zeros((W, C), wp.float64))
        self.edge_l = self.add_input("edge_l", ctx.zeros((W, C), wp.int32))
        self.edge_r = self.add_input("edge_r", ctx.zeros((W, C), wp.int32))
        self._count = ctx.zeros(W, wp.int32)
        self._buf = ctx.zeros((W, MAX_BEAR), wp.float32)
        self._median = ctx.zeros(W, wp.float32)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        ctx, p = self.ctx, self.paste
        W, C = ctx.world_count, self.flow.tables.cell_count
        self._count.zero_()
        wp.launch(_gather_kernel, dim=(W, C), inputs=[p.f.bear, p.f.tau_s], outputs=[self._count, self._buf],
                  device=ctx.device)
        wp.launch(_median_kernel, dim=W, inputs=[self._count, self._buf], outputs=[self._median], device=ctx.device)
        wp.launch(_contact_kernel, dim=(W, C),
                  inputs=[p.f.bear, p.f.age, self.flow.h, p.pressure, self._median, self.edge_l, self.edge_r,
                          wp.float64(self.config.skin)],
                  outputs=[self.contact_l, self.contact_r, self.seal_age], device=ctx.device)

    def set_start(self, *, contact_l: Any = None, contact_r: Any = None, edge_l: Any = None,
                  edge_r: Any = None) -> None:
        """Between periods: the walls' contact at the start and the edge masks ([C] or [W, C])."""
        for name, values in (("edge_l", edge_l), ("edge_r", edge_r)):
            if values is not None:
                self.set_input(name, np.asarray(values, np.int32))
        for arr, values in ((self.contact_l, contact_l), (self.contact_r, contact_r)):
            if values is not None:
                arr.assign(np.broadcast_to(np.asarray(values, np.int32), arr.shape).copy())


__all__ = ["SealantContact"]
