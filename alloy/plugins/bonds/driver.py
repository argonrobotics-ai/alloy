# SPDX-License-Identifier: Apache-2.0
"""``GantryScript``: a companion plug-in that drives the peel rig's gantry targets from each world's clock, inside
the step. It launches kernels only, so it runs under graph capture.

Bind it after the build (``bind(sim, bonds)``): it writes ``sim.control.joint_target_q``, and the 90° modes read the
bonds' front point.

Paths, relative to the start, at speed v:
- ``follow``: z = v·t, with x above the front point. This is the 90° fixture: a sliding table keeps the arm upright.
- ``pull180``: lift to ``lift`` over ``t_lift``, then x = v·(t - t_lift) over the bonded tape. The arm folds back.
- ``slack``: as ``follow`` until ``t_turn``, then the grip comes back down at v. The arm goes slack.
- ``press``: z = -``lift`` (below the plate: the force-limited drive presses) until ``t_turn``, then z = +``lift``.
- ``roll``: as ``press`` in z, and x back and forth over ``span`` at v (a roller's passes, one each way).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.base import Forces, ForcePlugin
from alloy.plugins.errors import PluginError

MODES = {"follow": 0, "pull180": 1, "slack": 2, "press": 3, "roll": 4}


@wp.kernel
def drive(mode: int, t_now: wp.array[wp.float64], dt: float, ix: wp.array[wp.int32], iz: wp.array[wp.int32],
          v: float, lift: float, t_lift: float, t_turn: float, x0: float, span: float, front: wp.array[wp.vec3],
          target: wp.array[float]):
    w = wp.tid()
    t = float(t_now[w]) + dt                          # target for the end of this exchange
    x = float(0.0)
    z = float(0.0)
    if mode == 1:
        z = lift * wp.min(t / t_lift, 1.0)
        x = v * wp.max(t - t_lift, 0.0)
    elif mode == 3 or mode == 4:
        z = -lift
        if t > t_turn:
            z = lift
        if mode == 4:
            u = v * wp.min(t, t_turn) / span                # passes so far
            k = wp.floor(u)
            x = span * (u - k)
            if int(k) % 2 == 1:
                x = span * (1.0 - (u - k))
    else:
        z = v * t
        if mode == 2 and t > t_turn:
            z = v * (2.0 * t_turn - t)
        fx = front[w][0]
        if fx > 0.0:
            x = fx - x0
    target[ix[w]] = x
    target[iz[w]] = z


class GantryScript(ForcePlugin):
    """Drives the gantry of ``alloy.plugins.bonds.rig`` (module docstring). Applies no force."""

    name = "bonds_gantry_script"
    summary = "test companion: scripted gantry targets for the bonds peel rig"

    @dataclass
    class Config:
        mode: str = "follow"
        speed: float = 0.05                 # m/s
        lift: float = 3.0e-3                # m (pull180)
        t_lift: float = 0.1                 # s (pull180)
        t_turn: float = 0.2                 # s (slack)
        x0: float = 2.5e-3                  # m, the grip point's start x (the tab's middle)
        span: float = 0.03                  # m, one roll pass (roll)

        def __post_init__(self) -> None:
            if self.mode not in MODES:
                raise ValueError(f"mode {self.mode!r}: one of {sorted(MODES)}")

    def setup(self, ctx: Any) -> None:
        m = ctx.model
        starts, labels = m.joint_q_start.numpy(), list(m.joint_label)
        cols = [[int(starts[j]) for j, lab in enumerate(labels) if lab.endswith(f"gantry_{a}")] for a in "xz"]
        if len(cols[0]) != ctx.world_count or len(cols[1]) != ctx.world_count:
            raise PluginError("bonds_gantry_script: needs joints gantry_x and gantry_z in every world (bonds.rig)")
        self.ix, self.iz = ctx.array(np.asarray(cols[0]), wp.int32), ctx.array(np.asarray(cols[1]), wp.int32)
        self.target: wp.array | None = None
        self.front = ctx.zeros(ctx.world_count, wp.vec3)

    def bind(self, sim: Any, bonds: Any | None = None) -> None:
        self.target = sim.control.joint_target_q
        if bonds is not None:
            self.front = bonds.state_arrays()["front_point"]

    def apply(self, state: Any, forces: Forces, dt: float) -> None:
        if self.target is None:
            raise PluginError("bonds_gantry_script: call bind(sim, bonds) after building the simulation")
        c = self.config
        wp.launch(drive, dim=self.ctx.world_count,
                  inputs=[MODES[c.mode], self.ctx.time, float(dt), self.ix, self.iz, c.speed, c.lift, c.t_lift,
                          c.t_turn, c.x0, c.span, self.front], outputs=[self.target], device=self.ctx.model.device)


__all__ = ["GantryScript", "MODES"]
