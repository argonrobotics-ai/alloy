# SPDX-License-Identifier: Apache-2.0
"""Contact-driven deposition: markers lay ink on a board face and felt wipers lift it, into a per-world colour map.

One instance owns one board face (a ``removal.surface`` patch, the same per-world grid as the removal plug-in) and
any number of tools. Each tool is a body carrying a compliant tip, removal v1's pad (Winkler pressure, damping,
Coulomb friction), optionally domed (a marker nib):

- ``apply`` (before the solver): tip pressure and friction on every tool body, with the reactions on a floating board;
- ``update`` (after the solver): markers lay ``flow·p·A·dt`` into their colour channel (limited by the reservoir and
  by the board's ``saturation``); wipers lift ``1 - exp(-cover·ds/wipe_length)`` of every channel (removal v1's wipe
  law) into their sink.

The plug-in owns the tip contact, so the solver never collides the tips. Keep each tool's rigid shapes above its tip
apex (a nib's length, a felt pad's thickness). Ink is conserved exactly per world:
``Σ map + Σ reservoirs + Σ sinks`` is constant to float64 rounding.

Defaults (sources and choices in docs/notes/deposition-v1.md): a dry-erase marker with a 3 mm domed felt nib
(1e9 Pa/m, 3 g of ink, flow 1e-7 kg/(N·s): about 2 µg/mm at 2 N and 0.1 m/s) and a 25 mm felt wiper (3e6 Pa/m,
wipe length 20 mm). The board holds at most 2e-3 kg/m² of ink.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.deposition import laws
from alloy.plugins.deposition.kernels import tip_force_kernel, transfer_kernel
from alloy.plugins.deposition.tools import ToolSpec, tip_array, tool_specs
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import Law, TestScene
from alloy.plugins.removal.kernels import Grid
from alloy.plugins.deposition.stability import stability_numbers, tip_response
from alloy.plugins.removal.surface import patch_of, patch_tables

log = logging.getLogger("alloy.plugins.deposition")

STABLE = 1.0


class Deposition(ForcePlugin):
    """Markers lay ink on a board and wipers lift it: pressure-driven deposition into a per-world colour map, with
    an exact ink ledger.

    Example::

        board = Deposition(tools=("pen", "eraser"), kinds=("marker", "wiper"), board="whiteboard",
                           channels=("black", "blue"), colours=(1, 0))
        sim = alloy.plugins.simulation(built, [board])
        ...; r = sim.plugins["deposition"].read(); r["ink"][0]       # [nx, ny, channels] kg per cell, world 0
    """

    name = "deposition"
    summary = "markers lay ink, wipers lift it: contact-pressure deposition into a per-world colour map, exact ledger"

    @dataclass
    class Config:
        tools: tuple[str, ...] = ("marker", "wiper")             # body labels of the tools
        kinds: tuple[str, ...] = ("marker", "wiper")             # per tool: "marker" or "wiper"
        colours: tuple[int, ...] = (0,)                          # per marker: ink channel index
        channels: tuple[str, ...] = ("black",)                   # names of the ink channels
        tip_center: tuple[Any, ...] = (0.0, 0.0, -0.015)         # tip apex in the tool frame [m] (one, or per tool)
        tip_normal: tuple[Any, ...] = (0.0, 0.0, -1.0)           # tip face normal, out of the tip (one, or per tool)
        tip_radius: tuple[float, ...] = (0.003, 0.025)           # m: nib / pad radius
        dome_radius: tuple[float, ...] = (0.003, 0.0)            # m: nib dome radius; 0: flat face
        stiffness: tuple[float, ...] = (1.0e9, 3.0e6)            # Pa/m: pressure per metre of indentation
        damping: tuple[float, ...] = (1.0e-3, 2.0e-3)            # s: Kelvin-Voigt time
        friction: tuple[float, ...] = (0.3, 0.5)                 # Coulomb coefficient on the board
        flow: tuple[float, ...] = (1.0e-7, 0.0)                  # marker: kg of ink per N of contact force per s
        reservoir: tuple[float, ...] = (3.0e-3, 0.0)             # marker: kg of ink at the start
        wipe_length: tuple[float, ...] = (0.0, 0.02)             # wiper: m of slide per e-fold of the ink
        slip_eps: float = 1.0e-3                                 # m/s: friction is viscous below this slip speed
        board: str = "board"                                     # shape label of the board
        size: tuple[float, float] | None = None                  # face size in the shape's local x, y; None: box
        surface_offset: float | None = None                      # face height in the shape frame; None: box top
        cells: tuple[int, int] = (160, 120)                      # map resolution along local x, y (see below)
        saturation: float = 2.0e-3                               # kg/m^2 the board holds at most; 0: no limit
        # Resolution rule (WP2 review F1): contact is sampled at cell centres, so a cell may not be larger than any
        # tip's radius (refused), and a domed nib's light-touch patch, sqrt(2·R·0.1·R) ≈ 0.45·R, should span a
        # cell (warned): a 3 mm nib wants cells of about 1.3 mm or less.

        def __post_init__(self) -> None:
            if not self.channels:
                raise ValueError("channels must name at least one ink channel")
            if min(self.cells) < 1:
                raise ValueError(f"cells must be >= 1, got {self.cells}")
            if not (math.isfinite(self.saturation) and self.saturation >= 0.0):
                raise ValueError(f"saturation must be finite and >= 0, got {self.saturation}")
            if not (math.isfinite(self.slip_eps) and self.slip_eps >= 0.0):
                raise ValueError(f"slip_eps must be finite and >= 0, got {self.slip_eps}")
            tool_specs(self)                                     # per-tool checks

    def setup(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        self.specs: list[ToolSpec] = tool_specs(c)
        W, T = ctx.world_count, len(self.specs)
        patches = [patch_of(model, int(s), c.board, c.size, c.surface_offset) for s in ctx.shapes(c.board).numpy()]
        self.face_body, self.face_local = patch_tables(patches, ctx.device)
        bodies = np.stack([ctx.bodies(s.label).numpy() for s in self.specs], axis=1).astype(np.int32)   # [W, T]
        self.tools = ctx.array(bodies, wp.int32)
        self.tips = tip_array(self.specs, c.slip_eps, ctx.device)
        self.grid = Grid()
        self.grid.half = wp.vec2(0.5 * patches[0].size[0], 0.5 * patches[0].size[1])
        self.grid.cell = wp.vec2(patches[0].size[0] / c.cells[0], patches[0].size[1] / c.cells[1])
        self.grid.feedback = 0
        _check_resolution(self.specs, self.grid, c.board)
        self.response = tip_response(model, bodies, patches, self.specs)
        self.mobility = ctx.array(self.response.friction_mobility(), wp.float32)
        start = np.tile(np.asarray([s.reservoir for s in self.specs], np.float64), (W, 1))
        self.ink = self.add_state("ink", ctx.zeros((W, *c.cells, len(c.channels)), wp.float32))
        self.reservoir = self.add_state("reservoir", ctx.array(start, wp.float64))
        self.laid = self.add_state("laid", ctx.zeros((W, T), wp.float64))
        self.sink = self.add_state("sink", ctx.zeros((W, T), wp.float64))
        self.load = self.add_state("load", ctx.zeros((W, T), wp.float32))
        log.info("deposition %s: %d tools %s on %r, map %s x %d channels, cell %.3g mm", self.label, T,
                 [s.label for s in self.specs], c.board, c.cells, len(c.channels), 1e3 * float(self.grid.cell[0]))

    def check_dt(self, dt: float) -> None:
        """Each tip's spring and damper (normal + tilt, whole flat face pressed), with every tool's load on a floating
        board added (``stability.py``), on the worst world. The kernel bounds the friction."""
        bad = {}
        for label, (spring, damping) in stability_numbers(self.response, self.specs, dt).items():
            for name, value in ((f"{label} tip spring", spring), (f"{label} tip damping", damping)):
                if value >= STABLE:
                    bad[name] = round(value, 3)
        if bad:
            raise PluginError(f"{self.label}: unstable at dt={dt:.3g} s: {bad} (each must be < {STABLE}); lower that "
                              f"tool's stiffness or damping, shrink the exchange step, or give the tool (and a floating "
                              f"board) more mass")

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        c, m = self.config, self.ctx.model
        wp.launch(tip_force_kernel, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, m.body_com, self.tools, self.face_body, self.face_local,
                          self.tips, self.grid, int(c.cells[0]), int(c.cells[1]), float(dt), self.mobility],
                  outputs=[forces.body_f], device=self.ctx.device)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        wp.launch(transfer_kernel, dim=self.ctx.world_count,
                  inputs=[state_in.body_q, state_in.body_qd, state_out.body_q, self.ctx.model.body_com, self.tools,
                          self.face_body, self.face_local, self.tips, self.grid, float(dt),
                          float(self.config.saturation)],
                  outputs=[self.ink, self.reservoir, self.laid, self.sink, self.load], device=self.ctx.device)

    # --- host side (between periods) ---------------------------------------------------------------------------

    def set_ink(self, mass: Any) -> None:
        """Ink already on the board [kg per cell]: ``[nx, ny, C]`` or ``[W, nx, ny, C]``. Host side, between periods;
        the simulation's ``reset`` returns to the built state (a clean board)."""
        src = np.broadcast_to(np.asarray(mass, np.float32), self.ink.shape)
        if not (np.isfinite(src).all() and (src >= 0.0).all()):
            raise PluginError(f"{self.label}: ink must be finite and >= 0")
        self.ink.assign(np.ascontiguousarray(src))

    @property
    def cell_area(self) -> float:
        return float(self.grid.cell[0]) * float(self.grid.cell[1])

    def tool_index(self, label: str) -> int:
        names = [s.label for s in self.specs]
        if label not in names:
            raise PluginError(f"{self.label}: no tool {label!r}; tools: {names}")
        return names.index(label)

    def read(self) -> dict[str, np.ndarray]:
        out = super().read()
        ink = out["ink"].astype(np.float64)
        out["by_channel"] = ink.sum(axis=(1, 2))                              # [W, C] kg
        out["ink_total"] = out["by_channel"].sum(axis=1)                      # [W] kg on the board
        out["ledger"] = out["ink_total"] + out["reservoir"].sum(axis=1) + out["sink"].sum(axis=1)
        return out

    def test_scene(self) -> TestScene:
        return laws.marker_and_wiper_scene(self.config)

    def laws(self) -> list[Law]:
        return laws.deposition_laws(self.config)


def _check_resolution(specs: list[ToolSpec], grid: Grid, board: str) -> None:
    """WP2 review F1: refuse cells larger than a tip's radius; warn when a domed nib's light-touch patch is smaller
    than a cell (its force then steps as the nib crosses cell centres)."""
    cell = max(float(grid.cell[0]), float(grid.cell[1]))
    for s in specs:
        if cell > s.radius:
            raise PluginError(f"board {board!r}: cells of {cell * 1e3:.3g} mm are larger than tool {s.label!r}'s tip "
                              f"radius {s.radius * 1e3:.3g} mm, so its contact can fall between cell centres; raise "
                              f"cells to at least {cell / s.radius:.2g}x the current count")
        light = math.sqrt(0.2) * s.dome_radius
        if s.dome_radius > 0.0 and light < cell:
            log.warning("board %r: cells of %.3g mm are larger than tool %r's light-touch patch (%.3g mm radius): its "
                        "contact force steps as it crosses cells; use cells <= %.3g mm", board, cell * 1e3, s.label,
                        light * 1e3, light * 1e3)


__all__ = ["Deposition"]
