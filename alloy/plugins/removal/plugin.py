# SPDX-License-Identifier: Apache-2.0
"""Material removal by contact: sanding, polishing, deburring and wiping a workpiece face with a compliant pad.

A tool body carries a circular pad face. The workpiece face is a per-world grid (``surface.py``, generalized from the
``ink`` plug-in). Every step the plug-in:

- ``apply`` (before the solver): pad pressure and friction on the tool, reaction on the workpiece body if it has
  one. The pad is a compliant layer the plug-in owns (``kernels.py``), so the solver never collides the pad face;
  keep the tool's rigid shapes above the pad face (a sanding pad's foam thickness) so they only touch when the pad
  bottoms out.
- ``update`` (after the solver): Archard wear under the pad, ``dh = (K / H)·p·ds`` per cell, into ``depth``. The
  removed mass goes to ``loose`` (left on the part) or ``sink`` (captured by the tool); a wiping pad lifts loose
  material into ``sink``.

Pressure and slip come from geometry and the pre-solve state, so the plug-in works with every solver and its hooks are
pure functions of the state. ``feedback`` lowers the contact surface by the removed depth (and raises it by heights
set with ``set_profile``, e.g. burrs). Mass is conserved per world to float64 rounding:
``removed + loose(t=0) = loose + sink``.

Defaults (sources and choices in docs/notes/removal-v1.md): a foam sanding pad (E about 0.3 MPa over 10 mm:
3e7 Pa/m), two-body abrasive wear K = 1e-2 (Rabinowicz; Hutchings), an aluminium workpiece (H = 1 GPa,
2700 kg/m^3).
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
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import Law, TestScene
from alloy.plugins.removal import laws
from alloy.plugins.removal.kernels import Grid, Pad, Wear, pad_force_kernel, wear_kernel
from alloy.plugins.removal.stability import pad_response
from alloy.plugins.removal.surface import patch_of, patch_tables

log = logging.getLogger("alloy.plugins.removal")

STABLE = 1.0   # explicit pad springs and dampers need omega·dt and c·dt/m below this (GUIDE section 5)


class Removal(ForcePlugin):
    """Sanding, polishing, deburring or wiping: a compliant pad on a tool body removes material from a workpiece face
    (Archard / Preston wear into a per-world depth grid, with an exact mass ledger).

    Example::

        sander = Removal(tool="sander", workpiece="plate", pad_center=(0, 0, -0.03), cells=(128, 128))
        sim = alloy.plugins.simulation(built, [sander])
        ...; r = sim.plugins["removal"].read(); r["depth"][0]       # [nx, ny] removed depth [m], world 0
    """

    name = "removal"
    summary = "material removal by a compliant pad: Archard/Preston wear into a per-world depth grid, mass ledger"

    @dataclass
    class Config:
        tool: str = "tool"                                       # body label of the tool
        pad_center: tuple[float, float, float] = (0.0, 0.0, -0.025)   # pad face centre, tool frame [m]
        pad_normal: tuple[float, float, float] = (0.0, 0.0, -1.0)     # pad face normal (out of the pad), tool frame
        pad_radius: float = 0.025                                # m
        pad_stiffness: float = 3.0e7                             # Pa/m: pressure per metre of indentation
        pad_damping: float = 2.0e-3                              # s: Kelvin-Voigt time of the pad
        friction: float = 0.5                                    # Coulomb coefficient of the pad on the face
        slip_eps: float = 1.0e-3                                 # m/s: friction is viscous below this slip speed
        workpiece: str = "workpiece"                             # shape label of the workpiece
        size: tuple[float, float] | None = None                  # face size in the shape's local x, y; None: box
        surface_offset: float | None = None                      # face height in the shape frame; None: box top
        cells: tuple[int, int] = (64, 64)                        # grid resolution along local x, y
        wear_coefficient: float = 1.0e-2                         # Archard K (two-body abrasion: 1e-3 .. 1e-1)
        hardness: float = 1.0e9                                  # Pa (aluminium 6061: about 1 GPa)
        density: float = 2700.0                                  # kg/m^3 of the workpiece material
        capture: float = 0.0                                     # fraction of removed mass the tool collects
        wipe_length: float = 0.0                                 # m: loose pick-up 1 - exp(-slide/length); 0: none
        feedback: bool = True                                    # removed depth and profile move the contact

        def __post_init__(self) -> None:
            positive = {"pad_radius": self.pad_radius, "pad_stiffness": self.pad_stiffness,
                        "hardness": self.hardness, "density": self.density}
            bad = [k for k, v in positive.items() if not (math.isfinite(v) and v > 0.0)]
            if bad:
                raise ValueError(f"{bad} must be finite and > 0")
            nonneg = {"pad_damping": self.pad_damping, "friction": self.friction, "slip_eps": self.slip_eps,
                      "wear_coefficient": self.wear_coefficient, "wipe_length": self.wipe_length}
            bad = [k for k, v in nonneg.items() if not (math.isfinite(v) and v >= 0.0)]
            if bad:
                raise ValueError(f"{bad} must be finite and >= 0")
            if not 0.0 <= self.capture <= 1.0:
                raise ValueError(f"capture must be in [0, 1], got {self.capture}")
            if min(self.cells) < 1:
                raise ValueError(f"cells must be >= 1, got {self.cells}")
            if np.linalg.norm(self.pad_normal) <= 0.0:
                raise ValueError("pad_normal must be a non-zero vector")

    def setup(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        self.tool = ctx.bodies(c.tool)
        patches = [patch_of(model, int(s), c.workpiece, c.size, c.surface_offset)
                   for s in ctx.shapes(c.workpiece).numpy()]
        self.face_body, self.face_local = patch_tables(patches, ctx.device)
        self.pad = _pad(c)
        self.grid = _grid(c, patches[0].size)
        self.wear = _wear(c)
        self.response = pad_response(model, self.tool.numpy(), patches, c.pad_center, c.pad_normal, c.pad_radius)
        self.mobility = ctx.array(self.response.mobility, wp.float32)
        W = ctx.world_count
        f64 = wp.float64
        self.profile = self.add_state("profile", ctx.zeros((W, *c.cells), f64))
        self.depth = self.add_state("depth", ctx.zeros((W, *c.cells), f64))
        self.loose = self.add_state("loose", ctx.zeros((W, *c.cells), f64))
        self.removed = self.add_state("removed", ctx.zeros(W, f64))
        self.sink = self.add_state("sink", ctx.zeros(W, f64))
        self.load = self.add_state("load", ctx.zeros(W, wp.float32))
        self.contact = self.add_state("contact", ctx.zeros(W, wp.int32))
        log.info("removal %s: tool %r on %r, grid %s of %.3g x %.3g m, pad r=%.3g m, K/H=%.3g 1/Pa", self.label,
                 c.tool, c.workpiece, c.cells, *patches[0].size, c.pad_radius, self.wear.rate)

    def check_dt(self, dt: float) -> None:
        """The pad's explicit spring and damper, normal and tilt together, on the worst world's normal mobility and
        reduced tilt inertia (tool + floating workpiece). Friction needs no check: the force kernel bounds it
        (``stability.py``)."""
        c, r = self.config, self.response
        stiffness = c.pad_stiffness * math.pi * c.pad_radius**2                 # all of the pad pressed in [N/m]
        # normal springs resist a tilt about the COM through their in-plane offsets only (k_w·∫x²dA = stiffness·R²/4);
        # the COM's height above the pad moves pad points tangentially, which the friction bound covers
        tilt = stiffness * 0.25 * c.pad_radius**2                                # [N·m/rad]
        omega_sq = stiffness * r.normal_mobility + 2.0 * tilt / r.tilt_inertia    # bound of ω² (stability.py)
        checks = {"pad spring (normal + tilt)": math.sqrt(omega_sq) * dt,
                  "pad damping (normal + tilt)": c.pad_damping * omega_sq * dt}
        bad = {k: round(v, 3) for k, v in checks.items() if v >= STABLE}
        if bad:
            raise PluginError(f"{self.label}: unstable at dt={dt:.3g} s: {bad} (each must be < {STABLE}); lower "
                              f"pad_stiffness or pad_damping, shrink the exchange step, or give the tool (and a "
                              f"floating workpiece) more mass and inertia")

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        m = self.ctx.model
        wp.launch(pad_force_kernel, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, m.body_com, self.mobility, self.tool, self.face_body,
                          self.face_local, self.pad, self.grid, float(dt), self.profile, self.depth],
                  outputs=[forces.body_f], device=self.ctx.device)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        wp.launch(wear_kernel, dim=self.ctx.world_count,
                  inputs=[state_in.body_q, state_in.body_qd, state_out.body_q, self.ctx.model.body_com, self.tool,
                          self.face_body, self.face_local, self.pad, self.grid, self.wear, self.profile],
                  outputs=[self.depth, self.loose, self.removed, self.sink, self.load, self.contact],
                  device=self.ctx.device)

    # --- host side (between periods) ---------------------------------------------------------------------------

    def set_profile(self, heights: Any) -> None:
        """Initial surface heights above the nominal face [m], ``[nx, ny]`` or ``[W, nx, ny]`` (burrs, waviness).
        Host side, between periods. The simulation's ``reset`` returns to the state it was built with (a flat face)."""
        self._assign(self.profile, heights)

    def set_loose(self, mass: Any) -> None:
        """Loose material lying on each cell [kg], ``[nx, ny]`` or ``[W, nx, ny]`` (dirt, dust, polish). Host side,
        between periods; ``reset`` returns to the built state (clean)."""
        self._assign(self.loose, mass)

    def _assign(self, dst: wp.array, values: Any) -> None:
        src = np.broadcast_to(np.asarray(values, np.float64), dst.shape)
        if not np.isfinite(src).all():
            raise PluginError(f"{self.label}: values must be finite")
        dst.assign(np.ascontiguousarray(src))

    @property
    def cell_area(self) -> float:
        return float(self.grid.cell[0]) * float(self.grid.cell[1])

    def read(self) -> dict[str, np.ndarray]:
        out = super().read()
        out["volume"] = out["depth"].reshape(out["depth"].shape[0], -1).sum(axis=1) * self.cell_area
        out["loose_total"] = out["loose"].reshape(out["loose"].shape[0], -1).sum(axis=1)
        return out

    def test_scene(self) -> TestScene:
        return laws.sanding_scene(self.config)

    def laws(self) -> list[Law]:
        return laws.removal_laws(self.config)


def _pad(c: Removal.Config) -> Pad:
    pad = Pad()
    pad.center = wp.vec3(*c.pad_center)
    pad.normal = wp.vec3(*(np.asarray(c.pad_normal, np.float64) / np.linalg.norm(c.pad_normal)))
    pad.radius = float(c.pad_radius)
    pad.stiffness = float(c.pad_stiffness)
    pad.damping = float(c.pad_damping)
    pad.friction = float(c.friction)
    pad.slip_eps = float(c.slip_eps)
    return pad


def _grid(c: Removal.Config, size: tuple[float, float]) -> Grid:
    grid = Grid()
    grid.half = wp.vec2(0.5 * size[0], 0.5 * size[1])
    grid.cell = wp.vec2(size[0] / c.cells[0], size[1] / c.cells[1])
    grid.feedback = int(c.feedback)
    return grid


def _wear(c: Removal.Config) -> Wear:
    wear = Wear()
    wear.rate = float(c.wear_coefficient / c.hardness)
    wear.density = float(c.density)
    wear.capture = float(c.capture)
    wear.wipe_length = float(c.wipe_length)
    return wear


__all__ = ["Removal"]
