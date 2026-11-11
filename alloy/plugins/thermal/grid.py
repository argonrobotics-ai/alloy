# SPDX-License-Identifier: Apache-2.0
"""``GridThermal``: heat v3, temperature on a rectilinear grid over a metal workpiece, with a moving arc
(docs/notes/heat-v3-design.md; Gym-Anything's welding model, report_welding §3-4).

A force plug-in that adds no forces (any route; no contacts). One grid per instance, in the frame of a carrier body
(``body``; "" = the world frame), offset by ``origin``. Cells whose centre lies inside one of ``boxes`` are metal of
that box's part (``parts``); the others are air. Each exchange, after the solver step (``update``):

1. the beam: a torch body's pose (``torch``: the tip ``torch_tip`` and the direction ``torch_axis`` in its frame,
   taken midway between the step's start and end poses) or the ``arc_pose`` input [W, 6] (point and direction, world
   frame), with the inputs ``arc_power`` [W] (W into the work: the caller's η·V·I) and ``arc_sigma`` [W] (m);
2. the arc's power per cell: a Gaussian beam of standard deviation σ across the torch axis on every metal face that
   is open to air, faces the torch and sees it along the axis with no metal in between, times the face's cosine
   (``grid_kernels.arc_power``); exact power η·V·I when the whole footprint lands on metal;
3. ``substeps`` explicit enthalpy steps (``stable_dt``): conduction between metal cells through the series resistance
   of the two half cells, plus 1 / ``contact`` between cells of different parts until both have reached the solidus
   (fused); faces open to air lose ``convection`` (W/(m²·K)) and grey radiation (``emissivity``) to the ``ambient``
   input [W] (K); the grid's boundary faces named in ``sink`` ("z-": the plate's underside on a table) conduct to the
   ambient through ``sink_conductance``; H above the material's boiling cap is booked as evaporated.

State ``[W, N]`` (N = nx·ny·nz, z fastest): ``enthalpy`` + ``enthalpy_lo`` (float32 H [J/kg] and its remainder, heat
v2's convention, ``grid_materials``), ``peak`` (K, the hottest each cell has been: fusion is peak >= T_s), ``cell``
(0 air, part + 1), and the float64 ledger ``arc_J``, ``lost_J``, ``evaporated_J`` per cell plus ``stored0_J`` [W].
``read()`` gives per world ``stored_J`` (Σ m H), the ledger sums and ``residual_J`` = stored − stored0 − (arc + filler
− lost − evaporated), ``t_max``, ``fused_cells``, ``molten_cells``, ``liquid_m3`` (Σ φ V), the wire's ``fed_m3``,
``deposited_m3``, ``missed_m3`` and ``weld_cells``, and ``probe_T`` [W, P] at ``probes`` (grid frame);
``fields(world)`` gives the whole grid.

Deposition (``grid_deposit``, ``grid_cap``): the ``wire_feed`` input [W] (m/s) deposits weld metal in the slices
around the beam's anchor; the droplets' heat goes into a cylinder below the arc root, and the beam carries the arc's
power less the droplets' share. ``bead_model="cap"`` holds each slice's liquid between its contact points.

Limits (v3): the bead's shape is not GA's pool (mass and energy are exact; the shape is not claimed); parts are fixed
in the grid (a loose part moved by a robot before it is tacked is not followed); constant density (no expansion or
shrinkage); no pool flow (the liquid's conductivity stands in for it, as in GA); the grid takes no heat from fixtures
beyond the sink. Exclusivity: a carrier body must not also be a ``Thermal`` (v1) body, or its heat is counted
twice; setup cannot see the other plug-ins, so ``exclusive(host)`` checks it and the weld scenes call it.

Example (a 40 x 30 x 6 mm plate on a table, an arc of 2 kW along x)::

    grid = GridThermal(x=graded(0, 0.04, 5e-4), y=graded(-0.015, 0.015, 5e-4), z=graded(-0.006, 0, 5e-4),
                       boxes=((0, -0.015, -0.006, 0.04, 0.015, 0),), sink=("z-",))
    sim.plugins["grid_thermal"].set_input("arc_power", np.full(W, 2000.0))
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.thermal import grid_cap as C
from alloy.plugins.thermal import grid_deposit as D
from alloy.plugins.thermal import grid_flow as F
from alloy.plugins.thermal import grid_kernels as K
from alloy.plugins.thermal.grid_materials import GridMaterial, material

log = logging.getLogger("alloy.plugins")

FACES = ("x-", "x+", "y-", "y+", "z-", "z+")
BYTES_PER_CELL = 52  # state 4+4+4+4 (H, remainder, peak, cell) + 3 x 8 ledger + 3 x 4 derived (T, k, arc power)
MAX_SUBSTEPS = 100_000
BEAD_MODELS = C.BEAD_MODELS


def graded(lo: float, hi: float, fine: float, band: tuple[float, float] | None = None, growth: float = 1.2,
           coarse: float = 3.0e-3) -> tuple[float, ...]:
    """Cell edges from ``lo`` to ``hi``: ``fine`` cells across ``band`` (default the whole span), growing by
    ``growth`` per cell outside it up to ``coarse``, scaled so the last edge is exactly ``lo`` / ``hi`` (GA's grid:
    0.25-0.5 mm in the joint band growing 1.2x to 2.5-3 mm, report_welding §3)."""
    a, b = band if band is not None else (lo, hi)
    if not lo <= a < b <= hi or fine <= 0.0 or growth < 1.0:
        raise ValueError(f"graded: needs lo <= band[0] < band[1] <= hi, fine > 0, growth >= 1 (got {lo}, {band}, "
                         f"{hi}, {fine}, {growth})")
    inner = np.linspace(a, b, max(1, round((b - a) / fine)) + 1)

    def outward(length: float) -> np.ndarray:
        if length <= 1e-12:
            return np.zeros(0)
        widths, w = [], fine
        while sum(widths) < length:
            w = min(w * growth, coarse)
            widths.append(w)
        out = np.asarray(widths)
        return np.cumsum(out * (length / out.sum()))

    right = b + outward(hi - b)
    left = a - outward(a - lo)
    edges = np.concatenate([left[::-1], inner, right])
    edges[0], edges[-1] = lo, hi
    return tuple(float(e) for e in edges)


_PLATE = (graded(0.0, 0.02, 1.0e-3), graded(0.0, 0.02, 1.0e-3), graded(-0.005, 0.0, 1.0e-3))  # the default grid


@dataclass
class Config:
    material: str = "steel_a36"                          # grid_materials.MATERIALS
    x: tuple[float, ...] = _PLATE[0]                     # cell edges [m], grid frame (graded()); default 20 mm
    y: tuple[float, ...] = _PLATE[1]
    z: tuple[float, ...] = _PLATE[2]
    boxes: tuple[tuple[float, ...], ...] = ((0.0, 0.0, -0.005, 0.02, 0.02, 0.0),)   # metal: (x0, y0, z0, x1, y1, z1)
    parts: tuple[int, ...] = ()                          # part of each box (default: all part 0); later boxes win
    initial: float = 293.15                              # K
    body: str = ""                                       # carrier body label ("": the world frame)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)   # the grid frame's origin in the carrier's frame [m]
    torch: str = ""                                      # torch body label ("": the arc_pose input)
    torch_tip: tuple[float, float, float] = (0.0, 0.0, 0.0)  # a point on the beam axis, torch frame [m]
    torch_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)  # beam direction, torch frame
    arc_sigma: float = 4.0e-3                            # m (GA §5: 4 mm, flagged open; Penn State 5 mm)
    convection: float = 12.0                             # W/(m²·K) (GA §4)
    emissivity: float = 0.7                              # EN 1993-1-2 §2.2(2) (GA §4)
    contact: float = 1500.0                              # W/(m²·K) between unfused parts (GA §4, an estimate)
    sink: tuple[str, ...] = ()                           # boundary faces on a table, e.g. ("z-",)
    sink_conductance: float = 1500.0                     # W/(m²·K) (GA §4: the same contact conductance)
    ambient: float = 293.15                              # K, every world's start value of the input "ambient"
    probes: tuple[tuple[float, float, float], ...] = ()  # grid-frame points read as probe_T (must be metal)
    max_cells: int = 20_000_000                          # refuse a grid with more cells over all worlds
    safety: float = 0.9                                  # explicit step = safety x the positivity limit
    wire_diameter: float = 0.9e-3                        # m (GA: 0.035 in ER70S-6)
    deposit_efficiency: float = 0.98                     # deposited / fed (GA §5, Lincoln: spray 98 %)
    drop_temperature: float = 2673.0                     # K, the droplets' (GA §5)
    weld_part: int = -1                                  # part of deposited metal (-1: one past the highest part)
    deposit_reach: float = 8.0e-3                        # m, farthest deposited cell centre from the anchor
    bead_flatten: float = 1.0                            # κ on Δz in the fill score (grid_deposit)
    bead_norm: float = 2.0                               # p of the fill score
    max_fill: int = 256                                  # cells per world per exchange at most
    deposit_spread: float = 1.0e-3                       # m, σ sharing each exchange's wire along x (GA: 1 mm)
    drop_cylinder: tuple[float, float] = (0.5e-3, 1.5e-3)   # m, droplet heat's radius and depth below the arc root
    deposit_lag: float = 0.0                             # s: metal lands where the arc was (struck) this long ago;
    #   the arc's velocity along x only (straight travel along the grid's x)
    arc_pressure: tuple[float, float] = (0.0, 0.0)       # (N, m): the arc's force and σ; F > 0: the pool flows
    surface_tension: float = 1.2                         # N/m, liquid steel's (GA: POOL_TAU's 1.2)
    flow_moves: int = 64                                 # pool flow: liquid weld cells moved per world per exchange
    bead_model: str = "nearest"                          # "nearest" | "cap" (grid_deposit: a cap between fused toes)
    reshape_moves: int = 16                              # cap model: liquid weld cells moved per slice per exchange
    joint_root: tuple[float, float] | None = None        # (y, z) of the root, grid frame (None: the anchor's foot)
    joint_rays: tuple[tuple[float, float], tuple[float, float]] = ((1.0, 0.0), (-1.0, 0.0))  # surfaces from the root

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            e = np.asarray(getattr(self, name), np.float64)
            if e.size < 2 or np.any(np.diff(e) <= 0.0):
                raise ValueError(f"{name}: needs at least two strictly increasing cell edges (graded())")
        if not self.boxes or any(len(b) != 6 or b[3] <= b[0] or b[4] <= b[1] or b[5] <= b[2] for b in self.boxes):
            raise ValueError("boxes: one or more (x0, y0, z0, x1, y1, z1) with x1 > x0, y1 > y0, z1 > z0")
        if self.parts and (len(self.parts) != len(self.boxes) or min(self.parts) < 0):
            raise ValueError("parts: one non-negative part per box (or none)")
        bad = [f for f in self.sink if f not in FACES]
        if bad:
            raise ValueError(f"sink {bad}: use {FACES}")
        if min(self.convection, self.emissivity, self.contact, self.sink_conductance) < 0.0 or self.emissivity > 1.0:
            raise ValueError("convection, contact, sink_conductance >= 0 and emissivity in [0, 1]")
        if self.arc_sigma <= 0.0 or self.initial <= 0.0 or self.ambient <= 0.0 or not 0.0 < self.safety <= 1.0:
            raise ValueError("arc_sigma > 0, initial and ambient > 0 K, safety in (0, 1]")
        if self.wire_diameter <= 0.0 or not 0.0 < self.deposit_efficiency <= 1.0 or self.drop_temperature <= 0.0:
            raise ValueError("wire_diameter > 0, deposit_efficiency in (0, 1], drop_temperature > 0 K")
        if self.deposit_reach <= 0.0 or self.bead_flatten <= 0.0 or self.bead_norm < 1.0 or self.max_fill < 1 or \
                self.deposit_spread <= 0.0 or min(self.drop_cylinder) <= 0.0:
            raise ValueError("deposit_reach, bead_flatten, deposit_spread, drop_cylinder > 0, bead_norm >= 1, "
                             "max_fill >= 1")
        if self.bead_model not in BEAD_MODELS:
            raise ValueError(f"bead_model {self.bead_model!r}: use {BEAD_MODELS}")
        if self.arc_pressure[0] > 0.0 and self.arc_pressure[1] <= 0.0:
            raise ValueError("arc_pressure: a force > 0 needs σ > 0")
        if self.arc_pressure[0] > 0.0 and F.dent_cell(self.y, self.z, self.joint_root) > self.arc_pressure[1] / 3.0:
            log.warning("grid_thermal: the pool flow (arc_pressure) needs cells at the joint's root under a third of "
                        "its σ (%.2g m): coarser, the dent under the arc starves slices", self.arc_pressure[1])
        rays = np.asarray(self.joint_rays, np.float64)
        if rays.shape != (2, 2) or np.any(np.linalg.norm(rays, axis=1) == 0.0) or \
                rays[0, 0] * rays[1, 1] - rays[0, 1] * rays[1, 0] < -1e-9:
            raise ValueError("joint_rays: two non-zero (y, z) directions sweeping counter-clockwise through the bead")
        if np.linalg.norm(self.torch_axis) == 0.0:
            raise ValueError("torch_axis must be non-zero")
        material(self.material)


def edges(config: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.asarray(v, np.float64) for v in (config.x, config.y, config.z))


def occupancy(config: Config) -> np.ndarray:
    """``cell`` [N] int32: 0 air, part + 1 for cells whose centre lies inside a box (z fastest)."""
    cx, cy, cz = (0.5 * (e[1:] + e[:-1]) for e in edges(config))
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing="ij")
    cell = np.zeros(X.shape, np.int32)
    for i, b in enumerate(config.boxes):
        part = config.parts[i] if config.parts else 0
        inside = (X > b[0]) & (X < b[3]) & (Y > b[1]) & (Y < b[4]) & (Z > b[2]) & (Z < b[5])
        cell[inside] = part + 1
    return cell.reshape(-1)


def estimate(config: Config, worlds: int = 1) -> dict[str, float]:
    """Cells per world, metal cells, bytes per world and in all (``BYTES_PER_CELL``), before building."""
    n = int(np.prod([len(e) - 1 for e in edges(config)]))
    return {"cells": n, "metal": int(np.count_nonzero(occupancy(config))), "bytes_per_world": n * BYTES_PER_CELL,
            "bytes": worlds * n * BYTES_PER_CELL}


def _axis_rate(e: np.ndarray, alpha: float, sink_lo: float, sink_hi: float, expose: float) -> float:
    """max over cells along one axis of α Σ_faces 1 / (δ d) + the sink and exposure terms [1/s]."""
    d = np.diff(e)
    delta = 0.5 * (d[1:] + d[:-1])                      # centre distances between neighbours
    rate = np.zeros_like(d)
    rate[:-1] += alpha / (delta * d[:-1])
    rate[1:] += alpha / (delta * d[1:])
    rate += 2.0 * expose / d
    rate[0] += sink_lo / d[0]
    rate[-1] += sink_hi / d[-1]
    return float(rate.max())


def stable_dt(config: Config, mat: GridMaterial | None = None) -> float:
    """The explicit step [s]: ``safety`` x the positivity limit min_i C_i / Σ_j G_ij over every cell as if all were
    metal, with each face's conductance bounded by A·min(k_max / δ, γ k(T) / δ) (γ = 2 max δ / d) and every face
    possibly open to air (convection + radiation linearised at the boiling cap) or a sink (GA §4: the explicit limit
    over the smallest cells at the largest diffusivity)."""
    mat = mat or material(config.material)
    ex, ey, ez = edges(config)
    gamma = 1.0
    for e in (ex, ey, ez):
        d = np.diff(e)
        if d.size > 1:
            delta = 0.5 * (d[1:] + d[:-1])
            gamma = max(gamma, 2.0 * float(np.max(np.maximum(delta / d[1:], delta / d[:-1]))))
    kmax, cmin, ratio = mat.bounds(gamma)
    alpha = ratio / mat.density
    rho_c = mat.density * cmin
    expose = (config.convection + 4.0 * config.emissivity * K.SIGMA_SB * mat.t_boil ** 3) / rho_c
    rate = 0.0
    for axis, e in enumerate((ex, ey, ez)):
        dmin = float(np.diff(e).min())
        sinks = [min(config.sink_conductance, 2.0 * kmax / dmin) / rho_c if FACES[2 * axis + s] in config.sink
                 else 0.0 for s in (0, 1)]
        rate += _axis_rate(e, alpha, sinks[0], sinks[1], expose)
    return config.safety / rate


def substeps(config: Config, dt: float) -> int:
    """Explicit steps per exchange of ``dt`` seconds."""
    return max(1, math.ceil(dt / stable_dt(config) - 1e-9))


class GridThermal(ForcePlugin):
    """Temperature on a rectilinear grid over a workpiece, with a moving arc (see the module docstring)."""

    name = "grid_thermal"
    summary = "grid heat over a workpiece (heat v3): conduction, melting, losses, a moving Gaussian arc"
    needs_contacts: ClassVar[bool] = False
    speed_budget: ClassVar[float] = 4.0
    Config = Config

    def setup(self, ctx: Any) -> None:
        c = self.config
        self._mat = material(c.material)
        self._W = W = ctx.world_count
        ex, ey, ez = edges(c)
        self._shape = (len(ex) - 1, len(ey) - 1, len(ez) - 1)
        self.N = N = int(np.prod(self._shape))
        if W * N > c.max_cells:
            raise PluginError(f"{self.label}: {W} worlds x {N} cells = {W * N:,} cells exceeds max_cells "
                              f"{c.max_cells:,} (~{W * N * BYTES_PER_CELL / 1e9:.1f} GB); coarsen the grid outside "
                              f"the joint band (graded(coarse=...)), use fewer worlds, or raise max_cells")
        cell = occupancy(c)
        d = [np.diff(e) for e in (ex, ey, ez)]
        vol = (d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :]).reshape(-1)
        self._mass = (self._mat.density * vol).astype(np.float32)
        self._metal = cell != 0
        self._ga, self._tb = self._structs(ctx, ex, ey, ez)
        h0, lo0 = self._split(np.full((W, N), c.initial))
        self.enthalpy = self.add_state("enthalpy", ctx.array(h0, wp.float32))
        self.enthalpy_lo = self.add_state("enthalpy_lo", ctx.array(lo0, wp.float32))
        self.peak = self.add_state("peak", ctx.full((W, N), c.initial, wp.float32))
        self.cell = self.add_state("cell", ctx.array(np.tile(cell, (W, 1)), wp.int32))
        self.arc_J = self.add_state("arc_J", ctx.zeros((W, N), wp.float64))
        self.lost_J = self.add_state("lost_J", ctx.zeros((W, N), wp.float64))
        self.evaporated_J = self.add_state("evaporated_J", ctx.zeros((W, N), wp.float64))
        self.stored0_J = self.add_state("stored0_J", ctx.array(self._stored(h0, lo0), wp.float64))
        self.deposit_budget = self.add_state("deposit_budget_m3", ctx.zeros((W, self._shape[0]), wp.float64))
        for name in ("fed_m3", "deposited_m3", "missed_m3", "filler_J"):
            setattr(self, name, self.add_state(name, ctx.zeros(W, wp.float64)))
        self.slice_fed_m3 = self.add_state("slice_fed_m3", ctx.zeros((W, self._shape[0]), wp.float64))
        self.toes = self.add_state("toes", ctx.zeros((W, self._shape[0]), wp.vec2))
        self.clock_s, self.arc_on_s = (self.add_state(n, ctx.zeros(W, wp.float64)) for n in ("clock_s", "arc_on_s"))
        self.liquid_s, self.live_s = (self.add_state(n, ctx.full((W, self._shape[0]), -1.0, wp.float64))
                                      for n in ("liquid_s", "live_s"))
        self._mass_d = ctx.array(self._mass, wp.float32)
        self._t, self._k = ctx.zeros((W, N), wp.float32), ctx.zeros((W, N), wp.float32)
        self._arc_w = ctx.zeros((W, N), wp.float32)
        self._pool = tuple(ctx.zeros((W, self._shape[0]), t) for t in (wp.float32, wp.float32, wp.vec2, wp.float32))
        self._drop = (ctx.zeros((W, 1024), wp.int32), ctx.zeros((W, 1024), wp.float32), ctx.zeros(W, wp.int32))
        self._arcs = wp.zeros(W, dtype=K.Arc, device=ctx.device)
        pose = np.zeros((W, 6), np.float32)
        pose[:, 5] = -1.0
        self.add_input("arc_pose", ctx.array(pose, wp.float32))
        self.add_input("arc_power", ctx.zeros(W, wp.float32))
        self.add_input("arc_sigma", ctx.full(W, c.arc_sigma, wp.float32))
        self.add_input("ambient", ctx.full(W, c.ambient, wp.float32))
        self.add_input("wire_feed", ctx.zeros(W, wp.float32))
        self._bead, self._drop_power = C.bead_struct(c, self._mat, cell)
        self._torch = ctx.bodies(c.torch) if c.torch else ctx.full(W, -1, wp.int32)
        self._carrier = ctx.bodies(c.body) if c.body else ctx.full(W, -1, wp.int32)
        self._no_q = wp.zeros(1, dtype=wp.transform, device=ctx.device)
        self._probes = self._probe_cells(ex, ey, ez)
        self._dt_stable = stable_dt(c, self._mat)
        self._derive()

    def _structs(self, ctx: Any, ex: np.ndarray, ey: np.ndarray, ez: np.ndarray) -> tuple[K.Grid, K.Table]:
        c, m = self.config, self._mat
        g = K.Grid()
        g.nx, g.ny, g.nz = self._shape
        g.ex, g.ey, g.ez = (ctx.array(e, wp.float32) for e in (ex, ey, ez))
        g.sink = int(sum(1 << FACES.index(f) for f in set(c.sink)))
        g.h_air, g.eps, g.h_sink, g.h_contact = c.convection, c.emissivity, c.sink_conductance, c.contact
        g.t_solidus = m.t_solidus
        g.weld = (c.weld_part if c.weld_part >= 0 else int(occupancy(c).max())) + 1
        tt, hh, kk = m.table()
        tb = K.Table()
        tb.t, tb.h, tb.k = (ctx.array(v, wp.float32) for v in (tt, hh, kk))
        tb.count = len(tt)
        tb.h_liquidus, tb.h_boil = m.h_liquidus, m.h_boil
        tb.c_low = float(m.c_solid(np.asarray(tt[0])))
        return g, tb

    def _split(self, temps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """float32 H and its float32 remainder for temperatures [W, N] (K)."""
        h = self._mat.enthalpy(temps)
        hi = h.astype(np.float32)
        return hi, (h - hi.astype(np.float64)).astype(np.float32)

    def _stored(self, h: np.ndarray, lo: np.ndarray) -> np.ndarray:
        """Σ m H over metal cells per world [J], float64."""
        hh = h.astype(np.float64) + lo.astype(np.float64)
        return (hh * self._mass.astype(np.float64) * self._metal).sum(axis=1)

    def _probe_cells(self, ex: np.ndarray, ey: np.ndarray, ez: np.ndarray) -> np.ndarray:
        out = []
        for p in self.config.probes:
            ijk = [int(np.searchsorted(e, v, side="right")) - 1 for e, v in zip((ex, ey, ez), p, strict=True)]
            if any(i < 0 or i >= n for i, n in zip(ijk, self._shape, strict=True)):
                raise PluginError(f"{self.label}: probe {p} is outside the grid")
            idx = (ijk[0] * self._shape[1] + ijk[1]) * self._shape[2] + ijk[2]
            if not self._metal[idx]:
                raise PluginError(f"{self.label}: probe {p} is in air (cell {ijk})")
            out.append(idx)
        return np.asarray(out, np.int64)

    def check_dt(self, dt: float) -> None:
        n = substeps(self.config, dt)
        if n > MAX_SUBSTEPS:
            raise PluginError(f"{self.label}: {n:,} explicit steps per {dt:g} s exchange (limit "
                              f"{self._dt_stable:.3g} s): coarsen the smallest cells or lower sink_conductance")
        log.info("grid_thermal %s: %d substeps of %.3g s per %.3g s exchange (%s cells per world)", self.label, n,
                 dt / n, dt, f"{self.N:,}")

    @property
    def applies_forces(self) -> bool:
        return False

    def _derive(self) -> None:
        wp.launch(K.derive, dim=(self._W, self.N), device=self.ctx.device,
                  inputs=[self._tb, self.cell, self._mass_d, self.enthalpy, self.enthalpy_lo],
                  outputs=[self._t, self._k, self.peak, self.evaporated_J])

    def _pool_time(self, dt: float) -> None:
        wp.launch(C.pool_time, dim=(self._W, self._shape[0]), device=self.ctx.device, inputs=[self._ga, self._bead,
                  self.cell, self._t, self.slice_fed_m3, self.clock_s, dt], outputs=[self.liquid_s, self.live_s])

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        c, dev, W, N = self.config, self.ctx.device, self._W, self.N
        q_in = state_in.body_q if self.ctx.model.body_count else self._no_q
        q_out = state_out.body_q if self.ctx.model.body_count else self._no_q
        inp = self.inputs()
        wp.launch(K.arc_frame, dim=W, device=dev,
                  inputs=[q_in, q_out, self._torch, self._carrier, wp.vec3(*c.torch_tip), wp.vec3(*c.torch_axis),
                          wp.vec3(*c.origin), inp["arc_pose"], inp["arc_power"], inp["arc_sigma"],
                          inp["wire_feed"], self._drop_power, float(dt)],
                  outputs=[self._arcs])
        wp.launch(D.deposit, dim=W, device=dev,
                  inputs=[self._ga, self._bead, self._tb, self._arcs, inp["wire_feed"], float(dt), self._mass_d],
                  outputs=[self.cell, self.enthalpy, self.enthalpy_lo, self.peak, self.deposit_budget, self.fed_m3,
                           self.deposited_m3, self.missed_m3, self.filler_J, self.slice_fed_m3, self.toes, self._t,
                           *self._drop, self.arc_on_s])
        if c.bead_model == "cap" and c.arc_pressure[0] > 0.0:
            wp.launch(F.flow, dim=W, device=dev, inputs=[self._ga, self._bead, 0.5 * self._mat.h_liquidus,
                      c.flow_moves, self._arcs], outputs=[self.cell, self.enthalpy, self.enthalpy_lo, self.peak,
                      self.deposit_budget, self.slice_fed_m3, self.toes, self._t, self.live_s, self.clock_s,
                      *self._pool, self._mass_d])
        if c.bead_model == "cap" and c.reshape_moves > 0:
            wp.launch(C.reshape, dim=(W, self._shape[0]), device=dev,
                      inputs=[self._ga, self._bead, 0.5 * self._mat.h_liquidus, c.reshape_moves],
                      outputs=[self.cell, self.enthalpy, self.enthalpy_lo, self.peak, self.deposit_budget,
                               self.slice_fed_m3, self.toes, self._t, self._mass_d])
        self._derive()
        self._pool_time(float(dt))                         # new metal is liquid now: a slice it froze in at once
        wp.launch(K.arc_power, dim=(W, N), device=dev, inputs=[self._ga, self.cell, self._arcs],
                  outputs=[self._arc_w])
        wp.launch(D.droplets, dim=W, device=dev, inputs=list(self._drop), outputs=[self._arc_w])
        n = max(1, math.ceil(dt / self._dt_stable - 1e-9))
        sub = float(dt) / n
        for _ in range(n):
            wp.launch(K.flux, dim=(W, N), device=dev,
                      inputs=[self._ga, self.cell, self._mass_d, self._t, self._k, self.peak, self._arc_w,
                              inp["ambient"], sub],
                      outputs=[self.enthalpy, self.enthalpy_lo, self.arc_J, self.lost_J])
            self._derive()
        self._pool_time(float(dt))
        wp.launch(C.tick, dim=W, device=dev, inputs=[inp["arc_power"], float(dt)],
                  outputs=[self.clock_s, self.arc_on_s])

    def set_temperature(self, values: Any) -> None:
        """Start temperatures [K]: a scalar, [N] or [W, N]; resets the peaks and the ledger (call before stepping)."""
        temps = np.broadcast_to(np.asarray(values, np.float64), (self._W, self.N))
        h, lo = self._split(temps)
        self.enthalpy.assign(h)
        self.enthalpy_lo.assign(lo)
        self.peak.assign(temps.astype(np.float32))
        for arr in (self.arc_J, self.lost_J, self.evaporated_J, self.deposit_budget, self.fed_m3, self.deposited_m3,
                    self.missed_m3, self.filler_J, self.slice_fed_m3, self.toes, self.clock_s, self.arc_on_s):
            arr.zero_()
        self.liquid_s.fill_(-1.0)
        self.live_s.fill_(-1.0)
        self.stored0_J.assign(self._stored(h, lo))
        self._derive()

    def _host_h(self) -> np.ndarray:
        return self.enthalpy.numpy().astype(np.float64) + self.enthalpy_lo.numpy().astype(np.float64)

    def read(self) -> dict[str, np.ndarray]:
        """Per-world sums (not the grid: see ``fields``), so a trace of every period stays small."""
        h, metal = self._host_h(), self.cell.numpy() != 0
        temp, phi = self._mat.temperature(h)
        stored = (h * self._mass.astype(np.float64) * metal).sum(axis=1)
        arc, lost, evap = (a.numpy().sum(axis=1) for a in (self.arc_J, self.lost_J, self.evaporated_J))
        filler = self.filler_J.numpy()
        out = {"stored_J": stored, "arc_J": arc, "lost_J": lost, "evaporated_J": evap, "filler_J": filler,
               "residual_J": stored - self.stored0_J.numpy() - (arc + filler - lost - evap),
               "fed_m3": self.fed_m3.numpy(), "deposited_m3": self.deposited_m3.numpy(),
               "missed_m3": self.missed_m3.numpy(), "weld_cells": (self.cell.numpy() == self._bead.weld).sum(axis=1),
               "t_max": np.where(metal, temp, 0.0).max(axis=1),
               "fused_cells": ((self.peak.numpy() >= self._mat.t_solidus) & metal).sum(axis=1),
               "molten_cells": ((phi > 0.0) & metal).sum(axis=1),
               "liquid_m3": (phi * (self._mass.astype(np.float64) / self._mat.density) * metal).sum(axis=1)}
        if self._probes.size:
            out["probe_T"] = temp[:, self._probes]
        return out

    def fields(self, world: int = 0) -> dict[str, np.ndarray]:
        """The whole grid of one world, shaped (nx, ny, nz): temperature [K], liquid fraction, peak [K], cell."""
        h = self._host_h()[world]
        temp, phi = self._mat.temperature(h)
        cell = self.cell.numpy()[world]
        return {"temperature": temp.reshape(self._shape), "liquid": phi.reshape(self._shape),
                "peak": self.peak.numpy()[world].reshape(self._shape), "cell": cell.reshape(self._shape)}

    @staticmethod
    def exclusive(host: Any) -> None:
        """Refuse a carrier body that a ``Thermal`` (heat v1) plug-in also heats: its heat would count twice."""
        from alloy.plugins.thermal.sensor import Thermal  # noqa: PLC0415

        plugins = list(getattr(host, "plugins", {}).values())
        lumped = {b for p in plugins if isinstance(p, Thermal) for b in p.config.bodies}
        for p in plugins:
            if isinstance(p, GridThermal) and p.config.body and p.config.body in lumped:
                raise PluginError(f"{p.label}: body {p.config.body!r} is also a Thermal (v1) body; its heat would "
                                  f"count twice: drop it from Thermal's bodies")

    def laws(self):
        from alloy.plugins.thermal.grid_laws import grid_laws  # noqa: PLC0415 - the laws import this module

        from alloy.plugins.thermal.grid_deposit_laws import deposit_laws  # noqa: PLC0415

        return grid_laws() + deposit_laws()

    def test_scene(self):
        from alloy.plugins.thermal.grid_laws import busy_scene  # noqa: PLC0415

        return busy_scene()


__all__ = ["FACES", "GridThermal", "estimate", "graded", "occupancy", "stable_dt", "substeps"]
