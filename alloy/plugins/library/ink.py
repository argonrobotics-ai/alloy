# SPDX-License-Identifier: Apache-2.0
"""Ink deposition: a pen tip that touches a paper surface lays ink into a per-world grid (or an eraser lifts it).

The paper is any shape (static or on a body) with a flat writing surface at local height ``surface_offset`` (a box's
top face by default) spanning ``paper_size`` in its local x, y. Every step, after the solver, the pen tip's height
``h`` above that surface is measured; when ``h <= contact_tolerance`` (and the tip is over the sheet) the pen lays
``rate·dt`` of ink (µL), limited by what is left in its reservoir, spread evenly over the grid cells whose centres lie
within ``brush_radius`` of the tip (at least the cell under the tip). An eraser removes up to ``rate·dt`` from those
cells instead and collects it. Ink is conserved exactly up to float32 rounding: Σ grid + reservoir (+ collected)
stays constant. ``touching [W]`` reports contact on the last step. Geometry, not contact points, decides touching,
so it works with every solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform

MAX_DEPTH = 0.02      # m below the surface still counted as touching (soft contacts sink; deeper = not on the sheet)


@wp.func
def _paper_frame(body_q: wp.array[wp.transform], paper_body: wp.int32, local: wp.transform) -> wp.transform:
    if paper_body >= 0:
        return wp.transform_multiply(body_q[paper_body], local)
    return local


@wp.kernel
def _ink_kernel(body_q: wp.array[wp.transform], pen: wp.array[wp.int32], tip: wp.vec3,
                paper_body: wp.array[wp.int32], paper_local: wp.array[wp.transform], half: wp.vec2,
                surface: wp.float32, tol: wp.float32, radius: wp.float32, amount: wp.float32, erase: wp.int32,
                ink: wp.array3d[wp.float32], reservoir: wp.array[wp.float32], touching: wp.array[wp.int32]):
    w = wp.tid()                                       # one thread per world owns its grid: no atomics
    r = radius
    frame = _paper_frame(body_q, paper_body[w], paper_local[w])
    p = wp.transform_point(wp.transform_inverse(frame), wp.transform_point(body_q[pen[w]], tip))
    h = p[2] - surface
    on = h <= tol and h >= -MAX_DEPTH and wp.abs(p[0]) <= half[0] + r and wp.abs(p[1]) <= half[1] + r
    touching[w] = wp.where(on, 1, 0)
    if not on:
        return
    nx, ny = ink.shape[1], ink.shape[2]
    dx, dy = 2.0 * half[0] / float(nx), 2.0 * half[1] / float(ny)
    u, v = p[0] + half[0], p[1] + half[1]
    i0, i1 = wp.max(int(wp.floor((u - r) / dx)), 0), wp.min(int(wp.floor((u + r) / dx)), nx - 1)
    j0, j1 = wp.max(int(wp.floor((v - r) / dy)), 0), wp.min(int(wp.floor((v + r) / dy)), ny - 1)
    count = int(0)
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            cu, cv = (float(i) + 0.5) * dx - u, (float(j) + 0.5) * dy - v
            if cu * cu + cv * cv <= r * r:
                count += 1
    if count == 0:                                     # brush smaller than a cell: the cell under the tip
        i0 = wp.clamp(int(wp.floor(u / dx)), 0, nx - 1)
        j0 = wp.clamp(int(wp.floor(v / dy)), 0, ny - 1)
        i1, j1 = i0, j0
        r = 1.0e9
        count = 1
    share = amount / float(count)
    if erase == 0:
        share = wp.min(amount, reservoir[w]) / float(count)
    total = float(0.0)
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            cu, cv = (float(i) + 0.5) * dx - u, (float(j) + 0.5) * dy - v
            if cu * cu + cv * cv <= r * r:
                moved = share
                if erase != 0:
                    moved = -wp.min(share, ink[w, i, j])
                ink[w, i, j] = ink[w, i, j] + moved
                total += moved
    reservoir[w] = reservoir[w] - total                # once per step: pen drains, eraser collects


class Ink(ForcePlugin):
    """A pen (or eraser) laying ink on a paper surface into a per-world grid; geometric contact, conserved ink.

    Example::

        ink = Ink(pen="pen", tip=(0, 0, -0.01), paper="paper", cells=(128, 128))
        sim = alloy.plugins.simulation(built, [ink])
        ...; grid = sim.plugins["ink"].read()["ink"][0]       # [nx, ny] µL per cell, world 0
    """

    name = "ink"
    summary = "pen or eraser on a paper shape: contact-driven ink deposition into a per-world grid"

    @dataclass
    class Config:
        pen: str = "pen"                                   # body label of the pen
        tip: tuple[float, float, float] = (0.0, 0.0, -0.01)    # tip in the pen body frame [m]
        paper: str = "paper"                               # shape label of the sheet (static or on a body)
        paper_size: tuple[float, float] | None = None      # sheet size in its local x, y [m]; None: the box's
        surface_offset: float | None = None                # writing surface height in the shape frame; None: box top
        cells: tuple[int, int] = (100, 100)                # grid resolution along local x, y
        contact_tolerance: float = 1.5e-3                  # m: tip this close above the surface counts as touching
        brush_radius: float = 1.5e-3                       # m
        rate: float = 5.0                                  # µL/s laid (pen) or lifted (eraser) while touching
        reservoir: float = 100.0                           # µL in the pen at the start (pen mode)
        mode: str = "pen"                                  # "pen" or "eraser"

        def __post_init__(self) -> None:
            if self.mode not in ("pen", "eraser"):
                raise ValueError(f"mode must be 'pen' or 'eraser', got {self.mode!r}")
            if min(self.cells) < 1 or self.rate < 0 or self.reservoir < 0 or self.brush_radius < 0:
                raise ValueError("cells >= 1, rate >= 0, reservoir >= 0 and brush_radius >= 0 are required")

    def setup(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        shapes = ctx.shapes(c.paper).numpy()
        size, top = _sheet(model, int(shapes[0]), c)
        self.half = wp.vec2(0.5 * size[0], 0.5 * size[1])
        self.surface = float(top)
        self.pen = ctx.bodies(c.pen)
        self.paper_body = ctx.array(model.shape_body.numpy()[shapes].astype(np.int32), wp.int32)
        self.paper_local = ctx.array(model.shape_transform.numpy()[shapes], wp.transform)
        W = ctx.world_count
        self.ink = self.add_state("ink", ctx.zeros((W, *c.cells)))
        start = c.reservoir if c.mode == "pen" else 0.0
        self.reservoir = self.add_state("reservoir", ctx.array(np.full(W, start, np.float32)))
        self.touching = self.add_state("touching", ctx.zeros(W, wp.int32))

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        c = self.config
        wp.launch(_ink_kernel, dim=self.ctx.world_count,
                  inputs=[state_out.body_q, self.pen, wp.vec3(*c.tip), self.paper_body, self.paper_local, self.half,
                          self.surface, float(c.contact_tolerance), float(c.brush_radius), float(c.rate * dt),
                          int(c.mode == "eraser")],
                  outputs=[self.ink, self.reservoir, self.touching], device=self.ctx.device)

    def read(self) -> dict[str, np.ndarray]:
        out = super().read()
        out["total"] = out["ink"].reshape(out["ink"].shape[0], -1).sum(axis=1, dtype=np.float64)
        return out

    def test_scene(self) -> TestScene:
        """A pen block (tip at its bottom face) slides 2-3 cm across a 20 cm sheet at 0.3 m/s and stops."""
        return TestScene(build=_pen_on_paper(self.config), init=_push_pen(0.3), periods=25)

    def laws(self) -> list[Law]:
        hover = TestScene(build=_pen_on_paper(self.config, lift=0.005), gravity=(0.0, 0.0, 0.0), periods=10)
        return [
            Law("ink_conserved", _check_conserved, None, "grid + reservoir constant per world"),
            Law("no_ink_when_lifted", _check_lifted, hover, "a pen 5 mm above the sheet lays nothing"),
            Law("track_length", _check_track, None, "inked track length = tip travel while touching"),
        ]


def _sheet(model: newton.Model, s: int, c: Ink.Config) -> tuple[tuple[float, float], float]:
    """Sheet size and surface height from the config, or from the shape if it is a box."""
    is_box = int(model.shape_type.numpy()[s]) == int(newton.GeoType.BOX)
    half = model.shape_scale.numpy()[s]
    if (c.paper_size is None or c.surface_offset is None) and not is_box:
        raise PluginError(f"paper shape {c.paper!r} is not a box: set paper_size=(x, y) and surface_offset=height")
    size = c.paper_size if c.paper_size is not None else (2.0 * float(half[0]), 2.0 * float(half[1]))
    top = c.surface_offset if c.surface_offset is not None else float(half[2])
    return (float(size[0]), float(size[1])), float(top)


def _pen_on_paper(c: Ink.Config, lift: float = 0.0):
    half_pen = max(0.005, -float(c.tip[2]))                  # the tip sits at the block's bottom face
    def build(b: newton.ModelBuilder) -> None:
        paper = newton.ModelBuilder.ShapeConfig(mu=0.2)
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, 0.001), hx=0.1, hy=0.1, hz=0.001, cfg=paper, label=c.paper)
        pen = b.add_body(xform=body_xform(-0.05, 0.0, 0.002 + half_pen + lift), label=c.pen)
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.2)
        b.add_shape_box(pen, hx=half_pen, hy=half_pen, hz=half_pen, cfg=cfg)
    return build


def _push_pen(speed: float):
    def init(sim: Any) -> None:
        qd = sim.state.joint_qd.numpy()
        qd.reshape(-1, 6)[:, 0] = speed              # one free joint per world: (vx, vy, vz, wx, wy, wz)
        sim.state.joint_qd.assign(qd)
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    return init


def _ink(run: Any) -> Ink:
    return run.plugin


def _check_conserved(run: Any) -> LawResult:
    ink = _ink(run)
    total = run.trace.plugin(ink.label, "total") + run.trace.plugin(ink.label, "reservoir").astype(np.float64)
    start = max(float(np.abs(total[0]).max()), 1e-9)
    drift = float(np.abs(total - total[0]).max()) / start
    laid = float(run.trace.plugin(ink.label, "total")[-1].min())
    if laid <= 0.0:
        return LawResult(False, 0.0, 1e-4, "no ink was laid in the scene", "the pen must touch the sheet")
    return LawResult.within(drift, 1.0e-4, f"relative change of grid + reservoir ({laid:.3g} µL laid)",
                            "move ink, never create it: subtract from the reservoir what you add to the grid")


def _check_lifted(run: Any) -> LawResult:
    ink = _ink(run)
    laid = float(np.abs(run.trace.plugin(ink.label, "total")).max())
    return LawResult.within(laid, 0.0, "µL laid by a pen hovering 5 mm above the sheet",
                            "deposit only when the tip height above the surface is <= contact_tolerance")


def _check_track(run: Any) -> LawResult:
    ink = _ink(run)
    c, tr = ink.config, run.trace
    grid = tr.plugin(ink.label, "ink")[-1, 0]                  # world 0, last period [nx, ny]
    cols = np.nonzero(grid.sum(axis=1) > 0)[0]
    if cols.size == 0:
        return LawResult(False, math.nan, 0.0, "no ink laid", "the pen must touch the sheet while it slides")
    dx = 2.0 * float(ink.half[0]) / c.cells[0]
    span = (cols[-1] - cols[0] + 1) * dx
    touching = tr.plugin(ink.label, "touching")[:, 0] > 0
    first = int(np.argmax(touching))
    touching[max(first - 1, 0)] = touching.any()       # the period that ended on the first touch began at its pose
    pen = run.bodies(c.pen)[0]
    x = tr.body_q[touching, pen, 0]
    travel = float(x.max() - x.min()) + 2.0 * c.brush_radius if x.size else 0.0
    bound = 2.0 * dx + 0.002
    return LawResult.within(span - travel, bound, f"inked span {span * 1e3:.1f} mm vs tip travel + brush "
                            f"{travel * 1e3:.1f} mm", "map the tip into the paper frame before picking cells")


__all__ = ["Ink"]
