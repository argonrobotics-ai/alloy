# SPDX-License-Identifier: Apache-2.0
"""Concatenated device tables of a plug-in's charts, and each world's chart frames (interface note §2-3).

``build_tables`` turns the chart specs into one cell list and one face list (global indices), plus per world and chart
the body the chart rides on and its frame on that body. Geometry is in each chart's frame; the kernels map it to world
with ``geometry.chart_xform``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.removal.surface import patch_of
from alloy.plugins.surface_flow.charts import (EDGE_KINDS, PLANE, REVOLUTION, Chart, Geometry, PlaneChart,
                                               RevolutionChart, Topology, plane_geometry, revolution_geometry,
                                               topology)
from alloy.plugins.surface_flow.links import apply_links

log = logging.getLogger("alloy.plugins.surface_flow")


@dataclass
class ChartTables:
    """Device tables (W worlds, K charts, C cells, F faces). Host copies: ``starts``, ``shapes``, ``area_np``."""

    charts: wp.array            # [K] Chart
    cbody: wp.array             # [W, K] int32: the body a chart rides on, -1 static
    clocal: wp.array            # [W, K] transform: chart frame in the body (or world) frame
    com: wp.array               # [W, K] vec3: the body's COM in the chart frame (centrifugal term)
    cshape: wp.array            # [W, K] int32: the shape a chart lies on (-1: none named), for occluder casts to skip
    prof: wp.array              # [P] vec3: revolution face nodes (r, z, s)
    cpos: wp.array              # [C] vec3 centres, chart frame
    cnrm: wp.array              # [C] vec3 outward normals, chart frame
    area: wp.array              # [C] float64 m²
    curv: wp.array              # [C] float32 substrate mean curvature sum (+ convex) [1/m]
    cchart: wp.array            # [C] int32
    cface: wp.array             # [C] vec4i faces i-, i+, j-, j+
    fa: wp.array                # [F] int32
    fb: wp.array                # [F] int32
    ftype: wp.array             # [F] int32: 0 interior, 1 wall, 2 open
    fchart: wp.array            # [F] int32
    fline: wp.array             # [F] vec2i
    fpar: wp.array              # [F] vec2i
    fperp: wp.array             # [F] vec4i
    fpos: wp.array              # [F] vec3
    fdir: wp.array              # [F] vec3
    flen: wp.array              # [F] float32
    fdist: wp.array             # [F] float32
    fdira: wp.array             # [F] vec3: the a side's tangent (fdir off chart links; links.py)
    fdirb: wp.array             # [F] vec3: the b side's tangent, in the b cell's chart frame
    starts: list[int]           # host: first cell of each chart
    shapes: list[tuple[int, int]]   # host: (nx, ny) per chart
    area_np: np.ndarray         # host: [C] float64
    link_cells: np.ndarray      # host: [L, 2] the cells (a, b) each chart link pairs

    @property
    def cell_count(self) -> int:
        return int(self.area_np.shape[0])

    @property
    def face_count(self) -> int:
        return int(self.fa.shape[0])

    def cells_of(self, k: int) -> slice:
        nx, ny = self.shapes[k]
        return slice(self.starts[k], self.starts[k] + nx * ny)


def _check_edges(edges: Sequence[str], n: int, what: str) -> None:
    if len(edges) != n or any(e not in EDGE_KINDS for e in edges):
        raise PluginError(f"{what}: edges must be {n} of {sorted(EDGE_KINDS)}, got {tuple(edges)}")


def _plane(model: newton.Model, ctx: Any, spec: PlaneChart) -> tuple[Geometry, Topology, Chart, np.ndarray,
                                                                     np.ndarray, np.ndarray]:
    _check_edges(spec.edges, 4, f"plane chart {spec.shape!r}")
    nx, ny = (int(v) for v in spec.cells)
    if min(nx, ny) < 1:
        raise PluginError(f"plane chart {spec.shape!r}: cells must be >= 1, got {spec.cells}")
    patches = [patch_of(model, int(s), spec.shape, spec.size, spec.surface_offset)
               for s in ctx.shapes(spec.shape).numpy()]
    size = patches[0].size
    if any(abs(p.size[0] - size[0]) > 1e-9 or abs(p.size[1] - size[1]) > 1e-9 for p in patches):
        raise PluginError(f"plane chart {spec.shape!r}: the face differs between worlds; one chart needs one size")
    geo = plane_geometry(nx, ny, size)
    chart = Chart()
    chart.kind, chart.nx, chart.ny, chart.wrap = PLANE, nx, ny, 0
    chart.half = wp.vec2(0.5 * size[0], 0.5 * size[1])
    chart.step = wp.vec2(size[0] / nx, size[1] / ny)
    chart.prof_start = -1
    body = np.asarray([p.body for p in patches], np.int32)
    local = np.asarray([p.local for p in patches], np.float64)
    shape = np.asarray([p.shape for p in patches], np.int32)
    return geo, topology(nx, ny, False, spec.edges), chart, body, local, shape


def _revolution(ctx: Any, spec: RevolutionChart) -> tuple[Geometry, Topology, Chart, np.ndarray, np.ndarray,
                                                          np.ndarray]:
    _check_edges(spec.ends, 2, "revolution chart")
    geo = revolution_geometry(spec)
    nx, ny = len(geo.prof) - 1, int(spec.n_theta)
    chart = Chart()
    chart.kind, chart.nx, chart.ny, chart.wrap = REVOLUTION, nx, ny, 1
    chart.half = wp.vec2(0.0, 0.0)
    chart.step = wp.vec2(float(geo.prof[1, 2] - geo.prof[0, 2]), 2.0 * np.pi / ny)
    W = ctx.world_count
    body = ctx.bodies(spec.body).numpy().astype(np.int32) if spec.body else np.full(W, -1, np.int32)
    if len(spec.frame) != 7:
        raise PluginError(f"revolution chart: frame must be (x, y, z, qx, qy, qz, qw), got {spec.frame}")
    local = np.tile(np.asarray(spec.frame, np.float64), (W, 1))
    shape = ctx.shapes(spec.shape).numpy().astype(np.int32) if spec.shape else np.full(W, -1, np.int32)
    return geo, topology(nx, ny, True, (spec.ends[0], spec.ends[1], "wall", "wall")), chart, body, local, shape


def _com_in_chart(model: newton.Model, body: np.ndarray, local: np.ndarray) -> np.ndarray:
    com = np.zeros((len(body), 3))
    coms = model.body_com.numpy() if model.body_count else np.zeros((0, 3))
    for w, (b, xf) in enumerate(zip(body, local)):
        if b < 0:
            continue
        X = wp.transform(wp.vec3(*xf[:3]), wp.quat(*xf[3:]))
        com[w] = np.asarray(wp.transform_point(wp.transform_inverse(X), wp.vec3(*coms[b])))
    return com


def build_tables(ctx: Any, specs: Sequence[PlaneChart | RevolutionChart], links: Sequence[Any] = ()) -> ChartTables:
    """Concatenate ``specs`` (in order) into device tables for ``ctx``'s worlds, joined by ``links`` (links.py)."""
    if not specs:
        raise PluginError("surface_flow needs at least one chart")
    model, W = ctx.model, ctx.world_count
    parts = []
    for spec in specs:
        if isinstance(spec, PlaneChart):
            parts.append(_plane(model, ctx, spec))
        elif isinstance(spec, RevolutionChart):
            parts.append(_revolution(ctx, spec))
        else:
            raise PluginError(f"unknown chart {spec!r}: use PlaneChart or RevolutionChart")
    c0 = f0 = p0 = 0
    charts, starts, shapes = [], [], []
    cat: dict[str, list[np.ndarray]] = {k: [] for k in ("cpos", "cnrm", "area", "curv", "cchart", "cface", "fa", "fb",
                                                        "ftype", "fchart", "fline", "fpar", "fperp", "fpos", "fdir",
                                                        "fdira", "fdirb", "flen", "fdist", "prof")}
    for k, (geo, top, chart, *_) in enumerate(parts):
        n, f = len(geo.area), len(top.fa)
        chart.cell_start, chart.iface_start = c0, f0
        chart.jface_start = f0 + (int(chart.nx) + 1) * int(chart.ny)
        chart.prof_start = p0 if int(chart.kind) == int(REVOLUTION) else -1
        charts.append(chart)
        starts.append(c0)
        shapes.append((int(chart.nx), int(chart.ny)))
        shift = lambda a, by: np.where(a >= 0, a + by, -1)  # noqa: E731
        for name in ("cpos", "cnrm", "area", "curv", "fpos", "fdir", "flen", "fdist"):
            cat[name].append(getattr(geo, name))
        for name in ("fdira", "fdirb"):                                # each side's tangent (fdir off revolutions)
            side = getattr(geo, name)
            cat[name].append(geo.fdir if side is None else side)
        cat["prof"].append(geo.prof)
        cat["cchart"].append(np.full(n, k))
        cat["fchart"].append(np.full(f, k))
        cat["cface"].append(shift(top.cface, f0))
        for name in ("fa", "fb"):
            cat[name].append(shift(getattr(top, name), c0))
        for name in ("fline", "fpar", "fperp"):
            cat[name].append(shift(getattr(top, name), f0))
        cat["ftype"].append(top.ftype)
        c0, f0, p0 = c0 + n, f0 + f, p0 + len(geo.prof)
    j = {k: np.concatenate(v) for k, v in cat.items()}
    body = np.stack([p[3] for p in parts], 1)                          # [W, K]
    local = np.stack([p[4] for p in parts], 1)                         # [W, K, 7]
    com = np.stack([_com_in_chart(model, p[3], p[4]) for p in parts], 1)
    shape = np.stack([p[5] for p in parts], 1)
    dira, dirb, pairs = apply_links(j, links, specs, charts, body, local)
    dev = ctx.device

    def arr(a: np.ndarray, dtype: Any) -> wp.array:
        return wp.array(np.ascontiguousarray(a), dtype=dtype, device=dev)

    log.info("surface_flow: %d charts, %d links (%d cell pairs), %d cells, %d faces per world (%d worlds)", len(specs),
             len(links), len(pairs), c0, f0, W)
    return ChartTables(
        charts=wp.array(charts, dtype=Chart, device=dev), cbody=arr(body.astype(np.int32), wp.int32),
        clocal=arr(local.astype(np.float32), wp.transform), com=arr(com.astype(np.float32), wp.vec3),
        cshape=arr(shape.astype(np.int32), wp.int32),
        prof=arr(j["prof"].reshape(-1, 3).astype(np.float32), wp.vec3),
        cpos=arr(j["cpos"].astype(np.float32), wp.vec3), cnrm=arr(j["cnrm"].astype(np.float32), wp.vec3),
        area=arr(j["area"].astype(np.float64), wp.float64), curv=arr(j["curv"].astype(np.float32), wp.float32),
        cchart=arr(j["cchart"].astype(np.int32), wp.int32), cface=arr(j["cface"].astype(np.int32), wp.vec4i),
        fa=arr(j["fa"].astype(np.int32), wp.int32), fb=arr(j["fb"].astype(np.int32), wp.int32),
        ftype=arr(j["ftype"].astype(np.int32), wp.int32), fchart=arr(j["fchart"].astype(np.int32), wp.int32),
        fline=arr(j["fline"].astype(np.int32), wp.vec2i), fpar=arr(j["fpar"].astype(np.int32), wp.vec2i),
        fperp=arr(j["fperp"].astype(np.int32), wp.vec4i), fpos=arr(j["fpos"].astype(np.float32), wp.vec3),
        fdir=arr(j["fdir"].astype(np.float32), wp.vec3), flen=arr(j["flen"].astype(np.float32), wp.float32),
        fdist=arr(j["fdist"].astype(np.float32), wp.float32), fdira=arr(dira.astype(np.float32), wp.vec3),
        fdirb=arr(dirb.astype(np.float32), wp.vec3),
        starts=starts, shapes=shapes, area_np=j["area"].astype(np.float64), link_cells=pairs)


__all__ = ["ChartTables", "build_tables"]
