# SPDX-License-Identifier: Apache-2.0
"""Read-only views of a ``SurfaceFlow`` plug-in's film for renderers (the camera's film look, ``alloy.camera.films``).

:func:`film_view` describes the charts once (cell corners and normals in each chart's frame, from the cell centres the
plug-in already built) and hands out the plug-in's own device arrays: film depth, species mass, soil, the chart
frames. Nothing here writes plug-in state, launches work or synchronises the device during a step; a renderer reads
the arrays between steps, so the simulation is the same bit for bit with or without it.

Example::

    from alloy.plugins.surface_flow.render_view import film_plugins, film_view
    views = [film_view(p) for p in film_plugins(sim.plugins)]
    views[0].depth            # wp.array [W, C] float64: the film depth (m), the plug-in's own buffer
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.surface_flow.charts import PlaneChart, resample_profile
from alloy.plugins.surface_flow.host_api import H_WET


@dataclass(frozen=True)
class ChartGrid:
    """One chart's cells: cell ``c`` of the chart is plug-in cell ``cell_start + c`` (``c = i * ny + j``).
    ``corners`` (n, 4, 3) and ``normals`` (n, 3) are in the chart frame; corners run counter-clockwise seen from the
    wetted side (along the normal)."""

    index: int
    kind: str                   # "plane" | "revolution"
    shape: str | None           # the shape the chart lies on (None: none named)
    cell_start: int
    nx: int
    ny: int
    corners: np.ndarray
    normals: np.ndarray


@dataclass(frozen=True)
class FilmView:
    """What a renderer reads of one film (device arrays are the plug-in's, read-only; ``None`` when it has none).
    ``cbody`` [W, K] int32 (the body index in ``state.body_q`` a chart rides on, -1 static) and ``clocal`` [W, K]
    transform place each chart; ``depth`` [W, C] float64 is the film depth (m) over ``base`` [W, C] float32 (the
    fixed base under the film along the normal, m: the film's surface is at base + depth); ``species_mass`` [W, C,
    S] float64 the species per area in ``species`` order (paint solids in m of dry film, soil in kg/m²); ``soil``
    / ``soil_start`` [W, C] float64 the adhered soil now and at the start (kg/m², the process stage's);
    ``hide_dft`` / ``phi_m`` the drying stage's hiding thickness (m) and packing fraction; ``wet_depth`` the depth
    under which a cell is dry."""

    label: str
    material: str
    worlds: int
    cells: int
    charts: tuple[ChartGrid, ...]
    cbody: wp.array
    clocal: wp.array
    depth: wp.array
    base: wp.array
    species: tuple[str, ...]
    species_mass: wp.array | None
    soil: wp.array | None
    soil_start: wp.array | None
    solids: int                 # index of the drying stage's solids species, -1 without drying
    hide_dft: float | None
    phi_m: float | None
    wet_depth: float = H_WET


def film_plugins(host: Any) -> list[Any]:
    """The ``SurfaceFlow`` plug-ins of a plug-in host (``sim.plugins``), in label order; [] for None or no films."""
    from alloy.plugins.surface_flow.plugin import SurfaceFlow  # noqa: PLC0415 - avoid a cycle at package import
    plugins = getattr(host, "plugins", None)
    if not isinstance(plugins, dict):
        return []
    return [plugins[k] for k in sorted(plugins) if isinstance(plugins[k], SurfaceFlow)]


def film_view(flow: Any) -> FilmView:
    """The read-only view of a set-up ``SurfaceFlow`` (after the simulation is built)."""
    t = flow.tables
    centres = t.cpos.numpy()                        # static geometry, read once (no state)
    normals = t.cnrm.numpy()
    charts = tuple(_grid(k, spec, t.starts[k], t.shapes[k], centres, normals)
                   for k, spec in enumerate(flow.config.charts))
    names = tuple(str(s) for s in flow.config.species)
    pr = flow.process
    drying = getattr(getattr(pr, "spec", None), "drying", None)
    soil = getattr(pr, "soil", None) if pr is not None else None
    W, C = int(flow.h.shape[0]), int(flow.h.shape[1])
    return FilmView(label=str(flow.label), material=str(getattr(flow.mat, "name", flow.config.material)), worlds=W,
                    cells=C, charts=charts, cbody=t.cbody, clocal=t.clocal, depth=flow.h, base=flow.base,
                    species=names,
                    species_mass=flow.sp.mass if names else None, soil=soil,
                    soil_start=getattr(pr, "soil0", None) if soil is not None else None,
                    solids=int(pr.k_solids) if drying is not None else -1,
                    hide_dft=float(drying.hide_dft) if drying is not None else None,
                    phi_m=float(drying.phi_m) if drying is not None else None)


def _grid(k: int, spec: Any, start: int, shape: tuple[int, int], centres: np.ndarray, normals: np.ndarray) -> ChartGrid:
    nx, ny = int(shape[0]), int(shape[1])
    cells = slice(int(start), int(start) + nx * ny)
    if isinstance(spec, PlaneChart):
        corners, kind, name = _plane_corners(centres[cells], nx, ny, spec.size), "plane", spec.shape
    else:
        corners, kind, name = _revolution_corners(spec, nx, ny), "revolution", spec.shape or spec.body
    corners = _outward(corners, normals[cells])
    return ChartGrid(k, kind, name, int(start), nx, ny, corners.astype(np.float32), normals[cells].astype(np.float32))


def _half_extent(c: np.ndarray, n: int, other_spacing: float | None) -> float:
    """Half the chart's length along one axis from its cell centres ``-h + (i + 1/2) d``, ``d = 2 h / n``."""
    if n > 1:
        return float(c.max()) / (1.0 - 1.0 / n)
    return 0.5 * (other_spacing or 1.0e-3)


def _plane_corners(c: np.ndarray, nx: int, ny: int, size: tuple[float, float] | None) -> np.ndarray:
    """Cell corners from the chart's ``size`` (d = size / n, charts.plane_geometry), else from the cell centres."""
    if size is not None:
        dx, dy = float(size[0]) / nx, float(size[1]) / ny
    else:
        hx = _half_extent(c[:, 0], nx, None)
        hy = _half_extent(c[:, 1], ny, 2.0 * hx / nx)
        if nx == 1:
            hx = _half_extent(c[:, 0], nx, 2.0 * hy / ny)
        dx, dy = 2.0 * hx / nx, 2.0 * hy / ny
    off = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]) * (dx, dy)
    out = np.zeros((len(c), 4, 3))
    out[:, :, :2] = c[:, None, :2] + off[None]
    out[:, :, 2] = c[:, None, 2]
    return out


def _revolution_corners(spec: Any, nx: int, ny: int) -> np.ndarray:
    """Cell (i, j) spans profile nodes i .. i + 1 and angles j dθ .. (j + 1) dθ (charts.revolution_geometry)."""
    nodes, _, _ = resample_profile(spec.profile, spec.ds)
    th = np.arange(ny + 1) * (2.0 * math.pi / ny)
    r, z = nodes[:, 0], nodes[:, 1]

    def at(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return np.stack([r[i] * np.cos(th[j]), r[i] * np.sin(th[j]), z[i]], -1)

    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    i, j = i.ravel(), j.ravel()
    return np.stack([at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)], 1)


def _outward(corners: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Reverse the corner order of cells whose winding faces away from the outward normal."""
    tri_n = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flip = np.einsum("ij,ij->i", tri_n, normals) < 0.0
    out = corners.copy()
    out[flip] = corners[flip][:, ::-1]
    return out


__all__ = ["ChartGrid", "FilmView", "film_plugins", "film_view"]
