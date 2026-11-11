# SPDX-License-Identifier: Apache-2.0
"""Charts: the surfaces a film lives on, as structured finite-volume grids in a body's frame.

Two kinds (design note §2, interface note §2-3):

- ``PlaneChart``: a flat face of a shape (``removal.surface.patch_of``: a box's top face by default), ``nx x ny``
  cells along the face's local x and y, +z out of the material (the wetted side);
- ``RevolutionChart``: a profile polyline ``(r, z)`` in the chart frame swept about its z axis. Cells run along the
  profile (``i``, arc length ``s``) and around it (``j``, θ wraps). The material is on the profile's right, so the
  outward normal is the tangent turned +90° in the (r, z) plane: a glass is one profile (inside floor outward, inside
  wall up, over the rim, outside wall down, outside base inward). The profile must stay off the axis
  (``r >= ds·n_theta/(2π)``, so the innermost ring's cells are about square); a cap chart linked to it is phase 2.

All charts of a plug-in are concatenated: chart ``k`` owns cells ``[cell_start, cell_start + nx·ny)`` with cell
``(i, j) = cell_start + i·ny + j``. Faces are numbered per chart: first the ``i``-faces ``(nx + 1)·ny`` (face ``(i, j)``
between cells ``(i - 1, j)`` and ``(i, j)``), then the ``j``-faces (``nx·ny`` when θ wraps, else ``nx·(ny + 1)``). A
face's cell ``a`` is on its minus side, ``b`` on its plus side (-1 outside the chart), and its direction points from
``a`` to ``b``. Geometry (centres, normals, face directions) is in the chart frame; ``chart_xform`` maps it to world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError

PLANE = wp.constant(0)
REVOLUTION = wp.constant(1)
EDGE_KINDS = {"wall": 1, "open": 2}                     # face types; 0 = interior


@dataclass(frozen=True)
class PlaneChart:
    """A flat face: the shape ``shape``'s box top (or ``size`` at ``surface_offset``), ``cells`` = (nx, ny).

    ``edges`` = (x-, x+, y-, y+): "wall" (no flux) or "open" (film leaves; booked as spilled)."""

    shape: str
    cells: tuple[int, int] = (64, 64)
    size: tuple[float, float] | None = None
    surface_offset: float | None = None
    edges: tuple[str, str, str, str] = ("wall", "wall", "wall", "wall")
    contact_angle: float | None = None                  # degrees; None: the material's


@dataclass(frozen=True)
class RevolutionChart:
    """A surface of revolution: ``profile`` (r, z) nodes [m] in the chart frame, material on the right.

    ``frame`` places the chart in the body ``body`` (None: the world) as (x, y, z, qx, qy, qz, qw). ``ds`` is the
    target cell length along the profile; ``ends`` = (start, end) of the profile: "wall" or "open". ``shape`` names
    the collision shape the chart lies on (its hull), so jets cast through it are not blocked by it."""

    profile: tuple[tuple[float, float], ...]
    n_theta: int = 32
    ds: float = 2.0e-3
    body: str | None = None
    frame: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    ends: tuple[str, str] = ("wall", "wall")
    contact_angle: float | None = None
    shape: str | None = None


@dataclass(frozen=True)
class Link:
    """Joins edge ``edge_a`` of chart ``a`` to edge ``edge_b`` of chart ``b`` (design note §2.1): the film crosses
    between them through the same face kernel. Charts by index or name (a plane chart's ``shape``, a revolution
    chart's ``shape``, else its ``body``). Edges: "i-", "i+", "j-", "j+" ((x-, x+, y-, y+) of a plane chart),
    "start" or "end" of a revolution chart's profile. A link joins a "+" edge to a "-" edge whose cells run the same
    way along it (a chart whose edge runs the other way is turned: a plate's underside is a shape turned 180° about
    y, not x); both charts ride one body (or both are static), and the edges have as many cells and meet."""

    a: int | str
    edge_a: str
    b: int | str
    edge_b: str


def link_spec(spec: Any) -> Link:
    """A link from a ``Link``, a 4-tuple ``(a, edge_a, b, edge_b)`` or a dict (JSON, cards)."""
    if isinstance(spec, Link):
        return spec
    try:
        if isinstance(spec, dict):
            return Link(**spec)
        if isinstance(spec, (tuple, list)) and len(spec) == 4:
            return Link(*spec)
    except TypeError as e:
        raise PluginError(f"a link takes a, edge_a, b, edge_b: {e}") from None
    raise PluginError(f"a link is a Link, (a, edge_a, b, edge_b) or a dict, got {spec!r}")


@wp.struct
class Chart:
    kind: wp.int32
    nx: wp.int32
    ny: wp.int32
    cell_start: wp.int32
    iface_start: wp.int32
    jface_start: wp.int32
    wrap: wp.int32                                      # 1: j (θ) wraps
    half: wp.vec2                                       # plane: half extents
    step: wp.vec2                                       # plane: (dx, dy); revolution: (ds, dθ)
    prof_start: wp.int32                                # revolution: first profile node (nx + 1 nodes: r, z, s)


@dataclass
class Geometry:
    """Host tables of one chart (chart frame), before concatenation."""

    cpos: np.ndarray                                    # [n, 3] cell centres
    cnrm: np.ndarray                                    # [n, 3] outward normals
    area: np.ndarray                                    # [n] m²
    curv: np.ndarray                                    # [n] substrate mean curvature sum, + when convex [1/m]
    fpos: np.ndarray                                    # [f, 3] face centres
    fdir: np.ndarray                                    # [f, 3] a -> b: the centre-to-centre chord / fdist (gravity)
    flen: np.ndarray                                    # [f] face length [m]
    fdist: np.ndarray                                   # [f] centre-to-centre distance [m]
    prof: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # revolution: face nodes (r, z, s)
    fdira: np.ndarray | None = None                     # [f, 3] the a side's unit tangent (None: fdir)
    fdirb: np.ndarray | None = None                     # [f, 3] the b side's unit tangent (None: fdir)


@dataclass
class Topology:
    """Index tables of one chart (local indices; -1 = none)."""

    fa: np.ndarray                                      # [f] cell on the minus side
    fb: np.ndarray                                      # [f] cell on the plus side
    ftype: np.ndarray                                   # [f] 0 interior, 1 wall, 2 open
    fline: np.ndarray                                   # [f, 2] collinear faces beyond a and beyond b
    fpar: np.ndarray                                    # [f, 2] parallel faces at -/+ in the transverse direction
    fperp: np.ndarray                                   # [f, 4] transverse faces: a-, a+, b-, b+
    cface: np.ndarray                                   # [n, 4] a cell's faces: i-, i+, j-, j+


def topology(nx: int, ny: int, wrap: bool, edges: tuple[str, str, str, str]) -> Topology:
    """Face and neighbour tables of an ``nx x ny`` grid; ``edges`` = (i-, i+, j-, j+) types (j ignored if wrap)."""
    nfi = (nx + 1) * ny
    nfj = nx * ny if wrap else nx * (ny + 1)
    jw = ny if wrap else ny + 1

    def cell(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        if wrap:
            j = np.mod(j, ny)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
        return np.where(ok, i * ny + j, -1)

    def iface(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        if wrap:
            j = np.mod(j, ny)
        ok = (i >= 0) & (i <= nx) & (j >= 0) & (j < ny)
        return np.where(ok, i * ny + j, -1)

    def jface(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        if wrap:
            j = np.mod(j, ny)
        ok = (i >= 0) & (i < nx) & (j >= 0) & (j < jw)
        return np.where(ok, nfi + i * jw + j, -1)

    I, J = np.meshgrid(np.arange(nx + 1), np.arange(ny), indexing="ij")
    I, J = I.ravel(), J.ravel()                                         # i-faces in order i·ny + j
    a_i, b_i = cell(I - 1, J), cell(I, J)
    t_i = np.zeros(nfi, np.int32)
    t_i[I == 0] = EDGE_KINDS[edges[0]]
    t_i[I == nx] = EDGE_KINDS[edges[1]]
    line_i = np.stack([iface(I - 1, J), iface(I + 1, J)], 1)
    par_i = np.stack([iface(I, J - 1), iface(I, J + 1)], 1)
    perp_i = np.stack([jface(I - 1, J), jface(I - 1, J + 1), jface(I, J), jface(I, J + 1)], 1)
    perp_i[:, :2][a_i < 0] = -1
    perp_i[:, 2:][b_i < 0] = -1

    I, J = np.meshgrid(np.arange(nx), np.arange(jw), indexing="ij")
    I, J = I.ravel(), J.ravel()                                         # j-faces in order i·jw + j
    a_j, b_j = cell(I, J - 1), cell(I, J)
    t_j = np.zeros(nfj, np.int32)
    if not wrap:
        t_j[J == 0] = EDGE_KINDS[edges[2]]
        t_j[J == ny] = EDGE_KINDS[edges[3]]
    line_j = np.stack([jface(I, J - 1), jface(I, J + 1)], 1)
    if not wrap:
        line_j[J == 0, 0] = -1
        line_j[J == ny, 1] = -1
    par_j = np.stack([jface(I - 1, J), jface(I + 1, J)], 1)
    perp_j = np.stack([iface(I, J - 1), iface(I + 1, J - 1), iface(I, J), iface(I + 1, J)], 1)
    perp_j[:, :2][a_j < 0] = -1
    perp_j[:, 2:][b_j < 0] = -1

    Ic, Jc = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    Ic, Jc = Ic.ravel(), Jc.ravel()
    cface = np.stack([iface(Ic, Jc), iface(Ic + 1, Jc), jface(Ic, Jc), jface(Ic, Jc + 1)], 1)
    return Topology(fa=np.concatenate([a_i, a_j]).astype(np.int32), fb=np.concatenate([b_i, b_j]).astype(np.int32),
                    ftype=np.concatenate([t_i, t_j]), fline=np.concatenate([line_i, line_j]).astype(np.int32),
                    fpar=np.concatenate([par_i, par_j]).astype(np.int32),
                    fperp=np.concatenate([perp_i, perp_j]).astype(np.int32), cface=cface.astype(np.int32))


def plane_geometry(nx: int, ny: int, size: tuple[float, float]) -> Geometry:
    hx, hy = 0.5 * size[0], 0.5 * size[1]
    dx, dy = size[0] / nx, size[1] / ny
    xc, yc = -hx + (np.arange(nx) + 0.5) * dx, -hy + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    n = nx * ny
    cpos = np.stack([X.ravel(), Y.ravel(), np.zeros(n)], 1)
    Xi, Yi = np.meshgrid(-hx + np.arange(nx + 1) * dx, yc, indexing="ij")
    Xj, Yj = np.meshgrid(xc, -hy + np.arange(ny + 1) * dy, indexing="ij")
    fi, fj = (nx + 1) * ny, nx * (ny + 1)
    fpos = np.concatenate([np.stack([Xi.ravel(), Yi.ravel(), np.zeros(fi)], 1),
                           np.stack([Xj.ravel(), Yj.ravel(), np.zeros(fj)], 1)])
    fdir = np.concatenate([np.tile([1.0, 0.0, 0.0], (fi, 1)), np.tile([0.0, 1.0, 0.0], (fj, 1))])
    return Geometry(cpos=cpos, cnrm=np.tile([0.0, 0.0, 1.0], (n, 1)), area=np.full(n, dx * dy), curv=np.zeros(n),
                    fpos=fpos, fdir=fdir, flen=np.concatenate([np.full(fi, dy), np.full(fj, dx)]),
                    fdist=np.concatenate([np.full(fi, dx), np.full(fj, dy)]))


def resample_profile(profile: tuple[tuple[float, float], ...], ds: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Face nodes ``(r, z, s)`` at equal arc length ``ds'`` (about ``ds``) and the cell centres' ``(r, z, s)``."""
    p = np.asarray(profile, np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2:
        raise PluginError(f"a revolution profile needs >= 2 (r, z) nodes, got {p.shape}")
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    if (seg <= 0.0).any():
        raise PluginError("a revolution profile repeats a node")
    s_nodes = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(s_nodes[-1])
    nx = max(int(round(length / ds)), 1)
    step = length / nx

    def at(s: np.ndarray) -> np.ndarray:
        return np.stack([np.interp(s, s_nodes, p[:, 0]), np.interp(s, s_nodes, p[:, 1]), s], 1)

    return at(np.arange(nx + 1) * step), at((np.arange(nx) + 0.5) * step), step


def _tangent_angle(nodes: np.ndarray) -> np.ndarray:
    """Unwrapped angle φ of the profile tangent (r, z) per segment between consecutive face nodes."""
    d = np.diff(nodes[:, :2], axis=0)
    return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def revolution_geometry(chart: RevolutionChart) -> Geometry:
    nodes, cells, step = resample_profile(chart.profile, chart.ds)
    nx, ny = len(cells), int(chart.n_theta)
    if ny < 3:
        raise PluginError(f"a revolution chart needs n_theta >= 3, got {ny}")
    dth = 2.0 * math.pi / ny
    r_min = float(min(nodes[:, 0].min(), cells[:, 0].min()))
    if r_min < 0.5 * step / dth:
        raise PluginError(f"revolution chart: the profile comes within r = {r_min * 1e3:.3g} mm of the axis, so the "
                          f"innermost ring's cells (r·dθ) are under half a cell length ({step * 1e3:.3g} mm); keep "
                          f"r >= {0.5 * step / dth * 1e3:.3g} mm or use fewer n_theta (a cap chart is phase 2)")
    phi = _tangent_angle(nodes)                                         # [nx] per cell (segment between its faces)
    # tangent at the face nodes: the mean of the adjoining cells' (ends: the end cell's)
    phi_f = np.concatenate([[phi[0]], 0.5 * (phi[:-1] + phi[1:]), [phi[-1]]])
    # meridian curvature, + when convex (the tangent turns clockwise): -dφ/ds across each cell
    kappa_m = -(phi_f[1:] - phi_f[:-1]) / step
    n_r, n_z = -np.sin(phi), np.cos(phi)                                # outward normal: tangent turned +90°
    th_c = (np.arange(ny) + 0.5) * dth
    th_f = np.arange(ny) * dth
    rc, zc = cells[:, 0], cells[:, 1]
    cpos = np.stack([np.outer(rc, np.cos(th_c)), np.outer(rc, np.sin(th_c)), np.repeat(zc[:, None], ny, 1)], -1)
    cnrm = np.stack([np.outer(n_r, np.cos(th_c)), np.outer(n_r, np.sin(th_c)), np.repeat(n_z[:, None], ny, 1)], -1)
    ring = step / 6.0 * (nodes[:-1, 0] + 4.0 * rc + nodes[1:, 0]) * dth            # Simpson: ∫ r ds · dθ
    curv = np.repeat((kappa_m + n_r / rc)[:, None], ny, 1)
    # i-faces at the nodes, j index = cell centre angle
    rf, zf = nodes[:, 0], nodes[:, 1]
    fpos_i = np.stack([np.outer(rf, np.cos(th_c)), np.outer(rf, np.sin(th_c)), np.repeat(zf[:, None], ny, 1)], -1)
    # each side's own tangent (its cell's), and gravity along the chord between the centres over their distance: the
    # unit mean tangent was 1/cos(turn/2) too steep at a corner (WP9's links review: 6.94 vs 4.905 at 90°)
    phi_a, phi_b = np.concatenate([[phi[0]], phi]), np.concatenate([phi, [phi[-1]]])

    def ring_dir(r_part: np.ndarray, z_part: np.ndarray) -> np.ndarray:
        return np.stack([np.outer(r_part, np.cos(th_c)), np.outer(r_part, np.sin(th_c)),
                         np.repeat(z_part[:, None], ny, 1)], -1)
    chord = np.diff(cells[:, :2], axis=0) / step                       # interior faces: (Δr, Δz) / fdist
    g_r = np.concatenate([[np.cos(phi[0])], chord[:, 0], [np.cos(phi[-1])]])
    g_z = np.concatenate([[np.sin(phi[0])], chord[:, 1], [np.sin(phi[-1])]])
    straight = np.abs(phi_a - phi_b) < 1.0e-12                         # no turn: the unit tangent, as before
    g_r, g_z = np.where(straight, np.cos(phi_f), g_r), np.where(straight, np.sin(phi_f), g_z)
    fdir_i = ring_dir(g_r, g_z)
    dira_i, dirb_i = ring_dir(np.cos(phi_a), np.sin(phi_a)), ring_dir(np.cos(phi_b), np.sin(phi_b))
    flen_i = np.repeat((rf * dth)[:, None], ny, 1)
    # j-faces at angle j·dθ through each cell centre's ring
    fpos_j = np.stack([np.outer(rc, np.cos(th_f)), np.outer(rc, np.sin(th_f)), np.repeat(zc[:, None], ny, 1)], -1)
    fdir_j = np.stack([-np.tile(np.sin(th_f), (nx, 1)), np.tile(np.cos(th_f), (nx, 1)), np.zeros((nx, ny))], -1)
    fdist_j = np.repeat((rc * dth)[:, None], ny, 1)
    return Geometry(cpos=cpos.reshape(-1, 3), cnrm=cnrm.reshape(-1, 3), area=np.repeat(ring, ny), curv=curv.ravel(),
                    fpos=np.concatenate([fpos_i.reshape(-1, 3), fpos_j.reshape(-1, 3)]),
                    fdir=np.concatenate([fdir_i.reshape(-1, 3), fdir_j.reshape(-1, 3)]),
                    flen=np.concatenate([flen_i.ravel(), np.full(nx * ny, step)]),
                    fdist=np.concatenate([np.full((nx + 1) * ny, step), fdist_j.ravel()]), prof=nodes,
                    fdira=np.concatenate([dira_i.reshape(-1, 3), fdir_j.reshape(-1, 3)]),
                    fdirb=np.concatenate([dirb_i.reshape(-1, 3), fdir_j.reshape(-1, 3)]))


def chart_spec(spec: Any) -> PlaneChart | RevolutionChart:
    """A chart from a spec or a dict (JSON, cards, ``dataclasses.asdict``): ``profile`` makes a revolution chart,
    else ``shape`` a plane chart."""
    if isinstance(spec, (PlaneChart, RevolutionChart)):
        return spec
    if isinstance(spec, dict):
        if "profile" in spec:
            return RevolutionChart(**{k: tuple(map(tuple, v)) if k == "profile" else v for k, v in spec.items()})
        if "shape" in spec:
            return PlaneChart(**spec)
    raise PluginError(f"a chart is a PlaneChart, a RevolutionChart or a dict with 'shape' or 'profile', got {spec!r}")


__all__ = ["Chart", "EDGE_KINDS", "Geometry", "Link", "PLANE", "PlaneChart", "REVOLUTION", "RevolutionChart",
           "Topology", "chart_spec", "link_spec", "plane_geometry", "resample_profile", "revolution_geometry",
           "topology"]
