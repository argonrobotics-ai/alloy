# SPDX-License-Identifier: Apache-2.0
"""Chart links (design note §2.1): an edge of one chart joined to an edge of another, so the film crosses between them.

``apply_links`` rewrites the concatenated host tables of ``tables.build_tables``. For each pair of edge cells, ``a`` on
chart A's "+" edge and ``b`` on chart B's "-" edge, in order along the edges:

- A's boundary face of ``a`` becomes the link face: interior, ``fa = a``, ``fb = b``, ``fchart = A``, its geometry in
  A's frame (B's mapped by ``T_AB = clocal_A⁻¹·clocal_B``, the same in every world).
  - Distance: the arc between the cell centres. That is half a cell on each side, plus the edges' gap across the
    edge (a plate's thickness at its lip).
  - Direction: the chord between the centres over that distance, so gravity's pull is the true drop over the arc at
    any fold (WP15's review).
  - Centre and length: the mean of the two old faces'.
- Each side reads the face along its own edge tangent (``dira``, A's outward one, and ``dirb``, B's inward one in
  B's frame) in the cells' vector sums, the supply's push and the tools' slide (``cell_dir``).
- A fold adds its turning to the seam cells' substrate curvature: ``±(θ/2)/dx`` on each side, + where it is convex.
- B's old boundary face becomes a wall of zero length, which no cell or stencil names any more.
- The link face's stencils: ``fline`` = (A's face beyond a, B's face beyond b), ``fpar`` = the link faces of a's edge
  neighbours, ``fperp`` = (a's transverse faces, b's). The transverse faces of a and b name each other as ``fpar``
  across the seam. The edges run the same way, so no stencil changes orientation.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.charts import EDGE_KINDS, Chart, PlaneChart, link_spec
from alloy.plugins.surface_flow.kernels_structs import Faces

FLAT = 1.0e-6                   # rad: a turn under this between linked edge tangents is a straight join
EDGES = {"i-": (0, -1), "i+": (0, 1), "j-": (1, -1), "j+": (1, 1), "start": (0, -1), "end": (0, 1)}


@wp.func
def cell_dir(faces: Faces, f: int, c: int) -> wp.vec3:
    """Face f's direction (a -> b) along cell c's side, in c's chart frame (off links: ``dir``; links.py)."""
    if faces.b[f] == c:
        return faces.dirb[f]
    return faces.dira[f]


def chart_name(spec: Any) -> str | None:
    """The name a link gives a chart by: a plane chart's shape, a revolution chart's shape, else its body."""
    if isinstance(spec, PlaneChart):
        return spec.shape
    return spec.shape or spec.body


def _index(key: Any, specs: Sequence[Any], what: str) -> int:
    if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
        if not 0 <= int(key) < len(specs):
            raise PluginError(f"{what}: chart {key} does not exist ({len(specs)} charts)")
        return int(key)
    hits = [k for k, s in enumerate(specs) if chart_name(s) == key]
    if len(hits) != 1:
        raise PluginError(f"{what}: chart {key!r} names {len(hits)} charts; give its index instead")
    return hits[0]


def edge_of(chart: Chart, edge: str, what: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """The global cells and boundary faces along ``edge`` of ``chart``, in order, the edge's sign (+1 / -1) and its
    axis (0: an i edge, 1: a j edge)."""
    wrap = int(chart.wrap) == 1
    if edge not in EDGES or (wrap and edge[0] == "j") or (not wrap and edge in ("start", "end")):
        kinds = '"start", "end"' if wrap else '"i-", "i+", "j-", "j+"'
        raise PluginError(f"{what}: edge {edge!r} is not one of this chart's: {kinds}")
    axis, sign = EDGES[edge]
    nx, ny = int(chart.nx), int(chart.ny)
    if axis == 0:
        j = np.arange(ny)
        cells = (0 if sign < 0 else nx - 1) * ny + j
        faces = (0 if sign < 0 else nx) * ny + j
    else:
        i = np.arange(nx)
        cells = i * ny + (0 if sign < 0 else ny - 1)
        faces = (nx + 1) * ny + i * (ny + 1) + (0 if sign < 0 else ny)
    return cells + int(chart.cell_start), faces + int(chart.iface_start), sign, axis


def _rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def relative(local_a: np.ndarray, local_b: np.ndarray, what: str) -> tuple[np.ndarray, np.ndarray]:
    """``T_AB`` (rotation, translation: a point of B's frame in A's), which must be the same in every world."""
    out = []
    for xa, xb in zip(local_a, local_b):
        ra = _rot(xa[3:])
        out.append((ra.T @ _rot(xb[3:]), ra.T @ (np.asarray(xb[:3]) - np.asarray(xa[:3]))))
    R, t = out[0]
    if any(not (np.allclose(r, R, rtol=0.0, atol=1e-9) and np.allclose(s, t, rtol=0.0, atol=1e-9)) for r, s in out):
        raise PluginError(f"{what}: the two charts sit differently on their body in different worlds")
    return R, t


def apply_links(j: dict[str, np.ndarray], links: Sequence[Any], specs: Sequence[Any], charts: Sequence[Chart],
                body: np.ndarray, local: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rewrite ``j`` (host tables, global indices) for ``links``. Returns ``dira``, ``dirb`` [F, 3] (``fdir`` off the
    link faces) and the linked cell pairs [L, 2] (a, b)."""
    old = {k: j[k].copy() for k in ("fline", "fperp", "fpos", "fdir", "flen", "fdist", "cpos", "cnrm")}
    dira, dirb = j["fdira"].astype(np.float64).copy(), j["fdirb"].astype(np.float64).copy()
    remap = np.arange(len(j["fa"]))
    pairs, used = [], set()
    for spec in links:
        link = link_spec(spec)
        what = f"link ({link.a!r}, {link.edge_a!r}, {link.b!r}, {link.edge_b!r})"
        ka, kb = _index(link.a, specs, what), _index(link.b, specs, what)
        for side in ((ka, link.edge_a), (kb, link.edge_b)):
            if side in used:
                raise PluginError(f"{what}: edge {side[1]!r} of chart {side[0]} is already linked")
            used.add(side)
        ca, fa, sa, xa = edge_of(charts[ka], link.edge_a, what)
        cb, fb, sb, xb = edge_of(charts[kb], link.edge_b, what)
        if sa == sb:
            raise PluginError(f"{what}: joins two {'+' if sa > 0 else '-'} edges. A link joins a '+' edge to a '-' "
                              "edge running the same way: turn one chart (a plate's underside: its shape turned 180° "
                              "about y)")
        if sa < 0:
            ka, kb, ca, cb, fa, fb, xa, xb = kb, ka, cb, ca, fb, fa, xb, xa
        _join(j, old, (dira, dirb), (ka, kb, ca, cb, fa, fb), body, local, what, specs)
        for s in (0, 1):                               # the transverse faces beside the seam: fpar across it
            ta, tb = j["cface"][ca, 2 * (1 - xa) + s], j["cface"][cb, 2 * (1 - xb) + s]
            j["fpar"][ta, 1], j["fpar"][tb, 0] = tb, ta                  # A's seam side is +, B's -
        remap[fb] = fa
        pairs.append(np.stack([ca, cb], 1))
    for name in ("cface", "fline", "fpar", "fperp"):
        a = j[name]
        a[a >= 0] = remap[a[a >= 0]]
    _refuse_narrow(used, charts, specs)
    return dira, dirb, (np.concatenate(pairs) if pairs else np.zeros((0, 2), np.int64))


def _join(j: dict[str, np.ndarray], old: dict[str, np.ndarray], dirs: tuple[np.ndarray, np.ndarray], side: tuple,
          body: np.ndarray, local: np.ndarray, what: str, specs: Sequence[Any]) -> None:
    """One link: A's edge faces ``fa`` (cells ``ca``) become link faces to B's cells ``cb``; B's faces ``fb`` walls."""
    ka, kb, ca, cb, fa, fb = side
    if len(ca) != len(cb):
        raise PluginError(f"{what}: the edges have {len(ca)} and {len(cb)} cells; links pair cells one to one")
    if (body[:, ka] != body[:, kb]).any():
        raise PluginError(f"{what}: the charts ride different bodies; a link joins charts on one body")
    R, t = relative(local[:, ka], local[:, kb], what)
    pa, pb = old["fpos"][fa], old["fpos"][fb] @ R.T + t
    tol = 0.5 * min(old["flen"][fa].min(), old["flen"][fb].min())
    gap = np.linalg.norm(pa - pb, axis=1).max()
    if gap > tol:
        turned = np.linalg.norm(pa - pb[::-1], axis=1).max() <= tol
        mixed = isinstance(specs[ka], PlaneChart) != isinstance(specs[kb], PlaneChart)
        raise PluginError(f"{what}: the edges {'run opposite ways' if turned else 'do not meet'} (face centres "
                          f"{gap * 1e3:.3g} mm apart, over half a cell); turn one chart or pick the meeting edges"
                          + ("; a revolution chart's edge is a ring, which never meets a plane chart's straight "
                             "edge" if mixed and not turned else ""))
    ta, tb = old["fdir"][fa], old["fdir"][fb] @ R.T                    # A's outward, B's inward tangent (A's frame)
    along = np.cross(old["cnrm"][ca], ta)                              # the edge's own direction
    along /= np.linalg.norm(along, axis=1, keepdims=True)
    lip = pb - pa
    lip = np.linalg.norm(lip - (lip * along).sum(1, keepdims=True) * along, axis=1)   # the gap across the edge only
    dist = 0.5 * (old["fdist"][fa] + old["fdist"][fb]) + lip
    chord = old["cpos"][cb] @ R.T + t - old["cpos"][ca]
    j["fpos"][fa], j["fdir"][fa], j["fdist"][fa] = 0.5 * (pa + pb), chord / dist[:, None], dist
    dirs[0][fa], dirs[1][fa] = old["fdir"][fa], old["fdir"][fb]
    j["flen"][fa] = 0.5 * (old["flen"][fa] + old["flen"][fb])
    j["fb"][fa], j["ftype"][fa] = cb, 0
    j["fline"][fa, 1] = old["fline"][fb, 1]
    j["fperp"][fa, 2:] = old["fperp"][fb, 2:]
    j["ftype"][fb], j["flen"][fb] = EDGE_KINDS["wall"], 0.0
    ta64, tb64 = ta.astype(np.float64), tb.astype(np.float64)
    turn = np.arctan2(np.linalg.norm(np.cross(ta64, tb64), axis=1), (ta64 * tb64).sum(1))   # the fold's angle
    turn[turn < FLAT] = 0.0                                            # a straight join (float32 tangents): no fold
    convex = np.where((old["cnrm"][ca] * chord).sum(1) < 0.0, 1.0, -1.0)   # the chord dips off A's wetted side
    j["curv"][ca] = j["curv"][ca] + convex * 0.5 * turn / old["fdist"][fa]
    j["curv"][cb] = j["curv"][cb] + convex * 0.5 * turn / old["fdist"][fb]


def check_wetting(links: Sequence[Any], specs: Sequence[Any], wet: Sequence[tuple[float, float]]) -> None:
    """A link joins charts that wet alike: their effective contact-line conditions ``wet`` ((h_cl, tan θ) per chart,
    ``plugin._wetting``: the chart's angle, else the material's) must agree."""
    def angle(k: int) -> str:
        return "none" if wet[k][1] < 0.0 else f"{math.degrees(math.atan(wet[k][1])):.4g}°"
    for spec in links:
        link = link_spec(spec)
        what = f"link ({link.a!r}, {link.edge_a!r}, {link.b!r}, {link.edge_b!r})"
        ka, kb = _index(link.a, specs, what), _index(link.b, specs, what)
        if tuple(wet[ka]) != tuple(wet[kb]):
            raise PluginError(f"{what}: the charts wet differently (contact angles {angle(ka)} and {angle(kb)}, the "
                              "chart's own or the material's); a link joins charts that wet alike")


def _refuse_narrow(used: set, charts: Sequence[Chart], specs: Sequence[Any]) -> None:
    """A chart under 3 cells across between two of its own linked edges would let film reach two links from one tile
    (the skip's near sets reach one link across)."""
    for k, ch in enumerate(charts):
        for axis, n in ((0, int(ch.nx)), (1, int(ch.ny))):
            edges = [e for e, (x, _) in EDGES.items() if x == axis]
            if sum((k, e) in used for e in edges) >= 2 and n < 3:
                raise PluginError(f"chart {chart_name(specs[k])!r}: linked on both sides of its "
                                  f"{'i' if axis == 0 else 'j'} axis with only {n} cells across; links need >= 3 "
                                  "cells between them (give the chart more cells, or link fewer edges)")


__all__ = ["EDGES", "FLAT", "apply_links", "cell_dir", "check_wetting", "chart_name", "edge_of", "relative"]
