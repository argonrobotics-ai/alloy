# SPDX-License-Identifier: Apache-2.0
"""What the dunk tasks (rinse_pan, rinse_glassware) read of a vessel each control period: one ``PanSample`` per
world for its ``PanTracker`` (their pan tracker and the v1 glass tracker share the rules).

Their tracker reads the vessel's faces and pools every 10 ms (``task.py:247-333``); here every control period, from the
film (depths, species, which pool floods each cell), the network (the bowls' levels, the cavity's water and soap)
and the vessel's pose. Definitions (``rinse_tracker.PanSample``):
- ``pool``: the bowl whose liquid holds the vessel's origin (in its footprint, below its level); ``pool_frac``: the
  area share of the vessel's cells below that level and inside its footprint (their "metal under the pool");
- ``inside_wet``: the area share of the inner cells flooded by any pool (the cavity's own water included);
- ``other_wet`` / ``touch_sanitizer`` / ``in_bowl``: a bowl's liquid holds the vessel or floods any of its cells;
- ``carried``: the soap on the vessel's faces (film and walls, adsorbed: their (s + gamma)·A, ``sanitizer.py:51``)
  plus the cavity's, when the cavity's water is stronger than the rinse strength (``carried_detergent_kg``);
- ``water_ml`` / ``inside_water_ml``: film (all / inner cells) plus cavity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.plugins.surface_flow import SurfaceFlow
from alloy.tasks.rinse_tracker import PanSample, carried_detergent_kg
from alloy.tasks.rinse_plates import gauges
from alloy.tasks.rinse_sink import BOWL, BOWLS, WALL
from alloy.tasks.rinse_vessels import Vessel
from alloy.tasks.rinse_wash_station import SPECIES, WashStation

SOAP_K = SPECIES.index("soap")


@dataclass(frozen=True)
class VesselCells:
    """A vessel chart's cells: ``inner`` mask (floor and inner wall), areas, centres in the body frame [n, 3]."""

    inner: np.ndarray
    area: np.ndarray
    local: np.ndarray

    @classmethod
    def of(cls, flow: SurfaceFlow, station: WashStation, k: int) -> VesselCells:
        return cls.of_vessel(flow, station.items()[k].vessel, k)

    @classmethod
    def of_vessel(cls, flow: SurfaceFlow, v: Vessel, k: int) -> VesselCells:
        """``v``'s chart, chart ``k`` of ``flow``."""
        t = flow.tables
        nx, ny = t.shapes[k]
        prof = np.asarray(v.profile())
        s_nodes = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(prof, axis=0).T))])
        s_rows = (np.arange(nx) + 0.5) * s_nodes[-1] / nx
        inner = np.repeat(s_rows < v.regions()["inner"][1], ny)
        cells = t.cells_of(k)
        return cls(inner=inner, area=t.area_np[cells], local=flow.cell_centres(k).reshape(-1, 3))


def face_soap_kg(flow: Any, area: np.ndarray, k: int, w: int) -> float:
    """World ``w``'s soap on vessel chart ``k``'s faces [kg]: the film's solution plus the walls' adsorbed load (their
    (s + gamma)·A, ``task_set/sanitizer.py:51``)."""
    cells = flow.tables.cells_of(k)
    soap = flow.sp.mass.numpy()[w, cells, SOAP_K]
    ads = getattr(flow, "adsorption", None)
    if ads is not None:
        soap = soap + ads.load.numpy()[w, cells, SOAP_K]
    return float((soap * area).sum())


def free_water_ml(flow: SurfaceFlow, cells: VesselCells, k: int, w: int, q: np.ndarray, h_break: float,
                  cavity_m3: float) -> float:
    """World ``w``'s free water on vessel chart ``k`` at body pose ``q`` (x, y, z, qx, qy, qz, qw) [mL]: the cavity's
    water plus the inner film above what clings, h_break/sqrt(max(sin slope, 0.04)) and at least 50 µm
    (``task.py:507-523``)."""
    x, y, z, qw = q[3:]
    up_local = np.array([2 * (x * z - qw * y), 2 * (y * z + qw * x), 1 - 2 * (x * x + y * y)])   # world z in body
    nz = flow.tables.cnrm.numpy()[flow.tables.cells_of(k)] @ up_local
    h_ret = np.maximum(h_break / np.sqrt(np.maximum(np.sqrt(np.clip(1.0 - nz * nz, 0.0, 1.0)), 0.04)), 50e-6)
    h = flow.h.numpy()[w, flow.tables.cells_of(k)]
    return (float((np.maximum(h - h_ret, 0.0) * cells.area)[cells.inner].sum()) + cavity_m3) * 1e6


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate rows of ``v`` by the quaternion ``q`` (x, y, z, w)."""
    u, w = q[:3], q[3]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


class VesselSense:
    """Builds each world's ``PanSample`` of item ``k`` (chart ``k``; ``bodies`` [W] its body index in each world, as
    ``PoseHand.held[:, i]``; see the module docstring)."""

    def __init__(self, flow: SurfaceFlow, station: WashStation, k: int, bodies: Any, c_rinse: float) -> None:
        self.flow, self.station, self.k, self.c_rinse = flow, station, k, c_rinse
        self.bodies = np.atleast_1d(np.asarray(bodies, np.int64))
        self.cells = VesselCells.of(flow, station, k)
        names = [p.tank for p in station.pools()]
        self.bowl_pool = {b: names.index(b) for b in BOWLS}
        nodes = [n.name for n in station.nodes()]
        self.cav_node = nodes.index(station.items()[k].cavity)
        self.bowl_node = {b: nodes.index(b) for b in BOWLS}

    def _levels(self, amount: np.ndarray) -> dict[str, float]:
        s = self.station
        return {b: s.floor_z + float(amount[self.bowl_node[b]]) / s.sink.area for b in BOWLS}

    def _under(self, world_pts: np.ndarray, k: int, level: float) -> np.ndarray:
        c = self.station.bowl_centre(k)
        hx, hy = 0.5 * BOWL[0] - WALL, 0.5 * BOWL[1]
        return ((np.abs(world_pts[:, 0] - c[0]) < hx) & (np.abs(world_pts[:, 1] - c[1]) < hy)
                & (world_pts[:, 2] < level))

    def sample(self, w: int, t: float, outlet_running: bool, body_q: np.ndarray, net: dict[str, Any]) -> PanSample:
        """World ``w``'s sample at time ``t``: ``body_q`` the bodies' poses, ``net`` the supply's read()."""
        f, cl = self.flow, self.cells
        q = body_q[self.bodies[w]]
        pts = q[:3] + _rotate(q[3:], cl.local)
        flooded = f.flooded.numpy()[w, f.tables.cells_of(self.k)]
        levels = self._levels(net["amount"][w])
        pool, frac, wet_by = None, 0.0, set()
        for k, b in enumerate(BOWLS):
            under = self._under(pts, k, levels[b])
            if (flooded == self.bowl_pool[b]).any():
                wet_by.add(b)
            if self._under(q[None, :3], k, levels[b])[0]:
                pool, frac = b, float(cl.area[under].sum() / cl.area.sum())
                wet_by.add(b)
        inside = cl.area[cl.inner & (flooded >= 0)].sum() / cl.area[cl.inner].sum()
        h = f.h.numpy()[w, f.tables.cells_of(self.k)]
        cav_v = float(net["amount"][w][self.cav_node])
        cav_soap = float(f.supply.fluid.species.numpy()[w, self.cav_node, SOAP_K])
        centred = bool(self._under(np.array([[q[0], q[1], -np.inf]]), BOWLS.index("sanitize"), np.inf)[0])
        return PanSample(
            t=t, gauge=float(gauges(f, f.process.soil.numpy()[w:w + 1], self.k)[0]) if f.process else 0.0,
            carried=carried_detergent_kg(face_soap_kg(f, cl.area, self.k, w), cav_soap, cav_v, self.c_rinse),
            z=float(q[2]), inside_wet=float(inside), pool=pool, pool_frac=frac,
            touch_sanitizer=pool == "sanitize" or "sanitize" in wet_by,
            other_wet=next((b for b in ("wash", "rinse") if b in wet_by), None), in_bowl=bool(wet_by),
            centred_over_sanitizer=centred, outlet_running=outlet_running,
            water_ml=float(((h * cl.area).sum() + cav_v) * 1e6),
            inside_water_ml=float(((h * cl.area)[cl.inner].sum() + cav_v) * 1e6))


__all__ = ["VesselCells", "VesselSense", "face_soap_kg", "free_water_ml"]
