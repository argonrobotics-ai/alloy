# SPDX-License-Identifier: Apache-2.0
"""Trapped air under a ``Bonds`` film (SCR-WP8-11 Part B2, ``Bonds.Config.trapped_air``).

**Pockets.** A pocket is a connected set of free film vertices (``pockets.py``, exact labelling). One that reaches the
label's edge is open, one wholly surrounded by bonds is enclosed. Each of the ``SLOTS`` = 8 largest pockets of a world
is one isothermal gas node of the shared fluid core (``alloy.plugins.fluid``), with amount a = P·V. Its volume is
V = sum A_i·h_i over its vertices: h_i is the gap between the vertex's contact surface and the nearest substrate
surface (box, capsule or cylinder), plus the layer the surfaces' roughness leaves at contact (``air_roughness``, 1 µm).
Without it a squeezed air film could never thin to contact (Reynolds: h ~ t^-1/2), and a flat pad would float a label
for ever. Air that would stand under 1 µm above that layer at P_atm is roughness air (``air_kernels.held``): carried by
the asperities, it pushes nothing, blocks no bond, is not trapped, and is booked ``air_vanished`` when its pocket's
last cell bonds (it ends in the adhesive's micro-voids). A pocket's pressure ramps in over the next micrometre.

**Pressure on the film (WP13 must-fix 2: implicit).** Before each VBD colour's solve, every pocket's pressure is
Boyle's at the current iterate, P = a/V, and each of its vertices of that colour feels (P - P_atm)·A_i along the
substrate normal, with the gas spring's Hessian kappa·A_i·s·n nᵀ (kappa = P/V, s the area of the colour's vertices in
the pocket). The coupling is rank one, kappa·A Aᵀ, and that lumped diagonal is at least half of it (Cauchy-Schwarz),
so the colour's simultaneous step lowers the energy and never overshoots; a uniform motion of the colour is solved
exactly. The gas spring is stiff (P·A²/V about 4e4 N/m on 1.6 mg of label for a 20 mm² pocket: a lagged force would
need dt < 13 µs); this holds at the film's step (laws T-1 at 0.5 and 1 ms).

**Vent.** After each film step an open pocket vents to P_atm through its throat on the label's edge, a laminar slot
g = P_mean·sum w_b·h³/(12·mu·s_b) over its edge vertices (``pockets.throat_coefficients``): one cell long, of the
edge's width, at the gap there but at least the pocket's mean gap. The floor is the squeeze film: under a rigid pad the
gap stays about uniform and the air leaves through all of it (a strip's mean-pressure conductance is this law with
s = half-width/3, so one cell is within 2x for label-sized pockets). Without it the label's edge vertices, which carry
a half or a quarter of an interior vertex's area but the same contact push, are pressed shut first and a flat pad
floats the label on its air (task_label_apply: 176 kPa for the whole press). The h³ still lets a pocket close as its
gap does; a pocket is trapped when bonds enclose it. The core steps the gas nodes (θ-method). An enclosed pocket keeps
its air, and its cells cannot bond while it holds air (``block_trapped``).

**Books (WP13 must-fix 3, ``air_books.py``).** Air moves with the labels through splits (by volume at a common
pressure), merges (summed), new cells (from the atmosphere, booked ``air_formed``), vents (``air_vented``), cells that
bond (their share, booked ``air_vanished``) and overflow (``air_untracked``): per world, sum of slot amounts +
vented + vanished + untracked - formed = 0 to float64 rounding (law T-3).

**Overflow (WP13 must-fix 5).** Every pocket's area comes from the labels: ``trapped_area`` is the largest enclosed
pocket holding air over all of them, and ``pockets`` counts them. Pockets beyond the slots carry no air
(``air_overflow`` counts them, a failed GA run); idle slots are parked at 1 cm³ with their vent shut.

**Limits.** The pocket is one pressure: a long narrow channel inside it adds no resistance (the vent is the edge's
throat, one cell long). The throat's mean-gap floor and the roughness rules all favour venting, so the model can
under-report trapped air from a transient seal; air is trapped when bonds enclose it (an edges-first application traps
197 mm² and fails GA, laws_air.py T-4). Pockets beyond the 8 largest keep their scored area but lose their air each
step (``air_untracked``, flagged by ``air_overflow``). Air does not dissolve or leak through the adhesive. GA's
3 mm² barcode criterion is not scored: it needs cells under 1.5 mm, where 60 µm stock freezes on the film route
(``BARCODE_NOT_SCORED``; docs/known_gaps/label-barcode-trapped-air.md).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.fluid import Edge, Network, Node

from . import air_books as B
from . import air_kernels as K
from .pockets import label, label_rounds, mesh_edges, throat_coefficients

log = logging.getLogger("alloy.plugins.bonds")

AIR_VISCOSITY = 1.81e-5                      # Pa·s, air at 20 °C
AIR_ROUGHNESS = 1.0e-6                       # m: the gas layer at contact (adhesive and substrate roughness, ESTIMATE)
GA_TRAPPED_AREA = 20.0e-6                    # m²: GA's label criterion, no trapped-air pocket larger
BARCODE_NOT_SCORED = "barcode trapped-air: not scored (needs < 1.5 mm cells; 60 µm stock freezes below that)"
SLOT_STATE = (("air_amount", wp.float64), ("air_p", wp.float64), ("air_root", wp.int32), ("air_area", wp.float64),
              ("air_volume", wp.float64), ("air_lam", wp.float64), ("air_enclosed", wp.int32))
READINGS = ("trapped_area", "pockets", "air_overflow", "air_unsettled", "air_formed", "air_vented", "air_vanished",
            "air_untracked")
GEO = {int(newton.GeoType.BOX): K.BOX, int(newton.GeoType.CAPSULE): K.CAPSULE, int(newton.GeoType.CYLINDER): K.CYLINDER}


def ga_trapped_air(trapped_area: np.ndarray) -> np.ndarray:
    """GA's trapped-air criterion per world: no enclosed pocket holding air larger than 20 mm²."""
    return np.asarray(trapped_area) <= GA_TRAPPED_AREA


def books_error(read: dict[str, np.ndarray]) -> np.ndarray:
    """Per world: (slots + vented + vanished + untracked - formed) / formed; 0 to rounding (law T-3)."""
    held = read["air_amount"].sum(axis=1)
    out = held + read["air_vented"] + read["air_vanished"] + read["air_untracked"] - read["air_formed"]
    return out / np.maximum(read["air_formed"], 1e-300)


def substrate_table(shapes: list[np.ndarray], kinds: np.ndarray) -> np.ndarray:
    """[W, S] substrate shape ids per world; every one a box, capsule or cylinder."""
    table = np.stack(shapes, axis=1).astype(np.int32)
    bad = sorted({int(kinds[s]) for s in table.ravel() if int(kinds[s]) not in GEO})
    if bad:
        raise PluginError(f"bonds: trapped air needs box, capsule or cylinder substrates (shape types {bad} found)")
    return table


class AirField:
    """The pockets of one film solver: arrays, the fluid core's gas nodes and the launches (module docstring)."""

    def __init__(self, view: Any, arrays: dict[str, wp.array], init: dict[str, Any], substrates: np.ndarray,
                 viscosity: float = AIR_VISCOSITY, roughness: float = AIR_ROUGHNESS) -> None:
        dev, n, W = view.device, int(view.particle_count), len(init["start"]) - 1
        self.dev, self.n, self.W = dev, n, W
        arr = lambda v, t: wp.array(np.ascontiguousarray(v), dtype=t, device=dev)  # noqa: E731
        x, tris = view.particle_q.numpy().astype(np.float64), view.tri_indices.numpy()
        edges, _ = mesh_edges(tris)
        self.edges = arr(edges, wp.vec2i)
        self.edge_world = arr(view.particle_world.numpy()[edges[:, 0]], wp.int32)
        self.rounds = label_rounds(n)
        self.state = arrays["bond_state"]
        self.gp = K.Gaps()
        self.gp.shapes = arr(substrate_table(list(substrates.T), view.shape_type.numpy()), wp.int32)
        self.gp.kind = arr([GEO.get(int(t), -1) for t in view.shape_type.numpy()], wp.int32)
        self.gp.scale, self.gp.xform = view.shape_scale, view.shape_transform
        self.gp.margin = arr(init["surface"]["margin"], float)
        self.gp.body, self.gp.radius, self.gp.world = view.shape_body, view.particle_radius, view.particle_world
        self.air = self._air_struct(view, arrays, init, x, tris, viscosity, arr)
        self.air.rough = float(roughness)
        self.books = K.Books()
        for name, reading in (("formed", "air_formed"), ("vented", "air_vented"), ("vanished", "air_vanished"),
                              ("untracked", "air_untracked"), ("trapped", "trapped_area"), ("pockets", "pockets"),
                              ("overflow", "air_overflow")):
            setattr(self.books, name, arrays[reading])
        self.unsettled = wp.zeros(W, dtype=wp.int32, device=dev)
        self.every = wp.zeros(n, dtype=wp.int32, device=dev)          # selects every vertex for vertex_gaps
        self.unsettled_f = arrays["air_unsettled"]
        self.scratch = self._scratch()
        self.fluid = self._network().compile(dev, W)

    def _air_struct(self, view: Any, arrays: dict, init: dict, x: np.ndarray, tris: np.ndarray, mu: float,
                    arr: Any) -> K.Air:
        a = K.Air()
        a.start, a.area = arr(init["start"], wp.int32), arr(init["area"], float)
        a.throat = arr(throat_coefficients(x, tris, init["h"], mu), wp.float64)
        a.slot = arrays["air_slot"]
        a.gap = wp.zeros(self.n, dtype=float, device=self.dev)
        a.nrm = wp.zeros(self.n, dtype=wp.vec3, device=self.dev)
        a.amount, a.p, a.root = arrays["air_amount"], arrays["air_p"], arrays["air_root"]
        a.area_k, a.volume, a.lam = arrays["air_area"], arrays["air_volume"], arrays["air_lam"]
        a.enclosed = arrays["air_enclosed"]
        a.vsum = wp.zeros((self.W, K.SLOTS), dtype=wp.float64, device=self.dev)
        a.csum = wp.zeros((self.W, K.SLOTS), dtype=wp.float64, device=self.dev)
        return a

    def _scratch(self) -> K.Scratch:
        sc, n, shape = K.Scratch(), self.n, (self.W, K.SLOTS)
        sc.parent = wp.zeros(n, dtype=wp.int32, device=self.dev)
        for name in ("amount", "area", "vol", "lam"):
            setattr(sc, name, wp.zeros(n, dtype=wp.float64, device=self.dev))
        sc.open_ = wp.zeros(n, dtype=wp.int32, device=self.dev)
        sc.slot = wp.zeros(n, dtype=wp.int32, device=self.dev)
        sc.sel_root = wp.zeros(shape, dtype=wp.int32, device=self.dev)
        sc.sel_area = wp.zeros(shape, dtype=wp.float64, device=self.dev)
        sc.new_root = wp.zeros(shape, dtype=wp.int32, device=self.dev)
        return sc

    @staticmethod
    def _network() -> Network:
        """SLOTS isothermal gas nodes, each venting to the air through its own linear edge (opening = g/G_REF)."""
        park, p_atm, g_ref = float(K.V_PARK.value), float(K.P_ATM.value), float(K.G_REF.value)
        nodes = [Node(f"pocket{k}", "gas", volume=park, pressure=p_atm) for k in range(K.SLOTS)]
        edges = [Edge(f"vent{k}", "linear", f"pocket{k}", reservoir=p_atm, conductance=g_ref) for k in range(K.SLOTS)]
        return Network(nodes, edges)

    # --- per film step -----------------------------------------------------------------------------------------
    def accumulate(self, solver: Any, colour: int, state: Any, contacts: Any, body_q: wp.array) -> None:
        """VBD after_accumulate slot: the pockets' pressure at the iterate, then the colour's forces and Hessians."""
        colors = solver.model.particle_colors
        wp.launch(K.vertex_gaps, dim=self.n, inputs=[self.gp, self.air.rough, self.air.slot, state.particle_q, body_q],
                  outputs=[self.air.gap, self.air.nrm], device=self.dev)
        wp.launch(K.pocket_pressure, dim=self.W, inputs=[colour, colors, self.air], device=self.dev)
        wp.launch(K.pocket_force, dim=self.n, inputs=[colour, colors, self.gp.world, self.air],
                  outputs=[solver.particle_forces, solver.particle_hessians], device=self.dev)

    def block(self, key: wp.array) -> None:
        """Before the dwell clocks: enclosed pockets holding air keep their cells free."""
        wp.launch(K.block_trapped, dim=self.n, inputs=[self.air, self.gp.world, key], device=self.dev)

    def after(self, state_out: Any, body_q: wp.array, dt: float) -> None:
        """After formation: relabel, move the air with the labels (exact books), then vent on the fluid core."""
        sc, air, bk, W = self.scratch, self.air, self.books, self.W
        label(self.state, self.edges, sc.parent, self.unsettled, self.rounds, self.edge_world)
        wp.launch(_to_reading, dim=W, inputs=[self.unsettled], outputs=[self.unsettled_f], device=self.dev)
        wp.launch(K.vertex_gaps, dim=self.n, inputs=[self.gp, air.rough, self.every, state_out.particle_q, body_q],
                  outputs=[air.gap, air.nrm], device=self.dev)
        wp.launch(B.survivors, dim=W, inputs=[air, sc, bk], device=self.dev)
        wp.launch(B.clear_roots, dim=self.n, inputs=[sc], device=self.dev)
        for kernel in (B.gather, B.select):
            wp.launch(kernel, dim=W, inputs=[air, sc, bk], device=self.dev)
        wp.launch(B.assign, dim=W, inputs=[air, sc], device=self.dev)
        wp.launch(B.vertex_slots, dim=self.n, inputs=[air, sc], device=self.dev)
        f = self.fluid
        wp.launch(K.vent_in, dim=(W, K.SLOTS), inputs=[air], outputs=[f.pressure, f.volume, f.volume_old, f.opening],
                  device=self.dev)
        f.step(dt)
        wp.launch(K.vent_out, dim=W, inputs=[air, f.pressure, bk], device=self.dev)


@wp.kernel
def _to_reading(count: wp.array[wp.int32], out: wp.array[wp.float64]):
    w = wp.tid()
    out[w] = wp.float64(count[w])


__all__ = ["AIR_ROUGHNESS", "AIR_VISCOSITY", "BARCODE_NOT_SCORED", "GA_TRAPPED_AREA", "READINGS", "SLOT_STATE",
           "AirField", "books_error", "ga_trapped_air", "substrate_table"]
