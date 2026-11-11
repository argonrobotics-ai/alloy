# SPDX-License-Identifier: Apache-2.0
"""``SurfaceFlow``'s host side, between periods: setting the film and its sources, reading it back, and the ledger.
A mixin of ``plugin.SurfaceFlow`` (it uses the plug-in's arrays and tables).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow import adsorption as AD
from alloy.plugins.surface_flow import cell_kernels as CK
from alloy.plugins.surface_flow import kernels as K
from alloy.plugins.surface_flow import stores_species as SS
from alloy.plugins.surface_flow import tiles as TL
from alloy.plugins.surface_flow import tool_geometry as TG

FREE_DRAIN = 1.0e30              # m/s: a drain cell that takes everything (finite, for the crash check)
H_DRY = 1.0e-6                   # m: a film thinner than this gives no flux: fronts do not creep on traces
H_WET = 2.0e-6                   # m: thinner cells carry no momentum and count as dry at a contact line
LEDGER = ("supplied", "spilled", "evaporated", "absorbed", "drained_total", "clipped", "dripped", "pooled_total")


class FilmHostAPI:
    """Host methods of ``SurfaceFlow`` (set_depth, set_source, set_input, depth, cell_centres, read)."""

    def _chart_cells(self, chart: int) -> slice:
        if not 0 <= chart < len(self.config.charts):
            raise PluginError(f"{self.label}: no chart {chart}; it has {len(self.config.charts)}")
        return self.tables.cells_of(chart)

    def _assign(self, array: wp.array, values: Any, chart: int | None, what: str) -> None:
        full = array.numpy()
        W = full.shape[0]
        sl = slice(None) if chart is None else self._chart_cells(chart)
        n = full[:, sl].shape[1]
        v = np.asarray(values, np.float64)
        if chart is not None and v.ndim >= 2 and v.shape[-2:] == self.tables.shapes[chart]:
            v = v.reshape(*v.shape[:-2], n)                   # (nx, ny) or (W, nx, ny) -> flat cells
        try:
            v = np.broadcast_to(v, (W, n))
        except ValueError:
            shape = "[C] or [W, C]" if chart is None else f"{self.tables.shapes[chart]}, (W, nx, ny) or [W, nx·ny]"
            raise PluginError(f"{self.label}: {what} of shape {np.shape(values)} does not fit; give a scalar or "
                              f"{shape} (pass chart= for per-chart grids)") from None
        if not np.isfinite(v).all() or (v < 0.0).any():
            raise PluginError(f"{self.label}: {what} must be finite and >= 0 (sinks are drains, open edges or "
                              f"evaporation)")
        full[:, sl] = v
        array.assign(full)

    def set_depth(self, depth: Any, chart: int | None = None) -> None:
        """The film [m]: a scalar, per cell of ``chart`` ((nx, ny) or (W, nx, ny)), or all cells ([C] or [W, C]).
        The ledger restarts from this film."""
        self._assign(self.h, depth, chart, "depth")
        self._refresh_wet()

    def set_source(self, rate: Any, chart: int | None = None, species: dict[str, float] | None = None) -> None:
        """A steady inflow [m/s] (volume per area per second) per cell, kept until changed. Never negative: film
        leaves through drains, open edges and evaporation (WP2 review M1). It may change every period, also on a
        strided film (``_rebank``). ``species``: what the inflow carries [kg/m³] by name (paint's solids, a soap
        solution); species left out carry 0; None keeps the last."""
        old = self.src_fixed.numpy().copy()              # numpy() is a view on the CPU
        old_conc = self.sp.src_conc.numpy().copy()
        self._assign(self.src_fixed, rate, chart, "source")
        if species is not None:
            conc = np.zeros_like(old_conc)
            for name, value in species.items():
                if not (np.isfinite(value) and value >= 0.0):
                    raise PluginError(f"{self.label}: source species {name!r} must be finite and >= 0, got {value}")
                conc[self._species_index(name)] = value
            self.sp.src_conc.assign(conc)
        self._rebank(old, old_conc)
        self._refresh_wet()

    def set_input(self, name: str, values: Any) -> None:
        old = self.src_fixed.numpy().copy() if name == "source" else None
        super().set_input(name, values)
        if old is not None:
            self._rebank(old, self.sp.src_conc.numpy().copy())
        if name in ("source", "evaporation", "absorption"):
            self._refresh_wet()
        if name == "base" and self.bulk is not None:           # the paste's surface starts where the user put it
            self.bulk.assign(self.base.numpy().astype(np.float64))

    def _rebank(self, old: np.ndarray, old_conc: np.ndarray) -> None:
        """A source changed between a strided film's steps: bank what the window's past exchanges were owed, volume
        and species (cell_kernels.rebank_kernel, rebank_species_kernel)."""
        if self.stride == 1:
            return
        dev = self.src_fixed.device
        prev = wp.array(old, dtype=wp.float64, device=dev)
        wp.launch(CK.rebank_kernel, dim=old.shape, inputs=[self.tick, self.stride, self._exchange_dt, prev,
                                                           self.src_fixed], outputs=[self.src_acc], device=dev)
        if self.sp.count:
            wp.launch(CK.rebank_species_kernel, dim=old.shape,
                      inputs=[self.tick, self.stride, self._exchange_dt, prev, self.src_fixed,
                              wp.array(old_conc, dtype=wp.float64, device=dev), self.sp], device=dev)

    def _refresh_wet(self) -> None:
        """Cells the host made wet (film, a source, inflow pending) join the dry-tile skip's wet set (tiles.py)."""
        src = self.src_fixed.numpy() + np.maximum(self.src_acc.numpy(), 0.0)
        sinks = self.mat.evaporation
        if self.fluid.per_sink:                                    # the process stage's per-cell rates (§4.8)
            sinks = self.fluid.evap_cell.numpy() + self.fluid.absorb_cell.numpy()
        self.cwet.assign(TL.wet_cells(self.h.numpy(), src, self.drain_of.numpy(), sinks, H_DRY,
                                      self.flooded.numpy()))

    def _species_index(self, name: str) -> int:
        names = tuple(self.config.species)
        if name not in names:
            raise PluginError(f"{self.label}: no species {name!r}; it carries {names} (Config.species)")
        return names.index(name)

    def set_species(self, mass: Any, name: str, chart: int | None = None) -> None:
        """A species' mass per area [kg/m²] (shapes as ``set_depth``); it moves with the film, and stays as residue
        where the film is gone."""
        k = self._species_index(name)
        full = self.sp.mass.numpy()
        layer = wp.array(np.ascontiguousarray(full[:, :, k]), dtype=wp.float64, device=self.sp.mass.device)
        self._assign(layer, mass, chart, f"species {name!r}")
        full[:, :, k] = layer.numpy()
        self.sp.mass.assign(full)

    def set_adsorbed(self, load: Any, name: str, chart: int | None = None) -> None:
        """Species ``name``'s load adsorbed on the walls [kg/m²] (shapes as ``set_depth``; ``Config.adsorption``)."""
        AD.set_adsorbed(self, load, name, chart, self._species_index(name))

    def species(self, name: str, chart: int = 0) -> np.ndarray:
        """``[W, nx, ny]`` mass per area [kg/m²] of species ``name`` on ``chart``."""
        nx, ny = self.tables.shapes[chart]
        return self.sp.mass.numpy()[:, self._chart_cells(chart), self._species_index(name)].reshape(-1, nx, ny)

    def set_regions(self, region: Any, chart: int | None = None) -> None:
        """Region ids per cell (ints; 0 everywhere at the start), the same in every world: cells of different regions
        exchange no film, the faces between them act as walls (no flux, no capillary climb). A stencil sheet,
        masking tape or an obstacle on a plate; set it again to move or lift them. Film on a region's cells stays
        there (paint on the sheet). Shapes as ``set_depth`` without the world axis."""
        t = self.tables
        regions = getattr(self, "_regions", None)
        if regions is None:
            regions = self._regions = np.zeros(t.cell_count, np.int64)
        sl = slice(None) if chart is None else self._chart_cells(chart)
        v = np.asarray(region)
        if v.dtype.kind not in "iub":
            raise PluginError(f"{self.label}: regions must be integers, got {v.dtype}")
        n = regions[sl].shape[0]
        try:
            regions[sl] = np.broadcast_to(v.reshape(-1) if v.ndim >= 2 else v, (n,))
        except ValueError:
            raise PluginError(f"{self.label}: regions of shape {v.shape} do not fit {n} cells") from None
        kind = t.ftype.numpy().copy()                    # the charts' own walls, open edges and interior faces
        a, b = t.fa.numpy(), t.fb.numpy()
        inner = (kind == int(K.INTERIOR)) & (a >= 0) & (b >= 0)
        split = inner & (regions[np.maximum(a, 0)] != regions[np.maximum(b, 0)])
        kind[split] = int(K.WALL)
        self.faces.kind.assign(kind.astype(np.int32))

    def depth(self, chart: int = 0) -> np.ndarray:
        """``[W, nx, ny]`` film depth [m] on ``chart``."""
        nx, ny = self.tables.shapes[chart]
        return self.h.numpy()[:, self._chart_cells(chart)].reshape(-1, nx, ny)

    def cell_centres(self, chart: int = 0) -> np.ndarray:
        """``[nx, ny, 3]`` cell centres of ``chart`` in its frame [m]."""
        nx, ny = self.tables.shapes[chart]
        return self.tables.cpos.numpy()[self._chart_cells(chart)].reshape(nx, ny, 3)

    def read(self) -> dict[str, np.ndarray]:
        out = super().read()
        area = self.tables.area_np
        out["film_volume"] = (out["h"] * area).sum(axis=1)                              # [W] m³
        for name in LEDGER:
            out[f"{name}_volume"] = out[name].sum(axis=1)
        out["pending_volume"] = (out["src_acc"] * area).sum(axis=1)                   # inflow awaiting the film step
        # health counters (WP2 review M3, M4): faces held at the speed cap (the film there is too thick), and cells
        # deeper than the depth the step bound assumed
        out["faces_at_cap"] = (np.abs(out["u"]) >= 0.999 * self.config.u_cap).sum(axis=1)
        out["cells_over_depth"] = (out["h"] > self.config.max_depth).sum(axis=1)
        out["tiles_active"] = self.tile_on.numpy().mean(axis=1)    # share of tiles the last film substep stepped
        if self.tools is not None:                                       # the tools' work fields, this exchange
            out["tool_cover"], out["tool_vel"] = self.tf.cover.numpy(), self.tf.vel.numpy()
            out["tool_pressure"] = self.tf.press.numpy()
            ceil = self.tf.ceil.numpy().astype(np.float64)               # [W, C] m above the chart; +inf: no tool down
            out["tool_ceiling"] = np.where(ceil >= TG.FREE_GAP, np.inf, ceil)
            work = self.tf.wceil.numpy().astype(np.float64)                # every tool's underside, store tools too
            out["tool_work_ceiling"] = np.where(work >= TG.FREE_GAP, np.inf, work)
            out["tool_work_cover"], out["tool_work_vel"] = self.tf.wcover.numpy(), self.tf.wvel.numpy()
            out["tool_drag_scale"] = self.tool_drag_scale.numpy()           # [W, tools] the last exchange's cap
            out["tool_drag_capped"] = self.tool_drag_capped.numpy()         # [W, tools] exchanges it bound
        if self.bulk is not None:
            out["bulk_volume"] = (out["bulk"] * area).sum(axis=1)       # the paste moved, not made: constant
        out["ledger"] = (out["film_volume"] + out["spilled_volume"] + out["evaporated_volume"] + out["absorbed_volume"]
                         + out["drained_total_volume"] + out["clipped_volume"] + out["dripped_volume"]
                         + out["pooled_total_volume"] - out["supplied_volume"])
        if self.tools is not None:                                         # what porous pads absorbed, not taken yet
            out["film_absorbed"] = (self.tf.absorb.numpy() * area).sum(axis=1)
            out["ledger"] = out["ledger"] + out["film_absorbed"]
        if self.sp.count:                                                  # [W, K] kg, and their ledgers
            out["species_mass"] = (out["species"] * area[None, :, None]).sum(axis=1)
            out["species_ledger"] = (out["species_mass"] + out["species_out"].sum(axis=1)
                                     + out["species_clipped"].sum(axis=1) - out["species_supplied"].sum(axis=1))
            AD.adsorbed_read(self, out)                                    # the walls' load (Langmuir)
            if getattr(self, "pool_adsorption", None) is not None:          # given to pools under water
                self.pool_adsorption.read(out)
            if self.tools is not None:
                absorbed = (self.tf.absorb_sp.numpy() * area[None, :, None]).sum(axis=1)
                out["species_ledger"] = out["species_ledger"] + absorbed
        if self.stores is not None:                                        # film the tools took (or laid: < 0)
            out["to_tools_volume"] = out["to_tools"].sum(axis=1)
            out["ledger"] = out["ledger"] + out["to_tools_volume"]
            out["store_ledger"] = (out["tool_load"].sum(axis=1) + out["tool_dripped"].sum(axis=1)
                                   + out["tool_released"].sum(axis=1) + out["tool_dropped"].sum(axis=1)
                                   + out["tool_overflow"].sum(axis=1) - out["tool_picked"].sum(axis=1)
                                   - out["to_tools_volume"])
            if "tool_overflow_off" in out:                                 # WP13's end bands: past the chart's edge
                out["store_ledger"] = out["store_ledger"] + out["tool_overflow_off"].sum(axis=1)
            out["tray_ledger"] = out["tray_volume"].sum(axis=1) + out["tool_picked"].sum(axis=1)
            SS.species_read(self, out)                                     # the stores' and trays' species books
        # the supply's cross ledger (sent + swept + withdrawn = supplied + to_tank + lost + overspray + mist + caught
        # + lumped) counts pending inflow as supplied: read supplied_volume + pending_volume there
        if self.supply is not None:
            sup = self.supply.read()
            out.update({f"supply_{k}": v for k, v in sup.items()})
            out["ledger"] = out["ledger"] + sup["swept"]          # the sweep moved film outside the film's books
            if self.sp.count:                                   # and its species, re-entered as supplied at the ring
                out["species_ledger"] = out["species_ledger"] + sup["swept_species"]
        reader = getattr(getattr(self, "process", None), "read", None)
        if reader is not None:                                  # the process stage's maps (dft, phi_s, laps, ...)
            out.update({k: v for k, v in reader().items() if k not in out})
        lumps = getattr(self, "lumped", None)
        if lumps is not None:                                   # lumped films (§4.9) and the film they took up
            lr = lumps.read()
            out.update({k: v for k, v in lr.items() if k not in out})
            out["ledger"] = out["ledger"] + lr["to_lumps_volume"]
            if self.sp.count:
                out["species_ledger"] = out["species_ledger"] + lumps.film_out.taken_sp.numpy()[:, :self.sp.count]
        return out



__all__ = ["FREE_DRAIN", "FilmHostAPI", "H_DRY", "H_WET", "LEDGER"]
