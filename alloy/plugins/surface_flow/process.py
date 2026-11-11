# SPDX-License-Identifier: Apache-2.0
"""Process states of ``SurfaceFlow`` (design §4.8 and §5; WP9, reviewed by WP15).

Per-cell states ride on the film: setting, soil, drying, absorption and the sanitizer clock.
``SurfaceFlow.Config.process`` sets them, as a ``Process`` or a dict (``process_specs.py`` has the parameters and their
sources). Each part is optional, and a part that is not set allocates nothing.

**Setting** (design §3.2, C:424-436). A paste stiffens with age: ``τ(a) = τ_0 + A_thix·a/s_T + τ_IS·(a/(t_IS·s_T))^m``.
- The film's at-rest yield is ``film_factor·τ`` (input ``yield_stress``); a forming bulk's ``bulk_yield`` is ``τ``.
  Setting is the only writer of both. WP15's thixotropy scales the film's yield in the face kernel and never writes it.
- ``age`` [s] is a state and a settable input (``set_input("age")``). With a forming bulk, every cell is placed at the
  start (``initial_age``) and ages wet or dry; its ``bulk_yield`` follows from its age every exchange (forming reads
  it each exchange). Otherwise a cell is placed when it first wets (before that, age < 0, counting as 0) and ages
  while its tile is active: a cell that goes bare and is wetted again comes back younger than its first wetting.
- A film step's yield is the law at the age the step starts with.

**Soil** (R:257-270). ``soil`` [kg/m²] soaks while wet, ``hyd ← 1 - (1 - hyd)·exp(-Δt/t_soak(T))``, and its strength
``σ_c = σ_dry·(1 - hyd)² + σ_wet`` falls. A wet cell loses soil (``process_specs.Soil``) to jet impact over the jet's
share of the cell, to shear (the jet sheet's ``jet_tau`` under a sheet, else the larger of it and the film's own;
cells under a sheet count as wet, though the supply sweeps their film; so do cells under a pool's water, ``flooded``,
whose film went to the pool, with the shear of the jets that enter the pool, DC5's ``pool_jet_tau``: the free jet's
decay with depth; their removed soil joins the pool's tank, ``soil_to_pools``), to Wilson's cleaning front
from clean neighbours, and to:
- film shear ``k_e·(τ_w - σ_c)+``, with ``τ_w = 3μ|q̄|/max(h, h_wet)²`` (plus the yield stress while the film flows);
- jet impact ``k_i·frac·(p_stag/σ_c - 1)+ = k_i·(impact_p/σ_c - frac)+``: the supply's per-cell ``impact_p`` is the
  stagnation pressure ``p_stag`` times the jets' share ``frac`` of the cell, its ``impact_share``;
- wiping ``k_wipe·cover·τ_wipe/σ_c``, with ``τ_wipe = μ|v|/max(gap, h)`` under a tool;
- abrasion under a ceiling tool, wet or dry, ``k_abr·cover·press·|v|·σ_ref/σ_c`` (Archard; a dry cell loses only this).
The loss is booked by mechanism in ``soil_removed_by`` (impact, shear, front, tool: wipe and abrasion;
``soil_removed`` is their sum).
When the film carries the soil species it joins the film (``src_species``, ``cwet``), or on a flooded cell the pool's
tank (DC5's ``pool_add_sp``, booked in the tank's species ledger and ``soil_to_pools``). Per world, ``soil +
soil_removed`` is the start load ``soil0`` (set at setup and by ``set_soil``).

**Drying** (P:704-728). The solids are the film's species ``solids`` in volume per area [m], so ``phi_s = solids/h``.
- Only solvent leaves, at ``k_m·c_sat·(1 - rh)/ρ`` (the per-cell ``evaporation`` input), never below ``solids/phi_m``.
  A substrate's uptake is solvent too, so absorption stops there as well. A cell first wetted during a film step
  starts drying at the next one.
- The dry film thickness is ``solids``. ``hide`` holds where it reaches ``hide_dft``.
- ``laps`` counts passes that land new paint (a rise in the solids' ``species_supplied``) on film already past the wet
  edge ``phi_we``.

**Absorption** (design §3.5). On a chart with a ``Substrate``, the per-cell ``absorption`` input is the exact
Lucas-Washburn increment of the depth taken so far (``absorbed = S·sqrt(t)`` under a standing film), up to the
capacity. The film books it in its ``absorbed`` ledger.

**Sanitizer** (FDA Food Code 4-703.11). ``expo`` [s] counts unbroken contact at ``c ≥ c_min``; ``expo_best`` keeps the
longest.

Stage calls (design §4.8). Kernels run per (world, tile), gated on the film's tick and its active tiles.
- ``before_film(state, dt)``: after the supply and drops, before the film step.
- ``rates(dt)``: inside the film's first substep, after its tile kernels. Writes yield, evaporation and absorption.
- ``after_film(state, dt)``: after the substeps. Advances age, soil, laps and the sanitizer clock.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow import process_kernels as PK
from alloy.plugins.surface_flow.host_api import H_WET
from alloy.plugins.surface_flow.process_specs import (Drying, Process, Sanitizer, Setting, Soil, Substrate,
                                                      process_spec)


class ProcessStage:
    """The stage ``SurfaceFlow`` calls (module docstring)."""

    def __init__(self, spec: Process | dict):
        self.spec = process_spec(spec)              # a dict from JSON, cards or the harness's config copies
        self.age = self.soil = self.laps = self.expo = self.sorp = None

    # --- setup ---------------------------------------------------------------------------------------------------

    def setup(self, ctx: Any, tables: Any, film: Any) -> None:
        self.ctx, self.tables, self.film = ctx, tables, film
        sp = self.spec
        if sp.setting is not None:
            self._setup_setting(sp.setting)
        if sp.soil is not None:
            self._setup_soil(sp.soil)
        if sp.drying is not None:
            self._setup_drying(sp.drying)
        if sp.substrates:
            self._setup_substrates(sp.substrates)
        if sp.sanitizer is not None:
            self._setup_sanitizer(sp.sanitizer)

    def _zeros(self, name: str, dtype: Any) -> wp.array:
        return self.film.add_state(name, self.ctx.zeros((self.ctx.world_count, self.tables.cell_count), dtype))

    def _chart(self, key: Any, part: str) -> int:
        """A chart index from an index or a name (the chart's ``shape``, else its ``body``)."""
        charts = self.film.config.charts
        if isinstance(key, (int, np.integer)) and 0 <= int(key) < len(charts):
            return int(key)
        names = [getattr(ch, "shape", None) or getattr(ch, "body", None) for ch in charts]
        if isinstance(key, str) and key in names:
            return names.index(key)
        raise PluginError(f"{self.film.label}: process.{part}: no chart {key!r}; charts by index 0..{len(charts) - 1} "
                          f"or by name {names}")

    def _species(self, name: str, part: str, required: bool) -> int:
        names = tuple(self.film.config.species)
        if name in names:
            return names.index(name)
        if required:
            raise PluginError(f"{self.film.label}: process.{part} needs the film to carry the species {name!r}; add "
                              f"it to Config.species (it carries {names})")
        return -1

    def _setup_setting(self, st: Setting) -> None:
        f, W, C = self.film, self.ctx.world_count, self.tables.cell_count
        if not f.config.per_cell_rheology:
            raise PluginError(f"{f.label}: process.setting writes the per-cell yield stress; set per_cell_rheology=True")
        self.every = 1 if f.bulk is not None else 0
        start = st.initial_age if self.every else -1.0
        self.age = f.add_state("age", self.ctx.full((W, C), start, wp.float64))   # stepped, reset, checkpointed
        f.add_input("age", self.age)                                               # and settable
        self.set_law = st.law()
        tau = float(st.yield_stress(start))                     # the yield before the first film step
        f.fluid.yield_cell.fill_(st.film_factor * tau)
        if self.every:
            f.bulk_yield.fill_(tau)

    def _setup_soil(self, so: Soil) -> None:
        f, W, C = self.film, self.ctx.world_count, self.tables.cell_count
        ks = [self._chart(k, "soil.charts") for k in so.charts]
        cchart = self.tables.cchart.numpy()
        on = np.isin(cchart, np.asarray(ks, np.int64)) if ks else np.ones(C, bool)
        init = np.where(on, so.initial, 0.0)[None, :].repeat(W, axis=0)
        self.soil = f.add_state("soil", self.ctx.array(init, wp.float64))
        self.soil0 = f.add_state("soil0", self.ctx.array(init, wp.float64))        # the start load (the front's)
        self.soil_prev = self.ctx.zeros((self.ctx.world_count, C), wp.float64)    # the step's start (scratch)
        self.soil_gate = self.ctx.zeros((self.ctx.world_count, self.film.tl.count), wp.int32)
        self.writers = self._writers()
        self.hyd = self._zeros("soil_hyd", wp.float64)
        self.removed_by = f.add_state("soil_removed_by", self.ctx.zeros((self.ctx.world_count, C, 4), wp.float64))
        self.soil_law = so.law(f.mat.density)
        self.soil_out = PK.SoilOut()
        o = self.soil_out
        o.soil, o.hyd, o.by, o.sp_src, o.cwet = self.soil, self.hyd, self.removed_by, f.sp.src, f.cwet
        self.k_soil = self._species(so.species, "soil", required=False)
        self.flood_to_tanks = self.k_soil >= 0 and getattr(f.supply, "flooding", False)
        o.pool_soil = self.ctx.zeros((W, C) if self.flood_to_tanks else (1, 1), wp.float64)
        self.soil_to_pools = f.add_state("soil_to_pools", self.ctx.zeros(W, wp.float64))   # kg into pools' tanks

    def _writers(self) -> PK.Writers | None:
        """The boxes jets and tools write each exchange (None: neither, and the soil runs on the film's tiles)."""
        f, ctx = self.film, self.ctx
        if f.supply is None and f.tools is None:
            return None
        wr = PK.Writers()
        none = ctx.zeros((1, 1), wp.vec4i)
        wr.boxes, wr.n_boxes = (f.supply.written, 1) if f.supply is not None else (none, 0)
        flood = getattr(f.supply, "flood_cur", None) if getattr(f.supply, "flooding", False) else None
        wr.flood, wr.n_flood = (flood, 1) if flood is not None else (none, 0)     # the cells DC5's pools flood
        if f.tools is not None:
            wr.wins, wr.tool_chart, wr.n_tools = f.tools.wins, f.tools.set.chart, f.tools.count
        else:
            wr.wins, wr.tool_chart, wr.n_tools = none, ctx.zeros(1, wp.int32), 0
        return wr

    def _setup_drying(self, dr: Drying) -> None:
        self.k_solids = self._species(dr.species, "drying", required=True)
        self.dry_law = dr.law()
        self.laps = self._zeros("laps", wp.int32)
        self.lap_open = self._zeros("lap_open", wp.int32)
        self.phi_prev = self._zeros("phi_prev", wp.float64)
        self.sup_prev = self._zeros("solids_supplied_prev", wp.float64)
        self.lap_seed = self._zeros("lap_seed", wp.int32)             # 1: the cell's solids once held solvent (not dust)

    def _setup_substrates(self, subs: dict[int, Substrate]) -> None:
        C = self.tables.cell_count
        cchart = self.tables.cchart.numpy()
        sorp, cap = np.zeros(C), np.zeros(C)
        for key, s in subs.items():
            k = self._chart(key, "substrates")
            sorp[cchart == k], cap[cchart == k] = s.sorptivity, s.capacity
        self.sorp, self.cap = self.ctx.array(sorp, wp.float64), self.ctx.array(cap, wp.float64)

    def _setup_sanitizer(self, sa: Sanitizer) -> None:
        self.k_san = self._species(sa.species, "sanitizer", required=True)
        self.c_min = float(sa.c_min)
        self.expo = self._zeros("expo", wp.float64)
        self.expo_best = self._zeros("expo_best", wp.float64)

    def set_soil(self, values: Any, chart: int | None = None, hyd: Any = 0.0) -> None:
        """Soil [kg/m²]: a scalar, per cell of ``chart`` ((nx, ny) or (W, nx, ny)), or all cells. The cells written
        start a new load (``soil0``, the front's) and a new ledger (removal 0), soaked to ``hyd`` (the same shapes;
        0: dry; a soil pre-soaked for t seconds: ``1 - exp(-t/Soil.t_soak_at)``); the others keep theirs."""
        if self.soil is None:
            raise PluginError(f"{self.film.label}: no process.soil configured")
        h = np.asarray(hyd, np.float64)
        if not np.all(np.isfinite(h)) or np.any(h < 0.0) or np.any(h > 1.0):
            raise PluginError(f"{self.film.label}: soil hydration lies in [0, 1], got {np.min(h)} to {np.max(h)}")
        kept = self.soil.numpy().copy(), self.hyd.numpy().copy()         # a bad shape leaves both as they were
        try:
            self.film._assign(self.soil, values, chart, "soil")
            self.film._assign(self.hyd, hyd, chart, "soil hydration")
        except Exception:
            self.soil.assign(kept[0])
            self.hyd.assign(kept[1])
            raise
        cells = slice(None) if chart is None else self.film._chart_cells(chart)
        soil0, by = self.soil0.numpy().copy(), self.removed_by.numpy().copy()
        soil0[:, cells], by[:, cells] = self.soil.numpy()[:, cells], 0.0
        self.soil0.assign(soil0)
        self.removed_by.assign(by)

    # --- stage calls ---------------------------------------------------------------------------------------------

    def _gate(self) -> list[Any]:
        f = self.film
        return [f.tl.tiles, f.tile_on, f.tick, f.stride]

    def before_film(self, state: Any, dt: float) -> None:
        """After the supply and the drops, before the film step and forming: a forming bulk's yield from its age, every
        exchange."""
        f = self.film
        if self.age is not None and f.bulk is not None:
            wp.launch(PK.bulk_yield_kernel, dim=(self.ctx.world_count, self.tables.cell_count),
                      inputs=[self.set_law, self.age], outputs=[f.bulk_yield], device=self.ctx.device)

    def rates(self, dt: float) -> None:
        """This film step's yield, evaporation and absorption (inside the first substep, after the tile kernels)."""
        f, dim, dev = self.film, (self.ctx.world_count, self.film.tl.count), self.ctx.device
        film_dt = float(dt) * f.stride
        if self.age is not None:
            wp.launch(PK.yield_kernel, dim=dim, inputs=[*self._gate(), self.every, self.set_law, self.age],
                      outputs=[f.fluid.yield_cell], device=dev)
        if self.laps is not None:
            wp.launch(PK.dry_rate_kernel, dim=dim,
                      inputs=[*self._gate(), self.dry_law, f.h, f.sp.mass, self.k_solids, film_dt],
                      outputs=[f.fluid.evap_cell], device=dev)
        if self.sorp is not None:                       # after drying: with drying on, never below solids/phi_m
            dry = self.laps is not None
            wp.launch(PK.absorb_rate_kernel, dim=dim,
                      inputs=[*self._gate(), self.sorp, self.cap, self.tables.area, f.book.absorbed, film_dt, int(dry),
                              self.dry_law if dry else PK.DryLaw(), f.h, f.sp.mass, self.k_solids if dry else 0,
                              f.fluid.evap_cell],
                      outputs=[f.fluid.absorb_cell], device=dev)

    def after_film(self, state: Any, dt: float) -> None:
        """Clocks and states of this film step (after the substeps)."""
        f, dim, dev = self.film, (self.ctx.world_count, self.film.tl.count), self.ctx.device
        film_dt = float(dt) * f.stride
        if self.age is not None:
            wp.launch(PK.age_kernel, dim=dim, inputs=[*self._gate(), self.every, f.h, H_WET, film_dt],
                      outputs=[self.age], device=dev)
        if self.soil is not None:
            gate = self._gate()                                               # the film's tiles, + soil under jets, tools
            if self.writers is not None:
                wp.launch(PK.soil_gate_kernel, dim=dim,
                          inputs=[*gate, f.cells, self.tables.charts, self.writers, self.soil, f.jet_cover, f.tf,
                                  f.flooded], outputs=[self.soil_gate], device=dev)
                gate = [f.tl.tiles, self.soil_gate, f.tick, f.stride]
            wp.launch(PK.soil_copy_kernel, dim=dim, inputs=[*gate, self.soil], outputs=[self.soil_prev],
                      device=dev)
            wp.launch(PK.soil_kernel, dim=dim,
                      inputs=[*gate, f.cells, f.faces, f.fluid, f.tf, self.soil_law, f.h, f.q, f.u,
                              f.impact_p, f.impact_share, f.jet_tau, f.jet_cover, f.pool_jet_tau, f.flooded,
                              self.soil_prev, self.soil0, film_dt, self.k_soil],
                      outputs=[self.soil_out], device=dev)
            if self.flood_to_tanks:                                       # flooded cells' soil into the tanks
                wp.launch(PK.pool_soil_kernel, dim=self.ctx.world_count,
                          inputs=[self.tables.charts, f.supply.flood_cur, f.flooded, self.tables.area, self.k_soil],
                          outputs=[self.soil_out.pool_soil, f.supply.pool_add_sp, self.soil_to_pools], device=dev)
        if self.laps is not None:
            wp.launch(PK.dry_state_kernel, dim=dim,
                      inputs=[*self._gate(), self.dry_law, f.h, H_WET, f.sp.mass, f.sp.supplied, self.k_solids],
                      outputs=[self.laps, self.lap_open, self.phi_prev, self.sup_prev, self.lap_seed], device=dev)
        if self.expo is not None:
            wp.launch(PK.sanitizer_kernel, dim=dim,
                      inputs=[*self._gate(), f.h, H_WET, f.sp.mass, self.k_san, self.c_min, film_dt],
                      outputs=[self.expo, self.expo_best], device=dev)

    # --- read ----------------------------------------------------------------------------------------------------

    def read(self) -> dict[str, np.ndarray]:
        """Per-cell maps ``[W, C]`` and per-world totals of the parts that are on."""
        out: dict[str, np.ndarray] = {}
        area = self.tables.area_np
        if self.soil is not None:
            soil, by = self.soil.numpy(), self.removed_by.numpy()
            removed = by.sum(axis=2)
            out.update(soil=soil, soil_hyd=self.hyd.numpy(), soil_removed=removed, soil_removed_by=by,
                       soil_mass=(soil * area).sum(axis=1), soil_removed_mass=(removed * area).sum(axis=1),
                       soil_to_pools_mass=self.soil_to_pools.numpy().copy(),
                       soil_removed_by_mass=(by * area[None, :, None]).sum(axis=1),   # [W, 4] impact/shear/front/wipe
                       soil_start_mass=(self.soil0.numpy() * area).sum(axis=1))
        if self.age is not None:
            out["age"] = self.age.numpy()
        if self.laps is not None:
            dft = self.film.sp.mass.numpy()[:, :, self.k_solids]
            h = self.film.h.numpy()
            phi = np.where(h > H_WET, dft / np.maximum(h, H_WET), 0.0)
            out.update(dft=dft, phi_s=np.minimum(phi, self.spec.drying.phi_m), laps=self.laps.numpy(),
                       hide=dft >= self.spec.drying.hide_dft)
        if self.sorp is not None:
            out["absorbed_depth"] = self.film.book.absorbed.numpy() / area
        if self.expo is not None:
            out.update(expo=self.expo.numpy(), expo_best=self.expo_best.numpy())
        return out


__all__ = ["Drying", "Process", "ProcessStage", "Sanitizer", "Setting", "Soil", "Substrate", "process_spec"]
