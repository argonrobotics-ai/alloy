# SPDX-License-Identifier: Apache-2.0
"""``SurfaceFlow``: a thin film of one material on a set of charts, stepped after every solver step.

``update`` resolves each chart's gravity in its frame (with the body's acceleration and spin), then runs ``n_sub``
film substeps (``kernels.py``; ``n_sub`` is fixed by ``check_dt`` for the exchange step, so CUDA graphs capture it),
or one strided step every k exchanges for a slow film. Only the tiles near wet cells are stepped (``tiles.py``).
Other stages run inside the hooks, each writing the film by the interface note's rules (``src_acc`` + ``cwet``):
- DC5's ``Supply`` (nozzles, drains, network): ``apply`` casts the jets and pushes the bodies; ``update`` deposits
  before the substeps (and clears what it wrote the step before), then drains into the tanks and steps the network;
- the applicators (``applicators.py``): the tools' ceilings and forces, and plastic forming of a paste's bulk;
- WP9's drop stage (design §4.7): absorption before the substeps, shedding after them.

Host API (between control periods): ``set_depth`` (the film at the start), ``set_source`` (a steady inflow per cell,
m/s), ``depth`` (the film per chart, ``[W, nx, ny]``), ``read`` (state plus the per-world ledger).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import Law, TestScene
from alloy.plugins.surface_flow import adsorption as AD
from alloy.plugins.surface_flow import process_adsorb as PA
from alloy.plugins.surface_flow import applicators as AP
from alloy.plugins.surface_flow import cell_kernels as CK
from alloy.plugins.surface_flow import forming as FM
from alloy.plugins.surface_flow import rheology as RH
from alloy.plugins.surface_flow import stores as ST
from alloy.plugins.surface_flow import kernels as K
from alloy.plugins.surface_flow import tiles as TL
from alloy.plugins.surface_flow import tool_steps as TS
from alloy.plugins.surface_flow.charts import PlaneChart, chart_spec, link_spec
from alloy.plugins.surface_flow.film import FilmMaterial, material
from alloy.plugins.surface_flow.host_api import FREE_DRAIN, H_DRY, H_WET, LEDGER, FilmHostAPI
from alloy.plugins.surface_flow.links import check_wetting
from alloy.plugins.surface_flow.supply import Supply
from alloy.plugins.surface_flow.tables import ChartTables, build_tables

log = logging.getLogger("alloy.plugins.surface_flow")



class SurfaceFlow(FilmHostAPI, ForcePlugin):
    """A depth-averaged film (lubrication, Bingham and shallow-water regimes in one update) on charts.

    Example::

        flow = SurfaceFlow(charts=(PlaneChart("plate", cells=(80, 40)),), material="water")
        sim = alloy.plugins.simulation(built, [flow])
        flow.set_depth(1e-3)                       # 1 mm everywhere, every world
        ...; flow.depth(0)[0]                      # world 0's film on chart 0 [m], (nx, ny)
    """

    name = "surface_flow"
    summary = "thin films on surfaces: a depth-averaged film (paint, paste, water) on plane and revolution charts"

    @dataclass
    class Config:
        charts: tuple[Any, ...] = (PlaneChart("plate", cells=(32, 32)),)
        links: tuple[Any, ...] = ()                   # charts.Link (or tuples, dicts): edges joined so film crosses
        material: Any = "water"                       # a name in film.MATERIALS or a FilmMaterial
        inertia: bool = True                          # momentum advection (off: the pure lubrication/creeping limit)
        u_cap: float = 1.5                            # m/s: the film's speed limit (interface note §4)
        max_depth: float = 5.0e-3                     # m: the deepest film the step bound assumes
        max_substeps: int = 64                        # refuse an exchange step needing more film substeps
        max_stride: int = 16                          # a slow film steps every k <= max_stride exchanges (1: every one)
        skip_dry: bool = True                         # step only the tiles near wet cells (exact: tiles.py)
        tools: tuple[Any, ...] = ()                   # applicators.Applicator (or dicts): tools working the film
        tool_speed: float = 0.5                       # m/s: the fastest a tool moves (a film step moves <= 1 cell)
        bulk: Any = None                              # forming.Bulk (or a dict): base is a paste's plastic surface
        trays: tuple[Any, ...] = ()                   # stores.Tray (or dicts): pools the tools' stores pick up from
        species: tuple[str, ...] = ()                 # names of species the film carries (surfactant, soil; <= 4)
        drops: Any = None                             # WP9's drop stage config (design §4.7); None: open edges spill
        sph: bool = False                             # drops become SPH particles and thin SPH liquid is absorbed
        per_cell_rheology: bool = False               # inputs "yield_stress" [Pa] and "viscosity" [Pa·s] per cell
        per_cell_sinks: bool = False                  # inputs "evaporation", "absorption" [m/s] (on with process)
        process: Any = None                           # WP9's process stage (design §4.8): drying, soil, setting ...
        lumped: tuple[Any, ...] = ()                  # WP9's lumped films (§4.9): lumped_specs.Lumped (or dicts)
        thixotropy: Any = None                        # rheology.Thixotropy (or a dict): Moore's structure on the yield
        krieger: float | None = None                  # phi_m: the 'solids' species raise the viscosity (rheology.py)
        krieger_ref: float = 0.0                      # the solids fraction where the viscosity is the card's (phi_0)
        adsorption: Any = None                        # adsorption.Adsorption(s): species on the walls (Langmuir)
        accel_margin: float = 1.5                     # the step bound assumes |g|·margin (frame accelerations)
        network: Any = None                           # the supply (DC5, interface note §1): network data in
        nozzles: tuple[Any, ...] = ()                 # FluidNetwork.from_dict's format (tanks, "pools"), the
        drains: tuple[Any, ...] = ()                  # nozzles and the drains

        def __post_init__(self) -> None:
            # dicts come from JSON, cards and the harness's copies (dataclasses.asdict): rebuild the dataclasses
            self.charts = tuple(chart_spec(ch) for ch in self.charts)
            self.links = tuple(link_spec(ln) for ln in self.links)
            self.tools = tuple(AP.applicator_spec(t) for t in self.tools)
            self.lumped = tuple(self.lumped)
            self.bulk = FM.bulk_spec(self.bulk)
            self.trays = tuple(ST.tray_spec(t) for t in self.trays)
            self.species = tuple(self.species)
            if len(self.species) > CK.MAX_SPECIES or len(set(self.species)) != len(self.species):
                raise ValueError(f"species must be at most {CK.MAX_SPECIES} distinct names, got {self.species}")
            if isinstance(self.material, dict):
                self.material = FilmMaterial(**self.material)
            material(self.material)
            for name in ("u_cap", "max_depth", "accel_margin", "tool_speed"):
                v = getattr(self, name)
                if not (math.isfinite(v) and v > 0.0):
                    raise ValueError(f"{name} must be finite and > 0, got {v}")
            if self.max_substeps < 1 or self.max_stride < 1:
                raise ValueError(f"max_substeps and max_stride must be >= 1, got {self.max_substeps}, "
                                 f"{self.max_stride}")

    # --- setup ---------------------------------------------------------------------------------------------------

    def setup(self, ctx: Any) -> None:
        c = self.config
        self.mat: FilmMaterial = material(c.material)
        t = self.tables = build_tables(ctx, c.charts, c.links)
        W, C, F, Kc = ctx.world_count, t.cell_count, t.face_count, len(c.charts)
        self.gravity = wp.vec3(*(float(v) for v in ctx.gravity))
        self.cells, self.faces = _cells(t), _faces(t)
        self.tl = TL.build_tiles(c.charts, t, ctx.device)
        self.cells.tile, self.faces.tile = self.tl.ctile, self.tl.ftile
        T = self.tl.count
        TS.setup_tools(self, ctx, W, C)                    # applicators, stores (tool_steps.py)
        self.fluid = K.Fluid()
        m = self.mat
        self.fluid.rho, self.fluid.nu, self.fluid.tau_y = m.density, m.nu, m.yield_stress
        self.fluid.sigma, self.fluid.beta, self.fluid.u_cap = m.surface_tension, 1.0 if c.inertia else 0.0, c.u_cap
        self.fluid.h_dry, self.fluid.h_wet = H_DRY, H_WET
        self.fluid.evaporation = m.evaporation
        self.fluid.per_cell = 1 if c.per_cell_rheology else 0
        n = (W, C) if c.per_cell_rheology else (1, 1)      # a setting layer stiffens the paste through these inputs
        self.fluid.yield_cell = ctx.full(n, m.yield_stress, wp.float32)
        self.fluid.visc_cell = ctx.full(n, m.viscosity, wp.float32)
        if c.per_cell_rheology:
            self.add_input("yield_stress", self.fluid.yield_cell)
            self.add_input("viscosity", self.fluid.visc_cell)     # never below the material's (the step bound)
        sinks = c.per_cell_sinks or c.process is not None
        self.fluid.per_sink = 1 if sinks else 0
        n = (W, C) if sinks else (1, 1)
        self.fluid.evap_cell = ctx.full(n, m.evaporation, wp.float32)
        self.fluid.absorb_cell = ctx.zeros(n, wp.float32)
        if sinks:
            self.add_input("evaporation", self.fluid.evap_cell)
            self.add_input("absorption", self.fluid.absorb_cell)
        wet = [self._wetting(ch) for ch in c.charts]
        check_wetting(c.links, c.charts, wet)                 # linked charts wet alike (links.py)
        self.wet = ctx.array(np.asarray(wet, np.float32), wp.vec2)
        RH.setup_rheology(self, ctx, W, C)               # thixotropy and solids (after the species are known)
        AD.setup_adsorption(self, ctx, W, C)             # species on the walls (Langmuir)
        self.frame = K.Frame()
        self.frame.g, self.frame.omega, self.frame.com = ctx.zeros((W, Kc), wp.vec3), ctx.zeros((W, Kc), wp.vec3), t.com
        self.frame.alpha = ctx.zeros((W, Kc), wp.vec3)
        self.g_sum = self.add_state("g_sum", ctx.zeros((W, Kc), wp.vec3))           # frame sums over a stride window
        self.alpha_sum = self.add_state("alpha_sum", ctx.zeros((W, Kc), wp.vec3))
        self.h = self.add_state("h", ctx.zeros((W, C), wp.float64))
        self.u = self.add_state("u", ctx.zeros((W, F), wp.float32))            # face velocity [m/s]
        self.q = self.add_state("q", ctx.zeros((W, F), wp.float32))            # face discharge [m²/s]
        self.gface = self.add_state("gface", ctx.zeros((W, F), wp.float32))
        self.base = self.add_input("base", ctx.zeros((W, C), wp.float32))
        self.src_fixed = self.add_input("source", ctx.zeros((W, C), wp.float64))
        # pending momentum per area [N·s/m²]: writers add (drops' arrivals), the film applies it over its step
        self.src_mom = self.add_state("src_mom", ctx.zeros((W, C), wp.vec3))
        self.impact_p = self.add_state("impact_p", ctx.zeros((W, C), wp.float32))
        self.impact_share = self.add_state("impact_share", ctx.zeros((W, C), wp.float32))   # impact_p's footprint
        self.jet_cover = self.add_state("jet_cover", ctx.zeros((W, C), wp.float32))
        self.jet_tau = self.add_state("jet_tau", ctx.zeros((W, C), wp.float32))
        self.pool_jet_tau = self.add_state("pool_jet_tau", ctx.zeros((W, C), wp.float32))    # under pools (DC5 writes)
        self.book = CK.Book()
        for name in LEDGER:
            setattr(self.book, name, self.add_state(name, ctx.zeros((W, C), wp.float64)))
        self.drained = self.book.drained = self.add_state("drained", ctx.zeros((W, C), wp.float64))
        self.drain_cap = self.book.drain_cap = self.add_state("drain_cap", ctx.full((W, C), FREE_DRAIN, wp.float64))
        self.drain_of = self.book.drain_of = self.cells.drain = ctx.full(C, -1, wp.int32)   # the supply fills it
        self.shed = self.book.shed = self.add_state("shed", ctx.zeros((W, C), wp.float64))       # drops (§4.7)
        self.flooded = self.book.flooded = self.add_state("flooded", ctx.full((W, C), -1, wp.int32))   # pools (DC5)
        self.pooled = self.book.pooled = self.add_state("pooled", ctx.zeros((W, C), wp.float64))
        self.travel = self.book.travel = self.add_state("travel", ctx.zeros((W, C), wp.float32))  # paint sag/run
        self._setup_species(ctx, W, C)
        self.shed_mom = self.book.shed_mom = self.add_state("shed_mom", ctx.zeros((W, C), wp.vec3))
        self.pot, self.q_new = ctx.zeros((W, C), wp.float32), ctx.zeros((W, F), wp.float32)
        self.u_new, self.g_new = ctx.zeros((W, F), wp.float32), ctx.zeros((W, F), wp.float32)
        self.ratio = ctx.zeros((W, C), wp.float64)
        self.tick = self.add_state("tick", ctx.zeros(W, wp.int32))
        self.cwet = self.add_state("cwet", ctx.zeros((W, C), wp.int32))            # cells whose film can change
        self.tile_act = self.add_state("tile_act", ctx.zeros((W, T), wp.int32))     # tiles active at the last substep
        self.tile_wet = ctx.zeros((W, T), wp.int32)
        self.tile_on = ctx.full((W, T), 0 if c.skip_dry else 1, wp.int32)          # tiles the substep steps
        self.src_acc = self.add_state("src_acc", ctx.zeros((W, C), wp.float64))   # inflow kept between film steps
        self.n_sub, self.stride, self._exchange_dt = 1, 1, 0.0
        self._dx = float(np.min(t.fdist.numpy()))
        self.lumped = None
        if c.lumped:                                         # WP9's lumped films (§4.9), before the supply (its hook)
            from alloy.plugins.surface_flow import lumped as LU  # noqa: PLC0415
            self.lumped = LU.LumpedStage(c.lumped)
            self.lumped.setup(ctx, t, self)
        self.supply = Supply(c.nozzles, c.drains, c.network) if self.has_supply else None
        if self.supply is not None:
            self.supply.setup(ctx, t, self)                  # registers its network, hits, ledger and inputs
            if self.lumped is not None:
                self.lumped.bind_pools(self.supply)          # DC5's pool exchange (lumped_pool.py)
        PA.setup_pool_adsorption(self, ctx)                  # walls under a pool's water (process_adsorb.py)
        self.drops = None
        if c.drops is not None or c.sph:                     # WP9's stage (design §4.7), bound to SPH lazily
            from alloy.plugins.surface_flow import drops as DR  # noqa: PLC0415
            self.drops = DR.Drops(c.drops, sph=c.sph)
            self.drops.setup(ctx, t, self)
        self.process = None
        if c.process is not None:                            # WP9's process stage (design §4.8)
            from alloy.plugins.surface_flow import process as PR  # noqa: PLC0415
            self.process = PR.ProcessStage(c.process)
            self.process.setup(ctx, t, self)
        log.info("surface_flow %s: %s on %d charts, %d cells, %d faces, smallest cell %.3g mm", self.label, m.name,
                 Kc, C, F, 1e3 * self._dx)
        from alloy.plugins.surface_flow.oneway import warn_one_way  # noqa: PLC0415
        self.one_way = warn_one_way(self, ctx)               # what passes through the film (known gap), or None

    def _setup_species(self, ctx: Any, W: int, C: int) -> None:
        """Species the film carries (cell_kernels.Species): mass per area, pending sources, ledgers."""
        K = len(self.config.species)
        sp = self.sp = CK.Species()
        sp.count = K
        n = (W, C, K) if K else (1, 1, 1)
        names = ("species", "src_species", "species_supplied", "species_out", "drained_species", "shed_species",
                 "species_clipped")
        arrays = [ctx.zeros(n, wp.float64) for _ in names]
        if K:
            arrays = [self.add_state(name, a) for name, a in zip(names, arrays)]
        sp.mass, sp.src, sp.supplied, sp.out, sp.drained, sp.shed, sp.clipped = arrays
        sp.conc = ctx.zeros(n, wp.float64)
        self.tf.sp_ex = ctx.zeros(n if self.tools is not None else (1, 1, 1), wp.float64)   # tools move species too
        self.tf.absorb_sp = ctx.zeros(n if self.tools is not None else (1, 1, 1), wp.float64)   # what pads absorb
        sp.src_conc = ctx.zeros(max(K, 1), wp.float64)

    def _wetting(self, chart: Any) -> tuple[float, float]:
        """(h_cl, tan θ) of ``chart``'s contact-line condition (tan θ = -1: none). The chart's angle overrides the
        material's (one liquid on glass, steel, PTFE ...)."""
        angle = chart.contact_angle if chart.contact_angle is not None else self.mat.contact_angle
        if angle is None:
            return (H_WET, -1.0)
        if not 0.0 < angle < 90.0:
            raise PluginError(f"{self.label}: contact angle {angle}° is outside (0, 90)°: the height-field film "
                              f"cannot hold steeper edges (phase 1)")
        return (H_WET, math.tan(math.radians(angle)))

    # --- stability -----------------------------------------------------------------------------------------------

    def step_bound(self) -> float:
        """The largest stable film substep [s] (design note §3.3) at ``max_depth``."""
        c, m, dx = self.config, self.mat, self._dx
        g = float(np.linalg.norm(np.asarray(self.gravity))) * c.accel_margin
        g_cap = 8.0 * m.surface_tension / (m.density * dx * dx)

        def bound(h: float, g_eff: float) -> float:
            if g_eff <= 0.0:
                return math.inf
            k4 = dx * dx * 3.0 * m.nu / (h * h) / 4.0
            a = g_eff * h
            return (k4 + math.sqrt(k4 * k4 + a * dx * dx)) / (2.0 * a)

        h = c.max_depth
        # the fastest film: the speed cap, or the creeping speed h²·G/(3ν) under gravity plus a one-cell bump
        u = min(c.u_cap, h * h / (3.0 * m.nu) * (g + (g + g_cap) * h / dx))
        if c.tools:
            u = max(u, 0.5 * c.tool_speed)                 # film sheared under a sliding tool (the Couette mean)
        return min(bound(h, g + g_cap), 0.5 * dx / u if u > 0.0 else math.inf)

    def check_dt(self, dt: float) -> None:
        """Film substeps per exchange (fast films), or a stride: a slow film (paint, paste) steps once every
        ``stride`` exchanges with the summed step, up to ``max_stride``."""
        self._film_steps(dt)
        if self.tools is not None:
            AP.check_tools(self.label, self.tools, dt)
        if self.stores is not None:
            ST.check_bins(self.label, self.config.tools, self.config.tool_speed, dt)

    def _film_steps(self, dt: float) -> None:
        self._exchange_dt = float(dt)
        bound = self.step_bound()
        cap = self.config.max_stride
        if self.tools is not None:                      # a tool moves at most a cell per film step
            cap = min(cap, max(1, int(self._dx / (self.config.tool_speed * dt))))
        if bound >= 2.0 * dt and cap > 1:
            self.n_sub, self.stride = 1, int(min(cap, math.floor(bound / dt)))
            return
        self.stride = 1
        n = max(1, math.ceil(dt / bound - 1e-9))
        if n > self.config.max_substeps:
            raise PluginError(f"{self.label}: an exchange step of {dt:.3g} s needs {n} film substeps (limit "
                              f"{self.config.max_substeps}; the stable substep is {self.step_bound():.3g} s): use "
                              f"coarser cells (smallest {self._dx * 1e3:.3g} mm), a smaller max_depth or u_cap, or a "
                              f"shorter exchange step")
        self.n_sub = n

    # --- the step ------------------------------------------------------------------------------------------------

    @property
    def has_supply(self) -> bool:
        c = self.config
        return bool(c.nozzles or c.drains or c.network)

    @property
    def applies_forces(self) -> bool:
        """The supply (jet impact and thrust, interface note §4), the applicators (pad pressure and shear) and the
        drop stage (film weight, absorbed impulse, debris drag: design §4.7) and the lumped bodies' water weight."""
        c = self.config
        return self.has_supply or bool(c.tools) or c.drops is not None or c.sph or bool(c.lumped)

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        if self.supply is not None:
            self.supply.apply(state, forces, dt)
        if self.drops is not None:
            self.drops.apply(state, forces, dt)
        if self.lumped is not None:
            self.lumped.apply(state, forces, dt)                 # the lumped bodies' water weight (§4.9)
        TS.tools_apply(self, state, forces, dt)                 # the tools' pads and drag (tool_steps.py)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        ctx, t = self.ctx, self.tables
        W, C, F = ctx.world_count, t.cell_count, t.face_count
        dev = ctx.device
        if self.lumped is not None:
            self.lumped.before_film(state_in, dt)              # sponges' room for their porous pads (tool_room)
        if self.supply is not None:
            self.supply.before_film(state_in, dt)              # deposits, the jet sweep, drain capacities
        if self.drops is not None:
            self.drops.absorb(state_in, state_out, dt)          # arriving drops and SPH liquid: src_acc, cwet
        if self.process is not None:
            self.process.before_film(state_in, dt)              # §4.8: sources it adds (src_species, cwet)
        k = self.stride
        wp.launch(CK.tick_kernel, dim=W, inputs=[], outputs=[self.tick], device=dev)
        wp.launch(K.frames_kernel, dim=(W, len(self.config.charts)),
                  inputs=[state_out.body_q, state_in.body_qd, state_out.body_qd, t.cbody, t.clocal, self.gravity,
                          float(dt), self.tick, k],
                  outputs=[self.g_sum, self.alpha_sum, self.frame.g, self.frame.omega, self.frame.alpha], device=dev)
        film_dt = float(dt) * k
        sub = film_dt / self.n_sub
        due = [self.tick, k, self.tile_on]
        TS.tools_exchange(self, state_in, state_out, dt, due)  # ceilings, stores, forming (tool_steps.py)
        T = self.tl.count
        inv = 1.0 / film_dt                                    # the pending inflow is applied over the film step
        for s in range(self.n_sub):
            if self.config.skip_dry:
                wp.launch(TL.tile_wet_kernel, dim=(W, T), inputs=[self.tl.tiles, self.tick, k, self.cwet],
                          outputs=[self.tile_wet], device=dev)
                wp.launch(TL.tile_on_kernel, dim=(W, T), inputs=[self.tl.tiles, self.tick, k, self.tile_wet],
                          outputs=[self.tile_act, self.tile_on], device=dev)
            if s == 0 and self.process is not None:
                self.process.rates(dt)                          # §4.8: per-cell sinks and yield, on this step's tiles
            if self.fluid.krieger:                              # the solids' viscosity factor (rheology.py)
                wp.launch(RH.solids_kernel, dim=(W, C),
                          inputs=[self.cells, *due, self.h, self.sp, self.solids_index, self.phi_m, self.phi_ref,
                                  H_WET],
                          outputs=[self.fluid.vfac], device=dev)
            wp.launch(K.potential_kernel, dim=(W, C),
                      inputs=[self.cells, self.faces, self.frame, self.fluid, self.wet, *due, self.h, self.base],
                      outputs=[self.pot], device=dev)
            wp.launch(K.face_kernel, dim=(W, F),
                      inputs=[self.cells, self.faces, self.frame, self.fluid, self.wet, self.tf, *due, self.h,
                              self.base, self.pot, self.u, self.src_mom, self.q, self.gface,
                              self.src_acc, inv,
                              self.src_fixed, sub], outputs=[self.u_new, self.q_new, self.g_new], device=dev)
            wp.launch(CK.limit_kernel, dim=(W, C),
                      inputs=[self.cells, self.faces, *due, self.h, self.q_new, self.src_acc, inv, self.src_fixed,
                              sub],
                      outputs=[self.ratio], device=dev)
            if self.sp.count:                           # the donors' species concentrations, before anything moves
                wp.launch(CK.conc_kernel, dim=(W, C), inputs=[self.cells, *due, H_DRY, self.h, self.sp], device=dev)
            wp.launch(CK.cell_kernel, dim=(W, C),
                      inputs=[self.cells, self.faces, self.fluid, self.wet, *due, self.q_new, self.u_new,
                              self.ratio,
                              inv, self.src_fixed, int(s == self.n_sub - 1), sub, self.book],
                      outputs=[self.h, self.cwet, self.src_acc, self.sp, self.tf, self.src_mom], device=dev)
            TS.tools_substep(self, due)                         # film above a tool's underside moves on
            wp.launch(CK.limited_kernel, dim=(W, F),
                      inputs=[self.faces, *due, self.q_new, self.ratio, self.u_new, self.g_new],
                      outputs=[self.q, self.u, self.gface], device=dev)
            if self.thixo is not None:                          # Moore's structure at the moved discharge
                wp.launch(RH.structure_kernel, dim=(W, C),
                          inputs=[self.cells, self.faces, *due, self.fluid, self.h, self.base, self.q, self.tf,
                                  self.thixo.rebuild, self.thixo.breakdown, sub], device=dev)
        AD.step_adsorption(self, dt, H_DRY)                     # species on and off the walls (Langmuir)
        if self.pool_adsorption is not None:
            self.pool_adsorption.step(dt)                       # and under a pool's water (process_adsorb.py)
        if self.drops is not None:
            self.drops.shed(state_out, dt)                      # shed, dripped: particles or ballistic drops
        if self.process is not None:
            self.process.after_film(state_out, dt)              # §4.8: clocks, drying outputs, erosion sources
        if self.lumped is not None:
            self.lumped.after_film(state_out, dt)               # §4.9: the bodies' water, soil, clocks, squeeze-out
        if self.supply is not None:
            self.supply.after_film(state_out, dt)              # drains into tanks, the network step, pool levels

    def test_scene(self) -> TestScene:
        from alloy.plugins.surface_flow import laws
        return laws.harness_scene(self.config)

    def laws(self) -> list[Law]:
        from alloy.plugins.surface_flow import laws
        return laws.surface_flow_laws(self.config)


def _cells(t: ChartTables) -> K.Cells:
    c = K.Cells()
    c.pos, c.nrm, c.area, c.curv, c.chart, c.face = t.cpos, t.cnrm, t.area, t.curv, t.cchart, t.cface
    return c


def _faces(t: ChartTables) -> K.Faces:
    f = K.Faces()
    f.a, f.b, f.chart, f.line, f.par, f.perp = t.fa, t.fb, t.fchart, t.fline, t.fpar, t.fperp
    f.kind = wp.clone(t.ftype)                        # the film's own: set_regions walls faces between regions
    f.pos, f.dir, f.dira, f.dirb, f.length, f.dist = t.fpos, t.fdir, t.fdira, t.fdirb, t.flen, t.fdist
    return f


__all__ = ["SurfaceFlow"]
