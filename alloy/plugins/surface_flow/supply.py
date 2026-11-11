# SPDX-License-Identifier: Apache-2.0
"""The surface-flow supply side (design note §4.3, interface note §4): a lumped network, nozzles and drains.

``Supply`` runs inside ``SurfaceFlow``'s hooks. It compiles the network from data (``network``: ``FluidNetwork``'s
dict format, tanks as ``medium="tank"`` nodes, plus ``pools``: where a tank's free surface is), and each exchange:

- ``apply``: casts every nozzle's rays along its ballistic path (charts, pools, occluding shapes) at the pre-solve
  poses, stores the hits, and puts the jets' momentum on the struck bodies and the thrust on the nozzles' bodies;
- ``before_film``: enters each nozzle's last-step discharge on the film (a solid jet at its jump radius after
  sweeping the film inside it; sprays over a footprint per ray), into pools, or books it lost; writes ``drain_cap``;
- ``after_film``: sums each drain's ``drained`` into its tank's inflow, steps the network, and publishes the pools'
  levels.

Example (a faucet over a basin floor chart that drains into the sink's tank)::

    supply = Supply(
        nozzles=(Nozzle("faucet", edge="spout", pos=(0.0, 0.0, 0.30), exit_area=1.3e-4),),
        drains=(Drain("waste", chart=0, center=(0.0, 0.0, 0.0), radius=0.02, to="sink"),),
        network={"nodes": [{"name": "line", "medium": "liquid", "volume": 1e-4},
                           {"name": "sink", "medium": "tank", "volume": 0.0, "area": 0.16}],
                 "edges": [{"name": "valve", "kind": "orifice", "a": "line", "reservoir": 4e5, "cd_area": 2e-5},
                           {"name": "spout", "kind": "orifice", "a": "line", "cd_area": 1.3e-4}],
                 "pools": [{"tank": "sink", "floor": (0.0, 0.0, -0.2), "half": (0.2, 0.2)}]})
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.fluid import G_TANK, Edge, FluidState, Network, Node
from alloy.plugins.surface_flow import supply_cavity as CV
from alloy.plugins.surface_flow import supply_deposit as D
from alloy.plugins.surface_flow import supply_carry as CA
from alloy.plugins.surface_flow import supply_float as FL
from alloy.plugins.surface_flow import supply_holes as HO
from alloy.plugins.surface_flow import supply_jetting as JT
from alloy.plugins.surface_flow import supply_kernels as K
from alloy.plugins.surface_flow import supply_lump as LM
from alloy.plugins.surface_flow import supply_pool as PL
from alloy.plugins.surface_flow import supply_shadow as SH
from alloy.plugins.surface_flow import supply_species as SP
from alloy.plugins.surface_flow import supply_stage as SG
from alloy.plugins.surface_flow.charts import REVOLUTION
from alloy.plugins.surface_flow.supply_spec import PATTERNS, STAGE, Drain, Nozzle, Pool, coerce, ray_table
from alloy.plugins.surface_flow.tables import ChartTables

log = logging.getLogger("alloy.plugins.surface_flow")

LEDGER = ("sent", "swept", "to_tank", "lost", "overspray", "away", "drained", "pooled", "withdrawn", "refused",
          "weir", "mist", "caught", "lumped", "holes", "lump_pool", "staged")


class Supply(JT.Jetting, PL.Flooding, LM.Lumping, SH.Spraying, FL.Floating, CA.Carrying, HO.Holing, SG.Staging):
    """Nozzles, drains and pools on a lumped network, run by ``SurfaceFlow`` (see the module docstring)."""

    def __init__(self, nozzles: Any = (), drains: Any = (), network: dict[str, Any] | None = None) -> None:
        self.nozzles = coerce(Nozzle, nozzles)
        self.drains = coerce(Drain, drains)
        spec = dict(network or {})
        self.pools = coerce(Pool, spec.pop("pools", ()))
        self.nozzles += HO.drip_nozzles(self.pools)            # a holed vessel's drips (supply_holes)
        self.floats = coerce(FL.Float, spec.pop("floats", ()))
        nodes = coerce(Node, spec.get("nodes", ())) or (Node("_none", medium="tank", area=1.0, volume=0.0),)
        self.net = Network(nodes, coerce(Edge, spec.get("edges", ())))
        self.openings = dict(spec.get("openings", {}))
        self.species_names = tuple(spec.get("species", ()))   # the network's species, by position
        self._validate()

    def _validate(self) -> None:
        net, tanks = self.net, {n.name for n in self.net.nodes if n.medium == "tank"}
        edges = {e.name: e for e in net.edges}
        owner: dict[str, str] = {}
        for noz in (n for n in self.nozzles if not n.drip_of):
            if noz.edge in owner:
                raise PluginError(f"nozzles {owner[noz.edge]!r} and {noz.name!r} share edge {noz.edge!r}: a nozzle "
                                  f"sends its edge's whole discharge, so give each nozzle its own edge")
            owner[noz.edge] = noz.name
            e = edges.get(noz.edge)
            kinds = ("orifice", "linear", "expel", "source") if noz.rim_of else ("orifice", "linear", "expel")
            if e is None or e.b is not None or e.kind not in kinds:
                rim = " (a rim nozzle: or its cavity's spill source edge)" if noz.rim_of else ""
                raise PluginError(f"nozzle {noz.name!r}: edge {noz.edge!r} must be an orifice or linear edge from a "
                                  f"liquid node, or an aerosol can's expel edge, to the atmosphere (b=None){rim}; "
                                  f"edges: {sorted(edges)}")
        for d in self.drains:
            if d.to is not None and d.to not in tanks:
                raise PluginError(f"drain {d.name!r}: {d.to!r} is not a tank node (tanks: {sorted(tanks)})")
            if not d.open_edges and d.radius <= 0:
                raise PluginError(f"drain {d.name!r}: give a radius > 0 or open_edges=True")
        for p in self.pools:
            if p.tank not in tanks:
                raise PluginError(f"pool of {p.tank!r}: not a tank node (tanks: {sorted(tanks)})")
        flooding = [p.tank for p in self.pools if p.flood]
        if len(set(flooding)) < len(flooding):
            raise PluginError(f"flooding pools need a tank each (each draws withdrawal films from its own): "
                              f"{sorted(flooding)}")
        for name in self.openings:
            if name not in edges:
                raise PluginError(f"openings: {name!r} is not an edge")
        CV.check_cavities(self.pools, self.nozzles, edges)

    # ---- setup -------------------------------------------------------------------------------------------------

    def setup(self, ctx: Any, charts: ChartTables, film: Any) -> None:
        """Compile the network, bind nozzles, drains and pools, register state and inputs on ``film`` (the
        ``SurfaceFlow`` plug-in: its film arrays plus ``add_state`` / ``add_input``)."""
        self.ctx, self.charts, self.film = ctx, charts, film
        self.nozzles = tuple(n for n in self.nozzles if n.drip_of != STAGE) + SG.stage_nozzles(film)   # its drips
        W, dev = ctx.world_count, ctx.device
        self.gravity = np.asarray(ctx.gravity, np.float64)
        g = float(np.linalg.norm(self.gravity))
        self.fluid: FluidState = self.net.compile(dev, W, gravity=g if g > 0.0 else G_TANK)
        for name, arr in self.fluid.state_arrays().items():
            film.add_state(f"net_{name}", arr)
        self._setup_openings(ctx, film)
        self._setup_nozzles(ctx)
        self._setup_pools(ctx)
        self._setup_cavities(ctx)
        self._setup_drains(ctx, film)
        occ = self._occluders(ctx)
        self.occ = ctx.array(occ, wp.int32)
        labels = list(getattr(ctx.model, "shape_label", None) or [])
        self.occluders = tuple(labels[s] if 0 <= s < len(labels) else str(s) for s in occ[0] if s >= 0)
        self.occluder_shapes = occ                          # [W, S] model shape per slot (-1: none), per world
        self.caught_by = film.add_state("supply_caught_by", ctx.zeros((W, occ.shape[1]), wp.float64))
        R = max(self.ray_count, 1)
        hit = {"kind": wp.int32, "idx": wp.int32, "cell": wp.int32, "p": wp.vec3, "w": wp.vec3, "s": wp.float32}
        self.hit = {k: film.add_state(f"jet_hit_{k}", ctx.zeros((W, R), t)) for k, t in hit.items()}
        self.hit["on"] = film.add_state("jet_hit_on", ctx.full((W, R), 1.0, wp.float32))   # sheet stays on the chart
        self.ledger = film.add_state("supply_ledger", ctx.zeros((W, K.N_LEDGER), wp.float64))
        self.level = film.add_state("tank_level", ctx.zeros((W, max(len(self.pools), 1)), wp.float32))
        self._setup_jets(ctx)
        self.ns = SP.net_species(ctx, film, self.net, self.species_names, self.fluid)
        names = tuple(film.config.species) if self.ns.n_film else ()
        conc, node = SP.nozzle_tables(self.nozzles, self.net, names, self.ns)
        self.nconc, self.nnode = ctx.array(conc, wp.vec4d), ctx.array(node, wp.int32)
        self._setup_flooding(ctx, film)
        self._setup_lumps(ctx, film)
        self._setup_holes(ctx, film)
        self._setup_stage(ctx, film)
        self._levels(ctx.model.body_q)                        # read before the first exchange (WP9's lumps)

    def _setup_openings(self, ctx: Any, film: Any) -> None:
        """One input per edge, named after it (rows of one [m, W] array copied into the network each exchange)."""
        m, W = max(len(self.net.edges), 1), ctx.world_count
        start = np.ones((m, W), np.float32)
        for name, value in self.openings.items():
            start[self.net.edge_index(name)] = float(value)
        self._open = ctx.array(start, wp.float32)
        for k, e in enumerate(self.net.edges):
            film.add_input(e.name, self._open[k])
        self.fluid.opening.assign(np.ascontiguousarray(start.T))

    def _bodies(self, ctx: Any, label: str | None) -> np.ndarray:
        if label is None:
            return np.full(ctx.world_count, -1, np.int32)
        return ctx.bodies(label).numpy().astype(np.int32)

    def _setup_nozzles(self, ctx: Any) -> None:
        specs, dirs, wts, fps, owner = [], [], [], [], []
        tanks = [p.tank for p in self.pools]
        for n, noz in enumerate(self.nozzles):
            d, wt, fp = ray_table(noz)
            s = K.NozzleSpec()
            s.edge = HO.drip_edge(self.nozzles, n, self.fluid.discharge.shape[1]) if noz.drip_of else \
                self.net.edge_index(noz.edge)
            s.pattern, s.drip = PATTERNS[noz.pattern], 2 if noz.drip_of == STAGE else int(noz.drip_of is not None)
            s.ray_start, s.ray_count, s.segments = sum(len(x) for x in wts), len(wt), int(noz.segments)
            s.exit_area, s.efficiency, s.reach = noz.exit_area, noz.efficiency, noz.reach
            s.rho, s.nu, s.sigma, s.jump_coeff = noz.density, noz.viscosity, noz.surface_tension, noz.jump_coeff
            s.h_min, s.u_max, s.froude = noz.h_min, noz.u_max, noz.froude
            vessel = noz.rim_of or (noz.drip_of if noz.drip_of != STAGE else None)
            s.own_pool = tanks.index(vessel) if vessel else -1
            specs.append(s)
            dirs.append(d)
            wts.append(wt)
            fps.append(fp)
            owner.append(np.full(len(wt), n))
        self.ray_count = int(sum(len(w) for w in wts))
        self.ray_starts = [int(sum(len(x) for x in wts[:n])) for n in range(len(wts))]
        if not specs:
            specs, dirs, wts, fps, owner = [K.NozzleSpec()], [np.zeros((1, 3))], [np.zeros(1)], [np.zeros(1)], [[0]]
            self.nozzle_spec = wp.array([], dtype=K.NozzleSpec, device=ctx.device)
        else:
            self.nozzle_spec = wp.array(specs, dtype=K.NozzleSpec, device=ctx.device)
        W = ctx.world_count            # per-world exit points and directions (rim pours move): state (snapshots)
        dirs = np.tile(np.concatenate(dirs).astype(np.float32)[None], (W, 1, 1))
        self.ray_dir = self.film.add_state("supply_ray_dir", ctx.array(dirs, wp.vec3))
        pos = np.asarray([n.pos for n in self.nozzles] or [(0.0, 0.0, 0.0)], np.float32)
        self.npos = self.film.add_state("supply_npos", ctx.array(np.tile(pos[None], (W, 1, 1)), wp.vec3))
        self.ray_wt = ctx.array(np.concatenate(wts).astype(np.float32), wp.float32)
        self.ray_fp = ctx.array(np.concatenate(fps).astype(np.float32), wp.float32)
        self.ray_noz = ctx.array(np.concatenate(owner).astype(np.int32), wp.int32)
        nb = [self._bodies(ctx, n.body) for n in self.nozzles] or [np.full(ctx.world_count, -1, np.int32)]
        self.nbody = ctx.array(np.stack(nb, 1), wp.int32)

    def _setup_pools(self, ctx: Any) -> None:
        kappa = self.net.tank_kappa(self.fluid.gravity)
        specs = []
        for p in self.pools:
            nd = self.net.nodes[self.net.node_index(p.tank)]
            s = K.PoolSpec()
            s.tank, s.kappa, s.p_ref, s.area = self.net.node_index(p.tank), kappa[p.tank], nd.p_ref, nd.area
            half = p.half if p.cavity is None else (float(p.cavity[-1][0]),) * 2   # a vessel catches over its mouth
            s.floor, s.half, s.round = wp.vec3(*p.floor), wp.vec2(*half), int(p.cavity is not None)
            s.flood, s.rho, s.nu, s.sigma = int(p.flood), nd.density, p.viscosity, p.surface_tension
            s.h_max, s.flat_cos = p.withdraw_max, math.cos(math.radians(p.puddle_tilt))
            specs.append(s)
            if p.cavity is None and abs(4.0 * p.half[0] * p.half[1] - nd.area) > 0.01 * nd.area:
                log.warning("pool of %r: footprint %.4g m² differs from the tank's area %.4g m² (the level uses the "
                                "tank's area)", p.tank, 4.0 * p.half[0] * p.half[1], nd.area)
        self.pool_spec = wp.array(specs, dtype=K.PoolSpec, device=ctx.device)
        pb = [self._bodies(ctx, p.body) for p in self.pools] or [np.full(ctx.world_count, -1, np.int32)]
        self.pbody = ctx.array(np.stack(pb, 1), wp.int32)

    def _setup_cavities(self, ctx: Any) -> None:
        """Cavity pools' tables and per-world surfaces and capacities (``supply_cavity``)."""
        W, P = ctx.world_count, max(len(self.pools), 1)
        cavs, slices, prof = CV.cavity_tables(self.pools, self.nozzles, self.net, self.ray_starts)
        self.cavities = any(c.on for c in cavs)
        self.cav = wp.array(cavs or [CV.Cavity()], dtype=CV.Cavity, device=ctx.device)
        self.cav_slices, self.cav_prof = ctx.array(slices, wp.vec3d), ctx.array(prof, wp.vec2)
        self._setup_floats(ctx)
        self._setup_carry(ctx)
        self.cav_surf, self.cav_cap = ctx.zeros((W, P), wp.vec4), ctx.zeros((W, P), wp.vec2d)
        self.cav_prev = self.film.add_state("supply_cavity_prev", ctx.zeros((W, P), wp.vec4))   # last flood's

    def _cavity_state(self, body_q: wp.array) -> None:
        wp.launch(CV.cavity_state, dim=self.cav_surf.shape,
                  inputs=[body_q, self.pool_spec, self.cav, self.cav_slices, self.pbody, self.fluid.pressure,
                          wp.vec3(*self.gravity)], outputs=[self.cav_surf, self.cav_cap, self.level],
                  device=self.ctx.device)

    def _drain_cells(self, d: Drain) -> np.ndarray:
        if d.chart < 0 or d.chart >= len(self.charts.starts):
            raise PluginError(f"drain {d.name!r}: chart {d.chart} does not exist ({len(self.charts.starts)} charts)")
        cells = np.arange(self.charts.cells_of(d.chart).start, self.charts.cells_of(d.chart).stop)
        if d.open_edges:
            faces = self.charts.cface.numpy()[cells]
            ftype = self.charts.ftype.numpy()
            opened = np.array([any(f >= 0 and ftype[f] == 2 for f in fs) for fs in faces], bool)
            return cells[opened]
        pos = self.charts.cpos.numpy()[cells].astype(np.float64)
        return cells[np.linalg.norm(pos - np.asarray(d.center), axis=1) <= d.radius]

    def _setup_drains(self, ctx: Any, film: Any) -> None:
        C = self.charts.cell_count
        drain_of = np.full(C, -1, np.int32)
        start, cells, kind, cd, to = [0], [], [], [], []
        for k, d in enumerate(self.drains):
            c = self._drain_cells(d)
            if c.size == 0:
                raise PluginError(f"drain {d.name!r}: no cells of chart {d.chart} (radius {d.radius}, "
                                  f"open_edges {d.open_edges})")
            if np.any(drain_of[c] >= 0):
                raise PluginError(f"drain {d.name!r} overlaps drain {self.drains[int(drain_of[c].max())].name!r}")
            drain_of[c] = k
            cells.append(c)
            start.append(start[-1] + c.size)
            kind.append(2 if d.open_edges else (0 if math.isinf(d.cd_area) else 1))
            cd.append(0.0 if math.isinf(d.cd_area) else d.cd_area)
            to.append(-1 if d.to is None else self.net.node_index(d.to))
        film.drain_of.assign(drain_of)
        D = max(len(self.drains), 1)
        self.dstart = ctx.array(np.asarray(start if self.drains else [0, 0], np.int32), wp.int32)
        self.dcell = ctx.array(np.concatenate(cells).astype(np.int32) if cells else np.zeros(1, np.int32), wp.int32)
        self.dkind = ctx.array(np.asarray(kind or [2], np.int32), wp.int32)
        self.dcd = ctx.array(np.asarray(cd or [0.0], np.float32), wp.float32)
        self.dto = ctx.array(np.asarray(to or [-1], np.int32), wp.int32)
        self.dopen = film.add_input("drain_open", ctx.array(np.ones((ctx.world_count, D), np.float32), wp.float32))
        self.n_drains = len(self.drains)

    def _occluders(self, ctx: Any) -> np.ndarray:
        """Per world, the colliding shapes that can block a ray: all but the nozzles' own bodies' shapes and each
        revolution chart's own shape (``cshape``: the chart is that shape's wetted surface, so a cup's hull does not
        cover its mouth; a plane chart's box still blocks off its face, where the tie rule makes the face the chart).
        A revolution chart on a body with no ``cshape`` (a chart without ``shape=``) exempts its whole body, with a
        warning: give the chart its shape so a handle on that body blocks again."""
        m, W = ctx.model, ctx.world_count
        flags, world, body = m.shape_flags.numpy(), m.shape_world.numpy(), m.shape_body.numpy()
        nb = self.nbody.numpy()[:, [n for n, noz in enumerate(self.nozzles) if not noz.drip_of]]   # drips: per ray
        skip_body = [set(nb[w].tolist()) - {-1} for w in range(W)]
        skip_shape: list[set[int]] = [set() for _ in range(W)]
        cb, cs = self.charts.cbody.numpy(), self.charts.cshape.numpy()
        for k in range(len(self.charts.starts)):
            if not self._is_revolution(k):
                continue
            for w in range(W):
                if cs[w, k] >= 0:
                    skip_shape[w].add(int(cs[w, k]))
                elif cb[w, k] >= 0:
                    skip_body[w].add(int(cb[w, k]))
            if np.any((cs[:, k] < 0) & (cb[:, k] >= 0)):
                log.warning("surface_flow supply: revolution chart %d has no shape: its whole body is exempt from "
                            "occluding jets (set RevolutionChart(shape=...))", k)
            if np.any((cs[:, k] < 0) & (cb[:, k] < 0)):
                log.warning("surface_flow supply: static revolution chart %d has no shape: a collision shape under it "
                            "would block jets from it (set RevolutionChart(shape=...))", k)
        collide = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
        rows = [[s for s in range(len(flags)) if collide[s] and world[s] in (w, -1) and int(body[s]) not in
                 skip_body[w] and s not in skip_shape[w]] for w in range(W)]
        out = np.full((W, max(max((len(r) for r in rows), default=0), 1)), -1, np.int32)
        for w, r in enumerate(rows):
            out[w, :len(r)] = r
        return out

    def _is_revolution(self, k: int) -> bool:
        return int(self.charts.charts.numpy()[k]["kind"]) == int(REVOLUTION)

    # ---- hooks (interface note §4) -----------------------------------------------------------------------------

    def _model_shapes(self) -> list[Any]:
        m = self.ctx.model
        return [m.shape_body, m.shape_transform, m.shape_type, m.shape_scale, m.shape_source_ptr]

    def _chart_args(self) -> list[Any]:
        c = self.charts
        return [c.charts, c.cbody, c.clocal, c.prof]

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        """Before the solver: openings in, rays cast at the pre-solve poses, jet momentum on the bodies."""
        W, dev, f = self.ctx.world_count, self.ctx.device, self.fluid
        self._bind_lumps()                                    # a lump stage may register after the supply's setup
        wp.launch(K.copy_openings, dim=W, inputs=[self._open], outputs=[f.opening], device=dev)
        self._float_forces(state, forces)                     # buoyancy, drag, vessels' water (supply_float)
        if not self.nozzles:
            return
        h = self.hit
        wp.launch(K.cast_rays, dim=(W, self.ray_count),
                  inputs=[state.body_q, *self._model_shapes(), self.occ, *self._chart_args(), self.pool_spec,
                          self.pbody, f.pressure, self.nozzle_spec, self.nbody, self.ray_noz, self.npos, self.ray_dir,
                          self.ray_p0, state.body_qd, self.ctx.model.body_com, self.nozzle_q, wp.vec3(*self.gravity)],
                  outputs=[h["kind"], h["idx"], h["cell"], h["p"], h["w"], h["s"]], device=dev)
        self._lump_kinds()
        if self.sprays:
            self._spray_shadows(state, dt)                   # footprints and hidden shares (supply_shadow)
        wp.launch(K.jet_forces, dim=W,
                  inputs=[state.body_q, self.ctx.model.body_com, self.ctx.model.shape_body, self.charts.cbody,
                          self.charts.clocal, self.nozzle_spec, self.nbody, self.npos, self.ray_dir, self.ray_wt,
                          self.ray_p0, self.nozzle_q,
                          h["kind"], h["idx"], h["cell"], h["p"], h["w"], h["on"], self.charts.cnrm, self.occ],
                  outputs=[forces.body_f], device=dev)
        self._shadow_forces(state, forces)                    # their momentum onto their occluders

    def before_film(self, state: newton.State, dt: float) -> None:
        """At the start of ``update`` (``state``: the pre-solve state): flooding, deposits, the sweep, ``drain_cap``."""
        W, dev, f, film, c = self.ctx.world_count, self.ctx.device, self.fluid, self.film, self.charts
        if self.cavities:
            self._cavity_state(state.body_q)
        if self.flooding:
            self._flood(state, dt)
        self._holes(dt, SP.refresh(self.ns, f))               # holes first: weirs and spills see what they took
        if self.cavities:                                     # rim pours' poses, weirs, spills (supply_cavity)
            wp.launch(CV.cavity_flows, dim=W,
                      inputs=[wp.float64(dt), state.body_q, self.pool_spec, self.cav, self.pbody, f.pressure,
                              wp.vec3(*self.gravity), self.cav_surf, self.cav_cap, SP.refresh(self.ns, f), f.n_species],
                      outputs=[f.opening, self.npos, self.ray_dir, f.inflow, self.ledger], device=dev)
            wp.copy(self.cav_prev, self.cav_surf)            # this exchange's surfaces: the next flood's motion
        if self.nozzles:
            h = self.hit
            wp.launch(D.clear_written, dim=(*self.written.shape, self.rows),
                      inputs=[c.charts, self.written, self.has_pool_tau],
                      outputs=[film.impact_p, film.impact_share, film.jet_cover, film.jet_tau, self.pool_tau],
                      device=dev)
            wp.launch(D.reset_written, dim=self.written.shape, inputs=[], outputs=[self.written], device=dev)
            wp.launch(D.deposit, dim=W,
                      inputs=[wp.float64(dt), self.nozzle_spec, self.ray_wt, self.pool_spec, self.nozzle_q, h["kind"],
                              h["idx"]], outputs=[f.inflow, self.ledger, self.caught_by], device=dev)
            if self.ns.n_film or f.n_species:                 # the species the nozzles' water carries, where it went
                wp.launch(SP.species_sent, dim=W,
                          inputs=[wp.float64(dt), self.nozzle_spec, self.ray_wt, self.pool_spec, self.nozzle_q,
                                  h["kind"], h["idx"], self.nconc, self.nnode, f.n_species, SP.refresh(self.ns, f)],
                          device=dev)
            self._lumps(state, dt, SP.refresh(self.ns, f))
            if self.sprays:
                self._sprays(state, dt)
            self._jets(state, dt)
            self._submerged(state)                            # jets plunging into pools (supply_jetting)
        if self.n_drains:
            dt_film = wp.float64(dt * getattr(film, "stride", 1))     # a strided film applies the cap over k·dt
            wp.launch(K.drain_capacity, dim=W,
                      inputs=[dt_film, self.dstart, self.dcell, self.dkind, self.dcd, self.dopen, film.h, c.area,
                              wp.float64(np.linalg.norm(self.gravity))],
                      outputs=[film.drain_cap], device=dev)
        self._publish_pools(dt)

    def after_film(self, state: newton.State, dt: float) -> None:
        """After the film's substeps: drains into tanks, the network step, the pools' levels."""
        W, dev, f = self.ctx.world_count, self.ctx.device, self.fluid
        self._carry_restore(state)                            # the dry body back (supply_carry)
        self.dt_x = dt                                        # the exchange: what flies in nozzle_q is Q over it
        if self.n_drains:
            wp.launch(K.drain_sums, dim=W, inputs=[wp.float64(dt), self.dstart, self.dcell, self.dto, self.charts.area],
                      outputs=[self.film.drained, f.inflow, self.ledger], device=dev)
        if self.ns.n_film and self.n_drains:              # their species into the drains' tanks, or away
            wp.launch(SP.drain_species, dim=W, inputs=[wp.float64(dt), self.dstart, self.dcell, self.dto,
                                                       self.charts.area, SP.refresh(self.ns, f)], device=dev)
        if self.flooding:                                     # the film the flooded cells gave up, into the tanks
            self._pool_sums(dt)
        self._pool_exchange(dt, SP.refresh(self.ns, f))      # the lumps' adds and takes (supply_lump)
        # One network step per exchange: the next exchange books this step's Q·dt (nozzle_q) at its c' (out_conc),
        # which equals the edge ledgers only so. Substepping here would have to book every substep's parcel.
        f.step(dt)
        self._drip_commit()                                   # this exchange's drips in flight (supply_holes)
        self._stage_commit(dt)                                # and the stages' (supply_stage)
        self._levels(state.body_q)

    def _levels(self, body_q: wp.array) -> None:
        """Each pool's free-surface height ``tank_level`` (cavity pools: their surfaces, normal to gravity)."""
        if self.pools:
            wp.launch(K.tank_levels, dim=self.ctx.world_count,
                      inputs=[body_q, self.pool_spec, self.pbody, self.fluid.pressure], outputs=[self.level],
                      device=self.ctx.device)
        if self.cavities:
            self._cavity_state(body_q)

    # ---- host reads --------------------------------------------------------------------------------------------

    def read(self) -> dict[str, np.ndarray]:
        """Host copies: the ledger columns [W] (m³), each node's pressure and amount, each edge's discharge, the
        pools' levels, and ``species_in_flight [W, K]`` kg (the film species in the nozzles' water in flight:
        ``supply_species.species_flight``)."""
        led = self.ledger.numpy()
        out = {name: led[:, k].copy() for k, name in enumerate(LEDGER)}
        p, v = self.fluid.pressure.numpy(), self.fluid.volume.numpy()
        out.update(pressure=p, amount=self.fluid.amount(p, v), discharge=self.fluid.discharge.numpy(),
                   edge_ledger=self.fluid.ledger.numpy(), inflow_total=self.fluid.inflow_total.numpy(),
                   level=self.level.numpy(), swept_species=self.swept_sp.numpy()[:, :self.n_sp],
                   caught_by=self.caught_by.numpy(), carried_kg=self.carried.numpy(),
                   species_in_flight=self._species_in_flight(),
                   **SP.species_reads(self.ns))
        return out


    def _species_in_flight(self) -> np.ndarray:
        """[W, K] kg of each film species in the nozzles' water in flight (zero before the first exchange)."""
        W, K = self.ctx.world_count, self.ns.n_film
        if not self.nozzles or not K or not getattr(self, "dt_x", 0.0):
            return np.zeros((W, K))
        out = self.ctx.zeros((W, K), wp.float64)
        wp.launch(SP.species_flight, dim=W,
                  inputs=[wp.float64(self.dt_x), self.nozzle_spec, self.nozzle_q, self.nconc, self.nnode,
                          SP.refresh(self.ns, self.fluid)], outputs=[out], device=self.ctx.device)
        return out.numpy()

__all__ = ["Drain", "LEDGER", "Nozzle", "PATTERNS", "Pool", "Supply", "ray_table"]
