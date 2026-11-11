# SPDX-License-Identifier: Apache-2.0
"""Lumped object films (design note §4.9; WP9 builds, WP15 reviews): one well-mixed film per small body.

Tomatoes, a sponge, a pan's handle: their wetting is one number per body, not a chart. Per world and body
(``lumped_kernels.lump_kernel``, every exchange after the film step):
1. the water and species the supply's jets brought (DC5's LUMP hit kind writes ``lump_in``, ``lump_sp_in``,
   ``lump_p``, ``lump_share``, ``lump_tau`` [W, B], cleared by its rule) enter the body's water;
2. on a wet body, its soil erodes by the §5 laws over the jets' share of its area: impact
   ``k_i·share·(p/σ_c - 1)+`` and the sheet's shear ``k_e·share·(τ - σ_c)+``, at the starting strength, then soaks;
   the soil joins the water (the film's soil species);
3. the sanitizer clock (the process stage's ``Sanitizer``): unbroken time at ``c >= c_min``;
4. a sponge (``Pores``) takes its strain from the pressure its pad carries (``p/E = ε + 3ε³``) and holds
   ``phi·A·H·(1 - ε)``; any body holds ``hold·area`` otherwise. A sponge under capacity takes up the film its pad
   touches or presses, on film steps (``uptake``, booked in the film's ``to_lumps``);
5. water over capacity leaves with its species: onto the pad's chart cells when the sponge is pressed (the writer rule:
   ``src_acc``, ``sp.src``, ``cwet``), else shed (``lump_shed``; the drop stage drips it from the body's lowest point
   when configured);
6. in one of DC5's pools, the body fills from it and swaps water with it (``lumped_pool.py``).
Ledger per body [m³]: water = in - shed - squeezed + pooled + uptake, exact in float64.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow import lumped_kernels as LK
from alloy.plugins.surface_flow import lumped_pool as LP
from alloy.plugins.surface_flow.host_api import H_WET
from alloy.plugins.surface_flow.lumped_specs import Lumped, lumped_spec, shape_area
from alloy.plugins.surface_flow.process_kernels import SoilLaw
from alloy.plugins.surface_flow.process_specs import Soil, process_spec

log = logging.getLogger("alloy.plugins.surface_flow")


class LumpedStage:
    """The stage ``SurfaceFlow`` calls (module docstring)."""

    def __init__(self, specs: Sequence[Any]):
        self.specs = tuple(lumped_spec(s) for s in specs)

    # --- setup ---------------------------------------------------------------------------------------------------

    def setup(self, ctx: Any, tables: Any, film: Any) -> None:
        self.ctx, self.tables, self.film = ctx, tables, film
        W, B, K = ctx.world_count, len(self.specs), max(len(film.config.species), 1)
        self.n_sp = len(film.config.species)
        self.bodies = np.stack([ctx.bodies(s.body).numpy().astype(np.int32) for s in self.specs], 1)   # [W, B]
        self.lump_of_shape, area = self._shapes()
        self.spec = self._spec(area)
        self.has_pads = bool((self.spec.tool.numpy() >= 0).any())
        st = self.state = LK.LumpState()
        z = lambda name, *shape: film.add_state(name, ctx.zeros((W, B, *shape), wp.float64))  # noqa: E731
        st.water, st.sp, st.hyd, st.by = z("lump_water"), z("lump_species", K), z("lump_hyd"), z("lump_removed_by", 4)
        st.ledger, st.expo, st.expo_best = z("lump_books", 5), z("lump_expo"), z("lump_expo_best")
        st.sp_in, st.sp_up = z("lump_species_in", K), z("lump_species_uptake", K)
        st.sp_shed, st.sp_sq = z("lump_species_shed", K), z("lump_species_squeezed", K)
        st.soil_sp, st.eps, st.drip = z("lump_soil_to_water"), z("lump_eps"), z("lump_drip")
        st.drip_sp, st.sp_wet = z("lump_drip_species", K), z("lump_species_wet", K)
        st.sp_drop, st.tank, st.sp_tank = z("lump_species_dropped", K), z("lump_drip_tanks"), z("lump_species_tanks", K)
        self.sp_set = z("lump_species_set", K)                      # what the host set (set_species), net
        self.water_set = film.add_state("lump_water_set", ctx.zeros((W, B), wp.float64))   # set_water, net
        st.pool_of = film.add_state("lump_pool_of_body", ctx.full((W, B), -1, wp.int32))
        st.vessel_of = film.add_state("lump_vessel_of_body", ctx.full((W, B), -1, wp.int32))
        soil0 = np.asarray([s.soil.initial if s.soil is not None else 0.0 for s in self.specs], np.float64)
        st.soil = film.add_state("lump_soil", ctx.array(np.tile(soil0, (W, 1)), wp.float64))
        inp = self.inputs = LK.LumpIn()
        inp.vol, inp.sp = z("lump_in"), z("lump_sp_in", K)
        f32 = lambda name: film.add_state(name, ctx.zeros((W, B), wp.float32))  # noqa: E731
        inp.p, inp.share, inp.tau = f32("lump_p"), f32("lump_share"), f32("lump_tau")
        self.lump_area = ctx.array(area, wp.float64)                    # DC5's inputs: the area and the shape map
        self.body_ids, self.pool = ctx.array(self.bodies, wp.int32), None
        self.film_out = self._film_out()
        self.drip = self._drip()
        self._warn_light()
        # the supply's LUMP hook (DC5, supply_lump.LUMP_ARRAYS) finds these on the film by name
        film.lump_of_shape, film.lump_area = self.lump_of_shape, self.lump_area
        film.lump_in, film.lump_sp_in, film.lump_p = inp.vol, inp.sp, inp.p
        film.lump_share, film.lump_tau = inp.share, inp.tau

    def bind_pools(self, supply: Any) -> None:
        """After the supply's setup: its pools' exchange arrays (``supply_lump``), when it has pools."""
        if supply is None or not getattr(supply, "pools", None):
            return
        ctx, f = self.ctx, self.film
        W, B, K = ctx.world_count, len(self.specs), max(self.n_sp, 1)
        if supply.pool_add_sp.shape[2] != K:
            raise PluginError(f"{f.label}: the supply's pools carry {supply.pool_add_sp.shape[2]} species columns, "
                              f"the lumped bodies {K}")
        pl = self.pool = LP.PoolLink()
        pl.pools, pl.pbody, pl.level, pl.avail = supply.pool_spec, supply.pbody, supply.level, supply.pool_avail
        pl.add, pl.add_sp, pl.take = supply.pool_add, supply.pool_add_sp, supply.pool_take
        pl.took, pl.took_sp = supply.pool_took, supply.pool_took_sp
        pl.rate = ctx.array(np.asarray([(s.k_fill, s.k_mix0, s.k_mix1) for s in self.specs], np.float64), wp.vec3d)
        pl.req = f.add_state("lump_pool_req", ctx.zeros((W, B), wp.float64))
        pl.req_pool = f.add_state("lump_pool_of", ctx.full((W, B), -1, wp.int32))
        pl.sp_net = f.add_state("lump_species_pooled", ctx.zeros((W, B, K), wp.float64))
        pl.cavs, pl.cprof = supply.cav, supply.cav_prof

    def _shapes(self) -> tuple[wp.array, np.ndarray]:
        """``lump_of_shape`` [W, S] (model shape id -> body, -1 none) and each body's area (its own, or its shapes');
        ``shape_ids[w][b]`` its shapes."""
        m, W = self.ctx.model, self.ctx.world_count
        S = int(m.shape_count)
        of = np.full((W, S), -1, np.int32)
        sb = m.shape_body.numpy()
        collide = (m.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
        area = np.zeros(len(self.specs))
        self.shape_ids = [[[] for _ in self.specs] for _ in range(W)]
        for b, spec in enumerate(self.specs):
            for w in range(W):
                if spec.shapes:
                    ids = [int(self.ctx.shapes(lab).numpy()[w]) for lab in spec.shapes]
                    if (sb[ids] != self.bodies[w, b]).any():
                        raise PluginError(f"{self.film.label}: lumped {spec.body!r}: shapes {spec.shapes} are not all "
                                          "on its body")
                else:                                                     # its collision shapes
                    ids = [int(s) for s in np.nonzero((sb == self.bodies[w, b]) & collide)[0]]
                if not ids:
                    raise PluginError(f"{self.film.label}: lumped {spec.body!r} has no collision shapes")
                self.shape_ids[w][b] = ids
                if (of[w, ids] >= 0).any():
                    raise PluginError(f"{self.film.label}: a shape of lumped {spec.body!r} belongs to another lump")
                of[w, ids] = b
                if w == 0:
                    area[b] = spec.area if spec.area is not None else sum(shape_area(m, s) for s in ids)
        return self.ctx.array(of, wp.int32), area

    def _spec(self, area: np.ndarray) -> LK.LumpSpec:
        f, ctx = self.film, self.ctx
        sp = LK.LumpSpec()
        sp.area = ctx.array(area, wp.float64)
        sp.hold = ctx.array(np.asarray([s.hold for s in self.specs], np.float64), wp.float64)
        laws = [(s.soil or Soil()).law(f.mat.density) for s in self.specs]
        sp.law = wp.array(laws, dtype=SoilLaw, device=ctx.device)
        sp.soil_on = ctx.array(np.asarray([s.soil is not None for s in self.specs], np.int32), wp.int32)
        pores = [(p.free, p.modulus, p.q_stop / p.height, p.tau_cap) if p is not None else (0.0, 1.0, 0.0, 1.0)
                 for p in (s.pores for s in self.specs)]
        sp.pore = ctx.array(np.asarray(pores, np.float64), wp.vec4d)
        sp.tool = ctx.array(np.asarray([self._tool(s) for s in self.specs], np.int32), wp.int32)
        sp.wet = ctx.array(H_WET * np.asarray(area, np.float64), wp.float64)
        names = tuple(f.config.species)
        soils = [s.soil.species if s.soil is not None else None for s in self.specs]
        sp.k_soil = ctx.array(np.asarray([names.index(n) if n in names else -1 for n in soils], np.int32), wp.int32)
        san = process_spec(f.config.process).sanitizer if f.config.process is not None else None
        sp.k_san = names.index(san.species) if san is not None and san.species in names else -1
        sp.c_min = float(san.c_min) if san is not None else 0.0
        sp.drip_on = 1 if (f.config.drops is not None or f.config.sph) else 0
        sp.built = ctx.array(np.asarray([s.built_water for s in self.specs], np.float64), wp.float64)
        return sp

    def _tool(self, spec: Lumped) -> int:
        """The Applicator a sponge is (by index or body), -1 for none."""
        if spec.pores is None or spec.pores.tool is None:
            return -1
        tools = self.film.config.tools
        key = spec.pores.tool
        if isinstance(key, (int, np.integer)) and 0 <= int(key) < len(tools):
            return int(key)
        names = [t.body for t in tools]
        if key in names:
            key = names.index(key)
        if not (isinstance(key, (int, np.integer)) and 0 <= int(key) < len(tools)):
            raise PluginError(f"{self.film.label}: lumped {spec.body!r}: pores.tool {key!r} is not an applicator "
                              f"(by index or body: {names})")
        if getattr(tools[int(key)], "store", None) is not None:
            raise PluginError(f"{self.film.label}: lumped {spec.body!r}: pores.tool {key!r} has a store; a sponge's "
                              "pad is a porous Applicator (its pores are the store)")
        if not getattr(tools[int(key)], "porous", False):
            raise PluginError(f"{self.film.label}: lumped {spec.body!r}: pores.tool {key!r} is not porous; a sponge's "
                              "pad absorbs the film it touches: Applicator(..., porous=True)")
        return int(key)

    def _drip(self) -> Any:
        """The drop stage's view of the bodies' drips (``lumped_drip.LumpDrip``: their shapes, their pending water)."""
        from alloy.plugins.surface_flow.lumped_drip import build_drip  # noqa: PLC0415
        return build_drip(self.ctx, self.bodies, self.shape_ids, self.state, self.n_sp)

    def _film_out(self) -> LK.FilmOut:
        f, t, ctx = self.film, self.tables, self.ctx
        fo = LK.FilmOut()
        fo.charts, fo.area = t.charts, t.area
        if f.tools is not None:
            fo.tool_chart, fo.wins = f.tools.set.chart, f.tools.wins
        else:
            fo.tool_chart, fo.wins = ctx.zeros(1, wp.int32), ctx.zeros((1, 1), wp.vec4i)
        fo.src_acc, fo.src_sp, fo.cwet = f.src_acc, f.sp.src, f.cwet
        fo.h, fo.base, fo.sp_mass = f.h, f.base, f.sp.mass
        W, K = ctx.world_count, max(len(f.config.species), 1)
        fo.taken = f.add_state("to_lumps", ctx.zeros(W, wp.float64))
        fo.taken_sp = f.add_state("species_to_lumps", ctx.zeros((W, K), wp.float64))
        fo.absorb, fo.absorb_sp, fo.room = f.tf.absorb, f.tf.absorb_sp, f.tool_room     # WP15's porous pads
        return fo

    # --- stage call ----------------------------------------------------------------------------------------------

    def before_film(self, state: Any, dt: float) -> None:
        """Every exchange, before the tool stage: each sponge's room for its porous pad (``tool_room``)."""
        if self.has_pads:
            wp.launch(LK.room_kernel, dim=self.ctx.world_count, inputs=[self.spec, self.state, self.film_out],
                      device=self.ctx.device)

    def after_film(self, state: Any, dt: float) -> None:
        """Every exchange, after the film step and the process stage (module docstring)."""
        W, dev = self.ctx.world_count, self.ctx.device
        if self.pool is not None:                     # last exchange's takes, confirmed by the supply
            wp.launch(LP.pool_book, dim=W, inputs=[self.state, self.pool, self.drip, state.body_q,
                                                   self.ctx.model.body_com, self.ctx.model.gravity, self.body_ids,
                                                   self.n_sp], device=dev)
        if self.has_pads:                             # what the porous pads absorbed this exchange
            wp.launch(LK.absorb_kernel, dim=W, inputs=[self.spec, self.state, self.film_out, self.n_sp], device=dev)
        wp.launch(LK.lump_kernel, dim=W,
                  inputs=[self.spec, self.state, self.inputs, self.film.tf, self.film_out, self.n_sp, float(dt),
                          self.film.tick, self.film.stride],
                  device=dev)
        if self.pool is not None:                     # this exchange's adds and asks (the supply's after_film)
            wp.launch(LP.pool_ask, dim=W,
                      inputs=[self.spec, self.state, self.pool, state.body_q, state.body_qd, self.ctx.model.body_com,
                              self.body_ids, self.n_sp, float(dt)], device=dev)

    # --- host ----------------------------------------------------------------------------------------------------

    def apply(self, state: Any, forces: Any, dt: float) -> None:
        """Each body carries its water's weight beyond its built mass (``Lumped.built_water``), at its COM."""
        del dt
        wp.launch(LK.weight_kernel, dim=self.ctx.world_count,
                  inputs=[self.spec, self.state, self.body_ids, self.ctx.model.gravity, float(self.film.mat.density)],
                  outputs=[forces.body_f], device=self.ctx.device)

    def set_water(self, values: Any, body: int | None = None) -> None:
        """A body's water [m³] (scalar, [W] per world, or [W, B]; with ``body`` a scalar or [W]); its species stay.
        The water ledger counts it as set (``lump_water_set``)."""
        new = self._values(values, body, "set_water [m³]")
        a, done = self.state.water.numpy().copy(), self.water_set.numpy().copy()
        sl = (slice(None), slice(None) if body is None else body)
        done[sl] += new - a[sl]
        a[sl] = new
        self.state.water.assign(a)
        self.water_set.assign(done)

    def _warn_light(self) -> None:
        """Warn once when a body's full water outweighs its built mass beyond what it was built with (WP15: a 2 mL
        film on a 2 g body falls at 19 m/s²; a dry-built 79 mL sponge of a few grams at 15-25 g)."""
        mass = self.ctx.model.body_mass.numpy()
        cap = [s.pores.free if s.pores is not None else s.hold * a for s, a in zip(self.specs, self.spec.area.numpy())]
        rho = float(self.film.mat.density)
        light = [f"{s.body!r} ({rho * (c - s.built_water) * 1e3:.3g} g of water over {float(mass[k]) * 1e3:.3g} g)"
                 for s, c, k in zip(self.specs, cap, self.bodies[0]) if rho * (c - s.built_water) > float(mass[k])]
        if light:
            log.warning("%s: lumped bodies whose full water outweighs their built mass beyond built_water: %s; it "
                        "weighs on them (Lumped.built_water) but its inertia stays the built one: build them at "
                        "their typical wet mass", self.film.label, ", ".join(light))

    def set_species(self, values: Any, name: str, body: int | None = None) -> None:
        """A body's mass of species ``name`` [kg] (as ``set_water``); its water stays. The species books count it as
        set (``lump_species_set``), so they start from it (JOB3's sponge of wash)."""
        names = tuple(self.film.config.species)
        if name not in names:
            raise PluginError(f"{self.film.label}: no species {name!r}; it carries {names} (Config.species)")
        new = self._values(values, body, f"set_species({name!r}) [kg]")
        k = names.index(name)
        sp, done = self.state.sp.numpy().copy(), self.sp_set.numpy().copy()
        sl = (slice(None), slice(None) if body is None else body, k)
        done[sl] += new - sp[sl]
        sp[sl] = new
        self.state.sp.assign(sp)
        self.sp_set.assign(done)

    def set_soil(self, values: Any, body: int | None = None) -> None:
        """A body's soil [kg/m²]; its hydration and removal restart."""
        self._set(self.state.soil, values, body, "set_soil [kg/m²]")
        for arr in (self.state.hyd, self.state.by):
            a = arr.numpy().copy()
            if body is None:
                a[:] = 0.0
            else:
                a[:, body] = 0.0
            arr.assign(a)

    def _set(self, arr: wp.array, values: Any, body: int | None, what: str = "value") -> None:
        a = arr.numpy().copy()
        if body is None:
            a[:] = self._values(values, body, what)
        else:
            a[:, body] = self._values(values, body, what)
        arr.assign(a)

    def _values(self, values: Any, body: int | None, what: str) -> np.ndarray:
        """Host values, checked: finite numbers >= 0; with ``body`` (an index 0..B-1) a scalar or [W], else a scalar,
        [W] (per world, every body) or [W, B]. Returns [W] or [W, B] float64."""
        W, B = self.bodies.shape
        if body is not None and (isinstance(body, bool) or not isinstance(body, (int, np.integer))
                                 or not 0 <= body < B):
            raise PluginError(f"{self.film.label}: {what}: body must be a lumped body's index 0..{B - 1}, got {body!r}")
        v = np.asarray(values)
        if v.dtype.kind not in "iuf" or not np.all(np.isfinite(v)) or np.any(v < 0):
            raise PluginError(f"{self.film.label}: {what} needs finite numbers >= 0, got {values!r}")
        v = v.astype(np.float64)
        if v.ndim == 0 or v.shape == (W,):
            out = np.broadcast_to(v.reshape(-1, 1) if v.ndim else v, (W, B))
        elif body is None and v.shape == (W, B):
            out = v
        else:
            want = "a scalar or [W]" if body is not None else "a scalar, [W] or [W, B]"
            raise PluginError(f"{self.film.label}: {what} takes {want} (W {W}, B {B}), got shape {v.shape}")
        return (out[:, 0] if body is not None else out).copy()

    def read(self) -> dict[str, np.ndarray]:
        st, area = self.state, self.lump_area.numpy()
        led = st.ledger.numpy()
        k = self.n_sp
        out = {"lump_water": st.water.numpy().copy(), "lump_soil": st.soil.numpy().copy(),
               "lump_soil_mass": st.soil.numpy() * area[None, :], "lump_hyd": st.hyd.numpy().copy(),
               "lump_removed_by_mass": st.by.numpy() * area[None, :, None], "lump_eps": st.eps.numpy().copy(),
               "lump_expo": st.expo.numpy().copy(), "lump_expo_best": st.expo_best.numpy().copy(),
               "lump_in_volume": led[..., LK.IN], "lump_shed": led[..., LK.SHED],
               "lump_squeezed": led[..., LK.SQUEEZED], "lump_pooled": led[..., LK.POOLED],
               "lump_uptake": led[..., LK.UPTAKE], "to_lumps_volume": self.film_out.taken.numpy().copy(),
               "lump_drip_pending": st.drip.numpy().copy(), "lump_drip_tanks": st.tank.numpy().copy()}
        out["lump_water_set"] = self.water_set.numpy().copy()
        out["lump_ledger"] = out["lump_water"] - (out["lump_water_set"] + led[..., LK.IN] - led[..., LK.SHED]
                                                  - led[..., LK.SQUEEZED] + led[..., LK.POOLED] + led[..., LK.UPTAKE])
        if k:
            out.update(self._species_read(k))
        if self.pool is not None:
            out.update(self._pool_read(k))
            if k:
                out["lump_species_ledger"] = out["lump_species_ledger"] - out["lump_species_pooled"]
        return out

    def _species_read(self, k: int) -> dict[str, np.ndarray]:
        """Species books [W, B, K] kg and the residual of: species = set + in + uptake + soil to water - shed -
        squeezed."""
        st = self.state
        out = {"lump_species": st.sp.numpy()[..., :k].copy()}
        # shed species go to the board under a resting body (wet), wait with the drips, leave with drops (dropped:
        # drops.read books where they went) or join a pool's tank with drips that start in its water (tanks)
        for key, arr in (("in", st.sp_in), ("uptake", st.sp_up), ("shed", st.sp_shed), ("squeezed", st.sp_sq),
                         ("drip_pending", st.drip_sp), ("wet", st.sp_wet), ("dropped", st.sp_drop),
                         ("tanks", st.sp_tank)):
            out[f"lump_species_{key}"] = arr.numpy()[..., :k].copy()
        soil = np.zeros_like(out["lump_species"])
        ks = self.spec.k_soil.numpy()
        for b in np.nonzero((ks >= 0) & (ks < k))[0]:
            soil[:, b, ks[b]] = st.soil_sp.numpy()[:, b]
        out["lump_species_soil"] = soil
        out["lump_species_set"] = self.sp_set.numpy()[..., :k].copy()
        out["lump_species_ledger"] = out["lump_species"] - (out["lump_species_set"] + out["lump_species_in"]
                                                            + out["lump_species_uptake"] + soil
                                                            - out["lump_species_shed"] - out["lump_species_squeezed"])
        return out

    def _pool_read(self, k: int) -> dict[str, np.ndarray]:
        """The species pooled in (net) and the takes the supply confirmed that the bodies book next exchange."""
        pl = self.pool
        req, of, took = pl.req.numpy(), pl.req_pool.numpy(), pl.took.numpy()
        pend, pend_sp = np.zeros(req.shape), np.zeros((*req.shape, max(k, 1)))
        took_sp = pl.took_sp.numpy()
        for w, b in zip(*np.nonzero(of >= 0)):
            p = of[w, b]
            share = req[w, b] / req[w, of[w] == p].sum()
            pend[w, b], pend_sp[w, b] = took[w, p] * share, took_sp[w, p] * share
        out = {"lump_pool_pending": pend}
        if k:
            out["lump_species_pooled"] = pl.sp_net.numpy()[..., :k].copy()
            out["lump_species_pool_pending"] = pend_sp[..., :k]
        return out


__all__ = ["LumpedStage"]
