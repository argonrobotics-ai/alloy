# SPDX-License-Identifier: Apache-2.0
"""Jets on lumped bodies (WP9's lumped object films: tomatoes, a sponge, a pan's handle; [internal note]
notes/wp9-sf-lumped-design.md). A body without a chart carries its film as one lump, so a jet that hits one of its
shapes neither lands on a chart nor is caught: it is booked ``LUMP`` and handed to the lump.

The film (WP9's stage) registers the lump arrays; the supply finds them by name and leaves them alone when absent:
- ``lump_of_shape [W, S]`` int32: the lump of each model shape, -1 for none; ``lump_area [B]``: each lump's area [m²];
- written here each exchange (zeroed at the next ``before_film``, the per-exchange clear rule), all ``[W, B]``:
  ``lump_in`` m³ that arrived, ``lump_sp_in [W, B, K]`` kg of each film species with it, ``lump_p`` the largest
  stagnation pressure ½·ρ·w_n², ``lump_share`` the jets' footprints over the lump's area (at most 1), ``lump_tau`` the
  largest jet-sheet shear (Blasius, as ``jet_tau``, the mean over a disc of the lump's equal-area sphere radius).

Jets only (v1): a fan, cone or spray ray on a lump is caught. The normal at the hit is taken from the body's centre
of mass (exact for a sphere, an estimate for a handle). The sheet's shear is averaged over a disc of the radius of
the sphere of the lump's area, √(A/4π) (a disc of the lump's own area, √(A/π), would give 1/√2 of it). Volumes go to
the supply ledger's ``lumped`` column. The arrays bind at the supply's setup or, when a stage registers them later,
at its first ``apply``.

The pool exchange (a sponge in a bowl swaps water with it): the supply publishes ``pool_avail [W, P]`` m³ at the end
of ``before_film``, each pool tank's volume after the supply's own transfers this exchange; a lump stage writes
``pool_add [W, P]`` m³ (>= 0) with ``pool_add_sp [W, P, K]`` kg of each film species, and ``pool_take [W, P]`` m³
(>= 0). At its next ``after_film``, before the network step, the supply adds, then takes at the tank's concentration
after the add and after everything booked into or out of the tank this exchange (at most what the tank then has, the
rest ``refused``), zeroes the three, and publishes the take in ``pool_took [W, P]`` m³ and ``pool_took_sp [W, P, K]``
kg for the stage to book. Ledger ``lump_pool`` (net in; species: the network's only, an add of a species no tank
carries goes ``away``). A negative add or take is refused (booked ``refused``), a negative species add ignored, and
requests to a scene without pools are refused and cleared. Several lumps in one pool share it through the stage: the
supply sees one request per pool, so the stage splits ``pool_took`` among its lumps (pro rata to their asks). Species
may come with no water (a sponge's soap diffusing out).

Species alone may leave a tank (SCR-WP8-10, WP9: a wall under a pool's water adsorbs its soap): ``pool_take_sp
[W, P, K]`` kg (>= 0), taken after the adds and before the lumps' takes, at most what the tank holds then (the rest,
a negative request, a species the network does not carry or a pool that is not there: species ``refused``); booked
``lump_pool`` (out), zeroed, and confirmed in ``pool_taken_sp [W, P, K]``. The lumps then take at the concentration
after it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import com_world
from alloy.plugins.surface_flow.supply_kernels import (JET, LUMP, LUMP_POOL, REFUSED, SHAPE, NozzleSpec, PoolSpec,
                                                       weight_sum)
from alloy.plugins.surface_flow.supply_species import SP_AWAY, SP_LUMP_POOL, SP_REFUSED, NetSpecies, nozzle_conc

PI = wp.constant(3.14159265358979)


@wp.struct
class Lumps:
    of_shape: wp.array2d[wp.int32]
    area: wp.array[wp.float64]
    inn: wp.array2d[wp.float64]
    sp_in: wp.array3d[wp.float64]
    p: wp.array2d[wp.float32]
    share: wp.array2d[wp.float32]
    tau: wp.array2d[wp.float32]


@wp.kernel
def lump_kinds(nozzles: wp.array[NozzleSpec], ray_noz: wp.array[wp.int32], occ: wp.array2d[wp.int32], lu: Lumps,
               hit_kind: wp.array2d[wp.int32], hit_idx: wp.array2d[wp.int32]):
    """A jet ray whose shape belongs to a lump is a ``LUMP`` hit (its index stays the occluder slot)."""
    w, r = wp.tid()
    if hit_kind[w, r] == SHAPE and nozzles[ray_noz[r]].pattern == JET:
        if lu.of_shape[w, occ[w, hit_idx[w, r]]] >= 0:
            hit_kind[w, r] = LUMP


@wp.kernel
def lump_clear(lu: Lumps):
    """The per-exchange lump fields, zeroed before this exchange's jets add to them."""
    w, b = wp.tid()
    lu.inn[w, b] = wp.float64(0.0)
    lu.p[w, b] = 0.0
    lu.share[w, b] = 0.0
    lu.tau[w, b] = 0.0
    for k in range(lu.sp_in.shape[2]):
        lu.sp_in[w, b, k] = wp.float64(0.0)


@wp.func
def sheet_shear(rho: float, nu: float, speed: float, d: float) -> float:
    """Blasius shear of a radial sheet at speed ``speed`` averaged over a disc of radius d (τ ∝ r^-1/2: 4/3 of τ(d))."""
    re = wp.max(speed * d / nu, 1.0e-6)
    return 4.0 / 3.0 * 0.332 * rho * speed * speed / wp.sqrt(re)


@wp.kernel
def lump_deposit(dt: wp.float64, nozzles: wp.array[NozzleSpec], ray_wt: wp.array[wp.float32],
                 discharge: wp.array2d[wp.float64], hit_kind: wp.array2d[wp.int32], hit_idx: wp.array2d[wp.int32],
                 hit_p: wp.array2d[wp.vec3], hit_w: wp.array2d[wp.vec3], occ: wp.array2d[wp.int32],
                 body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], shape_body: wp.array[wp.int32],
                 nconc: wp.array[wp.vec4d], nnode: wp.array[wp.int32], ns: NetSpecies, lu: Lumps):
    """Per world, the jets' rays on lumps (in ray order): their water and species, impact and shear (module doc)."""
    w = wp.tid()
    for n in range(nozzles.shape[0]):
        noz = nozzles[n]
        q_all = -discharge[w, noz.edge]
        if noz.pattern != JET or q_all <= wp.float64(0.0):
            continue
        norm = weight_sum(noz, ray_wt)
        for r in range(noz.ray_start, noz.ray_start + noz.ray_count):
            if hit_kind[w, r] != LUMP:
                continue
            q = q_all * wp.float64(noz.efficiency) * wp.float64(ray_wt[r]) / norm
            s = occ[w, hit_idx[w, r]]
            b = lu.of_shape[w, s]
            lu.inn[w, b] = lu.inn[w, b] + q * dt
            for k in range(ns.n_film):
                lu.sp_in[w, b, k] = lu.sp_in[w, b, k] + q * dt * nozzle_conc(ns, w, nconc[n], nnode[n], k)
            v = hit_w[w, r]
            speed = wp.max(wp.length(v), 1.0e-6)
            nrm = wp.normalize(hit_p[w, r] - com_world(body_q, body_com, shape_body[s]) + wp.vec3(0.0, 0.0, 1.0e-9))
            wn = wp.max(-wp.dot(v, nrm), 0.0)
            area = float(lu.area[b])
            if wn > 1.0e-3 * speed:
                lu.p[w, b] = wp.max(lu.p[w, b], 0.5 * noz.rho * wn * wn)
                lu.share[w, b] = wp.min(lu.share[w, b] + float(q) / wn / area, 1.0)    # footprint π·a²·w/w_n = q/w_n
            a = wp.sqrt(float(q) / (PI * speed))
            lu.tau[w, b] = wp.max(lu.tau[w, b], sheet_shear(noz.rho, noz.nu, speed,
                                                            wp.max(a, wp.sqrt(area / (4.0 * PI)))))


@wp.func
def tank_have(pool: PoolSpec, w: int, pressure: wp.array2d[wp.float64], inflow: wp.array2d[wp.float64],
              dt: wp.float64) -> wp.float64:
    """A pool tank's volume after the transfers booked into or out of it so far this exchange [m³]."""
    v = pool.kappa * wp.max(pressure[w, pool.tank] - pool.p_ref, wp.float64(0.0))
    return wp.max(v + inflow[w, pool.tank] * dt, wp.float64(0.0))


@wp.kernel
def pool_available(dt: wp.float64, pools: wp.array[PoolSpec], pressure: wp.array2d[wp.float64],
                   inflow: wp.array2d[wp.float64], avail: wp.array2d[wp.float64]):
    w, p = wp.tid()
    avail[w, p] = tank_have(pools[p], w, pressure, inflow, dt)


@wp.kernel
def pool_exchange(dt: wp.float64, pools: wp.array[PoolSpec], pressure: wp.array2d[wp.float64], ns: NetSpecies,
                  add: wp.array2d[wp.float64], add_sp: wp.array3d[wp.float64], take: wp.array2d[wp.float64],
                  take_sp: wp.array3d[wp.float64], inflow: wp.array2d[wp.float64], took: wp.array2d[wp.float64],
                  took_sp: wp.array3d[wp.float64], taken_sp: wp.array3d[wp.float64], ledger: wp.array2d[wp.float64]):
    """Per world, in pool order: the lumps' adds into each pool's tank, the species takes (walls), then the lumps'
    takes out of it at the tank's concentration after both (module docstring)."""
    w = wp.tid()
    zero = wp.float64(0.0)
    for p in range(add.shape[1]):
        a0 = add[w, p]
        t0 = take[w, p]
        add[w, p] = zero
        take[w, p] = zero
        if p >= pools.shape[0]:                                   # no pool there: refused
            ledger[w, REFUSED] = ledger[w, REFUSED] + wp.abs(a0) + wp.abs(t0)
            for k in range(add_sp.shape[2]):
                add_sp[w, p, k] = zero
                if k < ns.n_film:
                    ns.led[w, SP_REFUSED, k] = ns.led[w, SP_REFUSED, k] + wp.abs(take_sp[w, p, k])
                take_sp[w, p, k] = zero
                taken_sp[w, p, k] = zero
            continue
        pool = pools[p]
        a = wp.max(a0, zero)
        t = wp.max(t0, zero)
        have = tank_have(pool, w, pressure, inflow, dt)          # after what this exchange booked so far
        g = wp.min(t, have + a)                                   # at most what the tank has after the add
        ledger[w, REFUSED] = ledger[w, REFUSED] + (t - g) + (a - a0) + (t - t0)      # negatives refused
        took[w, p] = g
        inflow[w, pool.tank] = inflow[w, pool.tank] + (a - g) / dt
        ledger[w, LUMP_POOL] = ledger[w, LUMP_POOL] + (a - g)
        vol = have + a
        for k in range(ns.n_film):
            ma = wp.max(add_sp[w, p, k], zero)                    # a negative species add is ignored
            ts = take_sp[w, p, k]
            add_sp[w, p, k] = zero
            take_sp[w, p, k] = zero
            kn = ns.map[k]
            mt = zero
            mw = zero
            if kn >= 0:                                           # the walls' take: at most what the tank then holds
                m = wp.max(ns.species[w, pool.tank, kn] + ns.inflow[w, pool.tank, kn] * dt, zero)
                mw = wp.min(wp.max(ts, zero), m + ma)
                if vol > zero:                                    # the lumps' take at the concentration after both
                    mt = g * (m + ma - mw) / vol
            ns.led[w, SP_REFUSED, k] = ns.led[w, SP_REFUSED, k] + wp.abs(ts - mw)
            took_sp[w, p, k] = mt
            taken_sp[w, p, k] = mw
            if kn >= 0:
                ns.inflow[w, pool.tank, kn] = ns.inflow[w, pool.tank, kn] + (ma - mw - mt) / dt
                ns.led[w, SP_LUMP_POOL, k] = ns.led[w, SP_LUMP_POOL, k] + (ma - mw - mt)
            else:
                ns.led[w, SP_AWAY, k] = ns.led[w, SP_AWAY, k] + ma      # no tank holds it


# ---- host side -----------------------------------------------------------------------------------------------------

LUMP_ARRAYS = ("lump_of_shape", "lump_area", "lump_in", "lump_sp_in", "lump_p", "lump_share", "lump_tau")


def lumps_dummy(ctx: Any) -> Lumps:
    """A ``Lumps`` of one-element arrays for a film without lumps (the kernels' arguments)."""
    lu, W = Lumps(), ctx.world_count
    lu.of_shape = ctx.full((W, 1), -1, wp.int32)
    lu.area = ctx.array(np.ones(1), wp.float64)
    lu.inn, lu.sp_in = ctx.zeros((W, 1), wp.float64), ctx.zeros((W, 1, 1), wp.float64)
    lu.p, lu.share, lu.tau = (ctx.zeros((W, 1), wp.float32) for _ in range(3))
    return lu


def lumps_of(ctx: Any, film: Any, n_film: int) -> Lumps | None:
    """The film's lump arrays (``LUMP_ARRAYS``) as a ``Lumps``, None while it has none; their shapes and types are
    checked (host-only: no device reads, so a first bind inside a captured graph is safe)."""
    have = {name: getattr(film, name, None) for name in LUMP_ARRAYS}
    if all(v is None for v in have.values()):
        return None
    missing = sorted(n for n, v in have.items() if v is None)
    if missing:
        raise PluginError(f"lumped films: the film registers all of {LUMP_ARRAYS}; missing {missing}")
    W, S, B = ctx.world_count, int(ctx.model.shape_count), int(have["lump_area"].shape[0])
    want = {"lump_of_shape": ((W, S), wp.int32), "lump_area": ((B,), wp.float64), "lump_in": ((W, B), wp.float64),
            "lump_sp_in": ((W, B, max(n_film, 1)), wp.float64), "lump_p": ((W, B), wp.float32),
            "lump_share": ((W, B), wp.float32), "lump_tau": ((W, B), wp.float32)}
    for name, (shape, dtype) in want.items():
        a = have[name]
        if tuple(a.shape) != shape or a.dtype != dtype:
            raise PluginError(f"lumped films: {name} must be {shape} {dtype.__name__} (W worlds, S model shapes, B "
                              f"lumps, K film species); got {tuple(a.shape)} {a.dtype.__name__}")
    lu = Lumps()
    lu.of_shape, lu.area, lu.inn, lu.sp_in = have["lump_of_shape"], have["lump_area"], have["lump_in"], \
        have["lump_sp_in"]
    lu.p, lu.share, lu.tau = have["lump_p"], have["lump_share"], have["lump_tau"]
    return lu


class Lumping:
    """``Supply``'s lump hooks (a mixin: it uses the supply's nozzle tables, hits and species)."""

    def _setup_lumps(self, ctx: Any, film: Any) -> None:
        self.lu, self.n_lumps = lumps_dummy(ctx), 0
        self._bind_lumps()
        W, P, K = ctx.world_count, max(len(self.pools), 1), max(self.ns.n_film, 1)
        self.pool_avail, self.pool_add, self.pool_take, self.pool_took = (
            film.add_state(n, ctx.zeros((W, P), wp.float64)) for n in ("pool_avail", "pool_add", "pool_take",
                                                                        "pool_took"))
        self.pool_add_sp, self.pool_took_sp, self.pool_take_sp, self.pool_taken_sp = (
            film.add_state(n, ctx.zeros((W, P, K), wp.float64))
            for n in ("pool_add_sp", "pool_took_sp", "pool_take_sp", "pool_taken_sp"))

    def _bind_lumps(self) -> None:
        """Bind the film's lump arrays once they exist: at setup, or at the first ``apply`` after a stage that sets up
        after the supply registers them (WP9's lumped stage); shapes and types are checked then."""
        if self.n_lumps:
            return
        lu = lumps_of(self.ctx, self.film, self.ns.n_film)
        if lu is not None:
            self.lu, self.n_lumps = lu, int(lu.area.shape[0])

    def _publish_pools(self, dt: float) -> None:
        """End of ``before_film``: each pool tank's volume after the supply's transfers (``pool_avail``)."""
        if self.pools:
            wp.launch(pool_available, dim=self.pool_avail.shape,
                      inputs=[wp.float64(dt), self.pool_spec, self.fluid.pressure, self.fluid.inflow],
                      outputs=[self.pool_avail], device=self.ctx.device)

    def _pool_exchange(self, dt: float, ns: NetSpecies) -> None:
        """In ``after_film`` before the network step: the lumps' adds and takes (``pool_exchange``; without pools it
        refuses and clears what was asked)."""
        f = self.fluid
        wp.launch(pool_exchange, dim=self.ctx.world_count,
                  inputs=[wp.float64(dt), self.pool_spec, f.pressure, ns],
                  outputs=[self.pool_add, self.pool_add_sp, self.pool_take, self.pool_take_sp, f.inflow,
                           self.pool_took, self.pool_took_sp, self.pool_taken_sp, self.ledger],
                  device=self.ctx.device)

    def _lump_kinds(self) -> None:
        """After the cast: jet rays on a lump's shapes become ``LUMP`` hits."""
        if self.n_lumps:
            h = self.hit
            wp.launch(lump_kinds, dim=(self.ctx.world_count, self.ray_count),
                      inputs=[self.nozzle_spec, self.ray_noz, self.occ, self.lu], outputs=[h["kind"], h["idx"]],
                      device=self.ctx.device)

    def _lumps(self, state: Any, dt: float, ns: NetSpecies) -> None:
        """This exchange's lump fields: cleared, then the jets on lumps (``lump_deposit``)."""
        if not self.n_lumps:
            return
        W, dev, h, m = self.ctx.world_count, self.ctx.device, self.hit, self.ctx.model
        wp.launch(lump_clear, dim=(W, self.n_lumps), inputs=[self.lu], device=dev)
        wp.launch(lump_deposit, dim=W,
                  inputs=[wp.float64(dt), self.nozzle_spec, self.ray_wt, self.nozzle_q, h["kind"], h["idx"],
                          h["p"], h["w"], self.occ, state.body_q, m.body_com, m.shape_body, self.nconc, self.nnode, ns,
                          self.lu], device=dev)


__all__ = ["LUMP_ARRAYS", "Lumping", "Lumps", "lump_clear", "lump_deposit", "lump_kinds", "lumps_dummy", "lumps_of",
           "pool_available", "pool_exchange", "sheet_shear", "tank_have"]
