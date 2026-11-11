# SPDX-License-Identifier: Apache-2.0
"""The drop stage of ``SurfaceFlow`` (design §4.4, §4.5 and §4.7; WP9): film leaves as drops and comes back.

``SurfaceFlow`` builds ``Drops(config.drops, sph=config.sph)`` and calls it at three points of an exchange step:
- ``apply`` (before the solver): the film's forces on bodies (``body_forces.py``: weight, the absorbed impulse,
  debris drag).
- ``absorb`` (start of the film update, after the supply): arriving liquid joins the film (``src_acc``, ``cwet``):
  - with ``sph``: thin SPH liquid that reached a chart (fewer than ``absorb_column`` liquid particles stacked above
    it, so a resting pool stays SPH; within ``absorb_reach`` particle radii of the film top; not moving away), through
    the SPH port;
  - without: the ballistic drops that landed this step.
- ``shed`` (after the film's substeps):
  - the open-edge outflux and the drips of hanging films over their retention depth become pending volume per cell;
  - whole drops leave in request slots: SPH particles from the port's held reservoir (``sph``), or ballistic drops.
  - A drop's volume is the SPH particle volume, or with no SPH a sphere of ``drop_radius`` (default: the capillary
    length).

Retention (design §4.4, an *estimate* to fit to a drip test): ``h_ret = 2·l_c·sin(θ_r/2)``, with
``l_c = sqrt(σ/(ρ·g_n))``. ``θ_r`` is the chart's contact angle (the material's when the chart sets none, else
``retention_angle``, 90°: the puddle height) until the film has hysteresis. The excess drips over ``tau_drip``.

Ledger (law D-1). Per world, besides the film's own read() ledger:
- ``to_sph`` / ``from_sph``: volume handed to and taken from SPH;
- ``to_pool``, ``landed``, ``floored``, ``to_tanks``: ballistic drops made, landed on a chart, fallen below
  ``floor_z``, and fallen into a pool's water (its tank through DC5's ``pool_add``, booked there as ``lump_pool``);
- ``pending`` and ``in_flight``: volume waiting in cells, and drops in the air;
- with lumped bodies (``lumped_drip.py``): ``lump_wet`` (water a resting body gave the cell under it),
  ``lump_pending`` (its shed water still waiting), ``lump_drops`` (the drops made from it, in ``to_pool`` or
  ``to_sph``) and ``lump_tanks`` (its drips that started in a pool's water, into its tank: DC5's ``pool_add``,
  booked there as ``lump_pool``), so the bodies' shed = lump_wet + lump_pending + lump_drops + lump_tanks.
- species [W, K] kg: the film's species leave with its shed film and drips (the drip takes its share, so the film
  keeps its concentration; both in the film's ``species_out``) into the cells' pending species (``species_shed``),
  and each drop takes its share. A ballistic drop carries them (in flight) to the cell it lands on (``landed``, into
  the film's ``species_supplied``), below ``floor_z`` (``floored``) or into a pool's tank (``to_tanks``); SPH
  particles carry none, so a drop's species going to SPH are lost (``species_to_sph``; setup warns). With the
  lumps' dropped species: species_shed + lump_species_dropped = species_pending + species_to_sph +
  species_in_flight + species_landed + species_floored + species_to_tanks.
A full pool keeps drops waiting (none lost); ``read`` warns once and says the pool it needed (``pool_full``).

Liquid that arrives counts in the film's ``supplied``; liquid that leaves counts in its ``spilled`` or ``dripped``.
The ledger is exact in whole drops.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow import drops_flight as DF
from alloy.plugins.surface_flow import drops_kernels as DK
from alloy.plugins.surface_flow import lumped_drip as LD
from alloy.plugins.surface_flow.drops_pools import drop_pools
from alloy.plugins.surface_flow.body_forces import BodyForces, Debris

log = logging.getLogger("alloy.plugins.surface_flow")

OPEN_FACE = 2
DROP_RADIUS = 2.0e-3    # m: a ballistic drop when the film has no surface tension (no capillary length)


@dataclass
class DropsConfig:
    tau_drip: float = 0.05              # s: a hanging film's excess over h_ret drips over this time
    requests: int = 64                  # drops per world and exchange step at most (the rest wait, none lost)
    pool: int = 256                     # ballistic drops in flight per world at most
    drop_radius: float | None = None    # m: ballistic drop radius (None: the capillary length)
    floor_z: float = 0.0                # m: a ballistic drop below this is booked as floored
    retention_angle: float = 90.0       # deg: θ_r of h_ret where neither the chart nor the material sets an angle
    absorb_column: int = 1              # SPH liquid with fewer liquid particles stacked above it is thin: 1 = a one-deep sheet
    absorb_reach: float = 1.5           # x particle radius above (or into) the film top
    absorb_speed: float = 0.05          # m/s: v·n above this is moving away (not absorbed)
    weight: bool = True                 # the film's weight on its chart's body
    debris: tuple[Any, ...] = field(default_factory=tuple)   # Debris(body, radius): bodies dragged by the film

    def __post_init__(self) -> None:
        self.debris = tuple(Debris(**d) if isinstance(d, dict) else d for d in self.debris)
        for name in ("tau_drip", "absorb_reach"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"drops.{name} must be > 0")
        if self.requests < 1 or self.pool < 1 or self.absorb_column < 1:
            raise ValueError("drops.requests, drops.pool and drops.absorb_column must be >= 1")


def edge_cells(tables: Any, drain_of: np.ndarray) -> dict[str, np.ndarray]:
    """Host: the cells with an open face and no drain, each once (its first open face): cell, face centre, the unit
    directions out and along it, its length; and ``of[C]``, a cell's edge index (-1 none)."""
    ftype, fa, fb = tables.ftype.numpy(), tables.fa.numpy(), tables.fb.numpy()
    fpos, fdir, flen = tables.fpos.numpy(), tables.fdir.numpy(), tables.flen.numpy()
    cnrm = tables.cnrm.numpy()
    of = np.full(len(cnrm), -1, np.int32)
    rows: list[tuple] = []
    for f in np.flatnonzero(ftype == OPEN_FACE):
        c = int(fa[f]) if fa[f] >= 0 else int(fb[f])
        if c < 0 or of[c] >= 0 or drain_of[c] >= 0:
            continue
        out = fdir[f] if c == fa[f] else -fdir[f]
        along = np.cross(cnrm[c], out)
        along /= max(np.linalg.norm(along), 1e-12)
        of[c] = len(rows)
        rows.append((c, fpos[f], out, along, flen[f]))
    cols = list(zip(*rows)) if rows else [[], [], [], [], []]
    return {"cell": np.asarray(cols[0], np.int32), "pos": np.asarray(cols[1], np.float32).reshape(-1, 3),
            "out": np.asarray(cols[2], np.float32).reshape(-1, 3), "along": np.asarray(cols[3], np.float32).reshape(-1, 3),
            "length": np.asarray(cols[4], np.float32), "of": of}


class Drops:
    """The drop stage (module docstring). ``config``: a ``DropsConfig``, a dict or None (the defaults)."""

    def __init__(self, config: Any = None, *, sph: bool = False):
        self.config = config if isinstance(config, DropsConfig) else DropsConfig(**(config or {}))
        self.sph = bool(sph)
        self.port = None

    # --- setup ---------------------------------------------------------------------------------------------------

    def setup(self, ctx: Any, tables: Any, film: Any) -> None:
        c = self.config
        self.ctx, self.tables, self.film = ctx, tables, film
        W, C, T, K = ctx.world_count, tables.cell_count, film.tl.count, len(film.config.charts)
        self._no_lumps, self._no_lump_sp = ctx.zeros((1, 1), wp.float64), ctx.zeros((1, 1, 1), wp.float64)
        m = film.mat
        g = float(np.linalg.norm(ctx.gravity))
        e = edge_cells(tables, film.drain_of.numpy())
        self.edges = DK.Edges()
        for name in ("cell", "length", "of"):
            setattr(self.edges, name, ctx.array(e[name] if len(e[name]) else np.zeros(1, e[name].dtype),
                                                wp.int32 if e[name].dtype == np.int32 else wp.float32))
        for name in ("pos", "out", "along"):
            setattr(self.edges, name, ctx.array(e[name] if len(e[name]) else np.zeros((1, 3), np.float32), wp.vec3))
        self.n_edges = len(e["cell"])
        angles = [ch.contact_angle if ch.contact_angle is not None else
                  m.contact_angle if m.contact_angle is not None else c.retention_angle for ch in film.config.charts]
        ret = [2.0 * math.sin(math.radians(a) / 2.0) * math.sqrt(m.surface_tension / m.density) for a in angles]
        self.ret = ctx.array(ret, wp.float32)
        self.pend = DK.Pending()
        self.pend.vol = film.add_state("drops_pending", ctx.zeros((W, C), wp.float64))
        self.pend.mom = film.add_state("drops_pending_mom", ctx.zeros((W, C), wp.vec3))
        self.pend.tile = film.add_state("drops_pending_tile", ctx.zeros((W, max(T, 1)), wp.int32))
        n_sp = film.sp.count
        if n_sp:
            self.pend.sp = film.add_state("drops_pending_species", ctx.zeros((W, C, n_sp), wp.float64))
            self.pend.sp_in = film.add_state("drops_species_shed", ctx.zeros((W, C, n_sp), wp.float64))
        else:
            self.pend.sp, self.pend.sp_in = ctx.zeros((1, 1, 1), wp.float64), ctx.zeros((1, 1, 1), wp.float64)
        R = c.requests
        self.req = DK.Requests()
        self.req.pos, self.req.vel = ctx.zeros((W, R), wp.vec3), ctx.zeros((W, R), wp.vec3)
        self.req.cell, self.req.n = ctx.zeros((W, R), wp.int32), ctx.zeros(W, wp.int32)
        self.req.tile_n, self.req.tile_off = ctx.zeros((W, max(T, 1)), wp.int32), ctx.zeros((W, max(T, 1)), wp.int32)
        self.req.slot = ctx.zeros((W, R), wp.int32)
        self.granted = ctx.zeros(W, wp.int32)
        book = {n: film.add_state(f"drops_{n}", ctx.zeros(W, wp.float64))
                for n in ("to_sph", "from_sph", "to_pool", "landed", "floored", "lump_wet", "lump_drops", "to_tanks")}
        self.book = book
        self.full = film.add_state("drops_pool_full", ctx.zeros(W, wp.int32))    # exchanges a full pool held drops
        self.need = film.add_state("drops_pool_need", ctx.zeros(W, wp.int32))    # the most drops it was asked to hold
        self.req_full = film.add_state("drops_requests_full", ctx.zeros(W, wp.int32))   # exchanges at the request cap
        self._warned = False
        self.forces = BodyForces(film, c.weight, c.debris)
        self.arrive_sph = self._arrive(book["from_sph"])
        self.arrive_lump = self._arrive(book["lump_wet"])
        self.pools = drop_pools(ctx, getattr(film, "supply", None))     # drips, drops into pools (drops_pools.py)
        self.pool = DF.Pool()
        P = c.pool
        self.pool.pos = film.add_state("drops_pos", ctx.zeros((W, P), wp.vec3))
        self.pool.vel = film.add_state("drops_vel", ctx.zeros((W, P), wp.vec3))
        self.pool.alive = film.add_state("drops_alive", ctx.zeros((W, P), wp.int32))
        self.pool.landed, self.pool.floored = book["landed"], book["floored"]
        self._species(W, P)
        self._landed_sum = ctx.zeros(W, wp.float64)                 # arrive()'s total for drops (pool.landed books)
        self.arrive_pool = self._arrive(self._landed_sum)
        l_c = math.sqrt(m.surface_tension / (m.density * g)) if g > 0.0 else 0.0
        r_d = c.drop_radius if c.drop_radius is not None else (l_c if l_c > 0.0 else DROP_RADIUS)
        self.drop_radius, self.drop_volume = float(r_d), float(4.0 / 3.0 * math.pi * r_d ** 3)
        N = max(int(ctx.model.particle_count), 1)
        self.mask, self.at = ctx.zeros(N, wp.int32), ctx.zeros(N, wp.vec2i)

    def _species(self, W: int, P: int) -> None:
        """Drops carry the film's and the lumps' species (drops_flight.Pool): per drop, the film's source, and per
        world the species landed, floored, taken into tanks and lost to SPH [W, K] kg, as the water's books."""
        f, ctx = self.film, self.ctx
        n_sp = f.sp.count
        K = max(n_sp, 1)
        self.pool.n_sp, self.pool.src_sp = n_sp, f.sp.src
        self.pool.lump = f.add_state("drops_lump", ctx.full((W, P), -1, wp.int32))
        self.pool.sp = f.add_state("drops_species", ctx.zeros((W, P, K), wp.float64))
        self.pool.landed_sp, self.pool.floored_sp, self.pool.tanks_sp = (
            f.add_state(f"drops_species_{n}", ctx.zeros((W, K), wp.float64)) for n in ("landed", "floored", "tanks"))
        self.sph_sp = f.add_state("drops_species_to_sph", ctx.zeros((W, K), wp.float64))
        if self.sph and n_sp:                          # SPH particles carry no species (WP15's round-5 S4)
            log.warning("%s: species that leave with SPH drops are lost (SPH particles carry no species): booked "
                        "species_to_sph", f.label)

    def _arrive(self, total: wp.array) -> DK.Arrive:
        f = self.film
        a = DK.Arrive()
        a.src_acc, a.src_mom, a.cwet, a.h, a.area = f.src_acc, f.src_mom, f.cwet, f.h, self.tables.area
        a.impulse, a.moment, a.total = self.forces.impulse, self.forces.imp_moment, total
        a.max_depth = float(f.config.max_depth)
        return a

    def _bind(self) -> Any:
        """The SPH port of this model (first use; plug-ins are set up before the solvers exist)."""
        if self.port is None:
            from alloy.solvers.sph.ports import port_for  # noqa: PLC0415 - SPH only when asked for

            port = port_for(self.ctx.model)
            if port is None:
                raise PluginError("surface_flow: sph=True but the model has no SPH solver (no liquid entry)")
            rho = float(self.film.mat.density)
            if abs(port.rest_density - rho) > 1e-6 * rho:
                raise PluginError(f"surface_flow: the film's density {rho} kg/m³ and the SPH liquid's "
                                  f"{port.rest_density} kg/m³ differ: drops would change the liquid's mass")
            self.port = port
        return self.port

    # --- stages --------------------------------------------------------------------------------------------------

    def apply(self, state: Any, forces: Any, dt: float) -> None:
        from alloy.plugins.surface_flow.plugin import H_WET  # noqa: PLC0415 - plugin imports this module

        self.forces.apply(state, forces, dt, H_WET)

    def absorb(self, state_in: Any, state_out: Any, dt: float) -> None:
        del state_in
        ctx, t, dev = self.ctx, self.tables, self.ctx.device
        mdl = ctx.model
        if self.sph:
            port, c = self._bind(), self.config
            reach = c.absorb_reach * float(port.solver.config.particle_radius)
            wp.launch(DK.mark_kernel, dim=mdl.particle_count,
                      inputs=[port.particles(state_out), c.absorb_column, float(port.solver.config.spacing), t.charts,
                              t.prof, t.cbody, t.clocal, state_out.body_q, state_out.body_qd, mdl.body_com, self.film.h,
                              self.film.frame.g, self.film.tick, self.film.gravity, reach, c.absorb_speed],
                      outputs=[self.mask, self.at], device=dev)
            wp.launch(DK.deposit_kernel, dim=ctx.world_count,
                      inputs=[port.ws, state_out.particle_q, state_out.particle_qd, mdl.particle_mass,
                              port.particle_volume, t.charts, t.prof, t.cbody, t.clocal, state_out.body_q,
                              state_out.body_qd, mdl.body_com, self.at, self.arrive_sph], outputs=[self.mask],
                      device=dev)
            port.absorb(state_out, self.mask)
            return
        wp.launch(DF.advance_kernel, dim=ctx.world_count,
                  inputs=[self.pool, self.drop_volume, float(self.film.mat.density), mdl.gravity, self.config.floor_z,
                          float(dt), t.charts, t.prof, t.cbody, t.clocal, state_out.body_q, state_out.body_qd,
                          mdl.body_com, self.arrive_pool, self.pools, self.book["to_tanks"]], device=dev)

    def shed(self, state_out: Any, dt: float) -> None:
        f, ctx, t, c = self.film, self.ctx, self.tables, self.config
        W, T, dev = ctx.world_count, f.tl.count, ctx.device
        if self.n_edges:
            wp.launch(DK.collect_kernel, dim=(W, self.n_edges),
                      inputs=[self.edges, f.shed, f.shed_mom, f.cells.tile, f.sp, self.pend], device=dev)
        if not T:
            return
        decay = 1.0 - math.exp(-float(dt) / c.tau_drip)
        wp.launch(DK.drip_kernel, dim=(W, T),
                  inputs=[f.tl.tiles, f.tile_on, f.cells, f.frame, self.ret, 1.0e-6, decay, f.flooded, f.h, f.book.dripped,
                          f.sp, self.pend], device=dev)
        if self.sph:
            port = self._bind()
            vol, rad = port.particle_volume, float(port.solver.config.particle_radius)
        else:
            vol, rad = self.drop_volume, self.drop_radius
        wp.launch(DK.count_kernel, dim=(W, T), inputs=[f.tl.tiles, f.tile_on, self.pend, vol, self.req], device=dev)
        wp.launch(DK.offset_kernel, dim=W, inputs=[T, c.requests, self.req], device=dev)
        wp.launch(DK.fill_kernel, dim=(W, T),
                  inputs=[f.tl.tiles, f.tile_on, self.edges, f.cells, t.charts, t.cbody, t.clocal, state_out.body_q,
                          state_out.body_qd, ctx.model.body_com, f.h, self.pend, vol, rad, c.requests, self.req],
                  device=dev)
        lumps = getattr(f, "lumped", None)
        if lumps is not None:                          # lumped bodies' drips after the cells' (lumped_drip.py)
            wp.launch(LD.lump_drip_kernel, dim=W,
                      inputs=[lumps.drip, state_out.body_q, state_out.body_qd, ctx.model.body_com, ctx.model.gravity,
                              vol, rad, float(f.mat.density), c.requests, t.charts, t.prof, t.cbody, t.clocal, f.base,
                              self.arrive_lump, f.sp.src, self.req, self.pools], device=dev)
        if self.sph:
            port.emit(state_out, self.req.pos, self.req.vel, self.req.n)
            granted, out = port.granted, self.book["to_sph"]
        else:
            wp.launch(DF.spawn_kernel, dim=W, inputs=[self.req, self.pool, self.granted], device=dev)
            wp.launch(DF.pool_full_kernel, dim=W, inputs=[self.req, self.pool, self.granted, c.requests, self.full,
                                                          self.need, self.req_full], device=dev)
            granted, out = self.granted, self.book["to_pool"]
        lump_pend = lumps.drip.pending if lumps is not None else self._no_lumps
        lump_sp = lumps.drip.pending_sp if lumps is not None else self._no_lump_sp
        drop_sp = lumps.drip.drop_sp if lumps is not None else self._no_lump_sp
        wp.launch(DK.commit_kernel, dim=W, inputs=[self.req, granted, vol, f.cells, self.pend, lump_pend, lump_sp,
                                                   drop_sp, f.sp.count, int(not self.sph), self.pool.sp, self.sph_sp,
                                                   out, self.book["lump_drops"]], device=dev)

    # --- host side -----------------------------------------------------------------------------------------------

    def read(self) -> dict[str, np.ndarray]:
        """Per-world ledger [m³] (module docstring): to_sph, from_sph, to_pool, landed, floored, pending, in_flight."""
        W = self.ctx.world_count
        out = {f"{k}_volume": v.numpy()[:W].copy() for k, v in self.book.items()}
        out["pending_volume"] = self.pend.vol.numpy()[:W].sum(axis=1)
        out["in_flight_volume"] = self.pool.alive.numpy()[:W].sum(axis=1) * self.drop_volume
        lumps = getattr(self.film, "lumped", None)
        if lumps is not None:
            out["lump_pending_volume"] = lumps.drip.pending.numpy()[:W].sum(axis=1)
            out["lump_tanks_volume"] = lumps.drip.tank.numpy()[:W].sum(axis=1)
        out.update(self._species_read(self.film.sp.count))
        out["pool_full"], out["pool_need"] = self.full.numpy()[:W].copy(), self.need.numpy()[:W].copy()
        out["requests_full"] = self.req_full.numpy()[:W].copy()
        if (out["pool_full"].any() or out["requests_full"].any()) and not self._warned:
            self._warned = True
            log.warning("%s: drops waited (none lost): the drop pool (%d per world) was full on up to %d exchanges "
                        "(in flight plus asked for reached %d: DropsConfig(pool=%d) or more), and every request slot "
                        "(%d per exchange) was used on up to %d (DropsConfig(requests=...) higher)", self.film.label,
                        self.config.pool, int(out["pool_full"].max()), int(out["pool_need"].max()),
                        int(out["pool_need"].max()), self.config.requests, int(out["requests_full"].max()))
        return out

    def _species_read(self, k: int) -> dict[str, np.ndarray]:
        """The drops' species per world [W, K] kg (module docstring): shed by the cells, pending, lost to SPH,
        landed, floored, taken into tanks and in flight (with the lumps' wet, waiting, dropped and tanks books they
        close the lumps' shed species, lumped.read)."""
        if not k:
            return {}
        W, p = self.ctx.world_count, self.pool
        alive = p.alive.numpy()[:W].astype(bool)
        out = {f"species_{n}": a.numpy()[:W, :k].copy() for n, a in (("landed", p.landed_sp),
                                                                        ("floored", p.floored_sp),
                                                                        ("to_tanks", p.tanks_sp),
                                                                        ("to_sph", self.sph_sp))}
        out["species_shed"] = self.pend.sp_in.numpy()[:W, :, :k].sum(axis=1)
        out["species_pending"] = self.pend.sp.numpy()[:W, :, :k].sum(axis=1)
        out["species_in_flight"] = (p.sp.numpy()[:W, :, :k] * alive[:, :, None]).sum(axis=1)
        return out


__all__ = ["Drops", "DropsConfig", "edge_cells"]
