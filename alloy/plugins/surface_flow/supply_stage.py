# SPDX-License-Identifier: Apache-2.0
"""Stage drips (WP6's patch-flush sink, WP15's stores): volume a film stage releases from a tool (a store's drips, a
trowel's dumped load), handed to the supply and cast like any jet, so it lands where it falls: in a pool's tank (a
bucket), on a chart as film, caught by a shape, or lost.

The stage registers, before the supply's setup and as state, the arrays (E entries, one per tool):
- ``src_drip_v [W, E]`` float64: m³ released this exchange (accumulated);
- ``src_drip_p [W, E]`` vec3: the world exit point; ``src_drip_u [W, E]`` vec3: the world velocity there;
- ``src_drip_sp [W, E, max(K, 1)]`` float64: kg of each film species with it;
- optionally ``src_drip_body [W, E]`` int32: the tool's body, whose shapes the drip passes (-1: none).
The supply builds one internal one-ray jet nozzle per entry. At the end of ``after_film`` (``stage_commit``) it books
``staged`` += v, puts v/dt in the entry's column of ``nozzle_q`` and the drip's point, velocity and concentrations in
flight, and zeroes the requests: the next exchange casts and deposits it (so staged = what the stage drips sent + what
is in flight). A negative volume, and species with no volume, are refused (``refused``; species: their ``refused``
column). Into a pool, a film species becomes the network's of its name; one the network lacks goes ``away``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.surface_flow.supply_kernels import REFUSED, STAGED
from alloy.plugins.surface_flow.supply_spec import STAGE, WATER, Nozzle
from alloy.plugins.surface_flow.supply_species import SP_REFUSED, SP_STAGED, NetSpecies

STAGE_ARRAYS = ("src_drip_v", "src_drip_p", "src_drip_u", "src_drip_sp")
STAGE_SEGMENTS = 32              # chords of a drip's fall over the 1.5 m reach (within 0.1 mm of the parabola, WP15)


@wp.kernel
def stage_commit(dt: wp.float64, col0: int, ray0: wp.array[wp.int32], v: wp.array2d[wp.float64],
                 p: wp.array2d[wp.vec3], u: wp.array2d[wp.vec3], sp: wp.array3d[wp.float64], ns: NetSpecies,
                 nozzle_q: wp.array2d[wp.float64], ray_p0: wp.array2d[wp.vec3], ray_dir: wp.array2d[wp.vec3],
                 sdrip: wp.array3d[wp.float64], ledger: wp.array2d[wp.float64]):
    """Per world, the stage's requests in entry order: into flight for the next exchange (module docstring)."""
    w = wp.tid()
    zero = wp.float64(0.0)
    for e in range(v.shape[1]):
        vol = v[w, e]
        v[w, e] = zero
        ok = vol > zero
        if vol < zero:
            ledger[w, REFUSED] = ledger[w, REFUSED] - vol
        q = zero
        if ok:
            q = vol / dt
            ledger[w, STAGED] = ledger[w, STAGED] + vol
            ray_p0[w, ray0[e]] = p[w, e]
            ray_dir[w, ray0[e]] = u[w, e]
        nozzle_q[w, col0 + e] = -q
        for k in range(sp.shape[2]):
            m = sp[w, e, k]
            sp[w, e, k] = zero
            c = zero
            if ok:
                c = m / vol
                if k < ns.n_film:
                    ns.led[w, SP_STAGED, k] = ns.led[w, SP_STAGED, k] + m
            elif k < ns.n_film:
                ns.led[w, SP_REFUSED, k] = ns.led[w, SP_REFUSED, k] + m
            sdrip[w, e, k] = c


# ---- host side -----------------------------------------------------------------------------------------------------

def stage_entries(film: Any) -> int:
    """The film's stage-drip entries E (0: none registered); all of ``STAGE_ARRAYS`` or none."""
    have = {n: getattr(film, n, None) for n in STAGE_ARRAYS}
    if all(a is None for a in have.values()):
        return 0
    missing = sorted(n for n, a in have.items() if a is None)
    if missing:
        raise PluginError(f"stage drips: the film registers all of {STAGE_ARRAYS}; missing {missing}")
    return int(have["src_drip_v"].shape[1])


def stage_nozzles(film: Any) -> tuple[Nozzle, ...]:
    """One internal one-ray jet nozzle per stage-drip entry (world frame): ``STAGE_SEGMENTS`` chords of its fall, the
    film's density (its momentum on what it strikes)."""
    rho = float(getattr(getattr(film, "mat", None), "density", WATER[0]))
    return tuple(Nozzle(f"stage:{e}", edge="", drip_of=STAGE, exit_area=1.0, segments=STAGE_SEGMENTS, density=rho)
                 for e in range(stage_entries(film)))


def check_stage(ctx: Any, film: Any, n_film: int) -> None:
    """The stage's arrays' shapes and types (``PluginError`` naming the first wrong one)."""
    W, E = ctx.world_count, stage_entries(film)
    want = {"src_drip_v": ((W, E), wp.float64), "src_drip_p": ((W, E), wp.vec3), "src_drip_u": ((W, E), wp.vec3),
            "src_drip_sp": ((W, E, max(n_film, 1)), wp.float64), "src_drip_body": ((W, E), wp.int32)}
    for name, (shape, dtype) in want.items():
        a = getattr(film, name, None)
        if a is None and name == "src_drip_body":
            continue
        if tuple(a.shape) != shape or a.dtype != dtype:
            raise PluginError(f"stage drips: {name} must be {shape} {dtype.__name__} (W worlds, E entries, K film "
                              f"species); got {tuple(a.shape)} {a.dtype.__name__}")


class Staging:
    """``Supply``'s stage drips (a mixin: it uses the supply's nozzle tables, discharge table and species)."""

    def _setup_stage(self, ctx: Any, film: Any) -> None:
        """The stage's arrays checked, its nozzles' bodies (the drip passes them) and their drips in flight."""
        W = ctx.world_count
        stage = [n for n, noz in enumerate(self.nozzles) if noz.drip_of == STAGE]
        self.n_stage = len(stage)
        K = max(self.ns.n_film, 1)
        if not stage:
            self.ns.sdrip, self.stage_ray = ctx.zeros((1, 1, 1), wp.float64), ctx.zeros(1, wp.int32)
            return
        check_stage(ctx, film, self.ns.n_film)
        self.stage_col0 = self.fluid.discharge.shape[1] + sum(1 for noz in self.nozzles[:stage[0]] if noz.drip_of)
        self.stage_ray = ctx.array(np.asarray([self.ray_starts[n] for n in stage], np.int32), wp.int32)
        self.ns.sdrip = film.add_state("supply_stage_conc", ctx.zeros((W, self.n_stage, K), wp.float64))
        body = getattr(film, "src_drip_body", None)
        if body is not None:                                  # the drip passes its own tool's shapes
            nb = self.nbody.numpy()
            nb[:, stage] = body.numpy()
            self.nbody.assign(nb)

    def _stage_commit(self, dt: float) -> None:
        """End of ``after_film``: the stage's requests into flight (``stage_commit``)."""
        if self.n_stage:
            f = self.film
            wp.launch(stage_commit, dim=self.ctx.world_count,
                      inputs=[wp.float64(dt), self.stage_col0, self.stage_ray, f.src_drip_v, f.src_drip_p,
                              f.src_drip_u, f.src_drip_sp, self.ns],
                      outputs=[self.nozzle_q, self.ray_p0, self.ray_dir, self.ns.sdrip, self.ledger],
                      device=self.ctx.device)


__all__ = ["STAGE_ARRAYS", "STAGE_SEGMENTS", "Staging", "check_stage", "stage_commit", "stage_entries", "stage_nozzles"]
