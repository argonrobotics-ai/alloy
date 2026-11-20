# SPDX-License-Identifier: Apache-2.0
"""Cloth takes up liquid (opt-in), coupling side: the scene's request to the SPH entry, and the masses.

``add_cloth(absorb={"capacity": kg/m^2, "rate": kg/m^2/s, "wicking": m^2/s})`` writes ``features.absorb`` on the
sheet's card (``absorb_feature``). ``make_entries`` gives the SPH entry each absorbing sheet's particle id ranges
and values (``attach_absorb``, solver kwarg ``ABSORB_KEY``), and the SPH factory installs the uptake
(``install_absorb``; alloy/solvers/sph/cloth_absorb.py does the uptake itself).

Masses and momentum (``ClothAbsorbCoupling``; the coupled solver calls ``finish_exchange(state_out)`` at the end of
every exchange and ``restore()`` after a period rollback and at reset, before the energy reference is re-based).
One kernel writes every mass from the state, never adding to it:
- a cloth vertex: m = m_dry + w into the cloth entry's view, the parent model (mass and inverse mass together: kernels
  read inv_mass live), the SPH view's proxy (the build's mass_scale x m) and the endpoint set's ``meff``;
- a liquid particle: its build mass while live, 0 once ABSORBED (health's energy book reads the parent's masses, so the
  water counts once, on the cloth), and an absorbed particle's velocity 0.
At the exchange end the same kernel first merges the exchange's uptake into the vertex velocity, (m v + P) / (m + M),
from the int64 accumulators, then zeroes them: one deterministic merge per exchange, whatever the pass count.
Masses change only between exchanges. The exchange clock keeps the dry masses (``m_axis``, host, set at build): a
heavier vertex allows a coarser clock, so the dry masses are the conservative bound. Pinned vertices (zero mass) take
no water.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import warp as wp

log = logging.getLogger("alloy.coupling")

ABSORB_KEY = "alloy_cloth_absorb"  # sph solver kwarg: ((id ranges, capacity, rate, wicking, name), ...) per sheet
_FIELDS = ("capacity", "rate", "wicking")
_UNITS = {"capacity": "kg/m^2", "rate": "kg/m^2/s", "wicking": "m^2/s"}
W_SCALE_HOST, IMP_SCALE_HOST = float(2**40), float(2**50)   # cloth_absorb.W_SCALE_HOST, sph bodies' IMP_SCALE
W_SCALE, IMP_SCALE = wp.constant(W_SCALE_HOST), wp.constant(IMP_SCALE_HOST)


@wp.func
def _ufx(q: wp.int64, scale: float) -> float:
    return float(wp.float64(q) / wp.float64(scale))


# --------------------------------------------------------------------------------------------- API and entries
def absorb_feature(absorb: Any, extra: Mapping[str, Any]) -> dict[str, Any]:
    """``features.absorb`` for ``add_cloth``'s card from the user's dict. Refuses a film sheet and bad values."""
    from alloy.api._args import user_q  # noqa: PLC0415
    from alloy.api.errors import SceneError  # noqa: PLC0415

    fix = 'pass absorb={"capacity": kg/m^2, "rate": kg/m^2/s, "wicking": m^2/s} (wicking optional)'
    if (extra.get("route") or {}).get("route") == "cloth_film":
        raise SceneError("film and paper sheets take up no liquid", "leave absorb out for a film sheet, or use a "
                         "fabric material", field="absorb")
    if not isinstance(absorb, Mapping) or set(absorb) - set(_FIELDS) or not {"capacity", "rate"} <= set(absorb):
        raise SceneError(f"absorb must be a dict with capacity and rate, got {absorb!r}", fix, field="absorb")
    out = {}
    for k, v in absorb.items():
        x = float(v)
        if not (x > 0.0 or (k == "wicking" and x == 0.0)) or not np.isfinite(x):
            raise SceneError(f"absorb {k} must be {'>= 0' if k == 'wicking' else '> 0'}, got {v!r}", fix,
                             field="absorb")
        out[k] = user_q(x, f"cloth absorb {k} [{_UNITS[k]}] set through alloy.Scene.add_cloth")
    return out


def attach_absorb(specs: list, objs: Sequence[Any], strides: Mapping[str, int], worlds: int) -> None:
    """Give the SPH entry each absorbing sheet's particle id ranges and values (module doc)."""
    from alloy.coupling.liquid_cloth_selfcontact import id_ranges  # noqa: PLC0415
    from alloy.scene.entries import global_ids  # noqa: PLC0415

    from alloy.route.types import Route  # noqa: PLC0415

    asked = []
    for t in objs:
        a = getattr(getattr(getattr(t.obj, "card", None), "features", None), "absorb", None)
        if a is None:
            continue
        if t.decision.route != Route.CLOTH_VBD:
            log.warning("cloth absorb: %r routes %s, and only a cloth sheet takes up liquid: its absorb is ignored",
                        t.name, getattr(t.decision.route, "value", t.decision.route))
            continue
        ids = np.asarray(global_ids(t.ranges.particle, strides["particle"], worlds), np.int64)
        asked.append((id_ranges(ids), float(a.capacity.value), float(a.rate.value),
                      0.0 if a.wicking is None else float(a.wicking.value), t.name))
    if not asked:
        return
    for e in specs:
        if e.capability == "sph":
            e.solver_kwargs[ABSORB_KEY] = tuple(asked)


def install_absorb(solver: Any, asked: tuple | None) -> Any:
    """Install the uptake on an SPH solver for the asked sheets; returns the ClothAbsorb, or None."""
    from alloy.solvers.sph.cloth_absorb import ClothAbsorb, vertex_areas  # noqa: PLC0415

    sw = getattr(solver, "soft", None)
    if not asked or sw is None or not sw.count:
        if asked:
            log.warning("cloth absorb: no cloth faces reach the liquid's solver; nothing absorbs")
        return None
    vid = sw.vid.numpy()
    l2g = solver_l2g(solver, vid)
    tri = np.asarray([tuple(t) for t in sw.arrays.tri.numpy()], np.int64)
    comp = sw.arrays.face_comp.numpy()
    two = sw.arrays.two_sided.numpy()
    area = vertex_areas(solver.model.particle_q.numpy()[vid].astype(np.float64), tri)
    cap, rate, wick = np.zeros(len(vid)), np.zeros(len(vid)), np.zeros(len(vid))
    sheet = np.full(len(vid), -1)                                  # which asked sheet each vertex belongs to
    for k, (ranges, c, q, d, _name) in enumerate(asked):
        on = np.zeros(len(vid), bool)
        for a, b in ranges:
            on |= (l2g >= a) & (l2g < b)
        cap[on], rate[on], wick[on], sheet[on] = c * area[on], q * area[on], d, k
    absorbs = np.zeros(len(two), np.int32)
    face_on = (cap[tri] > 0).all(axis=1) & (two[comp] != 0)      # cloth faces of an absorbing sheet
    absorbs[np.unique(comp[face_on])] = 1
    on_face = np.zeros(len(vid), bool)
    on_face[np.unique(tri[face_on])] = True
    cap[~on_face], rate[~on_face] = 0.0, 0.0
    ph = solver.layout.phase.numpy()
    burst = float(solver.model.particle_mass.numpy()[ph != -1].max()) if (ph != -1).any() else 0.0
    ca = ClothAbsorb(sw, absorbs, cap, rate, float(solver.config.particle_radius), solver.model, solver.layout.phase,
                     int(solver.layout.world_count), burst, wick)
    solver.cloth_absorb = ca
    _reset_with(solver, ca)
    _report(asked, sheet, cap, area, burst)
    return ca


def _ceil3(x: float) -> float:
    """x rounded up to 3 significant figures, so a capacity printed from it is never short of the bound."""
    step = 10.0 ** (math.floor(math.log10(x)) - 2)
    return math.ceil(x * (1.0 + 1e-9) / step) * step


def _report(asked: tuple, sheet: np.ndarray, cap: np.ndarray, area: np.ndarray, burst: float) -> None:
    """The install line, and one warning per sheet with vertices that hold under a third of a liquid particle. A
    particle is taken only when every vertex of its face has room for its share, and the largest share is at least a
    third of it, so such vertices take up little or nothing (a fine sheet takes up none)."""
    least = burst / 3.0
    can = (cap > 0) & (cap >= least)
    for k, (_r, _c, _q, _d, name) in enumerate(asked):
        on = (sheet == k) & (cap > 0)
        short = on & ~can
        if not short.any():
            continue
        log.warning("cloth absorb: %d of %d vertices of %r hold under a third of a liquid particle (capacity x area "
                    "%.3g-%.3g mg, particle %.3g mg): a particle is taken only when each vertex of its face has room "
                    "for its share, so they take up little or nothing. Use a coarser resolution, a capacity of at "
                    "least %g kg/m^2, or alloy.Scene(liquid_accuracy=\"super\") (smaller particles)",
                    int(short.sum()), int(on.sum()), name, cap[on].min() * 1e6, cap[on].max() * 1e6, burst * 1e6,
                    _ceil3(least / float(area[on].min())))
    log.info("cloth absorb: %d of %d vertices of absorbing sheets can take up a particle (%.3g kg in all)",
             int(can.sum()), int((sheet >= 0).sum()), float(cap.sum()))


def _reset_with(solver: Any, ca: Any) -> None:
    """``solver.reset`` (masked worlds back to their build state) also zeroes those worlds' uptake state; wrapped per
    solver instance. The solver's own reset restores the absorbed particles' flags and phase."""
    reset = solver.reset

    def reset_and_absorb(state: Any, world_mask: Any = None, flags: Any = None) -> None:
        reset(state, world_mask, flags)
        ca.reset(world_mask)
    solver.reset = reset_and_absorb


def solver_l2g(solver: Any, local: np.ndarray) -> np.ndarray:
    """Global (parent) ids of a solver view's local particle ids; identity when the view keeps every particle."""
    m = getattr(solver, "alloy_particle_local_to_global", None)
    return local.astype(np.int64) if m is None else m.numpy()[local].astype(np.int64)


# ------------------------------------------------------------------------------------------------ masses
@wp.struct
class DryMasses:
    """Build-time values per endpoint: written exactly while a vertex holds no water (bit for bit when dry)."""

    m: wp.array[wp.float32]       # cloth view mass (= parent's)
    inv: wp.array[wp.float32]     # cloth view inverse mass
    pinv: wp.array[wp.float32]    # parent inverse mass
    sph: wp.array[wp.float32]     # SPH proxy mass
    sinv: wp.array[wp.float32]    # SPH proxy inverse mass
    meff: wp.array[float]         # the endpoint set's meff


@wp.kernel(enable_backward=False)
def cloth_mass_kernel(vtx: wp.array[wp.int32], water: wp.array[wp.int64], acc_m: wp.array[wp.int64],
                      acc_p: wp.array2d[wp.int64], merge: int, dry: DryMasses,
                      src_local: wp.array[int], src_global: wp.array[int], proxy_local: wp.array[int],
                      qd: wp.array[wp.vec3], vbd_m: wp.array[wp.float32], vbd_inv: wp.array[wp.float32],
                      par_m: wp.array[wp.float32], par_inv: wp.array[wp.float32], sph_m: wp.array[wp.float32],
                      sph_inv: wp.array[wp.float32], meff: wp.array[float]):
    """Each endpoint's cloth vertex (module doc): the merge (merge != 0), then every mass from m_dry + w; a dry vertex
    gets its build-time values exactly."""
    i = wp.tid()
    k = vtx[i]
    md = dry.m[i]
    if k < 0 or md <= 0.0:
        return
    g = src_global[i]
    if merge != 0:
        dm = _ufx(acc_m[k], W_SCALE)
        if dm > 0.0:
            P = wp.vec3(_ufx(acc_p[k, 0], IMP_SCALE), _ufx(acc_p[k, 1], IMP_SCALE), _ufx(acc_p[k, 2], IMP_SCALE))
            m0 = par_m[g]
            qd[g] = (m0 * qd[g] + P) / (m0 + dm)
        acc_m[k] = wp.int64(0)
        for d in range(3):
            acc_p[k, d] = wp.int64(0)
    if water[k] == wp.int64(0):
        vbd_m[src_local[i]] = md
        vbd_inv[src_local[i]] = dry.inv[i]
        par_m[g] = md
        par_inv[g] = dry.pinv[i]
        sph_m[proxy_local[i]] = dry.sph[i]
        sph_inv[proxy_local[i]] = dry.sinv[i]
        meff[i] = dry.meff[i]
        return
    m = md + _ufx(water[k], W_SCALE)
    vbd_m[src_local[i]] = m
    vbd_inv[src_local[i]] = 1.0 / m
    par_m[g] = m
    par_inv[g] = 1.0 / m
    ms = dry.sph[i] * (m / md)
    sph_m[proxy_local[i]] = ms
    sph_inv[proxy_local[i]] = 1.0 / ms
    meff[i] = ms


@wp.kernel(enable_backward=False)
def liquid_mass_kernel(phase: wp.array[wp.int8], absorbed: int, l2g: wp.array[int], m0: wp.array[wp.float32],
                       inv0: wp.array[wp.float32], zero_v: int, qd: wp.array[wp.vec3], par_m: wp.array[wp.float32],
                       par_inv: wp.array[wp.float32]):
    """A liquid particle's parent mass: its build mass while live, 0 once absorbed (and its velocity 0)."""
    j = wp.tid()
    g = l2g[j]
    ph = int(phase[j])
    if g < 0 or (ph < 0 and ph != absorbed):
        return
    if ph == absorbed:
        par_m[g] = 0.0
        par_inv[g] = 0.0
        if zero_v != 0:
            qd[g] = wp.vec3(0.0, 0.0, 0.0)
    else:
        par_m[g] = m0[j]
        par_inv[g] = inv0[j]


class _CompactMap:
    """One cloth -> SPH particle endpoint set's arrays for the mass kernel."""

    def __init__(self, es: Any, src_view: Any, dst_view: Any, ca: Any, model: Any) -> None:
        dev = es.world.device
        proxy = es.proxy_local.numpy().astype(np.int64)
        slot = {int(v): k for k, v in enumerate(ca.soft_walls.vid.numpy())}
        vtx = np.array([slot.get(int(p), -1) for p in proxy], np.int32)
        src = es.src_local.numpy()
        m_dry = src_view.particle_mass.numpy()[src]
        cap = ca.cap.numpy()
        cap[vtx[(vtx >= 0) & (m_dry <= 0.0)]] = 0                 # a pinned vertex takes no water
        ca.cap.assign(cap)

        def f32(a: np.ndarray, dt: Any = wp.float32) -> wp.array:
            return wp.array(np.asarray(a, np.float32), dtype=dt, device=dev)
        d = DryMasses()
        d.m, d.inv = f32(m_dry), f32(src_view.particle_inv_mass.numpy()[src])
        d.pinv = f32(model.particle_inv_mass.numpy()[es.src_global.numpy()])
        d.sph, d.sinv = f32(dst_view.particle_mass.numpy()[proxy]), f32(dst_view.particle_inv_mass.numpy()[proxy])
        d.meff = f32(es.meff.numpy(), float)
        self.es, self.src_view, self.dst_view, self.dry = es, src_view, dst_view, d
        self.vtx = wp.array(vtx, dtype=wp.int32, device=dev)


class ClothAbsorbCoupling:
    """The exchange-end merge and mass writes of a coupled solver whose SPH entry absorbs (module doc)."""

    def __init__(self, solver: Any, sph_name: str) -> None:
        from alloy.solvers.sph import bodies, cloth_absorb  # noqa: PLC0415

        if (cloth_absorb.W_SCALE_HOST, bodies.IMP_SCALE_HOST) != (W_SCALE_HOST, IMP_SCALE_HOST):
            raise RuntimeError("cloth absorb: fixed-point scales differ between the SPH uptake and the mass writes")
        sph = solver._entries[sph_name]
        self.ca, self.model = sph.solver.cloth_absorb, solver.model
        self.maps = [_CompactMap(es, solver._entries[es.src_name].view, sph.view, self.ca, solver.model)
                     for es in solver._sets if es.kind == "particle" and es.dst_name == sph_name]
        n = int(sph.solver.model.particle_count)
        dev = self.model.device
        l2g = getattr(sph, "particle_local_to_global", None)
        self.l2g = l2g if l2g is not None else wp.array(np.arange(n), dtype=int, device=dev)
        g = self.l2g.numpy()[:n]
        m0 = np.where(g >= 0, self.model.particle_mass.numpy()[np.maximum(g, 0)], 0.0).astype(np.float32)
        inv0 = np.where(g >= 0, self.model.particle_inv_mass.numpy()[np.maximum(g, 0)], 0.0).astype(np.float32)
        self.m0 = wp.array(m0, dtype=wp.float32, device=dev)
        self.inv0 = wp.array(inv0, dtype=wp.float32, device=dev)
        self.phase, self.n = sph.solver.layout.phase, n
        kg = np.full(int(self.ca.water.shape[0]), -1, np.int32)
        for c in self.maps:
            vtx, g = c.vtx.numpy(), c.es.src_global.numpy()
            kg[vtx[vtx >= 0]] = g[vtx >= 0]
        self.kglobal = wp.array(kg, dtype=wp.int32, device=dev)          # compact soft vertex -> parent particle
        from alloy.solvers.sph.cloth_absorb import ABSORBED  # noqa: PLC0415

        self.absorbed = int(ABSORBED)
        self._dummy = wp.zeros(1, dtype=wp.vec3, device=dev)     # restore(): no merge, so no state velocities
        log.info("cloth absorb: masses written at each exchange end for %d cloth -> liquid endpoint set(s)",
                 len(self.maps))

    def _write(self, qd: wp.array | None) -> None:
        """Device launches only (it runs inside captured periods): the merge when ``qd`` is given, then the masses."""
        m, ca, dev = self.model, self.ca, self.model.device
        merge = int(qd is not None)
        q = qd if qd is not None else self._dummy
        for c in self.maps:
            es, sv, dv = c.es, c.src_view, c.dst_view
            wp.launch(cloth_mass_kernel, dim=es.count, device=dev,
                      inputs=[c.vtx, ca.water, ca.acc_m, ca.acc_p, merge, c.dry, es.src_local,
                              es.src_global, es.proxy_local, q, sv.particle_mass, sv.particle_inv_mass,
                              m.particle_mass, m.particle_inv_mass, dv.particle_mass, dv.particle_inv_mass, es.meff])
        if self.n:
            wp.launch(liquid_mass_kernel, dim=self.n, device=dev,
                      inputs=[self.phase, self.absorbed, self.l2g, self.m0, self.inv0, merge, q, m.particle_mass,
                              m.particle_inv_mass])

    def finish_exchange(self, state_out: Any) -> None:
        """The exchange end: wicking (into the accumulators), the merge into the vertex velocities, every mass."""
        self.ca.wick(self.kglobal, state_out.particle_qd)
        self._write(state_out.particle_qd)

    def restore(self) -> None:
        """After a period rollback or a reset: every mass from the restored state (no merge)."""
        self._write(None)


def exchange_coupling(solver: Any) -> ClothAbsorbCoupling | None:
    """The coupled solver's absorb coupling, or None when no SPH entry absorbs (then nothing changes). Decided from the
    entries alone; a coupler without proxy endpoint sets (ADMM) gets an object back only so its builder refuses it
    (alloy.coupling.absorb_hook.refuse_with_admm)."""
    names = [n for n, e in getattr(solver, "_entries", {}).items()
             if getattr(getattr(e, "solver", None), "cloth_absorb", None) is not None]
    if not names:
        return None
    if not hasattr(solver, "_sets"):
        return _NoProxySets()
    return ClothAbsorbCoupling(solver, names[0])


class _NoProxySets:
    """Absorb asked on a coupler without proxy endpoint sets: the coupler's builder refuses it by name."""

    def finish_exchange(self, state_out: Any) -> None:
        raise RuntimeError("cloth absorb needs the proxy coupler")

    def restore(self) -> None:
        raise RuntimeError("cloth absorb needs the proxy coupler")


# ------------------------------------------------------------------------------------------------ state
def state_keys(engine: Any, name: str, rows: np.ndarray) -> dict[str, np.ndarray]:
    """``sim.state`` keys of an object in a scene whose liquid is taken up by cloth (none otherwise):
    ``"<sheet>.water"`` (worlds, vertices) kg held per vertex, ``"<liquid>.absorbed"`` (worlds,) particles taken up.
    ``rows``: the object's (worlds, n) global particle ids."""
    from alloy.api._liquid import liquid_solvers  # noqa: PLC0415
    from alloy.solvers.sph.cloth_absorb import ABSORBED  # noqa: PLC0415

    out: dict[str, np.ndarray] = {}
    for s in liquid_solvers(engine):
        ca = getattr(s, "cloth_absorb", None)
        if ca is None:
            continue
        holds = ca.cap.numpy() > 0                       # absorbing vertices only: other VBD objects get no key
        vid = solver_l2g(s, ca.soft_walls.vid.numpy())[holds]
        hit = np.isin(rows, vid)
        if hit.any():
            order = np.argsort(vid)
            pos = np.searchsorted(vid, rows, sorter=order)
            w = ca.water_kg()[holds][order[np.clip(pos, 0, len(vid) - 1)]]
            out[f"{name}.water"] = np.where(hit, w, 0.0)
        # phase is indexed by the SPH model's own ids and rows are global: they agree while the SPH model holds
        # every parent particle (every route today); otherwise the shape guard drops the key rather than miscount
        phase = s.layout.phase.numpy()
        if phase.shape[0] == s.model.particle_count and (phase[rows] != -1).all():
            out[f"{name}.absorbed"] = (phase[rows] == ABSORBED).sum(axis=1)
    return out


__all__ = ["ABSORB_KEY", "ClothAbsorbCoupling", "absorb_feature", "attach_absorb", "cloth_mass_kernel",
           "exchange_coupling", "install_absorb", "liquid_mass_kernel", "solver_l2g", "state_keys"]
