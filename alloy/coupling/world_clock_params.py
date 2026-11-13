# SPDX-License-Identifier: Apache-2.0
"""#10b S1b: a world's randomized parameters move with it ([internal note] notes/gap-10b/s1b_params.md).

``WorldTransfer`` (fine_pack) moves a world's state. A randomized scene (``Scene.randomize``) also gives each world
its own Model rows: masses and inertias (density), friction (the pair table's per-world rows, ``shape_material_mu``,
MuJoCo's pair rows), cloth and joint stiffness (youngs). A fine-pack slot holds the draws of the world it was built as
(worlds 0..n-1), so before S1b a world whose rows differ from its slot's was refused. ``WorldParams.move(e, s)``:
1. writes world ``e``'s rows of every per-world float Model array (start state left out) into slot ``s``, and its rows
   of the pair table's per-world tables and of ``built.randomized`` (readback names ``e``'s values);
2. tells the solvers (``notify_model_changed``, the flags of the frequencies that changed): Newton's coupled solver
   re-projects the entry views and each entry solver rebuilds what it derives (MuJoCo's per-world model fields);
3. refreshes what the coupler baked at build: every endpoint set's proxy masses (``meff``, ``m_axis``, ``touches``) take
   world ``e``'s rows from the source's sets, the stiffness caps re-read them, and the pair masses (the clock bound's
   and strong-pair class's input, a minimum over the pack's proxies) are taken again from the sets.
The slot's own rows are kept from the first move (``restore``). A move of rows that already agree does nothing, so the
way back into the coarse pack (whose world ``e`` rows never changed) costs no notification.

Refused, by name (``refusal``; the caller re-runs as today), never moved silently:
- parameters of particles and their elements (a randomized cloth or soft body: density, youngs): the entry views'
  copies and VBD's own rows are not re-projected by any notification;
- a contact gate: its per-proxy bounds are baked from the proxy masses at build with no refresh
  (contact_gate.py ``_rows``; the hook it needs: ``ContactGate.refresh_bounds()``, WP4 / WP10);
- a per-world Model array that is not a float array, or the pair table's per-world tables on one side only.
Not here: the watchdog's per-world scales (e_scale, z_ref, m_tot, l_char), moved with the world's state by the
transfer itself (fine_pack / fine_rerun). Laws: tests/coupling/test_world_clock_params.py.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.coupling.fine_pack import START, TransferError, _model_arrays, item_world
from alloy.coupling.stiffness_cap import bind_axis_masses

log = logging.getLogger("alloy.coupling")

_F = newton.ModelFlags
FLAGS = {"BODY": int(_F.BODY_PROPERTIES | _F.BODY_INERTIAL_PROPERTIES), "SHAPE": int(_F.SHAPE_PROPERTIES),
         "JOINT": int(_F.JOINT_PROPERTIES), "JOINT_DOF": int(_F.JOINT_DOF_PROPERTIES),
         "JOINT_COORD": int(_F.JOINT_PROPERTIES)}       # a frequency not named here notifies everything
PARTICLE = ("PARTICLE", "TRIANGLE", "EDGE", "TETRAHEDRON", "SPRING")     # refused: no notification re-projects them
TABLES = ("world_table", "world_stiffness")             # PairTable's per-world rows ([W, n, n, ...])
HOME = "_world_params_home"                              # on the destination simulation: {slot: its own rows}


def _frequency(model: Any, name: str) -> Any:
    try:
        return model.get_attribute_frequency(name)
    except (KeyError, AttributeError):
        return None


class _Row:
    """One per-world Model array on both sides, with each side's item -> world map (host)."""

    def __init__(self, name: str, freq: Any, a: wp.array, b: wp.array, wa: np.ndarray, wb: np.ndarray) -> None:
        self.name, self.freq, self.a, self.b, self.wa, self.wb = name, freq, a, b, wa, wb

    @property
    def kind(self) -> str:
        return str(getattr(self.freq, "name", self.freq))

    def rows(self, e: int, s: int) -> tuple[np.ndarray, np.ndarray]:
        return np.flatnonzero(self.wa == e), np.flatnonzero(self.wb == s)

    def differs(self, e: int, s: int) -> bool:
        ia, ib = self.rows(e, s)
        ra, rb = self.a.numpy()[ia], self.b.numpy()[ib]
        return ra.shape != rb.shape or not np.array_equal(ra, rb, equal_nan=True)


def _rows(src: Any, dst: Any) -> tuple[list[_Row], list[str]]:
    """The per-world float Model arrays both sides hold (``_Row``), and the names of per-world arrays of another
    dtype (never moved: a difference there refuses)."""
    other, out, odd = dict(_model_arrays(dst)), [], []
    for name, arr in _model_arrays(src):
        b = other.get(name)
        freq = _frequency(src, name)
        if name in START or arr.size == 0 or b is None or freq is None:
            continue
        ma, mb = item_world(src, freq), item_world(dst, freq)
        if ma is None or mb is None or ma.shape[0] != arr.shape[0] or mb.shape[0] != b.shape[0]:
            continue
        row = _Row(name, freq, arr, b, ma.numpy(), mb.numpy())
        if arr.numpy().dtype.kind == "f":
            out.append(row)
        else:
            odd.append(name)
    return out, odd


class WorldParams:
    """World ``e``'s randomized parameters of ``src`` into slot ``s`` of ``dst`` (module docstring). Both are
    simulations (``Sim.engine``) of the same scene."""

    def __init__(self, src: Any, dst: Any) -> None:
        self.src, self.dst = src, dst
        self.rows, self.odd = _rows(src.model, dst.model)
        self.tables = self._tables()

    def _tables(self) -> list[tuple[str, wp.array, wp.array]] | str:
        ta = getattr(getattr(self.src, "built", None), "pair_table", None)
        tb = getattr(getattr(self.dst, "built", None), "pair_table", None)
        out = []
        for name in TABLES:
            a, b = getattr(ta, name, None), getattr(tb, name, None)
            if (a is None) != (b is None):
                return f"the pair table's per-world rows ({name}) are on one simulation only"
            if a is not None:
                out.append((name, a, b))
        return out

    # ------------------------------------------------------------------ what differs, what cannot move
    def differing(self, e: int, s: int) -> list[str]:
        """The per-world parameter arrays whose world ``e`` rows differ from slot ``s``'s (Model arrays by name, the
        pair table's as ``pair_table.<name>``)."""
        out = [r.name for r in self.rows if r.differs(e, s)]
        if not isinstance(self.tables, str):
            out += [f"pair_table.{n}" for n, a, b in self.tables if not np.array_equal(a.numpy()[e], b.numpy()[s])]
        return out

    def refusal(self, e: int, s: int) -> str | None:
        """Why world ``e``'s parameters cannot move into slot ``s`` (module docstring), or None; no side effects."""
        if isinstance(self.tables, str):
            return self.tables
        names = set(self.differing(e, s))
        soft = sorted(r.name for r in self.rows if r.name in names and r.kind in PARTICLE)
        if soft:
            return (f"randomized particle parameters {soft[:6]} (a cloth's or soft body's density or stiffness: the "
                    "entry views and VBD keep their build copies)")
        if names and getattr(self.src, "pair_mu", None) is not None:
            return ("randomized or pair friction on a cloth or soft body (each simulation keeps its own per-world "
                    "contact friction: alloy.solvers.vbd.pair_mu)")
        if names and getattr(getattr(self.dst, "stepper", None), "_gate", None) is not None:
            return "randomized parameters with the contact gate (its per-proxy bounds are baked at build)"
        return None

    def check(self, e: int, s: int) -> None:
        why = self.refusal(e, s)
        if why is not None:
            raise TransferError(f"world {e} -> slot {s}: {why}")

    # ------------------------------------------------------------------ the move
    def move(self, e: int, s: int) -> list[str]:
        """Write world ``e``'s parameters into slot ``s`` and refresh what is derived from them; the names moved
        (empty, and nothing touched, when they already agree)."""
        self.check(e, s)
        names = self.differing(e, s)
        if not names:
            return names
        self._keep_home(s)
        flags = 0
        for r in self.rows:
            if r.name in names:
                ia, ib = r.rows(e, s)
                host = r.b.numpy().copy()
                host[ib] = r.a.numpy()[ia]
                r.b.assign(host)
                flags |= FLAGS.get(r.kind, int(_F.ALL))
        for name, a, b in self.tables:
            host = b.numpy().copy()
            host[s] = a.numpy()[e]
            b.assign(host)
        self._readback(e, s)
        refresh(self.dst, flags, like=self.src, world=e, slot=s)
        log.debug("world %d -> slot %d: moved parameters %s", e, s, names)
        return names

    def _readback(self, e: int, s: int) -> None:
        """``built.randomized`` (the drawn values, read back by name) names world ``e``'s draws in slot ``s``."""
        ra = getattr(getattr(self.src, "built", None), "randomized", None) or {}
        rb = getattr(getattr(self.dst, "built", None), "randomized", None) or {}
        for key, vals in ra.items():
            if key in rb:
                rb[key][s] = vals[e]

    # ------------------------------------------------------------------ the slot's own parameters
    def _keep_home(self, s: int) -> None:
        home = self.dst.__dict__.setdefault(HOME, {})
        if s in home:
            return
        kept: dict[str, np.ndarray] = {r.name: r.b.numpy()[r.wb == s].copy() for r in self.rows}
        kept.update({f"pair_table.{n}": b.numpy()[s].copy() for n, a, b in self.tables})
        built = getattr(self.dst, "built", None)
        kept.update({("randomized", k): float(v[s]) for k, v in (getattr(built, "randomized", None) or {}).items()})
        home[s] = kept

    def restore(self, s: int) -> list[str]:
        """Slot ``s`` of the destination back to the parameters it was built with (bit for bit); the names
        rewritten."""
        kept = getattr(self.dst, HOME, {}).pop(s, None)
        if kept is None:
            return []
        names, flags = [], 0
        for r in self.rows:
            host = r.b.numpy().copy()
            if not np.array_equal(host[r.wb == s], kept[r.name], equal_nan=True):
                host[r.wb == s] = kept[r.name]
                r.b.assign(host)
                names.append(r.name)
                flags |= FLAGS.get(r.kind, int(_F.ALL))
        for name, _, b in self.tables:
            host = b.numpy().copy()
            host[s] = kept[f"pair_table.{name}"]
            b.assign(host)
        rb = getattr(getattr(self.dst, "built", None), "randomized", None) or {}
        for key in rb:
            rb[key][s] = kept.get(("randomized", key), rb[key][s])
        if names:
            refresh(self.dst, flags)
        return names


def refresh(sim: Any, flags: int, like: Any = None, world: int = -1, slot: int = -1) -> None:
    """After ``sim``'s per-world Model rows changed: notify its solvers, then the coupler's build-time copies. With
    ``like`` (the simulation world ``world`` came from) slot ``slot``'s proxy rows are that world's, bit for bit;
    without it they are read again from the destination views."""
    sched = getattr(sim, "stepper", None)
    if flags:      # through the coupler when there is one: it re-projects its entry views before the solvers read them
        coupled = getattr(sched, "notify_model_changed", None) if hasattr(sched, "_entries") else None
        (coupled or sim._notify)(int(flags))
    sets = getattr(sched, "_sets", None)
    if not sets:
        return
    theirs = {(es.src_name, es.dst_name, es.kind): es for es in getattr(getattr(like, "stepper", None), "_sets", [])}
    for es in sets:
        src = theirs.get((es.src_name, es.dst_name, es.kind))
        if src is not None:
            _proxy_rows(es, src, world, slot)
        else:
            _reread(sched, es)
    bind_axis_masses(sched._caps, sets)
    sched._pair_mass = sched._pair_masses()


WATCH_ROWS = ("e_scale", "z_ref", "m_tot", "l_char")   # the energy watchdog's per-world reference rows


def move_watch_reference(src: Any, dst: Any, world: int, slot: int) -> None:
    """The energy watchdog's reference (E_scale, datum, mass, extent: set from each world's own start state and kept
    outside ``state_arrays``) of ``src``'s world ``world`` goes into ``dst``'s slot ``slot``; without it the moved world
    is judged on the slot's scale (P1, p1_watchdog.md); with them goes the world's column of the 1 s window ring. The ring's
    head and the elapsed time belong to the pack, not to a world: the caller sets them (the fine re-run's does)."""
    ws, wd = getattr(getattr(src, "stepper", None), "_watchdog", None), getattr(getattr(dst, "stepper", None),
                                                                                "_watchdog", None)
    if ws is None or wd is None:
        return
    for name in WATCH_ROWS:
        host = getattr(wd, name).numpy().copy()
        host[slot] = getattr(ws, name).numpy()[world]
        getattr(wd, name).assign(host)
    if ws.ring is None:
        return
    if wd.ring is None or wd.ring.shape[0] != ws.ring.shape[0]:     # as long as the pack's window, one column per slot
        wd.ring = wp.zeros((ws.ring.shape[0], wd.world_count), dtype=wp.float64, device=wd.device)
    ring = wd.ring.numpy().copy()
    ring[:, slot] = ws.ring.numpy()[:, world]
    wd.ring.assign(ring)


def _proxy_rows(es: Any, src: Any, world: int, slot: int) -> None:
    """Endpoint set ``es``'s proxies of ``slot`` take ``src``'s rows of ``world``: meff, m_axis, touches."""
    ia, ib = np.flatnonzero(src.world.numpy() == world), np.flatnonzero(es.world.numpy() == slot)
    if ia.shape != ib.shape:
        raise TransferError(f"endpoint set {es.src_name}->{es.dst_name}: world {world} has {ia.size} proxies, slot "
                            f"{slot} has {ib.size}")
    meff = es.meff.numpy().copy()
    meff[ib] = src.meff.numpy()[ia]
    es.meff.assign(meff)
    for name in ("m_axis", "touches"):
        a, b = getattr(src, name), getattr(es, name)
        if a is not None and b is not None:
            b[ib] = a[ia]


def _reread(sched: Any, es: Any) -> None:
    """Endpoint set ``es``'s proxy masses read again from its destination view, as ``build_endpoint_set`` does."""
    from alloy.coupling.axis_mass import axis_mass_ratio  # noqa: PLC0415

    view = sched._entries[es.dst_name].view
    mass = (view.body_mass if es.kind == "body" else view.particle_mass).numpy()
    meff = mass[es.proxy_local.numpy()].astype(np.float32)
    es.meff.assign(meff)
    ratio = axis_mass_ratio(sched._entries[es.src_name].solver) if es.kind == "body" else None
    if es.m_axis is not None:
        es.m_axis[:] = meff.astype(np.float64) * (ratio[es.src_local.numpy()] if ratio is not None else 1.0)


__all__ = ["FLAGS", "HOME", "PARTICLE", "WATCH_ROWS", "WorldParams", "move_watch_reference", "refresh"]
