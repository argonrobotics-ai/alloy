# SPDX-License-Identifier: Apache-2.0
"""``sim.reset()`` puts the coupled exchange clock back (WP4 red fix).

A Scheduler adapts its exchange clock while it runs: the level j (dt_x = control_dt / 2^j), each entry's substep rung,
their hysteresis counts, the sticky escalation floor and mode, the contact band's bound, the contact gate's arming
rows, the energy watchdog's window. ``reset`` puts the state arrays back but not this host state, so the run after a
reset started on the adapted clock and differed from the first run (a 3.4 mm difference in the water of a cloth +
liquid scene). ``ClockMemoryMixin`` takes a ``ClockMemory`` of the clock as the first period finds it (and again at
``sim.settle``, whose settled state is the reset target) and puts it back in ``reset_worlds``; the first period after
a reset of every world also reads the start's last report, as period 0 of a fresh build reads None.

- every world reset (``worlds is None``): everything in the memory goes back, then the entries' substeps.
- some worlds reset (a mask): ONLY the per-world rows of those worlds go back (the ladder's and onset's cooldown
  rows, the contact gate's rows of their proxies, the watchdog ring's columns). The shared level, rungs, floor, mode
  and band bound stay: one clock serves every world and the other worlds are still running on it.

Capture only reads, so a simulation that never resets is unchanged. Device arrays go back in place (``assign``): the
captured graphs and kernels hold their pointers. The host lists are those of ``fine_rerun`` (P1 mirrors the same
state into its fine pack), reused from there.
Laws: tests/coupling/test_reset_clock.py.
"""

from __future__ import annotations

import copy
import dataclasses
import types
from typing import Any

import numpy as np
import warp as wp

from alloy.coupling.fine_rerun import HOST, ROWS, SCALARS, _assign, _sticky_state
from alloy.coupling.match_reset import world_bools

OBJECTS = (*HOST, "_ladder", "_onset")        # host objects captured whole (their device arrays stay put)
EXTRA = ("_onset_last", "_rerun", "_short_rerun", "_gate_rerun", "_census_totals", "_census_kept", "_census_attempt",
         "_census_counted", "_census_used", "_census_last", "_census_warned")
STICKY = ("_trips", "_rho_raises", "_coupling")
GATE_DEVICE = ("armed", "touched", "near", "_lo", "_hi", "_vmax", "_count")   # first axis: proxy, or (dest, world)
GATE_PROXY_ROWS = ("armed", "touched", "near")    # indexed by proxy; the others by (destination, world)
SKIP_PASSES = ("world_count", "device")
SKIP_CONFIG = ("policy",)                     # configuration, not clock state: the live object stays


@dataclasses.dataclass
class ClockMemory:
    objects: dict[str, Any]                 # name -> namespace of the object's host attributes
    scalars: dict[str, Any]
    sticky: Any                             # shaped like a stepper for ``_sticky_state``
    passes: Any | None
    gate: dict[str, Any] | None
    band: dict[str, Any]
    watch: dict[str, Any]
    substeps: dict[str, int]
    last_report: Any


def _host(obj: Any, skip: tuple[str, ...] = ()) -> types.SimpleNamespace:
    """Deep copies of ``obj``'s host attributes (device arrays are not copied)."""
    return types.SimpleNamespace(**{k: copy.deepcopy(v) for k, v in vars(obj).items()
                                    if k not in skip and not isinstance(v, wp.array)})


def _sticky(sched: Any) -> types.SimpleNamespace:
    """The sticky ladder's pack-wide state in the shape ``fine_rerun._sticky_state`` reads."""
    names = [n for n in STICKY if hasattr(sched, n) and (n != "_coupling" or hasattr(sched, "_rho_raises"))]
    ns = types.SimpleNamespace(**{n: copy.deepcopy(getattr(sched, n)) for n in names})   # ADMM's config, as P1
    if hasattr(sched, "_strong"):
        ns._strong = set(sched._strong)
    ns._sets = [types.SimpleNamespace(staggered=s.staggered, strong=s.strong,
                                      mapping=types.SimpleNamespace(mode=s.mapping.mode))
                for s in getattr(sched, "_sets", None) or ()]
    loop = getattr(sched, "_loop", None)
    ns._loop = None if loop is None else types.SimpleNamespace(cfg=loop.cfg)
    return ns


def _gate_rows(gate: Any) -> dict[str, Any]:
    out = {"armed_host": gate.armed_host.copy(), "hold": gate.hold.copy(), "seen_state": gate.seen_state,
           "reruns": gate.reruns}
    out.update({n: getattr(gate, n).numpy().copy() for n in GATE_DEVICE})
    return out


def _band_rows(band: Any) -> dict[str, Any]:
    return {"last_speed": band.last_speed.copy(), "last_bound": band.last_bound}


def _watch_rows(wd: Any) -> dict[str, Any]:
    return {"ring": None if wd.ring is None else wd.ring.numpy().copy(), "head": wd.head,
            "elapsed_ns": wd.elapsed_ns.numpy().copy()}


def capture(sched: Any) -> ClockMemory:
    """Deep host copies of everything that decides the clock (module docstring); reads only."""
    cp = sched._captured_passes()
    gate = getattr(sched, "_gate", None)
    scalars = {n: copy.deepcopy(getattr(sched, n)) for n in (*SCALARS, *EXTRA) if hasattr(sched, n)}
    return ClockMemory(
        objects={n: _host(getattr(sched, n), SKIP_CONFIG) for n in OBJECTS}, scalars=scalars, sticky=_sticky(sched),
        passes=None if cp is None else _host(cp[0], SKIP_PASSES),
        gate=None if gate is None else _gate_rows(gate), band=_band_rows(sched._band),
        watch=_watch_rows(sched._watchdog), substeps=dict(sched._substeps_now),
        last_report=getattr(sched, "_last_report", None))


# ------------------------------------------------------------------------------------------------ every world
def _restore_gate(gate: Any, rows: dict[str, Any]) -> None:
    gate.armed_host, gate.hold = rows["armed_host"].copy(), rows["hold"].copy()
    gate.seen_state, gate.reruns = rows["seen_state"], rows["reruns"]
    for n in GATE_DEVICE:
        getattr(gate, n).assign(rows[n])


def _restore_watch(wd: Any, rows: dict[str, Any]) -> None:
    """The window ring in place when it exists with the saved shape, else dropped (``ring_for`` rebuilds it)."""
    ring = rows["ring"]
    if ring is not None and wd.ring is not None and tuple(wd.ring.shape) == ring.shape:
        wd.ring.assign(ring)
    else:
        wd.ring = None if ring is None else wp.array(ring, dtype=wp.float64, device=wd.device)
    wd.head = rows["head"]
    wd.elapsed_ns.assign(rows["elapsed_ns"])


def _restore_all(sched: Any, m: ClockMemory) -> None:
    for n, ns in m.objects.items():
        _assign(getattr(sched, n), ns)
    for n, v in m.scalars.items():
        setattr(sched, n, copy.deepcopy(v))
    for n in (*SCALARS, *EXTRA):                       # made during the run (read with a default): gone again
        if n not in m.scalars and n in vars(sched):
            delattr(sched, n)
    sched._last_report = m.last_report
    cp = sched._captured_passes()
    if cp is not None and m.passes is not None:
        _assign(cp[0], m.passes)
    _sticky_state(sched, copy.deepcopy(m.sticky))
    gate = getattr(sched, "_gate", None)
    if gate is not None and m.gate is not None:
        _restore_gate(gate, m.gate)
    sched._band.last_speed, sched._band.last_bound = m.band["last_speed"].copy(), m.band["last_bound"]
    sched._band.vmax.zero_()
    _restore_watch(sched._watchdog, m.watch)
    sched._apply_substeps(dict(m.substeps))


# ------------------------------------------------------------------------------------------------ a mask
def _restore_row(cur: np.ndarray, saved: Any, default: Any, worlds: np.ndarray) -> None:
    """``cur[worlds]`` from the saved row of the same length, else the row's default (a lazily sized row)."""
    saved = np.asarray(saved)
    cur[worlds[: cur.size]] = saved[worlds[: cur.size]] if saved.shape == cur.shape else default


def _restore_rows(sched: Any, m: ClockMemory, worlds: np.ndarray) -> None:
    for name, rows in ROWS.items():
        obj, saved = getattr(sched, name), m.objects[name]
        for attr, default in {"last": rows["last"], "per_world": rows.get("per_world", 0)}.items():
            cur = getattr(obj, attr, None)
            if isinstance(cur, np.ndarray) and cur.size and hasattr(saved, attr):
                _restore_row(cur, getattr(saved, attr), default, worlds)


def _restore_gate_worlds(gate: Any, rows: dict[str, Any], worlds: np.ndarray) -> None:
    proxy = worlds[np.asarray(gate.ends.world)] if gate.count else np.zeros(0, dtype=bool)
    gate.armed_host[proxy], gate.hold[proxy] = rows["armed_host"][proxy], rows["hold"][proxy]
    for n in GATE_DEVICE:
        arr = getattr(gate, n)
        cur = arr.numpy().copy()
        if n in GATE_PROXY_ROWS:
            cur[proxy[: cur.shape[0]]] = rows[n][proxy[: cur.shape[0]]]
        else:
            cur[:, worlds] = rows[n][:, worlds]
        arr.assign(cur)


def _restore_ring_worlds(wd: Any, rows: dict[str, Any], worlds: np.ndarray) -> None:
    if wd.ring is None:
        return
    ring, saved = wd.ring.numpy().copy(), rows["ring"]
    ring[:, worlds] = saved[:, worlds] if saved is not None and saved.shape == ring.shape else 0.0
    wd.ring.assign(ring)


def _restore_worlds(sched: Any, m: ClockMemory, worlds: np.ndarray) -> None:
    _restore_rows(sched, m, worlds)
    gate = getattr(sched, "_gate", None)
    if gate is not None and m.gate is not None:
        _restore_gate_worlds(gate, m.gate, worlds)
    _restore_ring_worlds(sched._watchdog, m.watch, worlds)


def restore(sched: Any, memory: ClockMemory, worlds: Any = None) -> None:
    """Put the clock back: every captured item when ``worlds`` is None, else only the per-world rows of the worlds in
    the mask (the shared level, rungs, floor, mode and band bound stay: the other worlds run on them)."""
    mask = world_bools(worlds)
    if mask is not None:
        mask = mask[: int(sched._watchdog.world_count)]      # Newton's mask may end in a global entry
    if mask is None:
        _restore_all(sched, memory)
    elif mask.any():
        _restore_worlds(sched, memory, mask)


class ClockMemoryMixin:
    """For the Scheduler, FIRST in its bases (its ``choose_clock``, ``step`` and ``reset_worlds`` wrap the others).

    The memory is the clock as the first period finds it: taken at each ``choose_clock`` until the first exchange is
    stepped (a simulation makes one ``choose_clock`` when it is built and the wiring sets flags after that, so the
    last one before an exchange is the start's). ``alloy.Sim.settle`` takes it again (``mark_start``). A reset before
    any exchange puts nothing back."""

    _start_memory: ClockMemory | None = None
    _exchanged = False           # an exchange has been stepped since this stepper was built
    _restarted = False           # every world was reset and no period has started since

    def mark_start(self) -> None:
        """The clock as it stands is the one ``reset`` returns to."""
        self._start_memory = capture(self)

    def choose_clock(self, control_dt: float, last: Any) -> tuple[float, dict[str, int]]:
        if not self._exchanged:
            self.mark_start()
        elif self._restarted:    # the first period after a reset of every world reads the start's last report (None
            last = self._start_memory.last_report                           # after a build), not the old run's
        self._restarted = False
        return super().choose_clock(control_dt, last)

    def step(self, *args: Any, **kwargs: Any) -> None:
        self._exchanged = True
        super().step(*args, **kwargs)

    def reset_worlds(self, world_mask: Any = None) -> None:
        super().reset_worlds(world_mask)
        if self._start_memory is not None and self._exchanged:
            restore(self, self._start_memory, world_mask)
            self._restarted = world_bools(world_mask) is None


__all__ = ["ClockMemory", "ClockMemoryMixin", "capture", "restore"]
