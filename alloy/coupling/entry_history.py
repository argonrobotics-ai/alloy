# SPDX-License-Identifier: Apache-2.0
"""Implicit MPM's per-particle history survives a coupled exchange (engine item 2, CPL-3).

Newton steps a coupled entry that is not ``in_place`` from ``state_0`` into ``state_1``. Every exchange it copies the
parent's core arrays (bodies, particles, joints) into ``state_0`` again (``_copy_state_to_entry``), and nothing copies
``state_1``'s namespaced arrays back. For SolverImplicitMPM those namespaces hold the material history: elastic strain,
plastic Jp, stress (``state.mpm.*``). So each exchange restarted the sand from the history of the first one.

Under a 28 g cup, that history is the bed's elastic settlement, about 0.6 µm. The sand gave it back and took it again
every exchange: the cup sank at 0.6 µm per exchange, 0.26 / 0.50 / 1.0 mm/s at E 2e7 / 1e7 / 5e6 Pa and 0.98 mm/s at
half the exchange interval. Material strength did not change it (φ 45°, 200 Pa cohesion), and neither did the solve
(1000 iterations at 1e-6, residual strain carried). A bare bed with no coupler kept its history and stood still.

``EntryHistory.commit`` copies the namespaced arrays of MPM entries from ``state_1`` into ``state_0`` once an exchange
is accepted (``SchedulerMixin.step``, after the coupled step). Restart passes inside an exchange still read
``state_0``, which is the exchange-start history, so lagged and gated passes stay repeatable. The copies are device
``wp.copy`` calls and are graph-capturable. In-place entries need nothing. ``snapshot_arrays`` puts that history
into the scheduler's ``state_arrays`` (with per-particle world maps), so a rollback or a latched world restores it with
the period start instead of keeping the discarded attempt's.

Other solvers with namespaced state have the same staleness, but they are left out: the ``sph`` namespace, for one,
belongs to WP9.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import warp as wp

from alloy._vendor import newton

log = logging.getLogger("alloy.coupling")

HISTORY_SOLVERS = (newton.solvers.SolverImplicitMPM,)


def _namespace_items(src: Any, dst: Any) -> list[tuple[str, wp.array, wp.array]]:
    """("<namespace>.<array>", src, dst) for every namespaced array both states carry, same name and shape."""
    out = []
    for ns_name, ns_src in vars(src).items():
        ns_dst = getattr(dst, ns_name, None)
        if not isinstance(ns_src, newton.Model.AttributeNamespace) or ns_dst is None:
            continue
        for name, a_src in vars(ns_src).items():
            a_dst = getattr(ns_dst, name, None)
            if isinstance(a_src, wp.array) and isinstance(a_dst, wp.array) and a_src.shape == a_dst.shape \
                    and a_src.ptr != a_dst.ptr:
                out.append((f"{ns_name}.{name}", a_src, a_dst))
    return out


def _namespace_pairs(src: Any, dst: Any) -> list[tuple[wp.array, wp.array]]:
    """(src, dst) array pairs of ``_namespace_items``."""
    return [(a_src, a_dst) for _, a_src, a_dst in _namespace_items(src, dst)]


class EntryHistory:
    """Copies MPM entries' namespaced history ``state_1 -> state_0`` after each accepted exchange (module docstring).
    The array pairs are cached per (state_0, state_1) identity and rebuilt if an entry's states are reallocated."""

    def __init__(self) -> None:
        self._key: tuple[int, ...] = ()
        self._pairs: list[tuple[wp.array, wp.array]] = []

    def _entries(self, entries: Mapping[str, Any]) -> list[Any]:
        return [e for e in entries.values() if not e.in_place and e.state_1 is not None and e.state_1 is not e.state_0
                and isinstance(e.solver, HISTORY_SOLVERS)]

    def commit(self, entries: Mapping[str, Any]) -> None:
        targets = self._entries(entries)
        key = tuple(x for e in targets for x in (id(e.state_0), id(e.state_1)))
        if key != self._key:
            self._key = key
            self._pairs = [p for e in targets for p in _namespace_pairs(e.state_1, e.state_0)]
            if self._pairs:
                log.info("coupled entries %s: %d history arrays carried state_1 -> state_0 per exchange",
                         [e.name for e in targets], len(self._pairs))
        for src, dst in self._pairs:
            wp.copy(dst, src)

    def snapshot_arrays(self, entries: Mapping[str, Any]) -> dict[str, wp.array]:
        """The MPM entries' history in ``state_0`` (what the next exchange starts from), for the scheduler's
        ``state_arrays``: a rollback or a latched world restores it with the rest of the period start. Per-particle
        arrays only, keyed ``<entry>.state0.<namespace>.<array>``."""
        return {k: a for k, (a, _) in self._snapshot(entries).items()}

    def snapshot_world_maps(self, entries: Mapping[str, Any]) -> dict[str, wp.array]:
        """Item -> world maps for ``snapshot_arrays`` (the entry view's ``particle_world``)."""
        return {k: w for k, (_, w) in self._snapshot(entries).items()}

    def _snapshot(self, entries: Mapping[str, Any]) -> dict[str, tuple[wp.array, wp.array]]:
        """The arrays ``commit`` carries (so the two cannot drift apart) with a leading axis of the entry's particle
        count, which are taken as per-particle (world map ``particle_world``)."""
        out = {}
        for e in self._entries(entries):
            view, world = e.view, getattr(e.view, "particle_world", None)
            n = int(getattr(view, "particle_count", 0) or 0)
            if world is None or n == 0:
                continue
            for key, _, arr in _namespace_items(e.state_1, e.state_0):
                if arr.ndim >= 1 and arr.shape[0] == n:
                    out[f"{e.name}.state0.{key}"] = (arr, world)
        return out
