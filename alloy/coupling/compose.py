# SPDX-License-Identifier: Apache-2.0
"""Coupled stepper composition with one-way followers (SCR-DC5-6/7/8): what ``make_stepper`` calls for a D5 choice.

- no one-way pair: ``Scheduler`` (proxy) or ``SchedulerADMM`` (admm), exactly as before;
- one hub and one one-way destination (two entries, e.g. MuJoCo -> SPH): the two-entry ``Scheduler`` with the one-way
  pair, one pass per exchange and no feedback (SCR-DC5-8);
- a hub with one two-way partner plus one-way destinations (DC5: MuJoCo <-> VBD and MuJoCo -> SPH): the two-entry
  ``Scheduler`` over the two-way pair, wrapped in a ``OneWayFollower`` whose leaves are the one-way destinations,
  built on the full model;
- two-way pairs over more than two entries (engine item 3 P3: MuJoCo -> VBD lagged, VBD -> SPH staggered, plus the
  one-way MuJoCo -> SPH): one ``Scheduler`` over every entry and pair. Newton's proxy coupler steps each entry once
  per pass in topological order of the acyclic source -> destination graph (WP11's N-entry change; before it, a
  coupler takes at most two entries and this raises);
- refused (CouplingError): the above on a Newton without N-entry proxy coupling, a one-way destination that owns
  bodies or joints, ADMM with one-way pairs, several one-way destinations with no two-way partner.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from alloy.core.errors import CouplingError
from alloy.coupling.one_way import OneWayFollower, OneWayLeaf, validate_one_way
from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy

log = logging.getLogger("alloy.coupling")


def proxy_dag_supported() -> bool:
    """True when Newton's ``SolverCoupledProxy`` couples more than two entries over an acyclic proxy graph (WP11's
    coupled-proxy-dag change: ``_build_proxy_entry_plan``)."""
    from alloy.core.newton_compat import SolverCoupledProxy

    return hasattr(SolverCoupledProxy, "_build_proxy_entry_plan")


def _require_dag(entries: Sequence[AlloyEntry], pairs: Sequence[PairPolicy]) -> None:
    if not proxy_dag_supported():
        raise CouplingError(f"two-way pairs {_routes(pairs)} over {len(entries)} entries need Newton's N-entry proxy "
                            "coupling (WP11's coupled-proxy-dag, in the Newton pin from 11-07); this Newton couples "
                            "at most two entries")


def build_coupled_stepper(model: Any, entries: Sequence[AlloyEntry], pairs: Sequence[PairPolicy],
                          policy: SchedulerPolicy = SchedulerPolicy(), *, kind: str = "proxy") -> Any:
    """The stepper for ``kind`` ("proxy" | "admm") over ``entries`` and ``pairs`` (module docstring)."""
    from alloy.coupling.scheduler import Scheduler, SchedulerADMM

    one_way = [p for p in pairs if p.one_way]
    if not one_way:
        if kind != "admm" and len(entries) > 2:
            _require_dag(entries, pairs)
        cls = SchedulerADMM if kind == "admm" else Scheduler
        return cls(model, list(entries), list(pairs), policy)
    if kind != "proxy":
        raise CouplingError(f"one-way pairs {_routes(one_way)} need the proxy coupler (kind {kind!r} has no one-way "
                            "stage)")
    two_way = [p for p in pairs if not p.one_way]
    coupled = {name for p in two_way for name in (p.source, p.destination)}
    chained = any(p.destination in coupled for p in one_way)      # a one-way pair into a two-way entry: not a leaf
    if len(coupled) > 2 or chained:   # inside one Scheduler a one-way pair is a proxy whose feedback stays zero
        _require_dag(entries, pairs)
        log.info("coupled stepper: one Scheduler over %s, one-way %s (N-entry proxy graph)", _routes(two_way),
                 _routes(one_way))
        return Scheduler(model, list(entries), list(pairs), policy)
    validate_one_way(pairs)
    if not two_way:
        if len(one_way) == 1 and len(entries) == 2:
            return Scheduler(model, list(entries), list(pairs), policy)          # one pass per exchange (DC5-8)
        raise CouplingError(f"one-way pairs {_routes(one_way)} without a two-way pair: only one hub and one one-way "
                            "destination are supported")
    by_name = {ae.entry.name: ae for ae in entries}
    base = Scheduler(model, [by_name[n] for n in by_name if n in coupled], two_way, policy)
    leaves = [_leaf(by_name[p.destination], model) for p in one_way]
    log.info("coupled stepper: Scheduler over %s plus one-way followers %s", _routes(two_way), _routes(one_way))
    return OneWayFollower(base, model, leaves, policy.clock)


def _leaf(ae: AlloyEntry, model: Any) -> OneWayLeaf:
    """A one-way destination entry as a follower leaf: its solver built on the full model, its particles only."""
    entry = ae.entry
    if list(entry.bodies or ()) or list(entry.joints or ()):
        raise CouplingError(f"one-way destination {entry.name!r} owns bodies or joints; a follower writes back "
                            "particles only")
    return OneWayLeaf(entry.name, entry.solver(model), list(entry.particles), rung=max(int(ae.initial_rung), 1),
                      adaptive=bool(ae.adaptive))


def _routes(pairs: Sequence[PairPolicy]) -> list[tuple[str, str]]:
    return [(p.source, p.destination) for p in pairs]
