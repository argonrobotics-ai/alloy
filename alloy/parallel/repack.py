# SPDX-License-Identifier: Apache-2.0
"""Repacking at reset boundaries (SPEC §3.9 ``WorldPacker.repack``, parallel §4.5).

A pack runs in lockstep, so it pays for its slowest and most-awake worlds. At a reset boundary every world of a
group (same scene, solver config, dt) is put in one of three lanes and the lanes are packed separately:

- ``slow``: per-world iterations p99 above ``slow_lane_iters_ratio`` x the pack median (needs
  ``PackStats.world_iters_p99``; skipped when the stats do not carry it);
- ``awake``: awake fraction >= ``AWAKE_LANE_FRACTION`` over the window;
- ``asleep``: the rest.

Lane packs keep the group's largest original pack size as their cap and the device that held most of their worlds.
World ids stay global, so a repacked pack's ids are generally not consecutive; build passes them as
``assemble(world_ids=...)`` (SCR-WP8-2).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from alloy.parallel import planning as pl
from alloy.parallel.types import Pack, PackPlan, PackPolicy, PackSpec, PackStats, WorldProfile

log = logging.getLogger("alloy.parallel")

AWAKE_LANE_FRACTION = 0.5
LANES = ("slow", "awake", "asleep")


@dataclass
class _Group:
    profile: WorldProfile
    n_cap: int = 0
    lanes: dict[str, list[tuple[int, str]]] = field(default_factory=lambda: {k: [] for k in LANES})  # (id, device)


def lane_of_worlds(spec: PackSpec, st: PackStats, policy: PackPolicy) -> list[str]:
    """Lane per world of one pack."""
    awake = np.asarray(st.awake_fraction, dtype=np.float64)
    if awake.shape != (spec.n,):
        raise ValueError(f"PackStats.awake_fraction for pack {st.pack} has shape {awake.shape}, pack has {spec.n}")
    lanes = np.where(awake >= AWAKE_LANE_FRACTION, "awake", "asleep").astype(object)
    iters = getattr(st, "world_iters_p99", None)
    if iters is not None:
        iters = np.asarray(iters, dtype=np.float64)
        median = float(np.median(iters))
        if median > 0.0:
            lanes[iters > policy.slow_lane_iters_ratio * median] = "slow"
    return [str(x) for x in lanes]


def repack_plan(packs: Sequence[Pack], stats: Sequence[PackStats],
                profiles: Mapping[int, WorldProfile | None], policy: PackPolicy) -> PackPlan:
    """Lane-split plan for the next episode; the caller rebuilds packs from it at the reset boundary."""
    by_pack = {st.pack: st for st in stats}
    groups: dict[tuple, _Group] = {}
    for i, pack in enumerate(packs):
        st = by_pack.get(i)
        prof = profiles.get(pack.spec.world_ids[0])
        if st is None or prof is None:
            raise ValueError(f"repack needs stats and a planned profile for every pack; pack {i} has "
                             f"stats={st is not None} profile={prof is not None}")
        key = (prof.scene_hash, prof.solver_config_hash, prof.dt)
        grp = groups.setdefault(key, _Group(profile=prof))
        grp.n_cap = max(grp.n_cap, pack.spec.n)
        for wid, lane in zip(pack.spec.world_ids, lane_of_worlds(pack.spec, st, policy), strict=True):
            grp.lanes[lane].append((wid, pack.spec.device))
    specs = [s for grp in groups.values() for s in _lane_specs(grp, policy)]
    specs.sort(key=lambda s: s.world_ids[0])
    plan = PackPlan(packs=tuple(specs), predicted_throughput=_throughput(specs, groups))
    log.info("repack: %d packs -> %d lane packs (%s)", len(packs), len(specs),
             ", ".join(f"{k}={sum(len(g.lanes[k]) for g in groups.values())}" for k in LANES))
    return plan


def _lane_specs(grp: _Group, policy: PackPolicy) -> list[PackSpec]:
    out = []
    for lane in LANES:
        worlds = sorted(grp.lanes[lane])
        if not worlds:
            continue
        for chunk in _chunks(worlds, grp.n_cap):
            device = Counter(dev for _, dev in chunk).most_common(1)[0][0]
            draft = pl.Draft(first_id=chunk[0][0], n=len(chunk), profile=grp.profile, kind="throughput")
            spec = pl.to_spec(draft, device, policy)
            out.append(replace(spec, world_ids=tuple(w for w, _ in chunk)))
    return out


def _chunks(worlds: list[tuple[int, str]], cap: int) -> list[list[tuple[int, str]]]:
    sizes = pl.split_counts(len(worlds), max(1, cap))
    out, start = [], 0
    for n in sizes:
        out.append(worlds[start:start + n])
        start += n
    return out


def _throughput(specs: Sequence[PackSpec], groups: Mapping[tuple, _Group]) -> float:
    profile_of = {}
    for grp in groups.values():
        for lane in LANES:
            for wid, _ in grp.lanes[lane]:
                profile_of[wid] = grp.profile
    loads: dict[str, pl.DeviceLoad] = {}
    for s in specs:
        loads.setdefault(s.device, pl.DeviceLoad(s.device, math.inf)).packs.append(s)
    return pl.predicted_throughput(list(loads.values()), profile_of)
