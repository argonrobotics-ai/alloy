# SPDX-License-Identifier: Apache-2.0
"""Pure planning steps behind ``WorldPacker.plan`` (SPEC §3.9, parallel §4.3).

1. Global world ids: request r owns ``[base_r, base_r + count_r)`` in request order (ids never depend on packing).
2. Runs: adjacent requests with the same (scene_hash, solver_config_hash, dt, mode) merge into one id run; only
   worlds of one run share a pack, so every pack's ids are consecutive.
3. Kind: latency (mode), big_world (one world > big_world_bytes_fraction of the device, or > 1 M particles), else
   throughput. Throughput runs split into ``ceil(count / N*)`` near-equal packs, ``N* = pack_size(...)``.
4. Placement: first-fit decreasing on predicted step time. Memory is hard. Occupancy u = N/(N + N_knee) is soft: a
   device takes packs while Σu <= 1 (latency devices only to ``latency_fill_occupancy``); packs at or past the knee
   claim the whole device. When no device has occupancy left, the pack time-shares the least-loaded device that has
   memory. No memory anywhere -> CapacityError.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from alloy.core.errors import CapacityError
from alloy.parallel import costmodel as cm
from alloy.parallel.types import PackPolicy, PackSpec, WorldProfile, WorldRequest

log = logging.getLogger("alloy.parallel")

BIG_WORLD_PARTICLES = 1_000_000
PRIORITY_NORMAL = 0
PRIORITY_HIGH = -1  # CUDA convention: lower value = higher stream priority


@dataclass(frozen=True)
class Run:
    """Consecutive global world ids that may share packs."""

    first_id: int
    count: int
    profile: WorldProfile
    mode: str
    request_index: int  # first request of the run


@dataclass(frozen=True)
class Draft:
    """A pack before placement."""

    first_id: int
    n: int
    profile: WorldProfile
    kind: str  # "throughput" | "latency" | "big_world"


@dataclass
class DeviceLoad:
    device: str
    mem_budget: float  # (1 - reserve) * free bytes
    mem_used: float = 0.0
    occupancy: float = 0.0
    has_latency: bool = False
    packs: list[PackSpec] = field(default_factory=list)


def request_bases(requests: Sequence[WorldRequest]) -> list[int]:
    """Global id of each request's first world (request order)."""
    bases, nxt = [], 0
    for req in requests:
        if req.count < 1:
            raise ValueError(f"WorldRequest.count must be >= 1, got {req.count}")
        bases.append(nxt)
        nxt += req.count
    return bases


def group_key(req: WorldRequest) -> tuple:
    p = req.profile
    return (p.scene_hash, p.solver_config_hash, p.dt, req.mode)


def make_runs(requests: Sequence[WorldRequest]) -> list[Run]:
    """Merge adjacent same-group requests into consecutive-id runs."""
    runs: list[Run] = []
    for i, (req, base) in enumerate(zip(requests, request_bases(requests), strict=True)):
        last = runs[-1] if runs else None
        if last is not None and group_key(requests[last.request_index]) == group_key(req):
            runs[-1] = Run(last.first_id, last.count + req.count, last.profile, last.mode, last.request_index)
        else:
            runs.append(Run(base, req.count, req.profile, req.mode, i))
    return runs


def is_big_world(p: WorldProfile, device_mem: float, policy: PackPolicy) -> bool:
    return p.bytes_fixed + p.bytes_per_world > policy.big_world_bytes_fraction * device_mem or (
        p.n_particles > BIG_WORLD_PARTICLES)


def split_counts(count: int, n_max: int) -> list[int]:
    """count split into ceil(count / n_max) packs whose sizes differ by at most one."""
    k = math.ceil(count / n_max)
    base, extra = divmod(count, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def draft_packs(run: Run, device_mem: float, policy: PackPolicy) -> list[Draft]:
    """Packs for one run, before placement."""
    p = run.profile
    if run.mode == "latency":
        return [Draft(run.first_id + i, 1, p, "latency") for i in range(run.count)]
    if is_big_world(p, device_mem, policy):
        return [Draft(run.first_id + i, 1, p, "big_world") for i in range(run.count)]
    n_star = cm.pack_size(p, run.count, int(device_mem), policy)
    drafts, nxt = [], run.first_id
    for n in split_counts(run.count, n_star):
        drafts.append(Draft(nxt, n, p, "throughput"))
        nxt += n
    return drafts


def claimed_occupancy(d: Draft) -> float:
    """Occupancy a pack claims: 1 at or past the knee or for a big world, else u(N)."""
    if d.kind == "big_world" or d.n >= cm.n_knee(d.profile):
        return 1.0
    return cm.occupancy(d.profile, d.n)


def make_loads(devices: Sequence[str], free_mem: Mapping[str, int], policy: PackPolicy) -> list[DeviceLoad]:
    return [DeviceLoad(d, (1.0 - policy.mem_reserve_fraction) * free_mem[d]) for d in devices]


def place(drafts: Sequence[Draft], loads: list[DeviceLoad], policy: PackPolicy) -> None:
    """First-fit decreasing on predicted step time; latency packs first (they set the 0.7 fill cap)."""
    order = sorted(drafts, key=lambda d: (d.kind != "latency", -cm.step_time(d.profile, d.n), d.first_id))
    for d in order:
        load = _choose_device(d, loads, policy)
        spec = to_spec(d, load.device, policy)
        load.packs.append(spec)
        load.mem_used += spec.predicted_bytes
        load.occupancy += claimed_occupancy(d)
        load.has_latency |= d.kind == "latency"


def _choose_device(d: Draft, loads: list[DeviceLoad], policy: PackPolicy) -> DeviceLoad:
    need_mem = cm.predicted_bytes(d.profile, d.n)
    u = claimed_occupancy(d)
    fits = [ld for ld in loads if ld.mem_used + need_mem <= ld.mem_budget]
    if not fits:
        raise CapacityError(
            f"pack of {d.n} worlds (scene {d.profile.scene_hash[:12]}, {need_mem} B) fits no device: "
            + ", ".join(f"{ld.device} {ld.mem_budget - ld.mem_used:.0f} B left" for ld in loads))
    for ld in fits:
        cap = policy.latency_fill_occupancy if (ld.has_latency or d.kind == "latency") else 1.0
        if d.kind == "big_world" and ld.packs:
            continue
        if ld.occupancy + u <= cap + 1e-12 or (not ld.packs):
            return ld
    shared = min(fits, key=lambda ld: (ld.occupancy, ld.mem_used))
    log.info("pack of %d worlds time-shares %s (occupancy %.2f + %.2f)", d.n, shared.device, shared.occupancy, u)
    return shared


def to_spec(d: Draft, device: str, policy: PackPolicy) -> PackSpec:
    caps = cm.capacities(d.profile, d.n, policy)
    prio = PRIORITY_HIGH if d.kind == "latency" and policy.latency_lane else PRIORITY_NORMAL
    return PackSpec(device=device, world_ids=tuple(range(d.first_id, d.first_id + d.n)), n=d.n,
                    stream_priority=prio, kind=d.kind, rigid_contact_max=caps["rigid_contact_max"],
                    njmax=caps["njmax"], nconmax=caps["nconmax"], shape_pairs_max=caps["shape_pairs_max"],
                    predicted_bytes=cm.predicted_bytes(d.profile, d.n))


def predicted_throughput(loads: Sequence[DeviceLoad], profiles: Mapping[int, WorldProfile]) -> float:
    """Σ over devices of Σ Θ_i, scaled by 1/Σu when packs oversubscribe a device (sim-s per GPU-s)."""
    total = 0.0
    for ld in loads:
        theta = sum(cm.throughput(profiles[s.world_ids[0]], s.n) for s in ld.packs)
        u = sum(cm.occupancy(profiles[s.world_ids[0]], s.n) for s in ld.packs)
        total += theta / max(1.0, u)
    return total
