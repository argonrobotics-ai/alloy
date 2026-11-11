# SPDX-License-Identifier: Apache-2.0
"""World packer: plan packs from measured profiles, build them through SceneAssembler, repack at resets.

SPEC §3.9 and parallel §4.3/§4.5. ``plan`` is pure host logic (``planning.py``); ``build`` calls
``assembler.assemble(scene, device, world_count=n, world_id_offset=..., capacities=spec)`` once per pack and wraps
each ``BuiltScene`` in a ``Simulation`` (injectable factory, so packing is testable without WP2); ``repack`` splits
packs by awake fraction and per-world iterations at reset boundaries (``repack.py``).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import warp as wp

from alloy.core.errors import AssemblyError
from alloy.parallel import planning as pl
from alloy.parallel.repack import repack_plan
from alloy.parallel.types import Pack, PackPlan, PackPolicy, PackSpec, PackStats, Trace, WorldProfile, WorldRequest

if TYPE_CHECKING:
    from alloy.cards.types import SceneCard
    from alloy.scene.types import BuiltScene, SceneAssembler

log = logging.getLogger("alloy.parallel")

SimFactory = Callable[["BuiltScene"], Any]


def device_free_memory(device: str) -> int:
    """Free bytes on a device: CUDA from the driver; CPU from available physical pages."""
    dev = wp.get_device(device)
    if dev.is_cuda:
        return int(dev.free_memory)
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError):  # pragma: no cover - non-POSIX hosts
        return 8 * 2**30


def _default_sim_factory(built: BuiltScene) -> Any:
    from alloy.sim import Simulation  # WP2; imported lazily so planning works without it

    return Simulation(built)


class WorldPacker:
    """Packs identical worlds onto devices from measured ``WorldProfile``s (SPEC §3.9)."""

    def __init__(self, devices: Sequence[str], policy: PackPolicy = PackPolicy(), *,
                 sim_factory: SimFactory | None = None, assembler: SceneAssembler | None = None) -> None:
        if not devices:
            raise ValueError("WorldPacker needs at least one device")
        self.devices = tuple(devices)
        self.policy = policy
        self.sim_factory = sim_factory or _default_sim_factory
        self.assembler = assembler
        self._profiles: dict[int, WorldProfile] = {}  # first global world id of each planned pack -> profile
        self.memory_checks: list[tuple[int, int, int | None]] = []  # (first id, predicted B, measured B) per build

    # ---- probe ----------------------------------------------------------------------------------------------
    def probe(self, scene: SceneCard, replay: Trace | None, device: str,
              n_values: Sequence[int] = (1, 64, 1024)) -> WorldProfile:
        """Measure t_floor, c_w, iterations, capacities and bytes per world on a replay (parallel §4.1)."""
        from alloy.parallel.probe import probe_scene  # heavy imports only when probing

        return probe_scene(scene, replay, device, n_values, assembler=self._require_assembler(),
                           sim_factory=self.sim_factory)

    # ---- plan -----------------------------------------------------------------------------------------------
    def plan(self, requests: Sequence[WorldRequest], free_mem: Mapping[str, int] | None = None) -> PackPlan:
        """Group, size and place packs; memory is a hard limit (CapacityError when a pack fits nowhere)."""
        free = self._free_memory(free_mem)
        loads = pl.make_loads(self.devices, free, self.policy)
        largest = max(free.values())
        drafts = [d for run in pl.make_runs(requests) for d in pl.draft_packs(run, largest, self.policy)]
        pl.place(drafts, loads, self.policy)
        self._profiles = {d.first_id: d.profile for d in drafts}
        packs = tuple(sorted((s for ld in loads for s in ld.packs), key=lambda s: s.world_ids[0]))
        plan = PackPlan(packs=packs, predicted_throughput=pl.predicted_throughput(loads, self._profiles))
        log.info("plan: %d requests -> %d packs on %s, predicted %.3g sim-s/GPU-s", len(requests), len(packs),
                 ",".join(self.devices), plan.predicted_throughput)
        return plan

    def _free_memory(self, free_mem: Mapping[str, int] | None) -> dict[str, int]:
        if free_mem is None:
            return {d: device_free_memory(d) for d in self.devices}
        missing = [d for d in self.devices if d not in free_mem]
        if missing:
            raise ValueError(f"free_mem has no entry for devices {missing}")
        return {d: int(free_mem[d]) for d in self.devices}

    # ---- build ----------------------------------------------------------------------------------------------
    def build(self, plan: PackPlan, requests: Sequence[WorldRequest],
              assembler: SceneAssembler | None = None) -> list[Pack]:
        """One ``BuiltScene`` + ``Simulation`` per pack, world ids global and consecutive within each pack."""
        asm = assembler or self._require_assembler()
        bases = pl.request_bases(requests)
        return [self._build_pack(spec, requests, bases, asm) for spec in plan.packs]

    def _build_pack(self, spec: PackSpec, requests: Sequence[WorldRequest], bases: Sequence[int],
                    asm: SceneAssembler) -> Pack:
        if len(spec.world_ids) != spec.n:
            raise AssemblyError(f"pack spec has {len(spec.world_ids)} world ids for n = {spec.n}")
        req = _owner(spec.world_ids[0], requests, bases)
        mem0 = _mem_current(spec.device)
        consecutive = _is_consecutive(spec.world_ids)
        extra = {} if consecutive else {"world_ids": spec.world_ids}  # repacked lanes (SCR-WP8-2)
        built = asm.assemble(req.scene, spec.device, world_count=spec.n, world_id_offset=spec.world_ids[0],
                             capacities=spec, **extra)
        _check_built(built, spec, consecutive)
        sim = self.sim_factory(built)
        measured = None if mem0 is None else _mem_current(spec.device) - mem0
        self.memory_checks.append((spec.world_ids[0], spec.predicted_bytes, measured))
        ids = wp.array(spec.world_ids, dtype=wp.int32, device=spec.device)
        log.info("built pack ids %d..%d (%s) on %s: predicted %d B, measured %s B", spec.world_ids[0],
                 spec.world_ids[-1], "consecutive" if consecutive else "repacked", spec.device, spec.predicted_bytes,
                 measured)
        return Pack(spec=spec, built=built, sim=sim, world_ids=ids)

    # ---- repack ---------------------------------------------------------------------------------------------
    def repack(self, packs: Sequence[Pack], stats: Sequence[PackStats]) -> PackPlan:
        """New plan at a reset boundary: slow lane (iters p99 > 3x pack median) and awake/asleep lanes."""
        profiles = {p.spec.world_ids[0]: self._profiles.get(p.spec.world_ids[0]) for p in packs}
        plan = repack_plan(packs, stats, profiles, self.policy)
        owner = {wid: profiles[p.spec.world_ids[0]] for p in packs for wid in p.spec.world_ids}
        self._profiles = {s.world_ids[0]: owner[s.world_ids[0]] for s in plan.packs}  # so repack can repeat
        return plan

    def _require_assembler(self) -> SceneAssembler:
        if self.assembler is None:
            raise AssemblyError("WorldPacker needs a SceneAssembler (pass assembler= to __init__ or build)")
        return self.assembler


def _is_consecutive(ids: Sequence[int]) -> bool:
    return all(b - a == 1 for a, b in zip(ids, ids[1:], strict=False))


def _check_built(built: BuiltScene, spec: PackSpec, consecutive: bool) -> None:
    """The assembler must have built exactly the worlds the pack asked for, under their global ids."""
    if built.world_count != spec.n:
        raise AssemblyError(f"assembler built {built.world_count} worlds, pack asked {spec.n}")
    if consecutive and built.world_id_offset != spec.world_ids[0]:
        raise AssemblyError(f"assembler offset {built.world_id_offset}, pack starts at {spec.world_ids[0]}")
    if not consecutive:
        got = None if built.world_ids is None else tuple(int(i) for i in built.world_ids.numpy())
        if got != tuple(spec.world_ids):
            raise AssemblyError(f"assembler ignored world_ids for a repacked pack (got {str(got)[:60]})")


def _owner(first_id: int, requests: Sequence[WorldRequest], bases: Sequence[int]) -> WorldRequest:
    for req, base in zip(requests, bases, strict=True):
        if base <= first_id < base + req.count:
            return req
    raise AssemblyError(f"world id {first_id} belongs to no request")


def _mem_current(device: str) -> int | None:
    dev = wp.get_device(device)
    if not (dev.is_cuda and wp.is_mempool_enabled(dev)):
        return None
    wp.synchronize_device(dev)
    return int(wp.get_mempool_used_mem_current(dev))
