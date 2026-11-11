# SPDX-License-Identifier: Apache-2.0
"""WorldPacker.build (consecutive global ids through SceneAssembler) and repack lanes (SPEC §3.9, parallel §4.5)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.errors import AssemblyError
from alloy.parallel.packer import WorldPacker
from alloy.parallel.repack import lane_of_worlds
from alloy.parallel.types import PackPolicy, PackStats, WorldRequest
from alloy.testing.parallel import GB, StubAssembler, fake_sim_factory, profile


def requests() -> list[WorldRequest]:
    return [WorldRequest(scene="scene-A", count=2500, mode="throughput", profile=profile()),
            WorldRequest(scene="scene-B", count=30, mode="throughput", profile=profile(scene_hash="b" * 16)),
            WorldRequest(scene="scene-A2", count=1, mode="latency", profile=profile(scene_hash="l" * 16))]


def built_packs(devices=("cpu",)):
    packer = WorldPacker(list(devices), sim_factory=fake_sim_factory)
    reqs = requests()
    plan = packer.plan(reqs, free_mem={d: 24 * GB for d in devices})
    asm = StubAssembler()
    return packer, plan, asm, packer.build(plan, reqs, asm), reqs


def test_build_assembles_each_pack_with_consecutive_global_ids():
    _, plan, asm, packs, _ = built_packs(devices=("cpu",))
    assert len(packs) == len(plan.packs) == len(asm.calls)
    covered = []
    for pack, call in zip(packs, asm.calls, strict=True):
        ids = pack.world_ids.numpy().tolist()
        assert ids == list(range(ids[0], ids[0] + pack.spec.n)) == list(pack.spec.world_ids)
        assert call["world_count"] == pack.spec.n and call["world_id_offset"] == ids[0]
        assert call["capacities"] is pack.spec and pack.sim["built"] is pack.built
        covered += ids
    assert sorted(covered) == list(range(2531))
    scenes = {c["world_id_offset"]: c["scene"] for c in asm.calls}
    assert scenes[2530] == "scene-A2" and all(scenes[k] == "scene-A" for k in scenes if k < 2500)


def test_build_rejects_a_mismatched_assembler():
    class Liar(StubAssembler):
        def assemble(self, scene, device, *, world_count=None, world_id_offset=0, capacities=None):
            return super().assemble(scene, device, world_count=(world_count or 1) + 1, world_id_offset=world_id_offset)

    packer = WorldPacker(["cpu"], sim_factory=fake_sim_factory)
    reqs = requests()[:1]
    with pytest.raises(AssemblyError):
        packer.build(packer.plan(reqs, free_mem={"cpu": 24 * GB}), reqs, Liar())


def stats_for(pack_index: int, n: int, awake: np.ndarray, iters: np.ndarray | None = None) -> PackStats:
    return PackStats(pack=pack_index, wall_s=1.0, sim_s=10.0, awake_fraction=awake, iters_p50=4.0, iters_p99=9.0,
                     contacts_p99=8, rows_p99=40, overflow=0, world_iters_p99=iters)


def test_lane_assignment():
    _, _, _, packs, _ = built_packs(devices=("cpu",))
    spec = packs[0].spec
    awake = np.zeros(spec.n)
    awake[:10] = 1.0
    iters = np.full(spec.n, 5.0)
    iters[-3:] = 40.0  # > 3x the median
    lanes = lane_of_worlds(spec, stats_for(0, spec.n, awake, iters), PackPolicy())
    assert lanes[:10] == ["awake"] * 10 and lanes[-3:] == ["slow"] * 3
    assert set(lanes[10:-3]) == {"asleep"}


def test_repack_groups_awake_worlds_and_keeps_every_world():
    packer, plan, _, packs, _ = built_packs(devices=("cpu",))
    rng = np.random.default_rng(0)
    stats = [stats_for(i, p.spec.n, (rng.random(p.spec.n) < 0.1).astype(float)) for i, p in enumerate(packs)]
    awake_ids = {wid for p, st in zip(packs, stats, strict=True)
                 for wid, a in zip(p.spec.world_ids, st.awake_fraction, strict=True) if a >= 0.5}
    new = packer.repack(packs, stats)
    ids = sorted(i for s in new.packs for i in s.world_ids)
    assert ids == list(range(2531))
    for s in new.packs:  # a lane pack is all awake or all asleep
        flags = {wid in awake_ids for wid in s.world_ids}
        assert len(flags) == 1
        assert s.n == len(s.world_ids) <= max(p.spec.n for p in packs)
    assert new.predicted_throughput > 0.0


def test_repack_requires_stats_for_every_pack():
    packer, _, _, packs, _ = built_packs(devices=("cpu",))
    with pytest.raises(ValueError):
        packer.repack(packs, [])


def test_build_passes_world_ids_for_repacked_packs_and_repack_repeats():
    packer, _, asm, packs, reqs = built_packs(devices=("cpu",))
    awake = np.zeros(packs[0].spec.n)
    awake[::2] = 1.0
    stats = [stats_for(i, p.spec.n, awake if i == 0 else np.zeros(p.spec.n)) for i, p in enumerate(packs)]
    new = packer.build(packer.repack(packs, stats), reqs, asm)
    repacked = [p for p in new if list(p.spec.world_ids) != list(range(p.spec.world_ids[0],
                                                                        p.spec.world_ids[0] + p.spec.n))]
    assert repacked, "the awake lane holds every other world: not consecutive"
    for p in repacked:
        call = next(c for c in asm.calls if c["world_ids"] is not None and c["world_ids"][0] == p.spec.world_ids[0])
        assert tuple(call["world_ids"]) == p.spec.world_ids == tuple(p.built.world_ids.numpy().tolist())
    again = packer.repack(new, [stats_for(i, p.spec.n, np.zeros(p.spec.n)) for i, p in enumerate(new)])
    assert sorted(i for s in again.packs for i in s.world_ids) == list(range(2531))


def test_build_rejects_an_assembler_that_ignores_world_ids():
    class Deaf(StubAssembler):
        def assemble(self, scene, device, *, world_count=None, world_id_offset=0, capacities=None, world_ids=None):
            return super().assemble(scene, device, world_count=world_count, world_id_offset=world_id_offset)

    packer, _, _, packs, reqs = built_packs(devices=("cpu",))
    awake = np.zeros(packs[0].spec.n)
    awake[::2] = 1.0
    stats = [stats_for(i, p.spec.n, awake if i == 0 else np.zeros(p.spec.n)) for i, p in enumerate(packs)]
    with pytest.raises(AssemblyError, match="ignored world_ids"):
        packer.build(packer.repack(packs, stats), reqs, Deaf())


def test_pack_runner_steps_every_pack_once_per_period():
    from alloy.parallel.streams import PackRunner

    class CountingSim:
        def __init__(self):
            self.periods = 0

        def step_period(self):
            self.periods += 1

    packer = WorldPacker(["cpu"], sim_factory=lambda built: CountingSim())
    reqs = requests()
    packs = packer.build(packer.plan(reqs, free_mem={"cpu": 24 * GB}), reqs, StubAssembler())
    runner = PackRunner(packs)
    for _ in range(3):
        runner.step_period()
    assert runner.periods == 3 and all(p.sim.periods == 3 for p in packs)
    assert all(s is None for s in runner.streams)  # CPU: no streams
    with pytest.raises(ValueError):
        PackRunner([])
