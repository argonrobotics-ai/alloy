# SPDX-License-Identifier: Apache-2.0
"""WorldPacker.plan (SPEC §3.9, parallel §4.3): grouping, sizing, memory as a hard limit, first-fit decreasing."""

from __future__ import annotations

import pytest

from alloy.core.errors import CapacityError
from alloy.parallel import costmodel as cm
from alloy.parallel import planning as pl
from alloy.parallel.packer import WorldPacker
from alloy.parallel.types import PackPolicy, WorldRequest

from alloy.testing.parallel import GB, profile

SCENE = object()  # plan never reads the scene; build does


def req(count: int, mode: str = "throughput", **kw) -> WorldRequest:
    return WorldRequest(scene=SCENE, count=count, mode=mode, profile=profile(**kw))


def all_ids(plan) -> list[int]:
    return sorted(i for s in plan.packs for i in s.world_ids)


def test_ids_are_global_consecutive_and_complete():
    reqs = [req(3000), req(10, scene_hash="b" * 16), req(500)]
    plan = WorldPacker(["cuda:0", "cuda:1"]).plan(reqs, free_mem={"cuda:0": 24 * GB, "cuda:1": 24 * GB})
    assert all_ids(plan) == list(range(3510))
    for s in plan.packs:
        assert s.n == len(s.world_ids) and list(s.world_ids) == list(range(s.world_ids[0], s.world_ids[0] + s.n))
    # request 0 split into ceil(3000 / 1055) = 3 near-equal packs (1055 = 9 N_knee)
    sizes = sorted(s.n for s in plan.packs if s.world_ids[0] < 3000)
    assert sizes == [1000, 1000, 1000]


def test_adjacent_same_group_requests_share_packs_but_not_across_groups():
    reqs = [req(40), req(40), req(40, scene_hash="x" * 16), req(40)]
    plan = WorldPacker(["cuda:0"]).plan(reqs, free_mem={"cuda:0": 24 * GB})
    firsts = sorted((s.world_ids[0], s.n) for s in plan.packs)
    assert firsts == [(0, 80), (80, 40), (120, 40)]


def test_memory_is_hard_per_device():
    pol = PackPolicy()
    reqs = [req(2000, bytes_per_world=5_000_000, scene_hash=f"{k}" * 16) for k in range(4)]
    free = {"cuda:0": 24 * GB, "cuda:1": 24 * GB}
    plan = WorldPacker(list(free), pol).plan(reqs, free_mem=free)
    for dev, mem in free.items():
        used = sum(s.predicted_bytes for s in plan.packs if s.device == dev)
        assert used <= (1 - pol.mem_reserve_fraction) * mem
    assert all_ids(plan) == list(range(8000))


def test_oom_200_boxes_predicted_and_refused():
    # 200-box pile: 11.8 MB dense H per world (parallel §3); 8,192 worlds need 97 GB on an 80 GB H100
    per_world = cm.estimate_bytes_per_world(cm.MemoryTerms(nv=1200, rows_max=4800, contacts_max=1600,
                                                           pairs_max=20_000, n_bodies=200))
    heavy = dict(nv=1200, island_nv_max=1200, bytes_per_world=per_world, bytes_fixed=0, t_floor_s=5e-3, c_w_s=4e-4)
    packer = WorldPacker(["cuda:0"])
    with pytest.raises(CapacityError):
        packer.plan([req(8192, **heavy)], free_mem={"cuda:0": 80 * GB})
    plan = packer.plan([req(4096, **heavy)], free_mem={"cuda:0": 80 * GB})
    assert sum(s.predicted_bytes for s in plan.packs) <= 0.9 * 80 * GB and all_ids(plan) == list(range(4096))


def test_first_fit_decreasing_places_largest_step_time_first():
    loads = pl.make_loads(["a", "b"], {"a": 10 * GB, "b": 10 * GB}, PackPolicy())
    small = pl.Draft(0, 10, profile(), "throughput")  # u ≈ 0.08
    big = pl.Draft(10, 2000, profile(), "throughput")  # past the knee: claims a device
    mid = pl.Draft(2010, 100, profile(), "throughput")  # u ≈ 0.46
    pl.place([small, big, mid], loads, PackPolicy())
    on = {ld.device: sorted(s.world_ids[0] for s in ld.packs) for ld in loads}
    assert on["a"] == [10]  # biggest first, alone
    assert on["b"] == [0, 2010]  # the two sub-knee packs share a device (Σu < 1)


def test_saturated_devices_time_share_least_loaded():
    loads = pl.make_loads(["a"], {"a": 24 * GB}, PackPolicy())
    drafts = [pl.Draft(i * 2000, 2000, profile(), "throughput") for i in range(3)]
    pl.place(drafts, loads, PackPolicy())
    assert len(loads[0].packs) == 3 and loads[0].occupancy == pytest.approx(3.0)


def test_latency_packs_single_world_high_priority_and_fill_cap():
    pol = PackPolicy()
    plan = WorldPacker(["cuda:0", "cuda:1"], pol).plan([req(2, mode="latency"), req(60), req(60, scene_hash="z" * 16)],
                                                       free_mem={"cuda:0": 24 * GB, "cuda:1": 24 * GB})
    lat = [s for s in plan.packs if s.kind == "latency"]
    assert [s.n for s in lat] == [1, 1] and all(s.stream_priority == pl.PRIORITY_HIGH for s in lat)
    for dev in ("cuda:0", "cuda:1"):
        packs = [s for s in plan.packs if s.device == dev]
        if any(s.kind == "latency" for s in packs):
            u = sum(cm.occupancy(profile(), s.n) for s in packs if s.kind != "latency")
            assert u <= pol.latency_fill_occupancy


def test_big_world_gets_a_device_alone():
    big = dict(bytes_per_world=int(0.85 * 24 * GB), bytes_fixed=0, scene_hash="B" * 16)
    plan = WorldPacker(["cuda:0", "cuda:1"]).plan([req(1, **big), req(50)],
                                                  free_mem={"cuda:0": 24 * GB, "cuda:1": 24 * GB})
    kinds = {s.kind: s.device for s in plan.packs}
    assert set(kinds) == {"big_world", "throughput"} and kinds["big_world"] != kinds["throughput"]


def test_predicted_throughput_adds_unsaturated_packs():
    p = profile()
    plan = WorldPacker(["cuda:0"]).plan([req(50)], free_mem={"cuda:0": 24 * GB})
    assert plan.predicted_throughput == pytest.approx(cm.throughput(p, 50))


def test_free_mem_must_cover_devices():
    with pytest.raises(ValueError):
        WorldPacker(["cuda:0", "cuda:1"]).plan([req(4)], free_mem={"cuda:0": GB})
