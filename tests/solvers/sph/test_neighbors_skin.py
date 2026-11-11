# SPDX-License-Identifier: Apache-2.0
"""Verlet-skin list reuse (LIQ-9 L4a): the rows the solver reads equal a fresh plain build at every step, the
rebuild flags are per world, and a skin row that overflows forces the next rebuild."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from alloy._vendor.newton import ParticleFlags

from alloy.solvers.sph.config import DFSPHConfig
from alloy.solvers.sph.neighbors import NeighborLists

H, SKIN = 1.9, 0.19


class _Pack:
    """Two worlds of jittered lattice points (contiguous ids) and plain + skin lists over them."""

    def __init__(self, device, rng, cap: int = 64, skin_rows: int | None = None):
        qs, ws = [], []
        for w, side in enumerate((4, 5)):
            ax = np.arange(side, dtype=np.float64)
            g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
            qs.append(g + rng.uniform(-0.2, 0.2, g.shape))
            ws.append(np.full(len(g), w, np.int32))
        self.q, self.world = np.concatenate(qs).astype(np.float32), np.concatenate(ws)
        self.flags = np.full(len(self.q), int(ParticleFlags.ACTIVE), np.int32)
        self.start = np.array([0, 64, 189])
        self.device, n = device, len(self.q)
        self.plain = NeighborLists(n, cap, H, wp.get_device(device), grid_dim=16, world_start=self.start)
        self.skin = NeighborLists(n, cap, H, wp.get_device(device), grid_dim=16, world_start=self.start, skin=SKIN,
                                  spacing=1.0, skin_rows=skin_rows)
        self.ovf_plain = wp.zeros(2, dtype=wp.int32, device=device)
        self.ovf_skin = wp.zeros(2, dtype=wp.int32, device=device)

    def step(self) -> np.ndarray:
        """Build both from the current arrays; assert equal rows; return the skin lists' rebuilt worlds."""
        arr = lambda x, t: wp.array(x, dtype=t, device=self.device)  # noqa: E731
        ins = (arr(self.q, wp.vec3), arr(self.flags, wp.int32), arr(np.ones(len(self.q), np.int8), wp.int8),
               arr(self.world, wp.int32))
        self.plain.build(*ins, self.ovf_plain)
        self.skin.build(*ins, self.ovf_skin)
        p_ids, p_cnt = self.plain.global_rows()
        s_ids, s_cnt = self.skin.global_rows()
        assert np.array_equal(p_cnt, s_cnt)
        assert np.array_equal(self.ovf_plain.numpy(), self.ovf_skin.numpy())
        for i in range(len(self.q)):
            assert np.array_equal(p_ids[i, : p_cnt[i]], s_ids[i, : s_cnt[i]]), i
        return self.skin.skin.rebuilt_worlds()


def test_skin_rows_equal_plain_rows_and_flags_are_per_world(device, rng):
    pk = _Pack(device, rng)
    assert pk.step().tolist() == [True, True]                 # first build: every world
    for _ in range(3):                                        # small moves (< skin/2 in total): reuse
        pk.q += rng.uniform(-0.01, 0.01, pk.q.shape).astype(np.float32)
        assert pk.step().tolist() == [False, False]
    pk.q[70] += np.float32(0.3)                               # one particle of world 1 moves more than skin/2
    assert pk.step().tolist() == [False, True]
    pk.flags[5] = 0                                           # world 0's active set changes (retired)
    assert pk.step().tolist() == [True, False]
    pk.flags[5] = int(ParticleFlags.ACTIVE)                   # and back (poured)
    assert pk.step().tolist() == [True, False]
    assert pk.step().tolist() == [False, False]


def test_plain_overflow_is_counted_as_a_plain_build(device, rng):
    pk = _Pack(device, rng, cap=8)                            # plain rows overflow (keep the 8 lowest ids)
    assert pk.step().tolist() == [True, True]
    assert pk.ovf_plain.numpy().sum() > 0
    assert pk.step().tolist() == [False, False]               # the superset fits: reused, overflow counted again


def test_full_skin_rows_are_plain_rows_and_force_the_next_rebuild(device, rng):
    pk = _Pack(device, rng, cap=8, skin_rows=12)              # superset rows overflow too
    assert pk.step().tolist() == [True, True]
    assert pk.step().tolist() == [True, True]                 # no motion, rebuilt anyway (forced)


def test_skin_lists_replay_in_a_cuda_graph(device, rng):
    """Two captured updates (graph replay) give the same rows as eager plain builds (CUDA only)."""
    if not wp.get_device(device).is_cuda:
        pytest.skip("CUDA graphs")
    pk = _Pack(device, rng)
    pk.step()                                                 # first build outside the graph
    arr = lambda x, t: wp.array(x, dtype=t, device=device)  # noqa: E731
    q = arr(pk.q, wp.vec3)
    ins = (arr(pk.flags, wp.int32), arr(np.ones(len(pk.q), np.int8), wp.int8), arr(pk.world, wp.int32))
    with wp.ScopedCapture(device=device) as cap:
        pk.skin.build(q, *ins, pk.ovf_skin)
    shift = np.zeros_like(pk.q)
    shift[70] = 0.3                                           # world 1 must rebuild inside the replay
    q.assign(pk.q + shift)
    wp.capture_launch(cap.graph)
    pk.plain.build(q, *ins, pk.ovf_plain)
    p_ids, p_cnt = pk.plain.global_rows()
    s_ids, s_cnt = pk.skin.global_rows()
    assert np.array_equal(p_cnt, s_cnt)
    for i in range(len(pk.q)):
        assert np.array_equal(p_ids[i, : p_cnt[i]], s_ids[i, : s_cnt[i]]), i


@pytest.mark.parametrize("skin, ok", [(None, True), (0.0, True), (0.1, True), (-0.1, False), (1.5, False)])
def test_skin_config_range(skin, ok):
    cfg = DFSPHConfig(reneighbor_skin=skin)
    if ok:
        cfg.validate()
    else:
        with pytest.raises(ValueError, match="reneighbor_skin"):
            cfg.validate()


def test_a_skin_needs_world_local_rows(device):
    with pytest.raises(ValueError, match="world-local"):
        NeighborLists(10, 16, 1.0, wp.get_device(device), grid_dim=8, skin=0.1)


def test_solver_with_a_skin_matches_the_plain_lists_bitwise():
    """A small dam break and a resting column (CPU): reneighbor_skin 0.1 gives the same end state, bit for bit, as
    rebuilding every step, with fewer rebuilds."""
    from alloy.testing.sph import dam_break, default_config, run, water_column

    for make, dt, steps in ((lambda c: dam_break(8, 16, 4, 32, cfg=c, device="cpu"), 0.02 / 64, 60),
                            (lambda c: water_column(4, 4, 8, cfg=c, device="cpu"), 1.0e-3, 60)):
        ends, solvers = [], []
        for skin in (None, 0.1):
            sc = make(default_config(2.0e-3, reneighbor_skin=skin))
            s = sc.make_solver()
            a, b = sc.model.state(), sc.model.state()
            s.prime(a, dt)
            a, b = run(s, a, b, steps, dt)
            ends.append(a.particle_q.numpy())
            solvers.append(s)
        assert np.array_equal(ends[0], ends[1])
        assert solvers[0]._nbrs.skin is None and 0 < solvers[1]._nbrs.skin.builds < steps
