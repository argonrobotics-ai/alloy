# SPDX-License-Identifier: Apache-2.0
"""CT-12: Newton and MJWarp overflow bits stay at 0 (no dropped contacts or rows) on the precision scenes, including
256 worlds, where the spike's explicit pipeline overflowed its automatic triangle-pair buffer (spike finding 3).
Hydroelastic runs 64 worlds: deterministic hydroelastic needs one pipeline per world here (345k iso voxels per world
against Newton's 419,430 cap; ~150 MB and ~5 ms per pipeline on an L4), so 256 worlds do not fit in 24 GB.
CPU: the layer's world chunking keeps every deterministic pipeline under the 2^20 - 1 triangle-pair and iso caps."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.contact.pipeline import (ISO_VOXEL_CAP, TRIANGLE_PAIR_CAP, _chunks, _tri_count,
                                    hydro_band_spans_clearance)
from alloy.contact.types import PrecisionPairSpec
from alloy.testing.contact import peg_hole_scene

from peg_harness import advance, l2_dt, make_run, overflow_total, perturb, ramp, record

WORLDS = {"sdf": 256, "hydroelastic": 64}


class _FakeMesh:
    def __init__(self, tris: int) -> None:
        self.indices = np.zeros(3 * tris, np.int32)


class _FakeModel:
    def __init__(self, worlds: int, tris: int) -> None:
        self.world_count = worlds
        self.shape_source = [_FakeMesh(tris), _FakeMesh(tris)]


def test_chunks_respect_triangle_cap():
    model = _FakeModel(4096, 50_000)
    wpairs = np.zeros((1, 4096, 2), np.int32)
    wpairs[0, :, 1] = 1
    chunks = _chunks(model, wpairs, deterministic=True)
    per_world = 2 * (_tri_count(model, 0) + _tri_count(model, 1))
    assert all(len(c) * per_world <= TRIANGLE_PAIR_CAP for c in chunks)
    assert sum(len(c) for c in chunks) == 4096
    assert len(_chunks(model, wpairs, deterministic=False)) == 1


def test_chunks_respect_iso_cap():
    model = _FakeModel(64, 100)
    wpairs = np.zeros((1, 64, 2), np.int32)
    wpairs[0, :, 1] = 1
    assert all(len(c) == 4 for c in _chunks(model, wpairs, True, iso_per_world=100_000))
    assert all(len(c) == 1 for c in _chunks(model, wpairs, True, iso_per_world=1.3 * ISO_VOXEL_CAP))
    assert len(_chunks(model, wpairs, False, iso_per_world=1.3 * ISO_VOXEL_CAP)) == 1


def test_hydro_band_vs_clearance():
    spec = PrecisionPairSpec(0, 1, mode="hydroelastic", voxel_size=5e-5, clearance=2e-4)
    assert hydro_band_spans_clearance(spec, 2 * 5e-4)        # the 0.5 mm default gap keeps the clearance shell
    assert not hydro_band_spans_clearance(spec, 2 * 5e-6)
    assert not hydro_band_spans_clearance(PrecisionPairSpec(0, 1), 1e-3)   # clearance unknown


@pytest.mark.gpu
@pytest.mark.parametrize("native", ["mujoco", "newton"])
@pytest.mark.parametrize("mode", ["hydroelastic", "sdf"])
def test_ct12_no_drops(mode, native):
    worlds = WORLDS[mode]
    scene = peg_hole_scene(worlds, device="cuda:0", mode=mode)
    run = make_run(scene, native, l2_dt(scene))
    perturb(run, np.random.default_rng(12))
    advance(run, int(0.05 / run.dt))
    ramp(run, np.array([0.0, 0.0, -1.0]), 0.0, 50.0, 0.1, n_seg=5)
    ovf = overflow_total(run)
    record("ct12", dict(metric="CT-12", mode=mode, native=native, worlds=worlds, overflow=ovf,
                        pipelines=len(run.layer.precision_streams)))
    assert sum(ovf.values()) == 0, ovf
