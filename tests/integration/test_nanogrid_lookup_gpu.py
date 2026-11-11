# SPDX-License-Identifier: Apache-2.0
"""The integer packed-grid cell lookup on CUDA (``alloy.integration.nanogrid_lookup``; ``-m gpu``, one L4).

Before it, MPM grains sank into bodies they touched at many GPU worlds, more the further out the world sits in the
shared sparse grid (known gap docs/known_gaps/mpm-moving-container-many-worlds.md, fixed for 64 worlds and below). Per
world, the largest number of grains more than 1 mm inside a box-shaped body at any control period, which must now be
flat in the world's offset:
- a kinematic plate pressing a dough block, 64 worlds: before, 1 in world 0 rising to about 32;
- a free 0.3 kg box resting on a sand slab, 64 worlds (AUDIT-STACK's case: 45 of 64 worlds before). Above 64 worlds
  Newton's sparse grid ran out of capacity (status 8) or memory with either lookup, so the law stops at 64;
- 1 world reads the same, bit for bit, with Warp's own lookup (run it in a process of its own: after the many-world
  rows a one-world build has run out of memory on an L4).

    PYTHONPATH=. python -m pytest tests/integration/test_nanogrid_lookup_gpu.py -q -m gpu -s
"""

import os

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.integration import nanogrid_lookup as N

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.timeout(3600)]

DEV = os.environ.get("ALLOY_DEVICE", "cuda:0")
DEPTH = 1.0e-3
BOX = 7                                    # Newton's GeoType.BOX


@pytest.fixture(autouse=True)
def _cuda():
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")


def plate(worlds: int) -> tuple[alloy.Scene, list]:
    s = alloy.Scene(dt=0.01, worlds=worlds)
    s.add_ground()
    s.add_box("dough", size=(0.06, 0.06, 0.03), pos=(0, 0, 0.015), material="dough_wheat")
    s.add_box("plate", size=(0.10, 0.10, 0.01), pos=(0, 0, 0.0352), kinematic=True, material="steel_carbon")
    return s, [(5, ("plate", (0, 0, -0.02), 0.75)), (100, None)]


def box_on_sand(worlds: int, size: float = 0.04) -> tuple[alloy.Scene, list]:
    s = alloy.Scene(dt=0.01, worlds=worlds)
    s.add_ground()
    s.add_box("sand", size=(size, size, size / 2), pos=(0, 0, size / 4), material="sand_silica_dry")
    s.add_box("box", size=(0.03, 0.03, 0.03), pos=(0, 0, size / 2 + 0.016), mass=0.3)
    return s, [(60, None)]


def _rot(q, v):
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _inside(sim, pids, pw, worlds) -> np.ndarray:
    """Per world, the grains more than DEPTH inside a box shape (static, kinematic or free) or below the ground."""
    m, st = sim.engine.model, sim.engine.state
    t, sb, sw = m.shape_type.numpy(), m.shape_body.numpy(), m.shape_world.numpy()
    sx, half = m.shape_transform.numpy(), m.shape_scale.numpy()
    x = st.particle_q.numpy()[pids].astype(np.float64)
    bq = st.body_q.numpy().astype(np.float64) if st.body_q is not None else np.zeros((0, 7))
    deep = x[:, 2] < -DEPTH
    for k in np.flatnonzero(t == BOX):
        p, q = sx[k, :3].astype(np.float64), sx[k, 3:7].astype(np.float64)
        if sb[k] >= 0:
            b = bq[sb[k]]
            p = b[:3] + _rot(b[3:7], p)
            u, w = b[3:6], b[6]
            q = np.concatenate([w * q[:3] + q[3] * u + np.cross(u, q[:3]), [w * q[3] - u @ q[:3]]])
        sel = np.flatnonzero(pw == sw[k]) if sw[k] >= 0 else np.arange(len(x))
        inv = np.concatenate([-q[:3], q[3:]])
        loc = _rot(inv[None, :], x[sel] - p[None, :])
        deep[sel[np.min(half[k][None, :3] - np.abs(loc), axis=1) > DEPTH]] = True
    return np.bincount(pw[deep], minlength=worlds)


def _run(make, worlds: int, **kw) -> tuple[np.ndarray, bytes]:
    scene, plan = make(worlds, **kw)
    sim = alloy.build(scene, device=DEV, seed=0)
    built = sim.engine.built
    pids = np.asarray(sorted(i for e in built.entries if e.capability == "mpm" for i in e.particles), np.int64)
    pw = built.model.particle_world.numpy()[pids].astype(np.int64)
    worst, step = np.zeros(worlds, np.int64), 0
    for at, drive in plan:
        while step < at:
            sim.step(1)
            step += 1
            worst = np.maximum(worst, _inside(sim, pids, pw, worlds))
        if drive:
            sim.drive(drive[0], velocity=drive[1], duration=drive[2])
    assert sim.check()["ok"]
    return worst, np.ascontiguousarray(sim.engine.state.particle_q.numpy()).tobytes()


def _flat(counts: np.ndarray, label: str) -> None:
    w = np.arange(len(counts))
    corr = float(np.corrcoef(w, counts)[0, 1]) if counts.std() > 0 else 0.0
    print(f"\n{label}: world 0 {counts[0]}, last 8 mean {counts[-8:].mean():.2f}, max {counts.max()}, "
          f"worlds with any {int((counts > 0).sum())}, corr with world {corr:.2f}")
    assert N.installed()
    assert counts[-8:].mean() <= counts[0] + 1.0, (label, counts.tolist())
    assert abs(corr) < 0.3, (label, corr)


def test_a_pressed_dough_block_is_flat_in_the_world_offset_at_64_worlds() -> None:
    counts, _ = _run(plate, 64)
    _flat(counts, "plate x64")


def test_a_free_box_resting_on_sand_is_flat_in_the_world_offset_at_64_worlds() -> None:
    counts, _ = _run(box_on_sand, 64)
    _flat(counts, "box on sand x64")


def test_one_world_is_bit_for_bit_with_warps_lookup(monkeypatch) -> None:
    """One world takes the lookup branch without offsets, which the copy keeps as Warp wrote it."""
    ours, ours_q = _run(plate, 1)
    monkeypatch.setattr(N.NanogridBase, "make_filtered_cell_lookup", N._WARP)
    monkeypatch.setattr(N, "install", lambda: False)
    theirs, theirs_q = _run(plate, 1)
    assert ours.tolist() == theirs.tolist() and ours_q == theirs_q
