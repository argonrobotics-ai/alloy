# SPDX-License-Identifier: Apache-2.0
"""PUN-9 determinism: events, lists and gathers run in fixed index order with integer-only atomics, so a puncture
run repeats bitwise, and identical worlds of a batch end bitwise identical (per-world kernels, no cross-world sums).
With suture threads, each world's thread ends bitwise equal to a one-world run's: a thread's chain step runs only in its
own colour, when Newton has accumulated that thread's force and Hessian."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.solvers.puncture import PunctureConfig, SolverPuncture, TissueMaterial
from alloy.solvers.puncture.scenes import Block, Drive, Thread, build_block

pytestmark = pytest.mark.compile_heavy
BL = Block(size=4e-3, thick=3e-3, material=TissueMaterial(fp=0.3, fcut=0.1, flen=50.0))
BL_THREAD = Block(size=4e-3, thick=2e-3, material=TissueMaterial(fp=0.3, fcut=0.1, flen=50.0), fix_sides=True,
                  needle_length=2.5e-3, thread=Thread(count=10, flen=50.0), from_below=True)


def _run(worlds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with wp.ScopedDevice("cpu"):
        m, specs = build_block(BL, worlds=worlds)
        s = SolverPuncture(m, puncture=PunctureConfig(), needles=specs, iterations=10,
                           integrate_with_external_rigid_solver=True)
        d = Drive(m, s, 5e-4)
        for _ in range(round((BL.gap + 1.5e-3) / 1e-2 / 5e-4)):
            d.step(lambda t: 1e-2 * t, BL.gap, record=False)
        return d.s0.particle_q.numpy(), s.anchors.w.numpy(), s.tracks.punctures.numpy()


def test_pun9_repeat_is_bitwise() -> None:
    a, b = _run(2), _run(2)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert a[2].tolist() == [1, 1]


def test_pun9_identical_worlds_match() -> None:
    q, w, punct = _run(2)
    half = q.shape[0] // 2
    assert np.array_equal(q[:half], q[half:])
    A = PunctureConfig().anchors_per_world
    assert np.array_equal(w[:A], w[A:2 * A])


def _run_threads(worlds: int, steps: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Needle and thread driven up into the block from below; per world: its particles, the thread's start and end."""
    with wp.ScopedDevice("cpu"):
        m, specs = build_block(BL_THREAD, worlds=worlds)
        s = SolverPuncture(m, puncture=PunctureConfig(), needles=specs, iterations=10,
                           integrate_with_external_rigid_solver=True)
        d = Drive(m, s, 2.5e-4, up_axis=True)
        start = d.s0.particle_q.numpy().copy()
        for _ in range(steps):
            d.step(lambda t: 2e-2 * t, BL_THREAD.gap, record=False)
        q = d.s0.particle_q.numpy()
    per = q.shape[0] // worlds
    threads = [slice(sp.thread_start, sp.thread_start + sp.thread_count) for sp in specs]
    return (q.reshape(worlds, per, 3), np.stack([start[t] for t in threads]), np.stack([q[t] for t in threads]))


def test_pun9_threads_of_a_batch_match_one_world_runs() -> None:
    one, start, end = _run_threads(1)
    assert np.abs(end[0] - start[0]).max() > 1e-5                    # the thread moved (drive and gravity)
    two, _, _ = _run_threads(2)
    for w in range(2):
        assert np.array_equal(two[w], one[0]), (w, float(np.abs(two[w] - one[0]).max()))


def test_pun9_a_thread_colour_steps_only_its_own_thread() -> None:
    """The chain step of thread 0's colour leaves thread 1's force and Hessian at zero. Newton zeroes them once per
    iteration and accumulates thread 1's terms only in thread 1's colour; a write here (Newton's steered I - m/dt^2
    and Delta - f0) would stand in for thread 1's inertia and gravity in its own solve."""
    with wp.ScopedDevice("cpu"):
        m, specs = build_block(BL_THREAD, worlds=2)
        s = SolverPuncture(m, puncture=PunctureConfig(), needles=specs, iterations=10,
                           integrate_with_external_rigid_solver=True)
        d = Drive(m, s, 2.5e-4, up_axis=True)
        d.step(lambda t: 2e-2 * t, BL_THREAD.gap, record=False)       # a real step: the poses, dt and inertia
        groups = [g.numpy() for g in m.particle_color_groups]
        colour0 = next(c for c, g in enumerate(groups) if g.size and int(g[0]) == specs[0].thread_start)
        s.particle_forces.zero_()
        s.particle_hessians.zero_()
        s._after_accumulate(s, colour0, d.s0, None)
        f, h = s.particle_forces.numpy(), s.particle_hessians.numpy()
    t0, t1 = (slice(sp.thread_start, sp.thread_start + sp.thread_count) for sp in specs)
    assert np.abs(h[t0]).max() > 0.0                                   # thread 0 was stepped
    assert not f[t1].any() and not h[t1].any(), (np.abs(f[t1]).max(), np.abs(h[t1]).max())
