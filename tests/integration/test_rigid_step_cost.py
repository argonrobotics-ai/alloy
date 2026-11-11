# SPDX-License-Identifier: Apache-2.0
"""SCR-WP0-RIGIDDT-2 creep guard: what the API adds around the MuJoCo solver step on a rigid-only free-body scene.

Counted on the CPU Warp route (the GPU's code path) as Warp launches + copies + zero_ calls, one GPU graph node each:
the wrapper per physics step (health checks, the latch, the period-start snapshots) and the snapshot launches per
period. On 11-08 the wrapper was 49 ops per step before the fused WorldSnapshot copy and 23 after; the solver step
itself is not pinned here: its cost is the solver settings' (SCR-WP0-RIGIDDT-2)."""

from __future__ import annotations

import collections

import pytest
import warp as wp

import alloy

WRAPPER_MAX = 26     # ops per physics step outside the solver (11-08: health 20.1, latch 2, snapshots 0.25, poll 0.1)
# The period-start snapshots fuse FUSED (16) word slots per launch; since the tunnel tracker (8efe08e56) _good holds
# 22 slots, so 2 launches. The bound catches one launch per slot, not slot growth.
# Measured cost of that one more launch per period: under 0.3 ms on this CPU stack scene at 64 worlds (_good.save
# 1.63-1.98 ms against 1.59-1.66 ms); the whole tunnel watch cost 3.5-3.8 % on one L4 at 1,024 worlds. The scene
# sat at exactly 16 slots, so packing the tracker cannot bring back one launch (FUSED 24 did: 13 ms per save).
SNAPSHOT_MAX = 3     # launches per period of the period-start snapshots (_good 2, health 1)
COUNTED = ("launch", "copy", "zero_")


def _stacks():
    scene = alloy.Scene(worlds=1, dt=0.01)
    scene.add_ground(friction=1.0)
    for i in range(3):                                   # 3 stacks of 2 resting cubes: 6 free bodies
        for k in range(2):
            scene.add_box(f"c{i}{k}", size=(0.1, 0.1, 0.1), pos=(0.3 * i, 0.0, 0.05 + 0.101 * k), density=500.0,
                          friction=1.0)
    return scene


def _counter(monkeypatch):
    """Patch the counted Warp calls; returns (counts by tag, current-tag cell)."""
    counts: collections.Counter = collections.Counter()
    tag = ["wrapper"]

    def counting(name, fn):
        def wrapped(*a, **k):
            counts[(tag[0], name)] += 1
            return fn(*a, **k)
        return wrapped

    monkeypatch.setattr(wp, "launch", counting("launch", wp.launch))
    monkeypatch.setattr(wp, "copy", counting("copy", wp.copy))
    monkeypatch.setattr(wp.array, "zero_", counting("zero_", wp.array.zero_))
    return counts, tag


def _tagged(obj, attr: str, name: str, tag: list[str]) -> None:
    fn = getattr(obj, attr)

    def run(*a, **k):
        prev, tag[0] = tag[0], name
        try:
            return fn(*a, **k)
        finally:
            tag[0] = prev
    setattr(obj, attr, run)


def test_the_api_wrapper_around_a_rigid_step_stays_small(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    sim = alloy.build(_stacks(), device="cpu")
    eng = sim.engine
    sim.step(3)                                          # warm: the start checks and first-step set-up
    counts, tag = _counter(monkeypatch)
    _tagged(eng.stepper, "step", "solver", tag)
    for snap in {id(s): s for s in (eng._good, eng._health_snap) if s is not None}.values():
        _tagged(snap, "save", "snapshot", tag)
    periods = 3
    steps0 = sim.physics_steps
    sim.step(periods)
    steps = sim.physics_steps - steps0
    ops = {t: sum(n for (tt, kind), n in counts.items() if tt == t and kind in COUNTED)
           for t in ("wrapper", "solver", "snapshot")}
    wrapper = (ops["wrapper"] + ops["snapshot"]) / steps
    assert ops["solver"] > 0 and steps == periods * eng.clock.steps_per_period
    assert ops["snapshot"] / periods <= SNAPSHOT_MAX, ops
    assert wrapper <= WRAPPER_MAX, (f"{wrapper:.1f} wrapper ops per physics step (bound {WRAPPER_MAX}): something "
                                    f"now launches per step around the solver; counts {dict(counts)}")
    assert sim.check()["ok"]
