# SPDX-License-Identifier: Apache-2.0
"""A world reset clears its health (SCR-WP5-19, coordinator GO 2026-11-08 21:17Z; CPU).

Before: a world that went INVALID stayed latched and frozen after sim.reset() (a blown sheet moved 0.0 m in 0.1 s),
and the run's sticky peaks carried over. RL users reset constantly.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import alloy
from alloy.precision.health_reset import zero_worlds


def _field_bytes(rec) -> dict[str, bytes]:
    """Each field of a structured record as its own bytes (nested structs field by field), so NaN bits count but the
    struct's padding does not: a masked restore may rewrite an untouched world's padding (seen on the test host)."""
    out: dict[str, bytes] = {}
    for name in rec.dtype.names:
        v = rec[name]
        if getattr(v, "dtype", None) is not None and v.dtype.names:
            out.update({f"{name}.{k}": b for k, b in _field_bytes(v).items()})
        else:
            out[name] = np.ascontiguousarray(v).tobytes()
    return out


def _boxes(worlds: int = 2):
    scene = alloy.Scene(dt=0.01, worlds=worlds)
    scene.add_ground()
    scene.add_box("block", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.3), mass=0.2)
    return alloy.build(scene, device="cpu")


def _fling(sim, world: int) -> None:
    """World ``world``'s block to 60 m/s (SPEED_FAIL: INVALID, latched)."""
    eng = sim.engine
    qd = eng.state.joint_qd.numpy()
    qd[eng.model.joint_qd_start.numpy()[world] + 2] = 60.0       # a free joint's qd: linear first
    eng.state.joint_qd.assign(qd)


def test_an_invalid_world_resets_to_healthy_and_steps_again() -> None:
    sim = _boxes()
    sim.step(1)
    _fling(sim, 1)
    sim.step(2)
    assert sim.health()["invalid_worlds"] == [1]
    sim.reset()
    z0 = sim.state()["block.pos"][:, 2].copy()
    sim.step(10)
    h = sim.health()
    assert h["invalid_worlds"] == [] and [w["severity"] for w in h["worlds"]] == ["OK", "OK"]
    assert np.all(sim.state()["block.pos"][:, 2] < z0 - 0.01)                # both fall again
    assert sim.check()["ok"]


def test_a_masked_reset_leaves_the_other_worlds_bit_for_bit() -> None:
    ref, sim = _boxes(), _boxes()
    for s in (ref, sim):
        s.step(1)
    _fling(sim, 1)
    for s in (ref, sim):
        s.step(2)
    record0 = sim.engine.health.record.numpy()[0].copy()
    sim.engine.reset(wp.array(np.array([0, 1], np.int32), dtype=wp.int32))    # world 1 only (Sim.reset: all)
    after = _field_bytes(sim.engine.health.record.numpy()[0])
    assert after == _field_bytes(record0)                                            # world 0's record, every field
    for s in (ref, sim):
        s.step(10)
    assert np.array_equal(sim.state(world=0)["block.pos"], ref.state(world=0)["block.pos"])
    assert sim.health()["invalid_worlds"] == []


def test_a_reset_clears_the_cloth_stretch_peak() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_health_cloth_stretch import _lift, _sheet

    sim = _sheet()
    sim.step(1)
    _lift(sim, 40, 0.5)
    sim.step(1)
    assert sim.health()["worlds"][0]["severity"] == "INVALID"
    sim.reset()
    p0 = sim.state()["sheet.points"][0].copy()
    sim.step(10)
    w = sim.health()["worlds"][0]
    assert w["severity"] == "OK" and "cloth_stretch" not in w and "CLOTH_STRETCH" not in w["flags"]
    assert float(np.abs(sim.state()["sheet.points"][0] - p0).max()) > 0.01         # it moves again


def test_a_counter_source_reset_zeroes_only_the_masked_worlds() -> None:
    counts = wp.array(np.array([3, 5, 7], np.int32), dtype=wp.int32)
    zero_worlds(counts)(wp.array(np.array([0, 1, 0], np.int32), dtype=wp.int32))
    assert counts.numpy().tolist() == [3, 0, 7]
    zero_worlds(counts)(None)
    assert counts.numpy().tolist() == [0, 0, 0]
