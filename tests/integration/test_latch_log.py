# SPDX-License-Identifier: Apache-2.0
"""SCR-JOB2-4: a world the health monitor latches is named once, with its flags and first bad step.

A synthetic health record (``StepHealth`` per world) on CPU (read at once) and on CUDA (read one poll later, from a
pinned copy behind an event). The scene path (a real latch in a coupled scene) is in ``test_rollback_latch.py``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.integration.latch_log import LatchLog
from alloy.precision.types import HealthFlags, StepHealth

W = 4


def _record(device: str) -> wp.array:
    h = wp.zeros(W, dtype=StepHealth, device="cpu").numpy()
    h["first_bad_step"] = -1
    return wp.array(h, dtype=StepHealth, device=device)


def _latch(record: wp.array, world: int, flags: HealthFlags, step: int, **fail) -> None:
    """Latch ``world``; ``fail`` fills WP5's first-failure snapshot (``fail_<field>``), with ``fail_flags`` = flags."""
    h = record.numpy()
    h["flags"][world] |= int(flags | HealthFlags.LATCHED)
    h["first_bad_step"][world] = step
    if fail:
        h["fail_flags"][world] = int(flags)
        for name, value in fail.items():
            h[f"fail_{name}"][world] = value
    record.assign(wp.array(h, dtype=StepHealth, device=record.device))


def test_for_health_finds_the_record_or_none():
    rec = _record("cpu")
    assert LatchLog.for_health(SimpleNamespace(record=rec, health=_record("cpu"))).record is rec   # WP5 monitor
    assert LatchLog.for_health(SimpleNamespace(health=rec)).record is rec                         # FakeHealth
    assert LatchLog.for_health(None) is None
    assert LatchLog.for_health(SimpleNamespace(health=wp.zeros(W, dtype=wp.int32))) is None


def test_cpu_warns_once_per_world_with_flags(caplog):
    rec = _record("cpu")
    ll = LatchLog(rec)
    with caplog.at_level(logging.WARNING, logger="alloy.integration"):
        assert ll.poll() == []
        _latch(rec, 2, HealthFlags.PEN_FAIL | HealthFlags.PEN_WARN, 47, time_s=0.029, pen_m=2.04e-3, shape_a=13,
               shape_b=-1, particle=812, speed_mps=0.3, body=-1)
        assert ll.poll() == [2]
        assert ll.poll() == []                                    # once per world
        _latch(rec, 0, HealthFlags.SPEED_FAIL, 90)
        _latch(rec, 3, HealthFlags.SPEED_FAIL, 95)
        assert ll.poll() == [0, 3]
    lines = [r.getMessage() for r in caplog.records]
    assert len(lines) == 2, lines
    assert "health latched 1 world(s) [2]: PEN_WARN|PEN_FAIL from step 47" in lines[0]
    assert ("first failure of world 2: check 47 (t 0.029 s): PEN_WARN|PEN_FAIL; pen 0.00204 m (particle 812 / "
            "shape 13)") in lines[0]
    assert "health latched 2 world(s) [0, 3]: SPEED_FAIL from step 90..95" in lines[1]
    assert "first failure" not in lines[1]                    # no snapshot written for these worlds
    assert ll.latched == {0, 2, 3}


def test_reset_forgets_the_reset_worlds(caplog):
    rec = _record("cpu")
    ll = LatchLog(rec)
    _latch(rec, 1, HealthFlags.NONFINITE_Q, 3)
    _latch(rec, 2, HealthFlags.NONFINITE_Q, 3)
    assert ll.poll() == [1, 2]
    ll.reset(wp.array(np.array([0, 1, 0, 0], np.int32), dtype=wp.int32))
    assert ll.latched == {2}
    assert ll.poll() == [1]                 # world 1 is still latched in the record (a reset world re-latched)
    ll.reset()
    assert ll.latched == set()


@pytest.mark.gpu
def test_cuda_reads_the_previous_poll_without_a_sync():
    if not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    rec = _record("cuda:0")
    ll = LatchLog(rec)
    assert ll.poll() == []                  # nothing copied yet
    _latch(rec, 3, HealthFlags.PEN_FAIL, 12)
    assert ll.poll() == []                  # reads the copy made before the latch, enqueues a fresh one
    assert ll.poll() == [3]                 # one poll (period) later
    assert ll.poll() == []
