# SPDX-License-Identifier: Apache-2.0
"""SCR-JOB2-1 item 2: a coupled scene with precision pairs runs its MuJoCo entry at one step per exchange (CPU).

``make_entries`` pins the entry (``tests/scene/test_entries_pin.py``); here the exchange interval is capped at its
target dt, and a world whose stability estimate asks for less raises the named DEGRADED flag, never a substep.
"""

from __future__ import annotations

import dataclasses
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.coupling.types import ClockPolicy, SchedulerPolicy
from alloy.integration import precision_pin as P
from alloy.testing import integration as I


def test_pinned_entries_and_the_exchange_cap():
    built = I.dc5_like(1, "cpu")                          # MuJoCo plug (target 1 ms, rung 1) + VBD pad, a pair
    assert P.pinned_entries(built) == {"mujoco": 1.0e-3}
    pol = P.pin_exchange(SchedulerPolicy(), built)
    assert pol.clock.dt_x_max == 1.0e-3
    finer = SchedulerPolicy(clock=ClockPolicy(dt_x_max=5.0e-4))
    assert P.pin_exchange(finer, built) is finer          # already finer: unchanged


def test_no_pin_without_pairs_or_with_a_substepping_entry():
    built = I.dc5_like(1, "cpu")
    bare = dataclasses.replace(built, precision_pairs=())
    assert P.pinned_entries(bare) == {} and P.pin_exchange(SchedulerPolicy(), bare) == SchedulerPolicy()
    mj, pad = built.entries
    ladder = dataclasses.replace(built, entries=[dataclasses.replace(mj, max_substeps=64), pad])
    assert P.pinned_entries(ladder) == {}
    single = dataclasses.replace(built, entries=[mj])
    assert P.RungPinWatch.for_built(single, SchedulerPolicy()) is None


def _report(dt_x: float, stable: dict | None) -> SimpleNamespace:
    return SimpleNamespace(dt_x=dt_x, notes={} if stable is None else {"stable_dt": stable})


def test_watch_flags_the_worlds_that_ask_for_a_smaller_step(caplog):
    watch = P.RungPinWatch({"mujoco": 1.0e-3}, ClockPolicy(safety=0.5), 3, "cpu")
    with caplog.at_level(logging.WARNING, logger="alloy.integration"):
        mask = watch.update(_report(1.0e-3, {"mujoco": np.array([4e-3, 1.5e-3, np.inf]), "pad": np.zeros(3)}))
    np.testing.assert_array_equal(mask, [False, True, False])          # 1 ms > 0.5 x 1.5 ms only in world 1
    np.testing.assert_array_equal(watch.flags.numpy(), [0, 1, 0])
    assert "precision rung pinned: 1 world(s)" in caplog.text
    watch.update(_report(1.0e-3, {"mujoco": np.full(3, 4e-3)}))
    np.testing.assert_array_equal(watch.flags.numpy(), [0, 0, 0])      # cleared when the estimate recovers


def test_watch_without_an_estimate_flags_nothing(caplog):
    watch = P.RungPinWatch({"mujoco": 1.0e-3}, ClockPolicy(), 2, "cpu")
    with caplog.at_level(logging.WARNING, logger="alloy.integration"):
        watch.update(_report(1.0e-3, None))
        watch.update(_report(1.0e-3, None))
    assert not watch.flagged.any() and caplog.text.count("no per-world stability estimate") == 1


def test_health_source_uses_the_named_flag_when_it_exists():
    from alloy.precision.types import HealthFlags  # noqa: PLC0415

    watch = P.RungPinWatch({"mujoco": 1.0e-3}, ClockPolicy(), 2, "cpu")
    src = watch.health_source()
    if not hasattr(HealthFlags, "RUNG_PINNED"):
        pytest.skip("HealthFlags.RUNG_PINNED not on this main yet (WP5); the source is None until it lands")
    assert src == (watch.flags, int(HealthFlags.RUNG_PINNED))


def test_watch_names_a_pinned_entry_without_an_estimate(caplog):
    watch = P.RungPinWatch({"mujoco": 1.0e-3}, ClockPolicy(), 2, "cpu")
    with caplog.at_level(logging.WARNING, logger="alloy.integration"):
        watch.update(_report(1.0e-3, {"pad": np.zeros(2)}))       # no estimator registered for the MuJoCo entry
        watch.update(_report(1.0e-3, {"pad": np.zeros(2)}))
    assert not watch.flagged.any() and caplog.text.count("entry 'mujoco' has no stability estimate") == 1
