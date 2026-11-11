# SPDX-License-Identifier: Apache-2.0
"""SCR-JOB1-3 (a), coordinator ruling 2026-11-04 22:03Z: a stepper that discards the rigid contact stream declares it.

- Background. JOB1's stepper runs Newton's collision pipeline for the particle-shape contacts and then zeroes
  ``rigid_contact_count``, since MuJoCo owns rigid contact. The narrow phase's reducer overflow still flagged
  CONTACT_OVERFLOW in every world, for contacts that reach no solver.
- Off (default): the rigid count over its buffer and the reducer overflow give CONTACT_OVERFLOW / INVALID in every
  world, as before.
- On (``HealthMonitor(rigid_contacts_unused=True)``): both are ignored.
- Soft-contact overflow stays fail closed either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.precision.health import HealthMonitor
from alloy.precision.types import HealthFlags

sys.path.insert(0, str(Path(__file__).parent))
from test_health_pen_measured import POLICY, _boxes  # noqa: E402

OVERFLOW = int(HealthFlags.CONTACT_OVERFLOW)


def _contacts(rigid_over: bool, reducer: bool, soft_over: bool) -> newton.Contacts:
    c = newton.Contacts(4, 4, device="cpu")
    c.rigid_contact_count.assign(np.array([5 if rigid_over else 0], dtype=np.int32))
    c.soft_contact_count.assign(np.array([5 if soft_over else 0], dtype=np.int32))
    c._reduction_overflow = wp.array([7 if reducer else 0], dtype=wp.int32, device="cpu")
    return c


def _flags(unused: bool, contacts: newton.Contacts) -> list[int]:
    model, s0 = _boxes("cpu")
    monitor = HealthMonitor(model, POLICY, rigid_contacts_unused=unused)
    monitor.update(0, s0, s0, contacts)
    return [int(f) & OVERFLOW for f in monitor.poll().flags]


def test_rigid_overflow_fails_closed_by_default():
    assert _flags(False, _contacts(rigid_over=False, reducer=True, soft_over=False)) == [OVERFLOW, OVERFLOW]
    assert _flags(False, _contacts(rigid_over=True, reducer=False, soft_over=False)) == [OVERFLOW, OVERFLOW]


def test_declared_unused_rigid_stream_is_ignored():
    assert _flags(True, _contacts(rigid_over=True, reducer=True, soft_over=False)) == [0, 0]


def test_soft_overflow_stays_fail_closed_when_rigid_is_unused():
    assert _flags(True, _contacts(rigid_over=False, reducer=False, soft_over=True)) == [OVERFLOW, OVERFLOW]
    assert _flags(False, _contacts(rigid_over=False, reducer=False, soft_over=True)) == [OVERFLOW, OVERFLOW]


def test_clean_contacts_raise_nothing_either_way():
    for unused in (False, True):
        assert _flags(unused, _contacts(rigid_over=False, reducer=False, soft_over=False)) == [0, 0]
