# SPDX-License-Identifier: Apache-2.0
"""A forklift plan whose job ends in a wait has finished once the truck rests there (rc1 red, WP5 2026-11-10).

AB-gym-5 judges GA's time_limit from whether the scripted plan finished before the limit. GA's load-trailer with
the trailer restraint red asks the truck to park short of the leveler and wait. Its operator stays in that wait
phase until the limit, so ``Operator.done`` never comes, and the 40 s run of
``test_forklift_ga_run.py::test_load_trailer_with_the_restraint_red_waits_off_the_leveler`` failed on time_limit.
"""

from __future__ import annotations

import pytest

from alloy.tasks.forklift_control import REST_SPEED
from alloy.tasks.forklift_ga_dock import task
from alloy.tasks.forklift_ga_run import WAIT_DWELL, WaitWatch
from alloy.tasks.forklift_operator import Operator


def test_a_plan_that_ends_in_its_wait_has_finished_once_the_truck_rests_there():
    """Finished ``WAIT_DWELL`` after the truck came to rest in the wait phase, and from then on while it stays. Not
    in another phase or while moving. The finish is withdrawn when the truck moves on or leaves the phase (a rest
    that does not last), and a new rest starts a new one. A task with no wait finishes only by ``Operator.done``."""
    w = WaitWatch("restraint_check")
    assert w.update("dock", 0.0, 1.0) is None                                 # at rest, but in another phase
    assert w.update("restraint_check", 2.0 * REST_SPEED, 2.0) is None         # still moving
    assert w.update("restraint_check", 0.0, 3.0) is None
    assert w.update("restraint_check", 0.0, 3.0 + 0.5 * WAIT_DWELL) is None
    assert w.update("restraint_check", 0.0, 3.0 + WAIT_DWELL) == pytest.approx(3.0 + WAIT_DWELL)
    assert w.update("restraint_check", 0.0, 30.0) == pytest.approx(3.0 + WAIT_DWELL)   # still waiting: the same
    assert w.update("restraint_check", 2.0 * REST_SPEED, 31.0) is None        # rests, then moves on: withdrawn
    assert w.update("restraint_check", 0.0, 32.0 + 0.5 * WAIT_DWELL) is None
    assert w.update("in", 0.0, 40.0) is None                                  # left the phase
    assert WaitWatch(None).update("restraint_check", 0.0, 10.0) is None


@pytest.mark.parametrize("secured", [True, False])
def test_only_load_trailer_with_the_restraint_red_ends_in_a_wait(secured):
    """The red-restraint task names its wait phase, and the operator has that phase; the green task has none (it
    finishes by ``Operator.done``)."""
    t = task(secured=secured)
    if secured:
        assert t.waits_at is None
    else:
        assert t.waits_at in Operator(t.plan).phase_list
