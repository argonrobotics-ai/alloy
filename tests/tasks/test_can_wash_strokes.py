# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can's robot stage: a whole revolution of the can in the UR10e's strokes on the turntable (fast:
the grip arc's geometry; CPU: four strokes of the empty can)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alloy.tasks.can_wash_strokes import (ARC_CENTRE, STROKE, arc_clearance, grip_azimuth, nozzle_azimuths,
                                         start_phi, wrap)
from alloy.tasks.can_wash_table import Turntable
from alloy.tasks.can_wash_tip import COARSE
from alloy.tasks.can_wash_turn import Lathe

LATHE = Lathe(dataclasses.replace(COARSE, table=Turntable()))


def test_the_grip_arc_stays_clear_of_the_fr3s_jets_and_nozzle():
    """Each stroke's grip runs over STROKE centred at ARC_CENTRE (95-155°), 55° or more from both aims (the inside
    jets' far wall at 38°, the outside wall at 278°) and 60° from the inside tip (over the can at 218°, 0.13 m in and
    0.3 m over the rim)."""
    a0 = grip_azimuth(LATHE, start_phi(LATHE))
    assert abs(wrap(a0 + 0.5 * STROKE - ARC_CENTRE)) < 1e-9
    az = nozzle_azimuths()
    assert min(arc_clearance(LATHE, az[k]) for k in ("inside aim", "outside aim")) > math.radians(55.0)
    assert arc_clearance(LATHE, az["inside tip"]) > math.radians(60.0)


def test_the_grip_turns_with_the_can():
    """A turn of the can by phi moves the grip's azimuth by phi."""
    for phi in (0.4, -1.1, 2.5):
        assert wrap(grip_azimuth(LATHE, phi) - grip_azimuth(LATHE, 0.0) - phi) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.compile_heavy
@pytest.mark.timeout(1800)
def test_strokes_turn_the_empty_can_one_whole_revolution():
    """60° strokes at 1 rad/s and a closing one for what is left (``revolve_to``): the can turns 360° within 0.5°
    (each full stroke within 6°: the pinch slips now and then, booked as WP15 asked), no stroke nears tipping it
    (tip ratio under 0.5), it slips under 3 mm on the table (as in the robot's run) and under 30 mm in the pads, its
    axis drifts under 5 mm, and it turns under 3° while the UR is off it (the swing; no brake here, no water aboard).
    Warp CPU route with the table's lip 4 mm clear, 7 strokes on the 95-155° arc (11-09): 58.9-60.4° a stroke and a
    closing one of 3.4°, pads 0.2-2.2 mm, tip ratio 0.31-0.37, slip 1.3-2.8 mm, drift under 3.1 mm, the axis 1.3-4.1
    mm off, the lip 0 N. The pinch holds the rim now, so what slips is the foot on the plate: MuJoCo Warp's contact
    creeps under the plate's drag (2.6 mm in the quarter turn against 0.33 mm on MuJoCo C), and the can pivots about
    the pinch. Before the jaws opened at ``RELEASE_N`` they hung shut between strokes: pads slipped up to 25 mm and
    the can rode the lip (0-10.2 N). 90° strokes lifted the wrist into the FR3's valve at the arc's end (-27 mm); 60°
    keeps the arms 82 mm or more apart."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415
    from alloy.tasks.can_wash_arms import ArmsBay, Rigs  # noqa: PLC0415
    from alloy.tasks.can_wash_strokes import StrokePlan, revolve_to  # noqa: PLC0415
    from alloy.tasks.can_wash_turn import LatheRun  # noqa: PLC0415
    from alloy.tasks.paint_arm import Plan, Tracker  # noqa: PLC0415

    task, rigs, pl = LATHE.task, Rigs.make(), Plan()
    plan = StrokePlan.make(LATHE, Tracker(rigs.ur, pl), 1.0)
    assert plan.misses == 0 and pl.near_limits == 0
    assert np.abs(plan.swing[-1] - plan.down[0]).max() < 1e-3          # the swing ends where the next stroke starts
    bay = ArmsBay(task, rigs, plan.down[0], np.asarray(rigs.fr3.specs[0].home, float))
    run = LatheRun(task, rigs.ur, build_simulation(bay.scene(lambda s: None, 1), [], "cpu"))
    rev = revolve_to(run, LATHE, Tracker(rigs.ur, pl), 1.0)
    print([dataclasses.asdict(s) for s in rev.strokes])
    assert abs(rev.turned - 2.0 * math.pi) < math.radians(0.5), math.degrees(rev.turned)
    assert 7 <= len(rev.strokes) <= 8 and pl.misses == 0      # a stroke the pinch slips turns short: one more
    assert all(abs(s.turned - STROKE) < math.radians(6.0) for s in rev.strokes[:6]), [s.turned for s in rev.strokes]
    assert rev.worst("tip_ratio") < 0.5 and rev.worst("slip") < 3e-3 and rev.worst("pad_slip") < 30e-3
    assert rev.worst("drift") < 5e-3
    assert max(abs(s.idle) for s in rev.strokes) < math.radians(3.0)     # it stays put while the UR is off it
    weight = run.mass * 9.81                    # the lip centres the can; it never bears half the can's weight
    assert all(np.isfinite(s.lip_force) and s.lip_force < 0.5 * weight for s in rev.strokes), [
        (s.lip_force, s.lip_touch) for s in rev.strokes]


def test_strokes_that_never_turn_the_can_raise():
    """``revolve_to`` stops with ``RevolutionError`` when three revolutions' worth of strokes leave the can short."""
    from types import SimpleNamespace  # noqa: PLC0415

    from alloy.tasks.can_wash_strokes import RevolutionError, StrokePlan, revolve_to  # noqa: PLC0415
    from alloy.tasks.can_wash_table import TurnLog  # noqa: PLC0415

    q = np.zeros(6)
    stuck = SimpleNamespace(step=lambda *a: None, turn=lambda qs, settle=0: TurnLog(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    plan = StrokePlan([q], [q], [q], [q])
    with pytest.raises(RevolutionError, match="0.00 of 360.00 deg in 18 strokes"):
        revolve_to(stuck, LATHE, None, 1.0, full=plan)
