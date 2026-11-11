# SPDX-License-Identifier: Apache-2.0
"""E09 Go2 stand: rest posture, joint-target offsets, falls and resets in the random phase.

Derivation (numbers of the request; nothing here depends on the engine's frames):
- Stand (zero action, 3 s from a fresh reset; the 2 s "rest" phase is held to the same height / tilt / drift test): the
  trunk height (base_pos z) stays within 0.24-0.40 m on every row, the tilt (angle of the trunk's z axis from world up,
  from base_quat wxyz) stays < 0.15 rad, and the trunk moves < 5 cm horizontally from its first row. The Go2 standing
  trunk height is about 0.27-0.32 m (the episode ends below 0.18 m).
- Act: +1 on every joint offsets each target by +0.3 rad from the standing pose. Joint angle change from the first row
  to the last of the 1 s phase, median over worlds. A position servo carrying the robot's weight reaches 0.3 rad plus or
  minus the load's sag (load torque / servo stiffness: a few 0.1 rad for a thigh or calf with the stiffness of
  Menagerie's Go2), so the request's "0.15-0.35 on every joint" holds for the unloaded hips only. Checked: the median
  over the 12 joints lies in 0.15-0.35, and no joint changes by less than 0.1 or by more than 0.75 rad. No done in the phase.
- Random: the request's "at least one world falls" is not decidable: independent uniform offsets of +-0.3 rad every
  control step average out in a servo and a Go2 stays up (the Alloy reference: lowest trunk 0.217 m in 2 x 10 s x 16
  worlds). Replaced by termination consistency: no row that is not a done row has the trunk below 0.18 m or tilted
  past 1 rad (such a row must end the episode), and every done row's new start state is standing (height 0.24-0.40,
  tilt < 0.15 rad) in >= 95 % of done rows (one world of 16 may not be wrong).
"""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

HEIGHT_LO, HEIGHT_HI, TILT_MAX, DRIFT_MAX = 0.24, 0.40, 0.15, 0.05
OFFSET_LO, OFFSET_HI = 0.15, 0.35
JOINT_MIN, JOINT_MAX = 0.1, 0.75
STANDING_SHARE = 0.95
FALL_HEIGHT, FALL_TILT = 0.18, 1.0


def tilt(ph: c.Phase) -> np.ndarray:
    return c.tilt_from_quat_wxyz(ph.probe["base_quat"])


def standing(ph: c.Phase) -> np.ndarray:
    """(T, N) rows whose height and tilt are those of a standing robot."""
    z = ph.probe["base_pos"][..., 2]
    return (z >= HEIGHT_LO) & (z <= HEIGHT_HI) & (tilt(ph) < TILT_MAX)


def _stand_checks(ph: c.Phase, tag: str) -> list[c.Check]:
    pos = ph.probe["base_pos"]
    drift = np.linalg.norm((pos - pos[0:1])[..., :2], axis=-1).max(0)
    return [c.at_least(f"{tag}_standing_rows", float(standing(ph).mean(0).min()), 1.0, 2),
            c.within(f"{tag}_drift", drift, 0.0, DRIFT_MAX, 2)]


def _act_checks(ph: c.Phase) -> list[c.Check]:
    q = ph.probe["joint_pos"]
    change = np.median(q[-1] - q[0], axis=0)
    return [c.within("act_joint_change_median", float(np.median(change)), OFFSET_LO, OFFSET_HI, 2),
            c.within("act_joint_change_each", change, JOINT_MIN, JOINT_MAX, 2),
            c.at_most("act_no_done", float(ph.done[:-1].sum()), 0.0, 2)]


def _random_checks(r: c.Rollout) -> list[c.Check]:
    ph = r.phase("random")
    rows = ph.done
    share = float(standing(ph)[rows].mean()) if rows.any() else 0.0
    z = ph.probe["base_pos"][..., 2]
    fallen = (z < FALL_HEIGHT) | (tilt(ph) > FALL_TILT)
    unterminated = float((fallen & ~rows).sum())
    return [c.at_most("random_unterminated_fall_rows", unterminated, 0.0, 2),
            c.at_least("random_resets_standing", share, STANDING_SHARE, 2)]


def check(rollout: c.Rollout, request: dict) -> c.CheckResult:
    checks = c.generic_checks(rollout, request)
    checks += _stand_checks(rollout.phase("rest"), "rest") + _stand_checks(rollout.phase("stand"), "stand")
    checks += _act_checks(rollout.phase("act_plus")) + _random_checks(rollout)
    return c.finish(checks)
