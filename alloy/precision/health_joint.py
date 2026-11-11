# SPDX-License-Identifier: Apache-2.0
"""Joint and weld violation in the health monitor: when it is measured, and the hold between measurements.

Joints and welds are measured on "full" checks only (``CheckPolicy.level == "full"``, on physics steps that are a
multiple of ``full_every``: every 10th step under the standard and fast profiles, every step under precision; before
audit ERR-13 the fast profile's checks were "cheap" and never measured them). The record must still say JOINT_FAIL on the checks in between,
or a read at the end of a control period never sees it (audit ERR-11: a weld pulled 7-70 mm apart read OK, because
the next check erased the flag and a 10 ms period, 8 checks, never ends on a measuring one). ``measure`` returns the
hold flag ``health_kernels.classify`` takes: 1 keeps the last measured JOINT_FAIL bit and joint errors in the record.
A weld that closes again clears on the next measurement (JOINT_FAIL is DEGRADED: not sticky), and a world reset
between two measurements clears the held bit with the rest of its record.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from alloy._vendor import newton
from alloy.precision import health_physics as hp

FLOOR_M = 1.0e-3      # thresholds.joint_m / joint_rad None: the flat floor for tree and loop joints (SPEC 4.3)
FLOOR_RAD = 1.0e-3


def measure(monitor: Any, step: int, state: newton.State) -> int:
    """Measure joints and welds into the monitor's accumulators if check ``step`` is a measuring one.

    Returns 1 when joints are checked under this policy but not on this step (``classify`` then holds the last
    verdict), else 0. Example: ``hold = health_joint.measure(self, step, state_out)``."""
    cfg = monitor.cfg
    if cfg.level != "full" or "joints" in cfg.skip:
        return 0
    if step % max(cfg.full_every, 1) != 0:
        return 1
    monitor._check_joints(state)          # = launch(monitor, state); a method so tests can count the calls
    return 0


def launch(monitor: Any, state: newton.State) -> None:
    """The joint and weld kernels on ``state`` (every judged joint, every connect / weld row)."""
    m, dev, t, acc = monitor.model, monitor.device, monitor.thresholds, monitor._acc
    if m.joint_count and m.body_count:
        wp.launch(hp.check_joints, dim=m.joint_count,
                  inputs=[m.joint_type, m.joint_parent, m.joint_child, m.joint_X_p, m.joint_X_c, m.joint_axis,
                          m.joint_qd_start, m.joint_world, state.body_q,
                          t.joint_m if t.joint_m is not None else FLOOR_M,
                          t.joint_rad if t.joint_rad is not None else FLOOR_RAD, acc], device=dev)
    welds = monitor._welds
    if welds is not None and m.body_count:
        wp.launch(hp.check_welds, dim=welds.kind.shape[0],
                  inputs=[welds, m.mujoco.equality_constraint_enabled, state.body_q, acc], device=dev)
