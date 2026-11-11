# SPDX-License-Identifier: Apache-2.0
"""BUL-0014 squeeze law reference (WP13 x WP3): MuJoCo C runs of ``library/robots/task_squeeze_{wedge,face}``.

The scenes are the minimal repro of BUL-0014 (T12, SB-7 trip 100100:316: a fingertip drives the 12.7 g block_3 into
the plywood sheet's edge; 11.45 mm on the GPU, reproduced by MuJoCo C from WP7's recorded window). They carry the
contact parameters T12 gets on Alloy's route, and keyframe ``init`` holds the start state; the tip is a 0.3 kg body
(T12's left fingertip effective mass along its velocity at BUL-0014's window, 0.29-0.34 kg; a 1 kg stand-in until the
coordinator's 2026-11-05 ruling) on three world-axis slides, servoed (kp 20000 N/m, kv 300 N s/m, force cap 30 N per
axis) along a straight path:

- ``wedge`` (case B): block_3 flicked toward the sheet's top edge (recorded state 1.4 ms before contact), the tip
  following at 1.27 m/s for 55 ms, then held. The deep contact is edge-edge (block bottom edge on the sheet corner).
- ``face`` (case A, control): block_3 at rest flat on the table against the sheet's side face, the tip pushing it
  face-on at 0.05 m/s until the 30 N cap holds it.

WP3's law (alloy/contact, tests/contact) gates the peak block-sheet depth <= 1 mm on Alloy's route; this module is
the MuJoCo C reference those numbers come from. Scene generator: [internal note].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from alloy.core.errors import TraceError
from alloy.cards.library import default_root

log = logging.getLogger("alloy.tasks")
ROBOTS = default_root() / "robots"


@dataclass(frozen=True)
class SqueezeDrive:
    velocity: tuple[float, float, float]    # [m/s] servo target velocity (world)
    push_s: float                           # [s] target moves for this long, then holds
    hold_s: float = 0.2                     # [s] run on after the push


@dataclass(frozen=True)
class SqueezeResult:
    peak_mm: float          # deepest block-sheet contact over the run [mm, > 0 = penetrating]
    peak_force_n: float     # normal force of that contact at that step [N]
    end_mm: float           # deepest block-sheet contact at the last step [mm]
    max_servo_n: float      # largest tip actuator force [N] (the cap is 30 N)


DRIVES = {
    "wedge": SqueezeDrive((0.948555, -0.847977, 0.078762), 0.055),
    "face": SqueezeDrive((0.0, -0.05, 0.0), 0.6),
}


def scene_path(case: str) -> Path:
    if case not in DRIVES:
        raise TraceError(f"no squeeze case {case!r}; known: {sorted(DRIVES)}")
    return ROBOTS / f"task_squeeze_{case}" / "scene.xml"


def _block_sheet(m: mujoco.MjModel, d: mujoco.MjData, sheet: int, block: int) -> tuple[float, float]:
    """(deepest dist [m], its normal force [N]) over the block-sheet contacts this step; (0, 0) without one."""
    worst, force = 0.0, 0.0
    f6 = np.zeros(6)
    for j in range(d.ncon):
        c = d.contact[j]
        if sheet in c.geom and block in {int(m.geom_bodyid[g]) for g in c.geom} and c.dist < worst:
            mujoco.mj_contactForce(m, d, j, f6)
            worst, force = float(c.dist), float(f6[0])
    return worst, force


def run_reference(case: str) -> SqueezeResult:
    """MuJoCo C (float64) from keyframe ``init``, the tip servoed along the case's straight path."""
    drive = DRIVES[case]
    m = mujoco.MjModel.from_xml_path(str(scene_path(case)))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("init").id)
    sheet, block = m.geom("sheet").id, m.body("block").id
    v, dt = np.asarray(drive.velocity), float(m.opt.timestep)
    peak, peak_f, end, servo = 0.0, 0.0, 0.0, 0.0
    for i in range(int(round((drive.push_s + drive.hold_s) / dt))):
        d.ctrl[:] = v * min((i + 1) * dt, drive.push_s)
        mujoco.mj_step(m, d)
        end, f = _block_sheet(m, d, sheet, block)
        if end < peak:
            peak, peak_f = end, f
        servo = max(servo, float(np.abs(d.actuator_force).max()))
    res = SqueezeResult(-peak * 1e3, peak_f, -end * 1e3, servo)
    log.info("squeeze %s (MuJoCo C): %s", case, res)
    return res
