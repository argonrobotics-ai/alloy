# SPDX-License-Identifier: Apache-2.0
"""The scripted policy of task rinse_pan: scrub the soaked egg off the pan's floor with the sponge, spray the band
where the floor meets the wall, rinse, sanitize, and set the pan mouth down on the shelf.

Geometry decides the split. The egg covers the inner floor (r < 97.5 mm) and the wall up to 8 mm (r < 99.9 mm),
where the s^1.5 flare rises almost vertically off the floor. A flat 111 x 66 mm pad slid along its short axis through
the centre covers a strip |x| < 55.5 mm, |y| < 77.7 mm with its corners 2 mm inside the floor's edge; four such strips
(0°, 45°, 90°, 135°) cover the floor to r 77 mm. The ring beyond, and the band up the wall, are left to the spray
valve's jets (28 m/s: an impact pressure far above soaked egg's 300 Pa), tilted out (``SPRAY_LAPS``) while the tip
circles them.

Hands (privileged, ``rinse_hand.PoseHand``): one holds the pan (its weight carried, 2 kN/m), one the sponge or the
valve (soft: the sponge weighs 80 g; the press is the hand's spring, ``press / kp`` below the face's contact).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from alloy.tasks.rinse_dunk import DunkRunner
from alloy.tasks.rinse_hand import PoseHand
from alloy.tasks.rinse_plates_scene import Pose, Step, move
from alloy.tasks.rinse_pan_scene import Y_BACK, Station

VALVE = 1                             # the tool hand's bodies: sponge 0, valve 1
TOOL_KP = 300.0                       # N/m, the tool hand's stiffness (rinse_pan.plugins)
STRIP = 0.0447                        # m, the slide either side of the centre that keeps the pad's corners on the floor
POUR_SIDE = 0.08                      # m, the pan's centre -x of the bowl's while it pours toward +x
POUR_TURN = math.radians(110.0)       # the roll about the handle's axis (world y) that empties the pan
SPRAY_UP = 0.08                       # m, the valve's tip over the pan's floor while spraying
CLEAR_UP = SPRAY_UP + 0.15            # m, the tip over the floor while the pan rolls (the rolled rim peaks ~135 mm)
SETTLE_S = 1.0                        # s, the valve's moves to a lap's start and clear of the pan
RAMP_S = 0.5                          # s, the valve's lever squeezed or let go at a lap's start and end
# (tip circle radius [m], tilt out from the pan's axis): the centre jet meets the wall band at r 0.0987, z 4 mm, the
# floor at r 0.091 and at r 0.083 (r0 = r_hit - (SPRAY_UP - z_hit)·tan(tilt)). Two laps (the wall, r 0.083) left one
# 1 cm cell at r 92 mm, 150° round, unhit (6,081 µg/cm²): the nine jets keep their heading while the tip circles,
# so at some angles their fan spans less of the ring between the laps (notes/wp8-results/surface/pan_review/).
SPRAY_LAPS = ((0.055, math.radians(30.0)), (0.058, math.radians(22.5)), (0.062, math.radians(15.0)))


def quat_z(angle: float) -> np.ndarray:
    return np.array([0.0, 0.0, math.sin(0.5 * angle), math.cos(0.5 * angle)])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


@dataclass
class PanRunner(DunkRunner):
    """``DunkRunner`` (body 0: the pan) plus a tool hand ``tools`` over the sponge (0) and the valve (1)."""

    tools: PoseHand | None = None
    balancer: PoseHand | None = None
    tool_pos: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __post_init__(self) -> None:
        super().__post_init__()
        q = self.sim.state.body_q.numpy()[self.tools.held.numpy()]
        self.tool_pos, self.tool_quat = q[..., :3].copy(), q[..., 3:].copy()
        self.tool_vel = np.zeros_like(self.tool_pos)
        self.tool_grip = np.zeros(self.tool_pos.shape[:2], np.float32)

    def step(self, moves: dict[int, Step]) -> None:
        self.tool_vel[:] = 0.0
        for i, s in moves.items():
            if i >= 10:                                           # tool moves are keyed 10 + tool index
                k = i - 10
                self.tool_pos[:, k], self.tool_quat[:, k], self.tool_vel[:, k] = s.pose.pos, s.pose.quat, s.vel
        for name, value in (("target_pos", self.tool_pos), ("target_quat", self.tool_quat),
                            ("target_vel", self.tool_vel), ("grip", self.tool_grip)):
            self.tools.set_input(name, value.astype(np.float32))
        super().step({i: s for i, s in moves.items() if i < 10})

    def grip_tool(self, k: int, on: bool) -> None:
        """The tool hand takes (``on``) or lets go of tool ``k``; the valve's balancer lets go while the hand holds it
        and takes it back after (its weight is carried once)."""
        self.tool_grip[:, k] = 1.0 if on else 0.0
        if k == VALVE and self.balancer is not None:
            self.balancer.set_input("grip", 0.0 if on else 1.0)

    def tool_pose(self, k: int) -> Pose:
        return Pose(self.tool_pos[0, k].astype(np.float64), self.tool_quat[0, k].astype(np.float64))

    def go_tool(self, k: int, target: Pose, seconds: float) -> None:
        for s in move(self.tool_pose(k), target, seconds, self.control_dt):
            self.step({10 + k: s})


def scrub_floor(run: PanRunner, st: Station, press: float, passes: int, speed: float = 0.15) -> None:
    """Take the sponge from the ledge, scrub four strips across the pan's floor (``passes`` there and back each,
    pressing ``press``), and put it back on the ledge."""
    floor = run.pose(0).pos                                     # the pan's inner floor centre (held still)
    home = run.tool_pose(0)
    run.tool_grip[:, 0] = 1.0
    up = Pose(home.pos + [0.0, 0.0, 0.10], home.quat)
    run.go_tool(0, up, 1.0)
    clear = np.array([floor[0], floor[1], st.sink.rim_z + 0.12])   # over the pan, above the rims (the bowls' walls)
    above = floor + [0.0, 0.0, 0.10]                              # in the water, just over the pan's floor
    run.go_tool(0, Pose(clear, home.quat), 2.0)
    run.go_tool(0, Pose(above, home.quat), 1.5)
    z = floor[2] - press / TOOL_KP                               # the hand's spring presses ``press``
    t_pass = 2.0 * STRIP / speed * 1.5                          # smooth-step: mean speed 2/3 of the peak
    for a in (0.0, 0.25 * math.pi, 0.5 * math.pi, 0.75 * math.pi):
        q = quat_mul(quat_z(a), home.quat)
        d = np.array([-math.sin(a), math.cos(a), 0.0]) * STRIP  # along the pad's short axis
        run.go_tool(0, Pose(above, q), 1.0)
        run.go_tool(0, Pose(np.r_[floor[:2] - d[:2], z], q), 1.0)
        for _ in range(passes):
            run.go_tool(0, Pose(np.r_[floor[:2] + d[:2], z], q), t_pass)
            run.go_tool(0, Pose(np.r_[floor[:2] - d[:2], z], q), t_pass)
        run.go_tool(0, Pose(above, q), 0.8)
    run.go_tool(0, Pose(above, home.quat), 1.0)
    run.go_tool(0, Pose(clear, home.quat), 1.5)
    run.go_tool(0, up, 2.0)
    run.go_tool(0, Pose(home.pos + [0.0, 0.0, 0.002], home.quat), 1.0)
    run.tool_grip[:, 0] = 0.0
    run.hold(0.5)


def spray_band(run: PanRunner, st: Station, laps: int = 1, seconds_per_lap: float = 4.0, dump: bool = True) -> None:
    """With the pan out of the water and level, circle the valve's tip ``SPRAY_UP`` over the floor on each of
    ``SPRAY_LAPS`` (radius, outward tilt), valve open, ``laps`` times each, emptying the pan after each lap
    (:func:`dump_pan`); then park it. Left level, the spray's water filled the pan to 700 mL, the hand sagged to 16.7°
    and the low side's water shielded the floor's outer ring: 4 cells of egg (8.4 mg) stayed at 150-165°. Emptied
    after each lap, it held at most 294 mL at 7.6° and none stayed (WP8's spray probe, [internal note]
    ``notes/wp8-results/surface/PAN_HANDOFF.md``).

    The valve reaches each lap's first pose (height and tilt) over ``SETTLE_S`` with the valve shut, opens and shuts
    over ``RAMP_S`` there (:func:`squeeze`), and leaves it ``CLEAR_UP`` over the floor before the pan rolls: held at
    the lap's height, the rolled pan's wall swept through it (WP15's review of 170948b34: the valve shoved 25 mm, the
    roll stalled at 87°; a one-period drop to the lap took the tip onto the floor). Spraying, the jets' thrust (about
    2 N on the 300 N/m hand) sits the valve about 6 mm up its axis from the target, 8.5 mm at most as it opens;
    rolling, the valve stays within 1.5 mm of its target, about 95 mm over the rim
    (``tests/tasks/test_rinse_pan_spray.py``)."""
    floor = run.pose(0).pos
    home = run.tool_pose(1)
    run.grip_tool(VALVE, True)
    run.go_tool(1, Pose(floor + [SPRAY_LAPS[0][0], 0.0, CLEAR_UP], home.quat), 2.0)
    n = max(int(seconds_per_lap / run.control_dt), 8)
    for r0, tilt in SPRAY_LAPS:
        for _ in range(laps):
            run.go_tool(1, lap_pose(floor, home.quat, r0, tilt, 0.0), SETTLE_S)
            squeeze(run, 0.0, 1.0)
            for i in range(n + 1):
                phi = 2.0 * math.pi * i / n
                tangent = np.array([-math.sin(phi), math.cos(phi), 0.0])
                run.step({11: Step(lap_pose(floor, home.quat, r0, tilt, phi),
                                   r0 * 2.0 * math.pi / seconds_per_lap * tangent)})
            squeeze(run, 1.0, 0.0)
            run.go_tool(1, Pose(floor + [r0, 0.0, CLEAR_UP], home.quat), SETTLE_S)
            if dump:
                dump_pan(run)
    run.go_tool(1, home, 2.0)
    run.grip_tool(VALVE, False)
    run.hold(0.5)


def squeeze(run: PanRunner, a: float, b: float) -> None:
    """Take the valve's lever from ``a`` to ``b`` over ``RAMP_S``, the tool held where it is. Opened in one period, the
    jets' thrust (about 2 N against the 300 N/m hand, ζ 0.12) rang the valve 26 mm off its target."""
    flow = run.senses[0].flow
    n = max(int(round(RAMP_S / run.control_dt)), 1)
    for i in range(1, n + 1):
        flow.set_input("valve", a + (b - a) * i / n)
        run.step({})


def lap_pose(floor: np.ndarray, park: np.ndarray, r0: float, tilt: float, phi: float) -> Pose:
    """The valve's pose at ``phi`` round a lap: its tip ``SPRAY_UP`` over the floor at radius ``r0``. The valve sprays
    along its z (down at its park ``park``); turning it by ``-tilt`` about the tangent tips the jets out, toward the
    wall."""
    tangent = np.array([-math.sin(phi), math.cos(phi), 0.0])
    pos = floor + r0 * np.array([math.cos(phi), math.sin(phi), 0.0]) + [0.0, 0.0, SPRAY_UP]
    return Pose(pos, quat_mul(_axis_angle(tangent, -tilt), park))


def dump_pan(run: PanRunner) -> None:
    """Empty the pan where it is: roll it ``POUR_TURN`` about the handle's axis (world y) over 2 s, hold 1 s, level it
    over 1.5 s (WP8's probe V1). The rolled wall sweeps up to about 135 mm over the floor's centre: the valve must be
    clear of it first (``spray_band`` lifts it to ``CLEAR_UP``)."""
    p = run.pose(0)
    run.go(0, Pose(p.pos, quat_mul(_axis_angle(np.array([0.0, 1.0, 0.0]), POUR_TURN), p.quat)), 2.0)
    run.hold(1.0)
    run.go(0, p, 1.5)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    return np.r_[math.sin(0.5 * angle) * a, math.cos(0.5 * angle)]


def lift_and_pour(run: PanRunner, st: Station, bowl: str, pace: float = 1.0) -> None:
    """Lift the pan straight out of ``bowl``, level it ``POUR_SIDE`` to the bowl's -x side, and pour toward +x by
    rolling it about its handle's axis (world y; the stream lands in the bowl), then level it.

    Not a pour over the front rim: that turn swept the pan's floor plane over the sponge resting on the back ledge, and
    surface_flow's revolution pad (tool_geometry.rev_window, plane rule per cell) then read the sponge as pressed 0.3 m
    into the floor and threw it onto the kitchen floor (13.6 m/s). The roll keeps every facing cell's footprint off
    the sponge (a replica of the pad rule over these motions: no covered cell behind it)."""
    p = run.pose(0)
    run.go(0, Pose(p.pos + [0.0, 0.0, st.sink.rim_z + 0.12 - p.pos[2]], p.quat), 2.0 * pace)
    c = st.bowl_centre(("wash", "rinse", "sanitize").index(bowl))
    level = np.asarray(st.pan_quat(), np.float64)
    over = np.array([c[0] - POUR_SIDE, c[1], st.sink.rim_z + 0.12])
    run.go(0, Pose(over, level), 1.5 * pace)
    run.go(0, Pose(over, quat_mul(_axis_angle(np.array([0.0, 1.0, 0.0]), POUR_TURN), level)), 3.0 * pace)  # +x rim down
    run.hold(1.5 * pace)
    run.go(0, Pose(over, level), 1.0 * pace)


def to_shelf(run: PanRunner, st: Station, pace: float = 1.0) -> None:
    """Carry the pan high over the clean board's shelf, flip it mouth down about its handle's axis (world y here,
    so the handle does not sweep into the shelf), set it on the shelf and let go."""
    p = run.pose(0)
    x0 = st.sink.x0 + 3 * 0.4572
    target = np.array([x0 + 0.3048, Y_BACK - 0.15])
    high = st.shelf_z + st.pan.depth + 0.25
    run.go(0, Pose(np.array([p.pos[0], p.pos[1], high]), p.quat), 1.5 * pace)
    run.go(0, Pose(np.array([target[0], target[1], high]), p.quat), 3.0 * pace)
    down = quat_mul(_axis_angle(np.array([0.0, 1.0, 0.0]), math.pi), p.quat)
    run.go(0, Pose(np.array([target[0], target[1], high]), down), 2.0 * pace)
    run.go(0, Pose(np.array([target[0], target[1], st.shelf_z + st.pan.depth + 0.002]), down), 2.0 * pace)
    run.release(0, 1.0)


__all__ = ["PanRunner", "dump_pan", "lap_pose", "lift_and_pour", "scrub_floor", "spray_band", "squeeze",
           "to_shelf"]
