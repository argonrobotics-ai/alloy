# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can, robot stage v1 (WP15's rig (b), the lathe; on its branch until after the launch).

Stage 2's bay, film and books (``can_wash_tip``), with the can on the turntable (``can_wash_table``), a UR10e + 2F-85
that turns it and an FR3 holding GA's hose valve fixed in its hand (``can_wash_arms``). Stage 2's order
(``can_wash_tip.script``): ``rinses`` times the inside bands and a dump, then the outside bands, then the dock.
- Spraying: the FR3 holds stage 1's nozzle pose for each band at its sector (``can_wash_strokes.SECTORS``) and the UR
  turns the can one revolution under it (``revolve_to``: 60° strokes, then one for what is left; each stroke planned
  about the can's measured axis, so the grip does not press it into the table's lip) at 1 rad/s, the rim
  passing the jets at stage 1's 0.25 m/s. The valve pulses while the can turns (``can_wash_pulse``: open in a share
  |ω| r / 0.25 m/s of the periods, from the can's measured turn), so every band, the floor's slower rings and the
  strokes' ramps included, takes stage 1's dose per metre; it is shut while the UR lets go and grips again.
- Dumping: stage 2's privileged pose hand on the can. The UR's rim pinch cannot dump it ([internal note]
  ``notes/job3/can_arm_reach.md``). The UR lets go and lifts clear, the FR3 backs off to its park, and the hand tips
  the can over the lip from where it stands (its yaw kept), holds, sets it back on the table and lets go.
- Docking: the UR lifts off the rim and the FR3 brings the valve to GA's holster (``parked``) and keeps it there. The
  valve is fixed in its hand (SOFT-3's pattern), so it is never released: ``water_off_and_hands_free`` fails in
  this stage (``Handles.tool_fixed``; audit AB-gym-3: GA's gate is "nothing held"), so a robot run does not succeed
  and scores 0.9 x progress. The run books the UR's last command (``ur_end``: jaws open, no squeeze) beside it.
- Books beyond stage 2's: each band's revolution (``revolution_books``), the water the turntable, the UR and the FR3
  caught per dump and in all (``caught_ml``: rays that hit them leave the film), the lip's peak force on the can in
  each dump, the can's turn after the last stroke (``idle_after_deg``).
Disclosed beyond stage 2's list: the turntable, its lip and its brake, which holds the plate whenever the UR is not
turning the can: a stand-in for G2's spurious yaw on the free plate until DC5's fix (``can_wash_table``); the arms'
links and fingers are not film tools (the film is one-way with them); every world runs world 0's arm commands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from alloy.tasks.can_wash_arms import WAND, ArmsBay, Rigs
from alloy.tasks.can_wash_pulse import Pulse, yaw_of
from alloy.tasks.can_wash_run import Rinse, can_z
from alloy.tasks.can_wash_strokes import SECTORS, StrokePlan, revolve_to, start_phi
from alloy.tasks.can_wash_tip import Driver, Empty, tip_pose
from alloy.tasks.can_wash_table import TABLE, footprint
from alloy.tasks.can_wash_turn import APPROACH, OPEN, RELEASE_N, Lathe, LatheRun, frame, put_ctrl, quat_of
from alloy.tasks.rinse_flush_floor import EDGE
from alloy.tasks.rinse_flush_policy import CONTROL_DT, NOZZLE

OMEGA = 1.0                  # rad/s, the can's turn: the rim at stage 1's 0.25 m/s
RETREAT = 0.25               # m, the UR's lift off the rim before a dump
FR3_RATE = 0.6               # rad/s, the FR3's mean joint rate between poses (SOFT-3's ``smooth``)
DUMP_PHASES = ("tip", "hold", "drain", "back")
LIFT_CLEAR = 0.003           # m, the can's foot over the table's lip while the hand tips it
LIP_SHARE = 0.3              # WP15: the lip's peak force under this share of the can's and the water's weight
WALK_MM = {"mujoco_c": 4.0, "mujoco_warp": 5.5}    # WP15 (11-09): the can's axis off the table's, every stroke. C:
                             # under the lip's 4 mm clearance (it never touches). Warp: the clearance plus the lip's
                             # contact give (4.80 mm in the full row: Warp's foot creeps on the plate, 8x C's)
RECENTRE = False             # strokes about the table's axis: about the can's measured axis they followed an
                             # off-centre can onto the lip (a 205 N lip and tip ratio 0.6 after a dump, MuJoCo C)
UR_SPEED = np.radians([120.0, 120.0, 180.0, 180.0, 180.0, 180.0])   # rad/s, the UR10e's joint speed limits (base
                             # and shoulder 120°/s, the others 180°/s)


@dataclass(frozen=True)
class Band:
    """One band's FR3 tool-point pose (x, y, z, qx, qy, qz, qw): stage 1's nozzle pose at the band's sector."""

    label: str
    kind: str                 # "inside" | "outside"
    tcp: np.ndarray
    r: float                  # m, the band's radius about the can's axis where the jets' aim meets it


def tcp_for(tip: np.ndarray, aim: np.ndarray) -> np.ndarray:
    """The FR3's tool point for the nozzle at ``tip`` spraying toward ``aim``: ``WAND`` behind it on the approach."""
    j = (aim - tip) / np.linalg.norm(aim - tip)
    x = np.cross(j, [0.0, 0.0, 1.0]) if abs(j[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    return np.array([*(tip - WAND * j), *quat_of(frame(j, x))])


def bands(task: Any, rs: Rinse) -> list[Band]:
    """Stage 1's bands as the FR3 sprays them: inside, the tip over the can at ``SECTORS["inside"]`` aiming across
    at the far wall; outside, aimed at the wall at ``SECTORS["outside"]`` from ``standoff`` out, ``lean`` up."""
    can, c = task.can, np.array([*task.at, can_z(task)])
    rim, out = float(can.radius(can.depth)), []
    u = np.array([math.cos(SECTORS["inside"]), math.sin(SECTORS["inside"]), 0.0])
    for r, z in [(r, 0.0) for r in rs.floor] + [(float(can.radius(z)), z) for z in rs.inner]:
        tip = c + 0.5 * rim * u + [0.0, 0.0, can.depth + rs.over_rim]
        out.append(Band(f"inside z {z:.2f} r {r:.2f}", "inside", tcp_for(tip, c - r * u + [0.0, 0.0, z]), r))
    v = np.array([math.cos(SECTORS["outside"]), math.sin(SECTORS["outside"]), 0.0])
    for z in rs.outer:
        aim = c + (float(can.radius(z)) + can.wall) * v + [0.0, 0.0, z]
        tip = aim + rs.standoff * (math.cos(rs.lean) * v + [0.0, 0.0, math.sin(rs.lean)])
        out.append(Band(f"outside z {z:.2f}", "outside", tcp_for(tip, aim), float(can.radius(z)) + can.wall))
    return out


def dock_tcp(task: Any) -> np.ndarray:
    """The FR3's tool point with the valve's body at the holster's docked centre, head down (``parked``)."""
    tip = np.asarray(task.bay.holster, np.float64) - [0.0, 0.0, NOZZLE]
    return tcp_for(tip, tip - [0.0, 0.0, 1.0])


def valve_body_label() -> str:
    """The hose valve's body label in the FR3's hand, as the bay's model names it (the flow's nozzles ride on it)."""
    from alloy._vendor import newton  # noqa: PLC0415
    from alloy.tasks.can_wash_arms import wand_xml  # noqa: PLC0415
    from alloy.tasks.rinse_flush_floor import VALVE  # noqa: PLC0415

    b = newton.ModelBuilder()
    b.add_mjcf(wand_xml(), ctrl_direct=True, parse_visuals=False)
    labels = [str(lb) for lb in b.body_label if str(lb).endswith("/" + VALVE)]
    if len(labels) != 1:
        raise ValueError(f"expected one body ending in /{VALVE} in the FR3's MJCF, found {labels}")
    return labels[0]


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


class ArmsDriver(Driver):
    """Stage 2's ``Driver`` with the two arms: every period writes the UR's command (``run``: the last one given) and
    the FR3's joint targets, then steps the plug-ins as stage 2 does. ``spray`` sprays a rinse's bands on the lathe;
    ``empty`` clears the arms and dumps with the pose hand; ``dock`` parks the valve."""

    def __init__(self, task: Any, hd: Any, hand: Any, run: LatheRun, rigs: Rigs, q_fr3: np.ndarray) -> None:
        super().__init__(task, hd, hand, None)
        self.run, self.rigs, self.lathe = run, rigs, Lathe(task)
        from alloy.tasks.paint_arm import Plan, Tracker  # noqa: PLC0415

        self.ur_plan, self.fr_plan = Plan(), Plan()
        self.ur_track, self.fr_track = Tracker(rigs.ur, self.ur_plan), Tracker(rigs.fr3, self.fr_plan)
        self.strokes = StrokePlan.make(self.lathe, self.ur_track, OMEGA)
        self.fr_bind = rigs.fr3.bind(hd.sim.model)
        self.fr_q = np.asarray(q_fr3, np.float64)
        self.fr_home = self.fr_q.copy()
        self.ur = (self.strokes.down[0], OPEN, RELEASE_N)
        self.yaw = np.array([0.0, 0.0, 0.0, 1.0])
        self.fr_steps: list[SimpleNamespace] = []
        self.ur_steps: list[np.ndarray] = []
        self.revolutions: list[dict] = []
        self.pulse: Pulse | None = None           # the band's valve pulses (``spray``)
        self.jets = False                         # the UR is turning the can: the valve may open
        self.ur_up = False                        # the UR retreated over the stroke's start (``clear``)
        self.dump_lip, self.caught0 = 0.0, {}     # the lip's peak force on the can in a dump [N], the catch before it
        run.stepper = self._ur_step

    def _ur_step(self, q: np.ndarray, width: float, force: float) -> None:
        self.ur = (np.asarray(q, np.float64), width, force)
        self.period()

    def period(self, valve: tuple[float, ...] | None = None) -> None:
        self.run.write(*self.ur)
        ctrl = self.hd.sim.control.mujoco.ctrl.numpy().copy()
        put_ctrl(ctrl, self.fr_bind.act[:, 0, :7], self.fr_q)              # every world runs world 0's commands
        self.hd.sim.control.mujoco.ctrl.assign(ctrl.astype(np.float32))
        if self.phase in DUMP_PHASES and self.run.lip is not None:
            self.dump_lip = max(self.dump_lip, self.run.lip.read())
        self.fr_steps.append(SimpleNamespace(q=self.fr_q.copy()))
        self.ur_steps.append(np.asarray(self.ur[0], np.float64).copy())
        if self.pulse is not None:
            yaw = yaw_of(self.hd.sim.state.body_q.numpy()[self.hd.can].astype(np.float64))
            self.hd.flow.set_input(EDGE, self.pulse.opening(yaw, self.jets).astype(np.float32))
        super().period(valve)

    def fr_to(self, q: np.ndarray) -> None:
        """The FR3 to joints ``q`` on SOFT-3's minimum-jerk profile (inside its acceleration limit), the UR held."""
        from alloy.tasks.paint_arm import smooth  # noqa: PLC0415

        for qk in smooth(self.fr_q, np.asarray(q, np.float64), CONTROL_DT, rate=FR3_RATE):
            self.fr_q = qk
            self.period()

    def fr_reach(self, tcp: np.ndarray) -> np.ndarray:
        return self.fr_track.track([tcp], self.fr_q[None, None])[0]

    def ur_to(self, q: np.ndarray) -> None:
        """The UR to joints ``q`` in the air, jaws open (SOFT-3's minimum-jerk ``smooth``)."""
        from alloy.tasks.paint_arm import smooth  # noqa: PLC0415

        for qk in smooth(np.asarray(self.ur[0], np.float64), np.asarray(q, np.float64), CONTROL_DT, rate=0.8):
            self.ur = (qk, OPEN, RELEASE_N)
            self.period()

    def spray(self, rs: Rinse) -> None:
        """Each of ``rs``'s bands: the FR3 to its pose (valve shut), then one revolution of the can under the valve's
        pulses (``can_wash_pulse``)."""
        self.phase = "spray"
        n = len(self.hd.can)
        jets = lambda on: lambda k: setattr(self, "jets", on)  # noqa: E731
        for band in bands(self.task, rs):
            self.fr_to(self.fr_reach(band.tcp))
            if self.ur_up:                                # back down from a dump's retreat, straight
                self.ur_line(RETREAT, APPROACH)
                self.ur_up = False
            self.ur_to(self.strokes.down[0])
            self.pulse = Pulse(band.r, rs.speed, n)
            rev = revolve_to(self.run, self.lathe, self.ur_track, OMEGA, before_turn=jets(True),
                             after_turn=jets(False), full=self.strokes, recentre=RECENTRE)
            share, duty, self.pulse = self.pulse.share(rev.turned), self.pulse.duty(), None
            self.hd.flow.set_input(EDGE, np.zeros(n, np.float32))
            self.revolutions.append(revolution_books(band.label, rev, share[0], duty[0]))
        self.fr_to(self.fr_home)

    def clear(self) -> None:
        """The UR straight up from the stroke's start to ``RETREAT`` over the pinch, the FR3 to its park."""
        if not self.ur_up:
            self.ur_line(APPROACH, RETREAT)
            self.ur_up = True
        self.fr_to(self.fr_home)

    def ur_line(self, a: float, b: float, seconds: float = 1.5) -> None:
        """The UR's tool point straight up or down over the stroke's start, from ``a`` to ``b`` over the pinch, jaws
        open, on a cosine ramp (a joint-space move back down from the retreat swung the fingers into the rim: the can
        was knocked 5 mm onto the lip)."""
        phi, n = start_phi(self.lathe), max(2, int(round(seconds / CONTROL_DT)))
        poses = [self.lathe.grip_pose(phi, a + (b - a) * 0.5 * (1.0 - math.cos(math.pi * k / n))) for k in range(n + 1)]
        for q in self.ur_track.track(poses, np.asarray(self.ur[0])[None, None]):
            self.ur = (np.asarray(q, np.float64), OPEN, RELEASE_N)
            self.period()

    def turn(self, a: float, b: float, seconds: float) -> None:
        for th in np.linspace(a, b, max(2, int(round(seconds / CONTROL_DT)))):
            p, q = tip_pose(self.task, self.origin, float(th), foot=footprint(self.task.can))
            self.pos[:, 0], self.quat[:, 0] = p, quat_mul(q, self.yaw)       # tipped about the table's axis
            self.period()

    def empty(self, em: Empty) -> float:
        """Stage 2's dump with the arms cleared: the hand raises the can over the table's lip, tips it about the
        table's axis (its yaw kept) and its colliders' footprint, stands it back up and lowers it centred onto the
        plate. Tipped from where it stood on the plate, about the foot ring, its floor boxes' corners swung under the
        pivot, its wall rode the lip and it came back 5-7 mm off, its foot on the lip. Raised 3 mm over the lip and
        tipped about the footprint, its colliders stay 3.4 mm or more over the lip's top and 8 mm over the plate."""
        self.clear()
        self.dump_lip, self.caught0 = 0.0, caught_ml(self.hd)
        X = self.hd.sim.state.body_q.numpy()[self.hd.can].astype(float)
        self.origin = np.array([*self.task.at, X[0, 2]])
        self.yaw = X[0, 3:].copy()
        self.lift = self.task.table.lip + LIFT_CLEAR          # over the lip: tipped about the foot, the wall rode it
        self.pos[:, 0], self.quat[:, 0] = X[:, :3], X[:, 3:]
        return super().empty(em)

    def dock(self) -> None:
        self.phase = "dock"
        self.clear()
        self.fr_to(self.fr_reach(dock_tcp(self.task)))

    def books(self) -> dict:
        """The plans' books: the FR3's joint path against its speed, acceleration and jerk limits, the IK's misses,
        the UR's commanded joint speeds against its limits (the largest share), each band's revolution."""
        self.fr_plan.steps = self.fr_steps
        fr = self.fr_plan.finish(CONTROL_DT).report()
        q = np.asarray(self.ur_steps, np.float64).reshape(-1, UR_SPEED.size)
        v = np.abs(np.diff(q, axis=0)) / CONTROL_DT if len(q) > 1 else np.zeros((1, UR_SPEED.size))
        return {"fr3": fr, "ur_misses": self.ur_plan.misses, "ur_speed_share": round(float((v / UR_SPEED).max()), 3),
                "revolutions": self.revolutions}


def run_robot(task: Any = None, em: Empty | None = None, worlds: int = 1, seed: int = 0, device: str = "cpu",
              check: Any = None) -> dict:
    """The robot stage in ``worlds`` worlds (module doc): stage 2's script with the arms, then the dock; each world's
    verdict, the dumps, and the plans' books."""
    import dataclasses  # noqa: PLC0415

    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415
    from alloy.tasks.can_wash_run import Handles, make_probe, readout  # noqa: PLC0415
    from alloy.tasks.can_wash_scene import CAN, POUR  # noqa: PLC0415
    from alloy.tasks.can_wash_table import Turntable  # noqa: PLC0415
    from alloy.tasks.can_wash_tip import COARSE, HAND, plugins, script  # noqa: PLC0415
    from alloy.tasks.paint_arm import Plan, Tracker  # noqa: PLC0415
    from alloy.tasks.rinse_flush_checks import Columns, shelf_slots  # noqa: PLC0415
    from alloy.tasks.rinse_flush_floor import VALVE  # noqa: PLC0415
    from alloy.tasks.rinse_flush_policy import bodies  # noqa: PLC0415
    from alloy.tasks.rinse_hand import PoseHand, mjw_solver, overflow_watch  # noqa: PLC0415

    task = task or dataclasses.replace(COARSE, table=Turntable(), valve_body=valve_body_label())
    em, flow, made, rigs = em or Empty(), task.make_flow(), {}, Rigs.make()
    hand = PoseHand(bodies=(CAN,), **HAND)
    q_ur = StrokePlan.make(Lathe(task), Tracker(rigs.ur, Plan()), OMEGA).down[0]
    q_fr3 = np.asarray(rigs.fr3.specs[0].home, np.float64)
    brake = task.table.brake()

    def init(sim: Any) -> None:
        flow.set_input(EDGE, np.zeros(worlds, np.float32))
        made.update(task.start(flow, worlds, seed))

    sim = build_simulation(ArmsBay(task, rigs, q_ur, q_fr3).scene(init, worlds), [*plugins(flow, hand), brake], device)
    net = flow.supply.net
    cols = Columns(outlet=net.edge_index(EDGE), shelf=shelf_slots(sim.model, flow.supply.occluder_shapes))
    hd = Handles(flow, sim, make_probe(task, flow), bodies(sim.model, CAN), bodies(sim.model, VALVE), cols,
                 net.edge_index(POUR), np.zeros(worlds, bool), mjw_solver(sim), watch=overflow_watch(sim),
                 tool_fixed=True)                                # the valve in the FR3's hand: never released
    drive = ArmsDriver(task, hd, hand, LatheRun(task, rigs.ur, sim, brake), rigs, q_fr3)
    drive.run.hold_table(True)                                   # held until the UR's first turn
    drive.run.aboard = lambda: float(np.sum(flow.supply.read()["carried_kg"][0]))
    r0, ref, _ = readout(task, hd)
    drive.ref, drive.check, dumps = ref, check, []

    def dumped(full: list, held: float) -> None:
        now = caught_ml(hd)
        dumps.append({"pool_ml_before": full, "pool_ml_after": drive.pool_ml().tolist(), "hold_s": round(held, 2),
                      "lost_ml": (flow.supply.read()["lost"] * 1e6).tolist(),
                      "caught_ml": {k: round(now[k] - drive.caught0.get(k, 0.0), 4) for k in now},
                      "lip_force_n": round(drive.dump_lip, 2)})

    script(drive, em, dumped)
    drive.dock()
    return finish(task, hd, drive, r0, ref, em, dumps, made, seed)


def finish(task: Any, hd: Any, drive: ArmsDriver, r0: Any, ref: Any, em: Empty, dumps: list, made: dict,
           seed: int) -> dict:
    """The settle and the verdict, as stage 2's ``run_empty`` reads them, plus the plans' books."""
    from alloy.tasks.can_wash_run import as_row, readout, settle  # noqa: PLC0415
    from alloy.tasks.can_wash_score import row, verdict  # noqa: PLC0415

    live, _, _ = readout(task, hd, ref)
    ends, at = settle(task, hd, ref, em.settle)
    out, rows = hd.flow.read(), []
    for w in range(len(ends)):
        v = verdict(r0, ends[w], w, live=live)
        rows.append(row(v, at[w]))
    return {"worlds": rows, "dumps": dumps, "start": as_row(r0), "live": as_row(live), "end": [as_row(e) for e in ends],
            "lost_ml": (hd.flow.supply.read()["lost"] * 1e6).tolist(), "overflow": hd.overflow.tolist(),
            "drained_L": (out["drained_total_volume"] * 1e3).tolist(), "plans": drive.books(), "seed": seed,
            "walk": walk_law(drive.revolutions, float(task.can.mass), route_of(hd.sim)),
            "strength": np.asarray(made["strength"]).tolist(), "caught_ml": caught_ml(hd),
            "idle_after_deg": round(math.degrees(drive.run.read()[1] - (drive.run.last_yaw or 0.0)), 2),
            "ur_end": {"width_m": float(drive.ur[1]), "force_n": float(drive.ur[2])}}


def route_of(sim: Any) -> str:
    """``"mujoco_c"`` on Alloy's MuJoCo C CPU route, else ``"mujoco_warp"`` (its ``mjw_data`` exists on both)."""
    from alloy.route.mjc_core import MjcCoreSolver  # noqa: PLC0415
    from alloy.tasks.rinse_hand import mjw_solver  # noqa: PLC0415

    return "mujoco_c" if isinstance(mjw_solver(sim), MjcCoreSolver) else "mujoco_warp"


def walk_law(revs: list[dict], mass: float, route: str) -> dict:
    """WP15's lip law over a run's revolutions (``revolution_books``): every band's peak lip force under ``LIP_SHARE``
    of the can's (``mass`` [kg]) and the water's weight aboard, and the can's axis within ``WALK_MM[route]`` of the
    table's at every stroke. A lip force not read (NaN) fails."""
    share = max(r["lip_force_n"] / ((mass + r["aboard_kg"]) * 9.81) for r in revs)
    walk = max(max(r["offset_mm_strokes"]) for r in revs)
    bound = WALK_MM[route]
    return {"route": route, "lip_share": round(share, 3), "walk_mm": round(walk, 2), "walk_bound_mm": bound,
            "ok": bool(share <= LIP_SHARE and walk <= bound)}


def revolution_books(label: str, rev: Any, share: float, duty: float) -> dict:
    """One band's revolution (``can_wash_strokes.Revolution``) as the run books it: the worst of each stroke's books
    (``TurnLog``), each stroke's offset off the table's axis and lip force (WP15: the creep, per stroke), the idle
    turn in each gap before a turn, the brake's grip while turning, the valve's duty and its dose per metre over stage
    1's."""
    lip = [s.lip_force for s in rev.strokes if np.isfinite(s.lip_force)]
    touch = [s.lip_touch for s in rev.strokes if np.isfinite(s.lip_touch)]
    return {"band": label, "strokes": len(rev.strokes), "turned_deg": round(math.degrees(rev.turned), 2),
            "pad_slip_mm": round(rev.worst("pad_slip") * 1e3, 2), "table_slip_mm": round(rev.worst("slip") * 1e3, 2),
            "drift_mm": round(rev.worst("drift") * 1e3, 2), "offset_mm": round(rev.worst("offset") * 1e3, 2),
            "offset_mm_strokes": [round(s.offset * 1e3, 2) for s in rev.strokes],
            "lip_n_strokes": [round(s.lip_force, 2) for s in rev.strokes],
            "rim_force_n": round(rev.worst("force"), 2),
            "tip_ratio": round(rev.worst("tip_ratio"), 3), "aboard_kg": round(rev.worst("aboard"), 3),
            "lip_force_n": round(max(lip), 2) if lip else float("nan"),
            "lip_touch": round(max(touch), 3) if touch else float("nan"),
            "lip_touch_full": sum(t >= 0.999 for t in touch),
            "idle_deg": round(max(abs(math.degrees(s.idle)) for s in rev.strokes), 2),
            "idle_deg_gaps": [round(math.degrees(s.idle), 2) for s in rev.strokes],
            "brake_grip": max(s.braked for s in rev.strokes), "duty": round(float(duty), 3),
            "dose_share": round(float(share), 3)}


def caught_ml(hd: Any) -> dict[str, float]:
    """World 0's water caught so far by shapes that are not charts [mL] (the supply's ``caught_by``: rays that hit
    them), by owner: the turntable (plate and lip), the UR, the FR3 with its valve, and the rest."""
    occ, caught = hd.flow.supply.occluder_shapes[0], hd.flow.supply.read()["caught_by"][0]
    labels = [str(lb) for lb in hd.sim.model.shape_label]
    out = dict.fromkeys(("table", "ur", "fr3", "other"), 0.0)
    for slot, shape in enumerate(occ):
        if shape < 0:
            break
        lb = labels[int(shape)]
        key = "table" if TABLE in lb else "ur" if "ur10e" in lb else "fr3" if "fr3" in lb else "other"
        out[key] += float(caught[slot]) * 1e6
    return out


__all__ = ["LIP_SHARE", "OMEGA", "WALK_MM", "ArmsDriver", "Band", "bands", "caught_ml", "dock_tcp", "finish",
           "quat_mul", "revolution_books", "route_of", "run_robot", "tcp_for", "valve_body_label", "walk_law"]
