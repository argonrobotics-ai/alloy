# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can, stage 2: rinse the inside, tip the can out toward the floor drain, again, then the outside
(GA's assisted demo A4, reduced; spec rinse_spec_more.md §2.6: tipping 105-107° toward the floor drain, the pool
pouring over the rim's lowest point).

A privileged hand (WP8's ``PoseHand``, no robot: GA's demo is two hands) grips the can and turns it about its foot's
edge on the drain's side, lifting it once the flared rim would meet the floor (``tip_pose``). The pool's water rides
on the can as mass, centre and inertia (DC5's G2, ``Pool(carry="mass")``), so the hand carries it; the flow is listed
before the hand, so the hand reads this exchange's mass. The sequence (``Empty``):
1. ``rinses`` times: the inside spray (``can_wash_run.plan`` without the outer heights), then the hose shut; grip the
   can, tip it to ``angle`` over ``tip_s``, hold until its pool is under ``dry_ml`` (at most ``hold_cap`` s) and
   ``drain_s`` more for the walls' film, stand it back up over ``back_s`` and let go.
2. The outside spray, last: tipping runs the inside's film over the rim onto the outer wall. Leaning 30° down from
   0.30 m, the cone's top rays meet the outer wall below the rim, so none enter the can.
3. Shut and dock, the score's live read, GA's settle (``can_wash_run.settle``), the gates.

``COARSE`` is the full bay at 20 mm cells with a 30 mm can chart: the tipped can pours about 0.9 m out from its axis,
past ``CROP``'s tile.

Disclosed:
- one privileged hand (GA's demo uses two, and a robot would need its own reach): ``PoseHand``'s PD (``HAND``) drives
  the gripped can along the turn's poses;
- the pool leaves over the rim's lowest point through ``can_lip`` (stage 1's spill), and rides the can as mass while
  it is there (DC5's G2): the water translates with the can and does not turn with it, as GA's;
- the drain after each dump (``Empty.drain_s``, 12 s) is the script's choice; the can keeps 18.1 mL of free water at
  the settled read (limit 100; docs/notes/can-wash-results.md);
- free water counts the cavity pool, which an emptied can restarts from its floor film (DC5's empty-cavity rule),
  plus the inner film above what clings. GA keeps an emptied can's puddle as film and leaves its bottom 0.5 mm
  uncounted, so Alloy's count is stricter, by up to about 70 mL (DC5's review: in a run whose tip stopped at 87°,
  88.7 of 88.9 mL was the restarted pool);
- the settle reaches GA's 30 s cap: the bay floor still drains about 7 g/s into the floor drain (GA's quiet rule:
  1 g/s); the can is still.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.tasks.can_wash_run import (Handles, Rinse, as_row, make_probe, plan, readout, region_soap, settle,
                                      step_watched)
from alloy.tasks.can_wash_scene import CAN, POUR, Can, CanWash
from alloy.tasks.can_wash_score import row, verdict
from alloy.tasks.rinse_flush_checks import Columns, shelf_slots
from alloy.tasks.rinse_flush_floor import EDGE, VALVE, Bay
from alloy.tasks.rinse_flush_policy import CONTROL_DT, bodies, parked, place, q_starts
from alloy.tasks.rinse_hand import OverflowWatch, PoseHand, mjw_solver, overflow_watch

HAND = {"kp": 4000.0, "kd": 300.0, "kr": 400.0, "dr": 40.0, "f_max": 300.0, "t_max": 200.0}   # on the 4.5 kg can
RIM_CLEAR = 0.010                     # m: the rim's lowest point over the floor once the tip would rest it there
HOLD_CHECK = 5                        # control periods between the hold's pool checks
LIFT_S = 0.5                          # s, the can raised (or lowered) by ``Driver.lift`` around a tip
COARSE = CanWash(bay=Bay(cell=0.02, debris=(), can=None), can=Can(ds=0.03, n_theta=32, ring=16), carry="mass")
# COARSE against GA: can cells 30 mm along the wall and 32 sectors (52 mm at the rim) against GA's 6 mm, bay floor
# cells 20 mm against GA's 10 mm; the 1 cm gauges read single coarse cells


@dataclass(frozen=True)
class Empty:
    """The stage-2 script (module doc): angles [rad], times [s], the pool left when the hold stops [mL]."""

    rinses: int = 2
    angle: float = math.radians(105.0)
    tip_s: float = 3.0
    hold_cap: float = 15.0                # a 6.5 L pool takes about 8 s over the rim
    dry_ml: float = 10.0
    drain_s: float = 12.0                 # held on after that, for the walls' film to run out
    back_s: float = 3.0
    rinse: Rinse = field(default_factory=lambda: Rinse(outer=()))
    outer: tuple[float, ...] = Rinse().outer
    settle: float = 30.0


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``v`` [..., 3] turned by the unit quaternion ``q`` (xyzw)."""
    u, w = np.asarray(q[:3], float), float(q[3])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def rim_points(can: Can, n: int = 72) -> np.ndarray:
    """``[n, 3]`` the rim's outer edge in the can's body frame (origin: the inner floor's centre)."""
    a = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    return np.stack([can.r_top * np.cos(a), can.r_top * np.sin(a), np.full(n, can.height - can.bottom)], 1)


def tip_pose(task: CanWash, origin: np.ndarray, theta: float,
             foot: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The can's body pose tipped ``theta`` toward the drain about its foot's edge: (origin [3], quat xyzw [4]). Its
    wall flares 3.2° out, so near 87° the rim would meet the floor first; from there the pose rises to keep the rim's
    lowest point ``RIM_CLEAR`` over the start's floor, and the hand holds the can up (DC5's review: about the foot's
    edge alone the 105° target sat 217 mm under the floor and the can stopped at 87° on its rim). ``foot``: the
    pivot's radius from the can's axis (default the foot's edge, ``r_bot``; the robot stage tips about the collider
    footprint, so the floor boxes' outer corners do not swing under the pivot)."""
    d = np.array([task.bay.drain[0] - task.at[0], task.bay.drain[1] - task.at[1], 0.0])
    d /= np.linalg.norm(d)
    pivot = origin + (task.can.r_bot if foot is None else foot) * d - [0.0, 0.0, task.can.bottom]
    k = np.cross([0.0, 0.0, 1.0], d)
    v = origin - pivot
    turned = v * math.cos(theta) + np.cross(k, v) * math.sin(theta) + k * np.dot(k, v) * (1.0 - math.cos(theta))
    pos, q = pivot + turned, np.array([*(k * math.sin(0.5 * theta)), math.cos(0.5 * theta)])
    low = (pos + rotate(q, rim_points(task.can)))[:, 2].min()
    pos[2] += max(0.0, origin[2] - task.can.bottom + RIM_CLEAR - low)
    return pos, q


class Driver:
    """Steps the run: the hand's targets on the can and the valve's pose, every world in lockstep. Every world turns
    about world 0's start pose: the can starts at one pose in every world (no start jitter, disclosed)."""

    def __init__(self, task: CanWash, hd: Handles, hand: Any, starts: np.ndarray) -> None:
        self.task, self.hd, self.hand, self.starts = task, hd, hand, starts
        q = hd.sim.state.body_q.numpy()[hd.can].astype(float)
        self.origin, self.quat0 = q[0, :3].copy(), q[0, 3:].copy()
        W = len(hd.can)
        self.pos, self.quat = np.tile(q[:, None, :3], 1), np.tile(q[:, None, 3:], 1)
        self.vel, self.grip = np.zeros((W, 1, 3)), np.zeros((W, 1))
        hd.grip = self.grip[:, 0]
        self.phase = "start"
        self.lift = 0.0                     # m, the can raised straight up before the tip and lowered after it (0 here)
        self.check: Callable[[Driver], None] | None = None     # called after every period (tests read the books)
        self.ref: dict | None = None                           # the start's books, for ``check``

    def period(self, valve: tuple[float, ...] | None = None) -> None:
        for name, v in (("target_pos", self.pos), ("target_quat", self.quat), ("target_vel", self.vel),
                        ("grip", self.grip)):
            self.hand.set_input(name, v.astype(np.float32))
        if valve is not None:
            place(self.hd.sim, self.starts, valve)
        step_watched(self.hd)
        if self.check is not None:
            self.check(self)

    def spray(self, rs: Rinse) -> None:
        self.phase = "spray"
        self.hd.flow.set_input(EDGE, np.ones(len(self.hd.can), np.float32))
        for p in plan(self.task, rs):
            self.period(p)
        self.hd.flow.set_input(EDGE, np.zeros(len(self.hd.can), np.float32))
        self.period(parked(self.task.bay))

    def turn(self, a: float, b: float, seconds: float) -> None:
        for th in np.linspace(a, b, max(2, int(round(seconds / CONTROL_DT)))):
            p, q = tip_pose(self.task, self.origin, float(th))
            self.pos[:, 0], self.quat[:, 0] = p, q
            self.period()

    def raise_by(self, base: np.ndarray, a: float, b: float) -> None:
        """The can held upright from ``a`` to ``b`` over ``base`` on a cosine ramp over ``LIFT_S``; the tip's origin
        is left at ``base`` raised ``b`` (no periods when ``a == b``)."""
        n = int(round(LIFT_S / CONTROL_DT)) if a != b else 0
        for k in range(1, n + 1):
            self.origin = base + [0.0, 0.0, a + (b - a) * 0.5 * (1.0 - math.cos(math.pi * k / n))]
            self.turn(0.0, 0.0, CONTROL_DT)
        self.origin = base + [0.0, 0.0, b]

    def pool_ml(self) -> np.ndarray:
        """``[W]`` the cavity pool [mL], from the supply's read alone (not the film's)."""
        return self.hd.flow.supply.read()["amount"][:, self.hd.probe.tank] * 1e6

    def empty(self, em: Empty) -> float:
        """Grip, tip, hold until dry (or the cap), stand back up, let go; the hold's length [s]. With ``lift`` the can
        is raised that far before the tip, tips about the raised pose, and is lowered straight down before release."""
        self.grip[:] = 1.0
        lift = getattr(self, "lift", 0.0)
        base = self.origin.copy() if lift else None
        if lift:
            self.raise_by(base, 0.0, lift)
        self.phase = "tip"
        self.turn(0.0, em.angle, em.tip_s)
        self.phase, checks = "hold", 0
        while checks < round(em.hold_cap / (HOLD_CHECK * CONTROL_DT)) and (self.pool_ml() > em.dry_ml).any():
            for _ in range(HOLD_CHECK):                                        # until every world's pool is out
                self.period()
            checks += 1
        held = checks * HOLD_CHECK * CONTROL_DT
        self.phase = "drain"
        for _ in range(int(round(em.drain_s / CONTROL_DT))):
            self.period()
        self.phase = "back"
        self.turn(em.angle, 0.0, em.back_s)
        if lift:
            self.raise_by(base, lift, 0.0)
        self.grip[:] = 0.0
        self.phase = "free"
        for _ in range(10):
            self.period()
        return held


def script(drive: Any, em: Empty, dumped: Callable[[list, float], None]) -> None:
    """The stage-2 sequence (module doc): ``em.rinses`` times the inside spray and a dump (``dumped(pool_ml before,
    hold s)`` after each), then the outside, last."""
    for _ in range(em.rinses):
        drive.spray(em.rinse)
        full = drive.pool_ml().tolist()
        dumped(full, drive.empty(em))
    drive.spray(dataclasses.replace(Rinse(), floor=(), inner=(), outer=em.outer))   # last: the dumps run the inside's
                                                                                  # film over the rim onto it


def plugins(flow: Any, hand: Any) -> list:
    """The plug-ins in their order: the flow before the hand, so the hand reads this exchange's wet mass (G2), and
    WP8's overflow watch."""
    return [flow, hand, OverflowWatch()]


def run_empty(task: CanWash | None = None, em: Empty | None = None, worlds: int = 1, seed: int = 0,
              device: str = "cpu", check: Callable[[Driver], None] | None = None) -> dict:
    """Stage 2 (module doc) in ``worlds`` worlds; each world's verdict, the pool after each dump, the holds.
    ``check(driver)`` runs after every period (``driver.phase``, ``driver.ref``: the start's books)."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run

    task, em = task or COARSE, em or Empty()
    flow, made = task.make_flow(), {}
    pose_hand = PoseHand(bodies=(CAN,), **HAND)

    def init(sim: Any) -> None:
        flow.set_input(EDGE, np.zeros(worlds, np.float32))
        made.update(task.start(flow, worlds, seed))

    sim = build_simulation(task.scene(init, worlds), plugins(flow, pose_hand), device)
    net = flow.supply.net
    cols = Columns(outlet=net.edge_index(EDGE), shelf=shelf_slots(sim.model, flow.supply.occluder_shapes))
    hd = Handles(flow, sim, make_probe(task, flow), bodies(sim.model, CAN), bodies(sim.model, VALVE), cols,
                 net.edge_index(POUR), np.zeros(worlds, bool), mjw_solver(sim), watch=overflow_watch(sim))
    drive = Driver(task, hd, pose_hand, q_starts(sim.model, hd.valve))
    r0, ref, out0 = readout(task, hd)
    soap0 = region_soap(task, out0, flow.tables.area_np, hd.probe)
    drive.ref, drive.check, dumps = ref, check, []

    def dumped(full: list, hold: float) -> None:
        dumps.append({"pool_ml_before": full, "pool_ml_after": drive.pool_ml().tolist(), "hold_s": round(hold, 2),
                      "soap_mg": (readout(task, hd, ref)[0].soap_kg * 1e6).tolist()})

    script(drive, em, dumped)
    live, _, _ = readout(task, hd, ref)
    ends, at = settle(task, hd, ref, em.settle)
    out = flow.read()
    rows = []
    for w in range(worlds):
        v = verdict(r0, ends[w], w, live=live)
        rows.append(row(v, at[w]))
    return {"worlds": rows, "dumps": dumps, "start": as_row(r0), "live": as_row(live), "end": [as_row(e) for e in ends],
            "soap_mg_regions": {"start": soap0, "end": region_soap(task, out, flow.tables.area_np, hd.probe)},
            "carried_kg": np.asarray(flow.supply.read().get("carried_kg", np.zeros((worlds, 1)))).tolist(),
            "drained_L": (out["drained_total_volume"] * 1e3).tolist(), "overflow": hd.overflow.tolist(),
            "watch": hd.watch is not None and hd.watch.bits is not None,
            "seed": seed, "strength": np.asarray(made["strength"]).tolist()}


__all__ = ["COARSE", "Driver", "Empty", "HAND", "RIM_CLEAR", "plugins", "rim_points", "rotate", "run_empty", "script",
           "tip_pose"]
