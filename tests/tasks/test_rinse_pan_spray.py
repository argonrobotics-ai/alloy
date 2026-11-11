# SPDX-License-Identifier: Apache-2.0
"""rinse_pan's spray on the CPU, cut down: the wash pour, the carry over the rinse bowl, one lap of the valve (open
1.0 in world 0, 0.5 in world 1) and the dump after it (DC5's should-fix A and WP15's must-fix on 170948b34)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

import alloy.tasks.rinse_pan_policy as POL
from alloy.tasks.rinse_books import Books
from alloy.tasks.rinse_pan import PanTask, outlets_running, setup
from alloy.tasks.rinse_pan_scene import Station
from alloy.tasks.rinse_plates_scene import Pose
from alloy.tasks.rinse_vessels import Pan

W = 2
OPENING = np.array([1.0, 0.5])        # the valve per world, so the worlds' water differs
# The offsets on 170948b34 in brackets are this test's 3D offset of the valve's origin; WP15's probes read 25 mm (the
# valve's shove) and 105 mm (the tip's plunge below its target) for the same two faults.
DUMP_SLACK = 0.003                    # m, the held valve off its target while the pan rolls (27.7 mm when the wall hit
                                      # it)
VALVE_SLACK = 0.010                   # m, off it otherwise: the jets' thrust (~2 N on 300 N/m) sits it ~6 mm off while
                                      # it sprays (141 mm when lap 1 began with a one-period drop)
ROLL_SLACK = math.radians(3.0)        # the dump's roll short of POUR_TURN (87° of 110° when the wall hit the valve)
LEFT = 1.0e-6                         # m³, the most a dumped pan keeps at the end of the level move (0 measured; 3-4 mL
                                      # drains back from the faces within 2.5 s; the stalled 87° roll left 3.5-8.1 mL)
LEVEL = math.radians(2.0)             # the pan's tilt after the dump
SHED = 0.05                           # the most of the dumped water shed off the chart's open edge (``spilled``, not
                                      # dripped: rinse_pan's disclosures). Measured: 0.07 / 11.4 mL (to 3.4 %) on the
                                      # Warp CPU route the suite runs, 0.04 / 3.3 mL on MuJoCo C (DC5's probe)
BALANCE = 1.0e-6                      # the dump's own water balance, relative to the water dumped
FILM = ("film_volume", "pending_volume", "spilled_volume", "evaporated_volume", "absorbed_volume")


def tilt(q: np.ndarray) -> float:
    """The angle between a body's z and world z (quaternion x, y, z, w)."""
    x, y = float(q[0]), float(q[1])
    return math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y))))


@dataclass
class Watch:
    """Per control period, per world: the held valve's offset from its target, the pan's tilt, the outlets while
    the pan rolls; the cavity and the rinse bowl around each dump."""

    run: POL.PanRunner
    flow: object
    st: Station
    dumping: bool = False
    offset: np.ndarray = field(default_factory=lambda: np.zeros(W))
    offset_dump: np.ndarray = field(default_factory=lambda: np.zeros(W))
    roll: np.ndarray = field(default_factory=lambda: np.zeros(W))
    running: np.ndarray = field(default_factory=lambda: np.zeros(W, bool))
    dumps: list[dict[str, np.ndarray]] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [n.name for n in self.st.nodes()]
        self.cav, self.rinse = names.index("pan_cavity"), names.index("rinse")
        self.outlets = outlets_running(self.flow, self.st)
        self.valve = self.run.tools.held.numpy()[:, POL.VALVE]
        self.pan = self.run.hand.held.numpy()[:, 0]
        self.step = self.run.step

    def tanks(self) -> tuple[np.ndarray, np.ndarray]:
        a = self.flow.supply.read()["amount"]
        return a[:, self.cav].copy(), a[:, self.rinse].copy()

    def watched_step(self, moves) -> None:
        self.step(moves)
        bq = self.run.sim.state.body_q.numpy()
        held = self.run.tool_grip[:, POL.VALVE] > 0.5
        off = np.where(held, np.linalg.norm(bq[self.valve, :3] - self.run.tool_pos[:, POL.VALVE], axis=1), 0.0)
        self.offset = np.maximum(self.offset, off)
        if self.dumping:
            self.offset_dump = np.maximum(self.offset_dump, off)
            self.roll = np.maximum(self.roll, [tilt(bq[b, 3:7]) for b in self.pan])
            self.running |= self.outlets()

    def film(self) -> dict[str, np.ndarray]:
        fr = self.flow.read()
        return {k: np.asarray(fr[k], np.float64).copy() for k in FILM}

    def dump(self, run: POL.PanRunner) -> None:
        (cav, rinse), film = self.tanks(), self.film()
        self.dumping = True
        _DUMP(run)
        self.dumping = False
        (cav1, rinse1), film1 = self.tanks(), self.film()
        bq = run.sim.state.body_q.numpy()
        self.dumps.append(dict(cav=cav, cav1=cav1, gained=rinse1 - rinse,
                               shed=film1["spilled_volume"] - film["spilled_volume"],
                               film=sum(film1[k] - film[k] for k in FILM),
                               level=np.array([tilt(bq[b, 3:7]) for b in self.pan])))


_DUMP = POL.dump_pan


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_one_lap_and_its_dump_empty_the_pan_with_the_valve_clear(monkeypatch):
    """After the wash pour, the pan over the rinse bowl takes one lap of the spray and one dump. The held valve stays
    within a few mm of its target while the pan rolls (the rolled wall does not shove it) and within 10 mm otherwise
    (its approach to the lap does not drop it onto the floor). The dump rolls the pan to POUR_TURN with no outlet
    running and levels it; the pan keeps under 1 mL, the water dumped is the rinse bowl's gain plus the film's change
    (the film shed off the chart's open edge under 5 % of it), nothing is lost, every ``rinse_books`` book closes per
    world and no substep overflowed."""
    task = PanTask(station=Station(weir=("wash", "rinse", "sanitize"), pan=Pan(ds=5.0e-3, n_theta=24)), passes=1)
    st = task.station
    sim, flow, run = setup(task, W, "cpu")
    books = Books.start(flow)
    watch = Watch(run, flow, st)
    monkeypatch.setattr(run, "step", watch.watched_step)
    monkeypatch.setattr(POL, "dump_pan", watch.dump)
    monkeypatch.setattr(POL, "SPRAY_LAPS", POL.SPRAY_LAPS[:1])
    set_input = flow.set_input
    monkeypatch.setattr(flow, "set_input", lambda name, v: set_input(name, OPENING * v if name == "valve" else v))
    run.grip[:, 0] = 1.0
    POL.lift_and_pour(run, st, "wash")
    run.go(0, Pose(st.bowl_centre(1) + [0.0, 0.0, 0.254 + 0.25], run.pose(0).quat), 2.0)
    POL.spray_band(run, st)
    assert len(watch.dumps) == 1, watch.dumps
    d = watch.dumps[0]
    print({k: np.round(v * 1e6, 2).tolist() for k, v in d.items() if k != "level"}, np.degrees(watch.roll),
          np.round(watch.offset * 1e3, 2), np.round(watch.offset_dump * 1e3, 2), np.degrees(d["level"]))
    assert (watch.offset < VALVE_SLACK).all() and (watch.offset_dump < DUMP_SLACK).all(), (watch.offset,
                                                                                           watch.offset_dump)
    assert (watch.roll > POL.POUR_TURN - ROLL_SLACK).all(), np.degrees(watch.roll)
    assert not watch.running.any() and (d["level"] < LEVEL).all(), (watch.running, np.degrees(d["level"]))
    out = d["cav"] - d["cav1"]
    assert (d["cav"] > 50 * LEFT).all() and (d["cav1"] < LEFT).all(), d
    assert (np.abs(out - d["gained"] - d["film"]) <= BALANCE * out).all(), (out - d["gained"] - d["film"]) / out
    assert (d["shed"] <= SHED * out).all(), d["shed"] / out
    assert (flow.supply.read()["lost"] == 0.0).all()
    closed, worst = books.closed(flow)
    assert closed.tolist() == [True] * W, worst
    assert not run.overflowed.any()
