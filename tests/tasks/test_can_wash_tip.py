# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can stage 2 (alloy/tasks/can_wash_tip.py): the tip about the foot's edge toward the drain."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.tasks.can_wash_run import Rinse
from alloy.tasks.can_wash_run import readout
from alloy.tasks.can_wash_tip import (COARSE, RIM_CLEAR, Driver, Empty, plugins, rim_points, rotate, run_empty, script,
                                      tip_pose)
from alloy.tasks.rinse_hand import OverflowWatch
from alloy.tasks.rinse_score import ledger_closed
from alloy.tasks.rinse_flush_policy import CONTROL_DT


def _setup() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    task = COARSE
    origin = np.array([*task.at, 0.35])
    d = np.array([task.bay.drain[0] - task.at[0], task.bay.drain[1] - task.at[1], 0.0])
    d /= np.linalg.norm(d)
    edge = origin + task.can.r_bot * d - [0.0, 0.0, task.can.bottom]
    return origin, d, edge, origin[2] - task.can.bottom


@pytest.mark.parametrize("deg", [0.0, 30.0, 80.0, 105.0])
def test_the_tip_leans_the_axis_toward_the_drain(deg: float):
    origin, d, _, _ = _setup()
    _, q = tip_pose(COARSE, origin, math.radians(deg))
    axis = rotate(q, np.array([0.0, 0.0, 1.0]))
    assert np.linalg.norm(q) == pytest.approx(1.0)
    assert axis @ d == pytest.approx(math.sin(math.radians(deg)), abs=1e-12)
    assert axis[2] == pytest.approx(math.cos(math.radians(deg)), abs=1e-12)


def test_the_tip_turns_about_the_foot_edge_until_the_rim_would_land_then_lifts():
    """The foot's edge on the drain's side stays put while the rim is over the floor (to about 86°: the wall flares
    3.2°); after that the pose rises so the rim's lowest point stays ``RIM_CLEAR`` over the floor, smoothly, to 105°
    (DC5's review: the can had stopped at 87° on its rim)."""
    origin, _, edge, floor = _setup()
    rim = rim_points(COARSE.can)
    prev, lifted_from = None, None
    for deg in np.linspace(0.0, 105.0, 211):
        pos, q = tip_pose(COARSE, origin, math.radians(deg))
        low = (pos + rotate(q, rim))[:, 2].min()
        foot = pos + rotate(q, edge - origin)
        assert low >= floor + RIM_CLEAR - 1e-12 or deg < 87.0
        if np.allclose(foot, edge, atol=1e-12):
            assert lifted_from is None, deg                         # once lifted, it stays up
        else:
            lifted_from = deg if lifted_from is None else lifted_from
            assert low == pytest.approx(floor + RIM_CLEAR, abs=1e-12) and foot[2] > edge[2]
        if prev is not None:
            assert np.linalg.norm(pos - prev) < 0.01                 # 0.5° steps: no jump
        prev = pos
    assert 84.0 < lifted_from < 87.0
    assert floor + RIM_CLEAR > low - 1e-12 and foot[2] - edge[2] > 0.1   # at 105°: up on the hand


def test_the_script_tips_past_the_rim_and_sprays_the_outside_last():
    """GA's 105-107° (spec §2.6); the dumps' rinse has no outer heights (the outside is sprayed once, after them)."""
    em = Empty()
    assert math.degrees(em.angle) == pytest.approx(105.0) and em.rinse.outer == () and em.outer
    assert COARSE.carry == "mass"


class _Drive:
    """Records the script's calls (no simulation)."""

    def __init__(self) -> None:
        self.calls: list = []

    def spray(self, rs: Rinse) -> None:
        self.calls.append(("spray", rs))

    def pool_ml(self) -> np.ndarray:
        return np.array([6400.0])

    def empty(self, em: Empty) -> float:
        self.calls.append(("empty", em))
        return 11.5


def test_the_script_dumps_after_each_inside_rinse_and_sprays_the_outside_last():
    em, drive, dumps = Empty(), _Drive(), []
    script(drive, em, lambda full, hold: dumps.append((full, hold)))
    assert [c[0] for c in drive.calls] == ["spray", "empty", "spray", "empty", "spray"]
    assert drive.calls[0][1] == em.rinse and drive.calls[2][1] == em.rinse and em.rinse.outer == ()
    last = drive.calls[-1][1]
    assert last.outer == em.outer and last.floor == () and last.inner == () and em.outer
    assert dumps == [([6400.0], 11.5)] * 2


def _driver(dry_after: tuple[int, ...]) -> tuple[Driver, dict]:
    """A driver without a simulation: each world's pool is out after ``dry_after`` checks of the hold."""
    d, n = Driver.__new__(Driver), {"periods": 0}
    d.grip, d.check = np.zeros((len(dry_after), 1)), None
    d.turn = lambda a, b, seconds: None
    d.period = lambda valve=None: n.__setitem__("periods", n["periods"] + 1)
    d.pool_ml = lambda: np.array([0.0 if n["periods"] // 5 >= k else 50.0 for k in dry_after])
    return d, n


def test_the_flow_steps_before_the_hand():
    """G2 puts the pool's mass on the can in the flow's apply: the hand, listed after it, reads the wet mass."""
    flow, hand = object(), object()
    order = plugins(flow, hand)
    assert order.index(flow) < order.index(hand) and isinstance(order[-1], OverflowWatch)


def test_the_hold_waits_until_every_worlds_pool_is_out():
    d, _ = _driver((2, 6))
    assert d.empty(Empty(drain_s=0.0)) == pytest.approx(6 * 5 * CONTROL_DT)
    d, _ = _driver((2, 10_000))
    assert d.empty(Empty(drain_s=0.0, hold_cap=1.0)) == pytest.approx(1.0)
    assert d.grip.tolist() == [[0.0], [0.0]]                       # let go after


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_one_short_dump_empties_the_can_and_keeps_its_books():
    """CPU, COARSE: one floor ring pools about 480 mL in the can; the hand tips it 105° toward the drain with the pool
    on it as mass (G2), holds until under 10 mL, stands it back up and lets go; a 2 s settle. At every period the
    books close (DC5's review: a pour's species in flight, pending film species) and through the dump the hand grips;
    the can carries its pool's mass when the tip starts and holds the target's 105° in the drain. The pool leaves over
    the rim, the can stands upright and is let go, nothing overflows (WP8's watch attached)."""
    rows: list[tuple] = []

    def check(d: Driver) -> None:
        r, _, out = readout(d.task, d.hd, d.ref)
        closed = ledger_closed(float(r.water_rel[0]), float(r.species_rel[0]), float(r.species_kg[0]))
        pool_kg = 998.0 * out["supply_amount"][0, d.hd.probe.tank]
        carried = float(np.sum(d.hd.flow.supply.read()["carried_kg"][0]))
        rows.append((d.phase, closed, bool(r.held[0]), float(r.tilt_deg[0]), carried, float(pool_kg)))

    em = Empty(rinses=1, rinse=Rinse(floor=(0.09,), inner=(), outer=()), outer=(), drain_s=1.0, settle=2.0)
    res = run_empty(em=em, check=check)
    (dump,) = res["dumps"]
    w, end = res["worlds"][0], res["end"][0]
    fault = w["gates"]["no_simulation_fault"]
    assert dump["pool_ml_before"][0] > 200.0 and dump["pool_ml_after"][0] <= em.dry_ml, dump
    assert dump["hold_s"] < em.hold_cap
    dump = [x for x in rows if x[0] in ("tip", "hold", "drain", "back")]
    bad = [(i, x[0]) for i, x in enumerate(rows) if not x[1]] + [x[0] for x in dump if not x[2]]
    assert {x[0] for x in rows} >= {"spray", "tip", "hold", "drain", "back", "free"} and not bad, bad[:5]
    assert dump[0][4] == pytest.approx(dump[0][5], rel=0.01) and dump[0][4] > 0.3     # the pool on the can (G2)
    assert min(x[3] for x in rows if x[0] == "drain") == pytest.approx(105.0, abs=0.5)
    assert end["tilt_deg"][0] < 1.0 and w["gates"]["emptied_upright"]["ok"] and end["held"] == [False]
    assert w["gates"]["water_off_and_hands_free"]["ok"] is None                # AB-gym-3: let go, no pads to read
    assert "water_off_and_hands_free" in w["not_judged"] and w["judged"][1] == 8
    assert res["overflow"] == [False] and res["watch"] and fault["ok"], fault
    assert fault["value"]["water_rel"] < 1e-11
