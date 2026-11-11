# SPDX-License-Identifier: Apache-2.0
"""rinse-sponge scene and run (alloy/tasks/sponge_rinse_scene.py, sponge_rinse_run.py): the sponge against
Gym-Anything's numbers (spec rinse_spec_more.md §3: 79.1 mL of pores, 59.3 mL of wash at the start, a 120 N wring
leaving 41.4 mL), the press law, the books on synthetic reads, a short scripted rinse and the overflow watch."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from alloy.tasks import sponge_rinse_run as R
from alloy.tasks.can_wash_books import closed
from alloy.tasks.rinse_score import ledger_closed
from alloy.tasks.rinse_sink import BOWLS, drain_edge
from alloy.tasks.sponge_rinse_scene import (A_EFF, BLOCK, BUILT_WATER, Q_STOP, START_FILL, WASH, WET_MASS, Sponge,
                                            SpongeSink)

SINK = SpongeSink()


def test_the_sponge_is_theirs():
    sp = Sponge()
    pores, pad = sp.pores(), sp.pad(0)
    assert pores.free * 1e6 == pytest.approx(79.1, abs=0.05)
    assert START_FILL * pores.free * 1e6 == pytest.approx(59.3, abs=0.05)
    assert 4.0 * pad.size[0] * pad.size[1] == pytest.approx(A_EFF) and pad.size[0] / pad.size[1] == pytest.approx(
        BLOCK[0] / BLOCK[1])
    assert pad.porous and pad.center[2] == -0.5 * BLOCK[2] and pad.stiffness == pytest.approx(5e4 / 0.012)
    assert WET_MASS == pytest.approx(0.008 + 998.0 * BUILT_WATER) and WET_MASS * 1e3 == pytest.approx(67.2, abs=0.1)
    assert sp.lumped().built_water == BUILT_WATER


def test_a_wring_of_120_n_leaves_their_41_mL():
    """The pad's spring is GA's foam spring E A_eff (ε + 3ε³): 120 N settles the pad F / K_PAD deep, ε 0.48, 52 %
    full; the hand's target, F / KP under the touch (hand and pad in series), puts it there."""
    pores = Sponge().pores()
    depth = 120.0 / R.K_PAD
    eps = float(pores.strain(depth * R.K_PAD / A_EFF))
    assert (1.0 - eps) * pores.free * 1e6 == pytest.approx(41.4, abs=1.5) and depth < Q_STOP
    kp = R.GAINS["kp"]
    assert (120.0 / R.KP) * kp / (kp + R.K_PAD) == pytest.approx(depth)        # the target offset, split in series


def test_the_ledge_gate_box_is_the_charted_deck():
    lo, hi = np.asarray(SINK.ledge().lo), np.asarray(SINK.ledge().hi)
    (ledge,) = SINK.charts()
    assert ledge.cells == (137, 11) and hi[0] - lo[0] == pytest.approx(137 * 0.010, abs=0.002)
    assert lo[2] < SINK.sink.rim_z < hi[2] and lo[1] < SINK.at[1] < hi[1]
    assert SINK.rest_z - 0.5 * BLOCK[2] == pytest.approx(SINK.sink.rim_z)          # its scour face on the deck


def test_the_scour_face_turns_with_the_sponge():
    flat = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]])
    s = math.sin(math.radians(45.0))
    rolled = np.array([[1.0, 2.0, 3.0, s, 0.0, 0.0, math.cos(math.radians(45.0))]])    # 90° about x
    assert R.scour_face(flat)[0] == pytest.approx([1.0, 2.0, 3.0 - 0.5 * BLOCK[2]])
    assert R.scour_face(rolled)[0] == pytest.approx([1.0, 2.0 + 0.5 * BLOCK[2], 3.0], abs=1e-12)


def test_start_fills_the_pores_with_wash():
    calls: dict = {}
    lumped = SimpleNamespace(set_water=lambda v, body: calls.__setitem__("water", (np.asarray(v), body)),
                             set_species=lambda v, name, body: calls.__setitem__(name, (np.asarray(v), body)))
    flow = SimpleNamespace(lumped=lumped, set_input=lambda name, v: calls.__setitem__(name, np.asarray(v)))
    s = SINK.start(flow, 8, seed=2)
    water = START_FILL * Sponge().pores().free
    assert s[0] == WASH and ((s[1:] >= 0.67 * WASH) & (s[1:] <= 1.67 * WASH)).all()
    assert np.allclose(calls["water"][0], water) and calls["water"][1] == 0
    assert np.allclose(calls["soap"][0], s * water) and calls["soap"][0][0] * 1e6 == pytest.approx(44.5, abs=0.05)
    assert all((calls[e] == 0.0).all() for e in (*(drain_edge(b) for b in BOWLS), "faucet"))


def _sys_read(W: int = 2, K: int = 3) -> tuple[dict, dict, np.ndarray]:
    from alloy.plugins.surface_flow.supply_species import SP_LEDGER  # noqa: PLC0415

    out = {"species_mass": np.zeros((W, K)), "species_out": np.zeros((W, 3, K))}
    out |= {f"supply_species_{c}": np.zeros((W, K)) for c in SP_LEDGER}
    out["supply_species_in_flight"] = np.zeros((W, K))
    lumps = {"lump_species": np.tile([[[44.5e-6, 0.0, 0.0]]], (W, 1, 1))}
    return out, lumps, np.zeros((W, 3, K))


def test_the_system_book_follows_soap_between_the_pores_the_ledge_and_the_bowls():
    out, lumps, net = _sys_read()
    ref = R.system_book(out, lumps, net)
    lumps["lump_species"][:, 0, 0] -= 10e-6                       # squeezed onto the ledge
    out["species_mass"][:, 0] += 6e-6
    out["species_out"][:, 1, 0] += 4e-6                            # 4 µg of it off the ledge's open front, to a bowl
    out["supply_species_tanks"][:, 0] += 4e-6
    net[:, 1, 0] += 4e-6
    lumps["lump_species"][:, 0, 0] -= 3e-6                         # swapped into the rinse bowl
    net[:, 1, 0] += 3e-6
    drift = np.abs(R.system_book(out, lumps, net) - ref)
    rel, kg = closed(drift, np.full((2, 3), 44.5e-6))
    assert [ledger_closed(0.0, float(rel[w]), float(kg[w])) for w in range(2)] == [True, True]
    net[1, 1, 0] += 1e-8                                           # world 1's bowl gains 10 µg from nowhere
    rel, kg = closed(np.abs(R.system_book(out, lumps, net) - ref), np.full((2, 3), 44.5e-6))
    assert [ledger_closed(0.0, float(rel[w]), float(kg[w])) for w in range(2)] == [True, False]


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_one_dunk_and_the_wring_rinse_and_wring_the_sponge():
    """CPU: one dunk-and-squeeze, the 120 N wring, set down 45 mm from the squeeze spot (over the ring its squeezes
    wet: DC5's drink-back check), GA's settle. The books close (the bowls' 212 L round at 1e-13), no contact
    overflows, the pores lose most of their soap (one cycle: not yet under 0.02 g/L, so no_suds fails alone), wrung
    near their 41.4 mL and taking nothing back from the ledge while they sit, the scour face on the deck, gripped
    until the release (then hands-free is not judged: the hand has no pads)."""
    res = R.run(wring=R.Wring(cycles=1, soak=0.5, settle=4.0, rest_at=R.Wring.end_inset + 0.045))
    w = res["worlds"][0]
    fault = w["gates"]["no_simulation_fault"]
    assert res["overflow"] == [False] and fault["ok"] and fault["value"]["water_rel"] < 1e-11, fault
    assert w["failed"] == ["no_suds"] and 3.0 <= w["settle_s"] <= 4.0
    assert res["pore_g_l"][0] < 0.2 * WASH and 38.0 < res["pore_ml"][0] < 0.60 * 79.1
    assert res["end"][0]["pore_m3"][0] <= res["live"]["pore_m3"][0] + 1e-9         # no drink-back: 1 µL
    lo, hi = np.asarray(SINK.ledge().lo), np.asarray(SINK.ledge().hi)
    spot = [lo[0] + R.Wring.end_inset + 0.045, 0.5 * (lo[1] + hi[1]), SINK.sink.rim_z]     # where it was set down
    assert res["end"][0]["pos"][0] == pytest.approx(spot, abs=1e-3)
    assert res["gripped"]["held"] == [True] and res["end"][0]["held"] == [False]
    hands = w["gates"]["water_off_and_hands_free"]                 # AB-gym-3: the faucet off, no pads to read
    assert hands["ok"] is None and hands["value"]["faucet_gpm"] == 0.0 and w["judged"] == [5, 6]
    assert w["not_judged"] == ["water_off_and_hands_free"]
    assert w["terms"]["wrung"] == 1.0 and w["terms"]["returned"] == 1.0


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_a_starved_press_marks_its_world(monkeypatch):
    """CPU: room for 2 contacts per world against the 4 a 120 N press makes (its core comes within the 10 mm
    detection gap of the deck): the watch marks the world and the fault gate's ``finite`` fails (DC5's review)."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415
    from alloy.tasks import sponge_rinse_scene as S  # noqa: PLC0415
    from alloy.tasks.rinse_hand import OverflowWatch, PoseHand  # noqa: PLC0415

    monkeypatch.setattr(S, "CAPACITY", {**S.CAPACITY, "nconmax": 2})
    task = SpongeSink()
    flow, hand = task.make_flow(), PoseHand(bodies=(S.SPONGE,), f_max=150.0, **R.GAINS)
    sim = build_simulation(task.scene(lambda sim: task.start(flow, 1), 1), [flow, hand, OverflowWatch()], "cpu")
    h = R.Hand(sim, hand, 1)
    h.grip[:] = 1.0
    h.move(h.pos[0, 0] - [0.0, 0.0, 120.0 / R.KP], 0.05)
    h.hold(0.2)
    r, _ = R.readout(flow, h, R.bodies(sim.model, S.SPONGE), R.Where.of(flow), None)
    assert h.watch is not None and h.watch.bits is not None                     # WP8's watch, attached
    assert h.overflow.tolist() == [True] and r.finite.tolist() == [False]


def test_the_hand_steps_or_the_overflow_bits_into_its_watch():
    """The hand's record on a fake solver without a watch (the real starve: ``test_a_starved_press_marks_its_world``;
    the watch's substeps: WP8's ``test_rinse_overflow``): a narrowphase overflow in world 1 marks world 1 (and stays
    marked); the iteration-limit bits do not count (``rinse_hand.period_overflow``)."""
    from alloy._vendor.mujoco_warp import OverflowType  # noqa: PLC0415

    bits = np.array([int(OverflowType.ITERATIONS), int(OverflowType.NARROWPHASE)])
    hand = R.Hand.__new__(R.Hand)
    hand.sim = SimpleNamespace(step_period=lambda: None)
    hand.hand = SimpleNamespace(set_input=lambda name, value: None)
    hand.pos, hand.quat, hand.vel = np.zeros((2, 1, 3)), np.zeros((2, 1, 4)), np.zeros((2, 1, 3))
    hand.grip, hand.overflow = np.zeros((2, 1), np.float32), np.zeros(2, bool)
    hand.solver = SimpleNamespace(mjw_data=SimpleNamespace(overflow=SimpleNamespace(numpy=lambda: bits)))
    hand.watch = None
    hand.step()
    assert hand.overflow.tolist() == [False, True]
    bits[:] = 0
    hand.step()
    assert hand.overflow.tolist() == [False, True]
