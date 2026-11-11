# SPDX-License-Identifier: Apache-2.0
"""The taped carton (alloy/tasks/gift_carton_tape.py; docs/notes/paper-pack.md). CPU.

Layout: the tape covers both majors along the seam and its tab; the tine, as wide as the tape, lies on the tab and short
of the bonded tape, the shaft past the tab's end; the peel pulls the grip twice the peeled length, inside the gantry's
travel. Build: the tape starts bonded on the majors and held by the tine at the tab only. The held major: taped shut
it starts closed, the tape's torque keeps it shut, and let go it springs up to its set as the untaped carton's does.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks.gift_carton import FLAPS, CartonSpec, carton_mjcf, flap_angles, hold_torque
from alloy.tasks.gift_carton_tape import CLEAR, OpenSpec, build, clear_gap, clear_plan, layout, peel_plan
from alloy.tasks.gift_gantry import FINGER_LEN

S = OpenSpec()


def test_tape_tab_and_tine_layout():
    lay = layout(S)
    c, t = S.flaps.carton, S.tape
    assert lay["x1"] - lay["x0"] == pytest.approx(c.length + c.caliper)            # the majors' width along x
    assert lay["centre"][0] - 0.5 * t.length == pytest.approx(lay["x0"] - t.tab)    # tab past the -x edge
    assert lay["centre"][0] + 0.5 * t.length == pytest.approx(lay["x1"])
    lo, hi = lay["tine"]
    assert lo < lay["x0"] - t.tab and lay["x0"] - hi > t.grip_reach                 # over the whole tab, not the bonds
    assert lay["palm"][0] + 0.004 < lay["x0"] - t.tab - 1e-3                        # the shaft past the tab
    r = lay["palm"][2] - FINGER_LEN - lay["z"]                                      # tine bottom over the tab
    assert r == pytest.approx(5e-4 + t.tine_gap) and r < t.grip_reach              # clear of it, the row in reach
    assert t.tine_width > 0.048                                                     # as wide as the tape


def test_peel_pulls_twice_the_peeled_length_inside_the_gantry():
    lay = layout(S)
    plan = dict((p, g) for p, _, g in peel_plan(S, lay))
    dx = plan["pull"]["gantry_x"]
    assert dx == pytest.approx(2.0 * (lay["x1"] - lay["x0"]) + 0.03, abs=0.05)
    assert lay["palm"][0] + dx < lay["palm"][0] + 0.9                               # gantry_x range +-0.9 m
    assert plan["pull"]["gantry_z"] == pytest.approx(S.tape.pull_height)             # flat over the tape


def test_tape_is_carried_clear_of_the_carton_before_it_is_let_go():
    lay = layout(S)
    dx = dict((p, g) for p, _, g in peel_plan(S, lay))["pull"]["gantry_x"]
    (_, _, tgt), = clear_plan(S, lay)
    assert dx < tgt["gantry_x"] == CLEAR[0] < 0.9                                  # on along +x, inside the travel
    assert tgt["gantry_z"] > S.tape.pull_height + 0.1                               # and up
    assert clear_gap(S, lay) > 0.1                                                  # flat, its end 0.1 m past the edge


@pytest.mark.timeout(240)
def test_tape_starts_bonded_on_the_majors_and_held_at_the_tab():
    sim, bonds, lay = build(S, device="cpu")
    st = np.asarray(bonds.read()["bond_state"]).ravel()
    x = sim.state(world=0)["tape.points"]
    on = (x[:, 0] > lay["x0"] + 1e-3) & (x[:, 0] < lay["x1"] - 1e-3)
    gap = np.abs(x[:, 1]) < 2e-3                                                    # over the 4 mm seam gap
    assert (st[on & ~gap] == 1).mean() > 0.95 and np.all(st[on & gap] != 1)         # pre-bonded where it lies
    assert np.all(st[~on & (x[:, 0] > lay["x0"])] != 2)
    held = x[st == 2]
    assert len(held) > 10 and held[:, 0].max() < lay["x0"] - S.tape.tab + 2e-3      # the tab's outer row only
    assert sim.check()["ok"]
    caps = [(c.destination, c.c, c.c_d) for c in sim.engine.stepper._caps]           # the task's exchange caps
    assert caps == [("film", 4.0, 0.5)]
    from alloy.integration.profiles import PROFILES                                 # and the profile back
    assert PROFILES["standard"].scheduler_policy is None


@pytest.mark.timeout(120)
def test_held_major_starts_closed_and_springs_to_its_set():
    import alloy

    out = {}
    for held in (0.0, 12.0):
        s = CartonSpec(held_deg=held)
        scene = alloy.Scene(dt=0.01)
        scene.add_ground()
        scene.add_robot("carton", mjcf=carton_mjcf(s))
        sim = alloy.build(scene, device="cpu")
        if held:
            sim.set_controls({"carton/major_p_hold": hold_torque(s)})               # the tape's hold
        a = []
        for k in range(3):
            x = sim.state(["carton/carton"] + [f"carton/{f}" for f in FLAPS], world=0)
            a.append(flap_angles(s, x["carton/carton.quat_wxyz"], {f: x[f"carton/{f}.quat_wxyz"] for f in FLAPS}))
            if held and k == 1:
                sim.set_controls({"carton/major_p_hold": 0.0})                      # the tape is off
            sim.run(1.0)
        out[held] = [x["major_p"] for x in a]
    assert out[12.0][0] == pytest.approx(0.0, abs=0.1) and abs(out[12.0][1]) < 0.5  # taped shut, held shut
    assert out[0.0][0] == pytest.approx(12.0, abs=0.1)
    assert out[12.0][2] == pytest.approx(out[0.0][1], abs=2.0)                      # let go: back to its set, sagged
