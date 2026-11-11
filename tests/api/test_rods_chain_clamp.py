# SPDX-License-Identifier: Apache-2.0
"""A clamped rod on the chain route (opt-in ``add_rod(clamp="start", route="chain")``, S4 2026-11-09), CPU.

A clamped rod steps on the elastic rod (``rod_vbd``). Beside a robot that costs 83-95 s per simulated second on the
CPU, where the chain steps inside MuJoCo with the robot at about 1.3 s. With ``route="chain"`` and clamp="start" the
rod stays on the chain: the router's decision carries ``ROD_CLAMP_NUMBER``, and the chain emitter welds link 0 to the
world, as it does for a tree weld's root. The chain-grip rule still covers it, since a clamp is not an attachment.
- The arguments: the chain holds its first end only (clamp "end" or "both" with route="chain" raises), and an unknown
  route raises. route= survives Scene JSON.
- A clamped chain hangs from a fixed first link: link 0 does not move, the far end sags.
- R4's gripper rig (tests/api/test_rods_grip.py, sliding pads squeezing 4.2 N, the wrist rolled 0.3 rad over 0.5 s,
  then held 0.7 s) on the clamped chain, on users' CPU route (MuJoCo C): the gripped segment turns with the wrist
  within 2 % when the ramp ends (1.0000) and stays within 1 % over the hold (-0.0003), the free span twists uniformly
  (every joint within 1 % of the mean), and the check is ok. On the same host it ran 1.29 s per simulated second,
  against 82.9 s for the rod_vbd rig. On MuJoCo Warp (the GPU's route; the suite's CPU default) the held cable stays
  with the grip since the stick anchor (alloy.contact.mjw_stick_anchor): 1.0000 at the ramp's end and after the
  hold, bound 1 %. Before it, it crept back about 2 % per second (0.9976, -1.7 % after the hold).
- The wrist carries the cable's torque (G J / h) x the mean free joint twist within 3 %, h = L / (N - 1) the joint
  spacing, as in R4's law: measured 0.997 (MuJoCo C, gravity on), with the pads welded to the wrist at a 0.1 mm
  overlap (a MuJoCo-sized one; R4's 2.1 um is sized for the elastic rod's contact). Sliding pads read 0.89: the
  servoed pads let the sagging cable end sit 4.4 mm below the roll axis, and they carry its weight there, a moment
  that is not the cable's twist ([internal note] notes/wp9-runs/s4/chain_torque: with gravity off the sliding rig reads
  0.9999, and the pads' moment about the end link's own axis is 1.002 of the springs').
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

import alloy
from alloy.scene.emitters.chain import ROD_CLAMP_NUMBER

_spec = importlib.util.spec_from_file_location("grip", Path(__file__).with_name("test_rods_grip.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


OVERLAP = 1.0e-4                                    # welded pads' overlap with the cable (m)


def _welded_wrist() -> str:
    """R4's welded-pad wrist with the pads OVERLAP into the cable instead of R4's 2.1 um."""
    y0, y1 = G.R + 0.004 - G.SQUEEZE / G.KE, G.R + 0.004 - OVERLAP
    return G._wrist(True).replace(f"{y0:.9f}", f"{y1:.9f}")


def _cable(sc: alloy.Scene, **kw) -> None:
    sc.add_rod("cable", length=G.L, radius=G.R, pos=(0.0, 0.0, G.Z), material="cable_usb_composite", segments=G.N,
               **kw)


@pytest.mark.parametrize("clamp", ["end", "both"])
def test_the_chain_holds_its_first_end_only_and_routes_are_checked(clamp):
    with pytest.raises(alloy.SceneError, match="first end only"):
        _cable(alloy.Scene(), clamp=clamp, route="chain")
    with pytest.raises(alloy.SceneError, match="route must be one of"):
        _cable(alloy.Scene(), clamp="start", route="cloth")
    sc = alloy.Scene()
    _cable(sc, clamp="start", route="chain")
    assert alloy.Scene.from_dict(json.loads(json.dumps(sc.to_dict()))).items["cable"]["rod"]["route"] == "chain"


def test_a_clamped_chain_hangs_from_its_fixed_first_link():
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    _cable(sc, clamp="start", route="chain")
    sim = alloy.build(sc, device="cpu")
    row = sim.routes()["cable"]
    assert row["route"] == "chain" and any("link 0 welded" in r for r in row["reasons"]), row
    assert sim.built.report.decisions["cable"].numbers[ROD_CLAMP_NUMBER] == 1.0
    p0 = sim.state(["cable"], world=0)["cable.points"].copy()
    sim.step(30)
    p = sim.state(["cable"], world=0)["cable.points"]
    assert np.abs(p[:2] - p0[:2]).max() < 1e-6, np.abs(p[:2] - p0[:2]).max()     # link 0 fixed
    assert p0[-1, 2] - p[-1, 2] > 0.01, p[-1, 2]                                    # the free end sags
    assert sim.check()["ok"], sim.check()["problems"]


@pytest.mark.law
@pytest.mark.timeout(600)
@pytest.mark.parametrize("route,hold", [("mjc", 0.01), ("warp", 0.01)])
def test_a_gripper_twists_a_clamped_chain_and_holds_it(route, hold, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)        # mjc: users' CPU route (the suite's default is MuJoCo Warp)
    sc = alloy.Scene(dt=0.01)
    _cable(sc, clamp="start", route="chain")
    sc.add_robot("hand", mjcf=G._wrist(False))
    sim = alloy.build(sc, device="cpu")
    assert sim.routes()["cable"]["route"] == "chain"
    seg = [n for n in sim.names()["bodies"] if n.startswith("cable/")][-1]
    q0 = sim.state([seg], world=0)[f"{seg}.quat_xyzw"].copy()

    def angles() -> tuple[float, float, np.ndarray]:
        st = sim.state(["cable", "hand/roll", seg], world=0)
        return float(st["hand/roll.q"]), G._about_x(st[f"{seg}.quat_xyzw"], q0), st["cable.twist"][:-2]

    sim.set_targets({"lf": 0.02, "rf": 0.02})
    sim.step(20)
    sim.run(0.5, targets={"roll": (0.0, G.PHI)})
    wrist0, end0, _ = angles()
    sim.set_targets({"roll": G.PHI})
    sim.step(70)
    wrist, end, free = angles()
    print(f"ramp end {end0 / wrist0:.4f}; run-on {end / wrist - 1.0:+.4f}; "
          f"spread {float(np.abs(free / free.mean() - 1.0).max()):.2e}")
    assert end0 == pytest.approx(wrist0, rel=0.02, abs=0)
    assert end == pytest.approx(wrist, rel=hold, abs=0)
    assert float(np.abs(free / free.mean() - 1.0).max()) < 0.01
    assert math.isfinite(float(free.sum())) and sim.check()["ok"], sim.check()["problems"]


@pytest.mark.law
@pytest.mark.timeout(600)
def test_the_wrist_carries_the_clamped_chains_twist_torque(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")        # users' CPU route: one MjData per world
    sc = alloy.Scene(dt=0.01)
    _cable(sc, clamp="start", route="chain")
    sc.add_robot("hand", mjcf=_welded_wrist())
    sim = alloy.build(sc, device="cpu")
    d = sim.built.report.decisions["cable"].numbers
    k = d["GJ"] / (d["rod_length"] / (G.N - 1))         # N m per rad, per joint
    mj = sim.engine.solvers[0]
    m, x = mj.mj_model, mj.datas[0]
    roll = next(j for j in range(m.njnt) if m.joint(j).name.endswith("roll"))
    dof, act = int(m.jnt_dofadr[roll]), next(i for i in range(m.nu) if int(m.actuator_trnid[i, 0]) == roll)
    sim.run(0.5, targets={"roll": (0.0, G.PHI)})
    sim.set_targets({"roll": G.PHI})
    ratio = []
    for _ in range(14):
        sim.step(5)
        wrist = float(x.qfrc_bias[dof] - x.actuator_force[act] - x.qfrc_passive[dof])
        ratio.append(-wrist / (k * float(sim.state(["cable"], world=0)["cable.twist"][:-2].mean())))
    torque = float(np.mean(ratio[2:]))
    print(f"torque {torque:.4f} ({min(ratio[2:]):.3f}-{max(ratio[2:]):.3f})")
    assert torque == pytest.approx(1.0, rel=0.03, abs=0)
    assert sim.check()["ok"], sim.check()["problems"]
