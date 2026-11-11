# SPDX-License-Identifier: Apache-2.0
"""Released spinning objects with per-world masses stay put (F5 of the 2026-11-07 fresh-agent trial).

In the trial, a cylinder put down by an arm at (0.5, 0.17) was launched at 28-704 m/s in 5 of 8 worlds on the CPU route
(MuJoCo C) once the scene drew a density per world. The per-world MjModels kept world 0's ``body_subtreemass``; MuJoCo
C divides by it for the subtree centre of mass (``mj_comPos``), and a free body spinning away from the origin with the
wrong value diverges within a few steps (a plain MuJoCo cylinder at x = 0.5 spinning at 10 rad/s: 0.44x or 2.26x its
mass blows up, 1.0x does not). The law: a robot platter at (0.5, 0.17) spins a cylinder to 15 rad/s and stops under
it, so the object spins on by itself; with densities drawn per world, every world stays OK and the object never
moves faster than its slip at spin-up (0.1 m/s without draws)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy

C = (0.5, 0.17)
WZ = 15.0
V_BOUND = 0.25          # m/s; without density draws the peak is 0.098 m/s (the spin-up slip)
PLATTER = f"""<mujoco><compiler angle="radian"/><worldbody>
  <body name="platter" pos="{C[0]} {C[1]} 0.011"><joint name="spin" type="hinge" axis="0 0 1" damping="0.5"/>
    <geom type="cylinder" size="0.1 0.01" mass="2" friction="0.9 0.005 0.0001"/></body></worldbody>
  <actuator><position joint="spin" kp="400" kv="20"/></actuator></mujoco>"""
# (seconds, platter rate rad/s): spin up in three steps, slow down, stop
SCHEDULE = ((0.2, WZ / 3), (0.2, 2 * WZ / 3), (0.6, WZ), (0.1, WZ / 2), (0.7, 0.0))


def _spin_and_stop(draws: bool, worlds: int = 4) -> tuple[np.ndarray, dict]:
    """Per-world peak linear speed [m/s] of the cylinder over the schedule, and sim.check()."""
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_robot("p", mjcf=PLATTER)
    scene.add_cylinder("obj", radius=0.022, height=0.05, pos=(C[0], C[1], 0.047), mass=0.1, friction=0.9)
    if draws:
        scene.randomize("obj", "density", 600.0, 2000.0)          # the trial's draw: 0.05-0.13 kg
    sim = alloy.build(scene, seed=0, device="cpu")
    th, peak = 0.0, np.zeros(worlds)
    for seconds, rate in SCHEDULE:
        tr = sim.run(seconds, record=["obj"], targets={"spin": (th, th + rate * seconds)})
        th += rate * seconds
        peak = np.maximum(peak, np.linalg.norm(tr["obj.vel"][..., :3], axis=-1).max(axis=0))
    return peak, sim.check()


@pytest.mark.parametrize("draws", [True, False], ids=["density_draws", "no_draws"])
def test_spinning_object_with_drawn_mass_stays_put(draws: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALLOY_CPU_ROUTE", raising=False)          # the default CPU route: MuJoCo C per world
    peak, check = _spin_and_stop(draws)
    assert check["ok"], check["problems"][:3]
    assert peak.max() < V_BOUND, f"object launched: peak speeds {np.round(peak, 3).tolist()} m/s"
