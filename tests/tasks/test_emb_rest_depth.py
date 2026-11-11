# SPDX-License-Identifier: Apache-2.0
"""The shipped robots that stand on a floor rest on it, not in it (GAP-E1 slice 4b, coordinator 2026-11-10 00:42Z).

Health flags contact depth on the MuJoCo route: a world is DEGRADED past 1 mm, and INVALID when the depth stays past
1 cm for 0.1 s or passes 2 cm. Three shipped robots then read INVALID or DEGRADED standing still, each for contact
parameters that let a hard part sink into a hard floor:
- the Go2's feet (Menagerie's solimp 0.015 1 0.022): 13.7 mm deep at rest (tests/tasks/test_emb_go2_stand.py);
- the Duo's casters (no contact parameters in the URDF, so MuJoCo's defaults under 3.2 kN): 10.0 mm deep at rest,
  peak 13.5 mm; with firm wheels (alloy.tasks.emb_robots.firm_wheels) 0.43 mm, peak 0.51 mm;
- the H1-2 did not rest at all: its leg servos (a seated operator's gains) could not hold it up, it leaned from 1 s,
  lay flat at 2 s and landed 20-33 mm deep. With the leg and torso servos x 16 (alloy.tasks.emb_robots.STAND_GAIN)
  it stands: deepest contact 0.33 mm, 0.25 mm at rest.
Laws: started as a user starts it (``add_robot``, default targets), health's contact depth read every 2 ms never
passes 1 mm, on MuJoCo Warp (the GPU's route) and on the CPU's MuJoCo C route; and the H1-2 is still standing after
2.5 s (the earlier pack was on the floor by 2 s).
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from alloy.tasks.emb_robots import DUO_WHEELS, HARD_SOLIMP
from alloy.tasks.emb_scene import scene_path
from alloy.tasks.emb_specs import EMBODIMENTS, STIFF_SOLREF

REST_DEPTH_M = 1e-3            # [m] health's PEN_WARN: a world deeper than this is DEGRADED
DUO = ("e5_fr3_duo", "e5_fr3_duo_stiff", "e5_fr3_duo_route")
H1_2 = ("e7_h1_2", "e7_h1_2_stiff", "e7_h1_2_route")
# [s] past each robot's deepest moment (Duo 0.04 s, H1-2 0.25 s) and into its rest; H1-2: past its old fall (2 s)
SECONDS = {**dict.fromkeys(DUO, 1.0), **dict.fromkeys(H1_2, 2.5)}
STAND_DROP_M, STAND_LEAN = 0.01, 0.035     # [m], [rad]: a standing H1-2's pelvis sinks < 1 cm and leans < 2 deg


def depth_trace(key: str, seconds: float, body: str | None = None) -> tuple[np.ndarray, dict]:
    """Health's deepest contact [m] after each 2 ms period of ``seconds`` from the start, default targets, and the
    start and end state of ``body`` (when given)."""
    import alloy

    scene = alloy.Scene(dt=0.002)
    scene.add_robot("r", scene_path(EMBODIMENTS[key]))
    sim = alloy.build(scene, device="cpu")
    names = [f"r/{body}"] if body else []
    start, out = sim.state(names, world=0), []
    for _ in range(int(round(seconds / sim.dt))):
        sim.step(1)
        out.append(max((w.get("max_penetration_m") or 0.0) for w in sim.health()["worlds"]))
    assert sim.check()["ok"] and not sim.health()["invalid_worlds"], sim.check()["problems"]
    return np.asarray(out), {"start": start, "end": sim.state(names, world=0)}


@pytest.mark.parametrize("key", DUO)
def test_duo_wheels_are_firm(key: str) -> None:
    """The four wheel geoms are hard and stiff; their shapes and friction are the URDF's."""
    m = mujoco.MjModel.from_xml_path(str(scene_path(EMBODIMENTS[key])))
    wheels = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in DUO_WHEELS]
    assert len(wheels) == 4
    for g in wheels:
        assert tuple(m.geom_solimp[g]) == HARD_SOLIMP and m.geom_solref[g, 0] <= STIFF_SOLREF[0]
        assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER and m.geom_friction[g, 0] == 1.0


@pytest.mark.timeout(900)
@pytest.mark.parametrize("route", ["warp", "mjc"])
@pytest.mark.parametrize("key", list(SECONDS))
def test_a_standing_robot_rests_on_the_floor_not_in_it(key: str, route: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    depth, state = depth_trace(key, SECONDS[key], "pelvis" if key in H1_2 else None)
    tail = depth[-len(depth) // 4:]
    print(f"{key} {route}: peak {1e3 * depth.max():.3f} mm at {0.002 * (int(depth.argmax()) + 1):.3f} s, "
          f"last quarter {1e3 * tail.max():.3f} mm")
    assert 0.0 < depth.max() <= REST_DEPTH_M, (depth.max(), int(depth.argmax()))
    if key in H1_2:
        a, b = state["start"], state["end"]
        drop = float(a["r/pelvis.pos"][2] - b["r/pelvis.pos"][2])
        lean = 2.0 * float(np.arccos(min(1.0, abs(float(np.dot(a["r/pelvis.quat_wxyz"], b["r/pelvis.quat_wxyz"]))))))
        print(f"{key} {route}: pelvis sank {1e3 * drop:.2f} mm, leaned {np.degrees(lean):.2f} deg")
        assert abs(drop) <= STAND_DROP_M and lean <= STAND_LEAN, (drop, lean)
