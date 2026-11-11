# SPDX-License-Identifier: Apache-2.0
"""The pile the inertia floor was made for (WP13's clutter pile with YCB's large marker; coordinator rulings 2026-11-08).

100 objects (4 of them YCB 040 large markers, 15.8 g, axial r_g 6.2 mm) fall up to 0.55 m into a bin, 8 worlds, 0.7 s,
at the rigid step of a 10 ms control period (1.25 ms). Without the floor a marker spins up 0.4-0.7 rad per step in
contact and RK4 runs away: on the CPU's MuJoCo C route 3 of 8 worlds go INVALID and bodies fly at 8-240 m/s.

Clean (ruling 13:36Z, ``marker_pile.PileWatch``):
- every world's mechanical energy stays at or below its start + 1 % (``inertia_floor.ENERGY_TOL``);
- no body is faster than 2 x the fall's 3.3 m/s (``CLEAN_SPEED`` 6.6 m/s). A light body struck by a heavier one moving
  at V leaves at up to about (1 + e) V. On seed 1 a 2.4 m/s pudding box hands a marker 4.47 m/s while the bodies
  around it lose 0.6 J;
- no world is INVALID and none raises SPIN.

With the floor, seeds 0, 1 and 2 are clean, and the floored bodies are the 4 markers and nothing else. CPU, main
d7503c225: fastest 3.26 / 4.47 / 3.76 m/s, no energy gained, spin peak 0.27-0.35 rad per step. The two mutants are
not clean: an 8 mm floor flings a body to 17 m/s (SPIN 0.65), and plain implicitfast (the free-body rk4 rule off)
leaves a world INVALID at 12 m/s (SPIN 0.94).

The CPU runs MuJoCo C (users' CPU route): the suite's Warp CPU route steps 800 bodies too slowly for a law (over 2 h).
About 6.5 min per case on the CPU. ALLOY_DEVICE=cuda:0 runs it on MuJoCo Warp. Needs the YCB meshes (``alloy
fetch``); skipped without them."""

from __future__ import annotations

import pytest

import alloy
from alloy.contact import inertia_floor as IF
from alloy.scene import integrator_route as IR
from marker_pile import PileWatch, fetched, kind, pile

WORLDS, PERIODS = 8, 70


def _run(device, monkeypatch, seed: int) -> tuple[dict, list[str], list[str]]:
    if not fetched():
        pytest.skip("needs the YCB meshes (alloy fetch ycb_040_large_marker and the pile's other YCB cards)")
    if str(device).startswith("cpu"):
        monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")
    scene, names = pile(100, worlds=WORLDS, seed=seed)
    sim = alloy.build(scene, device=device)
    watch = PileWatch(sim, names)
    for _ in range(PERIODS):
        sim.step(1)
        watch.update()
    out = watch.verdict()
    print(f"marker pile seed {seed}: {out}")
    return out, sorted(sim.built.report.inertia_floor), sorted(n for n in names if kind(n) == "ycb_040_large_marker")


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_marker_pile_is_clean_with_the_floor(device, monkeypatch, seed):
    out, floored, markers = _run(device, monkeypatch, seed)
    assert floored == markers and len(markers) == 4, floored
    assert out["clean"], out


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mutant", ["floor_8mm", "plain_implicitfast"])
def test_the_law_catches_a_weaker_floor_and_plain_implicitfast(device, monkeypatch, mutant):
    if mutant == "floor_8mm":
        monkeypatch.setattr(IF, "R0_MIN", 8.0e-3)
    else:
        monkeypatch.setattr(IR, "choose_integrator", lambda *a, **k: IR.IntegratorChoice(False, "mutant: implicitfast"))
    out, _, _ = _run(device, monkeypatch, 0)
    assert not out["clean"], out
    assert max(out["fastest_m_s"]) > IF.CLEAN_SPEED or out["invalid"] > 0, out
