# SPDX-License-Identifier: Apache-2.0
"""Granular MPM's friction cone (``alloy.integration.mpm_solve``, gap #5b): ``ALLOY_MPM_CONE`` picks Newton's
Drucker-Prager coefficient for a card's friction angle. "tan" (default) is mu = tan(phi), bitwise as before; "mc" is
Mohr-Coulomb's compression-cone match. The slope laws (slow, CPU, soil_sand_dry at 34.5 deg, a plane-strain layer 4
cells deep on a rough plate): between end walls the layer rests at 33 deg and flows at 36 with "tan", and still rests
at 36 with "mc" (too strong); with a free end it relaxes to a wedge at the repose angle
(docs/known_gaps/mpm-grain-slope-resolution.md)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy
from alloy.core.errors import SolverOptionError
from alloy.integration import mpm_solve

PHI = math.radians(34.5)


def test_the_default_cone_is_tan_phi_and_mc_is_the_compression_match(monkeypatch) -> None:
    monkeypatch.delenv(mpm_solve.CONE_ENV, raising=False)
    assert mpm_solve.cone_friction(PHI) == math.tan(PHI)
    assert "friction cone" not in mpm_solve.describe()            # the route reason is unchanged by default
    monkeypatch.setenv(mpm_solve.CONE_ENV, "mc")
    assert mpm_solve.cone_friction(PHI) == pytest.approx(0.8063, abs=1e-4)
    assert mpm_solve.cone_friction(math.radians(30.0)) == pytest.approx(2.0 * math.sqrt(3.0) / 5.0)
    assert "friction cone mc (ALLOY_MPM_CONE)" in mpm_solve.describe()


def test_an_unknown_cone_is_refused_with_the_fix(monkeypatch) -> None:
    monkeypatch.setenv(mpm_solve.CONE_ENV, "outer")
    with pytest.raises(SolverOptionError, match="ALLOY_MPM_CONE=mc"):
        mpm_solve.cone_friction(PHI)


def test_a_sand_card_gets_the_cone_coefficient(monkeypatch) -> None:
    from alloy.cards.library import CardLibrary, default_root
    from alloy.scene.emitters.deformable import mpm_values

    mat = CardLibrary(roots=[default_root()]).material("soil_sand_dry")
    phi = float(mat.plastic.friction_angle.value)
    monkeypatch.delenv(mpm_solve.CONE_ENV, raising=False)
    assert mpm_values(mat)["friction"] == math.tan(phi)
    monkeypatch.setenv(mpm_solve.CONE_ENV, "mc")
    assert mpm_values(mat)["friction"] == pytest.approx(6 * math.sin(phi) / (math.sqrt(3) * (3 - math.sin(phi))))


W, H = 0.06, 0.02                                                   # the layer's width and depth (4 cells of 5 mm)


def _layer(theta_deg: float, length: float, end_walls: bool):
    """A soil_sand_dry layer ``length`` x 6 x 2 cm on a rough plate tilted ``theta_deg``, between frictionless side
    walls (plane strain), with frictionless end walls or both ends free: (sim, local), ``local()`` the particles in
    (down the slope, across, normal to the plate) from the layer's centre on the plate [m]."""
    th = math.radians(theta_deg)
    plate = length + (0.14 if end_walls else 0.30)                  # a free end needs room to run out
    ex, ez = np.array([math.cos(th), 0.0, -math.sin(th)]), np.array([math.sin(th), 0.0, math.cos(th)])
    o = 0.15 * ez + np.array([0.0, 0.0, 0.2 + 0.5 * plate * math.sin(th)])
    pc = o if end_walls else o + (0.5 * (plate - length) - 0.02) * ex
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("plate", size=(plate, 0.12, 0.02), pos=tuple(pc - 0.01 * ez), rpy=(0.0, th, 0.0), static=True)
    s.add_box("sand", size=(length, W, H), pos=tuple(o + (H / 2 + 0.001) * ez), rpy=(0.0, th, 0.0),
              material="soil_sand_dry")
    s.set_friction("plate", "sand", 1.2)
    side, up = np.array([0.0, W / 2 + 0.005, 0.0]), (H + 0.01) * ez
    tall = 2 * H + 0.02
    walls = [("wall_l", (plate, 0.01, tall), pc + up + side), ("wall_r", (plate, 0.01, tall), pc + up - side)]
    if end_walls:
        walls += [("end_lo", (0.01, 0.12, tall), o + up + (length / 2 + 0.005) * ex),
                  ("end_hi", (0.01, 0.12, tall), o + up - (length / 2 + 0.005) * ex)]
    for name, size, pos in walls:                                   # frictionless: normal support only
        s.add_box(name, size=size, pos=tuple(pos), rpy=(0.0, th, 0.0), static=True)
        s.set_friction(name, "sand", 0.001)
    sim = alloy.build(s, device="cpu", seed=0)

    def local() -> np.ndarray:
        d = sim.state(world=0)["sand.points"].astype(np.float64) - o
        return np.stack([d @ ex, d[:, 1], d @ ez], 1)

    return sim, local


def _use_cone(monkeypatch, cone: str) -> None:
    if cone == "tan":
        monkeypatch.delenv(mpm_solve.CONE_ENV, raising=False)
    else:
        monkeypatch.setenv(mpm_solve.CONE_ENV, cone)


BOX = 0.64                                                          # 32 depths long: the toe wall carries little


@pytest.mark.slow
@pytest.mark.timeout(5400)
@pytest.mark.parametrize(("cone", "theta_deg", "flows"),
                         [("tan", 33.0, False), ("tan", 36.0, True), ("mc", 36.0, False)])
def test_a_walled_sand_layer_rests_below_its_cone_angle_and_flows_above(monkeypatch, cone, theta_deg, flows) -> None:
    """The tilting-box law: the top of the layer's middle third, 0.3 s after it is laid on the plate. Measured (CPU,
    2026-11-09, Newton 1.8): "tan" at 33 deg 0.52 mm and 0.35 mm/s, slowing (at 30 deg 0.08 mm); at 36 deg 6.9 mm and 22 mm/s;
    "mc" at 36 deg 0.12 mm, at rest (it flows from 42 deg: its cone is a 38.9 deg sand's)."""
    _use_cone(monkeypatch, cone)
    sim, local = _layer(theta_deg, BOX, end_walls=True)
    p0 = local()
    top = (np.abs(p0[:, 0]) < BOX / 6) & (np.abs(p0[:, 1]) < W / 4) & (p0[:, 2] > 0.75 * H + 0.001)
    sim.step(20)
    p1 = local()
    sim.step(10)
    p2 = local()
    assert sim.check()["ok"]
    travel, speed = float((p2[top, 0] - p0[top, 0]).mean()), float((p2[top, 0] - p1[top, 0]).mean() / 0.1)
    if flows:
        assert travel > 4e-3 and speed > 8e-3, (travel, speed)
    else:
        assert travel < 1.5e-3 and abs(speed) < 1.5e-3, (travel, speed)


def _toe_wedge(p0: np.ndarray, p: np.ndarray, length: float) -> tuple[float, float]:
    """(the toe wedge's slope to the plate [deg], how far up the slope from the old toe the top has dropped [m]),
    from the layer's top in 5 mm bins along the slope (middle half of the width)."""
    edges = np.arange(-length / 2 - 0.02, length / 2 + 0.30 + 1e-9, 0.005)

    def profile(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = q[np.abs(q[:, 1]) < W / 4]
        idx = np.digitize(q[:, 0], edges)
        keep = [i for i in range(1, len(edges)) if (idx == i).sum() >= 4]
        return edges[np.array(keep) - 1] + 0.0025, np.array([q[idx == i, 2].max() for i in keep]) - 0.001

    h0 = float(profile(p0)[1].max())
    x, h = profile(p)
    back = length / 2 - float(x[(h < 0.95 * h0) & (x > -length / 2 + 0.02)].min())   # past the head's own slump
    fit = (h > 0.15 * h0) & (h < 0.85 * h0) & (x > 0)
    return math.degrees(math.atan(-np.polyfit(x[fit], h[fit], 1)[0])), back


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_a_free_end_relaxes_to_a_wedge_at_the_repose_angle(monkeypatch) -> None:
    """A 16 cm layer on a 26 deg plate with both ends free. An end cannot stand: it relaxes to a wedge whose surface
    is at the friction angle to the horizontal, so it slumps back about H / (2 tan(phi - beta)) from the toe (6.7 cm
    here). Measured (CPU, 2026-11-09, Newton 1.8): a surface at 33.1 deg (card 34.5; a moving toe overshoots), the top
    dropped 8.2 cm back, at rest by 0.7 s (0.1 mm/s). This is the set-up gap #5b first read as "flows below the
    friction angle"."""
    _use_cone(monkeypatch, "tan")
    beta, length = 26.0, 0.16
    sim, local = _layer(beta, length, end_walls=False)
    p0 = local()
    sim.step(60)
    p1 = local()
    sim.step(10)
    p2 = local()
    assert sim.check()["ok"]
    assert float(np.abs(p2[:, 0] - p1[:, 0]).mean() / 0.1) < 1e-3                    # at rest
    wedge_deg, back = _toe_wedge(p0, p2, length)
    assert 31.5 < beta + wedge_deg < 35.5, wedge_deg
    estimate = H / (2.0 * math.tan(PHI - math.radians(beta)))
    assert estimate < back < 1.5 * estimate, (back, estimate)
