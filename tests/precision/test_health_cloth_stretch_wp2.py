# SPDX-License-Identifier: Apache-2.0
"""WP2's blown-up 5 mm drape rung is CLOTH_STRETCH's law (SCR-WP5-17, coordinator 2026-11-08 14:18Z; CPU, slow).

WP2's D03M Cusick drape (harness at 7b7d6c3ff, row b5_super_e5mm_L): a 30 cm specimen disc of the Petrak 2021 F1
twill (arm L card), meshed by the R6 twin's mesher at 5 mm, clamped between two 18 cm static discs, VBD cloth at
cloth_accuracy="super" with self-contact. The mesher's ring edges reach 1.1 mm, under the sheet's 2 mm contact
thickness, so the sheet starts in contact with itself and blows up: edges at 2,839-49,235 % on the L4, every particle
finite, and sim.check() read ok. On CPU, one world, through WP2's harness files: INVALID by CLOTH_STRETCH at check 62
(t 0.079 s, 525 %); the same run with the measure off (health.ranges.cloth = None) reads OK, check() ok, at 0.15 s
(scr/WP5.md, SCR-WP5-17 law row).

Since 71010132b (WP2, cloth_accuracy "default" and "super" hold self-contact particles out of the Chebyshev step) the
rung no longer blows up (bisected 2026-11-10 on CPU VMs: 84a569b64 INVALID, 71010132b DEGRADED). Measured on main
ec7a47cb2 through 0.15 s: every particle finite, the largest edge 89 % past its rest length, the 99th percentile 2 %,
and CLOTH_STRETCH at its 117.8 % peak from the first period (the start push of the 1.1 mm ring edges): DEGRADED,
not INVALID. This law now pins that reading, so a return of the blow-up fails it; the INVALID path past 500 % is
test_health_cloth_stretch.py::test_a_blown_sheet_is_invalid_and_check_names_it.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy

H = 2.0e-3                                 # the sheet's contact thickness: max(KES T0 0.405 mm, 2 mm)
F1 = {"areal_mass": 0.1352, "bending": 0.5 * (1.95e-6 + 6.1e-7), "t0": 4.05e-4}   # WP2's pack, petrak2021_F1
CARD = """schema: alloy.material/1
name: wp2_f1_arm_l
density: {{value: {density}, source: measured, note: "WP2 D03M petrak2021_F1: W 0.1352 kg/m2 over the 2 mm sheet"}}
surface:
  mu_static: {{value: 0.5, source: model, note: "clamped specimen"}}
  mu_dynamic: {{value: 0.5, source: model, note: "clamped specimen"}}
  restitution: {{value: 0.0, source: model, note: "fabric impacts fully damped"}}
elastic:
  youngs: {{value: {youngs}, source: model, note: "WP2 arm L: E = 12 (1 - nu^2) B / T0^3"}}
  poisson: {{value: 0.3, source: model, note: "WP2 SCR-WP2-14, pre-registered"}}
  bending: {{value: {bending}, source: measured, note: "KES B, mean of warp and weft"}}
"""


def _rig(tmp_path):
    from alloy.scene.twins import drape as R6

    (tmp_path / "materials").mkdir()
    youngs = 12.0 * (1.0 - 0.09) * F1["bending"] / F1["t0"] ** 3
    (tmp_path / "materials" / "wp2_f1_arm_l.yaml").write_text(
        CARD.format(density=F1["areal_mass"] / H, youngs=youngs, bending=F1["bending"]))
    v2, tris = R6.disc_mesh(0.005, 0)
    v = np.column_stack([v2, np.zeros(len(v2))])
    scene = alloy.Scene(dt=0.01, roots=[str(tmp_path)], cloth_accuracy="super")
    scene.add_ground(z=-0.5)
    scene.add_cylinder("support", R6.R_SUPPORT, 0.1, pos=(0, 0, -0.05), static=True)
    scene.add_cylinder("clamp", R6.R_SUPPORT, 0.01, pos=(0, 0, H + 0.005), static=True)
    start = np.random.default_rng(0).uniform(-2.0e-3, 2.0e-3, 2)
    d = 2.0 * float(np.abs(v[:, :2]).max())
    scene.add_cloth("specimen", size=(d, d), pos=(start[0], start[1], 0.5 * H), thickness=H, material="wp2_f1_arm_l",
                    self_contact=True)
    item = scene.items["specimen"]                           # the disc mesh in place of the square (WP2's harness)
    item["shape"] = {"mesh": {"path": "", "sha256": "", "scale": [1.0, 1.0, 1.0], "collision": "convex"}}
    item["mesh"] = {"vertices": v.tolist(), "faces": np.asarray(tris).tolist()}
    area = 0.5 * np.linalg.norm(np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]]), axis=1).sum()
    item["volume"] = float(area) * H
    item["card_extra"].setdefault("budget", {})["particles"] = len(v)
    return alloy.build(scene, device="cpu", seed=0)


def _gravity(sim, t: float) -> None:
    """WP2's 0.5 s smoothstep gravity ramp."""
    s = min(max(t / 0.5, 0.0), 1.0)
    g = np.array([0.0, 0.0, -9.81 * s * s * (3.0 - 2.0 * s)], np.float32)
    sim.engine.model.gravity.assign(np.tile(g, (sim.engine.model.gravity.shape[0], 1)))


@pytest.mark.law
@pytest.mark.slow
@pytest.mark.timeout(1800)                                   # CPU: build + 0.15 s of super-accuracy VBD, ~10 min
def test_wp2s_5_mm_drape_starts_stretched_and_no_longer_blows_up(tmp_path) -> None:
    sim = _rig(tmp_path)
    for k in range(15):                                      # 0.15 s
        _gravity(sim, k * sim.dt)
        sim.step(1)
    w = sim.health()["worlds"][0]
    assert w["severity"] == "DEGRADED" and "CLOTH_STRETCH" in w["flags"], w
    cs = w["cloth_stretch"]
    assert cs["cloth"] == "specimen" and cs["level"] == "warn", cs
    assert 50.0 < cs["percent"] < 500.0, cs                  # 117.8 % on main ec7a47cb2: the start push, no blow-up
    pts = sim.state(["specimen"])["specimen.points"]
    assert np.isfinite(pts).all()
