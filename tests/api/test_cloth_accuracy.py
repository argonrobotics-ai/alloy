# SPDX-License-Identifier: Apache-2.0
"""``alloy.Scene(cloth_accuracy=)`` (SCR-WP2-13): "default" or "super", the same words as ``liquid_accuracy``, and
"fast" (VBD's own 10 iterations; ruling 2026-11-08 15:58Z).

Both settings reach the cloth solver (SolverVBDAlloy's adaptive start + Chebyshev, at
``cloth_accuracy.DEFAULT_ITERATIONS`` or ``SUPER_ITERATIONS``); the setting round-trips through Scene JSON and is left out of it at the default (no hash
churn); a bad value is an error naming the keyword. The accuracy and cost of each setting are measured on a
Cusick-style drape (docs/notes/wp2-results.md, SCR-WP2-13).
"""

from __future__ import annotations

import pytest

import alloy
from alloy.scene import cloth_accuracy as CA


def _sheet(accuracy: str) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01, cloth_accuracy=accuracy)
    scene.add_ground()
    scene.add_cloth("c", size=(0.2, 0.2), pos=(0.0, 0.0, 0.05), resolution=7, self_contact=False)
    return scene


def test_the_setting_reaches_the_cloth_solver():
    for accuracy, iterations, guess, rho in (("default", CA.DEFAULT_ITERATIONS, "adaptive", 0.9),
                                             ("super", CA.SUPER_ITERATIONS, "adaptive", 0.9),
                                             ("fast", CA.FAST_ITERATIONS, "inertial", None)):
        sim = alloy.build(_sheet(accuracy), device="cpu")
        vbd = [s for s in sim.engine.solvers if hasattr(s, "particle_enable_self_contact")]
        assert len(vbd) == 1
        assert vbd[0].iterations == iterations
        assert vbd[0]._accel.cfg.initial_guess == guess and vbd[0]._accel.cfg.chebyshev_rho == rho
        sim.close()


def test_scene_json_keeps_super_and_leaves_out_default():
    assert "cloth_accuracy" not in alloy.Scene().to_dict()
    data = alloy.Scene(cloth_accuracy="super").to_dict()
    assert data["cloth_accuracy"] == "super"
    assert alloy.Scene.from_dict(data).cloth_accuracy == "super"
    assert alloy.Scene.from_dict(alloy.Scene(cloth_accuracy="fast").to_dict()).cloth_accuracy == "fast"


def test_a_bad_value_names_the_keyword():
    with pytest.raises(alloy.SceneError) as err:
        alloy.Scene(cloth_accuracy="high")
    assert err.value.field == "cloth_accuracy"


def test_estimate_shows_both_settings():
    """``alloy.estimate`` gives both cloth rows (accuracy text, cost) and charges the scene's own setting at the built
    entry's iterations. Only the cloth solver's term follows the iteration count, so the scene's cost ratio is above 1
    and at most the iteration ratio."""
    default = alloy.estimate(_sheet("default"), worlds=4, seconds=1.0, device="cpu")
    sup = alloy.estimate(_sheet("super"), worlds=4, seconds=1.0, device="cpu")
    assert set(default["cloth"]) == set(CA.CLOTH_ACCURACY) and default["cloth_accuracy"] == "default"
    assert default["vbd"]["iterations"] == CA.DEFAULT_ITERATIONS and sup["vbd"]["iterations"] == CA.SUPER_ITERATIONS
    ratio = default["cloth"]["super"]["cost_vs_default"]
    assert 1.0 < ratio <= CA.SUPER_ITERATIONS / CA.DEFAULT_ITERATIONS
    assert default["cloth"]["default"]["cost_vs_default"] == 1.0
    fast = default["cloth"]["fast"]["cost_vs_default"]      # each setting at its own iterations (ruling 15:58Z)
    assert CA.FAST_ITERATIONS / CA.DEFAULT_ITERATIONS <= fast < 1.0
    assert all(default["cloth"][acc]["accuracy"] == CA.ACCURACY[acc] for acc in CA.CLOTH_ACCURACY)
    assert default["wall_time_s"] == pytest.approx(default["cloth"]["default"]["wall_time_s"], rel=1e-12)
    assert sup["wall_time_s"] == pytest.approx(default["cloth"]["super"]["wall_time_s"], rel=1e-12)
    for acc in ("default", "super"):                     # the rows do not depend on the scene's own setting
        assert sup["cloth"][acc]["wall_time_s"] == pytest.approx(default["cloth"][acc]["wall_time_s"], rel=1e-12)


#: measured wall seconds (plain python, one host-1 core, after a warm-up step) of ``_sheet_over_ball`` for 0.3 s at
#: the 40-iteration default: docs/notes/wp2-results.md (estimate calibration, 2026-11-08)
MEASURED_CPU_S = {True: 57.16, False: 33.74}


def _sheet_over_ball(self_contact: bool) -> alloy.Scene:
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_sphere("ball", 0.08, pos=(0, 0, 0.08), static=True)
    scene.add_cloth("sheet", size=(0.4, 0.4), pos=(0, 0, 0.2), resolution=20, self_contact=self_contact)
    return scene


@pytest.mark.parametrize("self_contact", [True, False])
def test_cpu_estimate_is_within_its_stated_3x(self_contact):
    """The CPU cloth estimate (VBD_COST: per particle-iteration, self-contact factor) is within the 3x its basis
    states of the measured run, with and without self-contact (OSS-17: it was 28x low at the 40-iteration default)."""
    est = alloy.estimate(_sheet_over_ball(self_contact), 2, 0.3, device="cpu")
    assert est["vbd"]["self_contact"] is self_contact and est["vbd"]["particles"] == 441
    ratio = est["wall_time_s"] / MEASURED_CPU_S[self_contact]
    assert 1 / 3 < ratio < 3, (est["wall_time_s"], MEASURED_CPU_S[self_contact])


ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""
#: OSS-17's row: AGENTS.md section 5's arm (a 1 kg link) and a 0.3 m sheet falling onto it, CPU, 1 world, 5 control
#: periods: 49.88 s (WP2 on host-1: 49.47 s; 32 force exchanges per period, 2 passes each)
MEASURED_ARM_SHEET_S = 49.88


def _sheet_over_arm(resolution: int = 15) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    scene.add_cloth("sheet", size=(0.3, 0.3), pos=(0.15, 0, 0.7), resolution=resolution, mass=0.05)
    return scene


def test_cloth_beside_a_robot_is_charged_its_exchange_clock():
    """A coupled scene runs a physics step per force exchange, each with the lagged pair's 2 passes (OSS-17: the
    estimate counted 8 steps per period, 18x low): within the stated 3x of the measured run."""
    est = alloy.estimate(_sheet_over_arm(), 1, 0.05, device="cpu")
    clock = est["vbd"]["exchange_clock"]
    assert est["vbd"]["coupled"] and clock["exchanges"] == 32 and clock["passes"] == 2
    assert est["physics_steps_per_world"] == 5 * 32
    ratio = est["wall_time_s"] / MEASURED_ARM_SHEET_S
    assert 1 / 3 < ratio < 3, est["wall_time_s"]


def test_the_estimated_clock_is_the_schedulers():
    """The static clock (``_exchange_clock``) gives the exchanges per period the built scheduler steps."""
    scene = _sheet_over_arm(resolution=6)
    est = alloy.estimate(scene, 1, 0.01, device="cpu")
    sim = alloy.build(scene, device="cpu")
    sim.step(1)
    assert sim.physics_steps == est["physics_steps_per_world"] == est["vbd"]["exchange_clock"]["exchanges"]
    sim.close()
