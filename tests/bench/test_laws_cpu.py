# SPDX-License-Identifier: Apache-2.0
"""The WP6 laws run end to end on CPU at --quick (marker `law`); numbers are hand-checked where exact."""

from __future__ import annotations

from pathlib import Path

import pytest

from alloy.bench import determinism as D
from alloy.bench.analytic import coulomb, energy, freefall, momentum, neohookean, pendulum, restitution
from alloy.bench.rigs import load_rig
from alloy.bench.runner import run_law
from alloy.bench.twins import bounce, incline

pytestmark = pytest.mark.law
RIGS = Path(__file__).resolve().parents[2] / "bench" / "rigs"


def run(law, twin, solver, data=None):
    return run_law(law, twin(), data, device="cpu", solver=solver, quick=True)


def test_a1_symplectic_euler_is_half_the_bound() -> None:
    r = run(freefall.LAW, freefall.FreeFallTwin, "xpbd")
    v = {x.name: x for x in r.verdicts}
    # semi-implicit Euler: x_n = x0 + v0 t - g t^2/2 - g h t/2 -> ratio (g h t / 2) / (g h t) = 0.5 (+ rounding)
    assert v["pos_err_ratio"].verdict == "pass" and v["pos_err_ratio"].error == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("solver", ["mujoco", "xpbd"])
def test_a1_fails_under_a_wrong_gravity(solver) -> None:
    """Audit AB-laws-2: the reference's g is the law's own 9.81, so worlds that fall at 9.0 m/s2 fail (the reference
    used to read model.gravity and followed the mutant: pass at ratio 0.003)."""
    class Slow(freefall.FreeFallTwin):
        def world_gravity(self, cond):
            return (0.0, 0.0, -9.0)

    r = run(freefall.LAW, Slow, solver)
    v = {x.name: x for x in r.verdicts}
    # after 1 s: x off by 0.81/2 m against a bound of g h t = 12 mm (ratio ~33 at 1.25 ms), v by 0.81 m/s against g h
    assert v["pos_err_ratio"].verdict == "fail" and v["vel_err_ratio"].verdict == "fail", r.verdicts
    assert v["pos_err_ratio"].error > 10 and v["vel_err_ratio"].error > 10
    assert freefall.G == 9.81


@pytest.mark.parametrize("plant", ["none", "inertia", "arm"])
def test_a2_reference_is_the_laws_own_pendulum(plant) -> None:
    """Audit AB-laws-2 (reference audit): A2's period reference is a solid sphere on the law's arm, not the built
    model's inertia and bob position. A bob built as a shell (I = 2/3 m r^2: period +0.52 %) or 2 % further out
    (period +1 %) fails the 0.1 % law; both passed while the reference read them back from the model."""
    class Planted(pendulum.PendulumTwin):
        arm_built = pendulum.ARM * (1.02 if plant == "arm" else 1.0)

        def build(self, b, cond, sample):
            law_arm, self.arm = self.arm, self.arm_built
            try:
                super().build(b, cond, sample)
            finally:
                self.arm = law_arm
            if plant == "inertia":
                b.body_inertia[-1] = b.body_inertia[-1] * (5.0 / 3.0)

    r = run(pendulum.LAW_A2, Planted, "mujoco")
    v = r.verdicts[0]
    assert v.name == "period_rel_err" and v.verdict == ("pass" if plant == "none" else "fail"), (v, r.monitors)
    assert v.error == pytest.approx({"none": 0.0, "inertia": 5.2e-3, "arm": 9.4e-3}[plant], abs=1e-3)


@pytest.mark.parametrize(("law", "twin"), [
    (pendulum.LAW_A2, pendulum.PendulumTwin), (pendulum.LAW_A3, pendulum.PendulumEnergyTwin),
    (energy.LAW, energy.ContactEnergyTwin), (momentum.LAW, momentum.MomentumTwin),
    (coulomb.LAW_ONSET, coulomb.CoulombOnsetTwin), (coulomb.LAW_SLIDE, coulomb.CoulombSlideTwin),
    (coulomb.LAW_ISO, coulomb.CoulombIsotropyTwin),
    (restitution.LAW, restitution.RestitutionTwin)])
def test_rigid_laws_run_on_featherstone(law, twin) -> None:
    r = run(law, twin, "featherstone")
    assert "error" not in r.monitors and {v.name for v in r.verdicts} == set(law.metrics)
    assert all(v.verdict in ("pass", "fail") for v in r.verdicts)


def test_a6_xpbd_restitution_passes() -> None:
    r = run(restitution.LAW, restitution.RestitutionTwin, "xpbd")
    assert r.verdicts[0].verdict == "pass"


def test_a6_uses_the_damping_path() -> None:
    # SPEC §5.3: MuJoCo gets native gains on the ball and STATIC_SOLMIX on the floor, as the assembler writes them
    twin = restitution.RestitutionTwin()
    run(restitution.LAW, lambda: twin, "mujoco")
    ke, kd = twin.model.shape_material_ke.numpy(), twin.model.shape_material_kd.numpy()
    ball = 1  # world 0 (e = 0): floor is shape 0, ball shape 1
    mass = float(twin.model.body_mass.numpy()[0])
    # a free ball routes rk4 (SCR-WP6-4; no F6 R2), on rk4's quarter-cap gains
    assert restitution.DR.routed_integrator(twin.model) == "rk4" == twin.monitors["integrator"]
    assert restitution.DR.routed_gains(twin.model) == "rk4" == twin.monitors["gains"]
    want = restitution.native_gains(restitution.K_MATERIAL, mass, 0.0, dt=restitution.LAW.dt, integrator="rk4")
    assert ke[ball] == pytest.approx(want[0], rel=1e-5) and kd[ball] == pytest.approx(want[1], rel=1e-5)
    solmix = twin.model.mujoco.geom_solmix.numpy()
    assert solmix[0] == pytest.approx(restitution.STATIC_SOLMIX)  # floor: the assembler's static solmix
    assert restitution.K_PAIR == pytest.approx(restitution.K_MATERIAL / 2)


def test_a15_vbd_matches_stable_neohookean() -> None:
    assert neohookean.uniaxial_stress(1.0) == pytest.approx(0.0, abs=1e-9)
    assert neohookean.stretch_for_stress(neohookean.uniaxial_stress(0.8)) == pytest.approx(0.8)
    r = run(neohookean.LAW, neohookean.NeoHookeanTwin, "vbd")
    assert r.verdicts[0].verdict == "pass"


def test_d1_rerun_bitwise_on_cpu() -> None:
    assert D.ulp_distance([1.0, -0.0], [1.0, 0.0]) == 0 and D.ulp_distance([1.0], [1.0000001]) == 1
    r = run_law(D.LAW_D1, D.RerunTwin(), None, device="cpu", solver="xpbd", quick=True)
    assert r.verdicts[0].verdict == "pass" and r.verdicts[0].error == 0.0


def test_r4_twin_on_synthetic_data() -> None:
    data = load_rig(RIGS / "R4.drop_bounce" / "synthetic-v1")
    r = run(bounce.LAW, bounce.BounceTwin, "xpbd", data)
    assert r.monitors["e_fit_calib"] == pytest.approx(0.7, abs=0.01)
    assert all(v.verdict == "pass" for v in r.verdicts) and len(r.verdicts) == 3


def test_r3_twin_on_synthetic_data_uses_calib_only() -> None:
    data = load_rig(RIGS / "R3.incline_friction" / "synthetic-v1")
    twin = incline.InclineTwin()
    seen = []
    orig = twin.bind_calib
    twin.bind_calib = lambda card, trials: (seen.extend(t.split for t in trials), orig(card, trials))[1]
    r = run_law(incline.LAW, twin, data, device="cpu", solver="featherstone", quick=True)
    assert set(seen) == {"calib"} and len(r.verdicts) == 6


def test_a5_isotropy_mujoco_static_cone_is_isotropic() -> None:
    r = run(coulomb.LAW_ISO, coulomb.CoulombIsotropyTwin, "mujoco")
    assert r.verdicts[0].verdict == "pass" and r.verdicts[0].error < 0.01


def test_a5_slide_mujoco_gates_azimuth_zero_spread_is_known_gap() -> None:
    # SCR-WP3-7: MuJoCo's off-axis slider spins up, so only azimuth 0 is gated; the 16-azimuth spread is a known gap
    r = run(coulomb.LAW_SLIDE, coulomb.CoulombSlideTwin, "mujoco")
    assert all(v.verdict == "pass" for v in r.verdicts) and r.monitors["accel_spread"] > 0.01
    assert r.monitors["gated_azimuths"].startswith("0 only")


def test_mujoco_laws_run_the_routed_settings() -> None:
    # SCR-WP6-4: free-body scenes route rk4, a hinged pendulum keeps implicitfast; both on the builtin kwargs
    a4 = run(momentum.LAW, momentum.MomentumTwin, "mujoco")
    a2 = run(pendulum.LAW_A2, pendulum.PendulumTwin, "mujoco")
    assert a4.config["solver_settings"]["kwargs"]["integrator"] == "rk4"
    assert a4.config["solver_settings"]["kwargs"]["cone"] == "elliptic"
    assert a2.config["solver_settings"]["kwargs"]["integrator"] == "implicitfast"
    assert all(v.verdict == "pass" for v in a4.verdicts)  # rk4 holds |dL|/L under 1e-3 (quick)

