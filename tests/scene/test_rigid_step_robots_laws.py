# SPDX-License-Identifier: Apache-2.0
"""GAP-R1 physics laws behind the robot caps of the opt-in rigid step (``alloy.scene.rigid_step_robots``), on raw
MuJoCo C so they run in seconds and do not depend on the module (notes: [internal note]).

Law A, servo cap ``omega dt <= max(1, zeta)`` (omega = sqrt(kp / I_eff), I_eff the joint-space inertia with armature,
  zeta = (kv + joint damping) / (2 sqrt(kp I_eff))):
  implicitfast steps  v+ = (v - dt omega^2 x) / (1 + 2 zeta omega dt),  x+ = x + dt v+  (kv is implicit, kp explicit),
  stable iff  omega dt < 2 zeta + 2 sqrt(zeta^2 + 1)  (2 at zeta 0; Jury's test on the 2x2 update). Euler keeps the
  actuator's kv explicit: stable iff omega dt < -2 zeta + 2 sqrt(zeta^2 + 1), so robot entries must not run it.
  Accuracy is first order: a step response lags the fine one by about half a step, so it differs by at most
  0.5 omega dt x (peak speed / omega) of the step. An overdamped servo follows its slow pole kp / b: at
  omega dt = zeta (dt kp / b = 0.5) it differs by 0.12 of the step, with a stability margin of 4 or more.
Law B, contact time constant cap ``dt <= tau / 2``: MuJoCo clamps a solref time constant to 2 dt (refsafe), so a
  contact row carrying force f rests at  |r| = f A (1 - d) dmax^2 tau_eff^2 zeta^2 / d^2,  tau_eff = max(tau, 2 dt),
  A the two bodies' translational inverse weights, d = d(|r|) the solimp impedance.
Law E, damping cap ``(1 + dt b / I_eff) / (1 + dt0 b / I_eff) <= 1.25`` (b = kv + joint damping, dt0 the fixed step):
  implicitfast solves a contact against the inertia M, then integrates with M + dt D, so the contact's push moves a
  damped joint 1 / (1 + dt b / I) as much and a driven link goes that much deeper before it stops.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

INERTIA = 0.02                  # kg m^2 about the hinge
OMEGA = 100.0                   # rad/s: kp = OMEGA^2 INERTIA
FINE = 0.05                     # omega dt of the reference run
G = 9.81
TAU = 0.004                     # s: the contact time constant under test
D0, DMAX, WIDTH = 0.9, 0.95, 0.001      # MuJoCo's default solimp (power 2, midpoint 0.5)


# --- law A: the servo ---------------------------------------------------------------------------------------------

def servo(zeta: float, dt: float, integrator: str, armature: float = 0.0, joint_share: float = 0.0) -> mujoco.MjModel:
    """A hinge of joint-space inertia INERTIA (``armature`` of it as armature) under a position servo, no gravity;
    ``joint_share`` of the damping b = 2 zeta omega I sits on the joint, the rest is the actuator's kv."""
    kp, b, body = OMEGA**2 * INERTIA, 2.0 * zeta * OMEGA * INERTIA, INERTIA - armature
    kv, damping = (1.0 - joint_share) * b, joint_share * b
    xml = f"""<mujoco><option timestep="{dt!r}" integrator="{integrator}" gravity="0 0 0"/>
    <worldbody><body><joint name="j" type="hinge" axis="0 1 0" armature="{armature!r}" damping="{damping!r}"/>
    <inertial pos="0 0 0" mass="1" diaginertia="{body!r} {body!r} {body!r}"/></body></worldbody>
    <actuator><position joint="j" kp="{kp!r}" kv="{kv!r}"/></actuator></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def step_response(m: mujoco.MjModel, steps: int) -> np.ndarray:
    """Joint angle after each step of a unit target step; inf from the first non-finite or runaway value on."""
    d = mujoco.MjData(m)
    d.ctrl[0] = 1.0
    q = np.full(steps + 1, np.inf)
    q[0] = 0.0
    for i in range(steps):
        mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos[0]) or abs(d.qpos[0]) > 1e3:
            break
        q[i + 1] = d.qpos[0]
    return q


def response(zeta: float, a: float, integrator: str = "implicitfast", horizon: float | None = None) -> np.ndarray:
    T = horizon if horizon is not None else settle_time(zeta)
    return step_response(servo(zeta, a / OMEGA, integrator), int(round(T * OMEGA / a)))


def settle_time(zeta: float) -> float:
    """Eight time constants of the slowest pole [s]."""
    return 8.0 * max(1.0 / (zeta * OMEGA), 2.0 * zeta / OMEGA)


def tracking(zeta: float, a: float, integrator: str = "implicitfast") -> tuple[float, float]:
    """(largest difference from the fine run, overshoot beyond the fine run's), in units of the step."""
    fine, q = response(zeta, FINE), response(zeta, a, integrator)
    ref = np.interp(np.arange(len(q)) * a, np.arange(len(fine)) * FINE, fine)
    return float(np.max(np.abs(q - ref))), float(np.max(q) - np.max(fine))


def peak_speed(zeta: float) -> float:
    """Peak speed of the continuous unit step response, over omega."""
    if zeta < 1.0:
        return math.exp(-zeta / math.sqrt(1.0 - zeta**2) * math.acos(zeta))
    if zeta == 1.0:
        return math.exp(-1.0)
    return math.exp(-zeta / math.sqrt(zeta**2 - 1.0) * math.acosh(zeta))


def stable_limit(zeta: float, implicit_kv: bool = True) -> float:
    """The largest stable omega dt (module docstring)."""
    return (2.0 if implicit_kv else -2.0) * zeta + 2.0 * math.sqrt(zeta**2 + 1.0)


@pytest.mark.parametrize("zeta", [0.3, 1.0])
@pytest.mark.parametrize("a", [0.25, 0.5, 1.0])
def test_servo_tracks_the_fine_run_up_to_omega_dt_1(zeta: float, a: float) -> None:
    """Measured (zeta 0.3 / 1.0): 0.046 / 0.017 at 0.25, 0.10 / 0.040 at 0.5, 0.23 / 0.087 at omega dt 1; the bound
    is a half-step lag at the response's peak speed. Overshoot grows by at most 0.11 of the step (bound 0.13 a)."""
    track, overshoot = tracking(zeta, a)
    assert track <= 0.5 * a * peak_speed(zeta), (track, overshoot)
    assert overshoot <= 0.13 * a, (track, overshoot)


def test_servo_error_is_first_order_in_the_step() -> None:
    """Halving omega dt halves the difference (ratio 2.0-2.3 measured): the cap's constant trades accuracy linearly."""
    for zeta in (0.3, 1.0):
        coarse, fine = tracking(zeta, 1.0)[0], tracking(zeta, 0.5)[0]
        assert 1.8 <= coarse / fine <= 2.5, (zeta, coarse, fine)


@pytest.mark.parametrize("zeta", [0.05, 0.3, 1.0, 5.0])
def test_implicitfast_stability_limit(zeta: float) -> None:
    """Bounded 10 % under omega dt = 2 zeta + 2 sqrt(zeta^2 + 1), diverging 10 % over it (400 steps)."""
    limit = stable_limit(zeta)
    under = response(zeta, 0.9 * limit, horizon=400 * 0.9 * limit / OMEGA)
    over = response(zeta, 1.1 * limit, horizon=400 * 1.1 * limit / OMEGA)
    assert np.all(np.isfinite(under)) and abs(under[-1] - 1.0) < 1e-6, under[-1]
    assert not np.all(np.isfinite(over))


def test_lightly_damped_servo_diverges_at_omega_dt_2p2() -> None:
    """zeta 0.05: omega dt 1.9 rings but settles, 2.2 diverges (limit 2.10) under implicitfast and Euler."""
    for integrator in ("implicitfast", "Euler"):
        assert np.all(np.isfinite(response(0.05, 1.9, integrator, horizon=4.0)))
        assert not np.all(np.isfinite(response(0.05, 2.2, integrator, horizon=4.0)))


def test_euler_keeps_kv_explicit() -> None:
    """Why the cap assumes implicitfast: at zeta 5 Euler diverges at omega dt 0.25 (its limit 0.198) and at zeta 1 at
    omega dt 1 (limit 0.83), where implicitfast tracks the fine run."""
    assert stable_limit(5.0, implicit_kv=False) == pytest.approx(0.198, abs=1e-3)
    assert not np.all(np.isfinite(response(5.0, 0.25, "Euler")))
    assert not np.all(np.isfinite(response(1.0, 1.0, "Euler", horizon=4.0)))
    assert np.all(np.isfinite(response(1.0, 0.5, "Euler", horizon=4.0)))
    assert tracking(5.0, 0.25)[0] < 0.01 and tracking(1.0, 1.0)[0] < 0.1


@pytest.mark.parametrize("zeta", [2.0, 5.0, 10.0])
def test_overdamped_servo_at_its_cap_follows_the_slow_pole(zeta: float) -> None:
    """The cap's overdamped branch, omega dt <= zeta (dt kp / b <= 0.5): the response differs from the fine run by
    0.117 of the step at zeta 5 and 10 (bound 0.15) without overshoot, and the step is a quarter of the stability
    limit or less. At omega dt = 1 it differs by 0.018 / 0.009 (zeta 5 / 10)."""
    assert stable_limit(zeta) / zeta >= 4.0
    assert tracking(zeta, 1.0)[0] <= 0.1 / zeta
    track, overshoot = tracking(zeta, zeta)
    assert track <= 0.15 and overshoot <= 0.01, (track, overshoot)
    assert tracking(zeta, 2.0 * zeta)[0] > 0.3          # twice the cap: a third of the step off


def test_joint_damping_counts_as_kv() -> None:
    """zeta takes kv + joint damping: moving half of b from the actuator to the joint leaves the response alone."""
    for zeta, a in ((0.3, 1.0), (5.0, 5.0)):
        split = step_response(servo(zeta, a / OMEGA, "implicitfast", joint_share=0.5), 200)
        assert np.allclose(split, step_response(servo(zeta, a / OMEGA, "implicitfast"), 200), rtol=0, atol=1e-9)


def test_servo_inertia_counts_armature() -> None:
    """I_eff is the joint-space inertia diagonal: half of it moved to armature leaves the stability limit in place."""
    limit = stable_limit(0.05)
    for a, bounded in ((0.9 * limit, True), (1.1 * limit, False)):
        m = servo(0.05, a / OMEGA, "implicitfast", armature=0.5 * INERTIA)
        assert m.dof_M0[0] == pytest.approx(INERTIA, rel=1e-12) and m.dof_armature[0] == 0.5 * INERTIA
        assert bool(np.all(np.isfinite(step_response(m, 400)))) is bounded


# --- law B: the contact time constant -----------------------------------------------------------------------------

def impedance(pen: float) -> float:
    """MuJoCo's default solimp at penetration ``pen`` [m]."""
    x = min(pen / WIDTH, 1.0)
    y = 2.0 * x * x if x < 0.5 else 1.0 - 2.0 * (1.0 - x) ** 2
    return D0 + (DMAX - D0) * y


def rest_penetration(row_acc: float, dt: float) -> float:
    """|r| = f A (1 - d) dmax^2 tau_eff^2 / d^2 at zeta 1 (``row_acc`` = f A [m/s^2]), by fixed point in d(|r|)."""
    tau = max(TAU, 2.0 * dt)
    r = 0.0
    for _ in range(60):
        d = impedance(r)
        r = (1.0 - d) * row_acc * DMAX**2 * tau**2 / d**2
    return r


def settle(xml: str, ctrl: float = 0.0, seconds: float = 1.0) -> mujoco.MjData:
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.ctrl[:] = ctrl
    for _ in range(int(round(seconds / m.opt.timestep))):
        mujoco.mj_step(m, d)
    return d


def resting(shape: str, dt: float) -> float:
    """Rest penetration [m] of a 1 kg sphere (one contact) or box (four) on a plane, solref (TAU, 1)."""
    geom = 'type="sphere" size="0.05"' if shape == "sphere" else 'type="box" size="0.05 0.05 0.05"'
    d = settle(f"""<mujoco><option timestep="{dt!r}" integrator="implicitfast"/>
    <default><geom solref="{TAU} 1"/></default>
    <worldbody><geom type="plane" size="1 1 0.1"/>
    <body pos="0 0 0.05"><freejoint/><geom {geom} mass="1"/></body></worldbody></mujoco>""")
    return 0.05 - float(d.qpos[2])


FINGER_KG, BOX_KG, GRIP_KP = 0.2, 0.1, 100.0


def squeeze(dt: float, force: float = 5.0) -> tuple[float, float]:
    """Two slide fingers under position servos pinch a free 4 cm box (no gravity): (pad penetration [m], grip force
    [N]) at rest. Each pad makes four contacts."""
    finger = f'<geom type="box" size="0.01 0.015 0.015" mass="{FINGER_KG}"/>'
    d = settle(f"""<mujoco><option timestep="{dt!r}" integrator="implicitfast" gravity="0 0 0"/>
    <default><geom solref="{TAU} 1"/></default>
    <worldbody>
    <body pos="0 0 0"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="{BOX_KG}"/></body>
    <body pos="-0.03 0 0"><joint name="l" type="slide" axis="1 0 0" damping="20"/>{finger}</body>
    <body pos="0.03 0 0"><joint name="r" type="slide" axis="-1 0 0" damping="20"/>{finger}</body>
    </worldbody>
    <actuator><position joint="l" kp="{GRIP_KP}"/><position joint="r" kp="{GRIP_KP}"/></actuator></mujoco>""",
               ctrl=force / GRIP_KP)
    travel = 0.5 * float(d.qpos[7] + d.qpos[8])         # each finger's travel past touching (the box stays centred)
    return travel, GRIP_KP * (force / GRIP_KP - travel)


@pytest.mark.parametrize(("shape", "contacts"), [("sphere", 1), ("box", 4)])
@pytest.mark.parametrize("ratio", [0.125, 0.5, 1.0, 2.0])
def test_rest_penetration_follows_the_clamped_time_constant(shape: str, contacts: int, ratio: float) -> None:
    """Measured, sphere: 17.48 um at dt <= tau / 2, 69.54 um at dt = tau, 257.5 um at dt = 2 tau (formula to 1e-9);
    the box rests a quarter as deep (four rows share the weight)."""
    assert resting(shape, ratio * TAU) == pytest.approx(rest_penetration(G / contacts, ratio * TAU), rel=0.01, abs=0)


def test_rest_penetration_grows_as_the_square_of_the_clamp() -> None:
    """(tau_eff / tau)^2 = 4 and 16, less what the impedance gains with depth: x3.98 and x14.7 measured."""
    base = resting("sphere", 0.5 * TAU)
    assert resting("sphere", 0.125 * TAU) == pytest.approx(base, rel=1e-6, abs=0)
    assert resting("sphere", TAU) / base == pytest.approx(4.0, rel=0.2)
    assert resting("sphere", 2.0 * TAU) / base == pytest.approx(16.0, rel=0.2)


def test_grasp_squeeze_is_step_independent_up_to_half_tau() -> None:
    """The grasp-squeeze cap: a 5 N pinch sits 33.36 um into each pad at dt = tau / 8, tau / 4 and tau / 2."""
    base = squeeze(0.125 * TAU)[0]
    assert base == pytest.approx(rest_penetration(5.0 / 4 * (1 / FINGER_KG + 1 / BOX_KG), 0.0), rel=0.01, abs=0)
    for ratio in (0.25, 0.5):
        assert squeeze(ratio * TAU)[0] == pytest.approx(base, rel=0.05, abs=0)


@pytest.mark.parametrize(("ratio", "grown"), [(1.0, 3.9), (2.0, 12.6)])
def test_grasp_squeeze_deepens_past_half_tau(ratio: float, grown: float) -> None:
    """dt = tau: x3.9 as deep (130.6 um); dt = 2 tau: x12.6 (420.1 um): the formula with the pad's row force F / 4
    and A = 1 / m_finger + 1 / m_box, to 1 %."""
    travel, force = squeeze(ratio * TAU)
    row_acc = force / 4 * (1 / FINGER_KG + 1 / BOX_KG)
    assert travel == pytest.approx(rest_penetration(row_acc, ratio * TAU), rel=0.01, abs=0)
    assert travel / squeeze(0.125 * TAU)[0] == pytest.approx(grown, rel=0.05)


# --- law E: joint damping dilutes a contact ----------------------------------------------------------------------

ARM_REACH, ARM_RADIUS, ARM_TOUCH, ARM_INERTIA = 0.3, 0.02, 0.3, 0.0302
FIXED_DT = 1.25e-3


def driven_impact(b: float, dt: float, torque: float | None = None) -> float:
    """Peak depth [m] of a 1 kg hinge arm's tip in a thick block it is driven onto at 0.9 m/s (joint damping ``b``,
    torque 3 b unless given, so the joint turns at 3 rad/s; no gravity), implicitfast."""
    top = -ARM_REACH * math.sin(ARM_TOUCH) - ARM_RADIUS
    xml = f"""<mujoco><compiler angle="radian"/><option timestep="{dt!r}" integrator="implicitfast" gravity="0 0 0"/>
    <worldbody><geom type="box" size="0.1 0.1 0.05" pos="{ARM_REACH * math.cos(ARM_TOUCH)} 0 {top - 0.05}"/>
    <body><joint name="j" type="hinge" axis="0 1 0" damping="{b}"/>
    <geom type="capsule" fromto="0 0 0 {ARM_REACH} 0 0" size="{ARM_RADIUS}" mass="1"/></body></worldbody>
    <actuator><motor joint="j"/></actuator></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.ctrl[0] = 3.0 * b if torque is None else torque
    deepest = 0.0
    for _ in range(int(round(0.6 / dt))):
        mujoco.mj_step(m, d)
        deepest = max(deepest, top + ARM_REACH * math.sin(d.qpos[0]) + ARM_RADIUS)
    return deepest


def dilution(b: float, dt: float) -> float:
    """The damping cap's ratio F = (1 + dt b / I) / (1 + dt0 b / I)."""
    return (1.0 + dt * b / ARM_INERTIA) / (1.0 + FIXED_DT * b / ARM_INERTIA)


@pytest.mark.parametrize("b", [1.5, 3.0, 6.0, 12.0, 24.0, 48.0])
@pytest.mark.parametrize("dt", [2.5e-3, 5.0e-3])
def test_joint_damping_deepens_a_driven_impact_by_the_dilution_ratio(b: float, dt: float) -> None:
    """Peak depth at ``dt`` over the fixed step's, against F (1.06 to 3.0 here). Measured: within -6 % / +1 % of F up
    to F = 1.35 (so 1.15-1.25x deeper at the cap's 1.25), then under it, as F^0.6-0.8 (x2.11 at F = 3.0); never more
    than 8 % over F."""
    F, grown = dilution(b, dt), driven_impact(b, dt) / driven_impact(b, FIXED_DT)
    assert math.sqrt(F) <= grown <= 1.08 * F, (F, grown)
    if F <= 1.35:
        assert grown == pytest.approx(F, rel=0.08), (F, grown)


def test_undamped_impact_does_not_deepen_with_the_step() -> None:
    """Law E's control: with no joint damping (0.25 N m drive, 0.65 m/s) the step alone changes the depth by
    -5 % and -16 % (3.95 mm at 1.25 ms, 3.74 at 2.5 ms, 3.33 at 5 ms)."""
    def undamped(dt: float) -> float:
        return driven_impact(0.0, dt, torque=0.25)

    assert 0.8 <= undamped(5.0e-3) / undamped(FIXED_DT) <= 1.0
    assert 0.9 <= undamped(2.5e-3) / undamped(FIXED_DT) <= 1.0
