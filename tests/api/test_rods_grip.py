# SPDX-License-Identifier: Apache-2.0
"""SCR-WP9-ROD R4: a robot's fingers twist an elastic rod (ROD_VBD) through contact, on the CPU.

A clamped USB cable (0.4 m, r 3 mm, 20 segments) has its free end held between two 0.2 kg pads on a MuJoCo wrist that
rolls about the cable's axis: a position servo ramped to 0.3 rad over 0.5 s, then held. The pads are boxes on the
wrist, their faces 2.1 um into the cable (4.2 N each at the cards' 2e6 N/m contact stiffness): a closed gripper.
- The free span twists uniformly: every joint within 0.5 % of the mean (measured 0.04 %).
- The gripped segment turns with the wrist: within 2 % when the ramp ends (measured 1.0000; rotation about the axis
  from the quaternions).
- While held it stays with the wrist within 1 % (measured +0.0000). A rod entry's proxy contacts match "sticky"
  (alloy/integration/stepper.py), with the replay keeping the fresh normal (vendored contact-match patch): friction
  holds the cable. Before F3 (SCR-WP9-ROD, 2026-11-08) Newton VBD's friction dragged it on: +1.9 % after this hold.
- The wrist's servo carries the cable's torque (G J / h) x the joint twist within 3 % (measured 0.976), h = L / (N - 1)
  the emitter's joint spacing (the model's own twist stiffness, checked): MuJoCo's balance on the roll joint at rest,
  qfrc_bias - actuator_force - qfrc_passive, averaged over the hold. Before F3 it read 1.038, the friction pushing the
  cable on; before 2026-11-08 this law used h = L / N, 5.3 % stiff.
- `sim.check()` is ok.
The fast tests pin the exchange clock that sets the cost (slow laws need a fast clock check): 0.5 sqrt(m / k) with
the lightest robot body near the cable (0.2 kg sliding pads) is 158 us, so j = 6 (156 us); and that only a rod entry's
contacts match sticky (a cloth beside the same wrist keeps unmatched contacts).
Measured rows, the sliding-pad creep and the probes: docs/notes/wp9-results-rod.md (R4).
"""

from __future__ import annotations

import gc
import math

import numpy as np
import pytest

import alloy
from alloy._vendor import newton
from alloy.api import _rods
from alloy.solvers.rod.topology import ROD

L, R, Z, N, PHI = 0.4, 0.003, 0.3, 20, 0.3
KE, SQUEEZE = 2.0e6, 4.2                     # N/m (card default), N per pad


def _wrist(welded: bool) -> str:
    if welded:
        y = R + 0.004 - SQUEEZE / KE
        pads = (f'<geom type="box" pos="0 {y:.9f} 0" size="0.01 0.004 0.01" mass="0.2" friction="1.0"/>'
                f'<geom type="box" pos="0 {-y:.9f} 0" size="0.01 0.004 0.01" mass="0.2" friction="1.0"/>')
        act = ""
    else:
        pads = ''.join(f'<body name="{n}" pos="0 {s * 0.013} 0"><joint name="{n[0]}f" type="slide" axis="0 {-s} 0" '
                       f'range="0 0.02" damping="1"/><geom type="box" size="0.01 0.004 0.01" mass="0.2" '
                       f'friction="1.0"/></body>' for n, s in (("left", 1), ("right", -1)))
        act = '<position joint="lf" kp="300" kv="5"/><position joint="rf" kp="300" kv="5"/>'
    return (f'<mujoco><compiler angle="radian"/><worldbody><body name="wrist" pos="{L - 0.012} 0 {Z}">'
            f'<joint name="roll" type="hinge" axis="1 0 0" damping="0.02"/>'
            f'<inertial pos="0 0 0" mass="0.3" diaginertia="2e-4 2e-4 2e-4"/>{pads}</body></worldbody>'
            f'<actuator><position joint="roll" kp="20" kv="0.5"/>{act}</actuator></mujoco>')


def _scene(welded: bool) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01)
    sc.add_rod("cable", length=L, radius=R, pos=(0.0, 0.0, Z), material="cable_usb_composite", segments=N,
               clamp="start")
    sc.add_robot("hand", mjcf=_wrist(welded))
    return sc


def _about_x(q: np.ndarray, q0: np.ndarray) -> float:
    """Rotation about world x of xyzw quaternion q relative to q0 (swing-twist)."""
    w = q[3] * q0[3] + q[0] * q0[0] + q[1] * q0[1] + q[2] * q0[2]
    x = -q[3] * q0[0] + q[0] * q0[3] - q[1] * q0[2] + q[2] * q0[1]
    return 2.0 * math.atan2(x, w)


def _wrist_torque(sim) -> float:
    """The cable's torque on the roll joint at rest: MuJoCo's balance, gravity and the servo taken out."""
    mj = sim.engine.solvers[0]
    if hasattr(mj, "datas"):                                   # the CPU route: MuJoCo C, one MjData per world
        d = mj.datas[0]
        return float(d.qfrc_bias[0] - d.actuator_force[0] - d.qfrc_passive[0])
    d = mj.mjw_data                                           # MuJoCo Warp (GPU, or ALLOY_CPU_ROUTE=warp)
    return float(d.qfrc_bias.numpy()[0, 0] - d.actuator_force.numpy()[0, 0] - d.qfrc_passive.numpy()[0, 0])


def test_the_grip_clock_is_set_by_the_lightest_pad():
    sim = alloy.build(_scene(welded=False), device="cpu")
    assert sim.routes()["cable"]["route"] == "rod_vbd" and sim.routes()["hand"]["solver"] == "mujoco"
    assert sim.engine.clock.j == 6
    sim.close()


def _new_pipelines(build) -> list[str]:
    """The contact_matching mode of every CollisionPipeline that `build()` creates."""
    gc.collect()
    before = {id(o) for o in gc.get_objects() if isinstance(o, newton.CollisionPipeline)}
    sim = build()
    modes = [o.contact_matching for o in gc.get_objects()
             if isinstance(o, newton.CollisionPipeline) and id(o) not in before]
    sim.close()
    return modes


def test_only_a_rod_entry_matches_its_contacts_sticky():
    rod = _new_pipelines(lambda: alloy.build(_scene(welded=False), device="cpu"))
    assert rod.count("sticky") == 1, rod

    def cloth():
        sc = alloy.Scene(dt=0.01)
        sc.add_cloth("sheet", size=(0.08, 0.08), pos=(L - 0.012, 0.0, Z - 0.05), resolution=4, mass=0.01,
                     self_contact=False)
        sc.add_robot("hand", mjcf=_wrist(welded=False))
        return alloy.build(sc, device="cpu")

    others = _new_pipelines(cloth)
    assert others and "sticky" not in others, others


@pytest.fixture(scope="module")
def grip() -> dict[str, float]:
    """One run of the closed gripper (module scope: one run however many checks read it)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ALLOY_CPU_ROUTE", "mjc")              # the CPU route as shipped, MuJoCo C (the suite forces Warp)
        return _closed_gripper()


def _closed_gripper() -> dict[str, float]:
    sim = alloy.build(_scene(welded=True), device="cpu")
    assert sim.engine.clock.j == 6
    seg = f"cable/cable_edge_body_{N - 1}"
    q0 = sim.state([seg], world=0)[f"{seg}.quat_xyzw"].copy()
    d = sim.built.report.decisions["cable"].numbers
    k_twist = d["GJ"] / (d["rod_length"] / (N - 1))                    # the emitter's joint spacing
    rod = [s for s in (x[0] if isinstance(x, tuple) else x for x in _rods._rod_solvers(sim.engine))
           if type(s).__name__.endswith("Rod")][0]
    jt, qs = rod.model.joint_type.numpy(), rod.model.joint_qd_start.numpy()
    twist_ke = rod.model.joint_target_ke.numpy()[[qs[j] + 5 for j in range(len(jt)) if int(jt[j]) == ROD]]
    assert twist_ke == pytest.approx(k_twist, rel=1e-5, abs=0)        # the model's own twist stiffness

    def row() -> tuple[float, ...]:
        st = sim.state(["cable", "hand/roll", seg], world=0)
        free = st["cable.twist"][:-2]                        # the last joints lie under the pads
        return (float(st["hand/roll.q"]), _about_x(st[f"{seg}.quat_xyzw"], q0), _wrist_torque(sim),
                k_twist * float(free.mean()), float(np.abs(free / free.mean() - 1.0).max()))

    sim.run(0.5, targets={"roll": (0.0, PHI)})
    wrist0, seg0 = row()[:2]
    sim.set_targets({"roll": PHI})
    rows = []
    for _ in range(14):
        sim.step(5)
        rows.append(row())
    wrist, seg_rot, t_mj, t_rod, spread = (np.array(c) for c in zip(*rows))
    ratio = -t_mj[2:] / t_rod[2:]
    out = {"wrist0": wrist0, "seg0": seg0, "spread": float(spread.max()), "run_on": seg_rot[-1] / wrist[-1] - 1.0,
           "torque": -t_mj[2:].mean() / t_rod[2:].mean(), "ok": bool(sim.check()["ok"])}
    print(f"ramp end {seg0 / wrist0:.4f}; spread {out['spread']:.2e}; run-on {out['run_on']:+.4f}; "
          f"torque {out['torque']:.4f} ({ratio.min():.3f}-{ratio.max():.3f})")
    sim.close()
    return out


@pytest.mark.slow
@pytest.mark.law
@pytest.mark.timeout(1800)
def test_a_closed_gripper_twists_a_clamped_cable(grip):
    assert grip["seg0"] == pytest.approx(grip["wrist0"], rel=0.02, abs=0)
    assert grip["spread"] < 0.005
    assert abs(grip["run_on"]) < 0.01
    assert grip["torque"] == pytest.approx(1.0, rel=0.03, abs=0)
    assert grip["ok"]

