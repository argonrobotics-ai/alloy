# SPDX-License-Identifier: Apache-2.0
"""The friction pull cap (alloy.contact.mjw_friction_cap, RIGID-DEFAULT (b)): its switch, its install, and its law on
the pushed cube of tests/scene/test_integrator_route.py (CPU, MuJoCo Warp route, implicitfast at auto's 5 ms).

Rows (main ca9e66eec, CPU warp; pitch peak to peak over t >= 0.1 s of a 0.3 m cube pushed at 1.5 mu m g, mu 0.5):
0.4 s push: rk4 0.200 deg, implicitfast 0.528, implicitfast + cap 20 0.326; 2 s push beside 8 resting cubes: rk4 3.86,
implicitfast 19.15, implicitfast + cap 20 0.412 (not growing: 0.316 over the last 0.5 s). Coast at 2.5 m/s after a 1 s
push, deceleration over mu g: implicitfast 1.023, + cap 20 1.000, + cap 10 0.996."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.contact import mjw_friction_cap as FC
from alloy.scene import entries as EN
from alloy.scene import integrator_route as IR

MU, G = 0.5, 9.81


def test_cap_value_switch(monkeypatch):
    assert FC.cap_value(None) is None and FC.cap_value(0) is None
    assert FC.cap_value(20) == 20.0
    with pytest.raises(ValueError, match="m/s2"):
        FC.cap_value(-1.0)
    monkeypatch.setenv(FC.ENV, "0")
    assert FC.cap_value(20) is None


class _Solver:
    use_mujoco_cpu = False

    def _mujoco_warp_step(self):
        from alloy._vendor.mujoco_warp._src import constraint

        return constraint.make_constraint


def test_install_swaps_make_constraint_only_inside_the_step():
    from alloy._vendor.mujoco_warp._src import constraint

    stock = constraint.make_constraint
    s = _Solver()
    assert FC.install_friction_cap(s, 20.0) and FC.install_friction_cap(s, 20.0)      # a second call: same cap
    inside = s._mujoco_warp_step()
    assert inside._alloy_friction_cap == 20.0 and constraint.make_constraint is stock
    with pytest.raises(RuntimeError, match="already capped"):
        FC.install_friction_cap(s, 10.0)
    assert not FC.install_friction_cap(_Solver(), None)
    c = _Solver()
    c.use_mujoco_cpu = True
    assert not FC.install_friction_cap(c, 20.0)


def _scene(size: float = 0.3):
    import alloy

    scene = alloy.Scene(dt=0.01)
    scene.add_ground(friction=MU)
    scene.add_box("blk", size=(size,) * 3, pos=(0.0, 0.0, 0.5 * size + 1e-4), density=1000.0, friction=MU)
    return scene


def _build(monkeypatch, rk4: bool, scene=None):
    """``alloy.build`` of the law's cube with the integrator rule forced: rk4, or R1's path (implicitfast on rk4's
    gains)."""
    import alloy

    choice = IR.IntegratorChoice(True, "rk4 (test)", free=True) if rk4 else \
        IR.IntegratorChoice(False, "implicitfast (test)", r1=True, free=True)
    with monkeypatch.context() as m:
        m.setattr(IR, "_clutter_or_rk4", lambda *_: choice)
        return alloy.build(scene or _scene(), device="cpu")


def _cap_of(sim) -> float | None:
    stepper = sim.engine.stepper
    for e in getattr(stepper, "entries", None) or [stepper]:
        solver = getattr(e, "solver", e)
        if hasattr(solver, "_mujoco_warp_step"):
            return getattr(solver, "_alloy_friction_cap", None)
    return None


def test_fast_mode_and_r1_entries_take_the_cap(monkeypatch):
    over = EN.mujoco_overrides({EN.RIGID_FAST_EFFECT}, False, None)
    assert over[FC.CAP_KEY][0] == FC.CAP_DEFAULT == 20.0 and "MuJoCo Warp" in over[FC.CAP_KEY][1]
    assert FC.CAP_KEY not in EN.mujoco_overrides(set(), False, None)
    assert FC.CAP_KEY not in EN.mujoco_overrides({IR.R1_EFFECT}, False, None)   # a CHAIN's effect too: not by effect
    r1, rk4 = _build(monkeypatch, rk4=False), _build(monkeypatch, rk4=True)
    assert (_cap_of(r1), _cap_of(rk4)) == (20.0, None)
    r1.close()
    rk4.close()
    monkeypatch.setenv(FC.ENV, "0")
    off = _build(monkeypatch, rk4=False)
    assert _cap_of(off) is None
    off.close()


def _pushed(monkeypatch, cap: bool, rk4: bool, push_s: float = 0.4, coast_s: float = 0.0):
    """(pitch p-p over t >= 0.1 s of the push, coast deceleration / mu g or None) of the law's 0.3 m cube; ``cap``
    False turns the cap off (``ALLOY_FRICTION_CAP=0``)."""
    with monkeypatch.context() as m:
        if not cap:
            m.setenv(FC.ENV, "0")
        sim = _build(monkeypatch, rk4)
    assert sim.built.report.rigid_step.dt == pytest.approx(5e-3)
    assert _cap_of(sim) == (FC.CAP_DEFAULT if cap and not rk4 else None)
    sim.run(0.5)
    sim.apply_wrench("blk", force=(1.5 * MU * 27.0 * G, 0.0, 0.0))
    tr = sim.run(push_s, record=["blk"]).world(0)
    t, (w, x, y, z) = tr["t"] - tr["t"][0], tr["blk.quat_wxyz"].T
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    coast = None
    if coast_s:
        sim.apply_wrench("blk", force=(0.0, 0.0, 0.0))
        tc = sim.run(coast_s, record=["blk"]).world(0)
        tt, px = tc["t"] - tc["t"][0], tc["blk.pos"][:, 0]
        k = tt >= 0.02
        coast = float(-2.0 * np.polyfit(tt[k], px[k], 2)[0] / (MU * G))
    assert sim.check()["ok"], sim.check()["problems"]
    sim.close()
    return float(np.ptp(pitch[t >= 0.1])), coast


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_capped_implicitfast_rocks_no_more_than_todays_rk4(monkeypatch):
    """The rocking law's bound (1.1x rk4 + 0.2 deg) for an R1 entry (implicitfast + cap 20) against rk4 (no cap: an rk4
    entry takes none), and the cap does cut implicitfast's rocking (0.528 -> 0.326 deg)."""
    rk4, _ = _pushed(monkeypatch, True, rk4=True)
    stock, _ = _pushed(monkeypatch, False, rk4=False)
    capped, _ = _pushed(monkeypatch, True, rk4=False)
    assert capped <= 1.1 * rk4 + 0.2 < stock, (capped, rk4, stock)


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_capped_coast_keeps_coulomb(monkeypatch):
    """A cube coasting from 2.5 m/s decelerates at mu g within 1 % under the cap (1.000 at cap 20)."""
    _, coast = _pushed(monkeypatch, True, rk4=False, push_s=1.0, coast_s=0.4)
    assert coast == pytest.approx(1.0, abs=0.01)
