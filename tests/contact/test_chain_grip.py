# SPDX-License-Identifier: Apache-2.0
"""SCR-WP3-CHAINGRIP: a CHAIN beside a moving body in the MuJoCo entry (`alloy.contact.chain_grip`), on the CPU.

- Scope: a 0.4 m USB cable (20 segments, CHAIN) on the ground, alone or beside a static box, is left alone. Beside a
  dynamic box or a robot, its link shapes take solmix 1000 and Newton runs impratio 300 on MuJoCo C, 30 on MuJoCo Warp,
  whose model is marked for the float32-noise pivot floor (SCR-WP3-CHAINGRIP-GPU). The rule is named in the cable's
  route reasons.
- Guard: with the rule switched off, every entry it does not cover builds bit for bit the same (entry kwargs, MuJoCo
  geom contact parameters, options, no pivot floor). A covered entry differs only in the cable's solmix, the rule's
  kwarg, the impratio and (MuJoCo Warp) the pivot floor.
- Grip: WP9's R4 rig on CHAIN, link 0 welded as [internal note] does. Sliding pads
  squeeze 4.2 N; the wrist rolls 0.3 rad over 0.5 s. On MuJoCo C the cable end follows within 0.5 % at the ramp's end
  and after a 0.2 s hold (measured 1.0000 / 0.9999), and the pads sit under 0.2 mm deep (0.103 mm). With the rule off
  it follows about half (0.498 / 0.281), and the pads sink 1.5 mm. On MuJoCo Warp (CPU) the same rig follows within
  1 % at impratio 30 (L4: 0.9976 / 0.9828 after 0.7 s). At 300 float32 Newton went non-finite at 25 ms, or (floored,
  damped) did not converge on a resting cable.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

import alloy
from alloy.contact import chain_grip as CG
from alloy.precision import mjw_chol
from alloy.scene.emitters import chain as CH

L, R, N, Z = 0.4, 0.003, 20, 0.3


def _wrist(pos: tuple[float, float, float]) -> str:
    """WP9's R4 wrist: a roll hinge carrying two 0.2 kg pads on servoed slides (closed by targets lf, rf = 0.02)."""
    pads = "".join(f'<body name="{n}" pos="0 {s * 0.013} 0"><joint name="{n[0]}f" type="slide" axis="0 {-s} 0" '
                   f'range="0 0.02" damping="1"/><geom type="box" size="0.01 0.004 0.01" mass="0.2" '
                   f'friction="1.0"/></body>' for n, s in (("left", 1), ("right", -1)))
    return (f'<mujoco><compiler angle="radian"/><worldbody><body name="wrist" pos="{pos[0]} {pos[1]} {pos[2]}">'
            f'<joint name="roll" type="hinge" axis="1 0 0" damping="0.02"/>'
            f'<inertial pos="0 0 0" mass="0.3" diaginertia="2e-4 2e-4 2e-4"/>{pads}</body></worldbody>'
            f'<actuator><position joint="roll" kp="20" kv="0.5"/><position joint="lf" kp="300" kv="5"/>'
            f'<position joint="rf" kp="300" kv="5"/></actuator></mujoco>')


def _scene(beside: str) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_rod("cable", length=L, radius=R, pos=(0.0, 0.0, R + 5e-4), segments=N, material="cable_usb_composite")
    box = {"size": (0.05, 0.05, 0.05), "pos": (0.2, 0.3, 0.025)}
    if beside == "static":
        sc.add_box("box", **box, static=True)
    elif beside == "box":
        sc.add_box("box", **box, mass=0.2)
    elif beside == "robot":
        sc.add_robot("hand", mjcf=_wrist((0.2, 0.3, 0.1)))
    return sc


def _floor(mj) -> float:
    """The pivot floor a compact-solve copy of the MuJoCo Warp model sees: the mark must ride on the Data too."""
    m = mj.mjw_model
    return mjw_chol.pivot_floor(dataclasses.replace(m, opt=dataclasses.replace(m.opt)), mj.mjw_data)


def _facts(sim) -> dict:
    """The MuJoCo entry's kwargs, its geoms' contact parameters and the options the CPU route steps with."""
    mj = sim.engine.solvers[0]
    m = mj.worlds[0] if getattr(mj, "worlds", None) else mj.mj_model
    kw = next(e for e in sim.built.entries if e.name == "mujoco").solver_kwargs
    cable = [g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith("cable_")]
    return {"kw": dict(kw), "geoms": {k: np.array(getattr(m, k)) for k in ("geom_solref", "geom_solimp", "geom_solmix",
                                                                            "geom_friction", "geom_condim")},
            "opt": {k: float(getattr(m.opt, k)) for k in ("impratio", "solver", "iterations", "cone", "integrator")},
            "cable": cable, "reasons": sim.routes()["cable"]["reasons"],
            "floor": None if getattr(mj, "worlds", None) else _floor(mj)}


def _build(beside: str, rule: bool, monkeypatch) -> dict:
    with monkeypatch.context() as mp:
        if not rule:
            mp.setattr(CG, "gripped", lambda objs: [])
        sim = alloy.build(_scene(beside), device="cpu")
    out = _facts(sim)
    assert sim.routes()["cable"]["route"] == "chain"
    sim.close()
    return out


@pytest.fixture(autouse=True)
def _c_route(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")          # users' CPU route (the suite's default is MuJoCo Warp)


@pytest.mark.parametrize("route", ["mjc", "warp"])
@pytest.mark.parametrize("beside", ["alone", "static"])
def test_a_cable_beside_nothing_that_moves_is_left_alone(beside, route, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    on, off = _build(beside, True, monkeypatch), _build(beside, False, monkeypatch)
    assert on["opt"]["impratio"] == 1.0 and on["floor"] in (None, 0.0)               # CHAIN_NEWTON_HOLD stands
    assert not any("chain grip" in r for r in on["reasons"])
    assert on["kw"] == off["kw"] and on["opt"] == off["opt"]                            # guard: bit for bit
    for k, v in on["geoms"].items():
        assert np.array_equal(v, off["geoms"][k]), k


@pytest.mark.parametrize("route", ["mjc", "warp"])
@pytest.mark.parametrize("beside", ["box", "robot"])
def test_a_cable_beside_a_moving_body_takes_the_rule(beside, route, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    on, off = _build(beside, True, monkeypatch), _build(beside, False, monkeypatch)
    near = "box" if beside == "box" else "hand"
    assert any(f"chain grip beside {near}" in r for r in on["reasons"])
    assert on["kw"].pop(CG.KEY) == {"impratio": CG.IMPRATIO} and on["kw"] == off["kw"]
    want = CG.IMPRATIO if route == "mjc" else CG.WARP_IMPRATIO                           # Newton: 300 / 30
    assert (on["opt"].pop("impratio"), off["opt"].pop("impratio")) == (want, 1.0)
    assert on["opt"] == off["opt"]
    if route == "warp":                                                                 # the pivot floor marks it
        assert (on["floor"], off["floor"]) == (CG.PIVOT_FLOOR, 0.0)
    mix, mix0 = on["geoms"]["geom_solmix"], off["geoms"]["geom_solmix"]
    assert np.all(mix[on["cable"]] == CG.SOLMIX) and np.all(mix0[on["cable"]] == 1.0)
    rest = np.setdiff1d(np.arange(len(mix)), on["cable"])
    assert np.array_equal(mix[rest], mix0[rest])
    for k in ("geom_solref", "geom_solimp", "geom_friction", "geom_condim"):
        assert np.array_equal(on["geoms"][k], off["geoms"][k]), k


def _about_x(q: np.ndarray, q0: np.ndarray) -> float:
    w = q[3] * q0[3] + q[0] * q0[0] + q[1] * q0[1] + q[2] * q0[2]
    x = -q[3] * q0[0] + q[0] * q0[3] - q[1] * q0[2] + q[2] * q0[1]
    return 2.0 * math.atan2(x, w)


def _pad_depth(mj) -> float:
    """The deepest pad-cable penetration of world 0 [m], on MuJoCo C or MuJoCo Warp."""
    if getattr(mj, "worlds", None):
        m, d = mj.worlds[0], mj.datas[0]
        rows = [(float(d.contact[i].dist), int(d.contact[i].geom1), int(d.contact[i].geom2)) for i in range(d.ncon)]
    else:
        m, x = mj.mj_model, mj.mjw_data
        n = int(x.nacon.numpy()[0])
        dist, geom = x.contact.dist.numpy()[:n], x.contact.geom.numpy()[:n]
        rows = [(float(r), int(g[0]), int(g[1])) for r, g in zip(dist, geom)]
    pads = {g for g in range(m.ngeom) if "wrist" in m.body(m.geom_bodyid[g]).name and m.geom_type[g] == 6}
    return max((-dist for dist, a, b in rows if {a, b} & pads), default=0.0)


def _grip(rule: bool, monkeypatch) -> tuple[float, float, float]:
    """WP9's F2 rig: (follow at the ramp's end, follow after a 0.2 s hold, deepest pad-cable penetration in m)."""
    build0 = CH.build_chain
    with monkeypatch.context() as mp:
        mp.setattr(CH, "build_chain", lambda b, rod, pts, opts, label=None: build0(
            b, rod, pts, dataclasses.replace(opts, clamp=True), label=label))   # weld link 0 (the probe's clamp)
        if not rule:
            mp.setattr(CG, "gripped", lambda objs: [])
        sc = alloy.Scene(dt=0.01)
        sc.add_rod("cable", length=L, radius=R, pos=(0.0, 0.0, Z), material="cable_usb_composite", segments=N)
        sc.add_robot("hand", mjcf=_wrist((L - 0.012, 0.0, Z)))
        sim = alloy.build(sc, device="cpu")
    seg = [n for n in sim.names()["bodies"] if n.startswith("cable/")][-1]
    q0 = sim.state([seg], world=0)[f"{seg}.quat_xyzw"].copy()

    def follow() -> float:
        st = sim.state(["hand/roll", seg], world=0)
        return _about_x(st[f"{seg}.quat_xyzw"], q0) / float(st["hand/roll.q"])

    sim.set_targets({"lf": 0.02, "rf": 0.02})
    sim.step(20)
    sim.run(0.5, targets={"roll": (0.0, 0.3)})
    ramp = follow()
    sim.set_targets({"roll": 0.3})
    sim.step(20)
    out = (ramp, follow(), _pad_depth(sim.engine.solvers[0]))
    assert sim.check()["ok"]
    sim.close()
    return out


@pytest.mark.timeout(600)
def test_a_gripped_chain_follows_the_wrist(monkeypatch):
    ramp, held, deep = _grip(True, monkeypatch)
    print(f"chain grip: ramp end {ramp:.4f}, after 0.2 s {held:.4f}, pad depth {1e3 * deep:.3f} mm")
    assert ramp == pytest.approx(1.0, abs=0.005) and held == pytest.approx(1.0, abs=0.005)
    assert deep < 2e-4
    ramp0, held0, deep0 = _grip(False, monkeypatch)                        # the slip the rule fixes
    print(f"rule off: ramp end {ramp0:.4f}, after 0.2 s {held0:.4f}, pad depth {1e3 * deep0:.3f} mm")
    assert ramp0 < 0.7 and deep0 > 1e-3


@pytest.mark.timeout(900)
def test_a_gripped_chain_follows_the_wrist_on_mujoco_warp(monkeypatch):
    """The GPU's code path on the CPU (SCR-WP3-CHAINGRIP-GPU): Newton at impratio 30, the pivot floor as safety net."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    ramp, held, deep = _grip(True, monkeypatch)
    print(f"chain grip (MuJoCo Warp): ramp end {ramp:.4f}, after 0.2 s {held:.4f}, pad depth {1e3 * deep:.3f} mm")
    assert ramp == pytest.approx(1.0, abs=0.01) and held == pytest.approx(1.0, abs=0.01)
    assert deep < 2e-4


def test_a_cg_entry_keeps_its_own_impratio():
    """The rule's impratio is for Newton entries only: an entry that resolves to CG (a chain attachment's,
    ``mujoco:solver=cg``) keeps its own, as the 17:41Z ruling refused CG at 300 on MuJoCo Warp (WP9's review)."""
    from types import SimpleNamespace as NS

    from alloy.route.types import Route
    from alloy.scene.entries import entry_overrides

    chain = NS(name="cable", obj=NS(card=NS(motion="dynamic")),
               decision=NS(route=Route.CHAIN, entry="mujoco", scene_effects=(), numbers={}, reasons=()))
    hand = NS(name="hand", obj=None, decision=NS(route=Route.RIGID, entry="mujoco", scene_effects=(), numbers={}))
    mjcf = NS(values={"impratio": 1.0})
    assert CG.KEY in entry_overrides([chain, hand], set(), False, mjcf, True)
    assert CG.KEY not in entry_overrides([chain, hand], {"mujoco:solver=cg"}, False, mjcf, False)
