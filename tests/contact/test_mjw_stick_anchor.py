# SPDX-License-Identifier: Apache-2.0
"""The stick anchor (alloy.contact.mjw_stick_anchor; known gap rod-chain-grip-slips): its switch, its install, and its
laws on the CPU's MuJoCo Warp route.

Laws (pre-registered 2026-11-09, scr/WP3.md "GAP WAVE ROUND 2"):
- hold: a cable on the chain route, twisted 0.3 rad by a gripper and held (R4's rig on a clamped chain,
  tests/api/test_rods_chain_clamp.py), stays with the wrist within 0.1 % over the hold, with sliding pads and with pads
  welded to the wrist. Without the anchor: 0.9976 -> 0.9465 over 2 s (sliding pads), 0.978 -> -10 % in 0.7 s (welded).
- Coulomb, on a cube with the anchor on its geom: pushed at 0.3 of its friction limit it creeps under a tenth of what
  it creeps without the anchor; pushed past the limit it slides and coasts as the same cube without the anchor does
  (travel within 0.5 %, deceleration within 0.01 mu g, and mu g within 5 %). The anchor never holds past the cone and
  adds no pull to a sliding contact. With the anchor on every moving geom the Coulomb
  laws read as without it (CPU warp, quick: A5.onset 0.0 deg, creep 5.2e-6; A5.slide 0.45 % / 0.33 %; A5.isotropy
  0.093 %; A5.speed force / mu N 1.000000 at 1 N and 20 N).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
import warp as wp

from alloy._vendor.mujoco_warp._src import constraint
from alloy._vendor.mujoco_warp._src.types import IntegratorType
from alloy.contact import chain_grip
from alloy.contact import mjw_stick_anchor as SA
from alloy.scene import integrator_route as IR

MU, G, MASS = 0.5, 9.81, 27.0
_spec = importlib.util.spec_from_file_location("grip", Path(__file__).parents[1] / "api" / "test_rods_grip.py")
RIG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RIG)


def test_cap_value_switch(monkeypatch):
    assert SA.cap_value(None) is None and SA.cap_value(0) is None and SA.cap_value(20) == 20.0
    with pytest.raises(ValueError, match="m/s2"):
        SA.cap_value(-1.0)
    monkeypatch.setenv(SA.ENV, "0")
    assert SA.cap_value(20) is None


def _solver(integrator: IntegratorType = IntegratorType.DISCRETE):
    s = NS(use_mujoco_cpu=False, mjw_model=NS(opt=NS(integrator=int(integrator)), ngeom=5),
           mjw_data=NS(nworld=3, nacon=wp.zeros(1, dtype=int, device="cpu")))
    s._mujoco_warp_step = lambda: constraint.make_constraint
    return s


def test_install_swaps_make_constraint_only_inside_the_step(monkeypatch):
    stock, s = constraint.make_constraint, _solver()
    assert SA.install_stick_anchor(s, [1, 3, 3])
    st = s._alloy_stick_anchor
    assert st.partner.shape == (3, 2, SA.SLOTS) and st.index.numpy().tolist() == [-1, 0, -1, 1, -1]
    assert s._mujoco_warp_step()._alloy_stick_anchor == SA.CAP_DEFAULT and constraint.make_constraint is stock
    assert set(SA.state_arrays(s)) == {"stick_anchor.partner", "stick_anchor.slip", "stick_anchor.sliding"}
    with pytest.raises(RuntimeError, match="already anchored"):
        SA.install_stick_anchor(s, [1])
    assert not SA.install_stick_anchor(_solver(), [])                          # nothing marked
    assert not SA.install_stick_anchor(_solver(IntegratorType.RK4), [1])       # four constraint builds a step
    cpu = _solver()
    cpu.use_mujoco_cpu = True
    assert not SA.install_stick_anchor(cpu, [1]) and SA.state_arrays(cpu) == {}
    monkeypatch.setenv(SA.ENV, "0")
    assert not SA.install_stick_anchor(_solver(), [1])


def _one_contact(addresses: tuple[int, int, int]):
    """A 1-world stand-in for MuJoCo Warp's model and data with one elliptic contact between the marked geom 1 and
    geom 0 (normal x, tangents y and z), its three rows at ``addresses``, and the pair seated with 0.1 mm of slip."""
    arr = lambda v, dtype: wp.array(v, dtype=dtype, device="cpu")                                    # noqa: E731
    elliptic = int(SA._types.ConstraintType.CONTACT_ELLIPTIC)
    con = NS(dim=arr([3], int), worldid=arr([0], int), geom=arr([(0, 1)], wp.vec2i),
             efc_address=arr([list(addresses)], int), frame=arr([np.eye(3)], wp.mat33), dist=arr([0.0], float),
             includemargin=arr([0.0], float), solref=arr([(0.02, 1.0)], wp.vec2),
             solreffriction=arr([(0.0, 0.0)], wp.vec2), solimp=arr([(0.9, 0.95, 0.001, 0.5, 2.0)], SA._vec5),
             friction=arr([(1.0, 1.0, 0.0, 0.0, 0.0)], SA._vec5))
    efc = NS(type=arr(np.full((1, 4), elliptic), int), vel=arr(np.zeros((1, 4)), float),
             D=arr(np.ones((1, 4)), float), aref=arr(np.array([[10.0, 0.0, 0.0, 7.0]]), float))
    m = NS(opt=NS(timestep=arr([1.25e-3], float), disableflags=0, integrator=int(IntegratorType.IMPLICITFAST)))
    d = NS(nacon=arr([1], int), naconmax=1, contact=con, efc=efc)
    st = SA.make_state([1], 2, 1, SA.CAP_DEFAULT, "cpu")
    st.partner.assign(np.array([[[0, -1, -1, -1]]], np.int32))
    slip = np.zeros((1, 1, SA.SLOTS, 3), np.float32)
    slip[0, 0, 0] = (0.0, 1.0e-4, 0.0)
    st.slip.assign(slip)
    return st, m, d


def test_a_contact_whose_sliding_rows_were_dropped_is_left_alone():
    """MuJoCo Warp drops the rows past njmax one by one: a contact can keep its normal row and have -1 as the address
    of a sliding row. The anchor then leaves that contact alone (it indexed row -1 before 2026-11-10)."""
    st, m, d = _one_contact((0, 1, 2))
    SA.launch_anchor(st, m, d)
    pulled = d.efc.aref.numpy()[0]
    assert pulled[0] == 10.0 and pulled[3] == 7.0 and pulled[2] == 0.0          # the normal row and a stranger's row
    assert -1.0 < pulled[1] < -0.1, pulled                 # -k d s on the y row: about 2.6e3 /s2 x 0.1 mm
    for dropped in ((0, -1, -1), (0, 1, -1), (0, -1, 2)):
        st, m, d = _one_contact(dropped)
        SA.launch_anchor(st, m, d)
        assert d.efc.aref.numpy()[0].tolist() == [10.0, 0.0, 0.0, 7.0], dropped


def test_the_fixed_point_sums_keep_large_and_small_terms():
    """The pairs' sums are integers (rule D3: no float atomics); their fixed point must hold a creeping velocity and a
    light body's direct stiffness alike."""
    @wp.kernel
    def roundtrip(x: wp.array[float], scale: float, limit: float, out: wp.array[float]):
        i = wp.tid()
        out[i] = SA._real(SA._fixed(x[i], scale, limit) + SA._fixed(x[i], scale, limit), scale)

    for values, scale, limit in (([1.0e-11, -3.0e-7, 2.5], SA._V_SCALE, SA._V_MAX),
                                 ([2.0e9, 6.2e7, 1.0e3], SA._K_SCALE, SA._K_MAX),
                                 ([1.0e-3, 0.25, 4.0e5], SA._C_SCALE, SA._C_MAX)):
        x, out = wp.array(values, dtype=float, device="cpu"), wp.zeros(3, dtype=float, device="cpu")
        wp.launch(roundtrip, dim=3, inputs=[x, scale, limit], outputs=[out], device="cpu")
        assert out.numpy() == pytest.approx(2.0 * x.numpy(), rel=1e-4), (values, out.numpy())


def test_the_chain_grip_rule_marks_its_links():
    s = _solver()
    s.mj_model = NS(geom_solmix=np.array([1.0, chain_grip.SOLMIX, 1.0, chain_grip.SOLMIX, 1e-3]))
    assert not chain_grip.install_anchor(s, None) and getattr(s, "_alloy_stick_anchor", None) is None
    assert chain_grip.install_anchor(s, {"impratio": 30.0})
    assert s._alloy_stick_anchor.index.numpy().tolist() == [-1, 0, -1, 1, -1]


ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""


@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_a_scene_without_chain_grip_links_is_bit_for_bit_unchanged(monkeypatch):
    """The anchor is installed for the chain-grip rule's links only. An arm pushing a box down onto a table (no cable
    on the chain route) gets no anchor and no wrapped constraint build, and steps the same bits with the switch off."""
    import alloy

    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    stock, runs = constraint.make_constraint, []
    for off in (False, True):
        if off:
            monkeypatch.setenv(SA.ENV, "0")
        scene = alloy.Scene(dt=0.01)
        scene.add_ground()
        scene.add_box("table", size=(0.3, 0.3, 0.2), pos=(0.25, 0.0, 0.1), static=True)
        scene.add_box("box", size=(0.06, 0.06, 0.06), pos=(0.25, 0.0, 0.23), mass=0.2)
        scene.add_robot("arm", mjcf=ARM, pos=(0.0, 0.0, 0.25))
        sim = alloy.build(scene, device="cpu")
        solver = _mujoco(sim)
        assert getattr(solver, "_alloy_stick_anchor", None) is None and SA.state_arrays(solver) == {}
        assert not any("chain grip" in r for v in sim.routes().values() for r in v["reasons"])
        tr = sim.run(0.4, record=["box", "arm/shoulder"], targets={"shoulder": (0.0, 0.3)}).world(0)
        runs.append({k: np.asarray(v).tobytes() for k, v in tr.items() if k != "t"})
        assert constraint.make_constraint is stock
        sim.close()
    assert runs[0] == runs[1]


@pytest.mark.parametrize("task", ["task_t12", "task_t09"])
def test_the_pinned_tasks_have_no_chain_grip_links(task):
    """T12 / SB-7 S1 and T09 are untouched, read from their routes: no object is under the chain-grip rule, so the
    build marks no geom and installs no anchor (``chain_grip.install_anchor`` needs a marked geom). T09's USB cable
    is on the chain route, but welded to its plug (``chain_grip.attached``): not gripped, not marked."""
    import alloy

    routes = alloy.estimate(alloy.Scene.load(task), worlds=1, seconds=0.1, device="cpu")["routes"]
    assert routes and not any("chain grip" in r for v in routes.values() for r in v["reasons"])
    assert task != "task_t09" or routes["usb_cable_1m"]["route"] == "chain"


def _mujoco(sim):
    stepper = sim.engine.stepper
    for e in getattr(stepper, "entries", None) or [stepper]:
        solver = getattr(e, "solver", e)
        if hasattr(solver, "_mujoco_warp_step"):
            return solver
    raise AssertionError("no MuJoCo Warp solver")


def _cube(monkeypatch, anchored: bool):
    """A 0.3 m cube on the ground on implicitfast (the anchor refuses rk4), the anchor on its geom when asked."""
    import alloy

    scene = alloy.Scene(dt=0.01)
    scene.add_ground(friction=MU)
    scene.add_box("blk", size=(0.3,) * 3, pos=(0.0, 0.0, 0.15 + 1e-4), density=1000.0, friction=MU)
    with monkeypatch.context() as m:
        m.setattr(IR, "_clutter_or_rk4", lambda *_: IR.IntegratorChoice(False, "implicitfast (test)"))
        sim = alloy.build(scene, device="cpu")
    solver = _mujoco(sim)
    if anchored:
        moving = np.flatnonzero(np.asarray(solver.mj_model.geom_bodyid) != 0)
        assert SA.install_stick_anchor(solver, moving)
    sim.run(0.5)
    return sim


def _travel(sim, push: float, seconds: float) -> float:
    x0 = float(sim.state(["blk"], world=0)["blk.pos"][0])
    sim.apply_wrench("blk", force=(push * MU * MASS * G, 0.0, 0.0))
    sim.run(seconds)
    return float(sim.state(["blk"], world=0)["blk.pos"][0]) - x0


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
def test_an_anchored_contact_holds_below_the_cone_and_slides_at_coulomb(monkeypatch):
    free, held = _cube(monkeypatch, False), _cube(monkeypatch, True)
    for sim in (free, held):                 # the push's first second: the contact takes up its elastic offset
        _travel(sim, 0.3, 1.0)
    crept, stayed = _travel(free, 0.3, 2.0), _travel(held, 0.3, 2.0)
    print(f"pushed at 0.3 of the limit, travel over 2 s: {1e6 * crept:.3f} um without the anchor, "
          f"{1e6 * stayed:.4f} um with it")
    assert abs(stayed) <= 0.1 * abs(crept), (crept, stayed)
    rows = []
    for sim in (free, held):                 # past the cone: 0.3 mu g for 0.6 s, then a coast
        slid = _travel(sim, 1.3, 0.6)
        sim.apply_wrench("blk", force=(0.0, 0.0, 0.0))
        tr = sim.run(0.15, record=["blk"]).world(0)
        t, x = tr["t"] - tr["t"][0], tr["blk.pos"][:, 0]
        k = t >= 0.02
        rows.append((slid, float(-2.0 * np.polyfit(t[k], x[k], 2)[0] / (MU * G))))
        assert sim.check()["ok"], sim.check()["problems"]
        sim.close()
    print(f"past the limit, travel in 0.6 s / coast deceleration over mu g: {rows[0][0]:.4f} m / {rows[0][1]:.4f} "
          f"without the anchor, {rows[1][0]:.4f} m / {rows[1][1]:.4f} with it")
    assert rows[1][0] == pytest.approx(rows[0][0], rel=0.005) and rows[1][1] == pytest.approx(rows[0][1], abs=0.01)
    assert rows[1][0] == pytest.approx(0.5 * 0.3 * MU * G * 0.36, rel=0.05)      # 0.2649 m (measured 0.2723, both)
    assert rows[1][1] == pytest.approx(1.0, abs=0.05)                            # measured 0.9604, both


def _wrist(welded: bool) -> str:
    if not welded:
        return RIG._wrist(False)
    y0, y1 = RIG.R + 0.004 - RIG.SQUEEZE / RIG.KE, RIG.R + 0.004 - 1.0e-4      # pads 0.1 mm into the cable
    return RIG._wrist(True).replace(f"{y0:.9f}", f"{y1:.9f}")


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.timeout(1200)
@pytest.mark.parametrize("welded", [False, True], ids=["sliding_pads", "welded_pads"])
def test_a_held_cable_stays_with_its_grip(welded, monkeypatch):
    import alloy

    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    sc = alloy.Scene(dt=0.01)
    sc.add_rod("cable", length=RIG.L, radius=RIG.R, pos=(0.0, 0.0, RIG.Z), material="cable_usb_composite",
               segments=RIG.N, clamp="start", route="chain")
    sc.add_robot("hand", mjcf=_wrist(welded))
    sim = alloy.build(sc, device="cpu")
    assert getattr(_mujoco(sim), "_alloy_stick_anchor", None) is not None       # the chain-grip rule installed it
    seg = [n for n in sim.names()["bodies"] if n.startswith("cable/")][-1]
    q0 = sim.state([seg], world=0)[f"{seg}.quat_xyzw"].copy()

    def follow() -> float:
        st = sim.state(["hand/roll", seg], world=0)
        return RIG._about_x(st[f"{seg}.quat_xyzw"], q0) / float(st["hand/roll.q"])

    def twist_and_hold() -> tuple[float, float]:
        sim.set_targets({"roll": 0.0, **({} if welded else {"lf": 0.02, "rf": 0.02})})
        sim.step(20)
        sim.run(0.5, targets={"roll": (0.0, RIG.PHI)})
        start = follow()
        sim.set_targets({"roll": RIG.PHI})
        sim.step(200)
        return start, follow()

    start, end = twist_and_hold()
    print(f"welded {welded}: end / wrist {start:.4f} at the ramp's end, {end:.4f} after 2 s held")
    assert start == pytest.approx(1.0, abs=0.005)
    assert end == pytest.approx(start, abs=0.001)
    assert sim.check()["ok"], sim.check()["problems"]
    sim.reset()                 # the anchors go back with the worlds: a reset run repeats the first one
    assert twist_and_hold() == pytest.approx((start, end), abs=1e-6)
    _a_restored_period_repeats(sim, follow)
    sim.close()


def _a_restored_period_repeats(sim, follow) -> None:
    """A re-run of a period (a rollback beside cloth or liquid) starts from the period's own anchors: the snapshot the
    simulation rolls back to carries each slot's partner, slip and sliding flag. Ten steps after the wrist starts to
    turn back, restored and stepped again, give the same anchors and the same cable, bit for bit."""
    from alloy.core.snapshot import WorldSnapshot

    engine, anchors = sim.engine, SA.state_arrays(_mujoco(sim))
    extra = dict(engine.state_arrays())
    assert all(any(k.endswith(name) for k in extra) for name in anchors), sorted(extra)
    snap = WorldSnapshot(engine.model, engine.state, extra, extra_world=dict(engine.state_world_maps()) or None)
    snap.save(engine.state)
    at_start = anchors["stick_anchor.slip"].numpy().copy()
    sim.set_targets({"roll": 0.8 * RIG.PHI})

    def ten_steps() -> tuple[float, dict[str, bytes]]:
        sim.step(10)
        return follow(), {k: v.numpy().tobytes() for k, v in anchors.items()}

    first = ten_steps()
    assert not np.array_equal(anchors["stick_anchor.slip"].numpy(), at_start)        # the slip moved in between
    snap.restore(engine.state, None)
    assert np.array_equal(anchors["stick_anchor.slip"].numpy(), at_start)
    assert ten_steps() == first
