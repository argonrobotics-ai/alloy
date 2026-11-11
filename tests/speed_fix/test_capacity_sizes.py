# SPDX-License-Identifier: Apache-2.0
"""``alloy.build(scene, contact_capacity=n)`` sizes MuJoCo Warp's rows and CCD scratch from n (SPEED-FIX B1).

- Rows per world are r x n + R, read from the compiled MjModel: r the rows of one contact (largest condim on the
  model's cone), R the scene's own rows at their maximum. A two-hinge arm (both limited, one with friction loss)
  beside four boxes, at n = 40 over 2 worlds: njmax 3 x 40 + 3 = 123 (was max(6 x 40, the entry's 6 x 64) = 384),
  naconmax = naccdmax = 80 (the CCD scratch follows the pool).
- Without the option nothing changes: the entry's own sizes reach MuJoCo Warp (njmax 384 = 6 x max(64, 8 x 7 shapes)
  for that scene), and put_data is MuJoCo Warp's own again after any build.
- R equals MuJoCo C's nefc with every limit pushed past its range, every equality on and contacts off, on the
  compiled packs (Panda 10 = 1 equality + 9 limits; LEAP 32 = 16 friction loss + 16 limits; G1 86 = 43 + 43), and
  over 50 control periods of each pack every world's peak rows stay within r x its peak contacts + R
  (``alloy.api._capacity.watch_capacity``).
"""

from __future__ import annotations

import copy

import mujoco
import numpy as np
import pytest

import alloy
from alloy.api._capacity import contact_rows, model_rows, mujoco_solver, watch_capacity
from alloy.scene.entries import default_capacities

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="l1" pos="0 0 0.3">
  <joint name="j1" type="hinge" axis="0 1 0" range="-1 1" damping="2" frictionloss="0.1"/>
  <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/>
  <body name="l2" pos="0.2 0 0"><joint name="j2" type="hinge" axis="0 1 0" range="-1.5 1.5" damping="1"/>
  <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.5"/></body></body></worldbody>
  <actuator><position joint="j1" kp="100" kv="5"/><position joint="j2" kp="50" kv="3"/></actuator></mujoco>"""


def _arm_and_boxes(worlds: int = 2) -> alloy.Scene:
    sc = alloy.Scene(dt=0.01, worlds=worlds)
    sc.add_ground()
    sc.add_robot("arm", mjcf=ARM, pos=(0.0, 0.5, 0.0))
    for i in range(4):
        sc.add_box(f"b{i}", size=(0.04, 0.04, 0.04), pos=(0.08 * i, 0.0, 0.025), mass=0.1)
    return sc


def _kw(sim: alloy.Sim) -> dict:
    return next(e.solver_kwargs for e in sim.built.entries if e.capability == "mujoco")


@pytest.mark.law
def test_contact_capacity_sizes_rows_from_the_model_and_the_ccd_scratch_follows(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    sim = alloy.build(_arm_and_boxes(), device="cpu", contact_capacity=40)
    s = mujoco_solver(sim)
    d = s.mjw_data
    assert (contact_rows(s.mj_model), model_rows(s.mj_model)) == (3, 3)      # elliptic condim 3; 2 limits + 1 loss
    assert int(d.njmax) == 3 * 2 * 40 + 3, int(d.njmax)   # POOL_SHARE = 2
    assert int(d.naconmax) == int(d.naccdmax) == 40 * 2
    sim.step(20)
    assert sim.check()["ok"], sim.check()["problems"]


def test_no_contact_capacity_keeps_the_entrys_sizes(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    from alloy._vendor import mujoco_warp

    stock = mujoco_warp.put_data
    alloy.build(_arm_and_boxes(), device="cpu", contact_capacity=40)
    assert mujoco_warp.put_data is stock
    sim = alloy.build(_arm_and_boxes(), device="cpu")
    d = mujoco_solver(sim).mjw_data
    assert (_kw(sim)["nconmax"], _kw(sim)["njmax"]) == default_capacities(7) == (64, 384)   # ground, 2 links, 4 boxes
    assert (int(d.naconmax), int(d.njmax), int(d.naccdmax)) == (64 * 2, 384, 64 * 2)


def test_put_data_is_restored_after_a_failing_build_and_never_wrapped_without_the_option():
    from alloy._vendor import mujoco_warp
    from alloy.api._capacity import rows_from_contacts

    stock = mujoco_warp.put_data
    with pytest.raises(RuntimeError, match="boom"), rows_from_contacts(40):
        assert mujoco_warp.put_data is not stock
        raise RuntimeError("boom")
    assert mujoco_warp.put_data is stock
    with rows_from_contacts(None):                       # contact_capacity not given: no wrap at all
        assert mujoco_warp.put_data is stock


def test_a_sized_build_leaves_put_data_and_the_solver_patches_originals_alone(monkeypatch):
    """WP5: mjw_chol / mjw_block install again during the build and must not take the wrap for the original."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    from alloy._vendor import mujoco_warp
    from alloy.precision import mjw_block, mjw_chol

    alloy.build(_arm_and_boxes(), device="cpu")                      # installs whatever the route installs
    def originals() -> list:
        return [mujoco_warp.put_data, *mjw_chol._STATE.items(), *mjw_block._STATE.items()]

    before = originals()
    alloy.build(_arm_and_boxes(), device="cpu", contact_capacity=40)
    after = originals()
    assert len(after) == len(before) and all(a is b or (a[0] == b[0] and a[1] is b[1])
                                             for a, b in zip(after, before, strict=True))


def test_another_threads_build_is_untouched_inside_a_sized_builds_window(monkeypatch):
    """WP13: a build without the option that runs while a sized build holds the wrap keeps its own njmax."""
    import threading

    from alloy._vendor import mujoco_warp
    from alloy.api._capacity import rows_from_contacts

    seen = {}
    monkeypatch.setattr(mujoco_warp, "put_data", lambda mjm, mjd, **kw: seen.update(kw))
    stub = mujoco_warp.put_data
    inside, leave = threading.Event(), threading.Event()

    def sized_build() -> None:
        with rows_from_contacts(40):
            inside.set()
            leave.wait(30)

    t = threading.Thread(target=sized_build)
    t.start()
    assert inside.wait(30)
    mujoco_warp.put_data(None, None, nconmax=64, njmax=999, njmax_nnz=None)   # the other thread: passed through
    leave.set()
    t.join(30)
    assert seen == {"nconmax": 64, "njmax": 999, "njmax_nnz": None} and mujoco_warp.put_data is stub


def test_a_swap_made_inside_the_window_is_kept_and_the_wrapper_switches_off(monkeypatch):
    """WP0: the exit restores only its own wrapper; one that something else still holds passes through."""
    from alloy._vendor import mujoco_warp
    from alloy.api._capacity import rows_from_contacts

    seen = {}
    monkeypatch.setattr(mujoco_warp, "put_data", lambda mjm, mjd, **kw: seen.update(kw))

    def other(*a, **kw):
        return None

    with rows_from_contacts(40):
        wrapper = mujoco_warp.put_data
        mujoco_warp.put_data = other                     # e.g. an install or uninstall inside the window
    assert mujoco_warp.put_data is other
    wrapper(None, None, nconmax=64, njmax=999, njmax_nnz=None)
    assert seen == {"nconmax": 64, "njmax": 999, "njmax_nnz": None}


def test_constraint_rows_given_wins_over_the_model_count(monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    sim = alloy.build(_arm_and_boxes(), device="cpu", contact_capacity=40, constraint_rows=200)
    assert int(mujoco_solver(sim).mjw_data.njmax) == 200


def _pack(name: str) -> alloy.Scene:
    from alloy.cards.library import default_root

    sc = alloy.Scene(dt=0.005, worlds=2)
    sc.add_robot("r", default_root() / "robots" / name / "scene.xml")
    return sc


def _every_limit_rows(mjm) -> int:
    """MuJoCo C's nefc with contacts off, every equality on, each limited hinge / slide past its lower range."""
    m = copy.deepcopy(mjm)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    d = mujoco.MjData(m)
    d.eq_active[:] = 1
    hinge_slide = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    for j in np.flatnonzero(np.asarray(m.jnt_limited)):
        assert int(m.jnt_type[j]) in hinge_slide, "a limited ball joint: push it past its range too"
        d.qpos[m.jnt_qposadr[j]] = m.jnt_range[j][0] - 0.1
    mujoco.mj_forward(m, d)
    return int(d.nefc)


@pytest.mark.law
@pytest.mark.parametrize("pack, rows", [("emb_e2_panda", 10), ("emb_e9_leap", 32), ("emb_e6_g1", 86)])
def test_model_rows_are_mujocos_rows_with_every_limit_active(pack, rows, monkeypatch):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "warp")
    sim = alloy.build(_pack(pack), device="cpu")
    use = watch_capacity(sim)
    mjm = use.solver.mj_model
    assert model_rows(mjm) == _every_limit_rows(mjm) == rows
    sim.step(50)
    u, r = use.read(), contact_rows(mjm)
    assert all(n <= r * c + rows for n, c in zip(u["rows"], u["contacts"])), (u["rows"], u["contacts"], r)
    assert min(u["contacts"]) > 0                      # every world touched something (the packs rest on floors)
