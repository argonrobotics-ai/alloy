# SPDX-License-Identifier: Apache-2.0
"""Scoped noslip in C (alloy.route._mjc, SCR-WP8-6) against MuJoCo C's own noslip and alloy.contact.noslip.

- scope "all" with MuJoCo's restart and cold QCQPs is MuJoCo's noslip (engine_solver.c, monolithic) to rounding, for
  condim 3, 4 and 6 contacts, dense and sparse Jacobians;
- scope "robot" changes only the robot's rows (its dof friction rows, finger-block contacts) and leaves the other
  trees' qacc bit for bit; geom pairs and surface geoms add their contacts; extra slots keep the main solve's forces;
- the route's start (restart off) and warm QCQPs reach MuJoCo's forces up to the stopping tolerances;
- against ScopedNoslip on MuJoCo Warp (CPU, float32) from the same state: the same slots, and the same change of
  qacc to float32 accuracy.
"""

from __future__ import annotations

import types

import mujoco
import numpy as np
import pytest

from alloy.contact.noslip import ALL, ROBOT, NoslipSpec, ScopedNoslip, build_scope
from alloy.route import _mjc
from mjc_scenes import arm, c_step, copy_state, settled

pytestmark = pytest.mark.cpu_route
lib = c_step()
FRICTION = (mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF, mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC)


def noslip(m, d, spec, scope):
    cfg = _mjc.MjcConfig(noslip=spec, scope=scope, line=False)
    out = _mjc.MjcOut()
    assert lib.alloy_mjc_noslip(cfg.struct, m._address, d._address, out.struct) == 0
    return out.struct


def geom(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)


def contact_rows(m, d, pairs):
    """efc rows of the friction rows of contacts between the named geom pairs."""
    want = {frozenset((geom(m, a), geom(m, b))) for a, b in pairs}
    rows = []
    for c in d.contact[:d.ncon]:
        if frozenset(c.geom) in want:
            rows += range(c.efc_address + 1, c.efc_address + c.dim)
    return rows


@pytest.mark.parametrize("jacobian", [mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE])
@pytest.mark.parametrize("condim", [3, 4, 6])
def test_scope_all_with_restart_is_mujocos_noslip(condim: int, jacobian: int) -> None:
    models = [arm(condim), arm(condim)]
    for mm in models:
        mm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
        mm.opt.jacobian, mm.opt.tolerance = jacobian, 1e-12
    m, mref = models
    mref.opt.noslip_iterations, mref.opt.noslip_tolerance = 30, -1.0     # every sweep, as below
    d = settled(m)
    ref = copy_state(mref, d)
    mujoco.mj_forward(mref, ref)
    mujoco.mj_forward(m, d)
    start = d.efc_force.copy()
    out = noslip(m, d, NoslipSpec(30, -1.0, scope=ALL, restart=True, warm_qcqp=False, rows_per_world=1000),
                 build_scope(m, ALL))
    assert out.noslip_sweeps == 30 and out.noslip_slots == 2 + sum(c.dim > 1 for c in d.contact[:d.ncon])
    assert np.abs(ref.efc_force - start).max() > 1e-3                    # MuJoCo's noslip moved the forces
    assert np.abs(d.efc_force - ref.efc_force).max() < 1e-9                # measured 1.0e-11 N (forces to 26 N)
    assert np.abs(d.qacc - ref.qacc).max() < 1e-7                           # measured 4.5e-10


def test_robot_scope_moves_only_robot_rows() -> None:
    m = arm(6)
    d = settled(m)
    mujoco.mj_forward(m, d)
    f0, q0 = d.efc_force.copy(), d.qacc.copy()
    out = noslip(m, d, NoslipSpec(50, 1e-10), build_scope(m, ROBOT))
    dof_rows = list(np.flatnonzero(d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))
    scoped = dof_rows + contact_rows(m, d, [("finger", "block")])
    assert out.noslip_slots == 2 + sum(frozenset(c.geom) == {geom(m, "finger"), geom(m, "block")}
                                       for c in d.contact[:d.ncon]) and out.noslip_overflow == 0
    others = np.setdiff1d(np.arange(d.nefc), scoped)
    assert np.array_equal(d.efc_force[others], f0[others])
    assert np.abs(d.efc_force[scoped] - f0[scoped]).max() > 1e-3
    still = np.isin(m.dof_treeid, [m.body_treeid[m.body(b).id] for b in ("sheet", "slab")])
    assert np.array_equal(d.qacc[still], q0[still]) and np.abs(d.qacc[~still] - q0[~still]).max() > 1e-3


def test_pairs_surfaces_and_overflow() -> None:
    m = arm(3)
    d = settled(m)
    mujoco.mj_forward(m, d)
    n = {k: sum(frozenset(c.geom) == {geom(m, a), geom(m, b)} for c in d.contact[:d.ncon])
         for k, (a, b) in {"fb": ("finger", "block"), "bf": ("block", "floor"), "sf": ("sheet", "floor"),
                           "ss": ("sheet", "slab")}.items()}
    assert min(n.values()) > 0
    cases = ((build_scope(m, ROBOT, pairs=[(geom(m, "block"), geom(m, "floor"))]), 2 + n["fb"] + n["bf"]),
             (build_scope(m, ROBOT, surfaces=[geom(m, "sheet")]), 2 + n["fb"] + n["sf"] + n["ss"]),
             (build_scope(m, ROBOT, surfaces=[geom(m, "sheet"), geom(m, "slab")]), 2 + n["fb"] + n["sf"]))
    for scope, slots in cases:
        e = copy_state(m, d)
        mujoco.mj_forward(m, e)
        assert noslip(m, e, NoslipSpec(5, 1e-10), scope).noslip_slots == slots
    f0 = d.efc_force.copy()
    out = noslip(m, d, NoslipSpec(5, 1e-10, rows_per_world=1), build_scope(m, ROBOT))
    assert out.noslip_slots == 1 and out.noslip_overflow == 1 and out.noslip_sweeps >= 1
    first = int(np.flatnonzero(d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)[0])
    rest = np.setdiff1d(np.arange(d.nefc), [first])                         # j1's row only (here at its limit)
    assert np.array_equal(d.efc_force[rest], f0[rest])


@pytest.mark.parametrize("condim", [3, 6])
def test_route_start_and_warm_qcqp_reach_the_same_forces(condim: int) -> None:
    m = arm(condim)
    m.opt.tolerance, m.opt.iterations = 1e-14, 200
    base = settled(m)
    got = {}
    for restart in (False, True):
        for warm in (False, True):
            d = copy_state(m, base)
            mujoco.mj_forward(m, d)
            noslip(m, d, NoslipSpec(500, 1e-15, restart=restart, warm_qcqp=warm), build_scope(m, ROBOT))
            got[restart, warm] = d.efc_force.copy()
    ref = got[True, False]                                                  # MuJoCo's start and QCQP
    assert np.abs(got[False, False] - ref).max() < 1e-7                     # main solve tolerance
    assert np.abs(got[False, True] - ref).max() < 1e-4                      # QCQP stopping rule (1e-10)


@pytest.mark.compile_heavy
@pytest.mark.parametrize("condim", [3, 6])
def test_matches_scoped_noslip_on_mujoco_warp(condim: int) -> None:
    mjw = pytest.importorskip("alloy._vendor.mujoco_warp")
    wp = pytest.importorskip("warp")
    from alloy.contact.noslip import _route

    spec = NoslipSpec(50, 1e-10)
    m = arm(condim)
    m.opt.tolerance = 1e-10
    d = settled(m)
    mujoco.mj_forward(m, d)
    q0 = d.qacc.copy()
    out = noslip(m, d, spec, build_scope(m, ROBOT))
    with wp.ScopedDevice("cpu"):
        m.opt.noslip_iterations, m.opt.noslip_tolerance = spec.iterations, spec.tolerance
        wm, wm0 = mjw.put_model(m), mjw.put_model(m)
        m.opt.noslip_iterations = 0
        wd = mjw.put_data(m, d, nworld=1, nconmax=64, njmax=256)
        for x in (wm, wm0):
            x.opt.disableflags = int(x.opt.disableflags) | int(mjw.DisableBit.NATIVECCD)
        wm0.opt.noslip_iterations = 0
        inst = ScopedNoslip(types.SimpleNamespace(mjw_model=wm, mjw_data=wd, mj_model=m), spec)
        mjw.forward(wm0, wd)
        w0 = wd.qacc.numpy()[0].copy()
        _route(wm, inst)
        mjw.forward(wm, wd)
        dw = wd.qacc.numpy()[0] - w0
    dc = d.qacc - q0
    assert int(inst.count.numpy()[0]) == out.noslip_slots == 5             # 2 dof rows + 3 finger-block contacts
    assert np.abs(dc).max() > 1.0 and np.abs(dc - dw).max() < 2e-3 * np.abs(dw).max()
