# SPDX-License-Identifier: Apache-2.0
"""The CPU route's numpy plug-ins (alloy.route.mjc_core_noslip / mjc_core_line, SCR-WP8-6) on their own, and their
parity with WP2's C extension (alloy.route._mjc; skipped until ``_mjc.load()`` returns the library).

- numpy scoped noslip, scope "all" with MuJoCo's restart and cold QCQPs, is MuJoCo's own noslip to rounding (condim
  3, 4, 6; dense and sparse Jacobians); scope "robot" leaves the other trees' qacc bit for bit;
- numpy line rule on upstream package 2's pose (BUL-0014): MuJoCo's one end contact becomes the overlap's two ends;
- C vs numpy: the same noslip forces and qacc from the same state, the same line contacts on seeded poses;
- the stop rule scales by 1 / trace(M) at the current configuration, as MuJoCo 3.17 does (a folded arm, where it
  differs 5x from 3.16's qpos0 scale): numpy = MuJoCo's own noslip, C = numpy, sweep for sweep.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from alloy.contact.noslip import ALL, ROBOT, NoslipSpec, build_scope
from alloy.route.mjc_core_config import RouteConfig
from alloy.route.mjc_core_line import LineRule
from alloy.route.mjc_core_noslip import ScopedNoslipNumpy
from mjc_scenes import arm, copy_state, settled

pytestmark = pytest.mark.cpu_route

SLAB = '<geom name="slab" type="box" size="0.3 0.6 0.00635" pos="0.35 0 -0.00635"/>'
BOX_XML = f"""<mujoco><worldbody>{SLAB}
  <body name="box"><freejoint/><geom name="box" type="box" size="0.0375 0.0041667 0.0075"/></body>
</worldbody></mujoco>"""
POS2 = [0.378351959, 0.599323405, 6.33253313e-05]           # upstream package 2's pose (tests/route/test_mjc_boxbox)
QUAT2 = [-0.00729818454, 0.0299928649, -0.200104167, 0.979288255]


def numpy_noslip(m, d, spec):
    ns = ScopedNoslipNumpy(spec, build_scope(m, spec.scope), RouteConfig(noslip=spec).max_slots())
    ns.run(m, d)
    return ns.counters


def c_lib():
    from alloy.route import _mjc

    lib = _mjc.load()
    if lib is None:
        pytest.skip(f"C extension unavailable: {_mjc.reason}")
    return _mjc, lib


@pytest.mark.parametrize("jacobian", [mujoco.mjtJacobian.mjJAC_DENSE, mujoco.mjtJacobian.mjJAC_SPARSE])
@pytest.mark.parametrize("condim", [3, 4, 6])
def test_scope_all_with_restart_is_mujocos_noslip(condim: int, jacobian: int) -> None:
    m, mref = arm(condim), arm(condim)
    for mm in (m, mref):
        mm.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ISLAND
        mm.opt.jacobian, mm.opt.tolerance = jacobian, 1e-12
    mref.opt.noslip_iterations, mref.opt.noslip_tolerance = 30, -1.0     # every sweep, as below
    d = settled(m)
    ref = copy_state(mref, d)
    mujoco.mj_forward(mref, ref)
    mujoco.mj_forward(m, d)
    start = d.efc_force.copy()
    c = numpy_noslip(m, d, NoslipSpec(30, -1.0, scope=ALL, restart=True, warm_qcqp=False, rows_per_world=1000))
    assert c.sweeps == 30 and c.slots == 2 + sum(int(x.dim > 1) for x in d.contact[:d.ncon])
    assert np.abs(ref.efc_force - start).max() > 1e-3                     # MuJoCo's noslip moved the forces
    assert np.abs(d.efc_force - ref.efc_force).max() < 1e-9
    assert np.abs(d.qacc - ref.qacc).max() < 1e-7


def test_robot_scope_leaves_other_trees_alone() -> None:
    m = arm(3)
    d = settled(m)
    mujoco.mj_forward(m, d)
    qacc0 = d.qacc.copy()
    c = numpy_noslip(m, d, NoslipSpec(50, 1e-10, scope=ROBOT))
    assert c.slots >= 2 and c.overflow == 0                               # j1, j2 friction rows (+ finger contacts)
    tree = np.asarray(m.dof_treeid)
    robot = tree == m.body_treeid[m.body("base").id]
    away = np.isin(tree, [m.body_treeid[m.body(n).id] for n in ("sheet", "slab")])   # no scoped row touches them
    assert np.array_equal(d.qacc[away], qacc0[away])
    assert not np.array_equal(d.qacc[robot], qacc0[robot])


def _posed(pos=POS2, quat=QUAT2):
    m = mujoco.MjModel.from_xml_string(BOX_XML)
    d = mujoco.MjData(m)
    d.qpos[:3], d.qpos[3:7] = pos, quat
    mujoco.mj_kinematics(m, d)
    return m, d


def test_package2_edge_on_face_becomes_a_line_contact() -> None:
    m, d = _posed()
    mujoco.mj_step1(m, d)
    assert d.ncon == 1 and d.contact.pos[0, 0] == pytest.approx(0.3408, abs=1e-3)   # the defect: one end contact
    depth = float(d.contact.dist[0])
    assert LineRule(m).apply(m, d) == 1
    xs = sorted(d.contact.pos[: d.ncon, 0])
    assert d.ncon == 2 and xs[0] < 0.342 and xs[1] > 0.414                 # spans the slab edge inside the box
    assert (d.contact.dist[:2] < 0.0).all() and d.contact.dist[:2].min() == pytest.approx(depth, abs=2e-5)
    assert set(d.efc_id[: d.nefc].tolist()) == {0, 1}                     # constraint rows rebuilt for both


def _seeded_poses(n: int = 40, seed: int = 0):
    """Package 2's pose turned up to 3 degrees about random axes, shifted along the edge and in height."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        axis = rng.normal(size=3)
        dq, quat = np.zeros(4), np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, axis / np.linalg.norm(axis), np.radians(rng.uniform(0.0, 3.0)))
        mujoco.mju_mulQuat(quat, dq, np.array(QUAT2))
        yield [POS2[0] + rng.uniform(-0.02, 0.02), POS2[1], POS2[2] + rng.uniform(-2e-4, 1e-4)], quat


def test_line_rule_matches_c() -> None:
    _mjc, lib = c_lib()
    fired = 0
    for pos, quat in _seeded_poses():
        m, d = _posed(pos, quat)
        cp, cn, cd = np.zeros(150), np.zeros(150), np.zeros(50)
        n = lib.alloy_mjc_boxbox(m._address, d._address, 0, 1, 0.0, 1, np.ctypeslib.as_ctypes(cp),
                                 np.ctypeslib.as_ctypes(cn), np.ctypeslib.as_ctypes(cd))
        mujoco.mj_step1(m, d)
        fired += LineRule(m).apply(m, d)
        assert d.ncon == n
        if n:
            assert np.abs(d.contact.pos[:n] - cp[: 3 * n].reshape(n, 3)).max() < 1e-12
            assert np.abs(d.contact.dist[:n] - cd[:n]).max() < 1e-12
            assert np.abs(d.contact.frame[:n, :3] - cn[: 3 * n].reshape(n, 3)).max() < 1e-12
    assert fired >= 5


@pytest.mark.parametrize("condim", [3, 4, 6])
def test_noslip_matches_c(condim: int) -> None:
    _mjc, lib = c_lib()
    m = arm(condim)
    d = settled(m)
    mujoco.mj_forward(m, d)
    for spec in (NoslipSpec(50, 1e-10, scope=ROBOT), NoslipSpec(50, 1e-10, scope=ALL)):
        dc, dn = copy_state(m, d), copy_state(m, d)
        for dd in (dc, dn):
            mujoco.mj_forward(m, dd)
        scope = build_scope(m, spec.scope)
        out = _mjc.MjcOut()
        cfg = _mjc.MjcConfig(noslip=spec, scope=scope, line=False)
        assert lib.alloy_mjc_noslip(cfg.struct, m._address, dc._address, out.struct) == 0
        c = numpy_noslip(m, dn, spec)
        assert (c.slots, c.sweeps) == (out.struct.noslip_slots, out.struct.noslip_sweeps)
        assert np.abs(dn.efc_force - dc.efc_force).max() < 1e-9
        assert np.abs(dn.qacc - dc.qacc).max() < 1e-7
        assert np.abs(dn.qfrc_constraint - dc.qfrc_constraint).max() < 1e-9


FOLDED = """<mujoco><option timestep="0.002" cone="elliptic" tolerance="1e-12"/><worldbody>
 <body name="base" pos="0 0 .5"><joint name="j1" type="hinge" axis="0 0 1" frictionloss="5" damping="0.1"/>
  <geom type="capsule" fromto="0 0 0 .3 0 0" size=".01" mass=".1"/>
  <body name="l2" pos=".3 0 0"><joint name="j2" type="hinge" axis="0 0 1" frictionloss="3" damping="0.05"/>
   <geom type="capsule" fromto="0 0 0 .25 0 0" size=".01" mass=".05"/>
   <geom type="sphere" pos=".25 0 0" size=".03" mass="1"/></body></body></worldbody>
 <actuator><motor joint="j1"/><motor joint="j2"/></actuator></mujoco>"""


def _folded(noslip_tolerance: float | None = None):
    """A two-link arm with a heavy tip folded back (trace(M) 0.19 of qpos0's), both joints sticking in dry friction:
    noslip's coupled Gauss-Seidel needs several sweeps, so its stop rule decides the forces."""
    m = mujoco.MjModel.from_xml_string(FOLDED)
    if noslip_tolerance is not None:
        m.opt.noslip_iterations, m.opt.noslip_tolerance = 100, noslip_tolerance
    d = mujoco.MjData(m)
    d.qpos[:], d.qvel[:], d.ctrl[:] = (0.3, 0.95 * np.pi), (0.004, -0.007), (0.05, -0.02)
    mujoco.mj_forward(m, d)
    return m, d


@pytest.mark.parametrize("tol", [1e-4, 1e-5, 1e-8, 1e-9, 1e-10])
def test_stop_rule_scales_by_the_current_inertia_trace(tol: float) -> None:
    """MuJoCo 3.17 stops noslip on improvement / trace(M) at the current configuration (3.16: trace(M(qpos0)) =
    stat.meaninertia * nv); the numpy fallback stops on the same sweep, so its forces are MuJoCo's to rounding. At
    these tolerances the qpos0 scale stops one sweep early (forces 1e-7 to 1.5e-4 off)."""
    from alloy.route.mjc_core_noslip import inertia_scale

    mref, ref = _folded(tol)
    m, d = _folded()
    assert inertia_scale(m, d) * float(m.stat.meaninertia) * m.nv > 5.0          # the two scales differ 5x here
    c = numpy_noslip(m, d, NoslipSpec(100, tol, scope=ALL, restart=True, warm_qcqp=False, rows_per_world=1000))
    assert c.sweeps >= 4
    assert np.abs(d.efc_force[:d.nefc] - ref.efc_force[:ref.nefc]).max() < 1e-12


def test_stop_rule_matches_c() -> None:
    """The C port stops on the same sweep as the numpy fallback (the same 1 / trace(M) scale) at a tolerance where
    the qpos0 scale would stop earlier."""
    _mjc, lib = c_lib()
    spec = NoslipSpec(100, 1e-9, scope=ALL)
    m, dc = _folded()
    _, dn = _folded()
    out = _mjc.MjcOut()
    cfg = _mjc.MjcConfig(noslip=spec, scope=build_scope(m, spec.scope), line=False)
    assert lib.alloy_mjc_noslip(cfg.struct, m._address, dc._address, out.struct) == 0
    c = numpy_noslip(m, dn, spec)
    assert c.sweeps >= 4 and (c.slots, c.sweeps) == (out.struct.noslip_slots, out.struct.noslip_sweeps)
    assert np.abs(dn.efc_force - dc.efc_force).max() < 1e-12
