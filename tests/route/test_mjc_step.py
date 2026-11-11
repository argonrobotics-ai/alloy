# SPDX-License-Identifier: Apache-2.0
"""The CPU route's C tick (alloy.route._mjc, SCR-WP8-6): equal to mj_step with the plug-ins off, the agreed refusals,
and the contact export, state pull, trajectory, batch and health counters (ABI v2, alloy_mjc.h)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from alloy.route import _mjc
from alloy.route._mjc import abi
from mjc_scenes import CTRL, arm, c_step, copy_state, settled

pytestmark = pytest.mark.cpu_route
lib = c_step()
STATE = ("qpos", "qvel", "qacc", "efc_force", "qacc_warmstart", "act")


def tick(m, d, cfg, nstep, out=None):
    out = out or _mjc.MjcOut()
    return lib.alloy_mjc_tick(cfg.struct, m._address, d._address, nstep, out.struct), out


@pytest.mark.parametrize("integrator", ["Euler", "implicit", "implicitfast"])
def test_tick_equals_mj_step_bit_for_bit(integrator: str) -> None:
    m = arm(6, integrator)
    ref, got = mujoco.MjData(m), mujoco.MjData(m)
    for d in (ref, got):
        d.ctrl[:] = CTRL
    for _ in range(200):
        mujoco.mj_step(m, ref)
    rc, out = tick(m, got, _mjc.MjcConfig(line=False), 200)
    assert rc == abi.OK and out.struct.steps == 200 and got.ncon > 0
    for f in STATE:
        assert np.array_equal(getattr(ref, f), getattr(got, f)), f
    assert got.time == ref.time


def test_refusals_leave_the_state_alone() -> None:
    m = arm()
    d = settled(m, 10)
    qpos = d.qpos.copy()
    spec = type("S", (), dict(iterations=5, tolerance=1e-8, scope="robot", restart=False, warm_qcqp=True,
                              rows_per_world=None))()
    scope = type("C", (), dict(geom=np.ones(m.ngeom), dof=np.ones(m.nv), pairs=np.zeros((0, 2))))()
    noslip = _mjc.MjcConfig(noslip=spec, scope=scope, line=False)
    bad_abi = _mjc.MjcConfig()
    bad_abi.struct.abi = 99
    assert tick(m, d, bad_abi, 1)[0] == abi.E_ABI
    assert tick(m, d, _mjc.MjcConfig(), 2, _mjc.MjcOut(nbody=m.nbody, traj_cap=1))[0] == abi.E_ARGS
    assert tick(m, d, _mjc.MjcConfig(pull_state=True), 1)[0] == abi.E_ARGS          # no body_map
    for field, value, code, cfg in (("integrator", mujoco.mjtIntegrator.mjINT_RK4, abi.E_INTEGRATOR, bad_abi),
                                    ("enableflags", mujoco.mjtEnableBit.mjENBL_SLEEP, abi.E_SLEEP, noslip),
                                    ("noslip_iterations", 3, abi.E_NOSLIP_TWICE, noslip),
                                    ("cone", mujoco.mjtCone.mjCONE_PYRAMIDAL, abi.E_CONE, noslip)):
        mm = arm()
        setattr(mm.opt, field, value)
        cfg.struct.abi = abi.ABI
        assert tick(mm, d, cfg, 1)[0] == code, field
    assert np.array_equal(d.qpos, qpos)


def test_export_state_trajectory_and_counters() -> None:
    m = arm(4)
    d = settled(m, 100)
    step1 = copy_state(m, d)
    body_map = np.array([1, -1, *range(2, m.nbody)], np.int32)          # one row left unwritten
    cfg = _mjc.MjcConfig(line=False, export_contacts=True, pull_state=True, body_map=body_map)
    out = _mjc.MjcOut(ncon_max=4, nbody=body_map.size, traj_cap=8)
    out.body_q[1] = out.body_qd[1] = 7.0
    rc, _ = tick(m, d, cfg, 8, out)
    assert rc == abi.OK and d.ncon > 4
    assert out.struct.ncon == 4 and out.struct.contact_overflow == d.ncon - 4
    for i in range(4):
        c, f = d.contact[i], np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        assert list(out.geom[i]) == list(c.geom) and np.array_equal(out.pos[i], c.pos)
        assert np.array_equal(out.frame[i], c.frame) and out.dist[i] == c.dist and np.array_equal(out.force[i], f)
    mujoco.mj_forward(m, d)
    for row, b in enumerate(body_map):
        if b < 0:
            assert (out.body_q[row] == 7.0).all() and (out.body_qd[row] == 7.0).all()
            continue
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, int(b), v, 0)
        assert np.allclose(out.body_q[row], [*d.xpos[b], *d.xquat[b][1:], d.xquat[b][0]], atol=1e-12)
        assert np.allclose(out.body_qd[row], [*v[3:], *v[:3]], atol=1e-12)
    rows = body_map >= 0
    assert np.array_equal(out.traj_body_q[-1][rows], out.body_q[rows])
    one = _mjc.MjcOut(nbody=body_map.size, traj_cap=1)
    for k in range(3):
        assert tick(m, step1, cfg, 1, one)[0] == abi.OK
        assert np.array_equal(one.traj_body_q[0], out.traj_body_q[k])
    c = out.counters()
    assert c["steps"] == 8 and c["nonfinite"] == 0 and 0 < c["pen_max"] < 0.01 and c["qvel_max"] > 0


def test_batch_runs_active_worlds_in_order() -> None:
    m = arm()
    worlds = [settled(m, 20 + 10 * w) for w in range(3)]
    refs = [copy_state(m, d) for d in worlds]
    cfgs = [_mjc.MjcConfig(line=False) for _ in worlds]
    outs = (abi.Out * 3)()
    cfgp = (abi.ctypes.POINTER(abi.Config) * 3)(*[abi.ctypes.pointer(c.struct) for c in cfgs])
    mp = (abi.ctypes.c_void_p * 3)(*[m._address] * 3)
    dp = (abi.ctypes.c_void_p * 3)(*[d._address for d in worlds])
    active = (abi.ctypes.c_uint8 * 3)(1, 0, 1)
    status = (abi.ctypes.c_int32 * 3)()
    assert lib.alloy_mjc_tick_batch(3, cfgp, mp, dp, active, 5, outs, status) == 0
    for w, (d, r) in enumerate(zip(worlds, refs)):
        for _ in range(5 if active[w] else 0):
            mujoco.mj_step(m, r)
        assert np.array_equal(d.qpos, r.qpos), w
        assert outs[w].steps == (5 if active[w] else 0)
    mm = arm()
    mm.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
    mp[1] = mm._address
    active[1] = 1
    assert lib.alloy_mjc_tick_batch(3, cfgp, mp, dp, active, 1, outs, status) == 1
    assert list(status) == [abi.OK, abi.E_INTEGRATOR, abi.OK]


def test_a_reset_world_counts_as_nonfinite(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)                                              # MuJoCo writes MUJOCO_LOG.TXT here
    m = arm()
    d = settled(m, 5)
    d.qvel[0] = np.nan
    rc, out = tick(m, d, _mjc.MjcConfig(line=False), 3)
    assert rc == abi.OK and out.struct.nonfinite >= 1 and np.isfinite(d.qpos).all()
