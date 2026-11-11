# SPDX-License-Identifier: Apache-2.0
"""The CPU route on the MuJoCo C core (alloy.route.mjc_core, SCR-WP8-6), through the ``mujoco`` entry factory.

- the factory builds it on request (``alloy:cpu_route`` / ``ALLOY_CPU_ROUTE``), refuses the GPU-only entries and keeps
  MuJoCo Warp by default; the route's solver settings and the scene's noslip request as scoped noslip;
- with the plug-ins off a step is ``mj_step`` on the world's model bit for bit, on either backend;
- per-world models carry per-world gravity, uniform fields keep the template's values;
- ``step_many(n)`` equals ``n`` single steps bit for bit; an edited world is pushed, the others continue in float64
  (DirectSim's fast path: tests/bully/test_direct_mjc_fast.py);
- reset, the non-finite freeze, the Newton contact export and the health mirror;
- C and numpy backends agree (skipped until WP2's extension loads).
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp
import pytest

from alloy._vendor import newton

from alloy.core.errors import SolverOptionError
from alloy.route.mjc_core import MjcCoreSolver
from alloy.route.mjc_core_config import CPU_ROUTE_ENV, CPU_ROUTE_KEY
from alloy.scene.builtin_plugins import builtin_capabilities
from alloy.scene.noslip_route import NOSLIP_KEY
from mjc_scenes import ARM, CTRL

pytestmark = pytest.mark.cpu_route

DT = 0.002


def arm_model(worlds: int = 2, condim: int = 3, gravity: list | None = None) -> newton.Model:
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    t.add_mjcf(ARM.format(condim=condim, integrator="implicitfast"), ctrl_direct=True, parse_visuals=False)
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, copy_count=worlds)
    model = b.finalize(device="cpu")
    if gravity is not None:
        model.set_gravity(gravity)                       # one vector per world, before the solver exists
    return model


def factory(model: newton.Model, **kw):
    cap = next(c for c in builtin_capabilities() if c.name == "mujoco")
    return cap.factory(model, kw)


def route(model: newton.Model, backend: str = "auto", **kw) -> MjcCoreSolver:
    s = factory(model, **{CPU_ROUTE_KEY: "mjc", **kw})
    assert isinstance(s, MjcCoreSolver)
    if backend == "numpy":
        s.route = s.route.replace(backend="numpy")
        s._make_outputs()
    return s


def backends() -> list[str]:
    from alloy.route import _mjc

    return ["numpy"] + (["auto"] if _mjc.load() is not None else [])


class Run:
    def __init__(self, model: newton.Model, solver: MjcCoreSolver) -> None:
        self.m, self.s = model, solver
        self.s0, self.s1, self.ctl = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.s0)
        self.ctl.mujoco.ctrl.assign(np.tile(CTRL, model.world_count).astype(np.float32))

    def step(self, n: int = 1, many: bool = False) -> None:
        if many:
            self.s.step_many(self.s0, self.s1, self.ctl, DT, n)
            self.s0, self.s1 = self.s1, self.s0
            return
        for _ in range(n):
            self.s.step(self.s0, self.s1, self.ctl, None, DT)
            self.s0, self.s1 = self.s1, self.s0


def test_factory_builds_refuses_and_defaults(monkeypatch) -> None:
    import os

    assert os.environ[CPU_ROUTE_ENV] == "warp"                                      # the suite's (tests/conftest.py)
    m = arm_model(1)
    assert not isinstance(factory(m), MjcCoreSolver)
    assert isinstance(factory(m, **{CPU_ROUTE_KEY: "mjc"}), MjcCoreSolver)          # the entry key wins
    monkeypatch.delenv(CPU_ROUTE_ENV)
    assert isinstance(factory(m), MjcCoreSolver)                                    # users' default (SCR-WP8-8)
    assert not isinstance(factory(m, use_mujoco_cpu=True), MjcCoreSolver)           # Newton's own C step asked for
    monkeypatch.setenv(CPU_ROUTE_ENV, "mjc")
    s = factory(m, **{NOSLIP_KEY: (50, 1e-10)})
    assert isinstance(s, MjcCoreSolver) and s.route.noslip.scope == "robot" and s.route.noslip.iterations == 50
    w = s.worlds[0]
    assert w.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_WARMSTART and w.opt.tolerance <= 1e-9
    assert w.opt.noslip_iterations == 0 and not hasattr(s, "_alloy_carry")
    assert not isinstance(factory(m, use_mujoco_contacts=False), MjcCoreSolver)     # Newton pipeline: refused
    assert not isinstance(factory(m, **{CPU_ROUTE_KEY: "warp"}), MjcCoreSolver)
    with pytest.raises(SolverOptionError):
        factory(m, **{CPU_ROUTE_KEY: "fast"})


@pytest.mark.parametrize("backend", backends())
def test_plugins_off_is_mj_step_bit_for_bit(backend: str) -> None:
    m = arm_model(1)
    r = Run(m, route(m, backend, **{"alloy:boxbox_line": False}))
    r.step()                                                              # pushes the Newton state and ctrl
    s = r.s
    ref = mujoco.MjData(s.worlds[0])
    for f in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
        getattr(ref, f)[:] = getattr(s.datas[0], f)
    ref.time = s.datas[0].time
    r.step(50)
    for _ in range(50):
        mujoco.mj_step(s.worlds[0], ref)
    for f in ("qpos", "qvel", "qacc"):
        assert np.array_equal(getattr(ref, f), getattr(s.datas[0], f)), f


def test_per_world_gravity_and_uniform_fields() -> None:
    g = [(0.0, 0.0, -9.81), (0.0, 0.0, -1.0), (0.0, 0.0, 0.0)]
    m = arm_model(3, gravity=g)
    s = route(m)
    for w, gw in enumerate(g):
        assert np.allclose(s.worlds[w].opt.gravity, gw)
        assert np.array_equal(s.worlds[w].body_mass, s.mj_model.body_mass)
    r = Run(m, s)
    r.step(100)
    sheet = s.mj_model.body("worldbody_sheet").id
    z = [d.xpos[sheet, 2] for d in s.datas]
    assert z[0] < z[1] < z[2] == pytest.approx(0.1, abs=1e-7)             # falls by its gravity, stays at g = 0


@pytest.mark.parametrize("backend", backends())
def test_step_many_equals_single_steps(backend: str) -> None:
    runs = []
    for many in (False, True):
        m = arm_model(2)
        r = Run(m, route(m, backend, **{NOSLIP_KEY: (50, 1e-10)}))
        r.step()
        r.step(40, many=many)
        runs.append(r)
    a, b = runs
    assert np.array_equal(a.s0.body_q.numpy(), b.s0.body_q.numpy())
    assert np.array_equal(a.s0.joint_q.numpy(), b.s0.joint_q.numpy())
    assert all(np.array_equal(x.qpos, y.qpos) for x, y in zip(a.s.datas, b.s.datas))


def test_edited_world_is_pushed_and_others_continue() -> None:
    m = arm_model(2)
    runs = [Run(m, route(m)) for _ in range(2)]
    for r in runs:
        r.step(20)
    q = runs[1].s0.joint_q.numpy()
    starts = m.joint_q_start.numpy()
    jpw = m.joint_count // 2
    q[starts[jpw]] += 0.1                                                 # world 1's j1, through Newton's state
    for r in runs:
        r.step(10)
    a, b = runs[0].s, runs[1].s
    assert np.array_equal(a.datas[0].qpos, b.datas[0].qpos)              # world 0 kept its float64 state
    assert not np.allclose(a.datas[1].qpos, b.datas[1].qpos, atol=1e-3)


def test_reset_and_nonfinite_freeze(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)                                           # MuJoCo's warnings write MUJOCO_LOG.TXT here
    m = arm_model(2)
    r = Run(m, route(m))
    r.step(20)
    qd = r.s0.joint_qd.numpy()
    qd[m.joint_qd_start.numpy()[m.joint_count // 2]] = np.nan            # world 1 blows up
    r.step(2)
    assert r.s._frozen.tolist() == [False, True] and r.s.outs[1].counters()["nonfinite"] >= 1
    assert not np.isfinite(r.s0.body_q.numpy()[m.body_world.numpy() == 1]).all()
    frozen = r.s.datas[1].time
    r.step(3)
    assert r.s.datas[1].time == frozen                                    # skipped while frozen
    r.s.reset(r.s0)
    assert not r.s._frozen.any() and r.s.outs[1].counters()["steps"] == 0
    r.step(5)
    assert np.isfinite(r.s0.body_q.numpy()).all()
    assert np.allclose([d.time for d in r.s.datas], 5 * DT)


def test_contact_export_and_health_mirror() -> None:
    m = arm_model(2)
    s = route(m)
    r = Run(m, s)
    r.step(30)
    contacts = newton.Contacts(int(s.mjw_data.naconmax), 0, device="cpu")
    s.update_contacts(contacts, r.s0)
    n = int(contacts.rigid_contact_count.numpy()[0])
    assert n == sum(d.ncon for d in s.datas) > 0
    shapes = contacts.rigid_contact_shape0.numpy()[:n]
    assert (shapes >= 0).any() and np.allclose(np.linalg.norm(contacts.rigid_contact_normal.numpy()[:n], axis=1), 1)
    assert int(s.mjw_data.nacon.numpy()[0]) == n
    assert np.array_equal(np.sort(s.mjw_data.contact.dist.numpy()[:n]),
                          np.sort(np.concatenate([d.contact.dist[: d.ncon] for d in s.datas]).astype(np.float32)))


def test_backends_agree() -> None:
    if len(backends()) < 2:
        pytest.skip("C extension unavailable")
    runs = []
    for backend in ("numpy", "auto"):
        m = arm_model(2, condim=6)
        r = Run(m, route(m, backend, **{NOSLIP_KEY: (50, 1e-10)}))
        r.step(100, many=True)
        runs.append(r)
    assert runs[1].s.backend == "c"
    qa = np.array([d.qpos for d in runs[0].s.datas])
    qb = np.array([d.qpos for d in runs[1].s.datas])
    assert np.abs(qa - qb).max() < 1e-10                                   # measured 3e-14 over 300 steps
    ca, cb = (r.s.route_report()["counters"] for r in runs)
    assert (ca["noslip_slots"], ca["noslip_sweeps"]) == (cb["noslip_slots"], cb["noslip_sweeps"])


def test_rk4_entry_steps_plain_mj_step() -> None:
    m = arm_model(1)
    r = Run(m, route(m, integrator="rk4"))
    r.step()
    s = r.s
    assert s.backend == "mj_step" and s.worlds[0].opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    ref = mujoco.MjData(s.worlds[0])
    for f in ("qpos", "qvel", "act", "ctrl"):
        getattr(ref, f)[:] = getattr(s.datas[0], f)
    ref.time = s.datas[0].time
    r.step(20)
    for _ in range(20):
        mujoco.mj_step(s.worlds[0], ref)
    assert np.array_equal(ref.qpos, s.datas[0].qpos) and np.array_equal(ref.qvel, s.datas[0].qvel)


def test_c_error_codes_say_whether_the_state_advanced() -> None:
    if len(backends()) < 2:
        pytest.skip("C extension unavailable")
    from alloy.route._mjc import abi

    m = arm_model(1)
    s = route(m)
    out = s.outs[0]

    class Stub:
        def __init__(self, rc: int) -> None:
            self.rc = rc

        def alloy_mjc_tick(self, *args) -> int:
            return self.rc

    out.lib = Stub(abi.E_NOMEM)
    with pytest.raises(SolverOptionError, match="state advanced"):
        out.tick(s.worlds[0], s.datas[0], 4)
    out.lib = Stub(abi.E_ARGS)
    with pytest.raises(SolverOptionError, match="nothing was stepped"):
        out.tick(s.worlds[0], s.datas[0], 4)


def test_contact_force_observable_and_template_mirror() -> None:
    m = arm_model(1)
    s = route(m)
    kind = newton.solvers.SolverBase.ObservableKind.CONTACT_F
    assert s.supported_observables == frozenset({kind})
    contacts = newton.Contacts(s.get_max_contact_count(), 0, device="cpu")
    obs = s.observables({kind}, contacts=contacts)
    r = Run(m, s)
    for _ in range(30):
        s.step(r.s0, r.s1, r.ctl, None, DT, observables=obs)
        r.s0, r.s1 = r.s1, r.s0
    n = int(contacts.rigid_contact_count.numpy()[0])
    f = obs.contact_f.numpy()[:n]
    assert n == s.datas[0].ncon > 0 and np.isfinite(f).all() and np.abs(f[:, :3]).max() > 0.1   # blocks press
    assert s.mj_data.ncon == s.datas[0].ncon and np.array_equal(s.mj_data.qpos, s.datas[0].qpos)  # template mirror

    class Unsupported:
        def is_requested(self, k: str) -> bool:
            return k == newton.solvers.SolverBase.ObservableKind.BODY_QDD

    with pytest.raises(SolverOptionError, match="CONTACT_F only"):
        s.step(r.s0, r.s1, r.ctl, None, DT, observables=Unsupported())


def test_set_noslip_after_construction() -> None:
    from alloy.contact.noslip import NoslipSpec

    m = arm_model(1)
    s = route(m)
    assert s.route.noslip is None
    s.set_noslip(NoslipSpec(50, 1e-10, scope="robot"))
    r = Run(m, s)
    r.step(10)
    c = s.route_report()["counters"]
    assert c["noslip_slots"] >= 20 and s.worlds[0].opt.noslip_iterations == 0


def _limited_hinges(worlds: int) -> newton.Model:
    """A bar on a hinge with a ±0.1 rad limit, its weight pressing it into the upper stop (Newton's default limit
    gains, force space)."""
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    bar = t.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
    t.add_shape_box(bar, xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), hx=0.1, hy=0.01, hz=0.01)
    j = t.add_joint_revolute(-1, bar, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
                             axis=(0.0, 1.0, 0.0), limit_lower=-0.1, limit_upper=0.1)
    t.add_articulation([j])
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.replicate(t, copy_count=worlds)
    return b.finalize(device="cpu")


def test_limit_stiffness_is_every_worlds() -> None:
    """Newton's C path converts the joint-limit gains for world 0 only (rows 1+ of the Warp model keep MuJoCo's
    default stop, about 85x softer): every world's model carries the template's converted jnt_solref, and identical
    worlds rest on the stop at the same angle (WP15's review of rinse_drain, 2026-11-06)."""
    from alloy.route.mjc_core_models import batched_fields

    m = _limited_hinges(3)
    s = route(m).finalize_route()
    assert s.mj_model.jnt_limited.all() and not np.allclose(s.mj_model.jnt_solref, [0.02, 1.0])
    assert "jnt_solref" not in batched_fields(s, 3)
    for w in s.worlds:
        assert np.array_equal(w.jnt_solref, s.mj_model.jnt_solref)
    s0, s1, ctl = m.state(), m.state(), m.control()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
    for _ in range(200):
        s.step(s0, s1, ctl, None, DT)
        s0, s1 = s1, s0
    q = [d.qpos[0] for d in s.datas]
    assert q[0] > 0.1 and q[1] == q[0] and q[2] == q[0]                    # on the upper stop, alike
