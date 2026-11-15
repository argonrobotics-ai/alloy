# SPDX-License-Identifier: Apache-2.0
"""The CPU route's bodies welded to the world (MuJoCo mocap bodies) stand at each world's own pose, as on MuJoCo Warp.

Newton makes a body welded to the world a mocap body; its pose is data (``mocap_pos`` / ``mocap_quat``), one row per
world, written from the joint frames. The route used to copy only qpos / qvel into each world's MjData, so a body welded
at another pose in another world stood at world 0's pose there (a new MjData holds the template's) and the route wrote
that pose back into ``body_q``. Found by WP9 (tests/plugins/test_surface_flow_lumped_cast.py, ALLOY_TEST_CPU_ROUTE=mjc).

Laws: three worlds, each with a post welded at its own x and a ball dropped onto it; the route's posts and balls equal
MuJoCo Warp's (5e-4 m: float64 C against float32 Warp) at rest, after a joint-frame change moves world 2's post
(``notify_model_changed(JOINT_PROPERTIES)``) and after a reset. A public-API scene with per-world kinematic boxes, a
drive and a set_pose takes MuJoCo Warp on the CPU route request (the route refuses kinematic bodies), so its poses are
the warp route's bit for bit.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.route.mjc_core import MjcCoreSolver
from alloy.route.mjc_core_config import CPU_ROUTE_ENV, CPU_ROUTE_KEY
from alloy.scene.builtin_plugins import builtin_capabilities

pytestmark = pytest.mark.cpu_route

DT, XS, TOL = 0.002, (0.0, 0.3, 0.6), 5e-4
POST_TOP = 0.2                   # a 4 cm x 4 cm x 20 cm post welded with its foot on the ground


def _world(x: float) -> newton.ModelBuilder:
    t = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(t)
    xf = wp.transform(wp.vec3(x, 0.0, 0.1), wp.quat_identity())
    post = t.add_link(xform=xf, label="post")
    t.add_shape_box(post, hx=0.02, hy=0.02, hz=0.1)
    t.add_articulation([t.add_joint_fixed(-1, post, parent_xform=xf)])
    ball = t.add_body(xform=wp.transform(wp.vec3(x, 0.0, POST_TOP + 0.05), wp.quat_identity()), label="ball")
    t.add_shape_sphere(ball, radius=0.015)
    return t


def _model() -> newton.Model:
    root = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(root)
    for x in XS:
        root.add_world(_world(x))
    root.add_ground_plane()
    return root.finalize(device="cpu")


def _solver(model: newton.Model, route: str):
    cap = next(c for c in builtin_capabilities() if c.name == "mujoco")
    return cap.factory(model, {CPU_ROUTE_KEY: route})


class _Run:
    def __init__(self, route: str) -> None:
        self.m = _model()
        self.s = _solver(self.m, route)
        self.a, self.b, self.ctl = self.m.state(), self.m.state(), self.m.control()
        newton.eval_fk(self.m, self.m.joint_q, self.m.joint_qd, self.a)

    def step(self, n: int) -> None:
        for _ in range(n):
            self.s.step(self.a, self.b, self.ctl, None, DT)
            self.a, self.b = self.b, self.a

    def poses(self) -> np.ndarray:
        """(worlds, 2, 3): each world's post and ball positions."""
        return self.a.body_q.numpy()[:, :3].reshape(len(XS), 2, 3)

    def move_post(self, world: int, dx: float) -> None:
        """World ``world``'s post dx along x by its weld's joint frame (the engine layer's way to move it)."""
        xp = self.m.joint_X_p.numpy()
        j = int(np.flatnonzero(self.m.joint_type.numpy() == int(newton.JointType.FIXED))[world])
        xp[j, 0] += dx
        self.m.joint_X_p.assign(xp)
        self.s.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)


def _close(mjc: _Run, warp: _Run, what: str) -> None:
    a, b = mjc.poses(), warp.poses()
    worst = float(np.abs(a - b).max())
    assert worst <= TOL, f"{what}: C route vs MuJoCo Warp, worst {worst:.2e} m\nC route:\n{a}\nwarp:\n{b}"


def test_welded_bodies_stand_at_each_worlds_own_pose() -> None:
    mjc, warp = _Run("mjc"), _Run("warp")
    assert isinstance(mjc.s, MjcCoreSolver) and not isinstance(warp.s, MjcCoreSolver)
    assert int(mjc.s.mj_model.nmocap) == 1                       # the post, in every world
    for r in (mjc, warp):
        r.step(150)                                              # 0.3 s: each ball lands on its own post
    _close(mjc, warp, "at rest")
    np.testing.assert_allclose(mjc.poses()[:, 0, 0], XS, atol=1e-6)
    assert np.all(mjc.poses()[:, 1, 2] > POST_TOP)               # no ball fell past its post
    for r in (mjc, warp):
        r.move_post(2, 0.1)                                      # world 2's post leaves from under its ball
        r.step(150)
    _close(mjc, warp, "after world 2's post moved")
    assert mjc.poses()[2, 0, 0] == pytest.approx(XS[2] + 0.1, abs=1e-6) and mjc.poses()[2, 1, 2] < POST_TOP
    for r in (mjc, warp):
        r.s.reset(r.a)                                           # model defaults: the joint frames as they are now
        r.step(150)
    _close(mjc, warp, "after a reset")


def test_kinematic_objects_take_mujoco_warp_and_match_it(monkeypatch) -> None:
    import alloy

    def run(route: str):
        monkeypatch.setenv(CPU_ROUTE_ENV, route)
        s = alloy.Scene(worlds=len(XS), dt=0.01)
        s.add_ground()
        xs = np.stack([np.array(XS), np.zeros(len(XS)), np.full(len(XS), 0.1)], axis=1)
        s.add_box("k", size=(0.05, 0.05, 0.05), pos=xs, kinematic=True)
        s.add_box("free", size=(0.04, 0.04, 0.04), pos=xs + [0.0, 0.0, 0.1], mass=0.1)
        sim = alloy.build(s, device="cpu", seed=0)
        sim.step(5)
        sim.drive("k", velocity=(0.0, 0.0, 0.2), duration=0.1)
        sim.step(15)
        sim.set_pose("k", pos=sim.state(["k"])["k.pos"] + np.array([0.0, 0.05, 0.0]))
        sim.step(5)
        kinds = {type(s).__name__ for s in sim.engine.solvers}
        return kinds, sim.state(["k", "free"])

    kinds, mjc = run("mjc")
    _, warp = run("warp")
    assert "MjcCoreSolver" not in kinds                          # refused: kinematic bodies
    for key in ("k.pos", "free.pos"):
        np.testing.assert_array_equal(mjc[key], warp[key])
