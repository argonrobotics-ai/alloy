# SPDX-License-Identifier: Apache-2.0
"""Force plug-ins' work is external work in health's energy check (fresh-agent friction F4, 2026-11-07; CPU).

Before: a 10 m/s wind (quadratic drag plug-in) on a 10 g ball read as ENERGY_JUMP at t = 0.34 s at every dt and froze
the world (E_scale ~ 1 mJ), on the MuJoCo C and the warp CPU routes. The host now measures the forces' work
(``alloy.precision.health_work.ForceWork``) and health books it every step; an energy gain the forces do not explain
still trips ENERGY_JUMP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.plugins.base import ForcePlugin
from alloy.precision.health_kernels import EnergyRef
from alloy.precision.health_work import WORK_ATTR, ForceWork, book_work, register_work

M, R = 0.01, 0.015


def test_force_work_is_force_times_com_displacement_plus_torque_power_per_world() -> None:
    w = newton.ModelBuilder()
    w.add_body(xform=wp.transform_identity(), mass=1.0, com=wp.vec3(0.1, 0.0, 0.0))
    w.add_particle(pos=wp.vec3(1.0, 0.0, 0.0), vel=wp.vec3(), mass=0.1)
    s = newton.ModelBuilder()
    s.replicate(w, 2)
    model = s.finalize(device="cpu")
    s0, s1 = model.state(), model.state()
    turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5 * np.pi)
    q1 = s1.body_q.numpy()
    q1[0] = [0.5, 0.0, 0.0, *turn]                       # world 0's body: moved 0.5 m in x, turned 90 deg about z
    s1.body_q.assign(q1)
    s0.body_qd.assign(np.array([[0, 0, 0, 0, 0, 1.0], [0] * 6], np.float32))
    s1.body_qd.assign(np.array([[0, 0, 0, 0, 0, 3.0], [0] * 6], np.float32))
    x1 = s1.particle_q.numpy()
    x1[1] += [0.0, 2.0, 0.0]                             # world 1's particle: moved 2 m in y
    s1.particle_q.assign(x1)
    body_f = wp.array(np.array([[2.0, 1.0, 0, 0, 0, 3.0], [0] * 6], np.float32), dtype=wp.spatial_vector, device="cpu")
    particle_f = wp.array(np.array([[0, 0, 0], [0, 1.0, 0]], np.float32), dtype=wp.vec3, device="cpu")
    fw = ForceWork(model)
    fw.begin(s0)
    out = fw.end(body_f, particle_f, s1, 0.1).numpy()
    # COM (0.1, 0, 0) -> (0.5, 0.1, 0): 2 * 0.4 + 1 * 0.1; torque 3 x mean spin 2 rad/s x 0.1 s; particle 1 N x 2 m
    np.testing.assert_allclose(out, [0.8 + 0.1 + 0.6, 2.0], rtol=1e-6)
    register_work(model, fw.out)                           # ForceWork registered itself: not twice
    assert len(getattr(model, WORK_ATTR)) == 1 and getattr(model, WORK_ATTR)[0] is fw.out
    ref = EnergyRef()
    for name in ("z_ref", "e0", "e_max", "w_cum", "e_scale"):
        setattr(ref, name, wp.zeros(2, dtype=wp.float64, device="cpu"))
    book_work(model, ref)
    book_work(model, ref)                                  # booking uses the step's work up: not counted twice
    np.testing.assert_allclose(ref.w_cum.numpy(), [1.5, 2.0], rtol=1e-6)
    assert not fw.out.numpy().any()


def test_force_work_on_a_particles_only_model() -> None:
    w = newton.ModelBuilder()
    w.add_particle(pos=wp.vec3(), vel=wp.vec3(), mass=0.1)
    w.add_particle(pos=wp.vec3(1.0, 0.0, 0.0), vel=wp.vec3(), mass=0.1)
    s = newton.ModelBuilder()
    s.add_world(w)                                         # world 0 (global items count in no world's energy)
    model = s.finalize(device="cpu")
    s0, s1 = model.state(), model.state()
    s1.particle_q.assign(s0.particle_q.numpy() + [0.0, 0.0, -0.5])
    fw = ForceWork(model)
    fw.begin(s0)
    f = wp.array(np.array([[0, 0, -2.0], [0, 0, -4.0]], np.float32), dtype=wp.vec3, device="cpu")
    np.testing.assert_allclose(fw.end(None, f, s1, 0.01).numpy(), [3.0], rtol=1e-6)   # 2 x 0.5 + 4 x 0.5


@wp.kernel
def _wind(bodies: wp.array(dtype=wp.int32), body_qd: wp.array(dtype=wp.spatial_vector), wind: wp.vec3, k: float,
          body_f: wp.array(dtype=wp.spatial_vector)):
    w = wp.tid()
    b = bodies[w]
    if b < 0:
        return
    qd = body_qd[b]
    rel = wind - wp.vec3(qd[0], qd[1], qd[2])
    f = k * wp.length(rel) * rel
    body_f[b] = body_f[b] + wp.spatial_vector(f[0], f[1], f[2], 0.0, 0.0, 0.0)


class Wind(ForcePlugin):
    """Quadratic air drag towards a 10 m/s wind (F4's case)."""
    name = "test_wind_f4"

    @dataclass
    class Config:
        body: str = "ball"
        speed: float = 10.0
        k: float = 0.5 * 1.2 * 0.47 * np.pi * R ** 2          # 1/2 rho Cd A

    def setup(self, ctx):
        self.bodies = ctx.bodies(self.config.body)

    def apply(self, state, forces, dt):
        wp.launch(_wind, dim=self.ctx.world_count, inputs=[self.bodies, state.body_qd, wp.vec3(self.config.speed, 0, 0),
                  float(self.config.k), forces.body_f], device=self.ctx.model.device)


def _ball_in_wind(dt: float = 0.005):
    import alloy

    scene = alloy.Scene(dt=dt)
    scene.add_ground()
    scene.add_sphere("ball", R, pos=(0, 0, R), mass=M)
    return alloy.build(scene, device="cpu", plugins=[Wind()])


@pytest.mark.parametrize("route", ["mjc", "warp"])      # the users' CPU default (MuJoCo C) and the suite's (OSS-14)
def test_a_wind_plugin_on_a_light_ball_is_booked_work_not_an_energy_jump(route, monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    sim = _ball_in_wind()
    sim.run(1.0)
    eng = sim.engine
    assert sim.check()["ok"] and sim.health()["worlds"][0]["severity"] == "OK"
    assert eng.state.body_q.numpy()[0][0] > 0.5                       # rolled downwind (frozen at 0.08 m before)
    qd = eng.state.body_qd.numpy()[0]
    ke = 0.5 * M * qd[:3] @ qd[:3] + 0.5 * (0.4 * M * R * R) * qd[3:] @ qd[3:]
    assert float(eng.health._ref.w_cum.numpy()[0]) == pytest.approx(ke, rel=1e-3)   # all of the gain is the wind's


@pytest.mark.parametrize("route", ["mjc", "warp"])
def test_an_unexplained_gain_still_trips_energy_jump_beside_a_force_plugin(route, monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    sim = _ball_in_wind()
    sim.run(0.3)
    jqd = sim.engine.state.joint_qd.numpy()
    jqd[2] += 2.0                                                      # a 2 m/s upward kick nothing explains
    sim.engine.state.joint_qd.assign(jqd)
    sim.run(0.05)
    world = sim.health()["worlds"][0]
    assert world["severity"] == "INVALID" and "ENERGY_JUMP" in world["flags"]


def _particle_worlds(sizes: list[int]):
    s = newton.ModelBuilder()
    for n in sizes:
        w = newton.ModelBuilder()
        for k in range(n):
            w.add_particle(pos=wp.vec3(0.01 * k, 0.0, 0.0), vel=wp.vec3(), mass=0.1)
        s.add_world(w)
    return s.finalize(device="cpu")


def _work(model, sizes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """ForceWork on particles whose force and displacement depend only on their index k inside their own world."""
    k = np.concatenate([np.arange(n) for n in sizes]).astype(np.float64)[:, None]
    dx = (1e-3 * np.cos(k + np.array([0.0, 1.0, 2.0]))).astype(np.float32)
    f = np.sin(0.7 * k + np.array([0.3, 0.6, 0.9])).astype(np.float32)
    s0, s1 = model.state(), model.state()
    s1.particle_q.assign(s0.particle_q.numpy() + dx)
    fw = ForceWork(model)
    fw.begin(s0)
    out = fw.end(None, wp.array(f, dtype=wp.vec3, device="cpu"), s1, 0.01).numpy().copy()
    moved = (s1.particle_q.numpy() - s0.particle_q.numpy()).astype(np.float32)
    return out, (f * moved).sum(axis=1).astype(np.float64)


def test_force_work_sums_in_fixed_chunks_past_one_chunk() -> None:
    from alloy.precision.chunk_sums import CHUNK

    sizes = [2 * CHUNK + 44, 5]                                 # world 0: three chunks; world 1: one
    out, per = _work(_particle_worlds(sizes), sizes)
    np.testing.assert_allclose(out, [per[:sizes[0]].sum(), per[sizes[0]:].sum()], rtol=1e-9)
    assert np.array_equal(out, _work(_particle_worlds(sizes), sizes)[0])     # the same bits every run
    alone, _ = _work(_particle_worlds([5]), [5])
    assert alone[0] == out[1]                                   # batch-invariant: world 1 alone gives the same bits
