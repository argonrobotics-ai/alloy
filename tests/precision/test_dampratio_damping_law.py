# SPDX-License-Identifier: Apache-2.0
"""Law (Gemini audit S1-018 / S1-021, WP5 2026-11-13): an MJCF ``<position ... dampratio=...>`` servo has its
damping in every world, on the CPU (one MuJoCo C model per world) as on a GPU; Scene.randomize's kd_scale scales it,
a kp draw keeps the ratio (damping goes with sqrt(kp)), and a reset keeps it.

Before: on the CPU only world 0 got the resolved damping (joint_target_kd [1.188, 0, 0, 0]: worlds 1-3 swung
undamped, 0.798 rad from world 0, with health ok), because Newton's CPU path resolves the template world only;
kd_scale had no effect (kd 3.173 at every scale) and a when="reset" gain row zeroed the damping, because the
randomizer took its nominal kd (0) before the solver resolved the ratio.

Every CPU law runs on both CPU routes: ``mjc`` (one MuJoCo C model per world, what users get; S1-018 lived there only)
and ``warp`` (MuJoCo Warp on the CPU, which tests/conftest.py pins for the suite). The first version ran on ``warp``
alone and passed before the S1-018 fix.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy

ARM = """<mujoco model="arm_dampratio"><compiler angle="radian"/><worldbody><body name="link" pos="0 0 0.25">
  <joint name="j" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
  <geom type="capsule" fromto="0 0 0 0.22 0 0" size="0.025" mass="{mass}"/></body></worldbody>
  <actuator><position name="j" joint="j" kp="10" dampratio="2"/></actuator></mujoco>"""
STEPS = 30
SOLVER = {"mjc": "MjcCoreSolver", "warp": "SolverMuJoCo"}      # the rigid solver each CPU route builds


@pytest.fixture(params=sorted(SOLVER))
def route(request, monkeypatch) -> str:
    """Both CPU routes, whatever the conftest pinned."""
    monkeypatch.setenv("ALLOY_CPU_ROUTE", request.param)
    return request.param


def _sim(worlds: int, device: str, mass: float = 0.5, rows=(), seed: int = 1):
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM.format(mass=mass))
    for row in rows:
        s.randomize("arm", *row[:3], **row[3] if len(row) > 3 else {})
    return alloy.build(s, worlds=worlds, seed=seed, device=device)


def _cpu(route: str, worlds: int, **kw):
    """A CPU build on ``route``, checked to step on that route's solver."""
    sim = _sim(worlds, "cpu", **kw)
    solvers = sim.engine.solvers.values() if isinstance(sim.engine.solvers, dict) else sim.engine.solvers
    assert SOLVER[route] in [type(x).__name__ for x in solvers], (route, solvers)
    return sim


def _kd(sim) -> np.ndarray:
    return sim.built.model.joint_target_kd.numpy().astype(np.float64)


def _swing(sim, steps: int = STEPS) -> np.ndarray:
    sim.set_targets({"arm/j": 0.5})
    out = []
    for _ in range(steps):
        sim.step(1)
        out.append(sim.state()["arm/j.q"].copy())
    return np.asarray(out)


@pytest.mark.law
def test_every_world_takes_world_0s_damping_on_the_cpu(route: str) -> None:
    sim = _cpu(route, 4)
    kd = _kd(sim)
    q = _swing(sim)
    sim.close()
    assert kd[0] > 0.0 and np.all(kd == kd[0]), kd                     # was [1.188, 0, 0, 0]
    assert np.all(q == q[:, :1]), np.abs(q - q[:, :1]).max()             # identical worlds step identically
    assert q.max() < 0.52, q.max(0)                                      # damped: undamped worlds reached 1.087 rad


@pytest.mark.law
def test_a_world_with_its_own_mass_gets_the_damping_of_that_mass(route: str) -> None:
    sim = _cpu(route, 3, rows=[("mass_scale", 0.5, 2.0)], seed=3)
    kd, scale = _kd(sim), sim.randomized()["arm.mass_scale"]
    sim.close()
    assert len(set(np.round(scale, 6))) == 3, scale
    for w in range(3):                                                    # as if the file gave that mass
        one = _cpu(route, 1, mass=0.5 * float(scale[w]))
        assert kd[w] == pytest.approx(float(_kd(one)[0]), rel=1e-5), (w, scale[w])
        one.close()


@pytest.mark.law
@pytest.mark.parametrize("scale", [0.25, 2.0])
def test_kd_scale_scales_the_damping(scale: float, route: str) -> None:
    base, scaled = _cpu(route, 1), _cpu(route, 1, rows=[("kd_scale", scale, scale)])
    assert _kd(scaled)[0] == pytest.approx(scale * _kd(base)[0], rel=1e-6)   # was the 1x value at every scale
    assert np.abs(_swing(scaled) - _swing(base)).max() > 1e-3
    base.close(), scaled.close()


@pytest.mark.law
def test_a_kp_draw_keeps_the_damping_ratio(route: str) -> None:
    base, stiff = _cpu(route, 1), _cpu(route, 1, rows=[("kp_scale", 4.0, 4.0)])
    assert _kd(stiff)[0] == pytest.approx(2.0 * _kd(base)[0], rel=1e-6)    # 2 sqrt(4 kp I) = 2 x 2 sqrt(kp I)
    base.close(), stiff.close()


@pytest.mark.law
def test_a_reset_keeps_the_damping(route: str) -> None:
    sim = _cpu(route, 2, rows=[("kp_scale", 1.0, 1.0, {"when": "reset"}), ("kd_scale", 0.5, 0.5, {"when": "reset"})])
    kd0 = _kd(sim)
    first = _swing(sim)
    sim.reset()
    kd1 = _kd(sim)
    second = _swing(sim)
    sim.close()
    assert np.all(kd0 > 0.0) and np.array_equal(kd0, kd1), (kd0, kd1)      # a reset zeroed it
    assert np.array_equal(first, second)


@pytest.mark.gpu
@pytest.mark.law
def test_on_a_gpu_every_world_has_the_cpus_damping_and_kd_scale_scales_it(monkeypatch) -> None:
    monkeypatch.setenv("ALLOY_CPU_ROUTE", "mjc")                          # the CPU reference users get
    cpu, gpu = _cpu("mjc", 4), _sim(4, "cuda:0")
    kd_cpu, kd_gpu = _kd(cpu), _kd(gpu)
    q_gpu = _swing(gpu)
    cpu.close(), gpu.close()
    assert np.all(kd_gpu == kd_gpu[0]) and kd_gpu[0] == pytest.approx(kd_cpu[0], rel=1e-4), (kd_cpu, kd_gpu)
    assert np.all(q_gpu == q_gpu[:, :1]) and q_gpu.max() < 0.52
    half = _sim(2, "cuda:0", rows=[("kd_scale", 0.5, 0.5)])
    assert np.allclose(_kd(half), 0.5 * kd_gpu[:2], rtol=1e-6)
    half.close()
