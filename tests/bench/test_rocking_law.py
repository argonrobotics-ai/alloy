# SPDX-License-Identifier: Apache-2.0
"""Rocking law (release red 2026-11-09, coordinator 10:19Z; RIGID-DEFAULT 16:07Z): a cube pushed across the ground
rocks no more on the default route than under rk4 at the same step, on both CPU routes.

A cube of edge 30 or 5 cm (1000 kg/m3, mu 0.5), alone or beside 8 resting cubes far away (an entry of 9 free bodies:
F6 R1's count), rests 0.5 s, then is pushed at its centre of mass by 1.5 mu m g for 0.4 s (to 0.95 m/s) or 2 s (to
4.6 m/s). Its pitch peak to peak from 0.1 s into the push, and over the late part (from 0.25 s of 0.4 s, the last
third of 2 s: a rocking that keeps growing), stays within 1.1x rk4's + 0.2 deg. Both CPU routes: the suite's MuJoCo
Warp kernels (tests/conftest.py) and MuJoCo C, which alloy.build(device="cpu") steps free bodies on by default, so a
contact-layer fix on one route only shows here.
History (CPU MuJoCo C unless noted): F6 R2 failed it, 1.69 vs 0.33 deg (30 cm, 5 ms) and 4.35 vs 1.45 (5 cm, 1.25 ms);
F6 R1 the 9-body case, 0.53 vs 0.20 over 0.4 s and 19.1 vs 3.9 over 2 s (warp route; 36.4 vs 3.86 over the last third
on MuJoCo C). Full implicit and euler rock as implicitfast does (docs/notes/wp6-results.md, RIGID-DEFAULT row).
And the CPU-vs-GPU law (ruling 2026-11-09 16:32Z): MuJoCo C (rk4 for clutter) and MuJoCo Warp (R1 + the friction cap)
agree on XB4-9's stack and the pushed cube within the bounds its test states."""

from __future__ import annotations

import os

import numpy as np
import pytest

from alloy.route.mjc_core_config import CPU_ROUTE_ENV
from alloy.scene import integrator_route as IR

CASES = [(s, e, p) for p in (0.4, 2.0) for s in (0.3, 0.05) for e in (0, 8)]


def _rock(size: float, rk4: bool, monkeypatch, extra: int, push: float) -> tuple[float, float, float, float]:
    """(rigid step, pitch p-p from 0.1 s into the push, late p-p, slide speed over the last 0.1 s) of the pushed
    cube, built as alloy.build does (rk4 forced on the entry when ``rk4``)."""
    import alloy

    with monkeypatch.context() as m:
        if rk4:
            m.setattr(IR, "_clutter_or_rk4", lambda b, mj, dt, *_: IR.IntegratorChoice(True, "forced rk4 (test)"))
        scene = alloy.Scene(dt=0.01)
        scene.add_ground(friction=0.5)
        scene.add_box("blk", size=(size,) * 3, pos=(0.0, 0.0, 0.5 * size + 1e-4), density=1000.0, friction=0.5)
        for i in range(extra):          # resting cubes far away: a clutter-sized entry (R1's count)
            scene.add_box(f"rest{i}", size=(size,) * 3, pos=(5.0 + 2.0 * i, 5.0, 0.5 * size + 1e-4), density=1000.0)
        sim = alloy.build(scene, device="cpu")
    on_c = type(sim.engine.stepper.solvers()[0]).__name__ == "MjcCoreSolver"
    assert on_c == (os.environ[CPU_ROUTE_ENV] == "mjc"), type(sim.engine.stepper.solvers()[0])
    sim.run(0.5)
    sim.apply_wrench("blk", force=(0.75 * 1000.0 * size ** 3 * 9.81, 0.0, 0.0))
    tr = sim.run(push, record=["blk"]).world(0)
    assert sim.check()["ok"], sim.check()["problems"]
    t, (w, x, y, z) = tr["t"] - tr["t"][0], tr["blk.quat_wxyz"].T
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    late, xy = t >= push - max(0.15, push / 3), tr["blk.pos"][:, :2]
    v = float(np.linalg.norm(xy[-1] - xy[-11]) / (t[-1] - t[-11]))
    return sim.built.report.rigid_step.dt, float(np.ptp(pitch[t >= 0.1])), float(np.ptp(pitch[late])), v


@pytest.mark.law
@pytest.mark.timeout(300)       # a 2 s push at 1.25 ms (5 cm) takes 44-52 s on the warp route on a loaded host
@pytest.mark.parametrize("route", ["warp", "mjc"])
@pytest.mark.parametrize("size, extra, push", CASES)
def test_a_pushed_cube_rocks_no_more_than_under_rk4(size, extra, push, route, monkeypatch):
    monkeypatch.setenv(CPU_ROUTE_ENV, route)
    dt, pp, late, _ = _rock(size, False, monkeypatch, extra, push)
    dt_rk4, pp_rk4, late_rk4, _ = _rock(size, True, monkeypatch, extra, push)
    assert dt == dt_rk4
    assert pp <= 1.1 * pp_rk4 + 0.2 and late <= 1.1 * late_rk4 + 0.2, (route, size, dt, pp, pp_rk4, late, late_rk4)


def _stack_settle(route: str, monkeypatch, worlds: int = 2) -> tuple[float, int, float]:
    """XB4-9's resting stacks through the API on ``route``: 9 columns of 3 one-metre cubes (10 kg/m3, friction 1),
    2 m apart, each cube jittered by up to 1 cm and 10 deg of yaw over per-layer gaps up to 5 mm (seeded per world),
    3.3 s. Returns (worst settle error, m: distance from the ideal rest spot; falls past 0.25 m; rigid step)."""
    import alloy

    monkeypatch.setenv(CPU_ROUTE_ENV, route)
    rng = [np.random.default_rng(9000 + k) for k in range(worlds)]
    grid = np.array([((i - 1) * 2.0, (j - 1) * 2.0, k) for i in range(3) for j in range(3) for k in range(3)])
    pos = np.repeat(grid[None], worlds, axis=0)
    yaw = np.zeros((worlds, 27))
    for w, g in enumerate(rng):
        pos[w, :, :2] += g.uniform(-0.01, 0.01, (27, 2))
        pos[w, :, 2] = 0.5 + grid[:, 2] + g.uniform(0.0, 0.005, (9, 3)).cumsum(axis=1).reshape(-1)
        yaw[w] = np.radians(g.uniform(-10.0, 10.0, 27))
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground(friction=1.0)
    for c in range(27):
        scene.add_box(f"c{c}", size=(1.0,) * 3, pos=pos[:, c], density=10.0, friction=1.0)
    sim = alloy.build(scene, device="cpu")
    for c in range(27):             # XB4-9's yaw, as its harness sets it (box_gpu.py)
        q = np.stack([np.cos(yaw[:, c] / 2), np.zeros(worlds), np.zeros(worlds), np.sin(yaw[:, c] / 2)], axis=1)
        sim.set_pose(f"c{c}", pos=pos[:, c], quat_wxyz=q)
    sim.run(3.3)
    assert sim.check()["ok"], sim.check()["problems"]
    st = sim.state([f"c{c}" for c in range(27)])
    ideal = pos.copy()
    ideal[..., 2] = 0.5 + grid[:, 2]
    err = np.linalg.norm(np.stack([st[f"c{c}.pos"] for c in range(27)], axis=1) - ideal, axis=-1)
    return float(err.max()), int((err > 0.25).sum()), sim.built.report.rigid_step.dt


@pytest.mark.law
@pytest.mark.timeout(300)
def test_the_cpu_routes_agree_on_the_stack_and_the_pushed_cube(monkeypatch):
    """CPU-vs-GPU law (coordinator ruling 2026-11-09 16:32Z): MuJoCo C (rk4 for clutter) and MuJoCo Warp (the GPU's
    kernels: F6 R1, implicitfast with the friction-pull cap) give the same physics within stated bounds. XB4-9's
    stacks settle within 0.2 mm of each other and under 1 mm, none falls (rows: 0.22 / 0.22 mm at 2.5 / 5 ms); a 30 cm
    cube pushed beside 8 slides at the same Coulomb rate (speed after 2 s within 3 %: 4.96 / 4.86 m/s), its pitch
    within 0.25 deg over 0.4 s (0.20 / 0.32) and no worse on Warp than on C over 2 s (3.86 / 0.48 late)."""
    settle = {r: _stack_settle(r, monkeypatch) for r in ("mjc", "warp")}
    assert all(s[1] == 0 and s[0] < 1e-3 for s in settle.values()), settle
    assert abs(settle["mjc"][0] - settle["warp"][0]) <= 2e-4, settle
    rows = {}
    for r in ("mjc", "warp"):
        monkeypatch.setenv(CPU_ROUTE_ENV, r)
        rows[r] = _rock(0.3, False, monkeypatch, 8, 0.4)[1], _rock(0.3, False, monkeypatch, 8, 2.0)
    (c04, (_, c_pp, c_late, c_v)), (w04, (_, w_pp, w_late, w_v)) = rows["mjc"], rows["warp"]
    assert abs(w04 - c04) <= 0.25 and abs(w_v - c_v) <= 0.03 * c_v, rows
    assert w_pp <= 1.1 * c_pp + 0.2 and w_late <= 1.1 * c_late + 0.2, rows


def _rock_gpu(size: float, rk4: bool, monkeypatch, extra: int, push: float) -> tuple[float, float, float, float, bool]:
    """``_rock``'s case on cuda:0 (MuJoCo Warp: F6 R1 and the friction-pull cap where R1 applies): (rigid step, pitch
    p-p from 0.1 s into the push, late p-p, rise of the cube's centre in m, whether R1 runs it)."""
    import alloy

    with monkeypatch.context() as m:
        if rk4:
            m.setattr(IR, "_clutter_or_rk4", lambda b, mj, dt, *_: IR.IntegratorChoice(True, "forced rk4 (test)"))
        scene = alloy.Scene(dt=0.01)
        scene.add_ground(friction=0.5)
        scene.add_box("blk", size=(size,) * 3, pos=(0.0, 0.0, 0.5 * size + 1e-4), density=1000.0, friction=0.5)
        for i in range(extra):
            scene.add_box(f"rest{i}", size=(size,) * 3, pos=(5.0 + 2.0 * i, 5.0, 0.5 * size + 1e-4), density=1000.0)
        sim = alloy.build(scene, device="cuda:0")
    r1 = any("integrator implicitfast" in r for r in sim.routes()["blk"]["reasons"])
    sim.run(0.5)
    z0 = float(sim.state(world=0)["blk.pos"][2])
    sim.apply_wrench("blk", force=(0.75 * 1000.0 * size ** 3 * 9.81, 0.0, 0.0))
    tr = sim.run(push, record=["blk"]).world(0)
    assert sim.check()["ok"], sim.check()["problems"]
    t, (w, x, y, z) = tr["t"] - tr["t"][0], tr["blk.quat_wxyz"].T
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    late = t >= push - max(0.15, push / 3)
    rise = float(tr["blk.pos"][:, 2].max()) - z0
    return sim.built.report.rigid_step.dt, float(np.ptp(pitch[t >= 0.1])), float(np.ptp(pitch[late])), rise, r1


@pytest.mark.gpu
@pytest.mark.law
@pytest.mark.timeout(900)       # the first case compiles MuJoCo Warp's kernels on a cold cache (60 s default)
@pytest.mark.parametrize("size, extra, push", CASES)
def test_the_pushed_cube_law_on_the_gpu(size, extra, push, monkeypatch):
    """The law on cuda:0 (RIGID-DEFAULT, coordinator 2026-11-09, after the official rows; WP7). Where the default route
    differs from its rk4 twin (F6 R1 with the cap: a cube beside 8 others), the CPU's strict comparison. A lone cube
    keeps rk4, so its twin is the same route; the 5 cm cube's 2 s push then tumbles it past 100 deg and two rk4 runs
    part on CUDA's contact order (134 against 103 deg, L4, main 04f81f0ff): that case bounds the outcome instead, a
    valid world and a rise under 0.1 m (CPU: 160 deg, 23 mm). The other lone cases repeat to the digit, strict."""
    dt, pp, late, rise, r1 = _rock_gpu(size, False, monkeypatch, extra, push)
    assert r1 == (extra > 0), (size, extra)
    if not r1 and size == 0.05 and push == 2.0:     # the same route both ways through a tumble: a bound, not a match
        assert rise < 0.1, (size, push, rise)
        return
    dt_rk4, pp_rk4, late_rk4, _, _ = _rock_gpu(size, True, monkeypatch, extra, push)
    assert dt == dt_rk4
    assert pp <= 1.1 * pp_rk4 + 0.2 and late <= 1.1 * late_rk4 + 0.2, (size, extra, push, pp, pp_rk4, late, late_rk4)
