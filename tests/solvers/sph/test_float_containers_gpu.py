# SPDX-License-Identifier: Apache-2.0
"""WP9 (a1) floating containers on the GPU, through the public API (alloy.Scene -> MuJoCo + SPH proxy, cuda:0):
L-a1 the draft of floating open cups across D5 ratios 3-50 (16 worlds: the GPU smoke), L-a2 Froude similarity of
the release, L-a3 slosh in a floating cup with the horizontal momentum closed against the tank's wall meters, L-a4
thin walls (h/4, h/8) that do not leak. r = 2 mm (h = 8 mm). Each prints what it measured."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy._vendor import newton

import alloy
from alloy.coupling.types import InterfaceTolerance
from tests.solvers.sph.float_scenes import Float

DEV = "cuda:0"


def _sph(sim):
    return next(s for s in sim.engine.solvers if type(s).__name__ == "SolverDFSPH")


def _bottom(f: Float, traj) -> np.ndarray:
    """(frames, worlds) the cup's bottom relative to the still surface [m]."""
    return traj["cup.pos"][..., 2] - f.surface


def _inside(f: Float, sim, name: str = "water") -> np.ndarray:
    """Per world, liquid particles of ``name`` inside the cup's cavity (in the cup's frame)."""
    st = sim.state([name, "cup"])
    s = f.scale
    half, t, H = (0.5 * f.L - f.wall) * s, f.wall * s, f.H * s
    out = []
    for w in range(sim.worlds):
        q = st["cup.quat_xyzw"][w]
        x, y, z, qw = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * qw), 2 * (x * z + y * qw)],
                      [2 * (x * y + z * qw), 1 - 2 * (x * x + z * z), 2 * (y * z - x * qw)],
                      [2 * (x * z - y * qw), 2 * (y * z + x * qw), 1 - 2 * (x * x + y * y)]])
        p = st[f"{name}.points"][w][st[f"{name}.active"][w]]
        rel = (p - st["cup.pos"][w]) @ R
        out.append(int(((np.abs(rel[:, 0]) < half) & (np.abs(rel[:, 1]) < half) & (rel[:, 2] > t)
                        & (rel[:, 2] < H)).sum()))
    return np.array(out)


def _cup_masses(sim) -> np.ndarray:
    """(worlds,) the cup body's mass in each world (its density is drawn per world)."""
    rows = sim.index.per_world(sim.index.bodies["cup"], "body")
    return sim.engine.model.body_mass.numpy()[rows].astype(np.float64)


@pytest.mark.gpu
@pytest.mark.timeout(2400)
def test_la1_floating_cups_float_at_their_archimedes_draft_in_16_worlds():
    """L-a1 (and the <= 64-world GPU smoke): open cups 60 x 60 x 50 mm, 4 mm walls, flagged containers, one density
    per world for D5 ratios 3-50, released 1 mm over still water in a 140 mm tank. Over the last 0.5 s of 1.5 s each
    cup's bottom is the tank's rise minus its draft m/(ρ·L²) within 0.1 spacing (0.4 mm; drafts 0.8-16.6 mm), it
    stays upright (< 5°), no water enters any cup, and every world is healthy."""
    f = Float(10.0)
    sim = alloy.build(f.scene(worlds=16, ratios=(3.0, 50.0)), device=DEV, seed=1)
    traj = sim.run(1.5, record=["cup"], every=5)
    cups = _cup_masses(sim)
    z = _bottom(f, traj)[-10:].mean(axis=0)                            # frames every 0.05 s: the last 0.5 s
    want = f.bottom_expected(cups)
    q = traj["cup.quat_wxyz"][..., 0]
    tilt = np.degrees(2.0 * np.arccos(np.clip(np.abs(q), 0.0, 1.0))).max()
    ins = _inside(f, sim)
    ratio = 998.2 * f.L * f.L * f.H / cups
    for w in np.argsort(ratio):
        print(f"ratio {ratio[w]:5.1f}: bottom {1e3 * z[w]:+7.3f} mm, want {1e3 * want[w]:+7.3f}, "
              f"err {1e3 * (z[w] - want[w]):+.3f} mm")
    print(f"tilt max {tilt:.2f} deg, water in cups {ins.tolist()}, check {sim.check()['ok']}")
    assert ratio.min() >= 3.0 * (1 - 1e-6) and ratio.max() <= 50.0 * (1 + 1e-6) and np.ptp(ratio) > 20.0
    assert np.abs(z - want).max() <= 0.4e-3
    assert tilt < 5.0 and (ins == 0).all() and sim.check()["ok"]


def _first_extrema(t: np.ndarray, z: np.ndarray) -> tuple[float, float, float, float]:
    """(t, z) of the release's first trough and the crest after it."""
    i = int(np.argmin(z[: len(z) // 2]))
    j = i + int(np.argmax(z[i: i + len(z) // 3]))
    return float(t[i]), float(z[i]), float(t[j]), float(z[j])


@pytest.mark.gpu
@pytest.mark.timeout(2400)
def test_la2_a_release_scales_by_froude_similarity():
    """L-a2 heave: a cup at D5 ratio 6 released 1 mm over the water, and the same scene with every length doubled
    (mass x8; the particles stay 2 mm). Gravity sets the time scale, so the doubled release runs sqrt(2) slower
    whatever its added mass: its first trough and crest come sqrt(2) later (5 %). The trough's depth is printed,
    not asserted: viscosity and the 2 mm particles do not scale, so the larger cup's overshoot is less damped
    (measured 2.20x, not 2x). Its settled draft is L-a1's."""
    out = []
    for s in (1.0, 2.0):
        f = Float(6.0, scale=s)
        sim = alloy.build(f.scene(), device=DEV)
        traj = sim.run(0.6 * math.sqrt(s), record=["cup"], every=1)
        z = _bottom(f, traj)[:, 0]
        out.append(_first_extrema(traj["t"], z))
        print(f"scale {s}: trough {out[-1][1] * 1e3:+.2f} mm at {out[-1][0]:.3f} s, crest {out[-1][3] * 1e3:+.2f} mm "
              f"at {out[-1][2]:.3f} s; check {sim.check()['ok']}")
        assert sim.check()["ok"]
    a, b = out
    r2 = math.sqrt(2.0)
    print(f"time ratios {b[0] / a[0]:.3f} (trough), {b[2] / a[2]:.3f} (crest), sqrt 2 = {r2:.3f}; depth ratio "
          f"{b[1] / a[1]:.3f}")
    assert b[0] / a[0] == pytest.approx(r2, rel=0.05, abs=0.0) and b[2] / a[2] == pytest.approx(r2, rel=0.05, abs=0.0)


def _tank_shapes(sim) -> list[int]:
    labels = list(sim.engine.model.shape_label)
    return [i for i, n in enumerate(labels) if "tank" in str(n)]


@pytest.mark.gpu
@pytest.mark.timeout(1800)
def test_la3_slosh_in_a_floating_cup_keeps_momentum_closed():
    """L-a3: a cup (D5 ratio 6) holding 15 mm of water, released tilted 0.08 rad over the tank's water: it rights
    itself and rocks, and its water sloshes. Horizontal momentum, every control step (0.01 s) over 1 s:
    - the system: cup + all water changes only by the tank walls' push on the water (SolverDFSPH's meters, the pass
      that sticks), within 5e-3 of the walls' gross impulse (their hydrostatic pushes cancel pairwise; the pressure
      is solved to max_density_error 1e-3);
    - the hand-over: the cup's momentum (MuJoCo's) is the liquid's impulse on it (SPH's meter on its shape) within
      the staggered pair's own gate summed over the run: each exchange stops once the force mismatch is under
      max(force_rel·|F|, force_floor) (InterfaceTolerance: 1 %, 1e-3 N). Here the horizontal forces are a few mN,
      at that floor, so the gate (1e-3 N·s over 1 s) is loose: measured 9.9e-5 N·s, built up while the cup rocks
      and then constant (printed);
    - the inner water stays in the cup, and the world is healthy."""
    f = Float(6.0)
    sim = alloy.build(f.scene(inner=0.015, rpy=(0.08, 0.0, 0.0)), device=DEV)
    sph = _sph(sim)
    sph.reset_shape_meters()
    labels = [str(n) for n in sph.model.shape_label]
    tanks, cup = [i for i, n in enumerate(labels) if "tank" in n], labels.index("cup")
    mdl = sim.engine.model
    m_p = mdl.particle_mass.numpy().astype(np.float64)
    m_cup = float(_cup_masses(sim)[0])
    rows = []                                                           # per control step: p_cup, J_cup, P + J_walls
    gross = 0.0
    for _ in range(100):
        sim.step(1)
        qd = sim.engine.state.particle_qd.numpy().astype(np.float64)
        act = (mdl.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
        p_liq = (m_p[act, None] * qd[act]).sum(axis=0)[:2]
        p_cup = m_cup * sim.state(["cup"])["cup.vel"][0][:2].astype(np.float64)
        J = sph.shape_impulse()
        rows.append((p_cup, J[cup, :2].copy(), p_liq + p_cup + J[tanks, :2].sum(axis=0)))
        gross = max(gross, float(np.abs(J[tanks, :2]).sum()))
    p = np.array([r[0] for r in rows])
    jc = np.array([r[1] for r in rows])
    system = float(np.abs(np.array([r[2] for r in rows])).max())
    p_max = float(np.abs(p).max())
    same = float(np.abs(p - jc).max())                                  # the cup's momentum vs SPH's impulse on it
    step = float(np.abs(np.diff(p, axis=0)).max())                      # the most it changes in one step
    drift = float(np.abs(p[-1] - jc[-1]).max())
    tol = InterfaceTolerance()                   # the staggered pair's per-exchange force gate, summed over the run
    dj = np.linalg.norm(np.diff(np.vstack([np.zeros((1, 2)), jc]), axis=0), axis=1)
    gate = float(tol.force_floor * 1.0 + tol.force_rel * dj.sum())
    res = np.abs(p - jc).max(axis=1)
    print("residual |p_cup - J_cup| per control step (first 12):", np.array2string(res[:12], precision=3))
    ins = _inside(f, sim, "inner")
    n_in = len(sim.state(["inner"])["inner.points"][0])
    print(f"cup: max |p_cup - J_cup| {same:.3e} N·s ({same / gate:.2f} of the coupler's gate {gate:.3e}), at the end "
          f"{drift:.3e}; max |p_cup| {p_max:.3e}, max change per step {step:.3e}; system: max |P + J_walls| "
          f"{system:.3e} N·s of the walls' gross {gross:.3e} ({system / gross:.2e}); inner water in the cup "
          f"{ins.tolist()} of {n_in}; check {sim.check()['ok']}")
    assert p_max > 1.0e-6 and same <= gate
    assert system <= 5.0e-3 * gross
    assert ins[0] == n_in and sim.check()["ok"]


@pytest.mark.gpu
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("wall", [2.0e-3, 1.0e-3])
def test_la4_thin_walls_do_not_leak(wall):
    """L-a4: cups with walls of h/4 and h/8 (2 and 1 mm at h = 8 mm), D5 ratio 10, released tilted 0.2 rad so the
    water slaps their walls: over 2 s no water enters, the cup settles to its Archimedes bottom (0.1 spacing) and
    the world is healthy."""
    f = Float(10.0, wall=wall)
    sim = alloy.build(f.scene(rpy=(0.2, 0.0, 0.0)), device=DEV)
    traj = sim.run(2.0, record=["cup"], every=5)
    z = _bottom(f, traj)[-10:, 0].mean()
    ins = _inside(f, sim)
    print(f"wall {wall * 1e3:.0f} mm: water in the cup {ins.tolist()}, bottom {z * 1e3:+.3f} mm, want "
          f"{f.bottom_expected() * 1e3:+.3f}; check {sim.check()['ok']}")
    assert (ins == 0).all() and abs(z - f.bottom_expected()) <= 0.4e-3 and sim.check()["ok"]
