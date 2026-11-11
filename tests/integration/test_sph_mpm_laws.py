# SPDX-License-Identifier: Apache-2.0
"""SPH liquid with an MPM paste: the laws of the coupling (SCR-WP12-MIX), closed form, small scenes on the CPU.

The MPM particles are a particle wall of the liquid (``alloy.solvers.sph.particle_walls``) and the liquid's force
reaches them (``alloy.integration.mpm_proxy``). Test pastes (cards written here, high yield so a block stays whole):

- L1 Archimedes: a wide slab (2.4 x 2.4 x 1.2 cm, rho_s 600, stable upright) dropped on water floats at the draft
  d = (rho_s / rho_f) H, the surface placed by the liquid's own volume; at rest the liquid carries its weight.
- L2 load on a bed: water poured on a paste bed that spans the tank (no liquid under it) presses it down with the
  whole column's weight, rho_f g V.
- L4 momentum across the interface: through the slab's drop, the liquid's and the slab's momentum change by
  gravity's impulse minus what the tank's walls took, to 2 % of the slab's own weight impulse (0.65 % measured at
  0.3 s: the tank walls' projection and the paste's own contacts are outside the ledger). The interface alone is
  exact to float32 in test_sph_mpm_walls.py (a standalone liquid on particle walls).
- L6 no leak, books: no liquid inside the paste, the liquid's mass ledger exact, health ok.
- L7 determinism: a repeat is bitwise; a world steps the same alone or beside another.

At 4 mm liquid spacing the draft (7.2 mm) is under two particles: it read 7.73 mm, 7 % deep (an effective inset
of the paste's face of about 0.2 mm), so L1 holds to 10 % here; the 3 cm cube on one L4 ([internal note]
notes/wp12-results/mix/spike.md) gave the submerged fraction to 1.5 %. Stokes settling (L3) needs the
fine liquid setting and runs on the GPU (notes/wp12-results/mix).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import alloy
from alloy.integration.stepper import entry_solvers

RHO_F = 997.05
W_TANK, FLOOR = 0.05, 0.01               # inner width, floor top
CARD = """schema: alloy.material/1
name: {name}
density: {{value: {rho}, source: model, note: 'TEST paste (SCR-WP12-MIX laws)'}}
surface:
  mu_static: {{value: 0.0, source: model, note: 'TEST paste'}}
  mu_dynamic: {{value: 0.0, source: model, note: 'TEST paste'}}
  restitution: {{value: 0.0, source: model, note: 'TEST paste'}}
plastic:
  yield_stress: {{value: 5000.0, source: model, note: 'TEST: high yield keeps the block whole'}}
fluid:
  viscosity: {{value: 2.0, source: model, note: 'TEST: Bingham plastic viscosity'}}
"""


@pytest.fixture(scope="module")
def cards(tmp_path_factory) -> str:
    root = tmp_path_factory.mktemp("mix_cards")
    (root / "materials").mkdir()
    for name, rho in (("test_paste_600", 600.0), ("test_paste_1500", 1500.0)):
        (root / "materials" / f"{name}.yaml").write_text(CARD.format(name=name, rho=rho))
    return str(root)


def _tank(scene: alloy.Scene) -> None:
    scene.add_ground()
    scene.add_open_box("tank", inner=(W_TANK, W_TANK, 0.05), wall=0.01, pos=(0, 0, 0), static=True)


def _parts(sim):
    """(SPH solver, liquid particle ids, wall particle ids, particle masses) of the built scene."""
    sph = entry_solvers(sim.engine.parts.stepper)["sph"]
    liquid = np.flatnonzero(sph.layout.phase.numpy() >= 1)
    return sph, liquid, sph.pw.vid.numpy().astype(np.int64), sim.engine.model.particle_mass.numpy().astype(np.float64)


def _momentum(sim, ids: np.ndarray, mass: np.ndarray) -> np.ndarray:
    qd = sim.engine.state.particle_qd.numpy().astype(np.float64)
    return (mass[ids, None] * qd[ids]).sum(axis=0)


def _floating_slab(cards: str, worlds: int = 1, pos=(0.0, 0.0, FLOOR + 0.027)) -> alloy.Scene:
    s = alloy.Scene(dt=0.01, worlds=worlds, roots=[cards])
    _tank(s)
    s.add_liquid("water", size=(W_TANK, W_TANK, 0.02), pos=(0, 0, FLOOR + 0.01), material="water")
    s.add_box("slab", size=(0.024, 0.024, 0.012), pos=pos, material="test_paste_600")     # 1 mm above the water
    return s


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_a_light_slab_floats_at_its_draft_and_the_interface_keeps_momentum(cards: str) -> None:
    """L1, L4, L6 in one run (about 1 s simulated)."""
    sim = alloy.build(_floating_slab(cards), device="cpu", seed=0)
    sph, liquid, wall, mass = _parts(sim)
    sph.reset_shape_meters()
    m_f, m_s = mass[liquid].sum(), mass[wall].sum()
    p0 = _momentum(sim, liquid, mass) + _momentum(sim, wall, mass)
    sim.step(30)                                                    # the drop and the first bob (L4)
    t = sim.t
    dp = _momentum(sim, liquid, mass) + _momentum(sim, wall, mass) - p0
    expect = (m_f + m_s) * np.array([0.0, 0.0, -9.81]) * t - sph.shape_impulse().sum(axis=0)
    assert np.abs(dp - expect).max() < 2.0e-2 * m_s * 9.81 * t, (dp, expect)   # the slab's weight is 8 % of all
    drafts, forces = [], []
    a_t, a_b, v_w = W_TANK**2, 0.024**2, sim.liquid_volume("water")
    r_m = float(sim.engine.model.particle_radius.numpy()[wall].max())      # half the MPM spacing (0.75 mm)
    while sim.t < 1.0 - 1e-9:
        sim.step(5)
        if sim.t >= 0.7 - 1e-9:
            b = sim.state(world=0)["slab.points"]
            z_b = float(b[:, 2].min()) - r_m                         # bottom face: half the MPM spacing below
            z_s = (v_w + FLOOR * a_t - a_b * z_b) / (a_t - a_b)       # the surface from the liquid's own volume
            drafts.append(z_s - z_b)
            forces.append(float(sph.pw.particle_impulse().sum(axis=0)[2]) / float(sim.engine.parts.stepper._dt_x))
    ext = b.max(axis=0) - b.min(axis=0)
    np.testing.assert_allclose(ext, [0.0225, 0.0225, 0.0105], rtol=0.1)   # whole and upright: the test paste (no
    # tensile strength) breathes by about 1 mm as it bobs, 22.7-24.0 mm across
    assert abs(np.mean(drafts) / (600.0 / RHO_F * 0.012) - 1.0) < 0.10               # d = rho_s / rho_f H
    assert abs(np.mean(forces) / (m_s * 9.81) - 1.0) < 0.05     # still bobbing at 1 s (1.031 over 0.7-1.0 s)
    _no_leak_and_books(sim, sph, liquid, wall)


def _no_leak_and_books(sim, sph, liquid: np.ndarray, wall: np.ndarray) -> None:
    """L6: no liquid particle inside the paste (colour > 0.3), the liquid's mass ledger exact, health ok."""
    from scipy.spatial import cKDTree

    from alloy.solvers.sph.kernels import kernel_w_np

    q = sim.engine.state.particle_q.numpy().astype(np.float64)
    vol = (2.0 * sim.engine.model.particle_radius.numpy()[wall].astype(np.float64)) ** 3
    tree, h = cKDTree(q[wall]), float(sph.config.h)
    colour = [float((vol[n] * kernel_w_np(np.linalg.norm(q[wall][n] - x, axis=1), h)).sum()) if n else 0.0
              for x, n in zip(q[liquid], tree.query_ball_point(q[liquid], h), strict=True)]
    assert max(colour) < 0.3
    led = sph.mass_ledger()
    np.testing.assert_array_equal(led["active"] + led["out"], led["initial"] + led["in"])
    assert sim.check()["ok"], sim.check()["problems"][:3]


@pytest.mark.slow
@pytest.mark.timeout(2400)
def test_water_poured_on_a_paste_bed_presses_it_with_the_whole_column(cards: str) -> None:
    """L2: a paste bed from wall to wall (no liquid can get under it) under water poured on top carries the column's
    weight, F = -rho_f g h A with h A the liquid's own volume: F = -rho_f g V. (A slab resting on the floor is not
    closed form at this resolution: about 40 liquid particles got under its edges and lifted it.)"""
    s = alloy.Scene(dt=0.01, roots=[cards])
    _tank(s)
    s.add_box("bed", size=(W_TANK, W_TANK, 0.016), pos=(0, 0, FLOOR + 0.0005 + 0.008), material="test_paste_1500")
    s.add_pour("water", 50e-6, pos=(0.0, 0.0, 0.055), rate=2.0e-4, speed=1.0)
    sim = alloy.build(s, device="cpu", seed=0)
    sph, _liquid, _wall, _mass = _parts(sim)
    sim.step(60)                                                    # 0.25 s of pour, then settling
    forces = []
    for _ in range(6):
        sim.step(5)
        forces.append(float(sph.pw.particle_impulse().sum(axis=0)[2]) / float(sim.engine.parts.stepper._dt_x))
    want = -RHO_F * 9.81 * sim.liquid_volume("water")
    assert abs(np.mean(forces) / want - 1.0) < 0.05, (np.mean(forces), want)


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_the_coupled_scene_repeats_bitwise_and_a_world_ignores_the_others(cards: str) -> None:
    """L7 (D9): two builds give the same bits; world 0 steps the same whatever world 1 does (its slab 4 mm aside).
    Same world count throughout: MPM's grid bounds follow the world count (alloy.integration.mpm_worlds), so one world
    alone is not the reference."""
    def run(shift: float) -> dict:
        z = FLOOR + 0.027
        sim = alloy.build(_floating_slab(cards, 2, np.array([[0.0, 0.0, z], [shift, 0.0, z]])), device="cpu", seed=0)
        sim.step(8)
        st = sim.state(world=0)
        return {k: st[k].copy() for k in ("water.points", "slab.points")}

    a, b, c = run(0.0), run(0.0), run(0.004)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
        np.testing.assert_array_equal(a[k], c[k])


def test_the_law_cards_are_written(cards: str) -> None:
    assert sorted(p.name for p in (pathlib.Path(cards) / "materials").iterdir()) == ["test_paste_1500.yaml",
                                                                                     "test_paste_600.yaml"]
