# SPDX-License-Identifier: Apache-2.0
"""Per-world friction reaches VBD's particle-shape contacts (alloy/solvers/vbd/pair_mu.py).

A towel on a tray that is pulled away: how far the towel is carried says what friction its contact had. Before the
fix a towel with ``Scene.randomize(name, "friction", lo, hi)`` slid at ``lo`` in every world, ``Scene.set_friction``
did not reach it, and a towel beside a randomized one took that one's ``lo``.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.solvers.vbd.pair_mu import write_contact_mu

LO, HI = 0.0005, 1.5


def _tray(worlds: int):
    import alloy

    sc = alloy.Scene(worlds=worlds, dt=0.01, cloth_accuracy="fast")
    sc.add_ground(z=-0.5)
    sc.add_box("tray", size=(0.70, 0.40, 0.20), pos=(0.0, 0.0, 0.10), kinematic=True, friction=0.25)
    return sc


def _towel(sc, name: str = "towel", y: float = 0.0, **kw) -> None:
    sc.add_cloth(name, size=(0.10, 0.10), pos=(0.0, y, 0.206), resolution=4, mass=kw.pop("mass", 0.04),
                 self_contact=False, **kw)


def _carried(sc, seed: int = 123) -> tuple[dict[str, np.ndarray], object]:
    """Each cloth's mean x per world [m] after the tray moved 0.40 m in 0.25 s; the built sim (closed)."""
    import alloy

    sim = alloy.build(sc, device="cpu", seed=seed)
    sim.step(6)
    sim.drive("tray", velocity=(1.60, 0.0, 0.0), duration=0.25)
    sim.step(25)
    state = sim.state()
    out = {k[:-len(".points")]: v[:, :, 0].mean(axis=1) for k, v in state.items() if k.endswith(".points")}
    assert sim.check()["ok"]
    sim.close()
    return out, sim


def test_the_kernel_takes_the_scene_pair_then_the_worlds_own_value():
    """Record 0: world 1's particle of material 0 on a shape of material 1 with no scene pair: sqrt(mu[1, 0] *
    shape mu). Record 1: a scene pair: its value. Record 2: a shape without a material: the world's rule. Records
    past the count and particles without a material keep what the solver wrote."""
    dev = "cpu"
    arr = lambda v, t: wp.array(np.asarray(v), dtype=t, device=dev)                       # noqa: E731
    out = arr([9.0, 9.0, 9.0, 9.0, 9.0], wp.float32)
    wp.launch(write_contact_mu, dim=5, device=dev, inputs=[
        arr([4], wp.int32), arr([[2, -1, -1], [0, -1, -1], [3, -1, -1], [1, -1, -1], [0, -1, -1]], wp.vec3i),
        arr([0, 1, 2, 0, 0], wp.int32), arr([0, 0, 1, 1], wp.int32), arr([0, -1, 0, 0], wp.int32),
        arr([1, 2, -1], wp.int32), arr([[0.2, 0.0, 0.0], [0.8, 0.0, 0.0]], wp.float32),
        arr([[-1.0, -1.0, 0.07], [-1.0, -1.0, -1.0], [0.07, -1.0, -1.0]], wp.float32),
        arr([0.5, 0.5, 0.32], wp.float32)], outputs=[out])
    got = out.numpy()
    assert got[0] == pytest.approx(np.sqrt(0.8 * 0.5), rel=1e-6)          # particle 2: world 1, no scene pair
    assert got[1] == pytest.approx(0.07)                                   # particle 0 on material 2: the scene pair
    assert got[2] == pytest.approx(np.sqrt(0.8 * 0.32), rel=1e-6)          # particle 3: its shape has no material
    assert got[3] == 9.0 and got[4] == 9.0                                 # no particle material; past the count


def test_scenes_that_ask_nothing_keep_the_stock_path():
    """No per-contact table for a plain cloth scene, nor when only a rigid object's friction is randomized: the
    solver's contact refresh runs Newton's own code alone, so those scenes step as before, bit for bit."""
    import alloy

    for randomize_rigid in (False, True):
        sc = _tray(2)
        _towel(sc, friction=1.2)
        if randomize_rigid:
            sc.add_box("block", size=(0.04, 0.04, 0.04), pos=(0.2, 0.0, 0.23), mass=0.1)
            sc.randomize("block", "friction", 0.2, 0.8)
        sim = alloy.build(sc, device="cpu", seed=1)
        assert sim.engine.pair_mu is None
        assert all(getattr(s, "alloy_pair_mu", None) is None for s in sim.engine.solvers)
        sim.close()


@pytest.mark.timeout(900)
def test_randomized_friction_is_each_worlds_own_and_set_friction_moves_the_towel():
    """Four worlds drawn in [0.0005, 1.5]: the towel is carried farther the higher its world's draw (before: every
    world slid alike, at the lowest value). A scene pair friction of 0.0005 leaves the towel behind (before: as
    without it)."""
    sc = _tray(4)
    _towel(sc)
    sc.randomize("towel", "friction", LO, HI)
    x, sim = _carried(sc)
    draws = np.asarray(sim.built.randomized[("towel", "material.surface.mu_static")])
    assert np.ptp(draws) > 0.5
    assert np.array_equal(np.argsort(x["towel"]), np.argsort(draws)), (x["towel"], draws)
    assert np.ptp(x["towel"]) > 0.10                                       # measured 0.21 m (draws 0.055 to 1.32)
    table = sim.engine.pair_mu.mu_world.numpy()
    col = int(np.argmax(table[0] == np.float32(draws[0])))
    np.testing.assert_allclose(table[:, col], draws.astype(np.float32), rtol=1e-6)

    nominal = _tray(2)
    _towel(nominal, friction=1.2)
    x0, _ = _carried(nominal)
    paired = _tray(2)
    _towel(paired, friction=1.2)
    paired.set_friction("towel", "tray", LO)
    x1, sim1 = _carried(paired)
    assert sim1.engine.pair_mu is not None
    assert x0["towel"].min() > 0.25 and x1["towel"].max() < 0.01           # measured 0.285 m and 0.0004 m


@pytest.mark.timeout(900)
def test_a_towel_beside_a_randomized_one_keeps_its_own_friction():
    """A towel at friction 1.2 beside a heavier rag randomized in [0.0005, 1.5] is carried as it is alone (before: it
    slid at the rag's lowest value), and the rag follows its own draws."""
    alone = _tray(2)
    _towel(alone, y=-0.08, friction=1.2)
    x0, _ = _carried(alone)
    both = _tray(2)
    _towel(both, y=-0.08, friction=1.2)
    _towel(both, "wet_rag", 0.08, mass=0.10)
    both.randomize("wet_rag", "friction", LO, HI)
    x1, sim = _carried(both)
    np.testing.assert_allclose(x1["towel"], x0["towel"], atol=1e-4)
    draws = np.asarray(sim.built.randomized[("wet_rag", "material.surface.mu_static")])
    assert np.array_equal(np.argsort(x1["wet_rag"]), np.argsort(draws))


def test_a_redraw_reaches_the_table():
    """``refresh`` (the simulation's reset hook) takes the draws the built scene holds now."""
    import alloy

    sc = _tray(2)
    _towel(sc)
    sc.randomize("towel", "friction", LO, HI)
    sim = alloy.build(sc, device="cpu", seed=5)
    pm = sim.engine.pair_mu
    assert pm.refresh in sim.engine.reset_hooks
    draws = sim.built.randomized[("towel", "material.surface.mu_static")]
    col = int(np.argmax(pm.mu_world.numpy()[0] == np.float32(draws[0])))
    draws[:] = [0.3, 0.9]
    pm.refresh()
    np.testing.assert_allclose(pm.mu_world.numpy()[:, col], [0.3, 0.9], rtol=1e-6)
    sim.close()
