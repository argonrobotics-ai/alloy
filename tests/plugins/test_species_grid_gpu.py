# SPDX-License-Identifier: Apache-2.0
"""The species plug-in's diffusion grid keeps its own descriptor across CUDA graph replays (WP9's ``GridDescriptor``,
alloy.solvers.sph.grid_descriptor), GPU.

Warp 1.18 replays every captured float32 HashGrid build with the descriptor of the grid the host built last (OSS-4's
find, WP9's fix). The species plug-in builds its own grid each exchange beside the SPH solver's. The laws (the
coordinator's: species + SPH in one process, bitwise against alone):
- a species sim (the law block: two species diffusing in an SPH liquid cube, CUDA graphs) stepped in turn with an SPH
  pour sim ends bitwise equal to a twin stepped alone: amounts and positions;
- a foreign float32 HashGrid rebuilt between its periods changes nothing, bitwise;
- the guard is armed (its layout check passes on this warp);
- the harness's check on CUDA passes: its strict hooks raise on any host read in a hook (rc2-1109's red).
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not wp.is_cuda_available(), reason="needs CUDA")]

PERIODS = 12


def _species():
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415
    from alloy.plugins.species import Species  # noqa: PLC0415
    from alloy.plugins.species.laws import block_scene, split_config  # noqa: PLC0415

    plug = Species(**split_config())
    return build_simulation(block_scene(), [plug], "cuda:0", graph=True), plug


def _pour():
    import alloy  # noqa: PLC0415

    sc = alloy.Scene(dt=0.01, worlds=2)
    sc.add_ground()
    sc.add_open_cylinder("cup", inner_radius=0.03, depth=0.06, wall=0.004)
    sc.add_pour("water", 20e-6, pos=(0.0, 0.0, 0.12), rate=1e-4)
    return alloy.build(sc, device="cuda:0", seed=0)


def _books(sim, plug) -> tuple[np.ndarray, np.ndarray]:
    return plug.amount.numpy().copy(), sim.state.particle_q.numpy().copy()


def _same(a: tuple, b: tuple) -> None:
    for x, y, what in ((a[0], b[0], "amounts"), (a[1], b[1], "positions")):
        assert np.array_equal(x, y), (what, int((x != y).sum()), float(np.abs(x.astype(float) - y).max()))


def test_species_beside_an_sph_sim_stays_bitwise_equal_to_alone():
    twin, tp = _species()
    for _ in range(PERIODS):                         # alone first: only the other sim can differ below
        twin.step_period()
    sim, sp = _species()
    pour = _pour()
    for _ in range(PERIODS):
        sim.step_period()
        pour.step(1)
    _same(_books(sim, sp), _books(twin, tp))
    assert sp.desc.ready and tp.desc.ready
    assert np.abs(sp.read()["conservation_error"]).max() == 0


def test_a_foreign_float32_grid_between_periods_changes_nothing():
    twin, tp = _species()
    for _ in range(PERIODS):
        twin.step_period()
    sim, sp = _species()
    other = wp.HashGrid(32, 32, 32, device="cuda:0")
    pts = wp.array(np.random.default_rng(0).uniform(-1.0, 1.0, (500, 3)).astype(np.float32), dtype=wp.vec3,
                   device="cuda:0")
    for _ in range(PERIODS):
        sim.step_period()
        other.build(pts, 0.05)
    _same(_books(sim, sp), _books(twin, tp))


def test_the_harness_check_passes_on_cuda():
    import alloy.plugins as ap  # noqa: PLC0415

    report = ap.check("alloy.plugins.species.plugin:Species", device="cuda:0", quick=True)
    assert report.passed, report.summary()
