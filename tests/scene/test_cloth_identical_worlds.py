# SPDX-License-Identifier: Apache-2.0
"""Audit AGENTS-15: identical cloth worlds match on the CPU, also once the sheet touches itself.

Newton's self-contact kernel walks edge i and vertex i in one thread and adds both into per-particle sums. Counted
over the whole model, world w's edges and vertices start at different offsets, so worlds past the first added a
particle's edge-edge and vertex-triangle terms in another order: a last-bit difference that grew to centimetres in a
loose drape (two worlds of AGENTS.md's cloth scene: 19 mm apart after 0.8 s). The vendored patch
(`alloy/_vendor/patches/0012-vbd-self-contact-world-order.patch`) counts i inside each world.

Laws (CPU; every run must list self-contact pairs, and each scene's worlds differ with upstream's order):
- a pleating drape (a 24 cm sheet over a 6 cm stand, "fast"; its corners pleat onto each other from 0.12 s): three
  identical worlds end bit for bit the same (upstream's order: apart from 0.16 s on, 5 cm by 0.5 s);
- world 0 of that run is the one-world run bit for bit, so one world steps exactly as upstream does;
- a folding sheet (12 x 24 cm over an 8 mm rail, the default accuracy with its Chebyshev hold; the halves meet below
  the rail from 0.17 s): identical worlds match (upstream's order: apart from 0.18 s on, 22 mm by 0.6 s);
- AGENTS.md's own cloth scene (section 7, the default accuracy), two worlds for 1 s: they match (slow);
- on a GPU the same drape stays healthy (worlds may differ there: atomic adds in no fixed order);
- a model whose worlds are not a replicated template keeps upstream's order.

The order of the pairs inside one vertex's or edge's row comes from the BVH query. The laws show it is the same in
co-located identical worlds today (the rows were compared at every detection when the cause was found); a BVH
change that breaks them is that order, not this patch. Not covered: a cloth coupled to a robot through the
proxy, whose force harvest kernel has the same thread layout (`alloy/_vendor/VENDORED.md`, patch 0012).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import alloy
from alloy._vendor.newton._src.solvers.vbd.solver_vbd import _identical_world_counts

pytestmark = pytest.mark.timeout(300)      # 7-22 s each on an idle host; a loaded gate host runs several times slower


def _drape(worlds: int) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01, worlds=worlds, cloth_accuracy="fast")
    scene.add_ground()
    scene.add_box("stand", size=(0.06, 0.06, 0.3), pos=(0, 0, 0.15), static=True)
    scene.add_cloth("sheet", size=(0.24, 0.24), pos=(0, 0, 0.31), resolution=10, mass=0.023)
    return scene


def _fold(worlds: int) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01, worlds=worlds)
    scene.add_ground()
    scene.add_box("rail", size=(0.22, 0.008, 0.008), pos=(0, 0, 0.3), static=True)
    scene.add_cloth("sheet", size=(0.12, 0.24), pos=(0, 0, 0.308), resolution=8, mass=0.0115)
    return scene


def _agents_md(worlds: int) -> alloy.Scene:
    scene = alloy.Scene(dt=0.01, worlds=worlds)
    scene.add_ground()
    scene.add_box("stand", size=(0.2, 0.2, 0.3), pos=(0, 0, 0.15), static=True)
    scene.add_cloth("sheet", size=(0.5, 0.5), pos=(0, 0, 0.5), resolution=20, mass=0.1)
    return scene


def _run(scene: alloy.Scene, seconds: float, device: str = "cpu") -> np.ndarray:
    """Sheet points [worlds, n, 3] after ``seconds``; the run must be healthy and list self-contact pairs."""
    sim = alloy.build(scene, device=device, seed=0)
    info = sim.engine.stepper.solver.trimesh_collision_info
    pairs = 0
    for _ in range(int(round(seconds / scene.dt))):
        sim.step(1)
        pairs = max(pairs, int(info.vertex_colliding_triangles_count.numpy().sum()
                               + info.edge_colliding_edges_count.numpy().sum()))
    assert sim.check()["ok"]
    assert pairs > 0, "the law needs a sheet in self-contact"
    return np.array(sim.state(["sheet"])["sheet.points"])


def _all_match(points: np.ndarray) -> bool:
    return all(np.array_equal(points[0], world) for world in points[1:])


@pytest.fixture(scope="module")
def drape_3_worlds() -> np.ndarray:
    return _run(_drape(3), 0.3)


def test_identical_worlds_of_a_pleating_drape_match_bit_for_bit(drape_3_worlds):
    assert _all_match(drape_3_worlds)


def test_world_0_is_the_one_world_run_bit_for_bit(drape_3_worlds):
    assert np.array_equal(drape_3_worlds[0], _run(_drape(1), 0.3)[0])


def test_identical_worlds_of_a_folding_sheet_match_bit_for_bit():
    assert _all_match(_run(_fold(2), 0.3))


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_identical_worlds_of_the_agents_md_cloth_scene_match_bit_for_bit():
    assert _all_match(_run(_agents_md(2), 1.0))


@pytest.mark.gpu
def test_gpu_worlds_stay_healthy_through_self_contact(drape_3_worlds):
    """On a GPU the sums are atomic adds in no fixed order, so worlds still differ there (deformables.md, Accuracy),
    and this unconverged drape is chaotic: on an L4 its 3 worlds ended 17-27 mm apart and 39-43 mm from the CPU run
    after 0.3 s (upstream's order: 22-29 and 34-39 mm). The per-world thread order must leave the run healthy:
    ``_run`` asserts check ok and self-contact pairs."""
    points = _run(_drape(3), 0.3, "cuda:0")
    spread = max(float(np.abs(points[0] - world).max()) for world in points[1:])
    from_cpu = float(np.abs(points - drape_3_worlds[0]).max())
    print(f"GPU after 0.3 s: worlds at most {1e3 * spread:.3f} mm apart, {1e3 * from_cpu:.3f} mm from the CPU run")
    assert np.isfinite(points).all()


def _model(particle_world: list[int], edges: list[tuple[int, int]], worlds: int) -> SimpleNamespace:
    rows = np.array([(-1, -1, a, b) for a, b in edges], np.int32).reshape(-1, 4)
    return SimpleNamespace(world_count=worlds, particle_count=len(particle_world), edge_count=len(rows),
                           particle_world=SimpleNamespace(numpy=lambda: np.array(particle_world, np.int32)),
                           edge_indices=SimpleNamespace(numpy=lambda: rows))


def test_only_a_replicated_template_gets_the_per_world_order():
    assert _identical_world_counts(_model([0, 0, 1, 1], [(0, 1), (2, 3)], 2)) == (1, 2)
    assert _identical_world_counts(_model([0, 0], [(0, 1)], 1)) == (0, 0)                        # one world: upstream
    assert _identical_world_counts(_model([0, 0, 0, 1], [(0, 1), (1, 2)], 2)) == (0, 0)          # uneven worlds
    assert _identical_world_counts(_model([0, 0, 1, -1], [(0, 1), (2, 3)], 2)) == (0, 0)         # a global particle
    assert _identical_world_counts(_model([0, 0, 1, 1], [(0, 1), (0, 1)], 2)) == (0, 0)          # edges all in world 0
    assert _identical_world_counts(_model([0, 0, 1, 1], [], 2)) == (0, 0)                        # no edges
    assert _identical_world_counts(_model([0, 0, 1, 1], [(0, 1), (2, 9)], 2)) == (1, 2)          # column 2 names the world
    assert _identical_world_counts(_model([0, 0, 1, 1], [(0, 1), (9, 3)], 2)) == (0, 0)          # an edge past the particles
