# SPDX-License-Identifier: Apache-2.0
"""SCR-SOFT-2-2: an MPM fill stays in a free box at worlds > 1 (one MPM grid per world).

Newton's implicit MPM shares one grid among all worlds by default (``Config.separate_worlds=False``). Alloy's worlds
overlap in space, so every world's particles pooled their mass and stress in the same cells and moved alike, and every
world's box collider acted on all of them. SOFT-2's sand_box_200ml in a free liquid_box_56 rested at 1 world and went
through the box floor at 4 (L4). ``builtin_plugins._mpm`` now asks for one grid per world; Newton builds per-world
sparse grids on CUDA only, so on the CPU a multi-world entry takes the dense grid.

CPU, 2 worlds, 0.2 s (wp12-results "WP12-MPM-WORLDS"): before, the sand's mean height fell 35.25 -> 30.69 mm with its
lowest grain at 2.31 mm, under the 4 mm floor top; after, 35.25 -> 35.11 mm with the lowest grain at 4.05 mm.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.core.registry import isolated_registry
from alloy.testing import integration as I
from alloy.testing import scene as S

pytestmark = [pytest.mark.integration]

FLOOR_TOP_MM = 4.0          # liquid_box_56: 4 mm polypropylene floor on the ground


def _mpm_solver(sim):
    from alloy._vendor import newton  # noqa: PLC0415

    (solver,) = [s for s in sim.solvers if isinstance(s, newton.solvers.SolverImplicitMPM)]
    return solver


@pytest.mark.parametrize("worlds", [1, 2])
def test_mpm_entry_runs_one_grid_per_world_on_cpu(worlds: int) -> None:
    from alloy.integration import build_simulation  # noqa: PLC0415

    _, card, lib = next(x for x in S.mixed_scenes() if x[0] == "sand_heap")
    with isolated_registry(empty=True):
        built = I.assemble(dataclasses.replace(card, worlds=worlds), lib, "cpu", worlds)
        solver = _mpm_solver(build_simulation(built, "standard", graph="off"))
    assert solver._separate_worlds is (worlds > 1)          # Newton ignores the flag at 1 world
    assert solver.grid_type == ("dense" if worlds > 1 else "sparse")


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_sand_stays_in_a_free_box_at_two_worlds() -> None:
    import alloy  # noqa: PLC0415

    scene = alloy.Scene()
    scene.add_ground()
    scene.add_object("box", "liquid_box_56", static=False)
    scene.add_object("fill", "sand_box_200ml")
    sim = alloy.build(scene, worlds=2, device="cpu")
    pts = np.asarray(sim.run(0.1, record=["fill"])["fill.points"])           # (frames, worlds, n, 3)
    z0, z1 = pts[0, :, :, 2].mean(axis=1), pts[-1, :, :, 2].mean(axis=1)
    lowest = pts[-1, :, :, 2].min(axis=1) * 1e3
    print(f"mean z {z0 * 1e3} -> {z1 * 1e3} mm, lowest {lowest} mm")
    assert sim.check()["ok"]
    assert np.all(lowest >= FLOOR_TOP_MM - 0.5), f"sand under the box floor: lowest {lowest} mm"
    assert np.all(np.abs(z1 - z0) * 1e3 < 1.0), "the fill is sinking"
