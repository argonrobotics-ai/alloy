# SPDX-License-Identifier: Apache-2.0
"""``add_cloth(resolution=...)`` counts cells: a sheet of n x m cells has (n + 1) x (m + 1) vertices.

Laws (CPU): ``resolution=20`` on a square sheet builds 20 x 20 cells, 441 vertices in ``sim.state()``'s points and 800
triangles; on 0.5 x 0.25 m, 21 x 11 vertices; on 0.3 x 0.6 m with 12, 7 x 13; ``(11, 9)`` on 0.20 x 0.16 m is kept
(12 x 10 vertices); none of these warns. A grid the build cannot keep is said, with the grid it makes, which is the
one built: ``(10, 6)`` on a square sheet (cells are square: 8 x 8, 81 vertices) and ``resolution=60`` on a square
sheet (at most 1,600 cells: 40 x 40, 1,681 vertices). The API's cap is the emitter's. Before, the API counted
resolution x resolution vertices (400 for 441), its docstring said vertices, and neither grid was said.
"""

from __future__ import annotations

import logging

import pytest

import alloy.api as alloy
from alloy.api._items import CLOTH_CELLS_MAX


def _points(size, resolution, caplog) -> tuple[int, int, list[str]]:
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="alloy.api"):
        scene = alloy.Scene(dt=0.01)
        scene.add_ground()
        scene.add_cloth("sheet", size=size, pos=(0, 0, 0.5), resolution=resolution, mass=0.05)
    said = [r.getMessage() for r in caplog.records if "add_cloth" in r.getMessage()]
    sim = alloy.build(scene, device="cpu")
    try:
        return int(sim.state(["sheet"])["sheet.points"].shape[1]), int(sim.faces("sheet").shape[0]), said
    finally:
        sim.close()


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("size", "resolution", "vertices"), [
    ((0.5, 0.5), 20, 21 * 21), ((0.5, 0.25), 20, 21 * 11), ((0.3, 0.6), 12, 7 * 13), ((0.20, 0.16), (11, 9), 12 * 10)])
def test_resolution_counts_cells(size, resolution, vertices, caplog):
    n, faces, said = _points(size, resolution, caplog)
    assert n == vertices and not said, (n, said)                  # resolution=20: 441, the API counted 400
    if resolution == 20 and size == (0.5, 0.5):
        assert faces == 2 * 20 * 20


@pytest.mark.compile_heavy
@pytest.mark.parametrize(("resolution", "grid", "why"), [((10, 6), (8, 8), "square"), (60, (40, 40), "1,600")])
def test_a_grid_the_build_cannot_keep_is_said(resolution, grid, why, caplog):
    n, _, said = _points((0.5, 0.5), resolution, caplog)
    assert n == (grid[0] + 1) * (grid[1] + 1)
    assert len(said) == 1 and f"{grid[0]} x {grid[1]}-cell sheet ({n} vertices)" in said[0] and why in said[0], said


def test_the_cap_is_the_emitters():
    from alloy.scene.emitters.deformable import CLOTH_VERTS_TARGET  # noqa: PLC0415

    assert CLOTH_CELLS_MAX == CLOTH_VERTS_TARGET
