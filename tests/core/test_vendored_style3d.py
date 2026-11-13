# SPDX-License-Identifier: Apache-2.0
"""SCR-WP0-S3D-WORLDS: Style3D keeps each world to itself (alloy/_vendor/patches/0002-*, 0003-*).

- 0002: upstream's broad phase (vertex-face, edge-edge, edge-face) queried one BVH over every world's sheet, so
  overlaid worlds collided with each other; every world of a 2-world scene went INVALID at the first physics step
  (WP2's report, 2026-11-08). The BVH leaves are grouped by world, each query starting at its own world's root.
- 0003: upstream's PCG summed rTz and pTAp over every world's particles, so a world's solve depended on the others.
  The sums are per world now (deterministic chunks).
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy


def _scene(worlds: int, lift: float = 0.0) -> alloy.Scene:
    """An anisotropic cotton sheet (routed to Style3D) 2 mm over a table; worlds after the first start ``lift``
    higher."""
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_box("table", size=(0.3, 0.3, 0.4), pos=(0, 0, 0.2), static=True)
    pos = np.array([[0.0, 0.0, 0.402]] + [[0.0, 0.0, 0.402 + lift]] * (worlds - 1))
    s.add_cloth("sheet", size=(0.4, 0.4), pos=pos, resolution=12, mass=0.05, material="cotton_woven_aniso")
    # a woven card on a grid sheet routes to the orthotropic membrane by default (D16): these tests are Style3D's
    s.items["sheet"]["card_extra"]["route"] = {"route": "cloth_style3d", "reason": "test: the Style3D solver"}
    return s


def _collision(sim):
    """The Style3D solver's Collision object (searched from sim.engine)."""
    from alloy._vendor.newton._src.solvers.style3d.collision.collision import Collision

    seen, frontier = set(), [sim.engine]
    for _ in range(8):
        nxt = []
        for o in frontier:
            if id(o) in seen or isinstance(o, (int, float, str, bytes, np.ndarray)):
                continue
            seen.add(id(o))
            if isinstance(o, Collision):
                return o
            vals = o.values() if isinstance(o, dict) else o if isinstance(o, (list, tuple)) else vars(o).values() \
                if hasattr(o, "__dict__") else ()
            nxt.extend(vals)
        frontier = nxt
    raise AssertionError("no Style3D Collision under sim.engine")


def _cross_world_candidates(col, worlds: int) -> int:
    """Broad-phase candidates whose element lies in another world's block (replicated worlds are contiguous)."""
    m = col.model
    n_p, n_e, n_t = m.particle_count // worlds, m.edge_count // worlds, m.tri_count // worlds
    cross = 0
    for bp, n_q, n_hit in ((col.broad_phase_vf, n_p, n_t), (col.broad_phase_ee, n_e, n_e),
                           (col.broad_phase_ef, n_e, n_t)):
        a = bp.numpy()
        for q in range(a.shape[1]):
            cross += sum(1 for x in a[1:1 + a[0, q], q] if x // n_hit != q // n_q)
    return cross


def _run(worlds: int, steps: int = 3, lift: float = 0.0):
    sim = alloy.build(_scene(worlds, lift), device="cpu", seed=0)
    assert {r["route"] for r in sim.routes().values()} == {"cloth_style3d", "static"}, sim.routes()
    sim.step(steps)
    return sim, sim.state(["sheet"])["sheet.points"]


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_two_worlds_step_like_one() -> None:
    """2 worlds: healthy, identical to each other, world 0 within float32 rounding of the 1-world run, and no
    broad-phase candidate in another world (unpatched: both worlds INVALID at t = 1.25 ms)."""
    one, p1 = _run(1)
    two, p2 = _run(2)
    try:
        assert two.health()["invalid_worlds"] == [] and two.check()["ok"], two.health()
        assert np.array_equal(p2[0], p2[1])
        assert np.abs(p2[0] - p1[0]).max() < 1e-6
        assert _cross_world_candidates(_collision(two), 2) == 0
    finally:
        one.close()
        two.close()


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_one_world_keeps_the_ungrouped_tree() -> None:
    """1 world: no groups, so the queries are upstream's (bit for bit)."""
    sim, _ = _run(1, steps=1)
    try:
        col = _collision(sim)
        assert col.particle_groups is None and col.tri_bvh.groups is None and col.edge_bvh.groups is None
    finally:
        sim.close()


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_a_world_does_not_depend_on_the_others() -> None:
    """World 0 is bit for bit the same whether world 1 starts with it or 5 cm higher (unpatched PCG: 21 um apart
    after 3 periods, 0.06 um rounding floor)."""
    same, p_same = _run(2, steps=3)
    lifted, p_lifted = _run(2, steps=3, lift=0.05)
    try:
        assert np.abs(p_lifted[1] - p_lifted[0]).max() > 0.01                  # world 1 really is elsewhere
        assert np.array_equal(p_same[0], p_lifted[0])
        assert lifted.health()["invalid_worlds"] == []
    finally:
        same.close()
        lifted.close()
