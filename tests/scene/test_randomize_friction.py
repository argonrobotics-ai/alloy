# SPDX-License-Identifier: Apache-2.0
"""Per-world friction draws reach rigid contacts (WP3's report from F5, CPU, both CPU routes).

Before: ``Scene.randomize(name, "friction", lo, hi)`` drew ``material.surface.mu_static``. Contacts use one
coefficient per pair (μ_d), the API capped μ_d at the lowest draw, and MuJoCo's native contacts never read the
per-world pair table, so every world of a box on a slope slid exactly the same (96.5 mm). Now a μ_static draw sets
the pair's friction (μ_d follows) and every drawn world's MuJoCo pair rows and geom friction get it (the C route reads
each world's pair rows from Newton's attributes). The block starts in contact: a block landing from 0.1 mm above keeps
sliding at a few mm/s after it lands (WP3's contact-layer finding, 2026-11-07), which would blur the sticking worlds.
A sticking world moves < 0.1 mm in 0.5 s; a sliding one moves Coulomb's 1/2 g (sin - mu cos) t^2 within 15 %.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy

ANGLE = math.radians(35.0)            # tan 0.700: a box sticks where the pair friction is above it
T = 0.5                              # [s] run


def _slope(worlds: int = 8) -> alloy.Scene:
    n = np.array([math.sin(ANGLE), 0.0, math.cos(ANGLE)])
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_box("slope", size=(0.6, 0.3, 0.02), pos=(0, 0, 0.3), rpy=(0, ANGLE, 0), static=True, friction=1.0)
    s.add_box("blk", size=(0.05, 0.05, 0.02), pos=tuple(np.array([0, 0, 0.3]) + n * 0.02), rpy=(0, ANGLE, 0),
              mass=0.1, friction=0.9)
    s.randomize("blk", "friction", 0.3, 1.2)
    return s


@pytest.mark.parametrize("route", ["mjc", "warp"])
def test_each_world_slides_or_sticks_by_its_own_friction_draw(monkeypatch, route):
    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    sim = alloy.build(_slope(), seed=0, device="cpu")
    p0 = sim.state(["blk"])["blk.pos"].copy()
    sim.run(T)
    slide = np.linalg.norm(sim.state(["blk"])["blk.pos"] - p0, axis=-1)
    table = sim.engine.built.pair_table
    i, j = (table.material_index(k) for k in _keys(sim))
    mu = table.world_table.numpy()[:, i, j, 1]                    # each world's pair friction (μ_d)
    tan = math.tan(ANGLE)
    sticks, slides = mu > tan + 0.05, mu < tan - 0.05
    assert sticks.any() and slides.any(), mu
    assert np.all(slide[sticks] < 1e-4), (mu, slide)
    coulomb = 0.5 * 9.81 * (math.sin(ANGLE) - mu * math.cos(ANGLE)) * T ** 2
    np.testing.assert_allclose(slide[slides], coulomb[slides], rtol=0.15)


def test_two_surface_draws_on_one_object_both_hold():
    """A second surface draw rebuilds the object's pair rows from every draw so far, so it does not undo the first.
    The second draw is μ_dynamic, below the μ_static range (the pair keeps μ_static >= μ_d). It was the restitution
    until ``Scene.randomize`` refused that (578fb899a: rigid contacts keep one restitution for every world)."""
    s = _slope(4)
    s.randomize("blk", "material.surface.mu_dynamic", 0.1, 0.25)
    sim = alloy.build(s, seed=0, device="cpu")
    built = sim.engine.built
    table = built.pair_table
    i = table.material_index(_keys(sim)[0])
    drawn_s = built.randomized[("blk", "material.surface.mu_static")]
    drawn_d = built.randomized[("blk", "material.surface.mu_dynamic")]
    assert np.ptp(drawn_s) > 0.1 and np.ptp(drawn_d) > 0.01      # both differ between the worlds
    own = table.world_table.numpy()[:, i, i]                      # blk on blk: (μ_s, μ_d, e, lever) per world
    np.testing.assert_allclose(own[:, 0], drawn_s, rtol=1e-6)     # the friction draw was not undone
    np.testing.assert_allclose(own[:, 1], drawn_d, rtol=1e-6)


def _keys(sim) -> tuple[str, str]:
    """(block, slope) material keys of the built scene's plan."""
    plan = sim.engine.built._plan
    by = {t.name: t.material_key for t in plan.objects}
    return by["blk"], by["slope"]
