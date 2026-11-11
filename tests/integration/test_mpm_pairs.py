# SPDX-License-Identifier: Apache-2.0
"""Pair friction reaches implicit MPM's colliders (``alloy.integration.mpm_pairs``; coordinator GO 2026-11-08 14:59Z,
known gap mpm-grain-slope-resolution): ``set_friction`` between sand and a table acts like the table's own
``friction=``, per world; a shape the entry's materials disagree on is refused. CPU."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.core.errors import AssemblyError
from alloy.integration import mpm_pairs
from alloy.integration.stepper import entry_solvers


def _scene(mode: str = "none", worlds: int = 1, second: str | None = None, rand: bool = False) -> alloy.Scene:
    """A 4 x 4 x 2 cm block of soil_sand_dry on a static abs table (the table's own mu 0.4)."""
    s = alloy.Scene(dt=0.01, worlds=worlds)
    s.add_ground()
    s.add_box("table", size=(0.3, 0.3, 0.2), pos=(0, 0, 0.1), static=True, **({"friction": 0.8} if mode == "shape"
                                                                               else {}))
    s.add_box("sand", size=(0.04, 0.04, 0.02), pos=(0, 0, 0.211), material="soil_sand_dry")
    if second:
        s.add_box("sand2", size=(0.04, 0.04, 0.02), pos=(0.08, 0, 0.211), material=second)
    if mode == "pair":
        s.set_friction("table", "sand", 0.8)
    if rand:
        s.randomize("table", "friction", 0.5, 0.9)
    return s


def _build(scene: alloy.Scene):
    sim = alloy.build(scene, device="cpu", seed=0)
    mpm = next(s for s in entry_solvers(sim.engine.parts.stepper).values() if hasattr(s, "_mpm_model"))
    return sim, mpm


def _friction(mpm) -> np.ndarray:
    return mpm._mpm_model.collider.material_friction.numpy().copy()


def _notes(sim) -> list[str]:
    return [n for n in sim.engine.parts.notes if "collider pair friction" in n]


def test_set_friction_reaches_the_collider_like_the_shapes_own_friction() -> None:
    """Newton's materials: default 0.5, ground 0.3, table. Before the fix the pair left the table at its own 0.4."""
    none, pair, shape = (_build(_scene(m)) for m in ("none", "pair", "shape"))
    np.testing.assert_array_equal(_friction(none[1]), np.float32([0.5, 0.3, 0.4]))
    np.testing.assert_array_equal(_friction(pair[1]), _friction(shape[1]))
    np.testing.assert_array_equal(_friction(pair[1]), np.float32([0.5, 0.3, 0.8]))
    assert not _notes(none[0]) and not _notes(shape[0])
    assert _notes(pair[0]) == ["mpm entry mpm: collider pair friction (set_friction / pair cards): abs_for_table "
                               "(1 shapes) mu 0.4 -> 0.8"]


def test_the_pair_steps_the_sand_exactly_as_the_shapes_own_friction() -> None:
    out = []
    for mode in ("pair", "shape"):
        sim, _ = _build(_scene(mode))
        sim.step(3)
        out.append(sim.state(world=0)["sand.points"].copy())
    np.testing.assert_array_equal(out[0], out[1])


def test_each_world_takes_the_pair_and_a_randomized_table_without_one_keeps_its_draw() -> None:
    """Separate worlds: one static collider per world. Without a pair the table's per-world draw still reaches the
    sand through the shape's own friction, as before."""
    _, pair = _build(_scene("pair", worlds=2, rand=True))
    np.testing.assert_array_equal(_friction(pair), np.float32([0.5, 0.3, 0.8, 0.3, 0.8]))
    _, drawn = _build(_scene("none", worlds=2, rand=True))
    f = _friction(drawn)
    assert f[2] != f[4] and 0.5 <= f[2] <= 0.9 and 0.5 <= f[4] <= 0.9


def test_a_shape_the_entrys_materials_disagree_on_is_refused() -> None:
    """One MPM solver gives a shape one friction: a pair for one material and none for the other cannot be built."""
    with pytest.raises(alloy.BuildError, match="sand_silica_dry \\(no pair\\).*set_friction the same value"):
        _build(_scene("pair", second="sand_silica_dry"))


def test_a_collider_order_newton_changed_raises_instead_of_writing_the_wrong_shape() -> None:
    _, mpm = _build(_scene("none"))
    assert mpm_pairs.collider_shapes(mpm).tolist()[0] == -1
    f = mpm._mpm_model.collider.material_friction
    f.assign(np.float32([0.5, 0.4, 0.3]))                 # ground and table swapped
    with pytest.raises(AssemblyError, match="discovery order"):
        mpm_pairs.collider_shapes(mpm)
