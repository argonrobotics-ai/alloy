# SPDX-License-Identifier: Apache-2.0
"""``alloy.vary``: the public name for scene variants (``alloy.scene.variants``; its own laws are in
tests/scene/test_variants.py). CPU, no build."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import alloy
import alloy.api as api


def table() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=(0.3, 0.3, 0.4), pos=(0, 0, 0.2), static=True, material="wood_pine")
    scene.add_box("cube", size=(0.04, 0.04, 0.04), pos=(0.05, 0.0, 0.42), material="wood_beech")
    return scene


def test_vary_is_public_and_gives_seeded_variants_within_the_bounds():
    assert "vary" in alloy.__all__ and alloy.vary is api.vary and "Example" in inspect.getdoc(alloy.vary)
    base = table()
    scenes = alloy.vary(base, 3, pos=0.01, seed=2)
    assert len(scenes) == 3 and all(isinstance(s, alloy.Scene) for s in scenes)
    for s in scenes:
        assert s.items["table"]["pose"] == base.items["table"]["pose"]                   # static: stays
        d = np.asarray(s.items["cube"]["pose"][:3]) - np.asarray(base.items["cube"]["pose"][:3])
        assert np.all(np.abs(d[:2]) <= 0.01 + 1e-9) and d[2] == pytest.approx(0.0, abs=1e-12)
    assert len({tuple(s.items["cube"]["pose"]) for s in scenes}) == 3
    again = alloy.vary(table(), 3, pos=0.01, seed=2)
    assert [s.to_dict() for s in again] == [s.to_dict() for s in scenes]                # same seed, same variants
    one = alloy.vary(table(), 1, pos=0.01, seed=2, textures="keep", tint=0.0)            # keywords pass through
    assert one[0].items["cube"]["pose"] == scenes[0].items["cube"]["pose"]
    with pytest.raises(alloy.SceneError) as err:
        alloy.vary(table(), 0)
    assert err.value.field == "n"
