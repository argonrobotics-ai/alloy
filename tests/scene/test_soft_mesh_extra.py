# SPDX-License-Identifier: Apache-2.0
"""A soft mesh without the [mesh] extra (pytetwild): the engine's error offers the rigid way too (physics 2026-11-17).

An outside tester's ``ycb_013_apple`` (vinyl_soft) routed to a soft body and the build died with only "check the
object named in the message". ``alloy.build`` gives the rigid way as its fix (``alloy.api.errors``), but
``alloy.check`` and the engine layer carry only the engine message, so the message itself names it.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import alloy
import alloy.check

OCTAHEDRON = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * 0.04
FACES = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])


def test_without_the_mesh_extra_a_soft_mesh_is_offered_the_rigid_way(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "pytetwild", None)              # as on an install without the extra
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_mesh("apple", vertices=OCTAHEDRON, faces=FACES, pos=(0.0, 0.0, 0.05), material="vinyl_soft")
    with pytest.raises(alloy.FeatureUnavailableError) as err:
        alloy.build(s, device="cpu")
    message = err.value.message
    assert "apple" in message and "[mesh] extra" in message, message
    assert "rigid=True to its add_object / add_mesh call" in message and "route: rigid" in message, message
    s.save(tmp_path)
    found = alloy.check(str(tmp_path / "scenes" / "scene.yaml"), worlds=1, seconds=0.05).findings
    assert any("rigid=True" in f.message for f in found if f.level == "error"), found
