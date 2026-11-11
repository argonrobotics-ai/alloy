# SPDX-License-Identifier: Apache-2.0
"""alloy.api.meshes: closed, outward-wound meshes with the expected volumes."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from alloy.api.meshes import block_with_hole, box, open_box, open_cylinder


def _volume(v: np.ndarray, f: np.ndarray) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _closed(f: np.ndarray) -> bool:
    edges = Counter((int(t[i]), int(t[(i + 1) % 3])) for t in f for i in range(3))
    return all(n == 1 and edges.get((b, a)) == 1 for (a, b), n in edges.items())


@pytest.mark.parametrize("mesh, volume", [
    (box((0.1, 0.2, 0.3)), 0.1 * 0.2 * 0.3),
    (open_box((0.08, 0.08, 0.1), 0.004), 0.088 * 0.088 * 0.104 - 0.08 * 0.08 * 0.1),
    (block_with_hole((0.1, 0.1, 0.05), (0.0102, 0.0102), 0.03), 0.1 * 0.1 * 0.05 - 0.0102 ** 2 * 0.03),
])
def test_closed_and_volume(mesh, volume):
    v, f = mesh
    assert _closed(f) and _volume(v, f) == pytest.approx(volume, rel=1e-9) and v[:, 2].min() == 0.0


def test_cylinder_cup_close_to_round():
    v, f = open_cylinder(0.04, 0.1, 0.004, segments=96)
    exact = np.pi * 0.044 ** 2 * 0.104 - np.pi * 0.04 ** 2 * 0.1
    assert _closed(f) and _volume(v, f) == pytest.approx(exact, rel=0.01)


def test_hole_must_fit():
    with pytest.raises(ValueError, match="fit"):
        block_with_hole((0.1, 0.1, 0.05), (0.2, 0.01), 0.03)
