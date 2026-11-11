# SPDX-License-Identifier: Apache-2.0
"""ModalSound offline FEM modes (alloy/plugins/sound/fem.py): P2/P1 tets against closed forms, on CPU (seconds)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.plugins.sound import fem
from alloy.plugins.sound.modes import ModeSet
from alloy.plugins.sound.presets import material

STEEL = material("steel")
L, A = 0.3, 0.005                                                   # a square steel bar: Euler-Bernoulli holds
EB_FREE = (4.73004074, 7.85320462, 10.99560784, 14.13716549)
EB_CLAMPED = (1.87510407, 4.69409113)


def _eb(roots, length: float = L) -> np.ndarray:
    c = math.sqrt(STEEL.youngs * A * A / (12 * STEEL.density))
    return np.array(roots) ** 2 / (2 * math.pi * length ** 2) * c


def test_p2_mass_reference_and_rigid_motion():
    assert fem.P2_MASS.sum() == pytest.approx(1.0)
    assert fem.P2_MASS[0, 0] == pytest.approx(6 / 420) and fem.P2_MASS[4, 4] == pytest.approx(32 / 420)
    v, t = fem.box_mesh((L, A, A), (10, 1, 1))
    k, m, mass, n = fem.assemble(v, t, STEEL, 2)
    assert mass == pytest.approx(STEEL.density * L * A * A)
    shift = np.zeros(3 * n)
    shift[1::3] = 1.0                                               # rigid translation along y
    assert np.abs(k @ shift).max() < 1e-6 * abs(k).max() and shift @ (m @ shift) == pytest.approx(mass)


def test_p2_converges_to_euler_bernoulli_and_p1_locks():
    """Bending pairs (y and z, degenerate on a square bar). P2 is within 1 % of Euler-Bernoulli for modes 1-4 and
    moves < 1 % when the mesh doubles (0.03 % on mode 1). The small negative offset growing with the mode is the shear and rotary
    inertia that Euler-Bernoulli leaves out. P1 at the same mesh reads > 50 % high (locking)."""
    coarse = fem.box_modes((L, A, A), STEEL, 8, cells=(30, 1, 1)).freq_hz
    fine = fem.box_modes((L, A, A), STEEL, 8, cells=(60, 2, 2)).freq_hz
    assert fine[0::2] == pytest.approx(fine[1::2], rel=1e-3)                    # y/z pairs
    assert fine[0::2] / _eb(EB_FREE) - 1 == pytest.approx(np.zeros(4), abs=0.01)
    assert np.all(fine[0::2] < _eb(EB_FREE))                                    # Timoshenko lowers them
    assert np.abs(coarse / fine - 1).max() < 0.01 and abs(coarse[0] / fine[0] - 1) < 1e-3
    p1 = fem.box_modes((L, A, A), STEEL, 2, cells=(60, 2, 2), order=1).freq_hz
    assert p1[0] / _eb(EB_FREE)[0] > 1.5


def test_axial_mode_matches_rod_theory():
    ms = fem.box_modes((L, A, A), STEEL, 24, cells=(60, 1, 1))
    axial = [f for f, g in zip(ms.freq_hz, ms.gains) if np.mean(g[:, 0] ** 2) > 0.9 * np.mean(np.sum(g ** 2, 1))]
    assert axial[0] == pytest.approx(math.sqrt(STEEL.youngs / STEEL.density) / (2 * L), rel=2e-3)


def test_clamped_end_gives_the_cantilever():
    v, t = fem.box_mesh((L, A, A), (60, 2, 2))
    fixed = np.flatnonzero(np.isclose(v[:, 0], -0.5 * L))
    ms = fem.mesh_modes(v, t, STEEL, 4, fixed=fixed)
    assert ms.freq_hz[0] == pytest.approx(ms.freq_hz[1], rel=1e-2)            # the mesh splits y/z by 0.3 %
    assert ms.freq_hz[0::2] / _eb(EB_CLAMPED) - 1 == pytest.approx(np.zeros(2), abs=0.02)   # shear + 3D clamp: < 0
    assert np.abs(ms.gains[:, np.isclose(ms.points[:, 0], -0.5 * L)]).max() == 0.0


def test_modal_mass_samples_and_area(tmp_path):
    """Gains are sqrt(m)·φ (φᵀMφ = 1), so the modal mass is the body's; samples are the boundary vertices; the
    radiating area is half the boundary."""
    v, t = fem.box_mesh((0.1, 0.02, 0.01), (10, 2, 1))
    ms = fem.mesh_modes(v, t, STEEL, 3, order=1)
    assert ms.mass == pytest.approx(np.full(3, STEEL.density * 0.1 * 0.02 * 0.01))
    _, m, mass, n = fem.assemble(v, t, STEEL, 1)
    u = np.zeros(3 * n)
    u.reshape(n, 3)[np.unique(fem.boundary(t))] = ms.gains[0] / math.sqrt(mass)
    assert u @ (m @ u) == pytest.approx(1.0, rel=1e-6)          # every vertex of this mesh is on the boundary
    assert len(ms.points) == len(np.unique(fem.boundary(t)))
    assert ms.area == pytest.approx(0.1 * 0.02 + 0.1 * 0.01 + 0.02 * 0.01)
    ms.save(tmp_path / "m.npz")
    assert np.array_equal(ModeSet.load(tmp_path / "m.npz").freq_hz, ms.freq_hz)


def test_cli_from_an_obj(tmp_path):
    pytest.importorskip("pytetwild")
    v, t = fem.box_mesh((0.1, 0.02, 0.01), (1, 1, 1))
    tri = fem.boundary(t)
    obj = tmp_path / "box.obj"
    obj.write_text("".join(f"v {x} {y} {z}\n" for x, y, z in v) + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in tri))
    out = tmp_path / "box_modes.npz"
    assert fem.main([str(obj), "--modes", "4", "--edge", "0.004", "--out", str(out)]) == 0
    ms = ModeSet.load(out)
    ref = fem.box_modes((0.1, 0.02, 0.01), STEEL, 4)
    assert ms.count == 4 and ms.freq_hz[0] == pytest.approx(ref.freq_hz[0], rel=0.03)


def test_obj_reader_takes_negative_indices(tmp_path):
    obj = tmp_path / "tri.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\nf 1/1/1 2/2/2 3/3/3\n")
    v, f = fem.read_obj(str(obj))
    assert v.shape == (3, 3) and f.tolist() == [[0, 1, 2], [0, 1, 2]]
