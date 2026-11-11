# SPDX-License-Identifier: Apache-2.0
"""Thermal buoyancy's force (alloy.solvers.sph.buoyancy, gaps-now #8), CPU, on synthetic slots.

- ρ(T) is NIST's water table (phase_materials.WATER_DENSITY): a row reads back exactly, the maximum sits near 4 C, and it falls monotonically above.
- Each slot's force is m φ (ρ(T) - ρ̄) / ρ̄ g (g down: a hot slot is pushed up), with ρ̄ the liquid-mass-weighted
  mean of its heat material's liquid in its world. Those forces sum to zero in each world; two heat groups of one
  material (a hot liquid and a cold liquid) share one ρ̄, so the hot group rises against the cold one.
- Uniform temperature gives no force (to the fixed-point sums' 1e-8). Groups whose heat material has no table get
  none, and neither do slots with no liquid fraction. The forces add into what particle_f already holds.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.plugins.thermal.phase_materials import material
from alloy.solvers.sph import buoyancy as B

WATER = material("water").density_table

G = 9.81


def test_the_water_table_is_nists():
    assert float(B.density(WATER, 273.15 + 20.01)) == pytest.approx(998.205, abs=1e-9, rel=0)
    t = 273.15 + np.linspace(0.0, 100.0, 401)
    rho = B.density(WATER, t)
    assert 3.0 < float(t[np.argmax(rho)] - 273.15) < 6.0
    above = t > 273.15 + 6.0
    assert np.all(np.diff(rho[above]) < 0.0)


def _run(temperature: np.ndarray, liquid: np.ndarray, mass: np.ndarray, materials, slot_group, prefill: float = 0.0):
    W, P = temperature.shape
    tables = {"water": WATER, "dark_chocolate": material("dark_chocolate").density_table}
    b = B.Buoyancy.build(materials, tables, np.asarray(slot_group), W, "cpu")
    gidx = np.arange(W * P, dtype=np.int32)[::-1].copy() + 3                    # any global ids, here reversed
    f = wp.array(np.full((W * P + 3, 3), prefill, np.float32), dtype=wp.vec3, device="cpu")
    grav = wp.array(np.tile([0.0, 0.0, -G], (W, 1)).astype(np.float32), dtype=wp.vec3, device="cpu")

    def flat(a):
        return wp.array(np.asarray(a, np.float64).reshape(-1), dtype=wp.float64, device="cpu")

    b.apply(flat(temperature), flat(liquid), flat(mass), wp.array(gidx, dtype=wp.int32, device="cpu"), grav, f)
    out = f.numpy()[gidx].reshape(W, P, 3)
    return out, b


def _expected(temperature, liquid, mass, slot_group, has_table, material_of=None) -> np.ndarray:
    W, P = temperature.shape
    fz = np.zeros((W, P))
    mat = material_of or {g: g for g in set(slot_group)}
    for w in range(W):
        for g in set(mat.values()):
            idx = [p for p in range(P) if mat[slot_group[p]] == g]
            if not has_table[g]:
                continue
            wm = mass[w, idx] * liquid[w, idx]
            rho = B.density(WATER, temperature[w, idx])
            rho_bar = float((wm * rho).sum() / wm.sum())
            fz[w, idx] = -wm * (rho - rho_bar) / rho_bar * G          # c g with g = (0, 0, -G)
    return fz


def test_each_slot_feels_its_density_difference_and_an_object_feels_none():
    rng = np.random.default_rng(0)
    W, groups = 2, [0, 0, 0, 0, 0, 1, 1]
    T = np.array([[293.15, 293.15, 353.15, 323.15, 278.15, 353.15, 293.15],
                  [363.15, 300.0, 300.0, 290.0, 330.0, 280.0, 280.0]])
    m = rng.uniform(5e-5, 2e-4, T.shape)
    phi = np.ones(T.shape)
    phi[1, 3] = 0.4
    f, b = _run(T, phi, m, ("water", "dark_chocolate"), groups)
    assert b.skipped == ("dark_chocolate",)
    want = _expected(T, phi, m, groups, {0: True, 1: False})
    np.testing.assert_allclose(f[..., 2], want, rtol=2e-6, atol=0)
    assert not f[..., :2].any() and not f[:, 5:].any()                          # vertical only; no table: none
    assert float(f[0, 2, 2]) > 0.0 > float(f[0, 0, 2])                          # the hot slot rises, the cold sinks
    for w in range(W):
        net, scale = float(f[w, :5, 2].sum()), float(np.abs(f[w, :5, 2]).sum())
        assert abs(net) < 1e-6 * scale, (w, net, scale)


def test_uniform_temperature_and_no_liquid_give_no_force_and_forces_add():
    T = np.full((1, 4), 330.0)
    m = np.full((1, 4), 1e-4)
    f, _ = _run(T, np.ones((1, 4)), m, ("water",), [0, 0, 0, 0])
    assert float(np.abs(f).max()) < 1e-8 * 1e-4 * G
    T = np.array([[293.15, 353.15, 353.15, 293.15]])
    f, _ = _run(T, np.array([[1.0, 0.0, 1.0, 1.0]]), m, ("water",), [0, 0, 0, 0], prefill=0.5)
    assert float(f[0, 1, 2]) == 0.5 and np.all(f[..., :2] == 0.5)               # φ 0: none; forces add to particle_f
    assert float(f[0, 2, 2]) > 0.5 > float(f[0, 0, 2])


def test_two_groups_of_one_material_share_the_reference():
    """A hot liquid (group 0) and a cold liquid (group 1), both water: the hot one is pushed up as a whole."""
    T = np.array([[353.15, 353.15, 293.15, 293.15, 293.15]])
    m = np.full((1, 5), 1e-4)
    f, b = _run(T, np.ones((1, 5)), m, ("water", "water"), [0, 0, 1, 1, 1])
    want = _expected(T, np.ones((1, 5)), m, [0, 0, 1, 1, 1], {0: True}, {0: 0, 1: 0})
    np.testing.assert_allclose(f[..., 2], want, rtol=2e-6, atol=0)
    assert float(f[0, :2, 2].sum()) > 0.0 > float(f[0, 2:, 2].sum())
    assert abs(float(f[0, :, 2].sum())) < 1e-6 * float(np.abs(f[0, :, 2]).sum())
