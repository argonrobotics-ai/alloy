# SPDX-License-Identifier: Apache-2.0
"""``alloy.scene.mujoco_pairs``: the explicit MuJoCo pair rows are chosen in numpy (``_candidates``) and counted without
being built under ``count_only`` (``alloy.estimate``'s assembly; audit S1-002: the per-pair Python test made a build
of 16 decomposed bowls spend 17 s there). The rows must be the ones the per-pair loop over every shape pair gives, bit
for bit, and the count the number a build adds (CPU)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.api import _estimate as E
from alloy.api import meshes as M
from alloy.scene import mujoco_pairs as MP

WORLDS = 2


def _scene() -> alloy.Scene:
    """Every reason a row exists or not: convex meshes (CCD rows, on either side of a pair with the same μ), a static
    mesh, materials whose pair μ differs from MuJoCo's max, a drawn friction, a sphere, a precise fixture and same-μ
    boxes (no row)."""
    s = alloy.Scene(dt=0.01, worlds=WORLDS)
    s.add_ground()
    s.add_box("table", size=(0.6, 0.6, 0.4), pos=(0, 0, 0.2), static=True, material="wood_beech")
    v, f = M.open_cylinder(inner_radius=0.03, depth=0.04, wall=0.006, segments=12)
    for i in range(3):
        s.add_mesh(f"bowl_{i}", vertices=v, faces=f, pos=(-0.2 + 0.1 * i, 0.15, 0.42), collision="convex", mass=0.2)
    s.add_mesh("dish", vertices=v, faces=f, pos=(0.2, -0.2, 0.4), collision="convex", static=True)
    s.add_box("pom", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.43), material="pom_acetal")
    s.add_box("steel", size=(0.05, 0.05, 0.05), pos=(0.1, 0, 0.43), material="steel_carbon")
    s.add_box("steel2", size=(0.05, 0.05, 0.05), pos=(0.1, -0.1, 0.43), material="steel_carbon")
    s.add_sphere("ball", 0.03, pos=(-0.1, 0, 0.43), material="steel_carbon")
    s.add_capsule("cap", 0.015, 0.05, pos=(0, -0.1, 0.43), mass=0.1)
    s.add_box("plain", size=(0.04, 0.04, 0.04), pos=(-0.2, -0.15, 0.43), mass=0.1)   # the bowls' μ: a CCD-only row
    s.add_mesh("cup", vertices=v, faces=f, pos=(-0.2, -0.05, 0.42), collision="convex", mass=0.2)  # CCD after it
    s.randomize("pom", "friction", 0.6, 1.0)
    s.add_block_with_hole("blk", size=(0.08, 0.08, 0.04), hole=(0.02, 0.02), depth=0.03, pos=(0.2, 0.2, 0.4))
    return s


def _pair_arrays(scene: alloy.Scene) -> dict[str, np.ndarray]:
    sim = alloy.build(scene, device="cpu")
    model = sim.engine.built.model
    out = {f"{ns}:{k}": v.numpy() for ns in ("mujoco", "alloy") if getattr(model, ns, None) is not None
           for k, v in vars(getattr(model, ns)).items() if "pair" in k and hasattr(v, "numpy")}
    sim.close()
    return out


def _every_pair(b, shapes, objs, index, table):
    """The old double loop: every shape pair goes to ``_maybe_row``, which keeps the ones that need a row."""
    return [(i, j) for i in range(len(shapes)) for j in range(i + 1, len(shapes))]


@pytest.fixture(scope="module")
def rows() -> dict[str, np.ndarray]:
    return _pair_arrays(_scene())


def test_rows_are_the_per_pair_loops_bit_for_bit(rows, monkeypatch):
    assert len(rows["mujoco:pair_geom1"]) > 0
    monkeypatch.setattr(MP, "_candidates", _every_pair)
    old = _pair_arrays(_scene())
    assert set(old) == set(rows)
    for key, value in old.items():
        np.testing.assert_array_equal(rows[key], value, err_msg=key)


def test_count_only_counts_the_rows_a_build_adds_and_adds_none(rows):
    per_world = len(rows["mujoco:pair_geom1"]) // WORLDS
    template = E._template_counts(_scene(), "cpu")
    assert template[6]["pair_rows"] == per_world
    with MP.count_only() as counted:
        assert MP._COUNT_ONLY.get() is counted
    assert MP._COUNT_ONLY.get() is None


def test_groups_vectorized_is_the_scalar_rule():
    groups = np.array([-3, -2, -1, 0, 1, 2, 3])
    for g in groups:
        assert MP._groups_collide_np(int(g), groups).tolist() == [MP.groups_collide(int(g), int(x)) for x in groups]


def test_candidates_follow_the_scalar_rule_on_random_shapes(monkeypatch):
    """``_candidates`` against ``_maybe_row``'s own tests pair by pair, on random shapes: shared bodies and objects,
    massless bodies, collision groups (0, negative, positive), filtered pairs, CCD and drawn objects, and pair μ equal
    to MuJoCo's max for about half the material pairs (where only CCD or a draw makes a row)."""
    from types import SimpleNamespace

    from alloy.scene import entries

    rng = np.random.default_rng(7)
    for _ in range(40):
        n_obj, n = int(rng.integers(2, 7)), int(rng.integers(2, 16))
        ccd = rng.random(n_obj) < 0.4
        draws = rng.random(n_obj) < 0.2
        objs = [SimpleNamespace(obj=None if rng.random() < 0.1 else object(), material_key=f"m{o}", ccd=bool(ccd[o]),
                                placement=SimpleNamespace(randomize=[SimpleNamespace(field="material.surface.mu")]
                                                          if draws[o] else []))
                for o in range(n_obj)]
        obj = rng.integers(0, n_obj, n)
        body = np.where(rng.random(n) < 0.15, -1, obj * 2 + rng.integers(0, 2, n))
        mu = rng.choice([0.3, 0.5, 0.8], n)
        table = np.zeros((n_obj, n_obj, 4))
        for a in range(n_obj):
            for c in range(n_obj):
                table[a, c, 1] = table[c, a, 1] = 0.8 if rng.random() < 0.5 else rng.random()
        b = SimpleNamespace(shape_material_mu=list(mu), shape_collision_group=list(rng.choice([-2, -1, 0, 1, 2], n)),
                            body_mass=list(np.where(rng.random(2 * n_obj) < 0.2, 0.0, 1.0)),
                            _shape_collision_filter_pairs={(int(i), int(j)) for i, j in rng.integers(0, n, (3, 2))
                                                           if i < j})
        shapes = [(s, int(body[s]), int(obj[s])) for s in range(n)]
        index = {f"m{o}": o for o in range(n_obj)}
        monkeypatch.setattr(entries, "ccd_backed", lambda t: t.obj is not None and t.ccd)

        def scalar(i, j):
            (s0, b0, o0), (s1, b1, o1) = shapes[i], shapes[j]
            m0, m1 = MP._mass(b, b0), MP._mass(b, b1)
            if b0 == b1 or (m0 <= 0.0 and m1 <= 0.0) or MP._excluded(b, shapes[i], shapes[j], objs):
                return False
            mu_max = max(b.shape_material_mu[s0], b.shape_material_mu[s1])
            same = abs(table[o0, o1, 1] - mu_max) <= MP.MU_RTOL * max(abs(mu_max), 1e-12)
            special = any(objs[o].ccd or MP._draws_surface(objs[o]) for o in (o0, o1))
            return not same or special

        want = [(i, j) for i in range(n) for j in range(i + 1, n) if scalar(i, j)]
        assert MP._candidates(b, shapes, objs, index, table) == want
