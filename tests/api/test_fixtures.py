# SPDX-License-Identifier: Apache-2.0
"""Static fixtures from boxes, friction randomization, and the contact-pipeline choice."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.api import fixtures as F
from alloy.api._build import _contacts
from alloy.api.errors import BuildError, SceneError
from alloy.api.scene import Scene


def _boxes(parts):
    return [(np.array(c) - np.array(s) / 2, np.array(c) + np.array(s) / 2) for _, s, c, _ in parts]


def test_open_box_cavity_and_walls():
    parts = F.open_box_parts((0.08, 0.06, 0.10), 0.004)
    lo = np.min([b[0] for b in _boxes(parts)], axis=0)
    hi = np.max([b[1] for b in _boxes(parts)], axis=0)
    assert np.allclose(lo, [-0.044, -0.034, 0.0]) and np.allclose(hi, [0.044, 0.034, 0.104])
    for lo_k, hi_k in _boxes(parts):          # nothing intrudes into the cavity (|x| < 0.04, |y| < 0.03, z > 0.004)
        overlap = np.minimum(hi_k, [0.04, 0.03, 0.104]) - np.maximum(lo_k, [-0.04, -0.03, 0.004])
        assert (overlap <= 1e-12).any()


def _inside_any(parts, pts):
    boxes = _boxes(parts)
    return np.any([np.all((pts > lo) & (pts < hi), axis=1) for lo, hi in boxes], axis=0)


def test_block_with_hole_leaves_the_hole_open():
    parts = F.block_with_hole_parts((0.1, 0.1, 0.05), (0.0102, 0.0102), 0.03)
    g = np.stack(np.meshgrid(*(np.linspace(-0.0499, 0.0499, 41),) * 2, np.linspace(1e-4, 0.0499, 21),
                             indexing="ij"), axis=-1).reshape(-1, 3)
    hole = (np.abs(g[:, 0]) < 0.0051) & (np.abs(g[:, 1]) < 0.0051) & (g[:, 2] > 0.02)
    assert np.array_equal(_inside_any(parts, g), ~hole)          # the solid is the block minus the hole
    with pytest.raises(SceneError, match="fit"):
        F.block_with_hole_parts((0.1, 0.1, 0.05), (0.2, 0.01), 0.03)


def test_walls_reach_the_bottom_so_no_face_lies_inside_the_solid():
    """A wall standing on the floor puts its bottom face (normal down) on the floor's top; MPM's nearest-face
    colliders then push grains in the wall's foot down through the floor. Walls go to z = 0 instead."""
    fixtures = [F.open_box_parts((0.08, 0.06, 0.10), 0.004), F.open_cylinder_parts(0.04, 0.10, 0.004),
                F.block_with_hole_parts((0.1, 0.1, 0.05), (0.0102, 0.0102), 0.03)]
    for parts in fixtures:
        for (suffix, size, centre, _), (lo, _) in zip(parts, _boxes(parts)):
            assert lo[2] == pytest.approx(0.0, abs=1e-12), (suffix, size, centre)


def test_kinematic_fixture_parts_abut_without_overlap():
    """Kinematic parts abut: as separate moving bodies, overlapping ones read as penetrations (health latched). They
    are one body now (alloy.api.fixtures, tests/scene/test_fixed_pairs_body.py) and keep the same boxes."""
    fixtures = [F.open_box_parts((0.08, 0.06, 0.10), 0.004, overlap=False),
                F.block_with_hole_parts((0.1, 0.1, 0.05), (0.0102, 0.0102), 0.03, overlap=False)]
    for parts in fixtures:
        boxes = _boxes(parts)
        for i, (lo_a, hi_a) in enumerate(boxes):
            for lo_b, hi_b in boxes[i + 1:]:
                assert (np.minimum(hi_a, hi_b) - np.maximum(lo_a, lo_b) <= 1e-12).any()
    g = np.stack(np.meshgrid(*(np.linspace(-0.0499, 0.0499, 41),) * 2, np.linspace(1e-4, 0.0499, 21),
                             indexing="ij"), axis=-1).reshape(-1, 3)
    hole = (np.abs(g[:, 0]) < 0.0051) & (np.abs(g[:, 1]) < 0.0051) & (g[:, 2] > 0.02)
    assert np.array_equal(_inside_any(fixtures[1], g), ~hole)          # the same solid either way


def test_cylinder_slabs_are_tangent_to_the_radius():
    parts = F.open_cylinder_parts(0.04, 0.10, 0.004, segments=24)
    assert len(parts) == 25
    for _, size, (x, y, _), yaw in parts[1:]:
        assert math.hypot(x, y) - size[0] / 2 == pytest.approx(0.04)
        assert math.atan2(y, x) % (2 * math.pi) == pytest.approx(yaw % (2 * math.pi), abs=1e-9)


def test_fixture_names_and_cards(tmp_path):
    from alloy.cards.library import CardLibrary, default_root

    s = Scene()
    names = s.add_open_cylinder("cup", 0.04, 0.10, segments=12, pos=(0.2, 0, 0))
    names += s.add_block_with_hole("block", (0.1, 0.1, 0.05), (0.0102, 0.0102), 0.03, yaw=0.3)
    assert names[0] == "cup_floor" and names[-1] == "block_wall3" and len(names) == 13 + 5
    with pytest.raises(SceneError, match="already"):
        s.add_open_box("x", (0.1, 0.1, 0.1)) and s.add_box("x_floor", size=(1, 1, 1))
    s.save(tmp_path)
    assert len(CardLibrary(roots=[tmp_path, default_root()], assets=None).scene("scene").objects) == 23


def test_friction_randomization_caps_dynamic_friction(tmp_path):
    from alloy.cards.library import CardLibrary, default_root

    s = Scene(worlds=4)
    s.add_box("cube", size=(0.05, 0.05, 0.05), material="steel_carbon")
    s.randomize("cube", "friction", 0.3, 0.8)
    s.save(tmp_path)
    lib = CardLibrary(roots=[tmp_path, default_root()], assets=None)
    lib.scene("scene")                                            # validates with the drawn range
    mat = lib.material("steel_carbon_for_cube")
    assert mat.surface.mu_dynamic.value == pytest.approx(0.3)


def test_contacts_choice():
    s = Scene()
    s.add_box("cube", size=(0.1, 0.1, 0.1))
    assert _contacts(s, "auto") is None and _contacts(s, "newton") == "newton"
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    s.add_mesh("hull", vertices=v, faces=f)
    assert _contacts(s, "auto") is None
    s.add_mesh("cup", vertices=v, faces=f, collision="sdf", static=True)
    assert _contacts(s, "auto") == "newton"
    with pytest.raises(BuildError):
        _contacts(s, "physx")


def test_precise_fixture_makes_a_fine_step_and_stiff_contacts(tmp_path):
    from alloy.api._materialize import materialize
    from alloy.cards.units_yaml import yaml_load

    s = Scene(dt=0.001)
    s.add_ground()
    s.add_block_with_hole("block", (0.1, 0.1, 0.05), (0.0102, 0.0102), 0.03)
    s.add_box("peg", size=(0.01, 0.01, 0.04), pos=(0, 0, 0.075), mass=0.05)
    s.add_box("pad", size=(0.02, 0.02, 0.01), pos=(0.3, 0, 0.005), material="nitrile_70a", rigid=True)
    assert F.precise_substeps(s) == 4 and F.precise_substeps(Scene(dt=0.01)) == 1
    mat = materialize(s, tmp_path)
    card = yaml_load(mat.scene_file.read_text(), "scene")
    assert mat.substeps == 4 and card["control_dt"] == pytest.approx(0.00025)
    k = {n: (yaml_load((tmp_path / "materials" / f"{m}.yaml").read_text(), m)["surface"].get("contact_stiffness")
             if (tmp_path / "materials" / f"{m}.yaml").exists() else None) for n, m in mat.materials.items()}
    assert k["peg"]["value"] == F.PRECISE_K and k["block_floor"]["value"] == F.PRECISE_K
    assert k["pad"] is None                                       # a rubber keeps its own (soft) contact
    s2 = Scene(dt=0.001)
    s2.add_block_with_hole("block", (0.1, 0.1, 0.05), (0.0102, 0.0102), 0.03, precise=False)
    assert F.precise_substeps(s2) == 1


def test_fixtures_accept_static_true_as_every_add_does():
    """Agent trial r2, R2-D1: the docs list static=True for every add_*; fixtures raised TypeError on it."""
    a, b = Scene(), Scene()
    a.add_open_box("bin", inner=(0.05, 0.05, 0.04), static=True)
    a.add_open_cylinder("cup", inner_radius=0.03, depth=0.05, static=True)
    a.add_block_with_hole("block", (0.1, 0.1, 0.05), hole=(0.0102, 0.0102), depth=0.03, static=True)
    b.add_open_box("bin", inner=(0.05, 0.05, 0.04))
    b.add_open_cylinder("cup", inner_radius=0.03, depth=0.05)
    b.add_block_with_hole("block", (0.1, 0.1, 0.05), hole=(0.0102, 0.0102), depth=0.03)
    assert a.to_dict()["items"] == b.to_dict()["items"]                         # a no-op
    with pytest.raises(SceneError) as err:
        a.add_open_box("jug", inner=(0.05, 0.05, 0.05), static=True, kinematic=True)
    assert err.value.field == "kinematic"
    with pytest.raises(SceneError) as err:
        a.add_open_box("jug", inner=(0.05, 0.05, 0.05), static=False)
    assert err.value.field == "static" and "kinematic=True" in err.value.fix
    assert not any(k.startswith("jug") for k in a.items)


def test_randomize_names_the_valid_fields():
    """Agent trial r2, R2-D2: randomize("cube", "mass") failed at build with a bare CardError."""
    scene = Scene()
    scene.add_box("cube", size=(0.05,) * 3, pos=(0, 0, 0.1), mass=0.1)
    with pytest.raises(SceneError) as err:
        scene.randomize("cube", "mass", 0.1, 0.2)
    assert err.value.field == "field" and "density" in err.value.message
    assert all(f in err.value.fix for f in ("density", "friction", "youngs", "material.surface.mu_dynamic"))
    with pytest.raises(SceneError, match="not a field"):
        scene.randomize("cube", "material.nope", 0.1, 0.2)
    scene.randomize("cube", "density", 900, 1100)
    scene.randomize("cube", "material.surface.mu_dynamic", 0.3, 0.4)     # poisson: tests/api/test_audit_stack.py
    assert [r["field"] for r in scene.items["cube"]["randomize"]] == ["material.density", "material.surface.mu_dynamic"]
