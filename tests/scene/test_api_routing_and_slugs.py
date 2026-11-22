# SPDX-License-Identifier: Apache-2.0
"""Regression tests for robot cloth routing and material slug disambiguation."""

from __future__ import annotations

import hashlib

import pytest
import yaml

from alloy.api import _materialize as mat
from alloy.cards.library import CardLibrary, default_root
from alloy.cards.types import Placement, RobotRef, SceneCard
from alloy.cards.units_yaml import yaml_dump
from alloy.core.errors import AssemblyError
from alloy.route.types import Route
from alloy.scene.assembler import SceneAssembler
from alloy.scene.builtin_plugins import register_builtin_solvers


def test_anisotropic_cloth_beside_robot_routes_to_cloth_vbd(tmp_path):
    """plan() counts scene robots, so an anisotropic mesh cloth beside one routes to cloth_vbd, not Style3D (which
    cannot couple), and assemble() stops with the fix to make instead of an internal TypeError."""
    register_builtin_solvers()
    arm_xml = tmp_path / "arm.xml"
    arm_xml.write_text(
        '<mujoco><worldbody><body name="link" pos="0 0 0.5">'
        '<joint name="j0" type="hinge" axis="0 1 0"/>'
        '<geom type="box" size="0.05 0.05 0.05" mass="1.0"/>'
        '</body></worldbody></mujoco>',
        encoding="utf-8",
    )
    (tmp_path / "meshes").mkdir()
    (tmp_path / "objects").mkdir()
    mesh_bytes = b"v 0 0 0\nv 0.2 0 0\nv 0.2 0.2 0\nv 0 0.2 0\nf 1 2 3\nf 1 3 4\n"
    (tmp_path / "meshes" / "patch.obj").write_bytes(mesh_bytes)
    cloth_card = {
        "schema": "alloy.object/1",
        "name": "aniso_mesh_cloth",
        "material": "cotton_woven_aniso",
        "motion": "dynamic",
        "shape": {"mesh": {"path": "meshes/patch.obj", "sha256": hashlib.sha256(mesh_bytes).hexdigest()}},
        "features": {"thickness": {"value": 0.002, "source": "model", "note": "test sheet"}},
    }
    (tmp_path / "objects" / "aniso_mesh_cloth.yaml").write_text(yaml_dump(cloth_card), encoding="utf-8")
    lib = CardLibrary([tmp_path, default_root()])
    card = SceneCard(
        robots=(RobotRef(path=str(arm_xml), kind="mjcf", base_pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), name="arm"),),
        objects=(Placement(object="aniso_mesh_cloth", pose=(0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0)),),
        worlds=1,
    )
    plan = SceneAssembler(lib).plan(card)
    dec = plan.report.decisions["aniso_mesh_cloth"]
    assert dec.route == Route.CLOTH_VBD, dec
    with pytest.raises(AssemblyError, match="orthotropic membrane"):
        SceneAssembler(lib).assemble(card, "cpu")


def test_object_material_disambiguates_colliding_slugs(tmp_path):
    """Objects whose names collapse to the same slug must materialize distinct material cards with own friction."""
    (tmp_path / "materials").mkdir()
    item_a = {"kind": "box", "material": "steel_carbon", "overrides": {"friction": 0.2}, "randomize": [],
              "size": (0.1, 0.1, 0.1)}
    item_b = {"kind": "box", "material": "steel_carbon", "overrides": {"friction": 0.8}, "randomize": [],
              "size": (0.1, 0.1, 0.1)}
    m1 = mat._object_material("peg-1", item_a, tmp_path, [], (set(), False), yaml_dump)
    m2 = mat._object_material("peg_1", item_b, tmp_path, [], (set(), False), yaml_dump)
    assert m1 != m2
    d1 = yaml.safe_load((tmp_path / "materials" / f"{m1}.yaml").read_text(encoding="utf-8"))
    d2 = yaml.safe_load((tmp_path / "materials" / f"{m2}.yaml").read_text(encoding="utf-8"))
    assert d1["surface"]["mu_dynamic"]["value"] == 0.2
    assert d2["surface"]["mu_dynamic"]["value"] == 0.8
