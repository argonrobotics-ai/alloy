# SPDX-License-Identifier: Apache-2.0
"""Regression tests for MJCF writer error import, plane/connect export, and USD linear joint scaling."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy._vendor.newton import GeoType
from alloy.io import from_model, mjcf_write, usd_units
from alloy.io.errors import ImportFileError
from alloy.io.groups import Group
from alloy.io.model import IOBody, IOModel, IOShape


def test_write_group_mjcf_raises_import_file_error_on_invalid_xml(tmp_path):
    """write_group_mjcf must raise ImportFileError (not NameError) when MuJoCo rejects the written XML."""
    pytest.importorskip("mujoco")
    io = IOModel(name="bad")
    body = IOBody(name="b", pose=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), mass=1.0,
                  shapes=[IOShape(name="s", kind="sphere", size=(-1.0,))])
    io.bodies.append(body)
    grp = Group(name="bad", bodies=[body], root=body, fixed_base=False)
    with pytest.raises(ImportFileError, match="does not load in MuJoCo"):
        mjcf_write.write_group_mjcf(io, grp, tmp_path / "bad.xml")


def test_from_model_plane_half_extents_and_world_first_connect_anchor():
    """GeoType.PLANE exports half-extents and world-first <connect> anchors transform into body1's frame."""
    class _FakeArr(dict):
        m = SimpleNamespace(shape_source=[None], mujoco=None)

    arr = _FakeArr(
        shape_type=[int(GeoType.PLANE)],
        shape_scale=[[4.0, 6.0, 0.0]],
        shape_flags=[1],
        shape_transform=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
        shape_material_ke=[0.0],
        shape_material_kd=[0.0],
    )
    shape = from_model._shape(arr, 0, "floor")
    assert shape is not None and shape.kind == "plane"
    assert shape.size == (2.0, 3.0)

    io = IOModel(name="t")
    io.bodies.append(IOBody(name="b0", pose=(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0), mass=1.0))
    ns = SimpleNamespace(
        equality_constraint_count=1,
        equality_constraint_type=SimpleNamespace(numpy=lambda: np.array([from_model._CONNECT], np.int32)),
        equality_constraint_body1=SimpleNamespace(numpy=lambda: np.array([-1], np.int32)),
        equality_constraint_body2=SimpleNamespace(numpy=lambda: np.array([0], np.int32)),
        equality_constraint_anchor=SimpleNamespace(numpy=lambda: np.array([[1.5, 2.0, 3.0]], np.float64)),
    )
    from_model._equalities(SimpleNamespace(mujoco=ns), ["b0"], io)
    assert len(io.equalities) == 1
    assert io.equalities[0].body1 == "b0" and io.equalities[0].body2 is None
    np.testing.assert_allclose(io.equalities[0].anchor[:3], [0.5, 0.0, 0.0], atol=1e-9)


def test_usd_units_scales_cm_prismatic_joint_attributes(tmp_path):
    """A centimetre USD stage must scale linear drive targets/states and prismatic limits/spring refs by 0.01."""
    pytest.importorskip("pxr")
    usda = tmp_path / "slider_cm.usda"
    usda.write_text(
        '#usda 1.0\n(\n    metersPerUnit = 0.01\n    upAxis = "Z"\n)\n'
        'def Xform "World" {\n'
        '    def PhysicsPrismaticJoint "slide" (\n'
        '        prepend apiSchemas = ["PhysicsDriveAPI:linear"]\n'
        '    ) {\n'
        '        float physics:lowerLimit = -10.0\n'
        '        float physics:upperLimit = 20.0\n'
        '        float drive:linear:physics:targetPosition = 15.0\n'
        '        float drive:linear:physics:targetVelocity = 50.0\n'
        '        float state:linear:physics:position = 5.0\n'
        '        float state:linear:physics:velocity = 25.0\n'
        '        float newton:velocityLimit = 100.0\n'
        '        float physxJoint:maxJointVelocity = 80.0\n'
        '        float mjc:springref = 4.0\n'
        '        float mjc:ref = 2.0\n'
        '    }\n'
        '}\n',
        encoding="utf-8",
    )
    stage = usd_units.metre_source(usda)
    prim = stage.GetPrimAtPath("/World/slide")
    assert prim.GetAttribute("drive:linear:physics:targetPosition").Get() == pytest.approx(0.15)
    assert prim.GetAttribute("drive:linear:physics:targetVelocity").Get() == pytest.approx(0.50)
    assert prim.GetAttribute("state:linear:physics:position").Get() == pytest.approx(0.05)
    assert prim.GetAttribute("state:linear:physics:velocity").Get() == pytest.approx(0.25)
    assert prim.GetAttribute("newton:velocityLimit").Get() == pytest.approx(1.0)
    assert prim.GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(0.8)
    assert prim.GetAttribute("mjc:springref").Get() == pytest.approx(0.04)
    assert prim.GetAttribute("mjc:ref").Get() == pytest.approx(0.02)
