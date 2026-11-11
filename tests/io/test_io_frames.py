# SPDX-License-Identifier: Apache-2.0
"""Joint frames survive the URDF and USD writers: a two-link arm whose joint frames are offset and rotated from the
body frames lands at the same body poses when MuJoCo or Newton evaluates it at q = 0."""

from __future__ import annotations

import json

import numpy as np
import pytest

from alloy.io.geometry import compose, inverse, quat_matrix
from alloy.io.urdf_write import write_urdf
from alloy.io.usd_write import write_usd
from alloy.testing.io import arm_model

mujoco = pytest.importorskip("mujoco")


def _same_pose(a, b, tol=1e-5):
    np.testing.assert_allclose(a[:3], b[:3], atol=tol)
    np.testing.assert_allclose(quat_matrix(a[3:]), quat_matrix(b[3:]), atol=tol)


def test_urdf_arm_in_mujoco(tmp_path):
    io = arm_model()
    index = write_urdf(io, tmp_path)
    g = index["groups"][0]
    assert g["name"] == "arm" and g["fixed_base"] and g["articulated"]
    m = mujoco.MjModel.from_xml_path(str(tmp_path / g["file"]))
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    base = io.body("base").pose
    for name in ("link1", "link2"):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        x, wxyz = d.xpos[i], d.xquat[i]
        rel = compose(inverse(base), io.body(name).pose)     # URDF root = the base link frame
        # URDF link frames are joint frames; child_pose is the joint frame in the body: body = link * child_pose^-1
        link = compose((*x, *wxyz[1:], wxyz[0]), inverse(io.joint_of(name).child_pose))
        _same_pose(link, rel)
        assert m.body_mass[i] == pytest.approx(io.body(name).mass, rel=1e-6)
    j2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "j2")
    assert m.jnt_range[j2] == pytest.approx([-2.0, 2.0]) and m.dof_damping[m.jnt_dofadr[j2]] == pytest.approx(0.1)


def test_usd_arm_in_newton(tmp_path):
    pytest.importorskip("pxr")
    pytest.importorskip("newton_usd_schemas")
    from alloy._vendor import newton

    io = arm_model()
    path = write_usd(io, tmp_path / "arm.usda")
    b = newton.ModelBuilder()
    b.add_usd(str(path))
    model = b.finalize("cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    labels = [lbl.split("/")[-1] for lbl in model.body_label]
    q = state.body_q.numpy()
    for name in ("base", "link1", "link2"):
        _same_pose(q[labels.index(name)], io.body(name).pose)
    lo = model.joint_limit_lower.numpy()
    assert sorted(np.round(lo, 5))[:2] == [-2.0, -1.5]
    assert json.dumps(sorted(labels)) == json.dumps(["base", "link1", "link2"])
