# SPDX-License-Identifier: Apache-2.0
"""The MJCF piece and add_latch build the same mechanism (masses, inertias, joint springs, stops, excludes)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.latch.builder import add_latch
from alloy.latch.mjcf import latch_mjcf
from alloy.testing.latch import TabRig, tab_spec


def _model_from_mjcf(tmp_path, spec, gap=None):
    path = tmp_path / f"{spec.name}.xml"
    path.write_text(latch_mjcf(spec, gap=gap))
    b = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(str(path), floating=True)
    return b, b.finalize(device="cpu")


def _model_from_builder(spec):
    b = newton.ModelBuilder()
    h = add_latch(b, spec)
    return b, b.finalize(device="cpu"), h


def _dof_attr(model, name):
    return np.asarray(getattr(model.mujoco, name).numpy(), np.float64)


def test_mjcf_and_builder_agree(tmp_path):
    spec = tab_spec(TabRig())
    bm, mm = _model_from_mjcf(tmp_path, spec)
    bb, mb, h = _model_from_builder(spec)
    assert mm.body_count == mb.body_count == 2 and mm.joint_dof_count == mb.joint_dof_count == 7
    assert np.allclose(mm.body_mass.numpy(), mb.body_mass.numpy(), rtol=1e-6)
    assert np.allclose(mm.body_inertia.numpy(), mb.body_inertia.numpy(), rtol=1e-5, atol=1e-14)
    assert np.allclose(mm.body_com.numpy(), mb.body_com.numpy(), atol=1e-9)
    d, q = h.dofs["tab"], h.coords["tab"]
    m = spec.member("tab")
    for model in (mm, mb):                       # Newton 1.8: core joint spring (stiffness per dof, rest per coord)
        assert model.joint_stiffness.numpy()[d] == pytest.approx(m.spring.rate, rel=1e-6)
        assert model.joint_rest_q.numpy()[q] == pytest.approx(m.spring.springref, rel=1e-6)
    assert mm.joint_limit_lower.numpy()[d] == pytest.approx(0.0, abs=1e-9)
    assert mm.joint_limit_upper.numpy()[d] == pytest.approx(mb.joint_limit_upper.numpy()[d], rel=1e-6)
    assert np.allclose(_dof_attr(mm, "solreflimit")[d], _dof_attr(mb, "solreflimit")[d], rtol=1e-6)
    assert int(mm.mujoco.solreflimit_mode.numpy()[d]) == int(mb.mujoco.solreflimit_mode.numpy()[d]) == 1


def test_builder_excludes_every_cross_part_pair():
    spec = tab_spec(TabRig())
    b, m, h = _model_from_builder(spec)
    pairs = {tuple(sorted(p)) for p in b.shape_collision_filter_pairs}
    root = [h.shapes[x.name] for x in spec.root_shapes]
    tab = [h.shapes[x.name] for x in spec.member("tab").shapes]
    assert all(tuple(sorted((a, c))) in pairs for a in root for c in tab)
    assert set(h.roles["catch"]) == {h.shapes["rig_tab_catch"]} and h.dofs["tab"] == 6


def test_builder_slide_mount_has_a_position_drive():
    b = newton.ModelBuilder()
    h = add_latch(b, tab_spec(TabRig()), wp.transform_identity(), mount="slide", stage=(1500.0, 5.0))
    m = b.finalize(device="cpu")
    assert h.stage_dof == 0 and h.dofs["tab"] == 1
    assert m.joint_target_ke.numpy()[0] == pytest.approx(1500.0)
    assert int(m.joint_target_mode.numpy()[0]) == int(newton.JointTargetMode.POSITION)


def test_mjcf_gap_reaches_every_shape(tmp_path):
    """``gap`` sets the imported shapes' contact gap (SCR-DC1-4); without it Newton's 0.1 m default stays and the
    text carries no <default>, so earlier generated files keep their bytes."""
    spec = tab_spec(TabRig())
    assert "<default>" not in latch_mjcf(spec)
    _, m = _model_from_mjcf(tmp_path, spec, gap=0.005)
    _, m0 = _model_from_mjcf(tmp_path, spec)
    assert np.allclose(m.shape_gap.numpy(), 0.005)
    assert np.all(m0.shape_gap.numpy() > 0.05)
