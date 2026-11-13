# SPDX-License-Identifier: Apache-2.0
"""alloy.isaaclab.bridge and the step rule, without Isaac Lab: a model of Alloy's own Newton dressed in Newton 1.6.1's
attribute names stands in for the model Isaac Lab 3.0.0 builds."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton as ven
from alloy.isaaclab.bridge import RENAMES, bridge_model, resync_arrays
from alloy.isaaclab.solver import substeps_for

CARTPOLE = """<mujoco><worldbody><body name="cart" pos="0 0 1"><joint name="slide" type="slide" axis="1 0 0"
 damping="0.1"/><geom type="box" size=".2 .1 .1" mass="1"/><body name="pole"><joint name="hinge" type="hinge"
 axis="0 1 0" range="-90 90" armature="0.01"/><geom type="capsule" fromto="0 0 0 0 0 .6" size=".02" mass=".2"/>
 </body></body></worldbody><actuator><motor joint="slide" gear="2"/></actuator></mujoco>"""
MJ_FIELDS = ("body_mass", "body_inertia", "geom_size", "geom_friction", "jnt_range", "dof_damping", "dof_armature",
             "actuator_gear", "actuator_gainprm", "actuator_biasprm", "geom_solref", "geom_solimp", "body_pos")


def _native(worlds: int = 3) -> ven.Model:
    b = ven.ModelBuilder()
    ven.solvers.SolverMuJoCo.register_custom_attributes(b)
    b.add_mjcf(CARTPOLE)
    w = ven.ModelBuilder()
    ven.solvers.SolverMuJoCo.register_custom_attributes(w)
    w.replicate(b, worlds)
    return w.finalize("cpu")


def _as_newton_161(m: ven.Model) -> SimpleNamespace:
    """The same model under Newton 1.6.1's names: renamed attributes back, mimic as a constraint table, stiffness and
    springref in the mujoco namespace, and the MuJoCo attributes 1.6.1 does not declare left out."""
    back = {new: old for old, new in RENAMES.items()}
    host = {back.get(k, k): v for k, v in vars(m).items() if k not in ("joint_mimic_joint", "joint_mimic_coeffs",
                                                                         "joint_stiffness", "joint_rest_q", "mujoco")}
    ns = SimpleNamespace(**{k: v for k, v in vars(m.mujoco).items() if not k.startswith(("flag_", "actuator_d",
                                                                                           "actuator_armature"))})
    ns.dof_passive_stiffness, ns.dof_springref = m.joint_stiffness, m.joint_rest_q
    host.update(mujoco=ns, constraint_mimic_count=0)
    return SimpleNamespace(**host)


def _mj(model: ven.Model):
    return ven.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, separate_worlds=True).mj_model


def test_bridged_161_model_gives_the_same_mujoco_model():
    native = _native()
    host = _as_newton_161(native)
    bridged, report = bridge_model(host)
    a, b = _mj(bridged), _mj(native)
    for f in ("nq", "nv", "nu", "nbody", "ngeom", "njnt"):
        assert getattr(a, f) == getattr(b, f), f
    for f in MJ_FIELDS:
        np.testing.assert_allclose(getattr(a, f), getattr(b, f), err_msg=f)
    assert report.renamed["rigid_contact_max"] == "_rigid_contact_max"
    assert "mujoco:flag_contact" in report.defaulted


def test_bridge_shares_arrays_and_follows_replaced_ones():
    host = _as_newton_161(_native())
    bridged, _ = bridge_model(host)
    assert bridged.body_mass is host.body_mass            # shared, not copied: a write by Isaac Lab is seen
    host.body_mass = wp.clone(host.body_mass)              # Isaac Lab replaced the array
    assert resync_arrays(host, bridged) >= 1
    assert bridged.body_mass is host.body_mass


def test_mimic_table_becomes_per_joint():
    native = _native(worlds=1)
    host = _as_newton_161(native)
    host.constraint_mimic_count = 1
    host.constraint_mimic_joint0 = wp.array([1], dtype=wp.int32)
    host.constraint_mimic_joint1 = wp.array([0], dtype=wp.int32)
    host.constraint_mimic_coef0 = wp.array([0.5], dtype=wp.float32)
    host.constraint_mimic_coef1 = wp.array([-2.0], dtype=wp.float32)
    host.constraint_mimic_enabled = wp.array([True], dtype=wp.bool)
    bridged, _ = bridge_model(host)
    assert bridged.joint_mimic_joint.numpy().tolist() == [-1, 0]
    np.testing.assert_allclose(bridged.joint_mimic_coeffs.numpy()[1], [0.5, -2.0])


@pytest.mark.parametrize(("rule", "dt", "n"), [("alloy", 1 / 120, 8), ("alloy", 0.005, 4), ("alloy", 1 / 600, 1),
                                               ("task", 0.005, 1), (None, 0.02, 1), (0.002, 0.005, 3)])
def test_alloy_step_rule(rule, dt, n):
    assert substeps_for(rule, dt) == n


def test_bad_step_rule_says_what_is_allowed():
    with pytest.raises(ValueError, match="'alloy', 'task' or a positive number"):
        substeps_for(-1.0, 0.005)


def test_noslip_with_the_discrete_integrator_refuses_to_run():
    """MuJoCo Warp has no noslip under integrator='discrete': the run stops instead of stepping without it."""
    from alloy.isaaclab.solver import AlloyRigidSolver

    cfg = SimpleNamespace(integrator="discrete", noslip_iterations=100, health="off", use_mujoco_cpu=True)
    with pytest.raises(NotImplementedError, match="no noslip for the discrete integrator"):
        AlloyRigidSolver(_as_newton_161(_native(1)), cfg)


def test_log_line_names_the_step():
    from alloy.isaaclab.solver import step_note

    assert "the task's own" in step_note("task") and "the task's own" in step_note(None)
    assert "1/600 s" in step_note("alloy") and "0.002 s" in step_note(0.002)
