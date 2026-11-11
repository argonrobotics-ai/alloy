# SPDX-License-Identifier: Apache-2.0
"""Light, heavily preloaded slide pads (DC3's C13 contacts): armature regularisation keeps the plug's mass, and
``stop_stiffness`` makes the rest stop stiff without body mass (MJCF in MuJoCo C, and add_latch via SolverMuJoCo)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from alloy._vendor import newton

from alloy.latch.builder import add_latch
from alloy.latch.design import check_spec
from alloy.latch.mass import member_armature
from alloy.latch.mjcf import LIMIT_MARGIN, joint_impedance, joint_name, latch_mjcf, stop_inv_weights, total_mass
from alloy.latch.spec import Box, LatchSpec, Member, Spring

DT = 1.25e-3
PRELOAD, RATE, PAD_G = 6.84, 1000.0, 2.0e-3          # DC3 C13 pad: N, N/m, kg
K_STOP = 1.0e5                                       # N/m


def _pad_spec(regularise: str = "armature", stop_stiffness: float | None = K_STOP) -> LatchSpec:
    pad = Member("pad", "slide", anchor=(0.006, 0.0, 0.0), axis=(1.0, 0.0, 0.0), travel=0.002,
                 spring=Spring(RATE, PRELOAD, 5.0), shapes=(Box("pad_face", (0.001, 0.003, 0.004), role="pad"),),
                 mass=PAD_G, regularise=regularise, stop_stiffness=stop_stiffness)
    return LatchSpec("c13_pad_test", root_shapes=(Box("body", (0.004, 0.008, 0.01), role="body"),),
                     root_mass=0.022, members=(pad,), design_dt=DT)


def _rest_q(xml: str, spec: LatchSpec, seconds: float = 0.3) -> float:
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.timestep, m.opt.gravity[:] = DT, 0.0
    d = mujoco.MjData(m)
    for _ in range(int(seconds / DT)):
        mujoco.mj_step(m, d)
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name(spec, "pad"))
    return float(d.qpos[m.jnt_qposadr[j]])


def test_armature_keeps_the_physical_mass():
    spec = _pad_spec()
    assert total_mass(spec) == pytest.approx(0.022 + PAD_G)
    assert member_armature(spec.members[0], DT) == pytest.approx(RATE * (DT / 0.35) ** 2 - PAD_G)
    assert not [i for i in check_spec(spec) if i.level == "error"]
    assert total_mass(_pad_spec("mass")) > 0.022 + PAD_G + 0.01          # the mass route adds ~10.8 g


def test_stop_stiffness_holds_the_preload_at_the_stop():
    """Default stop on a 2 g pad: ~1 mm past the stop under 6.84 N (DC3 measured the same); stiff stop: <= F/K."""
    soft = _rest_q(latch_mjcf(_pad_spec(stop_stiffness=None)), _pad_spec(stop_stiffness=None))
    spec = _pad_spec()
    stiff = _rest_q(latch_mjcf(spec), spec)
    assert soft < -0.3e-3
    assert -PRELOAD / K_STOP <= stiff <= LIMIT_MARGIN          # the margin shifts the rest in by <= 1e-5


def test_inverse_weight_is_an_upper_bound():
    """The metric compile has no armature: the free-root value bounds the compiled one with armature."""
    spec = _pad_spec()
    a_up = stop_inv_weights(spec)["pad"]
    m = mujoco.MjModel.from_xml_string(latch_mjcf(spec))
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name(spec, "pad"))
    assert m.dof_invweight0[m.jnt_dofadr[j]] < a_up
    assert a_up == pytest.approx(1.0 / PAD_G, rel=0.5)


def test_builder_carries_the_same_armature_and_stop():
    """add_latch -> SolverMuJoCo (MuJoCo C): same armature and solimplimit as the MJCF, same rest offset."""
    spec = _pad_spec()
    arm, si, _ = joint_impedance(spec)["pad"]
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    h = add_latch(b, spec)
    model = b.finalize(device="cpu")
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_cpu=True, integrator="implicitfast")
    mm = solver.mj_model
    j = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, joint_name(spec, "pad"))
    assert mm.dof_armature[mm.jnt_dofadr[j]] == pytest.approx(arm, rel=1e-5)
    assert np.allclose(mm.jnt_solimp[j], si, rtol=1e-5)
    s0, s1, ctrl = model.state(), model.state(), model.control()
    for _ in range(int(0.3 / DT)):
        solver.step(s0, s1, ctrl, None, DT)
        s0, s1 = s1, s0
    q = float(s0.joint_q.numpy()[h.coords["pad"]])
    assert -PRELOAD / K_STOP <= q <= LIMIT_MARGIN
