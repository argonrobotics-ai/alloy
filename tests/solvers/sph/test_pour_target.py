# SPDX-License-Identifier: Apache-2.0
"""Bully B0 pour scene ``sph_pour``: registered on import, MuJoCo cup -> SolverDFSPH proxy, follows its tilt target and
keeps the liquid in bounds over a short scripted tilt (CPU, 1 world)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy.solvers.sph  # noqa: F401  (registers the target)
from alloy.bully.actions import global_action_index
from alloy.bully.targets import get_target


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_sph_pour_target_tilts_and_keeps_liquid():
    target = get_target("sph_pour")
    assert target.actions.dim == 2 and target.action_kind == "joint_target_q"
    sim = target.make_sim(1, "cpu")
    idx = global_action_index(sim.model, target.action_kind, target.action_index)
    sph = sim.solver.solver("sph")
    for k in range(8):
        cmd = sim.control.joint_target_q.numpy()
        cmd[idx[0, 1]] = min(0.15 * (k + 1), target.actions.high[1])
        sim.control.joint_target_q.assign(cmd)
        sim.step_period()
    q = sim.state.joint_q.numpy()
    assert 0.6 < q[idx[0, 1]] < 1.3  # the hinge follows the ramp (lagging the 1.2 rad target)
    r = sph.report()
    assert r.escaped[0] == 0 and np.isfinite(r.max_speed[0]) and r.max_speed[0] < 5.0
    assert np.isfinite(sim.state.particle_q.numpy()).all()
