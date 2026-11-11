# SPDX-License-Identifier: Apache-2.0
"""PUN-7 default off (SCR-SC11-1): SolverPuncture without puncturable tets or without needles steps bitwise as
SolverVBDMomentum, and with the feature on but no anchor yet (needle above the block) the colour callback adds nothing:
bitwise Newton's loop. The callback's slot is SolverVBDAlloy's after_accumulate (SCR-SC11-2), whose own tripwire
(after_accumulate_hook_ok) checks Newton's loop. CPU, small."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from alloy.solvers.puncture import SolverPuncture, TissueMaterial
from alloy.solvers.puncture.scenes import Block, build_block
from alloy.solvers.vbd.base import after_accumulate_hook_ok
from alloy.solvers.vbd import SolverVBDMomentum

pytestmark = pytest.mark.compile_heavy
BLOCK = Block(size=4e-3, thick=3e-3, gap=2e-3)


def _run(model, make, steps: int = 10) -> tuple[np.ndarray, np.ndarray]:
    s0, s1 = model.state(), model.state()
    solver = make(model)
    for _ in range(steps):
        solver.step(s0, s1, None, None, 1.0e-3)
        s0, s1 = s1, s0
    return s0.particle_q.numpy(), s0.particle_qd.numpy()


def _vbd(m):
    return SolverVBDMomentum(m, iterations=6, integrate_with_external_rigid_solver=True)


@pytest.mark.parametrize("case", ["no_tissue", "no_needle", "needle_not_touching"])
def test_puncture_off_is_vbd_momentum(case: str) -> None:
    with wp.ScopedDevice("cpu"):
        bl = replace(BLOCK, material=TissueMaterial(0.0, 0.0, 0.0)) if case == "no_tissue" else BLOCK
        m_ref, _ = build_block(bl)
        q_ref, qd_ref = _run(m_ref, _vbd)
        m, specs = build_block(bl)
        needles = [] if case == "no_needle" else specs
        q, qd = _run(m, lambda mm: SolverPuncture(mm, needles=needles, iterations=6,
                                                  integrate_with_external_rigid_solver=True))
        assert np.array_equal(q, q_ref) and np.array_equal(qd, qd_ref)


def test_after_accumulate_slot_available_on_pinned_newton() -> None:
    assert after_accumulate_hook_ok()
