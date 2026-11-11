# SPDX-License-Identifier: Apache-2.0
"""Vendored patch 0013: Alloy's copy of Newton's SolverVBD compiles its kernels without fast math unless asked.

Newton release-1.8 turns Warp's ``fast_math`` on for the VBD particle kernels, the rigid solve and the joint dual
update (CUDA only; the CPU ignores it). On one L4 that alone moved the 64-world drape off the previous pin's bits
(peak strain 0.364 % to 0.543 %, 7 % faster); with it off the drape is the previous pin's bit for bit
(``alloy/_vendor/VENDORED.md``). The default is therefore False here, and ``enable_fast_math=True`` still works.
"""

from __future__ import annotations

import inspect

import warp as wp

from alloy._vendor import newton


def _sheet() -> newton.Model:
    b = newton.ModelBuilder()
    b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, 0.1), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=2, dim_y=2,
                     cell_x=0.05, cell_y=0.05, mass=0.01, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=0.0, edge_ke=1.0e-3)
    b.color()
    return b.finalize(device="cpu")


def test_the_vendored_vbd_solver_has_fast_math_off_by_default() -> None:
    default = inspect.signature(newton.solvers.SolverVBD.__init__).parameters["enable_fast_math"].default
    assert default is False


def test_fast_math_can_still_be_asked_for_and_both_solvers_step() -> None:
    model = _sheet()
    for kwargs in ({}, {"enable_fast_math": True}, {"enable_fast_math": False}):
        solver = newton.solvers.SolverVBD(model, iterations=2, **kwargs)
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, model.control(), None, 1.0e-3)
        assert s1.particle_q.numpy()[:, 2].max() < 0.1                 # one step of free fall, either way
