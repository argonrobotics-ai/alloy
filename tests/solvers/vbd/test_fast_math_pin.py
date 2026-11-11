# SPDX-License-Identifier: Apache-2.0
"""Pin for vendored patch 0013 (coordinator ruling 2026-11-10 01:42Z): Alloy's cloth solvers build Newton's VBD
particle kernels with Warp's ``fast_math`` off.

Newton release-1.8 turns it on for CUDA; that alone moved the GPU drape off the previous build's bits (peak strain
0.364 % to 0.543 %; ``alloy/_vendor/VENDORED.md``, patch 0013). The CPU ignores the option, so this checks the
kernel module's option itself; the bits are pinned on an L4 in ``test_drape_l4_gpu.py``.
"""

from __future__ import annotations

import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton._src.solvers.vbd import particle_vbd_kernels
from alloy.solvers.vbd import SolverVBDAlloy, SolverVBDMomentum


def _sheet() -> newton.Model:
    b = newton.ModelBuilder()
    b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, 0.1), rot=wp.quat_identity(), vel=wp.vec3(0.0, 0.0, 0.0), dim_x=2, dim_y=2,
                     cell_x=0.05, cell_y=0.05, mass=0.01, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=0.0, edge_ke=1.0e-3)
    b.color()
    return b.finalize(device="cpu")


def _fast_math() -> bool:
    return bool(wp.get_module(particle_vbd_kernels.__name__).options.get("fast_math", False))


def test_alloys_cloth_solvers_build_the_particle_kernels_without_fast_math() -> None:
    model = _sheet()
    newton.solvers.SolverVBD(model, iterations=2, enable_fast_math=True)
    assert _fast_math() is True                                    # the option is read where the kernels are built
    for cls in (newton.solvers.SolverVBD, SolverVBDAlloy, SolverVBDMomentum):
        cls(model, iterations=2)
        assert _fast_math() is False, cls.__name__
