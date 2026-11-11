# SPDX-License-Identifier: Apache-2.0
"""SCR-WP9-ROD R3c: tension's geometric stiffness in SolverVBDRod's chain Hessian, CPU.

A taut, light rope (A10's: r 1 mm, 7,850 kg/m³, about 0.5 g segments), gravity off, pulled at both end segments by
T, with a 1 mN lateral kick at the middle for five steps, at the production 1.25 ms. Plain Gauss-Newton blocks
treat tension explicitly on segment rotation, and the rope diverges once T h passes I_t / dt² (here between 0.3 and
1 N). With the geometric term it stays finite to 10 N.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy._vendor import newton
from alloy.scene import analytic_capstan as AC
from alloy.scene.analytic_rods import rod_card
from alloy.scene.analytic_rods_vbd import vbd_card
from alloy.scene.emitters.rod_vbd import build_rod_vbd
from alloy.solvers.rod import SolverVBDRod

DT = 1.25e-3


def _pull(T: float, geometric: bool, seconds: float = 0.1) -> bool:
    """True when the rope stays finite and within 1 mm of its line."""
    w = AC.wrap()
    rod = vbd_card(rod_card(w.length, AC.EI, w.links, radius=AC.ROPE_R, density=AC.DENSITY), chain=True)
    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    h = build_rod_vbd(b, rod, [(0.0, 0.0, 1.0), (w.length, 0.0, 1.0)], label="rope")
    b.color()
    model = b.finalize(device="cpu")
    solver = SolverVBDRod(model, iterations=4, geometric_stiffness=geometric)
    s0, s1, c = model.state(), model.state(), model.control()
    f = np.zeros((model.body_count, 6), np.float32)
    f[h.bodies[0], 0], f[h.bodies[-1], 0] = -T, T
    mid = h.bodies[len(h.bodies) // 2]
    for k in range(int(round(seconds / DT))):
        g = f.copy()
        if k < 5:
            g[mid, 2] = 1e-3
        s0.clear_forces()
        s0.body_f.assign(g)
        solver.step(s0, s1, c, None, DT)
        s0, s1 = s1, s0
    q = s0.body_q.numpy()[list(h.bodies)]
    return bool(np.isfinite(q).all() and np.abs(q[:, 2] - 1.0).max() < 1e-3)


@pytest.mark.law
@pytest.mark.parametrize("T", [1.0, 10.0])
def test_a_taut_light_rope_stays_on_its_line_at_the_production_step(T):
    assert _pull(T, geometric=True)


@pytest.mark.law
def test_plain_gauss_newton_blocks_let_it_diverge():
    assert _pull(0.3, geometric=False)                  # below the threshold the plain blocks hold too
    assert not _pull(1.0, geometric=False)
