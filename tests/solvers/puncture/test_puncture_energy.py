# SPDX-License-Identifier: Apache-2.0
"""PUN-3 no energy gain (design note §6): net = E − W_needle + D_fric + R never rises beyond round-off.

Converged settings (dt 1e-4 s, 20 iterations): at dt 1e-3 s / 20 iterations stock VBD itself gains energy on a
sagging block (no needle), so the law is checked where the solver is converged. A straight needle is driven 1.5 mm
into a 3 mm block and held.
(a) conservative (no friction, cutting, damping or gravity): every step gain and the rise of net stay at round-off;
(b) full model: step gain ≤ 1e-3 E_scale, rise of net ≤ 1e-2 E_scale (E_scale = the D20 floor, 1 mJ);
(c) events never add energy: the booked release R never drops by more than 10 nJ in a step.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.solvers.puncture import PunctureConfig, SolverPuncture, TissueMaterial
from alloy.solvers.puncture.scenes import ROW, Block, Drive, build_block

pytestmark = pytest.mark.compile_heavy
C = {n: i for i, n in enumerate(ROW)}
E_SCALE = 1.0e-3
DT, SPEED, DEPTH, HOLD = 1e-4, 4e-2, 1.5e-3, 0.02


def _run(conservative: bool) -> np.ndarray:
    mat = TissueMaterial(fp=0.3, fcut=0.0, flen=0.0) if conservative else TissueMaterial(fp=0.3, fcut=0.1, flen=50.0)
    bl = Block(size=3e-3, thick=3e-3, material=mat, gravity=0.0 if conservative else 9.81)
    with wp.ScopedDevice("cpu"):
        m, specs = build_block(bl)
        s = SolverPuncture(m, puncture=PunctureConfig(), needles=specs, iterations=20,
                           integrate_with_external_rigid_solver=True)
        d = Drive(m, s, DT)
        t_in = (bl.gap + DEPTH) / SPEED
        for _ in range(round((t_in + HOLD) / DT)):
            d.step(lambda t: SPEED * min(t, t_in), bl.gap)
        return d.table()


@pytest.mark.parametrize("conservative", [True, False], ids=["conservative", "full"])
def test_pun3_no_energy_gain(conservative: bool) -> None:
    r = _run(conservative)
    assert r[-1, C["punctures"]] == 1 and r[-1, C["track"]] == 2 and r[-1, C["flags"]] == 0
    net = r[:, C["net"]]
    gain = np.diff(net).max()
    rise = net.max()
    if conservative:
        assert gain < 1e-9 and rise < 1e-9, (gain, rise)
    else:
        assert gain < 1e-3 * E_SCALE and rise < 1e-2 * E_SCALE, (gain, rise)
        assert r[-1, C["D"]] > 0.0
    assert np.diff(r[:, C["R"]]).min() > -1e-8
