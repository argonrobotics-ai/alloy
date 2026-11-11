# SPDX-License-Identifier: Apache-2.0
"""PUN-8 exit: a straight needle through a slab pinned at its sides leaves through the far face by a second
puncture (Frick et al. 2001: "two characteristic peaks, corresponding to initial penetration and emergence"):
the far face gets a look-ahead TIP contact, tents, and is punctured at fp; the track anchors stay on the needle."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.solvers.puncture import PunctureConfig, SolverPuncture, TissueMaterial
from alloy.solvers.puncture.events import INSIDE, OUTSIDE, TIP_EXIT
from alloy.solvers.puncture.scenes import ROW, Block, Drive, build_block

pytestmark = pytest.mark.compile_heavy
C = {n: i for i, n in enumerate(ROW)}
MAT = TissueMaterial(fp=0.3, fcut=0.1, flen=50.0)


def test_pun8_exit_by_second_puncture() -> None:
    bl = Block(size=6e-3, thick=3e-3, material=MAT, fix_sides=True)
    with wp.ScopedDevice("cpu"):
        m, specs = build_block(bl)
        s = SolverPuncture(m, puncture=PunctureConfig(), needles=specs, iterations=20,
                           integrate_with_external_rigid_solver=True)
        d = Drive(m, s, 2.5e-4)
        for _ in range(round((bl.gap + 5e-3) / 1e-2 / 2.5e-4)):
            d.step(lambda t: 1e-2 * t, bl.gap)
        r = d.table()
        an, used = s.anchors, int(s.tracks.used.numpy()[0])
        x, ti = d.s0.particle_q.numpy(), m.tet_indices.numpy()
        p = np.einsum("jk,jkd->jd", an.w.numpy()[:used], x[ti[an.tet.numpy()[:used]]])
        X = d.s0.body_q.numpy()[0]
    tr = r[:, C["track"]].astype(int)
    assert r[-1, C["punctures"]] == 2 and r[-1, C["flags"]] == 0 and tr[-1] == OUTSIDE
    k = int(np.flatnonzero((tr[:-1] == TIP_EXIT) & (tr[1:] == OUTSIDE))[0]) + 1
    assert MAT.fp <= r[k, C["fax"]] <= 1.02 * MAT.fp                      # exit at the threshold
    inside_peak = r[tr == INSIDE, C["axial"]].max()
    assert r[k - 1, C["axial"]] > inside_peak                            # the emergence peak
    assert r[:, C["offset"]].max() < 0.1e-3                              # anchors on the needle (lateral)
    assert np.hypot(p[:, 0] - X[0], p[:, 1] - X[1]).max() < 0.05e-3
