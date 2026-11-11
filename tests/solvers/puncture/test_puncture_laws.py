# SPDX-License-Identifier: Apache-2.0
"""PUN-1 (puncture at the threshold) and PUN-2 (insertion profile) on a small silicone block, CPU.

A straight needle is driven down at 10 mm/s. Before contact nothing acts; on contact a TIP anchor dimples the surface;
the step that converts it is the first whose start-of-step axial force reaches fp; afterwards the resistance is the
cutting force plus friction, growing with the inserted length at flen.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from alloy.solvers.puncture import PunctureConfig, SolverPuncture, TissueMaterial
from alloy.solvers.puncture.scenes import ROW, Block, Drive, build_block

pytestmark = pytest.mark.compile_heavy
MAT = TissueMaterial(fp=0.5, fcut=0.15, flen=50.0)
BLOCK = Block(size=6e-3, thick=5e-3, material=MAT)
CFG = PunctureConfig()
SPEED, DT, DEPTH = 1e-2, 1e-3, 4e-3
C = {n: i for i, n in enumerate(ROW)}


@pytest.fixture(scope="module")
def insertion() -> np.ndarray:
    with wp.ScopedDevice("cpu"):
        m, specs = build_block(BLOCK)
        s = SolverPuncture(m, puncture=CFG, needles=specs, iterations=20, integrate_with_external_rigid_solver=True)
        d = Drive(m, s, DT)
        steps = round((BLOCK.gap + DEPTH) / SPEED / DT)
        for _ in range(steps):
            d.step(lambda t: SPEED * t, BLOCK.gap)
        return d.table()


def test_pun1_puncture_fires_at_threshold(insertion: np.ndarray) -> None:
    r = insertion
    track = r[:, C["track"]].astype(int)
    assert r[-1, C["punctures"]] == 1 and r[-1, C["flags"]] == 0
    k = int(np.flatnonzero((track[:-1] == 1) & (track[1:] == 2))[0]) + 1     # first row inside
    assert MAT.fp <= r[k, C["fax"]] <= 1.02 * MAT.fp                            # converted at fp (start of step k)
    assert r[:k, C["fax"]].max() < MAT.fp                                      # never converted early
    assert r[k - 1, C["axial"]] >= MAT.fp                                      # ... on the first step that reached it
    assert r[:, C["axial"]].max() <= MAT.fp * 1.02                             # one step of overshoot at most
    assert r[k, C["axial"]] < 0.8 * MAT.fp                                     # the drop


def test_pun2_insertion_force_is_cut_plus_friction(insertion: np.ndarray) -> None:
    """Once inside, resistance = fcut + flen x inserted length: slope within 10 % of flen, intercept within 25 % of
    fcut (anchors own h_a of material, scaled by the tissue stretch along the needle: Chentanez's fperlen·l)."""
    r = insertion
    steady = np.r_[False, np.diff(r[:, C["anchors"]]) == 0]          # a new anchor slips partially in its first step
    rows = r[(r[:, C["track"]] == 2) & steady]
    rows = rows[rows[:, C["length"]] > rows[0, C["length"]] + 0.5e-3]
    slope, icpt = np.polyfit(rows[:, C["length"]], rows[:, C["axial"]], 1)
    assert abs(slope / MAT.flen - 1.0) < 0.10, slope
    assert abs(icpt / MAT.fcut - 1.0) < 0.25, icpt
