# SPDX-License-Identifier: Apache-2.0
"""CT-8: realised contact stiffness. sdf mode: K_eff = ΔF/Δpen at 1-50 N vs declared K and vs K_real (D22);
hydroelastic: F(δ) vs the pressure-law closed form ``F = (kh/2)·A·δ`` (two equal kh in series, flat bottom face).
Within 20 %. Penetration is GapMonitor's exact distance at the end of each 0.1 s hold.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alloy.contact.precision import k_real_mujoco, tau_for_stiffness
from alloy.testing.contact import peg_hole_scene

from peg_harness import advance, l2_dt, make_run, record, set_wrench

pytestmark = pytest.mark.gpu
WORLDS = 4
LEVELS = [1.0, 5.0, 10.0, 25.0, 50.0]


def _pen_at(run, force: float) -> float:
    set_wrench(run, np.tile(np.array([0.0, 0.0, -force]), (WORLDS, 1)))
    advance(run, int(0.1 / run.dt))
    run.monitor.reset()
    advance(run, 10)
    return float(np.median(run.monitor.report().world_max_m))


def _flat_area(g) -> float:
    r, n = 0.5 * g.peg_d, g.segments
    return 0.5 * n * r * r * math.sin(2 * math.pi / n)


@pytest.mark.parametrize("native", ["mujoco", "newton"])
@pytest.mark.parametrize("mode", ["sdf", "hydroelastic"])
def test_ct8_stiffness(mode, native):
    scene = peg_hole_scene(WORLDS, device="cuda:0", mode=mode, start_gap=2e-5)
    run = make_run(scene, native, l2_dt(scene))
    advance(run, int(0.1 / run.dt))
    pens = np.array([_pen_at(run, f) for f in LEVELS])
    forces = np.asarray(LEVELS) + scene.peg_mass * 9.81
    slope = np.polyfit(pens, forces, 1)[0]
    k_decl = scene.spec.pair_stiffness
    k_real = k_real_mujoco(scene.peg_mass, scene.spec.mj_imp, run.dt, tau_for_stiffness(k_decl, scene.peg_mass,
                                                                                       scene.spec.mj_imp))
    k_hydro = 0.5 * scene.spec.kh * _flat_area(scene.geometry)
    target = k_decl if mode == "sdf" else k_hydro
    row = dict(metric="CT-8", mode=mode, native=native, dt=run.dt, pen_m=pens.tolist(), forces_n=forces.tolist(),
               k_eff=float(slope), k_target=float(target), k_real_formula=float(k_real),
               rel_err=float(slope / target - 1.0))
    record("ct8", row)
    assert abs(slope / target - 1.0) <= 0.2, row
    if mode == "sdf":
        assert abs(slope / k_real - 1.0) <= 0.2, row
