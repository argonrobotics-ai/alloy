# SPDX-License-Identifier: Apache-2.0
"""LIQ-6 (SPEC §5.2): Torricelli drain through a thin-plate orifice of 8 particle diameters (r = 1 mm, 80 x 80 mm
tank, 100 mm deep, A_tank / A_orifice = 25). There is no rig C_d yet, so the run reports the discharge coefficient
the simulation implies (fit of sqrt(H) vs t between 0.8 and 0.2 H0) and the drain time; the acceptance row stays
not_run until R-rig data gives C_d. Results: ``$ALLOY_WP9_RESULTS/wp9_drain_gpu.json``."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from alloy._vendor.newton import ParticleFlags

from alloy.testing.sph import drain_tank, write_result

pytestmark = [pytest.mark.gpu, pytest.mark.timeout(3600)]
DT = 0.02 / 128
SAMPLE = 64  # steps between samples (10 ms)


def _height(sc, state) -> float:
    q = state.particle_q.numpy()
    active = (sc.model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
    inside = active & (q[:, 2] > 0.0)
    m = sc.model.particle_mass.numpy()
    return float(m[inside].sum() / 1000.0 / sc.base_area)


def test_liq6_drain_reports_discharge_coefficient():
    if not wp.is_cuda_available():
        pytest.skip("needs CUDA")
    sc = drain_tank(device="cuda:0")
    s = sc.make_solver()
    a, b = sc.model.state(), sc.model.state()
    s.prime(a, DT)
    h0 = _height(sc, a)
    t, h = [0.0], [h0]
    for k in range(1, 6 * 128 * 50 + 1):
        s.step(a, b, None, None, DT)
        a, b = b, a
        if k % SAMPLE == 0:
            t.append(k * DT)
            h.append(_height(sc, a))
            if h[-1] < 0.15 * h0:
                break
    t, h = np.array(t), np.array(h)
    sel = (h <= 0.8 * h0) & (h >= 0.2 * h0)
    slope = np.polyfit(t[sel], np.sqrt(h[sel]), 1)[0]
    a_ratio = sc.base_area / sc.bodies["hole"] ** 2
    cd = -slope * a_ratio / np.sqrt(9.81 / 2.0)
    t20 = float(np.interp(0.2 * h0, h[::-1], t[::-1]))
    t20_cd061 = a_ratio / 0.61 * (np.sqrt(2 * h0 / 9.81) - np.sqrt(2 * 0.2 * h0 / 9.81))
    r = s.report()
    out = {"h0_m": h0, "area_ratio": a_ratio, "cd_fit": float(cd), "t_to_0.2H0_s": t20, "t_pred_cd0.61_s": t20_cd061,
           "escaped": int(r.escaped[0]), "series_t": t.tolist()[::5], "series_h": h.tolist()[::5],
           "ledger": {k: v.tolist() for k, v in s.mass_ledger().items()}}
    write_result("wp9_drain_gpu", out)
    assert 0.3 < cd < 1.2  # a physical discharge coefficient (sharp-edged orifice ~0.6)
