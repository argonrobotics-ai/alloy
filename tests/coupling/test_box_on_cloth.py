# SPDX-License-Identifier: Apache-2.0
"""CPL-1 (interface force mismatch) and CPL-2 (kinematic mismatch) on the box-on-cloth law (SPEC §5.2)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.testing.coupling import cpl_force_mismatch, record_metric

T_SKIP = 0.1          # s, initial transient excluded from CPL-2 (coupling-time §3.2 convention)


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_cpl1_cpl2_box_on_cloth(cloth_run_02):
    log = cloth_run_02["runner"].log
    ratio = cpl_force_mismatch(log)
    med, p99 = float(np.median(ratio)), float(np.percentile(ratio, 99))
    t = np.cumsum(log.dt)
    keep = t >= T_SKIP
    rv = np.array(log.res_vel)[keep].ravel()
    rp = np.array(log.res_pos)[keep].ravel()
    v99, x99 = float(np.percentile(rv, 99)), float(np.percentile(rp, 99))
    dt_x = sorted(set(np.round(log.dt, 12)))
    record_metric("CPL-1", median=med, p99=p99, samples=int(ratio.size), dt_x=dt_x, box_mass=0.2)
    record_metric("CPL-2", vel_p99=v99, pos_p99=x99, samples=int(rv.size), dt_x=dt_x, box_mass=0.2)
    print(f"CPL-1 median {med:.3e} p99 {p99:.3e} (n={ratio.size}); CPL-2 vel p99 {v99:.3e} m/s, "
          f"pos p99 {x99:.3e} m; dt_x {dt_x}")
    assert ratio.size > 100
    assert med <= 0.01 and p99 <= 0.05                       # CPL-1
    assert v99 <= 1.0e-2 and x99 <= 1.0e-3                   # CPL-2, non-precision pair
