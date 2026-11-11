# SPDX-License-Identifier: Apache-2.0
"""CPL-10: rung changes per sim-second in steady contact (box resting on cloth, default adaptive clock)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.testing.coupling import record_metric


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_cpl10_rung_changes_in_steady_contact(cloth_run_02):
    reports = cloth_run_02["reports"]
    control_dt = 0.02
    last_second = int(round(1.0 / control_dt))
    changes = reports[-1].rung_changes - reports[-1 - last_second].rung_changes
    total = reports[-1].rung_changes
    record_metric("CPL-10", changes_per_s=float(changes), window_s=1.0, total_changes=int(total),
                  substeps=dict(reports[-1].substeps), dt_x=float(reports[-1].dt_x))
    print(f"CPL-10: {changes} rung changes in the last 1 s (total {total}); final {reports[-1].substeps}, "
          f"dt_x {reports[-1].dt_x}")
    assert changes <= 1
