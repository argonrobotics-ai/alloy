# SPDX-License-Identifier: Apache-2.0
"""CPL-4 energy: coupling injection ΔE_int/E_scale (dissipative box on cloth) and passive gain (E − E₀ − W_ext)/E_scale
(conservative variant: μ = 0, no damping); D20 datum test: E < 0 never makes the energy checks fire."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.defaults import ENERGY_JUMP_REL
from alloy.coupling.types import WatchState
from alloy.testing.coupling import record_metric

WINDOW = 1.0   # s


def _series(log, key):
    return np.array([lg[key][0] for lg in log.ledger])


def _rates(t, values, scale):
    """(per sim-minute rate from t=0, max increase over any 1 s window), both as fractions of scale."""
    rate = (values[-1] - 0.0) / scale * 60.0 / t[-1]
    worst = 0.0
    for k in range(len(t)):
        j = np.searchsorted(t, t[k] - WINDOW)
        base = values[j - 1] if j > 0 else 0.0
        worst = max(worst, (values[k] - base) / scale)
    return float(rate), float(worst)


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_cpl4_injection_dissipative(cloth_run_02):
    log = cloth_run_02["runner"].log
    t = np.cumsum(log.dt)
    scale = float(log.ledger[0]["e_scale"][0])
    rate, worst = _rates(t, _series(log, "interface_work"), scale)
    record_metric("CPL-4", scene="box_on_cloth (dissipative)", quantity="dE_int/E_scale", per_min=rate,
                  max_1s=worst, e_scale=scale, t_end=float(t[-1]))
    print(f"CPL-4 dissipative: dE_int/E_scale {rate:.3e}/min, max over 1 s {worst:.3e} (E_scale {scale:.3f} J)")
    assert rate <= 0.01 and worst <= 0.05


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_cpl4_passive_gain_conservative(cloth_run_conservative):
    log = cloth_run_conservative["runner"].log
    t = np.cumsum(log.dt)
    scale = float(log.ledger[0]["e_scale"][0])
    gain = _series(log, "gain")
    rate, worst = _rates(t, gain, scale)
    inj_rate, inj_worst = _rates(t, _series(log, "interface_work"), scale)
    record_metric("CPL-4", scene="box_on_cloth (mu=0, undamped)", quantity="(E-E0-W_ext)/E_scale", per_min=rate,
                  max_1s=worst, dE_int_per_min=inj_rate, dE_int_max_1s=inj_worst, e_scale=scale,
                  final_gain=float(gain[-1] / scale), t_end=float(t[-1]))
    print(f"CPL-4 conservative: gain {rate:.3e}/min, max 1 s {worst:.3e}; dE_int {inj_rate:.3e}/min, "
          f"max 1 s {inj_worst:.3e}; final G/E_scale {gain[-1] / scale:.3e}")
    assert rate <= 0.01 and worst <= 0.05
    assert inj_rate <= 0.01 and inj_worst <= 0.05


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_negative_energy_datum(cloth_run_heavy_cloth):
    """D20: the cloth sags below its reset height, so E < 0; ENERGY_JUMP (per-exchange gain net of W_ext >
    1e-2·E_scale) never fires, the watchdog never escalates, while the pre-D20 rule 1e-2 J + 0.1·E <= 0 would."""
    log = cloth_run_heavy_cloth["runner"].log
    e = _series(log, "energy")
    w_ext = _series(log, "external_work")
    scale = float(log.ledger[0]["e_scale"][0])
    assert e.min() < 0.0, "scene should go below its datum"
    jump = np.diff(e - w_ext) / scale
    old_rule_threshold = 1e-2 + 0.1 * e                        # the v1 threshold that went <= 0 for E < 0
    record_metric("D20-datum", e_min=float(e.min()), e_scale=scale, max_jump_rel=float(jump.max()),
                  old_rule_fires=int(np.sum(old_rule_threshold <= 0.0)))
    assert jump.max() <= ENERGY_JUMP_REL
    assert np.any(old_rule_threshold <= 0.0)
    # no ladder escalation; the first contact's force-jump re-run (SCR-WP11-4) leaves its period ROLLED_BACK, which
    # is a finer re-run, not an escalation of the energy rule
    assert cloth_run_heavy_cloth["runner"].rollbacks == 0
    states = np.array([r.watch_state[0] for r in cloth_run_heavy_cloth["reports"]])
    assert np.all((states < int(WatchState.ESCALATED)) | (states == int(WatchState.ROLLED_BACK))), states
