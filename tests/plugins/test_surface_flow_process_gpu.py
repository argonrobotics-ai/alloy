# SPDX-License-Identifier: Apache-2.0
"""Surface-flow process states on the GPU (design §4.8; WP9; light-test rule: one smoke): the full harness check with
every process state on, CUDA graphs included."""

from __future__ import annotations

import pytest

import alloy.plugins as ap
from alloy.plugins.surface_flow import SurfaceFlow
from alloy.plugins.surface_flow.process import Drying, Sanitizer, Setting, Soil, Substrate


def every_part() -> SurfaceFlow:
    """The harness's tilted plate with setting, soil, drying, a substrate and the sanitizer clock on."""
    return SurfaceFlow(per_cell_rheology=True, species=("solids", "soil", "sanitizer"),
                       process={"setting": Setting(tau_0=0.1, a_thix=0.01, tau_is=0.0),
                                "soil": Soil(initial=0.05, sigma_dry=10.0, sigma_wet=0.01, t_soak=0.05, k_e=0.01),
                                "drying": Drying(k_m=0.6), "substrates": {0: Substrate(capacity=1.0e-5)},
                                "sanitizer": Sanitizer(c_min=1.0e-9)})


@pytest.mark.gpu
def test_harness_full_check_with_every_process_state_on_gpu():
    report = ap.check(every_part(), device="cuda:0", laws=False)
    print(report.summary())
    assert report.passed, report.summary()
