# SPDX-License-Identifier: Apache-2.0
"""CPL-3: resting coupled load vs weight after 2 s, sleeping off (box on cloth; cup on MPM sand; rod clamp not_run)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.defaults import GRAVITY
from alloy.testing.coupling import record_metric

TAIL = 0.25           # s, averaging window at the end of the run


@pytest.mark.slow
@pytest.mark.timeout(1800)
def test_cpl3_box_on_cloth_static_load(cloth_run_02):
    log, spec = cloth_run_02["runner"].log, cloth_run_02["spec"]
    t = np.cumsum(log.dt)
    tail = t >= t[-1] - TAIL
    fz = np.array(log.f_raw)[tail, 0, 2]                      # harvested force on the box, z [N]
    weight = spec.box_mass * GRAVITY
    err = float(abs(fz.mean() - weight) / weight)
    record_metric("CPL-3", scene="box_on_cloth", rel_error=err, mean_fz=float(fz.mean()), weight=weight,
                  t_end=float(t[-1]), sleeping=False)
    print(f"CPL-3 box on cloth: mean F_z {fz.mean():.5f} N vs weight {weight:.5f} N -> {err:.3%}")
    assert t[-1] >= 2.0 - 1e-9
    assert err <= 0.01


@pytest.mark.slow
@pytest.mark.timeout(3000)
def test_cpl3_cup_on_mpm_sand_static_load():
    """CPL-3 cup on MPM sand (engine item 2): the default route (grain_mpm + mujoco, a strong pair kept lagged, MPM
    contact-free, its history carried across exchanges) holds a 28 g cup at rest on a sand bed; after 2 s the
    coupler's harvested force on the cup is its weight within 1 %, sleeping off."""
    from alloy.core.registry import isolated_registry
    from alloy.integration import build_simulation
    from alloy.testing.integration import assemble, cup_on_sand

    card, lib = cup_on_sand()
    with isolated_registry(empty=True):
        built = assemble(card, lib, "cpu", 1)
        (pair,) = built.coupler.pairs
        assert (built.coupler.kind, pair.destination, pair.mode) == ("proxy", "mpm", "lagged")
        sim = build_simulation(built, "standard", graph="off")
        cup = int(built.handles["cup"].template.body.start)
        weight = float(built.model.body_mass.numpy()[cup]) * GRAVITY
        periods = int(round(2.0 / built.control_dt))
        tail = int(round(TAIL / built.control_dt))
        fz = []
        for k in range(periods):
            sim.step_period()
            if k >= periods - tail:
                fz.append(float(sim.stepper.endpoint_sets[0].f_used.numpy()[0][2]))
        assert sim.health.poll().invalid_worlds == ()
    err = float(abs(np.mean(fz) - weight) / weight)
    record_metric("CPL-3", scene="cup_on_mpm_sand", rel_error=err, mean_fz=float(np.mean(fz)), weight=weight,
                  t_end=2.0, sleeping=False)
    print(f"CPL-3 cup on MPM sand: mean F_z {np.mean(fz):.5f} N vs weight {weight:.5f} N -> {err:.3%}")
    assert err <= 0.01
