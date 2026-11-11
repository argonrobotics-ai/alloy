# SPDX-License-Identifier: Apache-2.0
"""task_forklift_pick (vehicle + fluid network): approach, fork, lift, carry and set down a 725 kg pallet."""

from __future__ import annotations

import pytest

from alloy.tasks.forklift_pick import Pick, run

pytestmark = pytest.mark.compile_heavy


def test_forklift_picks_carries_and_sets_down_the_pallet():
    res = run(Pick(carry=1.0), timing=False)
    assert res.phases_done == 4
    assert abs(res.stop_error_mm) < 30.0
    assert res.lift_mm > 150.0
    assert res.carried_m > 0.95
    assert abs(res.slip_on_forks_mm) < 10.0
    assert abs(res.set_down_mm) < 5.0
    assert 20.0 < res.peak_bar < 200.0                          # 950 kg on a 60 mm bore is about 33 bar
    assert res.max_drive_slip < 0.5
    assert res.min_rear_load_n > 1000.0                         # the counterweight keeps the steer wheels loaded


@pytest.mark.gpu
def test_gpu_smoke_and_cost_per_step_at_64_worlds():
    """GPU smoke (light-test rule: <= 64 worlds, short): the whole pick on 64 worlds, with its cost per step."""
    res = run(Pick(carry=1.0), worlds=64, device="cuda:0")
    print(f"GPU 64 worlds: {res.ms_per_step:.3f} ms/step with vehicle, {res.ms_per_step_without:.3f} without")
    assert res.phases_done == 4 and res.lift_mm > 150.0
    assert res.ms_per_step < 2.0 * res.ms_per_step_without
