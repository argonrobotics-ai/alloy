# SPDX-License-Identifier: Apache-2.0
"""SCR-SC4-4d feedback slew limit (alloy/coupling/slew.py), CPU: the kernel on hand-made endpoints and the wiring."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from alloy.coupling import residuals as rs
from alloy.coupling import slew as sl
from alloy.coupling.types import SchedulerPolicy

DEV = "cpu"
DT_X = 6.25e-4


def _set(f_begin: np.ndarray, harvest: np.ndarray, meff: list[float], world: list[int]):
    """An endpoint set over global proxies 0..n-1 with the given exchange-start and harvested feedback."""
    n = len(meff)
    cf = wp.array(harvest.astype(np.float32), dtype=wp.spatial_vector, device=DEV)
    es = SimpleNamespace(kind="body", count=n, proxy_global=wp.array(np.arange(n), dtype=int, device=DEV),
                         world=wp.array(world, dtype=int, device=DEV), meff=wp.array(meff, dtype=float, device=DEV),
                         mapping=SimpleNamespace(coupling_forces=cf))
    lim = sl.FeedbackSlewLimit([es], c_s=0.1)
    lim.f_begin[id(es)].assign(f_begin.astype(np.float32))
    return es, lim


def test_a_jump_is_limited_to_c_s_m_eff_v_s_over_dt_x_and_torque_scales_with_it() -> None:
    m = 0.55
    lim_n = 0.1 * m * sl.SLEW_SPEED / DT_X                              # 44 N at C16's numbers
    begin = np.zeros((1, 6))
    harvest = np.array([[0.0, 900.0, 0.0, 0.0, 0.0, 9.0]])               # a one-pass 900 N spike, 9 N m
    es, lim = _set(begin, harvest, [m], [0])
    lim.apply(es, wp.zeros(1, dtype=wp.int32, device=DEV), DT_X)
    out = es.mapping.coupling_forces.numpy()[0]
    np.testing.assert_allclose(out[:3], [0.0, lim_n, 0.0], rtol=1e-5)
    np.testing.assert_allclose(out[5], 9.0 * lim_n / 900.0, rtol=1e-5)
    assert lim.peak_ratio()[0] == pytest.approx(900.0 / lim_n, rel=1e-5)


def test_a_held_force_frozen_worlds_and_kinematic_proxies_are_untouched() -> None:
    begin = np.array([[0.0, 0.0, 80.0, 0, 0, 0], [0.0, 0.0, 0.0, 0, 0, 0], [500.0, 0, 0, 0, 0, 0]])
    harvest = np.array([[0.0, 0.0, 81.0, 0, 0, 0], [0.0, 0.0, 600.0, 0, 0, 0], [0.0, 0, 0, 0, 0, 0]])
    es, lim = _set(begin, harvest, [0.55, 0.55, 0.0], [0, 1, 0])
    converged = wp.array([0, int(rs.FROZEN)], dtype=wp.int32, device=DEV)            # world 1 frozen
    lim.apply(es, converged, DT_X)
    np.testing.assert_allclose(es.mapping.coupling_forces.numpy(), harvest, rtol=1e-6)  # 80 -> 81 N: a held grip
    assert np.all(lim.peak_ratio() == 0.0)


def test_begin_keeps_the_exchange_start_feedback() -> None:
    es, lim = _set(np.zeros((1, 6)), np.array([[1.0, 2.0, 3.0, 0, 0, 0]]), [0.55], [0])
    lim.begin()
    np.testing.assert_allclose(lim.f_begin[id(es)].numpy(), [[1.0, 2.0, 3.0, 0, 0, 0]])


def test_off_by_default_and_c_s_must_be_positive() -> None:
    assert SchedulerPolicy().exchange_slew_c is None
    with pytest.raises(ValueError, match="c_s must be > 0"):
        sl.FeedbackSlewLimit([], 0.0)


def test_the_scheduler_builds_the_limit_from_the_policy() -> None:
    from alloy.coupling import pinch_rigs as pr

    on = pr._cloth_scheduler(pr.gripper_model(), True, policy=SchedulerPolicy(exchange_slew_c=0.1))
    off = pr._cloth_scheduler(pr.gripper_model(), True)
    assert off.feedback_slew is None
    assert on.feedback_slew is not None and on.feedback_slew.c_s == 0.1
    assert sum(es.count for es in on.feedback_slew.sets) == sum(es.count for es in on.endpoint_sets if es.kind == "body")
