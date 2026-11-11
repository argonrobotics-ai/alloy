# SPDX-License-Identifier: Apache-2.0
"""SB-7 counting on synthetic trip data (alloy.bully.rate)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.core.errors import BullyError
from alloy.bully.rate import SB7_TARGET, blowup_rate, blowup_rate_from_severity


def test_zero_trips_rule_of_three():
    r = blowup_rate(0, 300_000)
    assert r.method == "rule_of_three"
    assert r.upper95 == pytest.approx(1e-5)
    assert r.passes  # 3 / 300k = 1e-5 meets the target exactly
    assert not blowup_rate(0, 299_999).passes
    # 3/N is slightly conservative vs the exact bound 1 - 0.05^(1/N) ~= 2.996/N
    assert r.upper95 >= 1.0 - 0.05 ** (1.0 / 300_000)


def test_clopper_pearson_known_value():
    # one-sided 95 % upper limit for 1 trip in 10: Beta.ppf(0.95, 2, 9) = 0.3942
    r = blowup_rate(1, 10)
    assert r.method == "clopper_pearson"
    assert r.upper95 == pytest.approx(0.3942, abs=1e-4)
    assert r.rate == pytest.approx(0.1)
    assert blowup_rate(5, 5).upper95 == 1.0


def test_from_synthetic_severity_array():
    rng = np.random.default_rng(3)
    sev = np.zeros(400_000, dtype=np.int32)
    idx = rng.choice(sev.size, 52, replace=False)
    sev[idx[:2]] = 1  # two S1 trips
    sev[idx[2:]] = 2  # S2 episodes do not count for SB-7
    r = blowup_rate_from_severity(sev)
    assert r.trips == 2 and r.episodes == 400_000
    assert not r.passes  # 1.57e-5 > SB7_TARGET
    assert r.upper95 > SB7_TARGET
    assert r.upper95 == pytest.approx(1.574e-5, rel=1e-2)  # Beta.ppf(0.95, 3, 399998)


@pytest.mark.parametrize("trips,n", [(-1, 10), (11, 10), (0, 0)])
def test_rejects_impossible_counts(trips, n):
    with pytest.raises(BullyError):
        blowup_rate(trips, n)
