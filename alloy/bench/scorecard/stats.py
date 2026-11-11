# SPDX-License-Identifier: Apache-2.0
"""Bounds for scorecard rates."""

from __future__ import annotations

import math

Z95 = 1.959963984540054


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes of n independent trials (95 % by default).

    Trials must be independent: worlds that are copies of one start are one trial, not many."""
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"wilson needs 0 <= k <= n and n > 0, got k={k}, n={n}")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


__all__ = ["Z95", "wilson"]
