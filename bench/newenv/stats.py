# SPDX-License-Identifier: Apache-2.0
"""Small statistics: Wilson interval and medians.

Example:
    >>> lo, hi = wilson(3, 3)
    >>> round(lo, 3), hi
    (0.439, 1.0)
"""
from __future__ import annotations

import math
import statistics

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials; (0, 1) when n = 0."""
    if n < 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def median(xs: list[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.median(vals) if vals else None


__all__ = ["median", "wilson"]
