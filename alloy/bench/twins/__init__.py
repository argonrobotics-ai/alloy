# SPDX-License-Identifier: Apache-2.0
"""WP6 rig twins R3 (incline friction) and R4 (drop and bounce), registered on import (SPEC §7 WP6)."""

from __future__ import annotations

from alloy.bench.law import register_law
from alloy.bench.twins import bounce, incline

LAWS = ((incline.LAW, incline.InclineTwin), (bounce.LAW, bounce.BounceTwin))
for _spec, _factory in LAWS:
    register_law(_spec, _factory)

__all__ = ["LAWS"]
