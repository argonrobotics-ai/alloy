# SPDX-License-Identifier: Apache-2.0
"""SB-9 (CARD-1 coverage, §5.1) feeds: the coverage row measured on the pinned Newton decides.

Every Newton re-pin re-measures CARD-1 before SB-9 is re-fed (ruling scr/WP11.md 2026-11-04 23:10Z; wp0-scr-log
"Newton re-pin 47afecd7": WP11 re-measures, WP6 re-feeds SB-9). So a coverage row decides only when its label or value
cell names the pin `alloy.core.newton_compat.NEWTON_PIN` holds; a coverage row on another Newton is shown beside as
superseded and never decides. After the next re-pin SB-9 reads inconclusive until a row on the new pin lands.
"""

from __future__ import annotations

import importlib.util

from alloy.bench import gpu_repeat as GR
from alloy.bench.headline_types import Feed
from alloy.core.newton_compat import NEWTON_PIN

PIN = NEWTON_PIN[:8]
RULING = "ruling scr/WP11.md 2026-11-04 23:10Z"
_PCT = r"(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?) ?%"     # headline._NUM + " ?%": the first percentage in the value cell
_ON_PIN = rf"\b{PIN}"

FEEDS: tuple[Feed, ...] = (
    Feed("CARD-1", _PCT, "%", tag=f"coverage on Newton {PIN}", label="coverage", marker=_ON_PIN),
    Feed("CARD-1", _PCT, "%", tag="coverage on an earlier Newton (superseded)", label="coverage",
         not_marker=_ON_PIN, decides=False),
)


# Coordinator 2026-11-05 03:45Z: no N <= 10,000 iterations brings the CARD-1 sheet coupons within 10 % of Navier on
# the pin, so the profile is parked until the membrane fix 8c6b2e1e returns (wp11-results "Accurate-cloth profile").
_PARKED = ("parked (coordinator 2026-11-05 03:45Z: no N <= 10,000 converges the CARD-1 sheets on the pin; waits for "
           "the membrane fix 8c6b2e1e)")


def hinge_gap() -> str:
    """SB-9's known gap (coordinator 2026-11-04 16:25Z, SCR-WP11-3 (3)); the profile reads available once WP11's
    module is on the tree."""
    try:
        have = importlib.util.find_spec("alloy.solvers.vbd.cloth_profile") is not None
    except (ImportError, ValueError):
        have = False
    return ("known gap beside (SCR-WP11-3): default VBD hinge map is mesh-dependent (~1/120 of the card's D at "
            "8.6 mm); accurate-cloth profile " + ("available (alloy.solvers.vbd.cloth_profile)" if have else _PARKED))


def note() -> str:
    return (f"the CARD-1 coverage row measured on the pinned Newton {PIN} decides ({RULING}: re-measured after every "
            "re-pin); a coverage row on an earlier Newton is shown beside as superseded; the other CARD-1 rows are "
            f"findings; each card's verdict is one L4 run: {GR.SAMPLE}; " + hinge_gap())


__all__ = ["FEEDS", "PIN", "RULING", "hinge_gap", "note"]
