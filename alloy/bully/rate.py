# SPDX-License-Identifier: Apache-2.0
"""SB-7 blow-up rate: S1 episodes over N random-legal bully rollouts, with a one-sided 95 % upper bound.

At zero trips the bound is the rule of three, 3/N (Hanley & Lippman-Hand 1983, "If nothing goes wrong, is everything
all right?", JAMA 249:1743); the exact Clopper-Pearson value there is -ln(0.05)/N = 2.996/N, so 3/N is the slightly
conservative round number SPEC §5.1 names. With k > 0 trips the bound is the exact one-sided Clopper-Pearson upper
limit Beta(1 - alpha; k + 1, N - k).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

from alloy.core.errors import BullyError

SB7_TARGET = 1e-5  # SPEC §5.1: upper bound <= 1e-5 per scene class


@dataclass(frozen=True)
class BlowupRate:
    """SB-7 value for one scene class."""

    trips: int
    episodes: int
    rate: float  # trips / episodes
    upper95: float  # one-sided 95 % upper bound
    method: str  # "rule_of_three" or "clopper_pearson"

    @property
    def passes(self) -> bool:
        """True when the upper bound meets the SB-7 target (needs >= 300k clean episodes at 0 trips)."""
        return self.upper95 <= SB7_TARGET


def blowup_rate(trips: int, episodes: int, confidence: float = 0.95) -> BlowupRate:
    """Rate and one-sided upper bound from a trip count; raises ValueError on impossible counts."""
    if episodes <= 0 or trips < 0 or trips > episodes:
        raise BullyError(f"need 0 <= trips <= episodes and episodes > 0, got {trips}/{episodes}")
    if not 0.0 < confidence < 1.0:
        raise BullyError(f"confidence must be in (0, 1), got {confidence}")
    if trips == 0 and confidence == 0.95:
        return BlowupRate(0, episodes, 0.0, 3.0 / episodes, "rule_of_three")
    upper = 1.0 if trips == episodes else float(beta.ppf(confidence, trips + 1, episodes - trips))
    return BlowupRate(trips, episodes, trips / episodes, upper, "clopper_pearson")


def blowup_rate_from_severity(severity: np.ndarray) -> BlowupRate:
    """SB-7 from per-episode severities (0 none, 1 = S1, ...): an episode trips when its severity is S1."""
    sev = np.asarray(severity).ravel()
    return blowup_rate(int(np.count_nonzero(sev == 1)), int(sev.size))
