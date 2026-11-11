# SPDX-License-Identifier: Apache-2.0
"""WP6 analytic laws A1-A6, A15, A16 (SPEC §5.3), registered with `register_law` on import.

A16 (comparator self-test) is listed first: `python -m alloy.bench run` runs it before anything else and a failure
blocks every other law (bench §4.5).
"""

from __future__ import annotations

from alloy.bench.analytic import coulomb, energy, freefall, momentum, neohookean, pendulum, restitution, selftest
from alloy.bench.law import register_law

LAWS = (
    (selftest.LAW, selftest.SelfTestTwin),
    (freefall.LAW, freefall.FreeFallTwin),
    (pendulum.LAW_A2, pendulum.PendulumTwin),
    (pendulum.LAW_A3, pendulum.PendulumEnergyTwin),
    (energy.LAW, energy.ContactEnergyTwin),
    (momentum.LAW, momentum.MomentumTwin),
    (momentum.LAW_SMALL_LEVER, momentum.SmallLeverCollideTwin),
    (coulomb.LAW_ONSET, coulomb.CoulombOnsetTwin),
    (coulomb.LAW_SLIDE, coulomb.CoulombSlideTwin),
    (coulomb.LAW_ISO, coulomb.CoulombIsotropyTwin),
    (coulomb.LAW_SLIDE_SPREAD, coulomb.CoulombSlideSpreadTwin),
    (coulomb.LAW_SLIDE_RECIPE, coulomb.CoulombSlideRecipeTwin),
    (restitution.LAW, restitution.RestitutionTwin),
    (neohookean.LAW, neohookean.NeoHookeanTwin),
    (neohookean.LAW_EXPLICIT, neohookean.NeoHookeanTwin),
)

for _spec, _factory in LAWS:
    register_law(_spec, _factory)

__all__ = ["LAWS"]
