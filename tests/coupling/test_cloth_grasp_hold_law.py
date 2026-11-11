# SPDX-License-Identifier: Apache-2.0
"""CPU: does a two-pad pinch on a cloth's top edge HOLD a hanging load, or does the cloth creep out
(``alloy.coupling.pinch_hold.run_hold``)? Two MuJoCo pads squeeze a double layer of C7's towel (6 x 6 cells, top edge
at the pads' height, bottom and top free) at grip_n = 15 N; the carriage holds still; a downward load W is ramped on
the cloth's bottom row (an external particle force) over 0.3 s, then held while the clamped cloth's slide against the
pads is measured.

Coulomb limit: mu N = 2 faces x mu x 15 N = 12.7 N, mu = sqrt(0.45 x 0.4) = 0.424 (VBD's geometric mean of the model's
soft_contact_mu and the pad's mu; the clamp test's 0.42). Load ratio r = W / (mu N). Law: below the limit static
friction holds the cloth (no creep: the slide is the elastic settling of the cloth, and the pads' compliance); above it
the cloth slides out.

The engine's friction is Newton VBD's smoothed Coulomb, though: the slip is measured from each step's start and below
eps_u = friction_epsilon x dt the force is mu f_n (2 y - y^2), y = |u| / eps_u. A steady load F < mu f_n therefore keeps
a held vertex slipping y eps_u every step, a creep speed friction_epsilon x (1 - sqrt(1 - r)). The shipped cloth route
leaves friction_epsilon at Newton's 1e-2 (not set in alloy/scene/entries.py), so r = 0.5 predicts 2.93 mm/s.
Measured (2026-11-10, a CPU spot VM, the route's VBD via make_vbd at 40 iterations, full-surface proxy contacts as the
robot fingers use, 1 s hold after a 0.15 s close and a 0.3 s load ramp; creep, formula in brackets): r = 0.5 2.97 mm/s
[2.93], r = 0.75 5.05 mm/s [5.00]; r = 0.9 falls out of the pads; r = 1.2 and 2.0 slide out. friction_epsilon 1e-3:
r = 0.5 0.28 mm/s. So the creep is the smoothed Coulomb law to 2 %, and the pinch lets go below mu N.
An opt-in stick anchor for VBD body-particle friction is the planned fix; ``run_hold(stick=True)`` switches it on.
Slow (a CPU second of this rig costs 7-15 min), so the CPU gate suite skips it (marker ``slow``)."""

from __future__ import annotations

import pytest

from alloy.coupling.pinch_hold import run_hold

pytestmark = [pytest.mark.slow, pytest.mark.law, pytest.mark.timeout(3600)]

MU_N = 2.0 * 0.4243 * 15.0
EPS = 1.0e-2                    # the cloth route's friction_epsilon (Newton's default; the route does not set it)
HOLD_S = 1.0                    # a CPU second costs 7-15 min here; the bound below scales with it

# Static friction means no creep, so the bound covers only the cloth's elastic settling and the pads' compliance:
# 1 mm over 10 s of hold (0.1 mm/s), taken proportionally over the shorter hold.
SLIDE_BOUND = 1.0e-3 * HOLD_S / 10.0
CREEP_BOUND = 1.0e-4


@pytest.mark.xfail(strict=True, reason="VBD's smoothed Coulomb friction (slip measured from each step's start, force "
                   "mu f_n (2 y - y^2) below eps_u = friction_epsilon dt) lets a held vertex creep at "
                   "friction_epsilon (1 - sqrt(1 - r)): 2.97 mm/s measured at r = 0.5, eps 1e-2; "
                   "an opt-in stick anchor for "
                   "VBD body-particle friction is the planned fix")
def test_pinch_holds_half_the_coulomb_load():
    r = run_hold(0.5, hold_s=HOLD_S, friction_epsilon=EPS)
    print("HOLD r=0.5", {k: v for k, v in r.__dict__.items() if k != "trace"})
    assert r.finite and r.pass_through == 0, r
    assert abs(r.load_n - 0.5 * MU_N) < 0.1, r
    assert not r.left_pads, r                                 # it stays between the pads
    assert r.slide < SLIDE_BOUND and abs(r.creep) < CREEP_BOUND, (r.slide, r.creep)


def test_pinch_lets_go_above_the_coulomb_load():
    r = run_hold(2.0, hold_s=HOLD_S, friction_epsilon=EPS)
    print("HOLD r=2.0", {k: v for k, v in r.__dict__.items() if k != "trace"})
    assert r.finite and r.pass_through == 0, r
    assert r.slide > 10.0e-3 or r.left_pads, (r.slide, r.left_pads)
