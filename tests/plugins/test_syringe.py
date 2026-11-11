# SPDX-License-Identifier: Apache-2.0
"""Syringes, vials and needles (alloy/plugins/library/syringe.py, SCR-WP8-11 Part A): the harness laws G-1..G-6 and
null with bodies, the plunger's stability bound (check_dt refuses an unstable step and names the armature), the
config (CPU)."""

from __future__ import annotations

import pytest

import alloy.plugins as ap
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.check import _law, fresh
from alloy.plugins.harness.scene import build_simulation
from alloy.plugins.library.syringe import Syringes, inertance
from alloy.plugins.library.syringe_laws import PURGE, purge_scene

pytestmark = pytest.mark.law


LAWS = ["G-1_capped_isothermal", "G-1_capped_adiabatic", "G-2_purge_air_first", "G-3_vial_equal_volume",
        "G-4_nozzle_down_no_air", "G-5_books", "G-6_energy_ledger", "null_closed_at_atmosphere"]


def test_syringe_quick_check_passes_its_contract_and_lists_every_law():
    """The harness's contract, lint and stepping checks (laws run one per test below: all eight in one test ran
    57-80 s, at the 60 s timeout on a loaded host)."""
    report = ap.check(Syringes(), quick=True, laws=False)
    assert report.passed, report.summary()
    assert [law.name for law in Syringes().laws()] == LAWS


@pytest.mark.parametrize("name", LAWS)
def test_syringe_law(name):
    plug = Syringes()
    law = next(x for x in plug.laws() if x.name == name)
    res = _law(law, law.scene, [fresh(plug, law.config)], "cpu")
    assert res.passed, (res.detail, res.value, res.bound, res.fix)


def test_the_needle_column_sets_the_plungers_bound():
    """25 G x 16 mm on a 3 mL barrel: the column's added mass ρ·L·A²/a is 1.01 kg, and with it the needle damper's
    bound 2·I/c is ρ·r²/(4·μ)·... 4.2 ms and the capped all-liquid spring's 1.25 ms (WP13's review)."""
    i25 = inertance(8.59e-3, 0.26e-3, 0.016)
    assert i25 == pytest.approx(1.012, rel=2e-3)
    plug = Syringes(**PURGE)
    build_simulation(purge_scene(plug), [plug], "cpu")
    damper, spring = plug.plunger_limits(plug.ctx.model)["syringe"]
    assert damper == pytest.approx(4.2e-3, rel=2e-2) and spring == pytest.approx(1.25e-3, rel=2e-2)
    plug.check_dt(1.0e-3)
    with pytest.raises(PluginError, match="armature of at least"):
        plug.check_dt(2.0e-3)
