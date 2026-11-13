# SPDX-License-Identifier: Apache-2.0
"""A paste or liquid made rigid needs a friction (``alloy.route.router.refuse_frictionless_rigid``), AUDIT S1-011
(Gemini 0.1.2): paste and liquid cards have no Coulomb friction (friction 0: their wall drag is their viscosity), so
``rigid=True`` gave a block of mayonnaise mu 0 against everything (it slid 0.2555 m in 1 s down a 3 degree steel
incline, the frictionless 0.2567 m), MuJoCo warned of NaN (friction under MJ_MINMU) and ``sim.check()`` passed. The
laws: such a body is refused at build with a fix naming ``friction=``; with it, the block holds on the incline."""

from __future__ import annotations

import math
import warnings

import pytest

import alloy

TILT = math.radians(3.0)


def _incline(material: str, **kw) -> alloy.Scene:
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("plate", size=(0.80, 0.30, 0.02), pos=(0.0, 0.0, 0.10), rpy=(0.0, TILT, 0.0),
              material="stainless_304", static=True)
    s.add_box("blob", size=(0.04, 0.04, 0.02), pos=(-0.20, 0.0, 0.145), rpy=(0.0, TILT, 0.0), material=material, **kw)
    return s


@pytest.mark.parametrize("material", ["mayonnaise", "peanut_butter", "water"])
def test_a_rigid_paste_or_liquid_without_friction_is_refused_with_the_fix(material):
    with pytest.raises(alloy.BuildError) as err:
        alloy.build(_incline(material, rigid=True), device="cpu")
    assert "no Coulomb friction" in err.value.message and material in err.value.message
    assert "friction=" in err.value.fix


def test_with_a_friction_it_holds_on_the_incline():
    """mu 0.5 > tan 3 degrees: the block stays (it slid 0.2555 m in 1 s at mu 0), and MuJoCo has no zero friction
    to warn about. ~5 s."""
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        sim = alloy.build(_incline("mayonnaise", rigid=True, friction=0.5), device="cpu")
        sim.step(100)
    dx = float(sim.state(world=0)["blob.pos"][0]) + 0.20
    assert abs(dx) < 1.0e-3 and sim.check()["ok"], (dx, sim.check()["problems"])
    assert not [w for w in seen if "MJ_MINMU" in str(w.message)]
