# SPDX-License-Identifier: Apache-2.0
"""clear-coat books its overspray and dust from where the can was and whether its valve was open as each period ran,
so a policy that moves the can or holds the valve is booked as it sprayed (audit T1-020).

The books read the script's raster place and valve: a can a policy parked 1.5 m beside the box laid the script's mist
on the print and booked nothing outside. The scripted expert's episode is pinned: its verdict and books are those of
the script-only books, number for number.
"""
from __future__ import annotations

import numpy as np
import pytest

from alloy.tasks import paint_clear_coat as cc
from alloy.tasks.paint_clear_coat_checks import escape_share

#: the scripted expert at 0.30 and 0.40 m (``cc.SMALL``, 2 s settle), from the script-only books
PINNED = [{"dust_laid": 2.5712690114743325e-08, "on_print": 1.7254186101241476e-07, "on_bed": 4.196800754970767e-07,
           "gauge": [34.798, 35.059, 35.725], "even": 1.02664},
          {"dust_laid": 1.7071339844952763e-08, "on_print": 1.7346810686520756e-07, "on_bed": 3.8789937147737794e-07,
           "gauge": [32.161, 33.052, 34.956], "even": 1.086922}]
SENT, MIST = 5.411557985940106e-06, 3.246934920585661e-07


def test_a_turned_can_sprays_where_it_points():
    lay, can = cc.BoxLayout(), cc.AerosolCan()
    rays, low = can.ray_table(), (0.0, 0.0, lay.rim_z - 0.1)                 # over the print, below the rim
    assert escape_share(lay, can.efficiency, rays, low) == 0.0               # upright: the box catches all of it
    up = escape_share(lay, can.efficiency, rays, low, turn=np.array([1.0, 0.0, 0.0, 0.0]))   # upside down
    assert up == pytest.approx(can.efficiency, rel=1e-12)                    # every ray leaves through the opening


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_the_scripted_expert_episode_is_unchanged():
    res = cc.run(cc.SMALL, cc.Passes(), settle=2.0, worlds=2, standoffs=[0.30, 0.40])
    for w, pin in zip(res["worlds"], PINNED):
        b = w["books_m3"]
        assert w["succeeded"] and w["score"] == 1.0 and b["outside"] == 0.0
        assert b["sent"] == pytest.approx(SENT, rel=1e-12) and b["mist_solids"] == pytest.approx(MIST, rel=1e-12)
        for key in ("dust_laid", "on_print", "on_bed"):
            assert b[key] == pytest.approx(pin[key], rel=1e-9), key
        assert w["film"]["gauge_um"] == pin["gauge"] and w["checks"]["even"]["value"] == pin["even"]


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_a_can_held_open_beside_the_box_books_all_of_it_outside(monkeypatch):
    from alloy.plugins.surface_flow.plugin import SurfaceFlow

    place, set_input = cc._place, SurfaceFlow.set_input

    def beside(sim, idx, x, y, z) -> None:  # noqa: ANN001
        place(sim, idx, x + 1.5, y, z)                                      # the policy parks the can 1.5 m away

    def held(self, name, values) -> None:  # noqa: ANN001
        set_input(self, name, np.ones_like(values) if name == cc.EDGE else values)   # and holds the valve open

    monkeypatch.setattr(cc, "_place", beside)
    monkeypatch.setattr(SurfaceFlow, "set_input", held)
    w = cc.run(cc.SMALL, cc.Passes(), settle=2.0)["worlds"][0]
    b = w["books_m3"]
    assert b["sent"] > SENT                                                 # it sprayed through the settle too
    assert b["outside"] == pytest.approx(b["sent"], rel=1e-9)              # beside the box, below its rim: all leaves
                                                                           # (also what is in flight at each shut)
    assert b["dust_laid"] == 0.0 and b["on_print"] == 0.0                  # no mist falls on the print
    assert not w["succeeded"] and w["checks"]["overspray"]["ok"] is False
