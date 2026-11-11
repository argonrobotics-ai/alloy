# SPDX-License-Identifier: Apache-2.0
"""cook_core (alloy/tasks/cook_core.py): a short CPU run of the oven roast (light test; the full 40-minute task runs on
the GPU smoke), and the per-world oven input closing the energy ledger."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.plugins.harness.scene import build_simulation
from alloy.tasks import cook_core as T

pytestmark = pytest.mark.law


def test_each_world_heats_from_its_own_oven_and_the_ledger_closes():
    plug = T.plugin()
    sim = build_simulation(T.scene(2, 4, 0, "cpu"), [plug], "cpu", 2)
    ovens = plug.inputs()["ambient"].numpy().copy()
    assert ovens[0] != ovens[1] and (ovens >= T.T_OVEN[0]).all() and (ovens <= T.T_OVEN[1]).all()
    plug.set_input("ambient", [T.T_OVEN[0], T.T_OVEN[1]])
    start = plug.read()["stored_J"]
    for _ in range(4):
        sim.step_period()
    r = plug.read()
    gained = -r["heat_out_J"]                                       # the oven air heats the cube: a negative loss
    assert (gained > 0.0).all() and gained[1] > gained[0], gained   # the hotter oven puts in more
    assert np.allclose(r["stored_J"] - start, gained, rtol=1e-9, atol=0.0), (r["stored_J"] - start, gained)
    assert np.isfinite(r["temperature"]).all()
    assert (r["evaporated_in_solver"] == 0).all()


def test_short_roast_warms_the_surface_not_yet_the_core():
    row = T.run(2, "cpu", seed=0, periods=5)
    assert row["neighbor_overflow_max"] == 0, row
    assert row["core_reached"] == 0 and row["dried_max"] < 1.0, row
    assert row["surface_C_max"] > 10.0 and row["core_C_min"] < 7.0, row
