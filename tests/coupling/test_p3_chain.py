# SPDX-License-Identifier: Apache-2.0
"""Engine item 3 P3 (option A, coordinator 2026-11-06): MuJoCo -> VBD -> SPH as one coupled chain.

A block (MuJoCo) on a cloth (VBD) lying on water (SPH) routes to one Scheduler over the three entries: MuJoCo -> VBD
lagged, VBD -> SPH staggered (the cloth's surface is SPH boundary, SPH-VBD P1), MuJoCo -> SPH one-way. Newton steps
each entry once per pass in topological order (WP11's N-entry proxy coupling). Without that Newton the build refuses
with a CouplingError naming the pin. CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

import alloy
import alloy.coupling.compose as compose
from alloy.coupling.scheduler import Scheduler

CHAIN = [("mujoco", "vbd", "lagged", False), ("vbd", "sph", "staggered", False), ("mujoco", "sph", "lagged", True)]


def _scene() -> alloy.Scene:
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_open_box("tank", inner=(0.05, 0.05, 0.05), wall=0.006, pos=(0, 0, 0))
    scene.add_liquid("water", size=(0.05, 0.05, 0.016), pos=(0, 0, 0.008))
    scene.add_cloth("sheet", size=(0.04, 0.04), pos=(0, 0, 0.020), resolution=6, mass=0.003)
    # 0.2 kg (7.4 t/m^3): the MuJoCo -> VBD exchange bound is 0.5 sqrt(m/k) = 158 us on 2e6 N/m, dt_x 156 us; a 10 g
    # block clocks at 19.5 us (control_dt / 2^9) and costs ~40 s of CPU per period
    scene.add_box("block", size=(0.03, 0.03, 0.03), pos=(0, 0, 0.037), mass=0.2)
    return scene


def test_chain_needs_n_entry_proxy_coupling(monkeypatch):
    monkeypatch.setattr(compose, "proxy_dag_supported", lambda: False)
    with pytest.raises(alloy.BuildError, match="N-entry proxy coupling"):
        alloy.build(_scene(), device="cpu")


@pytest.mark.timeout(1200)
@pytest.mark.skipif(not compose.proxy_dag_supported(), reason="Newton without N-entry proxy coupling (pin < 11-07)")
def test_block_cloth_water_is_one_scheduler_over_three_entries():
    sim = alloy.build(_scene(), device="cpu")
    st = sim.engine.stepper
    assert isinstance(st, Scheduler)
    assert [(p.source, p.destination, p.mode, p.one_way) for p in st._alloy_pairs] == CHAIN
    assert st.iteration_loop.cfg.k_min >= 2                 # the lagged pair keeps its two passes in the chain
    sim.step(1)
    s = sim.state(world=0)
    assert np.all(np.isfinite(s["water.points"])) and np.all(np.isfinite(s["sheet.points"]))
    assert s["sheet.points"][:, 2].min() > 0.010             # the cloth rests on the water, not through it
    assert sim.check()["ok"], sim.check()["problems"]
