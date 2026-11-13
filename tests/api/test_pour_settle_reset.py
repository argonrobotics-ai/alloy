# SPDX-License-Identifier: Apache-2.0
"""A pour and ``sim.reset()``, with and without ``sim.settle()`` first. CPU.

``settle`` makes the settled state the start a reset returns to. For a pour that start holds the poured particles and
the emitter's place in its reservoir (cursor) and layer clock. The engine's start snapshot was taken before the
pour's emitter existed, and every reset rewound the emitter to 0 instead: after a settle, the settled particles came
back active while the emitter started over, so its first layer took settled particles out of the cup back to the
nozzle (21 cm up), the pour stopped at the settled count, and the mass ledger counted those particles in twice.
Laws (bound: exact, bit for bit on the CPU):
- after ``settle``, a reset puts the emitter back where the settle left it, and the run after the reset repeats the
  run after the settle bit for bit, with the ledger balanced (active + out == initial + in) at every step;
- without ``settle``, a reset still starts the pour over: 0 particles poured, cursor 0, the ledger balanced, and the
  run repeats the first one bit for bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import alloy.api as alloy
from alloy.api import _liquid


def _sink_and_tap() -> alloy.Scene:
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_open_box("sink", inner=(0.1, 0.1, 0.06), wall=0.006)
    scene.add_box("plate", size=(0.04, 0.04, 0.01), pos=(0.12, 0, 0.005), mass=0.1)      # a free body: the coupler
    scene.add_pour("tap", 10e-6, pos=(0, 0, 0.15), rate=1e-4, speed=0.45)               # 156 particles over 0.1 s
    return scene


def _books(sim: alloy.Sim) -> dict[str, Any]:
    """What the pour holds now: its points, which are poured, the emitters' cursors and clocks, the ledger's balance."""
    sph = _liquid.liquid_solvers(sim.engine)[0]
    st, led = sim.state(["tap", "plate"]), sph.mass_ledger()
    return {"points": st["tap.points"].copy(), "active": st["tap.active"].copy(), "plate": st["plate.pos"].copy(),
            "cursor": sph.emitters.dev.cursor.numpy().copy(), "clock": sph.emitters.dev.t_acc.numpy().copy(),
            "balance": led["active"] + led["out"] - led["initial"] - led["in"]}


def _run(sim: alloy.Sim, periods: int) -> list[dict[str, Any]]:
    out = []
    for _ in range(periods):
        sim.step(1)
        out.append(_books(sim))
    return out


def _same(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.compile_heavy
def test_a_reset_after_settle_goes_on_from_the_settled_pour():
    sim = alloy.build(_sink_and_tap(), device="cpu", seed=0)
    sim.settle(0.05)
    settled = _books(sim)
    assert 0 < int(settled["cursor"].min()) and int(settled["cursor"].max()) < 156     # part-way through the pour
    first = _run(sim, 8)
    sim.reset()
    assert _same(_books(sim), settled)                     # was: cursor 0 with the settled particles active
    again = _run(sim, 8)
    assert [int(b["active"].sum()) for b in again] == [int(b["active"].sum()) for b in first]
    assert all(_same(a, b) for a, b in zip(first, again))  # was: settled particles back at the nozzle, 21 cm up
    assert all(not b["balance"].any() for b in first + again)
    sim.close()


@pytest.mark.compile_heavy
def test_a_reset_without_settle_starts_the_pour_over():
    sim = alloy.build(_sink_and_tap(), device="cpu", seed=0)
    first = _run(sim, 8)
    assert int(first[-1]["cursor"].min()) > 0
    sim.reset()
    start = _books(sim)
    assert not start["active"].any() and not start["cursor"].any() and not start["clock"].any()
    assert not start["balance"].any()
    again = _run(sim, 8)
    assert all(_same(a, b) for a, b in zip(first, again))
    assert all(not b["balance"].any() for b in again)
    sim.close()
