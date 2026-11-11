# SPDX-License-Identifier: Apache-2.0
"""MPM in a coupled scene (engine item 2, CPL-3; WP4 reviewed).

MPM is a contact-free destination like SPH: SolverImplicitMPM.step ignores ``contacts`` ("collisions are handled
internally"), so a rigid -> MPM proxy pair gets no contact-spring clock bound (SCR-JOB2-3), its proxy no contact
pipeline (SCR-DC5-19), and the MPM particles no fallback soft candidates (SCR-JOB2-4). Before: the spring bound
clocked CPL-3's cup on sand at dt_x 3.9e-5 s, and the soft-contact PEN rule latched the cup's 1.4 mm overlap with the
1.5 mm particles. The MPM entry's namespaced history also survives each exchange (``alloy.coupling.entry_history``).
CPU, cup-on-sand scene, the pair run lagged (the mode the router gives a strong MPM pair, WP4's coupler_select line)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from alloy.core.registry import isolated_registry
from alloy.coupling.proxy_contacts import CONTACT_FREE_DESTINATIONS, contact_free
from alloy.integration.stepper import dynamic_entries
from alloy.integration.wiring import contact_free_particles
from alloy.testing.integration import assemble, cup_on_sand

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


def _lagged(built):
    """The cup/sand pair as one lagged proxy pair (what D5 routes a strong MPM pair to)."""
    (pair,) = built.coupler.pairs
    pair = dataclasses.replace(pair, mode="lagged")
    return dataclasses.replace(built, coupler=dataclasses.replace(built.coupler, kind="proxy", pairs=(pair,)))


@pytest.fixture(scope="module")
def cup_sim():
    from alloy.integration import build_simulation

    card, lib = cup_on_sand()
    with isolated_registry(empty=True):
        built = _lagged(assemble(card, lib, "cpu", 1))
        sim = build_simulation(built, "standard", graph="off")
        sim.step_period()
        yield built, sim


def test_mpm_is_contact_free():
    assert "mpm" in CONTACT_FREE_DESTINATIONS and "sph" in CONTACT_FREE_DESTINATIONS
    assert contact_free(None, "mpm")


def test_mpm_pair_gets_no_contact_stiffness_bound(cup_sim):
    built, sim = cup_sim
    st = sim.stepper
    assert st._default_stiffness() > 0.0                        # the cup's contact ke exists, but does not apply
    assert st._pair_stiffness_mass() == []
    assert math.isinf(st._interface_bound())
    assert sim.clock.dt >= 1.0e-3                               # was 3.9e-5 s under the spring bound
    (pair,) = st._alloy_pairs
    assert pair.proxy is not None and pair.proxy.collision_pipeline is None


def test_mpm_particles_get_no_fallback_soft_candidates(cup_sim):
    built, sim = cup_sim
    (mpm,) = [e for e in built.entries if e.capability == "mpm"]
    free = contact_free_particles(dynamic_entries(built))
    np.testing.assert_array_equal(free, np.sort(np.asarray(mpm.particles)))
    layer = sim.contact_layer
    assert hasattr(layer, "pipeline"), type(layer).__name__     # the fallback buffer the candidates come from
    pairs = layer.pipeline.soft_rigid_contact_pairs.numpy()
    assert not np.isin(pairs[:, 0], free).any()
    assert sim.health.poll().invalid_worlds == ()               # the cup's overlap no longer latches PEN_FAIL


def test_mpm_history_survives_exchanges():
    """The MPM entry steps state_0 -> state_1 and the coupler refreshes only core arrays in state_0; without carrying
    its namespaced history (elastic strain, Jp, stress) back, the bed gave up and re-took its elastic settlement every
    exchange and the cup sank 0.6 um per exchange (0.5 mm/s). With ``alloy.coupling.entry_history`` the cup rests."""
    from alloy.integration import build_simulation

    card, lib = cup_on_sand()
    with isolated_registry(empty=True):
        built = _lagged(assemble(card, lib, "cpu", 1))
        sim = build_simulation(built, "standard", graph="off", health=False)
        cup = int(built.handles["cup"].template.body.start)
        for _ in range(4):                                       # landing (the cup starts 1 mm above the sand)
            sim.step_period()
        z0 = float(sim.state.body_q.numpy()[cup, 2])
        for _ in range(4):
            sim.step_period()
        z1, vz = float(sim.state.body_q.numpy()[cup, 2]), float(sim.state.body_qd.numpy()[cup, 2])
    assert abs(z1 - z0) < 1.0e-6                                 # was 40 um over these 0.08 s
    assert abs(vz) < 1.0e-5                                      # was 5e-4 m/s
