# SPDX-License-Identifier: Apache-2.0
"""SCR-DC5-3: precision pairs under native="mujoco" beside non-MuJoCo entries (CPU).

``wire`` accepts them only when the collision filters prove that no non-MuJoCo entry can contact a precision-pair
shape, and otherwise refuses, naming the pair. The allowed case is DC5's shape: a MuJoCo plug on a jack (the pair)
and a VBD pad whose particles reach only a cold plate.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from alloy.core.errors import AssemblyError
from alloy.core.registry import isolated_registry
from alloy.integration import build_simulation, profile, wire
from alloy.integration import precision_reach as R
from alloy.testing import integration as I

G = 9.81
PAD_RADIUS = 0.004            # dc5_like cloth particle radius [m]; the cold plate's top is z = 0


def _refusal(built) -> str:
    with pytest.raises(AssemblyError) as err:
        R.check_isolated(built, built.entries)
    return str(err.value)


def test_dc5_like_pairs_are_isolated():
    built = I.dc5_like(2, "cpu")
    assert R.find_reach(built, built.entries) == []
    tray = I.dc5_like(2, "cpu", tray_group=I.PAIR_GROUP + 1)      # a rigid entry in another collision group
    assert R.find_reach(tray, tray.entries) == []
    R.check_isolated(tray, tray.entries)


def test_refused_when_particles_can_touch_a_pair_shape():
    msg = _refusal(I.dc5_like(2, "cpu", jack_particles=True))
    assert "precision pair 0 (jack / plug): entry 'pad' (vbd) can contact jack through its particles" in msg
    assert "in 2 world(s)" in msg and "native_contacts='newton'" in msg


def test_refused_when_a_rigid_entry_shares_the_pair_group():
    msg = _refusal(I.dc5_like(2, "cpu", tray_group=I.PAIR_GROUP))
    assert "entry 'tray' (xpbd) can contact jack through its shape tray" in msg
    assert "entry 'tray' (xpbd) can contact plug through its shape tray" in msg


def test_refused_for_mujoco_substeps_and_two_mujoco_entries():
    built = I.dc5_like(1, "cpu")
    mj, pad = built.entries
    sub = dataclasses.replace(built, entries=[dataclasses.replace(mj, max_substeps=2), pad])
    assert "substeps (rung 2)" in _refusal(sub)
    two = dataclasses.replace(built, entries=[mj, dataclasses.replace(pad, capability="mujoco")])
    assert "exactly one SolverMuJoCo entry" in _refusal(two)


@pytest.mark.parametrize(("a", "b", "hit"), [(0, 1, False), (1, 0, False), (1, 1, True), (1, 2, False),
                                             (1, -1, True), (-1, 1, True), (-1, -1, False), (-1, -2, True)])
def test_groups_follow_newton_test_group_pair(a, b, hit):
    assert bool(R._groups_collide(np.array([a]), b)[0]) is hit


def test_sph_skips_ignored_boundary_shapes():
    t = R._Shapes(np.array([R.COLLIDE_PARTICLES]), np.array([1]), np.array([0]), np.array([-1]), np.zeros(0, np.int64),
                  np.array([True]), None)
    worlds = np.array([0])
    assert not R._particle_reach(t, worlds, 0, "sph")
    assert R._particle_reach(t, worlds, 0, "vbd")
    assert not R._particle_reach(t, np.array([1]), 0, "vbd")      # other world


def test_wire_refuses_before_building_solvers():
    """The refused case through ``wire``: no solver registry is needed, the check runs first."""
    with isolated_registry(empty=True), pytest.raises(AssemblyError, match=r"precision pair 0 \(jack / plug\)"):
        wire(I.dc5_like(2, "cpu", jack_particles=True), "standard")


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_wire_dc5_like_allowed_case():
    """MuJoCo plug + VBD pad under the Scheduler: the plug rests at F/K on the appended precision rows (they carry the
    pair table's friction scale), the Newton fallback buffer has no pair rows and gives the pad its contacts, and the
    GapMonitor sees the pair."""
    with isolated_registry(empty=True):
        from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415

        register_builtin_solvers()
        built = I.dc5_like(2, "cpu")
        sim = build_simulation(built, "standard")
        layer = sim.contact_layer
        assert isinstance(layer, R.CoupledPrecisionContacts) and type(sim.stepper).__name__ == "Scheduler"
        assert not layer.bulk.rigid          # MuJoCo owns every rigid shape: soft-only fallback (SCR-WP4-7)
        assert "fallback contacts: soft-only, rigid contacts from MuJoCo (Newton auto sizes)" in sim.parts.notes
        assert sim.contacts is layer.bulk.contacts() and sim.contacts is not layer.layer.contacts()
        for _ in range(20):
            sim.step_period()
    mass = float(built.model.body_mass.numpy()[0])
    sink = 0.01 - sim.state.body_q.numpy()[:, 2]
    np.testing.assert_allclose(sink, mass * G / I.PAIR_K, rtol=0.02)
    prec = layer.layer.contacts()
    n = int(prec.rigid_contact_count.numpy()[0])
    assert n >= 2
    np.testing.assert_allclose(prec.rigid_contact_friction.numpy()[:n], 1.0)    # table mu 0.5 / geom mix 0.5
    bulk = sim.contacts
    nb = int(bulk.rigid_contact_count.numpy()[0])
    pairs = {tuple(int(v) for v in x) for x in layer.world_pairs.reshape(-1, 2)}
    rows = zip(bulk.rigid_contact_shape0.numpy()[:nb], bulk.rigid_contact_shape1.numpy()[:nb])
    assert not [r for r in rows if (min(map(int, r)), max(map(int, r))) in pairs]
    assert int(bulk.soft_contact_count.numpy()[0]) > 0
    assert sim.state.particle_q.numpy()[:, 2].min() > PAD_RADIUS - 2e-4      # the pad rests on the cold plate
    assert int(sim.parts.gap_monitor.report().pair_samples.sum()) > 0
    assert not sim.report()["health"].invalid_worlds


@pytest.mark.compile_heavy
@pytest.mark.timeout(600)
def test_coupled_fallback_takes_the_profile_budget():
    """The non-MuJoCo entries' buffer beside the precision stream is bounded by ``contact_budget`` like the plain
    fallback (SCR-WP4-6), and its overflow counter is one of the health monitor's capacity sources."""
    from alloy.scene.steppers import ContactBudget  # noqa: PLC0415

    with isolated_registry(empty=True):
        from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415

        register_builtin_solvers()
        parts = wire(I.dc5_like(1, "cpu"), profile("standard", contact_budget=ContactBudget(soft_contacts=64)))
    bulk = parts.contact_layer.bulk
    assert bulk.budget.soft_contacts == 64 and not bulk.rigid
    assert any(s.array is bulk.overflow for s in parts.health._sources)


# --- SCR-JOB2-1: a contact-free destination (SPH) on a pair shape it gets as a MuJoCo proxy -----------------------


def test_job2_like_sph_on_the_pair_is_boundary_only_and_accepted():
    """Liquid in a jar that holds a jar / lid pair: SPH reads no contacts, so its reach is boundary only (proxy
    wrench) and the check accepts it when the GapMonitor fails closed; it returns the reaches for the plan report."""
    built = I.job2_like(2, "cpu")
    reach = R.find_reach(built, built.entries)
    assert len(reach) == 4 and {r.channel for r in reach} == {R.BOUNDARY_ONLY}       # jar and lid, 2 worlds
    accepted = R.check_isolated(built, built.entries, gap_fail_closed=True)
    assert accepted == reach
    lines = R.describe_reach(built, accepted)
    assert lines == ["precision pair 0 (jar / lid): entry 'water' (sph) can contact jar through boundary only "
                     "(proxy wrench) in 2 world(s)",
                     "precision pair 0 (jar / lid): entry 'water' (sph) can contact lid through boundary only "
                     "(proxy wrench) in 2 world(s)"]


def test_job2_like_boundary_only_needs_a_fail_closed_gap_monitor():
    msg = _refusal(I.job2_like(1, "cpu"))
    assert "entry 'water' (sph) can contact jar through boundary only (proxy wrench)" in msg
    assert "GapMonitor on and failing closed" in msg


def test_job2_like_with_vbd_is_still_refused():
    built = I.job2_like(2, "cpu", liquid="vbd")
    with pytest.raises(AssemblyError, match=r"entry 'water' \(vbd\) can contact jar through its particles"):
        R.check_isolated(built, built.entries, gap_fail_closed=True)


def test_boundary_only_needs_the_shape_on_a_proxied_body():
    """A static jar is not a proxy the SPH entry receives: that reach refuses, the lid's stays boundary only."""
    built = I.job2_like(1, "cpu", static_jar=True)
    by_shape = {r.shape: r.channel for r in R.find_reach(built, built.entries)}
    assert by_shape == {0: "its particles (COLLIDE_PARTICLES)", 1: R.BOUNDARY_ONLY}
    with pytest.raises(AssemblyError, match=r"can contact jar through its particles"):
        R.check_isolated(built, built.entries, gap_fail_closed=True)


def test_wire_refuses_boundary_only_without_the_gap_monitor():
    """Through ``wire``: a profile without the GapMonitor refuses before any solver is built."""
    with isolated_registry(empty=True), pytest.raises(AssemblyError, match="GapMonitor on and failing closed"):
        wire(I.job2_like(1, "cpu"), profile("standard", gap_monitor="never"))
