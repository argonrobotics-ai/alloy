# SPDX-License-Identifier: Apache-2.0
"""Realised μ through SolverMuJoCo (Newton-contacts mode) and Featherstone equals the pair table within 1e-6 (WP2
acceptance, D13). A sphere slides on a static table with a PairCard μ_d = 0.23 that neither solver's own mix
(MuJoCo max 0.5, Featherstone mean 0.45 of the shape μ) would give."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.cards.types import Primitive
from alloy.core.registry import isolated_registry
from alloy.precision.types import PrecisionConfig
from alloy.route.router import Router
from alloy.route.types import Route
from alloy.scene.assembler import SceneAssembler
from alloy.sim import Simulation
from alloy.testing import route as T
from alloy.testing import scene as S

MU_PAIR, R, TOP = 0.23, 0.05, 0.02


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):
        yield


def sliding_sphere(entry: str, k: float = 2e5):
    ball = T.obj("ball", Primitive("sphere", (R,)), T.material("abs", density=1050.0, E=2.3e9, mu=0.4, e=0.3, k=k))
    tab = S.table()
    lib = S.library([tab, ball])
    m = 1050.0 * 4.0 / 3.0 * np.pi * R**3
    sink = m * 9.81 / (0.5 * k)                         # static sink in the series spring of two K materials
    card, _ = S.scene([tab, ball], [S.place("table"), S.place("ball", z=TOP + R - sink)],
                      pairs=[S.pair("abs", "wood", 0.3, MU_PAIR)], worlds=2)
    router = Router(entries={Route.RIGID: entry})
    built = SceneAssembler(lib, router).assemble(card, "cpu")
    return built


def _launch_sliding(sim: Simulation, v: float) -> None:
    qd = sim.state.joint_qd.numpy()
    qd[0::6] = v                                        # free joint linear x velocity, every world
    sim.state.joint_qd.assign(qd)
    sim.state.body_qd.assign(np.tile([v, 0, 0, 0, 0, 0], (sim.built.model.body_count, 1)).astype(np.float32))


def test_mujoco_newton_contacts_realised_mu():
    built = sliding_sphere("mujoco")
    sim = Simulation(built, precision=PrecisionConfig(native_contacts="newton"))
    _launch_sliding(sim, 0.05)
    sim._exchange(sim.clock.dt)
    solver = sim.stepper.solver
    n = int(solver.mjw_data.nacon.numpy()[0])
    mu = solver.mjw_data.contact.friction.numpy()[:n, 0]
    assert n >= 2                                       # one contact per world
    err = float(np.max(np.abs(mu / MU_PAIR - 1.0)))
    S.record("realised_mu_mujoco_newton_contacts", err, "pass" if err <= 1e-6 else "fail", mu=mu.tolist())
    np.testing.assert_allclose(mu, MU_PAIR, rtol=1e-6)
    scale = sim.contacts.rigid_contact_friction.numpy()[: int(sim.contacts.rigid_contact_count.numpy()[0])]
    np.testing.assert_allclose(scale, MU_PAIR / 0.5, rtol=1e-6)   # D13: a scale on MuJoCo's max rule


def test_featherstone_realised_mu():
    built = sliding_sphere("featherstone")
    assert [e.name for e in built.entries] == ["featherstone"] and built.pair_table.mu_mix_rule == 1
    sim = Simulation(built)
    _launch_sliding(sim, 0.05)
    dt = 1.0e-4
    sim._exchange(dt)
    f = getattr(sim.state, "body_f_ext", None)          # Featherstone's accumulated contact wrench (state_aug =
    f = (f if f is not None else sim.stepper.solver.body_f_ext).numpy()   # state_out, solver_featherstone.py:555)
    lin = f[:, :3]
    mu = np.hypot(lin[:, 0], lin[:, 1]) / np.abs(lin[:, 2])
    err = float(np.max(np.abs(mu / MU_PAIR - 1.0)))
    S.record("realised_mu_featherstone", err, "pass" if err <= 1e-6 else "fail", mu=mu.tolist())
    np.testing.assert_allclose(mu, MU_PAIR, rtol=1e-6)
    v = sim.state.body_qd.numpy()[:, 0]
    assert np.all(v < 0.05)                             # friction decelerates the +x slide (wrench stored negated)
    m = float(built.model.body_mass.numpy()[0])
    np.testing.assert_allclose((0.05 - v) * m / dt, np.abs(lin[:, 0]), rtol=2e-3)  # Δv·m/h = |f_t|


def test_mujoco_native_contacts_realised_mu_via_pair_rows():
    """D13 native path: an explicit mujoco:pair row carries μ_pair; MuJoCo's own contact uses it (not max 0.5), and
    the explicit pair replaces (does not duplicate) the automatic contact."""
    built = sliding_sphere("mujoco")
    sim = Simulation(built, precision=PrecisionConfig(native_contacts="mujoco"))
    assert sim.contacts is None
    _launch_sliding(sim, 0.05)
    sim._exchange(sim.clock.dt)
    d = sim.stepper.solver.mjw_data
    n = int(d.nacon.numpy()[0])
    mu = d.contact.friction.numpy()[:n, 0]
    err = float(np.max(np.abs(mu / MU_PAIR - 1.0))) if n else float("nan")
    S.record("realised_mu_mujoco_native_pair_rows", err, "pass" if n and err <= 1e-6 else "fail", n_contacts=n)
    assert n == 2                                       # one sphere-table contact per world, no duplicate
    np.testing.assert_allclose(mu, MU_PAIR, rtol=1e-6)
