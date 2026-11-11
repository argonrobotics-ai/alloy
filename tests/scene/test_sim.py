# SPDX-License-Identifier: Apache-2.0
"""Simulation (§2.3, D18) on CPU: single entry (MuJoCo native and Newton contacts), stock proxy, reset."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.core.registry import isolated_registry
from alloy.precision.types import PrecisionConfig
from alloy.route.router import Router
from alloy.scene.assembler import SceneAssembler
from alloy.sim import Simulation
from alloy.testing import scene as S
from alloy.testing.fakes import FakeHealth, FakeTopology

PASS = {"sag_err": -6e-4, "drop_err": -0.03, "period_err": -0.003, "finite": 1.0, "passed": 1.0}


@pytest.fixture(autouse=True)
def _registry():
    with isolated_registry(empty=True):
        yield


def build(card, lib):
    return SceneAssembler(lib, Router(rod_self_test=lambda rod, h: dict(PASS))).assemble(card, "cpu")


@pytest.mark.parametrize("native", ["mujoco", "newton"])
def test_box_settles_on_table(native):
    card, lib = S.box_on_table(worlds=2)
    built = build(card, lib)
    sim = Simulation(built, precision=PrecisionConfig(native_contacts=native))
    for _ in range(10):
        sim.step_period()
    z = sim.state.body_q.numpy()[:, 2]
    np.testing.assert_allclose(z, 0.02 + 0.03, atol=2e-3)       # table top 0.02 m, box half 0.03 m
    rep = sim.report()
    assert rep["periods"] == 10 and rep["ticks"] == 10 * 1024 and rep["t"] == pytest.approx(0.2)
    assert (sim.contacts is None) == (native == "mujoco")


def test_fake_parts_are_called_and_reset_restores():
    card, lib = S.box_on_table(worlds=2)
    built = build(card, lib)
    topo = FakeTopology(2)
    sim = Simulation(built, topology=topo)
    q0 = sim.state.body_q.numpy().copy()
    sim.step_period()
    assert topo.pre_steps == topo.applies == 16
    sim.reset(wp.array([1, 0], dtype=wp.int32))
    q = sim.state.body_q.numpy()
    np.testing.assert_array_equal(q[0], q0[0])
    assert not np.array_equal(q[1], q0[1])


THROWN = ("the stock-proxy stepper throws the box when the cloth lands on it (WP4's open item, 2026-11-10). CPU, 2 "
          "worlds: the box rests at 0.0500 m for 4 periods and is at 1.39 m after the fifth on main ad97479fc (20 m "
          "after 12), 37.6 m with the cable's stick anchor off on main ec7a47cb2, and gone (NaN) with it on. The same "
          "box and cloth built by the public API step on Alloy's Scheduler and stay put (box 0.0500 m, 0.000 m/s, 10 "
          "periods past the landing, both CPU routes; with the cable too, measured through the landing). Until "
          "2026-11-10 this test asserted only finite values and a cloth above the table, which a cloth carried up "
          "with the box passes.")


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=THROWN)
def test_two_entry_stock_proxy_box_cloth_cable():
    _, card, lib = next(s for s in S.mixed_scenes() if s[0] == "box_cloth_cable")
    built = build(card, lib)
    assert built.coupler.kind == "proxy" and [e.name for e in built.entries] == ["mujoco", "vbd"]
    sim = Simulation(built)
    assert type(sim.stepper).__name__ == "StockProxyStepper" and sim.native == "newton"
    z0 = sim.state.body_q.numpy()[:, 2].copy()
    for _ in range(5):
        sim.step_period()
    pq, z = sim.state.particle_q.numpy(), sim.state.body_q.numpy()[:, 2]
    assert np.isfinite(pq).all() and np.isfinite(z).all()
    assert pq[:, 2].min() > 0.02 - 5e-3                        # cloth stays above the table top
    # and nothing is thrown: the test passed for days with the box 1.4 m up after the cloth landed on it (finite, and
    # the cloth carried up with it). The box starts 1 mm above its rest on the table; the cloth is dropped from 0.12 m.
    boxes = z0 > 0.04
    assert boxes.any() and np.abs(z[boxes] - 0.05).max() < 1e-3, z[boxes]
    assert pq[:, 2].max() < 0.2, pq[:, 2].max()


class _RollbackOnce:
    """Wraps the default stepper: the first period asks for a rollback of world 1 (WatchState 2); the re-run adds a
    small vertical velocity, so a world that was really re-run differs from the fault-free run."""

    def __init__(self, inner, escalated: int, worlds: int):
        self.inner, self.periods, self.perturb = inner, 0, False
        self.watch = np.zeros(worlds, np.int32)
        self.watch[escalated] = 2

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def step(self, s0, s1, control, contacts, dt):
        self.inner.step(s0, s1, control, contacts, dt)
        if self.perturb:
            qd = s1.body_qd.numpy()
            qd[:, 2] += 1e-3
            s1.body_qd.assign(qd)

    def report(self):
        import types

        self.periods += 1
        return types.SimpleNamespace(watch_state=self.watch if self.periods == 1 else np.zeros_like(self.watch))

    def rollback_requested(self, rep):
        return bool(np.any(rep.watch_state == 2))

    def on_rollback(self, mask):
        self.perturb = True


def test_rollback_restores_only_escalated_worlds():
    from alloy.scene.steppers import default_stepper

    card, lib = S.box_on_table(worlds=4)
    ref = Simulation(build(card, lib))
    ref.step_period()
    built = build(card, lib)
    sim = Simulation(built, stepper=_RollbackOnce(default_stepper(built, "mujoco"), escalated=1, worlds=4))
    sim.step_period()
    q, q_ref = sim.state.body_qd.numpy(), ref.state.body_qd.numpy()
    np.testing.assert_array_equal(q[[0, 2, 3]], q_ref[[0, 2, 3]])     # kept worlds: their first end state
    assert not np.array_equal(q[1], q_ref[1])                        # the escalated world was re-run
    assert sim.steps == ref.steps == sim.clock.steps_per_period
    assert sim.report()["ticks"] == ref.report()["ticks"]
