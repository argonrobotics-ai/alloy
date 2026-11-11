# SPDX-License-Identifier: Apache-2.0
"""PRC-9 (SPEC §5.2): injected NaN, overflow, 1 mm precision-pair penetration (> S1 for 0.2 mm clearance) and an
energy jump of 2·E_scale are INVALID and latched in the step they appear; every other world stays bitwise equal to
a clean run. Scenes: box_on_plane(16) (XPBD, Newton contacts) and pendulum(16) (Featherstone, articulated)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy.contact.types import PrecisionPairSpec
from alloy.precision.types import CheckPolicy, HealthFlags, Severity
from alloy.testing.precision import HealthLoop, set_body, world_bodies
from alloy.testing.scenes import box_on_plane, pendulum

W, WARMUP, AFTER, INJECT_AT = 16, 20, 15, 20
PAIR = PrecisionPairSpec(0, 1, clearance=2e-4)  # 0.2 mm diametral: S1 = 0.1 mm, so 1 mm is S1


def _scene(name: str, device: str):
    return box_on_plane(W, device=device) if name == "box" else pendulum(W, 0.4, device=device)


def _loop(name: str, device: str, overflow: wp.array) -> HealthLoop:
    # XPBD with Warp's deterministic lowering, so the bitwise comparison with the clean run also holds on a GPU
    kwargs = {"deterministic": wp.DeterministicMode.RUN_TO_RUN} if name == "box" else {}
    return HealthLoop(_scene(name, device), checks=CheckPolicy(level="full", every=1, full_every=1),
                      capacity_flags=[(overflow, int(HealthFlags.CONTACT_OVERFLOW))], precision_pairs=[PAIR],
                      solver_kwargs=kwargs)


def _kick_energy(loop: HealthLoop, world: int, state) -> None:
    """Add 2·E_scale of kinetic energy upward to the first body of ``world``."""
    b = int(world_bodies(loop.scene.model, world)[0])
    e_scale = float(loop.monitor._ref.e_scale.numpy()[world])
    mass = float(loop.scene.model.body_mass.numpy()[b])
    qd = state.body_qd.numpy()
    v0 = qd[b, :3].astype(np.float64)
    ke0 = 0.5 * mass * float(v0 @ v0)
    v = math.sqrt(2.0 * (ke0 + 2.0 * e_scale) / mass)
    qd[b, :3] = (0.0, 0.0, v)
    state.body_qd.assign(qd)


INJECTIONS = {
    3: "nan", 5: "overflow", 7: "pair_pen", 9: "energy",
}


def _injector(loop: HealthLoop, overflow: wp.array):
    def inject(step: int, state) -> None:
        if step != INJECT_AT:
            return
        b3 = int(world_bodies(loop.scene.model, 3)[0])
        q = state.body_q.numpy()[b3].copy()
        q[2] = np.nan
        set_body(state.body_q, b3, q)
        overflow.assign(np.where(np.arange(W) == 5, 1, 0).astype(np.int32))
        loop.gap.set(7, 1e-3)
        _kick_energy(loop, 9, state)
    return inject


@pytest.mark.parametrize("name", ["box", "pendulum"])
def test_prc9_recall_and_isolation(device: str, name: str) -> None:
    overflow_clean = wp.zeros(W, dtype=wp.int32, device=device)
    clean = _loop(name, device, overflow_clean)
    clean.step(WARMUP + AFTER)
    ref_q, ref_qd = clean.body_state()
    rep_clean = clean.monitor.poll()
    assert rep_clean.invalid_worlds == ()

    overflow = wp.zeros(W, dtype=wp.int32, device=device)
    loop = _loop(name, device, overflow)
    loop.step(WARMUP)
    pre_q = loop.body_state()[0]
    loop.step(1, inject=_injector(loop, overflow))
    rep = loop.monitor.poll()
    flags_at = {3: HealthFlags.NONFINITE_Q, 5: HealthFlags.CONTACT_OVERFLOW, 7: HealthFlags.PAIR_PEN_FAIL,
                9: HealthFlags.ENERGY_JUMP}
    assert set(rep.invalid_worlds) == set(flags_at), rep.invalid_worlds
    for w, flag in flags_at.items():
        assert rep.severity[w] == Severity.INVALID
        assert rep.flags[w] & int(flag) and rep.flags[w] & int(HealthFlags.LATCHED)
        assert rep.first_bad_step[w] == INJECT_AT
    overflow.zero_()
    loop.gap.set(7, 0.0)
    loop.step(AFTER - 1)
    q, qd = loop.body_state()
    world = loop.scene.model.body_world.numpy()
    others = ~np.isin(world, list(flags_at))
    assert np.array_equal(q[others].view(np.uint32), ref_q[others].view(np.uint32))
    assert np.array_equal(qd[others].view(np.uint32), ref_qd[others].view(np.uint32))
    frozen = np.isin(world, list(flags_at))
    assert np.array_equal(q[frozen].view(np.uint32), pre_q[frozen].view(np.uint32))  # restored and frozen
    final = loop.monitor.poll()
    assert set(final.invalid_worlds) == set(flags_at)  # sticky


def test_raise_mode(device: str) -> None:
    from alloy.core.errors import PrecisionError

    loop = HealthLoop(box_on_plane(4, device=device), checks=CheckPolicy(on_invalid="raise"))
    loop.step(2)
    loop.monitor.poll()
    b = int(world_bodies(loop.scene.model, 1)[0])

    def nan(step: int, state) -> None:
        q = state.body_q.numpy()[b].copy()
        q[0] = np.inf
        set_body(state.body_q, b, q)

    loop.step(1, inject=nan)
    with pytest.raises(PrecisionError, match="NONFINITE_Q") as err:
        loop.monitor.poll()
    assert err.value.report.invalid_worlds == (1,)


def test_speed_and_penetration_flags(device: str) -> None:
    loop = HealthLoop(box_on_plane(4, device=device))
    loop.step(3)
    b1, b2 = int(world_bodies(loop.scene.model, 1)[0]), int(world_bodies(loop.scene.model, 2)[0])

    def inject(step: int, state) -> None:
        qd = state.body_qd.numpy()
        qd[b1, :3] = (25.0, 0.0, 0.0)  # > 20 m/s: SPEED_FAIL (INVALID)
        state.body_qd.assign(qd)
        q = state.body_q.numpy()
        q[b2, 2] -= 0.003  # 3 mm into the plane: PEN_WARN (DEGRADED), below the 1 cm S1
        state.body_q.assign(q)

    loop.step(1, inject=inject)
    rep = loop.monitor.poll()
    assert rep.flags[1] & int(HealthFlags.SPEED_FAIL) and rep.severity[1] == Severity.INVALID
    assert rep.flags[2] & int(HealthFlags.PEN_WARN) and rep.severity[2] == Severity.DEGRADED
    assert rep.max_pen_m[2] == pytest.approx(0.003, rel=0.2)
    assert rep.severity[0] == Severity.OK and rep.severity[3] == Severity.OK
