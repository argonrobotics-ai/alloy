# SPDX-License-Identifier: Apache-2.0
"""Watchdog acceptance (SPEC §7 WP4): the light box on stiff cloth contact is caught before 20 m/s and recovered by
halving dt_x through a Simulation-style rollback; snapshots of the MuJoCo + VBD state adapters restore bitwise."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.snapshot import WorldSnapshot
from alloy.coupling.types import WatchState
from alloy.testing.coupling import (ClothSceneSpec, PeriodRunner, cloth_scheduler, coupling_box_on_cloth,
                                    fixed_clock_policy)

CONTROL_DT = 1.0 / 60.0


def _max_speed(runner):
    """Largest body speed over every exchange, the rolled-back attempts included (the watchdog must catch it)."""
    return max(float(np.max(np.linalg.norm(qd[:, :3], axis=1)))
               for qd in runner.log.body_qd + runner.discarded.body_qd)


def _light_box(**caps):
    """0.05 kg box on the cloth at a fixed 4.17 ms clock, 30 periods; ``caps``: the SCR-SC4-4d exchange caps."""
    spec = ClothSceneSpec(box_mass=0.05)
    model = coupling_box_on_cloth(1, spec, device="cpu")
    sched = cloth_scheduler(model, spec, policy=fixed_clock_policy(1.0 / 240.0, **caps), vbd_adaptive=False)
    runner = PeriodRunner(model, sched, CONTROL_DT, log_exchanges=True)
    reports = runner.run(30)
    vmax = _max_speed(runner)
    print(f"0.05 kg box at 4.17 ms {caps}: rollbacks {runner.rollbacks}, final dt_x {reports[-1].dt_x:.6g} s, "
          f"max speed {vmax:.3f} m/s, interface work {reports[-1].interface_work}")
    return runner, reports, vmax


def test_light_box_caught_and_recovered():
    """0.05 kg box, k = 1e5 N/m: ω·dt = 5.9 at dt_x = 4.17 ms ejects it without the watchdog (coupling-time §3.2
    measured 5.86 m/s here); the box must be caught below 20 m/s, with a rollback and dt_x halved, and settle.
    Two things catch it: the watchdog's trip (dt_x halved for good) or a force-jump re-run (dt_x refined for that
    period only), so the test asserts that dt_x was halved in some period, not in the last one. The scene is sensitive
    to its start (6 starts raised 0-100 um, CPU, [internal note] notes/wp11-f32/wd_probe.py): on main before it 4 of 6
    pass and 2 eject at 62-79 m/s and end FAILED; with the float32 remainder (SCR-WP11-F32) the box is caught in 6 of
    6, by a re-run in 3 (1 rollback, dt_x down to 0.26-0.52 ms, back at 4.17 ms by period 23, never FAILED)."""
    runner, reports, vmax = _light_box()                    # the default policy: exchange caps off
    assert runner.rollbacks >= 1
    assert len(runner.discarded.dt) > 0 and math.isclose(sum(runner.log.dt), 30 * CONTROL_DT)   # log = accepted run
    assert min(r.dt_x for r in reports) <= 0.5 / 240.0 + 1e-12
    assert vmax < 20.0
    assert reports[-1].watch_state[0] != int(WatchState.FAILED)
    tail = [np.linalg.norm(qd[:, :3], axis=1).max() for qd in runner.log.body_qd[-20:]]
    assert max(tail) < 1.0                                    # settled after the recovery


def test_light_box_stays_stable_with_the_damping_cap():
    """SCR-SC4-4d (opt-in): the same box with the contact damping cap c_d m_eff / dt_x (c_d 0.5) stays stable at
    4.17 ms. Measured on CPU, main d81ddbc7 + the caps: 0 rollbacks, the clock stays at 4.17 ms, max speed 0.976 m/s,
    interface work 0.073 J (caps off: 1 rollback, the clock halves to 2.08 ms, 0.858 m/s, 0.049 J).
    The cap's signature is the first three (no rollback, no halving, no speed-up). The interface work is a sanity
    bound only: the capped box bounces on the cloth, and the work it sums over 30 periods follows the bounces. Starts
    raised by 0-100 um (8 worlds) give 0.041-0.085 J on Newton 1.7 (main 8b9715996), 0.037-0.224 J on Newton 1.8 +
    MuJoCo 3.17 (the bump; its 0-um start, this test's, is the 0.224). There the runs agree until the first bounce
    (period 6: 0.0105 vs 0.0107 J), and two later bounces (periods 21, 27) add 0.21 J, while the box stays under 1 m/s
    (WP4, 2026-11-09; [internal note] notes/wp4-bump/). Bound: 0.3 J, above two such bounces (0.08 and 0.13 J); the
    uncapped box (4.9-5.5 m/s before the watchdog catches it) fails the first three."""
    runner, reports, vmax = _light_box(exchange_damping_c=0.5)
    assert runner.rollbacks == 0
    assert reports[-1].dt_x == pytest.approx(1.0 / 240.0)
    assert vmax < 1.2
    assert float(np.max(np.abs(reports[-1].interface_work))) < 0.3


def _state_vector(state):
    return np.concatenate([state.body_q.numpy().ravel(), state.body_qd.numpy().ravel(),
                           state.particle_q.numpy().ravel(), state.particle_qd.numpy().ravel()])


def test_snapshot_with_state_adapters_restores_bitwise():
    """Save State + Scheduler.state_arrays() (feedback, Aitken, watchdog, MuJoCo mjw_data, VBD histories) mid-impact,
    run on, restore, re-run: bitwise equal (CPU)."""
    spec = ClothSceneSpec(box_mass=0.2, cells=8)
    model = coupling_box_on_cloth(2, spec, device="cpu")
    sched = cloth_scheduler(model, spec, vbd_adaptive=False)
    extra = sched.state_arrays()
    assert any(k.startswith("mjc.mjw.qacc_warmstart") for k in extra)
    assert any(k.startswith("vbd.vbd.body_q_prev") for k in extra)
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()

    def run(n):
        nonlocal s0, s1
        for _ in range(n):
            s0.clear_forces()
            pipe.collide(s0, contacts)
            sched.step(s0, s1, control, contacts, 1.0 / 480.0)
            s0, s1 = s1, s0
        return _state_vector(s0), {k: np.array(v.numpy(), copy=True) for k, v in extra.items()}

    run(12)                                                   # box is in contact with the cloth by now
    snap = WorldSnapshot(model, s0, extra=extra)
    snap.save(s0)
    first, first_extra = run(10)
    snap.restore(s0)
    second, second_extra = run(10)
    assert first.tobytes() == second.tobytes()
    for k in first_extra:
        assert first_extra[k].tobytes() == second_extra[k].tobytes(), k


def test_scheduler_state_has_world_maps_for_per_world_restore():
    """SCR-WP4-3: every per-item extra array has an item->world map, so a per-world restore does not raise."""
    spec = ClothSceneSpec(box_mass=0.2, cells=6)
    model = coupling_box_on_cloth(2, spec, device="cpu")
    sched = cloth_scheduler(model, spec, vbd_adaptive=False)
    extra, maps = sched.state_arrays(), sched.state_world_maps()
    s0 = model.state()
    snap = WorldSnapshot(model, s0, extra=extra, extra_world=maps)
    assert snap.whole_arrays == (), snap.whole_arrays
    mask = wp.array(np.array([0, 1], np.int32), dtype=wp.int32, device=model.device)
    snap.save(s0, mask)
    snap.restore(s0, mask)
