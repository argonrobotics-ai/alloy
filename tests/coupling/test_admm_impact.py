# SPDX-License-Identifier: Apache-2.0
"""SchedulerADMM's penalty per exchange (``SchedulerPolicy.admm_rho_dt``; [internal note] notes/wp4-admm-rho): Newton's
contact rows apply rho W^2 (u - Jv) as a force, W^2 the pair's reduced mass, so one iteration removes rho dt_x of a
contact's approach speed. At Newton's rho 1 (0.125 % per iteration at dt_x 1.25 ms, 2.5 % per 20-iteration exchange) a
20 g box landing at 0.1 m/s on a free 16 g silicone pad went 0.62 mm into a pad particle (radius 0.57 mm): INVALID at
18.75 ms (WP10's #10a probe, C3's platen and pad scaled down). At rho dt_x 0.25 it stops on the pad within a period."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy

M, V0, H, BH, R = 0.02, 0.1, 0.01, 0.02, 0.000571
G = 9.81


def _drop(device: str) -> alloy.Sim:
    """The box 3 particle radii over the pad, both mid-air, the box moving down at V0; world 1's pad 0.2 m lower (the
    free-fall reference: no contact)."""
    s = alloy.Scene(worlds=2, dt=0.0025)
    s.add_box("pad", size=(0.04, 0.04, H), pos=np.array([[0, 0, 0.3], [0, 0, 0.1]]), material="tissue_fat_silicone")
    s.add_box("platen", size=(0.06, 0.06, BH), pos=(0, 0, 0.3 + H / 2 + 3 * R + BH / 2), mass=M, material="wood_beech")
    sim = alloy.build(s, device=device)
    eng = sim.engine
    qd, jqd = eng.state.body_qd.numpy(), eng.state.joint_qd.numpy()
    child, start = eng.model.joint_child.numpy(), eng.model.joint_qd_start.numpy()
    for b in np.asarray(sim.index.per_world(sim.index.body("platen"), "body")).ravel():
        qd[b][2] = -V0
        jqd[int(start[int(np.flatnonzero(child == b)[0])]) + 2] = -V0
    eng.state.body_qd.assign(qd)
    eng.state.joint_qd.assign(jqd)
    return sim


@pytest.mark.law
@pytest.mark.timeout(900)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_a_light_box_landing_on_a_free_soft_pad_stops_on_it_under_admm(device: str) -> None:
    """On cuda:0 the exchanges run as captured graphs, each baking its own rho (the graph key holds dt_x)."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA device")
    sim = _drop(device)
    eng = sim.engine
    assert type(eng.stepper).__name__ == "SchedulerADMM"                # a strong pair (20 g on 16 g): ADMM rows
    assert (eng._graphs is not None) == device.startswith("cuda")
    try:
        pw, x = eng.model.particle_world.numpy(), []
        for _ in range(10):                                             # the hit at ~12 ms, then 12 ms on
            sim.step(1)
            top = float(eng.state.particle_q.numpy()[pw == 0][:, 2].max())
            x.append(top + R - (float(sim.state(["platen"], world=0)["platen.pos"][2]) - BH / 2))
        assert sim.check()["ok"], sim.health()
        assert max(x) < 0.5 * R, x                                      # rho 1: through the particle (0.62 mm deep)
        dt_x = float(eng.stepper._last_report.dt_x)
        assert float(eng.stepper._coupling.rho) == pytest.approx(0.25 / dt_x)
        mp = eng.model.particle_mass.numpy().astype(np.float64)
        v = eng.state.particle_qd.numpy()[:, 2].astype(np.float64)
        box = float(sim.state(["platen"], world=0)["platen.vel"][2])
        pad = float((mp[pw == 0] * v[pw == 0]).sum())
        m_pad = float(mp[pw == 0].sum())
        assert box - pad / m_pad > -0.2 * V0, (box, pad / m_pad)       # the approach stopped (no pass-through)
        expect = -M * V0 - (M + m_pad) * G * sim.t                      # only gravity acts on the pair
        assert M * box + pad == pytest.approx(expect, rel=0.02), (M * box + pad, expect)
    finally:
        sim.close()
