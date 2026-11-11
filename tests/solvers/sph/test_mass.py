# SPDX-License-Identifier: Apache-2.0
"""LIQ-1: exact mass ledger with an emitter, a sink and the escape box (SPEC §5.2)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.testing.sph import default_config, run, water_column


def test_liq1_ledger_exact_with_emitter_sink_escape():
    cfg = default_config(world_bounds=((-0.004, -0.004, -0.004), (0.012, 0.012, 0.060)))
    sc = water_column(4, 4, 4, worlds=2, fills=[4, 3], spares=160, headroom=20, cfg=cfg)
    s = sc.make_solver()
    sp = cfg.spacing
    # emitter above world 0's column pouring down; sink in world 1's floor corner
    e = s.add_emitter(0, wp.transform((2 * sp, 2 * sp, 9 * sp), wp.quat_identity()), wp.vec3(2 * sp, 2 * sp, 0.4 * sp),
                      rate=16 * sp**3 / 0.008, velocity=wp.vec3(0.0, 0.0, -0.25))
    k = s.add_sink(1, wp.transform((0.5 * sp, 0.5 * sp, 0.5 * sp), wp.quat_identity()), wp.vec3(sp, sp, sp))
    assert (e, k) == (0, 0)
    a, b = sc.model.state(), sc.model.state()
    # one particle of world 1 thrown out of the escape box
    v = a.particle_qd.numpy()
    w = sc.model.particle_world.numpy()
    phase = s.layout.phase.numpy()
    thrown = int(np.nonzero((w == 1) & (phase == 1))[0][-1])
    v[thrown] = (0.0, 0.0, 40.0)
    a.particle_qd.assign(v)
    a, b = run(s, a, b, 50, 1.0e-3)
    led = s.mass_ledger()
    assert np.array_equal(led["active"] + led["out"], led["initial"] + led["in"])  # exact, int64 fixed point
    r = s.report()
    assert r.mass_in[0] > 0.0 and r.mass_in[1] == 0.0
    assert r.mass_out[1] > 0.0 and r.escaped[1] >= 1 and r.escaped[0] == 0
    m = sc.model.particle_mass.numpy()[0]
    assert np.allclose(r.mass_in[0] / m, round(r.mass_in[0] / m), atol=1e-6)  # whole particles emitted
    liq1 = (r.mass_active + r.mass_out) / (led["initial"] / led["scale"] + r.mass_in) - 1.0
    assert np.all(np.abs(liq1) <= 1e-6)


def test_a_closed_emitter_emits_nothing_until_opened():
    """SolverDFSPH.set_emitter_enabled (JOB3's burette stopcock): closed for 30 steps, no mass comes in; opened, it
    emits whole particles and the ledger stays exact; an unknown emitter is a ValueError."""
    cfg = default_config(world_bounds=((-0.004, -0.004, -0.004), (0.012, 0.012, 0.060)))
    sc = water_column(4, 4, 4, worlds=1, fills=[3], spares=160, headroom=20, cfg=cfg)
    s = sc.make_solver()
    sp = cfg.spacing
    e = s.add_emitter(0, wp.transform((2 * sp, 2 * sp, 9 * sp), wp.quat_identity()), wp.vec3(2 * sp, 2 * sp, 0.4 * sp),
                      rate=16 * sp**3 / 0.008, velocity=wp.vec3(0.0, 0.0, -0.25))
    s.set_emitter_enabled(e, False)
    a, b = sc.model.state(), sc.model.state()
    a, b = run(s, a, b, 30, 1.0e-3)
    assert s.report().mass_in[0] == 0.0
    s.set_emitter_enabled(e, True)
    a, b = run(s, a, b, 30, 1.0e-3)
    led = s.mass_ledger()
    assert s.report().mass_in[0] > 0.0
    assert np.array_equal(led["active"] + led["out"], led["initial"] + led["in"])
    with pytest.raises(ValueError, match="does not exist"):
        s.set_emitter_enabled(5, True)
