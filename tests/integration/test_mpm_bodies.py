# SPDX-License-Identifier: Apache-2.0
"""MPM entries carry their proxy bodies to the step's end (``alloy.integration.mpm_bodies``, speed-now-1109 S5).

Newton's implicit MPM never writes ``state_out.body_q``, so a coupled MPM destination ended every step with stale
proxy bodies; the lagged residual read a moving body's travel as a source-proxy gap and the watchdog escalated the
clock for good (a Panda beside a sand tray: 16 exchanges per 10 ms instead of 8). The carry copies the input bodies
through (default, as SPH) or advances them as the MPM collider did (``ALLOY_MPM_BODY_CARRY=advance``)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.core.errors import SolverOptionError
from alloy.integration import mpm_bodies


def test_the_carry_copies_by_default_and_an_unknown_mode_is_refused(monkeypatch) -> None:
    monkeypatch.delenv(mpm_bodies.ENV, raising=False)
    assert mpm_bodies.mode() == "copy"
    for how in ("off", "advance"):
        monkeypatch.setenv(mpm_bodies.ENV, how)
        assert mpm_bodies.mode() == how
    monkeypatch.setenv(mpm_bodies.ENV, "stale")
    with pytest.raises(SolverOptionError, match="ALLOY_MPM_BODY_CARRY=off"):
        mpm_bodies.mode()


def test_a_body_advances_by_its_twist_about_its_centre_of_mass() -> None:
    """One body, COM 1 cm off its origin, moving at v and spinning at w about z for dt: the COM moves v dt, the
    frame turns w dt about the COM, the velocity is kept."""
    com = np.array([[0.01, 0.0, 0.0]], np.float32)
    q = wp.array([wp.transform((0.1, 0.2, 0.3), wp.quat_identity())], dtype=wp.transform, device="cpu")
    qd = wp.array([wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 2.0)], dtype=wp.spatial_vector, device="cpu")
    q_out = wp.zeros(1, dtype=wp.transform, device="cpu")
    qd_out = wp.zeros(1, dtype=wp.spatial_vector, device="cpu")
    dt = 0.01
    wp.launch(mpm_bodies._advance_kernel, dim=1, inputs=[q, qd, wp.array(com, dtype=wp.vec3, device="cpu"), dt],
              outputs=[q_out, qd_out], device="cpu")
    x = q_out.numpy()[0]
    c0 = np.array([0.11, 0.2, 0.3])
    c1 = c0 + np.array([1.0, 0.0, 0.0]) * dt
    a = 2.0 * dt
    origin = c1 - np.array([math.cos(a), math.sin(a), 0.0]) * 0.01
    np.testing.assert_allclose(x[:3], origin, atol=1e-6)
    np.testing.assert_allclose(x[3:], [0.0, 0.0, math.sin(a / 2), math.cos(a / 2)], atol=1e-6)
    np.testing.assert_array_equal(qd_out.numpy(), qd.numpy())


def _falling_box_rows(periods: int = 8) -> list[tuple[float, float, int]]:
    """A 2 cm steel box falling toward a small sand bed (no contact in these periods): per period the coupler's
    (res_pos [m], dt_x [s], rollbacks)."""
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("sand", size=(0.03, 0.03, 0.01), pos=(0.0, 0.0, 0.0055), material="soil_sand_dry")
    s.add_box("box", size=(0.02, 0.02, 0.02), pos=(0.0, 0.0, 0.09), material="steel_carbon")
    sim = alloy.build(s, device="cpu", seed=0)
    eng, rows = sim.engine, []
    for _ in range(periods):
        sim.step(1)
        rep = eng.stepper.report()
        rows.append((float(np.max(rep.res_pos)), float(rep.dt_x), int(eng.rollbacks)))
    assert float(sim.state(["box"], world=0)["box.pos"][2]) > 0.04          # still falling, above the sand
    return rows


@pytest.mark.law
@pytest.mark.timeout(1800)
def test_a_body_falling_toward_sand_keeps_its_exchange_clock_with_the_carry(monkeypatch) -> None:
    """Off (before 2026-11-09): the stale proxy reads the box's travel as the gap. Copy (the default): the gap is one
    exchange's travel (under 1 m/s x 1.25 ms), advance: the step's curvature (um); no re-run, the clock stays."""
    monkeypatch.delenv(mpm_bodies.ENV, raising=False)                    # copy
    rows = _falling_box_rows()
    assert max(r[0] for r in rows) < 1.5e-3, rows
    assert rows[-1][2] == 0 and len({r[1] for r in rows}) == 1, rows
    monkeypatch.setenv(mpm_bodies.ENV, "advance")
    rows = _falling_box_rows()
    assert max(r[0] for r in rows) < 1e-4, rows
    assert rows[-1][2] == 0 and len({r[1] for r in rows}) == 1, rows
    monkeypatch.setenv(mpm_bodies.ENV, "off")
    rows = _falling_box_rows()
    assert max(r[0] for r in rows) > 5e-3, rows                       # the known stale gap, as measured
