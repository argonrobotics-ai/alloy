# SPDX-License-Identifier: Apache-2.0
"""Laws of the self-contact reach from a motion bound (``alloy.solvers.vbd.reach``, coordinator ruling 2026-11-08 08:40Z
(b)): nothing in a self-contact entry slower than the bound is truncated.

- A 0.6 mm sheet (the towel card's thickness) dropped with self contact on falls freely: within 2 % of g t at 0.3 s.
  The control, the route's old reach (gap = the thickness), holds it at 0.85 x 2h / dt = 0.816 m/s at the 1.25 ms step.
- The gap follows each step's dt: max(0, min(v dt / 0.85, 0.75 l) - margin), l the median rest edge.
- On a fine mesh the reach stops at 0.75 of an edge (pair buffers overflow beyond a cell).
- An explicit gap wins: the rule is not mixed in.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy
from alloy._vendor import newton
from alloy.integration.throughput import with_entry_kwargs
from alloy.solvers.vbd import make_vbd
from alloy.solvers.vbd.reach import REACH_CELLS, SELF_CONTACT_SPEED, SPEED_KEY, ReachMixin, reach_gap

H = 6.0e-4          # [m] the towel card's thickness


def _fall_speed(override: dict, t: float = 0.3) -> float:
    """Mean downward speed [m/s] of a self-contact sheet at time t after release (the route's VBD entry)."""
    import alloy.api._plugins as B

    inner = B.build_engine
    B.build_engine = lambda built, *a, **k: inner(with_entry_kwargs(built, override, "vbd"), *a, **k)
    try:
        scene = alloy.Scene(dt=0.01)
        scene.add_ground()
        scene.add_cloth("sheet", size=(0.2, 0.2), pos=(0, 0, 1.2), resolution=8, mass=0.02, thickness=H)
        sim = alloy.build(scene, device="cpu")
    finally:
        B.build_engine = inner
    z = sim.run(t, record=["sheet"])["sheet.points"][:, 0, :, 2].mean(axis=1)
    return float(-(z[-1] - z[-2]) / 0.01)


def test_a_self_contact_sheet_falls_freely():
    assert _fall_speed({}) == pytest.approx(9.81 * 0.295, rel=0.02)            # the last period's midpoint


def test_the_old_reach_caps_the_fall():
    v = _fall_speed({"particle_self_contact_gap": H})
    assert v == pytest.approx(0.85 * 2.0 * H / 1.25e-3, rel=0.01), v


def _cloth_model(cell: float = 0.01) -> newton.Model:
    b = newton.ModelBuilder()
    b.add_cloth_grid(pos=wp.vec3(0.0, 0.0, 0.5), rot=wp.quat_identity(), vel=wp.vec3(0.0), dim_x=4, dim_y=4,
                     cell_x=cell, cell_y=cell, mass=1.0e-3, tri_ke=1.0e3, tri_ka=1.0e3, tri_kd=1.0e-2,
                     particle_radius=H / 2)
    b.color()
    return b.finalize(device="cpu")


def test_the_gap_follows_each_step():
    m = _cloth_model()
    solver = make_vbd(m, {"iterations": 2, "particle_enable_self_contact": True, "particle_self_contact_margin": H,
                          SPEED_KEY: SELF_CONTACT_SPEED})
    assert isinstance(solver, ReachMixin)
    s0, s1, ctl = m.state(), m.state(), m.control()
    assert solver.reach_cap == pytest.approx(REACH_CELLS * 0.01)                # median rest edge: the cell
    for dt in (1.25e-3, 2.5e-4):
        solver.step(s0, s1, ctl, None, dt)
        want = reach_gap(H, dt, SELF_CONTACT_SPEED, 0.85)                          # under the cap here
        assert solver.particle_self_contact_gap == pytest.approx(want)
        assert solver._self_contact_query_radius == pytest.approx(H + want)
        s0, s1 = s1, s0
    assert np.isfinite(s0.particle_q.numpy()).all()
    assert reach_gap(H, 2.5e-4, SELF_CONTACT_SPEED, 0.85) > 0.0 and reach_gap(4e-3, 2.5e-4, 5.0, 0.85) == 0.0


def test_on_a_fine_mesh_the_reach_stops_under_a_cell():
    m = _cloth_model(cell=0.002)
    solver = make_vbd(m, {"iterations": 2, "particle_enable_self_contact": True, "particle_self_contact_margin": H,
                          SPEED_KEY: SELF_CONTACT_SPEED})
    s0, s1, ctl = m.state(), m.state(), m.control()
    solver.step(s0, s1, ctl, None, 1.25e-3)
    assert solver._self_contact_query_radius == pytest.approx(REACH_CELLS * 0.002)    # 1.5 mm, not 4.4 mm


def test_an_explicit_gap_wins():
    m = _cloth_model()
    solver = make_vbd(m, {"iterations": 2, "particle_enable_self_contact": True, "particle_self_contact_margin": H,
                          "particle_self_contact_gap": 2.0e-4, SPEED_KEY: SELF_CONTACT_SPEED})
    assert not isinstance(solver, ReachMixin) and solver.particle_self_contact_gap == pytest.approx(2.0e-4)
