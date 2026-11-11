# SPDX-License-Identifier: Apache-2.0
"""SCR-SC5-3: soft contacts between a Blade and a twin whose bond it is cutting (phase 1) leave the PEN rule.

Rig: a 2 mm thick blade (a box on its own body, as WP10's Blade) and particles of radius 0.5 mm, two worlds, Newton's
own soft contacts. The normal soft-contact limit is min(pen_fail, half the thinner of particle and shape) = 0.5 mm; a
particle 0.3 mm inside the blade face penetrates 0.8 mm. The cut-front limit (re-ruled 19:05Z) is twice the blade's
half-thickness plus the particle radius = 2.5 mm, and a cut-front contact may not get more than 0.1 mm deeper than at
the twin's first cut-front check. The predicate is WP10's: ``cut_front_blade`` holds the Blade's body id.
- a twin at the cut front (``cut_front_blade`` = the blade body): no PEN_FAIL; ``cut_front_pen_m`` 0.8 mm, 1 step;
- a plain particle at the same depth (``-1``): PEN_FAIL, as before;
- the same twin once its bond is past phase 1 (entry back to -1) and still in the blade: PEN_FAIL;
- a cut-front contact deeper than its limit (contact point moved 2.1 mm, so 2.9 mm deep): PEN_FAIL;
- a cut-front contact 0.05 mm deeper than at its first check stays exempt; 0.15 mm deeper is PEN_FAIL;
- the growth limit never goes below a normal contact's (ruling 2026-11-04 11:30Z: max(0.5 mm, baseline + 0.1 mm)): a
  twin first seen at the surface (baseline 0) pressed 0.3 mm and 0.45 mm in stays exempt; 0.6 mm is PEN_FAIL;
- without ``set_cut_front`` the monitor behaves as before (both inside particles fail).
"""

from __future__ import annotations

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags

R, HALF_T, INSIDE = 5e-4, 1e-3, 3e-4        # particle radius, blade half thickness, depth of the centre past the face
POLICY = CheckPolicy(level="cheap", every=1, full_every=1)


def _rig():
    """World w: blade at the origin; particle 0 inside the +y face by INSIDE, particle 1 well outside."""
    b = newton.ModelBuilder()
    blade = b.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()), mass=0.1)
    b.add_shape_box(blade, hx=0.02, hy=HALF_T, hz=0.02)
    b.add_particle(pos=(0.0, HALF_T - INSIDE, 0.1), vel=(0.0, 0.0, 0.0), mass=1e-3, radius=R)
    b.add_particle(pos=(0.0, 0.05, 0.1), vel=(0.0, 0.0, 0.0), mass=1e-3, radius=R)
    s = newton.ModelBuilder()
    s.replicate(b, 2)
    model = s.finalize(device="cpu")
    state = model.state()
    pipeline = newton.CollisionPipeline(model, soft_contact_margin=0.01)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return model, state, contacts


def _blade_of(model, w: int) -> int:
    return int(np.flatnonzero(model.body_world.numpy() == w)[0])


def _half(model) -> wp.array:
    return wp.array(np.full(model.body_count, HALF_T, np.float32), dtype=wp.float32, device="cpu")


def _inside(model, w: int) -> int:
    return int(np.flatnonzero(model.particle_world.numpy() == w)[0])


def _pen_fail(report, w: int) -> bool:
    return bool(int(report.flags[w]) & int(HealthFlags.PEN_FAIL))


@pytest.fixture
def rig():
    model, state, contacts = _rig()
    n = int(contacts.soft_contact_count.numpy()[0])
    assert n >= 2, "the inside particles must touch their blades"
    return model, state, contacts


def test_cut_front_twin_is_reported_not_flagged(rig) -> None:
    model, state, contacts = rig
    front = np.full(model.particle_count, -1, np.int32)
    front[_inside(model, 0)] = _blade_of(model, 0)        # world 0: a twin at the cut front; world 1: a plain particle
    monitor = HealthMonitor(model, POLICY)
    monitor.set_cut_front(wp.array(front, dtype=wp.int32, device="cpu"), _half(model))
    monitor.update(0, state, state, contacts)
    report = monitor.poll()
    assert not _pen_fail(report, 0) and report.invalid_worlds == (1,) and _pen_fail(report, 1)
    np.testing.assert_allclose(report.cut_front_pen_m[0], INSIDE + R, rtol=1e-4)
    assert int(report.cut_front_steps[0]) == 1 and int(report.cut_front_steps[1]) == 0
    assert report.cut_front_pen_m[1] == 0.0


def test_twin_past_phase_one_still_in_the_blade_flags(rig) -> None:
    model, state, contacts = rig
    front = wp.array(np.full(model.particle_count, -1, np.int32), dtype=wp.int32, device="cpu")
    monitor = HealthMonitor(model, POLICY)
    monitor.set_cut_front(front, _half(model))
    f = front.numpy()
    f[_inside(model, 0)] = _blade_of(model, 0)
    front.assign(f)
    monitor.update(0, state, state, contacts)
    assert not _pen_fail(monitor.poll(), 0)
    f[_inside(model, 0)] = -1                              # the bond moved to phase 2; the twin is still inside
    front.assign(f)
    monitor.update(1, state, state, contacts)
    report = monitor.poll()
    assert _pen_fail(report, 0) and 0 in report.invalid_worlds


def test_cut_front_deeper_than_its_limit_flags(rig) -> None:
    model, state, contacts = rig
    pts = contacts.soft_contact_body_pos.numpy()
    part = contacts.soft_contact_particle.numpy()
    k = int(np.flatnonzero(part == _inside(model, 0))[0])
    pts[k, 1] += 2.1e-3                                    # 0.8 mm + 2.1 mm = 2.9 mm > 2 * 1 mm + 0.5 mm
    contacts.soft_contact_body_pos.assign(pts)
    front = np.full(model.particle_count, -1, np.int32)
    front[_inside(model, 0)] = _blade_of(model, 0)
    monitor = HealthMonitor(model, POLICY)
    monitor.set_cut_front(wp.array(front, dtype=wp.int32, device="cpu"), _half(model))
    monitor.update(0, state, state, contacts)
    report = monitor.poll()
    assert _pen_fail(report, 0) and report.cut_front_pen_m[0] == 0.0


def test_cut_front_may_not_grow_past_its_first_depth(rig) -> None:
    model, state, contacts = rig
    pts0 = contacts.soft_contact_body_pos.numpy().copy()
    k = int(np.flatnonzero(contacts.soft_contact_particle.numpy() == _inside(model, 0))[0])
    front = np.full(model.particle_count, -1, np.int32)
    front[_inside(model, 0)] = _blade_of(model, 0)
    monitor = HealthMonitor(model, POLICY)
    monitor.set_cut_front(wp.array(front, dtype=wp.int32, device="cpu"), _half(model))
    for step, grow in enumerate((0.0, 0.05e-3, 0.15e-3)):   # first check sets the 0.8 mm baseline
        pts = pts0.copy()
        pts[k, 1] += grow
        contacts.soft_contact_body_pos.assign(pts)
        monitor.update(step, state, state, contacts)
        report = monitor.poll()
        assert _pen_fail(report, 0) == (grow > 0.1e-3), (grow, report.flags)


def test_growth_limit_is_never_tighter_than_a_normal_contact(rig) -> None:
    model, state, contacts = rig
    pts0 = contacts.soft_contact_body_pos.numpy().copy()
    k = int(np.flatnonzero(contacts.soft_contact_particle.numpy() == _inside(model, 0))[0])
    front = np.full(model.particle_count, -1, np.int32)
    front[_inside(model, 0)] = _blade_of(model, 0)
    monitor = HealthMonitor(model, POLICY)
    monitor.set_cut_front(wp.array(front, dtype=wp.int32, device="cpu"), _half(model))
    for step, depth in enumerate((-0.05e-3, 0.3e-3, 0.45e-3, 0.6e-3)):   # first check: outside, baseline 0
        pts = pts0.copy()
        pts[k, 1] += depth - (INSIDE + R)                   # the rig's contact is INSIDE + R deep
        contacts.soft_contact_body_pos.assign(pts)
        monitor.update(step, state, state, contacts)
        report = monitor.poll()
        assert _pen_fail(report, 0) == (depth > 0.5e-3), (depth, report.flags)


def test_without_the_predicate_nothing_changes(rig) -> None:
    model, state, contacts = rig
    monitor = HealthMonitor(model, POLICY)
    monitor.update(0, state, state, contacts)
    report = monitor.poll()
    assert report.invalid_worlds == (0, 1) and all(_pen_fail(report, w) for w in range(2))
    assert np.all(report.cut_front_pen_m == 0.0) and np.all(report.cut_front_steps == 0)
    with pytest.raises(ValueError, match="int32 of length"):
        monitor.set_cut_front(wp.zeros(3, dtype=wp.int32, device="cpu"), _half(model))
