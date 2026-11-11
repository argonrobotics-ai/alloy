# SPDX-License-Identifier: Apache-2.0
"""SCR-JOB2-3 (CPU): the exchange-clock bound safety x omega_dt_max x sqrt(m_eff / k_c) models a penalty spring at the
interface. A contact-free destination (SPH: its wall force comes from the pressure solve) gets no default k_c, so no
bound; a pair-stated ``interface_stiffness`` (D17) still binds; contact-based destinations (VBD) keep the contact
stiffness."""

from __future__ import annotations

import dataclasses

import math

import pytest

from alloy.core.newton_compat import SolverCoupledProxy
from alloy.coupling.clocks import damping_bound, interface_bound
from alloy.coupling.proxy_contacts import contact_free
from alloy.coupling.scheduler import Scheduler
from alloy.coupling.types import AlloyEntry, ClockPolicy, PairPolicy, SchedulerPolicy
from alloy.solvers.sph.solver import SolverDFSPH
from alloy.testing.coupling import ClothSceneSpec, cloth_scheduler, coupling_box_on_cloth
from alloy.testing.sph import default_config, floating_box

pytestmark = pytest.mark.compile_heavy

BOX = {"nx": 8, "ny": 8, "nz": 6, "box_cells": 4, "box_height_cells": 3.25, "box_density": 800.0}
POLICY = SchedulerPolicy(clock=ClockPolicy(omega_dt_max=1.0))


def _mujoco_sph(stiffness: float | None) -> Scheduler:
    """Two-way lagged MuJoCo -> SPH pair (a floating box in a tank)."""
    from alloy._vendor.newton.solvers import SolverMuJoCo

    cfg = default_config(coupling="two_way")
    sc = floating_box(**BOX, cfg=cfg)
    m, box = sc.model, int(sc.bodies["box"])
    e_m = SolverCoupledProxy.Entry(name="mjc", bodies=[box], joints=list(range(m.joint_count)),
                                   solver=lambda v: SolverMuJoCo(model=v, use_mujoco_contacts=False, njmax=50))
    e_s = SolverCoupledProxy.Entry(name="sph", particles=list(range(m.particle_count)), substeps=4,
                                   solver=lambda v: SolverDFSPH(v, cfg))
    proxy = SolverCoupledProxy.Proxy(source="mjc", destination="sph", bodies=[box], mode="lagged",
                                     collision_pipeline=None)
    return Scheduler(m, [AlloyEntry(e_m, adaptive=False, capability="mujoco"),
                         AlloyEntry(e_s, initial_rung=4, adaptive=False, capability="sph")],
                     [PairPolicy("mjc", "sph", mode="lagged", proxy=proxy, interface_stiffness=stiffness)], POLICY)


def test_sph_destination_gets_no_contact_stiffness_bound():
    sched = _mujoco_sph(None)
    assert contact_free(sched._entries["sph"].solver, "sph")
    assert sched._default_stiffness() > 0.0                      # a contact ke exists, but does not apply
    assert sched._pair_stiffness_mass() == []
    assert math.isinf(sched._interface_bound())


def test_stated_interface_stiffness_still_binds():
    sched = _mujoco_sph(1.0e3)                                    # JOB2's interim K_INTERFACE_SPH
    ((k_c, m_eff),) = sched._pair_stiffness_mass()
    assert k_c == 1.0e3 and math.isfinite(m_eff)
    clock = POLICY.clock
    assert sched._interface_bound() == pytest.approx(interface_bound(1.0e3, m_eff, clock.safety, clock.omega_dt_max))


def test_vbd_destination_keeps_the_contact_stiffness_bound():
    spec = ClothSceneSpec(cells=8)
    model = coupling_box_on_cloth(1, spec, device="cpu")
    sched = cloth_scheduler(model, spec, policy=POLICY, vbd_adaptive=False)
    assert not contact_free(sched._entries["vbd"].solver)
    ((k_c, m_eff),) = sched._pair_stiffness_mass()
    assert k_c == sched._default_stiffness() == max(spec.contact_ke, spec.shape_ke)
    assert math.isfinite(sched._interface_bound())


DAMPED = SchedulerPolicy(clock=ClockPolicy(omega_dt_max=1.0, damping_bound=True))


def test_damping_bound_is_opt_in():
    spec = ClothSceneSpec(cells=8, box_mass=1.2e-3)
    model = coupling_box_on_cloth(1, spec, device="cpu")
    sched = cloth_scheduler(model, spec, policy=POLICY, vbd_adaptive=False)
    ((k_c, m_eff),) = sched._pair_stiffness_mass()
    clock = POLICY.clock
    assert sched._interface_bound() == pytest.approx(interface_bound(k_c, m_eff, clock.safety, clock.omega_dt_max))


def test_contact_damping_bounds_light_bodies_only():
    """The held contact damping force adds safety x omega_dt_max x m_eff / c_d (c_d: VBD's mean of soft_contact_kd and
    the shape kd). For a 0.2 kg box on cloth the spring bound still binds; for a 1.2 g box the damping bound does."""
    clock = DAMPED.clock
    for box, damping_binds in ((0.2, False), (1.2e-3, True)):
        spec = ClothSceneSpec(cells=8, box_mass=box)
        model = coupling_box_on_cloth(1, spec, device="cpu")
        sched = cloth_scheduler(model, spec, policy=DAMPED, vbd_adaptive=False)
        ((c_d, m_eff),) = sched._pair_damping_mass()
        assert c_d == pytest.approx(0.5 * (float(model.soft_contact_kd) + float(model.shape_material_kd.numpy().max())))
        spring = interface_bound(sched._default_stiffness(), m_eff, clock.safety, clock.omega_dt_max)
        damp = damping_bound(c_d, m_eff, clock.safety, clock.omega_dt_max)
        assert sched._interface_bound() == pytest.approx(min(spring, damp))
        assert (damp < spring) is damping_binds


def test_sph_destination_gets_no_damping_bound():
    assert _mujoco_sph(None)._pair_damping_mass() == []


@dataclasses.dataclass(frozen=True)
class _Rest:
    """A light box run [m], each relative to its start: the box (``low``, ``high``), the box over the cloth vertex
    under it (``on_low``, ``on_high``: below zero it presses in, above it lifts off), and that vertex's lowest."""

    low: float
    high: float
    on_low: float
    on_high: float
    vertex_low: float


def _light_box_heights(monkeypatch, damping: bool) -> _Rest:
    """A 5 g box landing on cloth with a soft contact (ke 500, Newton's default kd: c_d 55 N s/m), 0.1 s at the
    clock's dt_x, with or without the damping bound."""
    from alloy._vendor import newton
    import numpy as np

    from alloy.coupling.scheduler import Scheduler as S

    if not damping:
        monkeypatch.setattr(S, "_pair_damping_mass", lambda self: [])
    spec = ClothSceneSpec(cells=4, box_mass=5.0e-3, contact_ke=500.0, shape_ke=500.0)
    model = coupling_box_on_cloth(1, spec, device="cpu")
    sched = cloth_scheduler(model, spec, policy=DAMPED, vbd_adaptive=False)
    dt, _ = sched.choose_clock(1.0 / 60.0, None)
    s0, s1, ctl = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    con = pipe.contacts()
    b0, p0 = s0.body_q.numpy()[0].copy(), s0.particle_q.numpy().copy()
    under = int(np.argmin(np.linalg.norm(p0[:, :2] - b0[:2], axis=1)))      # the vertex under the box
    rows = []
    for _ in range(int(round(0.1 / dt))):
        s0.clear_forces()
        pipe.collide(s0, con)
        sched.step(s0, s1, ctl, con, dt)
        s0, s1 = s1, s0
        z, v = float(s0.body_q.numpy()[0, 2]) - float(b0[2]), float(s0.particle_q.numpy()[under, 2] - p0[under, 2])
        rows.append((z, z - v, v))
    z, on, v = (np.array(c) for c in zip(*rows, strict=True))
    return _Rest(float(z.min()), float(z.max()), float(on.min()), float(on.max()), float(v.min()))


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_light_box_rests_only_with_the_damping_bound(monkeypatch):
    """WP4 P3 law (2026-11-06; restated 2026-11-10 after WP11's F32 fb7d80a8a). The box lands on a 4-cell cloth
    released flat with 10 of its 25 vertices pinned; the cloth sags under its own weight and swings, the box rides it.
    With the spring bound alone (dt_x 1.04 ms, c_d dt / m = 11) the lagged exchange holds kd x v for the whole interval
    and throws the box 30.4 mm off the vertex under it (14.5 mm above its start). With the damping bound (dt_x 33 us) it
    stays on its vertex: at most 2.1 mm into it, never above, and never above its start (CPU, F32 and before alike).
    The cloth sags the same at both steps (lowest -20.6 mm). Before F32 it did not at 33 us: g dt^2 = 1e-8 m is under
    one float32 ulp of z at 1 m, so the cloth stayed frozen while the box crept into it at 19 mm/s, which this law
    read as "rests 3.4 mm below its start" (WP4, [internal note] scr/WP4.md 2026-11-10)."""
    on = _light_box_heights(monkeypatch, damping=True)
    assert on.high < 1.0e-3 and -3.0e-3 < on.on_low and on.on_high < 1.0e-3, on
    monkeypatch.undo()
    off = _light_box_heights(monkeypatch, damping=False)
    assert off.on_high > 5.0e-3 and off.high > 5.0e-3, off
    assert abs(on.vertex_low - off.vertex_low) < 2.0e-3 and off.vertex_low < -10.0e-3, (on, off)   # no freeze
