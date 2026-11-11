# SPDX-License-Identifier: Apache-2.0
"""SCR-DC5-19: the proxy CollisionPipeline sized by what can collide (``alloy.coupling.proxy_contacts``), CPU.

- the candidates are Newton's list minus shapes without COLLIDE_PARTICLES, in Newton's order (inactive particles stay:
  emitters activate them at runtime); contacts and Scheduler results are bitwise those of the stock pipeline;
- SolverVBD's construction-time contact state is sized like the pipeline, not every particle x shape;
- rigid pairs: only those with a shape of the destination's own bodies, none (no narrow phase, no reducer) for a
  particle-only destination, soft contacts and results unchanged;
- a full soft- or rigid-contact buffer, the reducer's loss flag or a flag re-enabled after construction is counted on
  the device and, through ``Scheduler.capacity_flags()``, makes HealthMonitor mark every world INVALID (SC0-1).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import CouplingError
from alloy.coupling.proxy_contacts import (COLLIDE_PARTICLES, ProxyContactPipeline, filtered_soft_pairs,
                                           owned_rigid_pairs, soft_pair_count)
from alloy.coupling.scheduler import Scheduler
from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy
from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags
from alloy.testing.coupling import ClothSceneSpec, _world_builder, cloth_entries, cloth_proxy
from alloy._vendor.newton._src.sim import collide as _collide
from alloy._vendor.newton._src.solvers.vbd import solver_vbd as _vbd

ACTIVE = int(newton.ParticleFlags.ACTIVE)
SPEC = ClothSceneSpec(box_mass=0.2, cells=12)     # 3 x 3 particles under each box
DT = 1.0 / 480.0
EXTRA = 5                     # static boxes per world that never meet particles (COLLIDE_PARTICLES off)


def _cell_model(worlds: int, *, inactive: tuple[int, ...] = (), global_particle: bool = False) -> newton.Model:
    """Box-on-cloth worlds (box resting on the cloth) plus EXTRA non-colliding static boxes and one colliding one per
    world, and a global ground plane."""
    top = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(top)
    for w in range(worlds):
        b = _world_builder(SPEC, -0.012 + 0.001 * w)
        off = b.default_shape_cfg.copy()
        off.has_particle_collision = False
        for k in range(EXTRA + 1):
            pose = wp.transform(wp.vec3(2.0 + 0.3 * k, 0.0, 0.5), wp.quat_identity())
            b.add_shape_box(-1, xform=pose, hx=0.1, hy=0.1, hz=0.1, cfg=off if k < EXTRA else b.default_shape_cfg)
        top.add_world(b)
    top.add_ground_plane()
    if global_particle:
        top.add_particle(pos=wp.vec3(5.0, 5.0, 5.0), vel=wp.vec3(0.0), mass=0.0)
    top.color()
    model = top.finalize(device="cpu")
    model.soft_contact_ke, model.soft_contact_mu = SPEC.contact_ke, SPEC.contact_mu
    if inactive:
        flags = model.particle_flags.numpy()
        flags[list(inactive)] &= ~ACTIVE
        model.particle_flags.assign(flags)
    return model


def _scheduler(model: newton.Model, pipeline) -> Scheduler:
    e_m, e_v = cloth_entries(model, SPEC)
    proxy = dataclasses.replace(cloth_proxy(model), collision_pipeline=pipeline)
    return Scheduler(model, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False)],
                     [PairPolicy("mjc", "vbd", mode="lagged", proxy=proxy)], SchedulerPolicy())


def _run(model: newton.Model, sched: Scheduler, steps: int) -> tuple[newton.State, list[dict[str, np.ndarray]]]:
    s0, s1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    out = []
    for _ in range(steps):
        s0.clear_forces()
        pipe.collide(s0, contacts)
        sched.step(s0, s1, control, contacts, DT)
        s0, s1 = s1, s0
        out.append({f: getattr(s0, f).numpy().copy() for f in ("body_q", "body_qd", "particle_q", "particle_qd")})
    return s0, out


def _vbd_hint(sched: Scheduler) -> int:
    return int(sched._entries["vbd"].solver.body_particle_contact_penalty_k.shape[0])


def test_filtered_pairs_are_newtons_minus_skipped():
    model = _cell_model(3, inactive=(0, 5, 50), global_particle=True)
    full = _collide._build_soft_particle_rigid_contact_pairs(model).numpy()
    pf, sf = model.particle_flags.numpy(), model.shape_flags.numpy()
    keep = (sf[full[:, 1]] & COLLIDE_PARTICLES) != 0
    got = filtered_soft_pairs(model).numpy()
    np.testing.assert_array_equal(got, full[keep])               # same pairs, same order
    assert soft_pair_count(model) == len(got) < len(full)
    assert {0, 5, 50} <= set(got[:, 0].tolist())                 # inactive particles stay (emitters activate them)
    assert ((pf[got[:, 0]] & ACTIVE) == 0).any()
    ok = np.arange(model.shape_count) % 2 == 0                   # Newton's own shape_ok composes with the filter
    full_ok = _collide._build_soft_particle_rigid_contact_pairs(model, shape_ok=ok).numpy()
    keep_ok = (sf[full_ok[:, 1]] & COLLIDE_PARTICLES) != 0
    np.testing.assert_array_equal(filtered_soft_pairs(model, shape_ok=ok).numpy(), full_ok[keep_ok])
    assert soft_pair_count(model, shape_ok=ok) == int(keep_ok.sum())


def test_contacts_unchanged():
    model = _cell_model(2)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    ref_pipe = newton.CollisionPipeline(model)
    ref = ref_pipe.contacts()
    ref_pipe.collide(state, ref)
    pipe = ProxyContactPipeline()(model)
    got = pipe.contacts()
    pipe.collide(state, got)
    n = int(ref.soft_contact_count.numpy()[0])
    assert n > 0 and int(got.soft_contact_count.numpy()[0]) == n
    for attr in ("soft_contact_particle", "soft_contact_shape", "soft_contact_body_pos", "soft_contact_normal"):
        np.testing.assert_array_equal(getattr(got, attr).numpy()[:n], getattr(ref, attr).numpy()[:n])
    assert pipe.soft_contact_pair_count < ref_pipe.soft_contact_pair_count


def test_scheduler_bitwise_and_vbd_state_sized():
    model = _cell_model(2)
    stock = _scheduler(model, lambda view: newton.CollisionPipeline(view))
    factory = ProxyContactPipeline(rigid_shapes=())          # VBD owns no bodies: soft pairs only
    sized = _scheduler(model, factory)
    assert _vbd._count_soft_particle_rigid_contact_pairs is _collide._count_soft_particle_rigid_contact_pairs
    pipeline = factory.built[0][1]
    assert _vbd_hint(sized) == int(pipeline.soft_contact_max) == int(pipeline.soft_contact_pair_count)
    assert _vbd_hint(sized) < _vbd_hint(stock)
    assert int(pipeline.rigid_contact_max) == 0
    _, ref = _run(model, stock, 12)
    _, got = _run(model, sized, 12)
    for k, (a, b) in enumerate(zip(ref, got)):
        for f in a:
            assert a[f].tobytes() == b[f].tobytes(), (k, f)
    flags = sized.capacity_flags()                  # the overflow counter + the pass-through flags (SCR-SC4-4)
    (overflow,) = [f for f in flags if not isinstance(f, tuple)]
    assert int(overflow.numpy()[0]) == 0
    assert all(int(f[0].numpy().max()) == 0 for f in flags if isinstance(f, tuple))
    assert stock.capacity_flags() == []


def test_full_buffer_marks_every_world_invalid():
    model = _cell_model(2)
    factory = ProxyContactPipeline(soft_contacts_per_world=1)
    sched = _scheduler(model, factory)
    assert int(factory.built[0][1].soft_contact_max) == 2 and _vbd_hint(sched) == 2
    state, _ = _run(model, sched, 2)
    flags = sched.capacity_flags()
    assert int(flags[0].numpy()[0]) >= 9 * 2 - 2                # 9 particles under each box, 2 slots
    monitor = HealthMonitor(model, CheckPolicy(level="cheap", every=1, full_every=1), capacity_flags=flags)
    monitor.update(0, state, state, None)
    report = monitor.poll()
    assert report.invalid_worlds == (0, 1)
    assert all(int(f) & int(HealthFlags.CONTACT_OVERFLOW) for f in report.flags)
    sched.reset(state)
    assert int(flags[0].numpy()[0]) == 0


def test_flag_enabled_after_construction_is_counted():
    model = _cell_model(1, inactive=(3,))
    factory = ProxyContactPipeline()
    factory(model).contacts()
    factory.update_overflow()
    assert int(factory.overflow.numpy()[0]) == 0
    original = model.shape_flags.numpy().copy()
    sf = original.copy()
    sf[int(np.flatnonzero((sf & COLLIDE_PARTICLES) == 0)[0])] |= COLLIDE_PARTICLES
    model.shape_flags.assign(sf)
    factory.update_overflow()
    assert int(factory.overflow.numpy()[0]) == 1                 # the shape's contacts would be missed
    model.shape_flags.assign(original)
    factory.reset_overflow()
    pf = model.particle_flags.numpy()
    pf[3] |= ACTIVE                                              # an emitter activates a spare: it was a candidate
    model.particle_flags.assign(pf)
    factory.update_overflow()
    assert int(factory.overflow.numpy()[0]) == 0                 # not a missed contact (WP12's drip report)


def test_missing_newton_hook_refuses(monkeypatch):
    monkeypatch.delattr(_collide, "_world_compatible_pairs")
    with pytest.raises(CouplingError, match="SCR-WP4-5"):
        ProxyContactPipeline()(_cell_model(1))
    with pytest.raises(ValueError):
        ProxyContactPipeline(soft_contacts_per_world=0)          # Newton would skip soft contacts, uncounted


def _boxes_on_ground(worlds: int) -> newton.Model:
    """One free box per world resting 1 mm into a global ground plane (4 rigid contacts per world)."""
    top = newton.ModelBuilder()
    for _ in range(worlds):
        b = newton.ModelBuilder()
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()))
        b.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        top.add_world(b)
    top.add_ground_plane()
    return top.finalize(device="cpu")


def test_rigid_pairs_only_on_owned_shapes():
    model = _cell_model(2)
    full = model.shape_contact_pairs.numpy().reshape(-1, 2)
    owned = np.flatnonzero(model.shape_body.numpy() == 0)              # world 0's box
    kept = owned_rigid_pairs(model, owned).numpy()
    np.testing.assert_array_equal(kept, full[np.isin(full, owned).any(axis=1)])   # same rows, same order
    assert 0 < len(kept) < len(full)
    assert owned_rigid_pairs(model, ()).shape[0] == 0
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    ref_pipe = newton.CollisionPipeline(model)
    ref = ref_pipe.contacts()
    ref_pipe.collide(state, ref)
    pipe = ProxyContactPipeline(rigid_shapes=())(model)               # particle-only destination
    got = pipe.contacts()
    pipe.collide(state, got)
    assert int(pipe.rigid_contact_max) == 0
    assert getattr(pipe.narrow_phase, "global_contact_reducer", None) is None
    n = int(ref.soft_contact_count.numpy()[0])
    assert n > 0 and int(got.soft_contact_count.numpy()[0]) == n
    np.testing.assert_array_equal(got.soft_contact_particle.numpy()[:n], ref.soft_contact_particle.numpy()[:n])


def test_rigid_cap_and_reducer_flag_count_as_overflow():
    model = _boxes_on_ground(2)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    boxes = np.flatnonzero(model.shape_body.numpy() >= 0)
    factory = ProxyContactPipeline(rigid_contacts_per_world=1, rigid_shapes=boxes)
    pipe = factory(model)
    contacts = pipe.contacts()
    assert int(pipe.rigid_contact_max) == 2
    pipe.collide(state, contacts)
    factory.update_overflow()
    assert int(factory.overflow.numpy()[0]) >= 4 * 2 - 2              # 4 contacts per box, 2 slots
    roomy = ProxyContactPipeline(rigid_contacts_per_world=8, rigid_shapes=boxes)
    pipe = roomy(model)
    contacts = pipe.contacts()
    pipe.collide(state, contacts)
    roomy.update_overflow()
    assert int(contacts.rigid_contact_count.numpy()[0]) == 8 and int(roomy.overflow.numpy()[0]) == 0
    contacts.contact_counters.assign(np.array([8, 0, 1, 0], dtype=np.int32))   # the reducer dropped contacts
    roomy.update_overflow()
    assert int(roomy.overflow.numpy()[0]) == 1
