# SPDX-License-Identifier: Apache-2.0
"""SCR-SC4-4 (CPU): a particle whose centre is inside a rigid or proxy shape is a pass-through event. Its world turns
INVALID (PAIR_PEN_FAIL) through the contact layer's or the Scheduler's capacity flags, sticky until that world's
flags are reset; particles in contact but outside raise nothing."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.newton_compat import SolverCoupledProxy
from alloy.coupling.pass_through import PASS_THROUGH_BITS
from alloy.coupling.proxy_contacts import ProxyContactPipeline
from alloy.coupling.scheduler import Scheduler
from alloy.coupling.types import AlloyEntry, PairPolicy, SchedulerPolicy
from alloy.precision.health import HealthMonitor
from alloy.precision.types import CheckPolicy, HealthFlags
from alloy.scene.steppers import NewtonContacts
from alloy.testing.coupling import BOX_HALF, ClothSceneSpec, cloth_entries, coupling_box_on_cloth
from alloy.coupling.pinch_rigs import cup_mesh

_CUBE_V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float32) * 0.1
_CUBE_F = np.array([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                    1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7], np.int32)
R = 0.005                                      # particle radius [m]


def _cube_and_particles(kind: str, inside_world: int | None, worlds: int = 2, depth: float = 0.05) -> newton.Model:
    """Per world a 0.2 m cube (box or mesh) on a body at z 0.1 and two particles in contact with its top face
    (centres 3 mm above it); in ``inside_world`` the second one's centre is ``depth`` below the top face instead."""
    b = newton.ModelBuilder()
    for w in range(worlds):
        sub = newton.ModelBuilder()
        body = sub.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
        if kind == "mesh":
            sub.add_shape_mesh(body, mesh=newton.Mesh(_CUBE_V, _CUBE_F))
        else:
            sub.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        sub.add_particle((0.0, 0.0, 0.203), (0.0, 0.0, 0.0), 0.01, radius=R)
        sub.add_particle((0.05, 0.05, 0.2 - depth if w == inside_world else 0.203), (0.0, 0.0, 0.0), 0.01, radius=R)
        b.add_world(sub)
    return b.finalize(device="cpu")


def _collided(model: newton.Model) -> tuple[ProxyContactPipeline, newton.State, int]:
    """A soft-only proxy pipeline on the whole model, collided once: (factory, state, soft contact count)."""
    factory = ProxyContactPipeline(rigid_shapes=())
    pipeline = factory(model)
    contacts = pipeline.contacts()
    state = model.state()
    pipeline.collide(state, contacts)
    return factory, state, int(contacts.soft_contact_count.numpy()[0])


@pytest.mark.parametrize(("kind", "depth"), [("box", 0.05), ("box", 0.002), ("mesh", 0.002)])
def test_centre_inside_flags_only_that_world(kind, depth):
    """A box is checked at any depth; a mesh only within margin + radius of its surface, where Newton's mesh query
    still returns the closest point (deeper, no contact is generated at all)."""
    factory, _, n = _collided(_cube_and_particles(kind, inside_world=1, depth=depth))
    assert n == 4                                                             # every particle is in contact
    assert factory.pass_through.flags.numpy().tolist() == [0, 1]
    clean, _, _ = _collided(_cube_and_particles(kind, inside_world=None))
    assert clean.pass_through.flags.numpy().tolist() == [0, 0]


def test_pass_through_turns_its_world_invalid_until_reset():
    model = _cube_and_particles("box", inside_world=1)
    factory, state, _ = _collided(model)
    monitor = HealthMonitor(model, CheckPolicy(level="cheap", every=1, full_every=1),
                            capacity_flags=[factory.pass_through.health_source()])
    monitor.update(0, state, state, None)
    report = monitor.poll()
    assert report.invalid_worlds == (1,)
    assert int(report.flags[1]) & int(HealthFlags.PAIR_PEN_FAIL)
    factory.pass_through.reset(wp.array([0, 1], dtype=wp.int32, device="cpu"))
    assert factory.pass_through.flags.numpy().tolist() == [0, 0]


def test_fallback_layer_does_not_check():
    """The fallback layer's particles include SPH (its own leak monitor): no pass-through flags there."""
    layer = NewtonContacts(_cube_and_particles("box", inside_world=1), rigid=False)
    assert layer._factory.pass_through is None


def _scheduler(model: newton.Model, spec: ClothSceneSpec) -> Scheduler:
    e_m, e_v = cloth_entries(model, spec)
    proxy = SolverCoupledProxy.Proxy(source="mjc", destination="vbd", bodies=list(range(model.body_count)),
                                     mode="lagged", proxy_relaxation_mode="fixed",
                                     collision_pipeline=ProxyContactPipeline(), collide_interval=1)
    return Scheduler(model, [AlloyEntry(e_m, adaptive=False), AlloyEntry(e_v, adaptive=False)],
                     [PairPolicy("mjc", "vbd", mode="lagged", proxy=proxy)], SchedulerPolicy())


def _run(model, sched, state, steps: int) -> newton.State:
    s1, control = model.state(), model.control()
    pipe = newton.CollisionPipeline(model)
    contacts = pipe.contacts()
    for _ in range(steps):
        state.clear_forces()
        pipe.collide(state, contacts)
        sched.step(state, s1, control, contacts, 1.0 / 480.0)
        state, s1 = s1, state
    return state


def _flags(sched) -> list[int]:
    (flags,) = [x[0] for x in sched.capacity_flags() if isinstance(x, tuple) and x[1] == PASS_THROUGH_BITS]
    return flags.numpy().tolist()


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_scheduler_flags_a_particle_inside_its_proxy():
    """Box on cloth (2 worlds): the box lands and rests with no event; a cloth particle of world 0 moved into the
    box's centre is flagged at the next exchange; resetting world 0's flags clears it."""
    spec = ClothSceneSpec(cells=8)
    model = coupling_box_on_cloth(2, spec, device="cpu")
    sched = _scheduler(model, spec)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    state = _run(model, sched, state, 96)                                     # drop and settle, 0.2 s
    assert _flags(sched) == [0, 0]
    box = state.body_q.numpy()[0, :3]
    q = state.particle_q.numpy()
    i = int(np.flatnonzero(model.particle_world.numpy() == 0)[np.argmin(
        np.linalg.norm(q[model.particle_world.numpy() == 0] - box, axis=1))])
    q[i] = box + np.array([0.0, 0.0, -0.5 * BOX_HALF], np.float32)          # centre 2.5 cm inside the box
    state.particle_q.assign(q)
    _run(model, sched, state, 2)
    assert _flags(sched) == [1, 0]
    sched.reset_world_flags(wp.array([1, 0], dtype=wp.int32, device="cpu"))
    assert _flags(sched) == [0, 0]


@pytest.mark.parametrize("depth", [0.002, 0.05])
def test_full_surface_mesh_flags_a_centre_at_any_depth(depth):
    """SCR-SC4-4: under full-surface contact Newton's mesh pass also reports a deep vertex (a recovery record), so a
    mesh is checked at any depth; resolved records only."""
    for inside, expected in ((1, [0, 1]), (None, [0, 0])):
        model = _cube_and_particles("mesh", inside_world=inside, depth=depth)
        factory = ProxyContactPipeline(rigid_shapes=(), full_surface=True)
        pipeline = factory(model)
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        assert factory.pass_through.flags.numpy().tolist() == expected


@pytest.mark.parametrize("full", [False, True])
def test_thin_walled_cup_flags_the_wall_not_the_cavity(full):
    """SCR-JOB2-4's hollow-mesh check: particles in a 1.8 mm-walled cup's cavity, 1.05 r off the inner wall, raise
    nothing; one with its centre inside the wall does (per-vertex and full-surface; Newton's full-surface mesh pass
    reports ~42 records per particle there, so the capacity is set explicitly)."""
    v, f = cup_mesh()
    for wall, expected in ((False, [0]), (True, [1])):
        b = newton.ModelBuilder()
        body = b.add_body(xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()))
        b.add_shape_mesh(body, mesh=newton.Mesh(v, f))
        for a in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
            r = 0.0382 - 1.05 * R
            b.add_particle((r * np.cos(a), r * np.sin(a), 0.055), (0.0, 0.0, 0.0), 1e-5, radius=R)
        if wall:
            b.add_particle((0.0391, 0.0, 0.055), (0.0, 0.0, 0.0), 1e-5, radius=R)
        model = b.finalize(device="cpu")
        factory = ProxyContactPipeline(rigid_shapes=(), full_surface=full, soft_contacts_per_world=4000)
        pipeline = factory(model)
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        assert int(contacts.soft_contact_count.numpy()[0]) <= int(contacts.soft_contact_max)
        assert factory.pass_through.flags.numpy().tolist() == expected, (full, wall)
