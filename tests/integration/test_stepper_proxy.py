# SPDX-License-Identifier: Apache-2.0
"""SCR-DC5-19 through ``make_stepper`` (CPU): a coupled scene's proxy pipeline gets soft pairs that can collide, no
rigid pairs for a particle-only destination, the profile's per-world caps, and a zero overflow counter after a period
on the health monitor's capacity flags."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alloy.core.errors import CouplingError
from alloy.core.registry import isolated_registry
from alloy.coupling.proxy_contacts import ProxyContactPipeline
from alloy.coupling.types import SchedulerPolicy
from alloy.integration import build_simulation, profile, wire
from alloy.coupling.full_surface import moving_shapes
from alloy.integration.stepper import _proxy_pipeline
from alloy.scene.builtin_plugins import register_builtin_solvers
from alloy.testing import integration as I
from alloy.testing import scene as S


def _randomized():
    return {n: (c, lib) for n, c, lib in S.mixed_scenes()}["randomized"]       # table + box (MuJoCo) + cloth (VBD)


def _factories(stepper):
    base = getattr(stepper, "base", stepper)
    return base._contact_factories()


@pytest.mark.compile_heavy
@pytest.mark.timeout(900)
def test_make_stepper_sizes_the_proxy_pipeline():
    card, lib = _randomized()
    with isolated_registry(empty=True):
        built = I.assemble(card, lib, "cpu", 2)
        pol = SchedulerPolicy(proxy_soft_contacts_per_world=64, proxy_rigid_contacts_per_world=4)
        parts = wire(built, profile("standard", scheduler_policy=pol))
        (factory,) = _factories(parts.stepper)
        assert isinstance(factory, ProxyContactPipeline)
        base = getattr(parts.stepper, "base", parts.stepper)               # SCR-JOB2-3: entries carry capability
        assert {n: ae.capability for n, ae in base._alloy_entries.items()} == \
            {e.name: e.capability for e in built.entries if e.name}
        assert factory.rigid_shapes == () and factory.soft_contacts_per_world == 64
        (view, pipeline, _c, _s) = factory.built[0]
        assert int(pipeline.rigid_contact_max) == 0                    # the cloth owns no bodies: no rigid pairs
        assert int(pipeline.soft_contact_max) <= 64 * 2
        sim = build_simulation(built, "standard")
        (factory,) = _factories(sim.parts.stepper)
        assert factory.soft_contacts_per_world is None and factory.rigid_contacts_per_world is None
        sim.step_period()
        assert np.isfinite(sim.state.particle_q.numpy()).all()
        flags = sim.parts.stepper.capacity_flags()       # overflow counter + pass-through flags (SCR-SC4-4)
        (overflow,) = [f for f in flags if not isinstance(f, tuple)]
        assert int(overflow.numpy()[0]) == 0
        assert all(int(f[0].numpy().max()) == 0 for f in flags if isinstance(f, tuple))


def test_proxy_pipeline_per_destination():
    """No pipeline for a destination that ignores contacts (SPH); rigid pairs only on the destination's own bodies."""
    model = SimpleNamespace(shape_count=4, shape_body=SimpleNamespace(numpy=lambda: np.array([-1, 0, 1, 1], np.int32)),
                            body_count=2, joint_count=0)
    pol = SchedulerPolicy(proxy_soft_contacts_per_world=8)
    with isolated_registry(empty=True):
        register_builtin_solvers()
        assert _proxy_pipeline(model, SimpleNamespace(capability="sph", bodies=()), pol) is None
        vbd = _proxy_pipeline(model, SimpleNamespace(capability="vbd", bodies=()), pol)
        assert vbd.rigid_shapes == () and vbd.soft_contacts_per_world == 8
        own = _proxy_pipeline(model, SimpleNamespace(capability="mujoco", bodies=(1,)), pol)
        assert own.rigid_shapes == (2, 3)


def test_full_surface_follows_the_destination_solver():
    """SCR-SC4-4: edge/face soft contacts for SolverVBD destinations (vbd and its film subclass), off elsewhere; the
    policy can turn them off, and turning them on for a solver that does not read them raises."""
    with isolated_registry(empty=True):
        register_builtin_solvers()                                     # stock solvers + plug-ins (film)
        _check_full_surface()


def _check_full_surface():
    model = SimpleNamespace(shape_count=2, shape_body=SimpleNamespace(numpy=lambda: np.array([-1, 0], np.int32)),
                            body_count=1, joint_count=0)
    auto, on = SchedulerPolicy(), SchedulerPolicy(proxy_full_surface=True)
    for cap, pol in (("vbd", auto), ("film", on)):
        pipe = _proxy_pipeline(model, SimpleNamespace(name=cap, capability=cap, bodies=()), pol)
        assert pipe.full_surface and pipe.kwargs["enable_rigid_soft_full_surface_contact"] is True
        assert pipe.full_surface_shapes.tolist() == [1]                  # the static shape keeps per-vertex
    assert not _proxy_pipeline(model, SimpleNamespace(name="film", capability="film", bodies=()), auto).full_surface
    mjc = SimpleNamespace(name="mujoco", capability="mujoco", bodies=(0,))
    assert not _proxy_pipeline(model, mjc, auto).full_surface
    off = _proxy_pipeline(model, SimpleNamespace(name="vbd", capability="vbd", bodies=()),
                          SchedulerPolicy(proxy_full_surface=False))
    assert not off.full_surface and "enable_rigid_soft_full_surface_contact" not in off.kwargs
    with pytest.raises(CouplingError, match="does not consume"):
        _proxy_pipeline(model, mjc, SchedulerPolicy(proxy_full_surface=True))


def test_c7_gets_full_surface_and_c16_keeps_per_vertex_contact():
    """Coordinator ruling 2026-11-04 20:08Z: C7's towel entry (capability vbd, alloy/clips/fold_scene.py) gets edge/face
    proxy contacts; C16's film entry (capability film, alloy/clips/bag_scene.py) keeps per-vertex contact under the
    automatic policy, a known gap until SCR-SC4-4b passes."""
    from alloy.integration.stepper import FULL_SURFACE_KNOWN_GAPS, _full_surface

    with isolated_registry(empty=True):
        register_builtin_solvers()
        auto = SchedulerPolicy()
        assert _full_surface(SimpleNamespace(name="vbd", capability="vbd"), auto) is True               # C7
        assert _full_surface(SimpleNamespace(name="film", capability="film"), auto) is False            # C16
        assert "SCR-SC4-4b" in FULL_SURFACE_KNOWN_GAPS["film"]
        assert _full_surface(SimpleNamespace(name="film", capability="film"),
                             SchedulerPolicy(proxy_full_surface=True)) is True                         # follow-up runs


def test_moving_shapes_leave_out_static_and_world_welded_bodies():
    """Edge/face contacts go to shapes that can move; a body welded to the world (SCR-SC8-2) and static shapes keep
    per-vertex contact."""
    from alloy._vendor import newton
    import warp as wp

    b = newton.ModelBuilder()
    b.add_shape_box(-1, hx=0.5, hy=0.5, hz=0.01)                               # 0: static
    table = b.add_link(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    b.add_shape_box(table, hx=0.3, hy=0.3, hz=0.01)                            # 1: welded to the world
    arm = b.add_link(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(arm, hx=0.05, hy=0.05, hz=0.05)                            # 2: on a revolute link
    b.add_articulation([b.add_joint_fixed(-1, table), b.add_joint_revolute(-1, arm, axis=(0.0, 0.0, 1.0))])
    assert moving_shapes(b.finalize(device="cpu")) == [2]
