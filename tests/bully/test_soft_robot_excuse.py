# SPDX-License-Identifier: Apache-2.0
"""The ledger's robot-touch excuse for soft contacts (WP15's ask, 2026-11-09): a robot body touching particles (a blade
in a VBD solid) excuses its world's step like a rigid robot contact, since the ledger does not measure that work; not
when a shape-work source books it (the driven SPH cup, SCR-WP9-6), and never for a passive body (CPU, kernel level)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.bully import kernels as K

DEV = "cpu"


def _touch(robot: int, excuse: int) -> tuple[int, int]:
    arr = lambda v, dt=wp.int32: wp.array(np.asarray(v), dtype=dt, device=DEV)  # noqa: E731
    touch_any, world_touch = arr([0, 0]), arr([0, 0])
    wp.launch(K.soft_touch_kernel, dim=2, device=DEV, inputs=[
        arr([1]), arr([1, -1]), arr([0, 1]), arr([0, robot]), arr([0, 1]), excuse, touch_any, world_touch])
    return int(touch_any.numpy()[1]), int(world_touch.numpy()[1])           # body 1 (world 1) touched particles


@pytest.mark.parametrize("robot, excuse, want", [(1, 1, (1, 1)), (1, 0, (1, 0)), (0, 1, (1, 0))])
def test_a_robot_touching_particles_excuses_its_world(robot, excuse, want):
    assert _touch(robot, excuse) == want


def test_the_monitor_excuses_only_without_a_shape_work_source(monkeypatch):
    """BenchMonitors passes the excuse when no ShapeWorkSource books the robot's work on particles."""
    from types import SimpleNamespace

    from alloy.bully import monitors as M

    seen = []
    real = wp.launch
    monkeypatch.setattr(M.wp, "launch", lambda k, *a, **kw: (seen.append(kw.get("inputs")) if k is K.soft_touch_kernel
                                                             else None, real(k, *a, **kw))[1])
    for sources, want in (((), 1), ((object(),), 0)):
        seen.clear()
        mon = SimpleNamespace(model=SimpleNamespace(body_count=1, shape_body=wp.zeros(1, dtype=wp.int32, device=DEV),
                                                    body_world=wp.zeros(1, dtype=wp.int32, device=DEV), device=DEV),
                              work_sources=sources, _robot=wp.zeros(1, dtype=wp.int32, device=DEV),
                              _touch_any=wp.zeros(1, dtype=wp.int32, device=DEV),
                              _touch_robot=wp.zeros(1, dtype=wp.int32, device=DEV),
                              _world_robot_touch=wp.zeros(1, dtype=wp.int32, device=DEV),
                              _flight=SimpleNamespace(touch_prev=wp.zeros(1, dtype=wp.int32, device=DEV),
                                                      robot_prev=wp.zeros(1, dtype=wp.int32, device=DEV)))
        contacts = SimpleNamespace(rigid_contact_max=0, soft_contact_max=1,
                                   soft_contact_count=wp.zeros(1, dtype=wp.int32, device=DEV),
                                   soft_contact_shape=wp.zeros(1, dtype=wp.int32, device=DEV))
        M.BenchMonitors._contacts(mon, None, contacts)
        assert len(seen) == 1 and seen[0][5] == want
