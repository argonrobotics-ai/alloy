# SPDX-License-Identifier: Apache-2.0
"""SolverVBD's history has an item -> world map for every per-joint, per-DOF and per-slot array (alloy.coupling.
adapters.vbd_world_maps), so a rollback restores only the escalated worlds.

Found in WP12's review of WP10's copies entry (2026-11-06): Newton sizes ``joint_angle_prev`` per DOF, the adapter
mapped it per joint, and ``joint_penalty_k`` (per joint-constraint slot) had no map. Either one left a whole array in
the snapshot, and Simulation then rolled back every world of any VBD scene with a joint.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.snapshot import WorldSnapshot
from alloy.coupling.adapters import vbd_state_arrays, vbd_world_maps

WORLDS = 2


def _solver() -> newton.solvers.SolverVBD:
    """Per world: a link welded to the world, a hinged link on it (one DOF, three constraint slots) and a free box."""
    b = newton.ModelBuilder()
    a = b.add_link(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), mass=1.0, label="a")
    d = b.add_link(xform=wp.transform((0.2, 0.0, 0.5), wp.quat_identity()), mass=1.0, label="d")
    for body in (a, d):
        b.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    b.add_articulation([b.add_joint_fixed(-1, a), b.add_joint_revolute(a, d, axis=(0.0, 1.0, 0.0))])
    free = b.add_body(xform=wp.transform((0.0, 0.5, 0.5), wp.quat_identity()), mass=1.0, label="free")
    b.add_shape_box(free, hx=0.05, hy=0.05, hz=0.05)
    scene = newton.ModelBuilder()
    scene.replicate(b, WORLDS)
    scene.color()
    return newton.solvers.SolverVBD(scene.finalize(device="cpu"))


def test_every_joint_dof_and_slot_array_has_a_world_map() -> None:
    solver = _solver()
    arrays, maps = vbd_state_arrays(solver), vbd_world_maps(solver)
    for key in ("vbd.joint_angle_prev", "vbd.joint_penalty_k", "vbd.joint_lambda_lin", "vbd.joint_drive_lambda"):
        assert key in arrays, f"{key} not in SolverVBD's history (Newton renamed it?)"
        assert key in maps and maps[key].shape[0] == arrays[key].shape[0], key
        assert np.bincount(maps[key].numpy(), minlength=WORLDS).tolist() == [arrays[key].shape[0] // WORLDS] * WORLDS
    whole = [k for k in arrays if k not in maps]
    assert all("contact" in k for k in whole), whole                   # only the contact history stays whole


def test_a_masked_restore_puts_back_only_world_0() -> None:
    solver = _solver()
    model, state = solver.model, solver.model.state()
    maps = vbd_world_maps(solver)
    extra = {k: v for k, v in vbd_state_arrays(solver).items() if k in maps}
    snap = WorldSnapshot(model, state, extra=extra, extra_world=maps)
    assert not snap.whole_arrays
    snap.save(state)
    saved = {k: v.numpy().copy() for k, v in extra.items()}
    filled = {k: np.full(v.numpy().shape, 7, dtype=v.numpy().dtype) for k, v in extra.items()}   # bool flags: True
    for k, arr in extra.items():
        arr.assign(filled[k])
    snap.restore(state, wp.array(np.array([1, 0], np.int32), dtype=wp.int32, device=model.device))
    for k, arr in extra.items():
        world = maps[k].numpy()
        now = arr.numpy()
        np.testing.assert_array_equal(now[world == 0], saved[k][world == 0], err_msg=k)
        np.testing.assert_array_equal(now[world == 1], filled[k][world == 1], err_msg=k)
