# SPDX-License-Identifier: Apache-2.0
"""Law: a topology snapshot covers a coupled MPM view's own particle flags.

The crumble kernel writes the view's ``particle_flags`` when that array is not the model's own, so
``mutated_model_arrays`` / ``mutated_model_worlds`` must list it, or a restore leaves crumbled flags behind.
"""

from __future__ import annotations

from types import SimpleNamespace

import warp as wp

from alloy.topology.solver_hooks import SolverHooks


def test_solver_hooks_snapshots_distinct_mpm_view_particle_flags():
    """mutated_model_arrays and mutated_model_worlds must include MPM view particle_flags when distinct."""
    m_flags = wp.zeros(4, dtype=wp.int32, device="cpu")
    v_flags = wp.zeros(4, dtype=wp.int32, device="cpu")
    mpm_flags = wp.zeros(4, dtype=wp.int32, device="cpu")
    mpm_mat = wp.zeros(4, dtype=wp.int32, device="cpu")
    mpm_vol = wp.zeros(4, dtype=wp.float32, device="cpu")
    pw = wp.zeros(4, dtype=wp.int32, device="cpu")

    hooks = object.__new__(SolverHooks)
    hooks.model = SimpleNamespace(
        device="cpu",
        world_count=1,
        spring_count=0,
        edge_count=0,
        particle_count=4,
        particle_flags=m_flags,
        particle_world=pw,
        body_count=0,
        mujoco=None,
    )
    hooks.tables = SimpleNamespace(edge_ids=wp.zeros(0, dtype=wp.int32, device="cpu"))
    hooks.eq_enabled = wp.zeros(1, dtype=wp.int32, device="cpu")
    hooks.eq_relpose = wp.zeros(1, dtype=wp.transform, device="cpu")
    hooks.mpm = [
        SimpleNamespace(
            model=SimpleNamespace(particle_flags=v_flags, particle_world=pw),
            _mpm_model=SimpleNamespace(
                particle_flags=mpm_flags,
                material_particle_flags=mpm_mat,
                material_particle_volume=mpm_vol,
            ),
        )
    ]
    arrays = hooks.mutated_model_arrays()
    worlds = hooks.mutated_model_worlds()
    assert arrays["topology.mpm0.view_particle_flags"] is v_flags
    assert "topology.mpm0.view_particle_flags" in worlds
