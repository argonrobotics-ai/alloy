# SPDX-License-Identifier: Apache-2.0
"""Per-world freeze of the entry solvers' internal state (warm starts, histories) during residual-gated passes.

A frozen world is re-stepped with frozen coupling inputs while the batch keeps iterating; that reproduces its
converging pass bitwise only if the solvers start each re-step from the same internal state. SolverMuJoCo carries
``mjw_data.qacc_warmstart`` from call to call (mujoco_warp@a9daa05f:mujoco_warp/_src/forward.py:472), so without this
the re-step starts from a different warm start and lands a few ulp away (measured on the ADMM scene: 1.5e-11 drift
of a world whose batch-mates converged later). Per pass p (W_x = internal state after pass x):

A. frozen worlds: arrays <- F_pre (W_{k-1}, the state their converging pass k started from);
B. P <- arrays (all worlds);
C. the pass runs;
D. worlds converged in pass p: F_post <- arrays (W_p), arrays <- P, F_pre <- arrays (W_{p-1});
E. exchange end: frozen worlds: arrays <- F_post, so the next exchange starts from W_k as if the world were alone.

The arrays are the entry solvers' registered state adapters with a per-world axis (world-major or with an item->world
map, SCR-WP4-3); arrays without one are left out (logged) and cannot be frozen per world.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import warp as wp

from alloy._vendor import newton
from alloy.core.registry import state_arrays_for, state_world_maps_for
from alloy.core.snapshot import WorldSnapshot
from alloy.coupling import residuals as rs

log = logging.getLogger("alloy.coupling")


@wp.kernel(enable_backward=False)
def freeze_masks_kernel(world_converged: wp.array[wp.int32], just: wp.array[wp.int32],
                        frozen: wp.array[wp.int32]):
    w = wp.tid()
    s = world_converged[w]
    just[w] = wp.where(s == rs.JUST_CONVERGED, 1, 0)
    frozen[w] = wp.where(s == rs.FROZEN, 1, 0)


def _per_world_arrays(solvers: Mapping[str, Any], world_count: int) -> tuple[dict, dict]:
    arrays: dict[str, wp.array] = {}
    maps: dict[str, wp.array] = {}
    skipped = []
    for name, solver in solvers.items():
        world_maps = state_world_maps_for(solver)
        for key, arr in state_arrays_for(solver).items():
            full = f"{name}.{key}"
            if key in world_maps:
                arrays[full], maps[full] = arr, world_maps[key]
            elif arr.ndim >= 1 and arr.shape[0] == world_count:
                arrays[full] = arr
            else:
                skipped.append(full)
    if skipped:
        log.warning("solver-state freeze: arrays without a per-world axis are not frozen: %s", skipped)
    return arrays, maps


class SolverStateFreeze:
    """Steps A-E above for the solvers of one coupler; a no-op when no solver has per-world internal state."""

    def __init__(self, model: Any, solvers: Mapping[str, Any], *, enabled: bool = True,
                 extra: Mapping[str, wp.array] | None = None, extra_world: Mapping[str, wp.array] | None = None) -> None:
        """``enabled=False`` when no world can freeze before the last pass (k_min == k_max): nothing to reproduce.
        ``extra``/``extra_world``: coupler buffers that also carry state between passes (ADMM ``*_qd_k``)."""
        self.world_count = int(model.world_count)
        self.device = model.device
        arrays, maps = _per_world_arrays(solvers, self.world_count) if enabled else ({}, {})
        if enabled:
            arrays.update(extra or {})
            maps.update(extra_world or {})
        self.active = bool(arrays)
        self.keys = tuple(sorted(arrays))
        if not self.active:
            return
        bare = newton.State()
        self._bare = bare
        self.pre = WorldSnapshot(model, bare, extra=arrays, extra_world=maps)
        self.f_pre = WorldSnapshot(model, bare, extra=arrays, extra_world=maps)
        self.f_post = WorldSnapshot(model, bare, extra=arrays, extra_world=maps)
        self.just = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        self.frozen = wp.zeros(self.world_count, dtype=wp.int32, device=self.device)
        log.debug("solver-state freeze over %s", self.keys)

    def _masks(self, world_converged: wp.array) -> None:
        wp.launch(freeze_masks_kernel, dim=self.world_count, inputs=[world_converged, self.just, self.frozen],
                  device=self.device)

    def begin_pass(self, world_converged: wp.array) -> None:
        """Steps A and B (call before the pass steps any solver)."""
        if not self.active:
            return
        self._masks(world_converged)
        self.f_pre.restore(self._bare, self.frozen)
        self.pre.save(self._bare)

    def capture_converged(self, world_converged: wp.array) -> None:
        """Step D (call after the convergence kernel marked JUST_CONVERGED, before settling them to FROZEN)."""
        if not self.active:
            return
        self._masks(world_converged)
        self.f_post.save(self._bare, self.just)
        self.pre.restore(self._bare, self.just)
        self.f_pre.save(self._bare, self.just)

    def finish(self, world_converged: wp.array) -> None:
        """Step E (call once the exchange's passes are done)."""
        if not self.active:
            return
        self._masks(world_converged)
        self.f_post.restore(self._bare, self.frozen)
