# SPDX-License-Identifier: Apache-2.0
"""Kinematic proxies: source bodies with no DOF to the world stay immovable in every destination (SCR-SC8-2).

A bench table or plywood sheet welded to the world (FIXED joints up to parent -1, recursively) is merged into the
world by MuJoCo, but Newton's proxy coupler gives every proxied body a FREE root in the destination view
(newton@ac7f7830:newton/_src/solvers/coupled/solver_coupled.py:201-221) and installs the source's effective mass
(solver_coupled_proxy.py:1284-1307). SolverVBD then integrates the 9 kg plywood under gravity each exchange, the
coupler rewinds it the next, and a film resting on it is pushed every exchange (SC8's C16: 2.7 mm per period, 77-85 J
of interface work into a 3.4 g bag, health silent).

``KinematicProxyMixin`` (first in the Scheduler's bases):
- classifies the model's bodies once (``world_welded_bodies``) before Newton builds the views;
- installs mass and inertia 0 for those proxies wherever Newton installs proxy inertia
  (``_apply_body_inertia_override``: Newton's kernel zeroes the inverses for mass <= 0 and notifies the destination
  solver) and re-zeroes them after Newton re-copies parent inertia on a model change
  (``_refresh_body_inertial_view_overrides``, which then notifies). They stay proxied: their shapes stay collidable
  in the destination. The source model is never touched;
- guards each exchange on the device: a kinematic proxy whose destination pose differs from the source pose after
  the exchange (> ``POS_TOL`` m or > ``ROT_TOL`` rad) flags its world, and ``capacity_flags()`` hands that per-world
  flag to HealthMonitor with JOINT_FAIL | PEN_FAIL (the weld is violated; PEN_FAIL makes it INVALID, fail closed);
- lists the kinematic proxies per pair (``kinematic_proxies``) for the plan report.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.precision.types import HealthFlags

log = logging.getLogger("alloy.coupling")

FIXED = int(newton.JointType.FIXED)
POS_TOL = 1e-9          # [m]   (coordinator ruling SCR-SC8-2)
ROT_TOL = 1e-9          # [rad]
GUARD_BITS = int(HealthFlags.JOINT_FAIL | HealthFlags.PEN_FAIL)


def world_welded_bodies(model: Any) -> np.ndarray:
    """bool[body_count]: True for a body whose chain to the world has no DOF, i.e. it has an incoming FIXED joint
    whose parent is the world (-1) or itself world-welded. A body with no incoming joint is not welded."""
    n = int(model.body_count or 0)
    welded = np.zeros(n, dtype=bool)
    if n == 0 or not int(model.joint_count or 0):
        return welded
    jt, jp, jc = model.joint_type.numpy(), model.joint_parent.numpy(), model.joint_child.numpy()
    fixed_parents: dict[int, list[int]] = {}
    for t, p, c in zip(jt, jp, jc, strict=True):
        if int(t) == FIXED and 0 <= int(c) < n:
            fixed_parents.setdefault(int(c), []).append(int(p))
    changed = True
    while changed:                      # fixed point: welded to the world or to a welded body
        changed = False
        for child, parents in fixed_parents.items():
            if not welded[child] and any(p < 0 or (p < n and welded[p]) for p in parents):
                welded[child] = changed = True
    return welded


@wp.kernel(enable_backward=False)
def kinematic_moved_kernel(dst_q: wp.array[wp.transform], local_ids: wp.array[wp.int32],
                           global_ids: wp.array[wp.int32], src_q: wp.array[wp.transform],
                           body_world: wp.array[wp.int32], pos_tol: float, rot_tol: float, out: wp.array[wp.int32]):
    """Flag the world of a kinematic proxy whose destination pose left the source pose (rotation from the relative
    quaternion's vector part: well conditioned for small angles, unlike acos of the dot product)."""
    i = wp.tid()
    a = dst_q[local_ids[i]]
    b = src_q[global_ids[i]]
    dp = wp.length(wp.transform_get_translation(a) - wp.transform_get_translation(b))
    rel = wp.mul(wp.quat_inverse(wp.transform_get_rotation(a)), wp.transform_get_rotation(b))
    ang = 2.0 * wp.length(wp.vec3(rel[0], rel[1], rel[2]))
    if dp > pos_tol or ang > rot_tol:
        w = body_world[global_ids[i]]
        if w >= 0:
            out[w] = 1
        else:
            for k in range(out.shape[0]):           # a global body: every world
                out[k] = 1


@wp.kernel(enable_backward=False)
def clear_masked_kernel(mask: wp.array[wp.int32], out: wp.array[wp.int32]):
    w = wp.tid()
    if mask[w] != 0:
        out[w] = 0


class KinematicProxyMixin:
    """See the module docstring. Call ``_init_kinematic(model)`` before the Newton coupler's ``__init__`` and
    ``_init_kinematic_guard()`` after it."""

    def _init_kinematic(self, model: Any) -> None:
        self._welded = world_welded_bodies(model)
        self._kinematic_local: dict[str, np.ndarray] = {}

    def _kinematic_local_ids(self, dst_name: str) -> np.ndarray:
        """View-local ids of the kinematic proxies in destination ``dst_name``."""
        if dst_name not in self._kinematic_local:
            ids = [m.proxy_ids_local.numpy()[self._welded[m.proxy_ids_global.numpy()]]
                   for m in self._proxy_mappings if m.dst_name == dst_name and m.proxy_ids_global is not None]
            self._kinematic_local[dst_name] = np.concatenate(ids).astype(np.int64) if ids else np.zeros(0, np.int64)
        return self._kinematic_local[dst_name]

    def _apply_body_inertia_override(self, entry: Any, body_indices: wp.array, body_mass: wp.array,
                                     body_inertia: wp.array) -> None:
        kinematic = self._kinematic_local_ids(entry.name) if hasattr(self, "_welded") else np.zeros(0, np.int64)
        hit = np.isin(body_indices.numpy(), kinematic)
        if hit.any():                                   # mass 0 -> Newton zeroes inv_mass and inv_inertia
            mass, inertia = body_mass.numpy().copy(), body_inertia.numpy().copy()
            mass[hit], inertia[hit] = 0.0, 0.0
            body_mass = wp.array(mass, dtype=body_mass.dtype, device=body_mass.device)
            body_inertia = wp.array(inertia, dtype=body_inertia.dtype, device=body_inertia.device)
        super()._apply_body_inertia_override(entry, body_indices, body_mass, body_inertia)

    def _refresh_body_inertial_view_overrides(self, entry: Any) -> None:
        super()._refresh_body_inertial_view_overrides(entry)
        kinematic = self._kinematic_local_ids(entry.name) if hasattr(self, "_welded") else np.zeros(0, np.int64)
        if kinematic.size:
            entry.view.disable_body_dynamics(wp.array(kinematic, dtype=int, device=self.model.device))

    def _init_kinematic_guard(self) -> None:
        """Device arrays of the guard, one group per proxy mapping with kinematic proxies; logs them per pair."""
        dev = self.model.device
        self._kinematic_groups: list[tuple[str, wp.array, wp.array]] = []
        self._kinematic_by_pair: dict[tuple[str, str], tuple[int, ...]] = {}
        for m in self._proxy_mappings:
            if m.proxy_ids_global is None:
                continue
            g, loc = m.proxy_ids_global.numpy(), m.proxy_ids_local.numpy()
            sel = self._welded[g]
            if not sel.any():
                continue
            self._kinematic_by_pair[(m.src_name, m.dst_name)] = tuple(int(b) for b in g[sel])
            self._kinematic_groups.append((m.dst_name, wp.array(loc[sel].astype(np.int32), dtype=wp.int32, device=dev),
                                           wp.array(g[sel].astype(np.int32), dtype=wp.int32, device=dev)))
            log.info("kinematic proxies %s -> %s: %d world-welded source bodies (mass and inertia 0 in the "
                     "destination)", m.src_name, m.dst_name, int(sel.sum()))
        self._kinematic_moved = wp.zeros(max(int(self.model.world_count or 0), 1), dtype=wp.int32, device=dev)

    def kinematic_proxies(self) -> dict[tuple[str, str], tuple[int, ...]]:
        """(source, destination) -> global ids of the source bodies proxied as kinematic (plan report)."""
        return dict(getattr(self, "_kinematic_by_pair", {}))

    def step(self, state_in: Any, state_out: Any, control: Any, contacts: Any, dt: float, **kw: Any) -> None:
        super().step(state_in, state_out, control, contacts, dt, **kw)
        for dst, local_ids, global_ids in getattr(self, "_kinematic_groups", ()):
            wp.launch(kinematic_moved_kernel, dim=local_ids.shape[0], device=self.model.device,
                      inputs=[self._entries[dst].state_1.body_q, local_ids, global_ids, state_out.body_q,
                              self.model.body_world, POS_TOL, ROT_TOL, self._kinematic_moved])

    def capacity_flags(self) -> list[Any]:
        flags = list(super().capacity_flags())
        if getattr(self, "_kinematic_groups", None):
            flags.append((self._kinematic_moved, GUARD_BITS))
        return flags

    def reset_world_flags(self, world_mask: Any = None) -> None:
        super().reset_world_flags(world_mask)
        moved = getattr(self, "_kinematic_moved", None)
        if moved is None:
            return
        if world_mask is None:
            moved.zero_()
        else:
            wp.launch(clear_masked_kernel, dim=moved.shape[0], device=moved.device, inputs=[world_mask, moved])
